"""Cross-cluster feature deduplication (Sprint 2).

Sprint 1 produces business-readable per-package sub-features but never
sees them all together — so the same product domain split across N
packages becomes N features (e.g. on documenso: ``lib/document-signing``,
``remix/document-signing``, ``trpc/document-signing``,
``ui/document-signing``, ``email/document-signing-emails``).

This module adds **one** Anthropic Sonnet call after
``deep_scan_workspace`` returns. The model sees every feature's name,
description, and a few sample paths; it proposes semantic merges with
a one-sentence rationale each. We apply the merges to the
``DeepScanResult`` and return the dedup'd map.

Public entry point: :func:`dedup_features`. The Anthropic client is
injectable so unit tests can mock it; the same fake-client pattern as
:mod:`faultline.llm.tool_use_scan` works here.

The pass is opt-in via the ``--dedup`` CLI flag and runs only when
``pipeline.run`` is called with ``dedup=True``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from .cost import CostTracker
    from .sonnet_scanner import DeepScanResult


logger = logging.getLogger(__name__)


DEFAULT_MAX_TOKENS = 8_192
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLE_PATHS = 5
# Hard cap on merges per pass. Prevents the model from collapsing the
# whole feature map into a small handful of mega-features. Original
# calibration at 12 was tight for documenso (5 document-signing
# siblings) and formbricks (~3 survey siblings). Bumped to 50 for n8n
# scale: 5 workflow-concept fragments + 3 credentials + 2 mcp-browser
# + 3 codemirror lang + ~10 other near-duplicates from a 78-feature
# scan benefit from a higher ceiling without the model going wild
# (it still has to justify each merge on shared paths/concepts).
MAX_MERGES_PER_PASS = 50

# Synthetic buckets that must never be touched by the merge pass.
# documentation collapses every doc bucket; shared-infra holds repo
# config; both are populated by stage-2 of pipeline.run AFTER
# deep_scan_workspace, but we exclude them defensively in case dedup
# is ever called against a result that already has them.
_PROTECTED_NAMES: frozenset[str] = frozenset({
    "documentation",
    "shared-infra",
    "examples",
})


# Universally tooling-only package names. When a workspace has one of
# these as a top-level package, it's almost never a product feature —
# it's eslint config, tsconfig template, etc. Folded into
# ``shared-infra`` automatically by the pipeline so users don't have
# to enumerate them in ``.faultline.yaml`` skip_features. List is
# intentionally narrow: only names that are tooling in ~99% of repos.
TOOLING_PACKAGE_NAMES: frozenset[str] = frozenset({
    "tsconfig", "typescript-config",
    "eslint-config", "prettier-config", "tailwind-config",
    "postcss-config", "babel-config", "stylelint-config",
    "vitest-config", "jest-config", "playwright-config",
    "config-eslint", "config-typescript", "config-prettier",
    "config-tailwind",
    "tooling", "build-config",
})


# ── System prompt ─────────────────────────────────────────────────────


_DEDUP_SYSTEM_PROMPT = """\
You are a senior engineer reviewing a list of detected features in
a single codebase. Some entries describe THE SAME business domain
split across packages — e.g. "lib/document-signing",
"remix/document-signing", "trpc/document-signing" all implement
parts of the document signing flow.

Your job: identify those semantic duplicates and propose merges.

INPUT
A JSON array of features:
  [{"name": "...", "description": "...", "file_count": N,
    "sample_paths": ["...", "..."]}]

OUTPUT (JSON only, no prose, no markdown fences)
{
  "merges": [
    {
      "into": "signing",
      "from": ["lib/document-signing", "remix/document-signing"],
      "description": "Document signing across UI, API, persistence,
                     and notifications.",
      "rationale": "One sentence explaining why these belong together."
    }
  ]
}

RULES
- Only merge features that genuinely describe the same business
  domain. When in doubt, do not merge.
- Pick the cleanest business name for "into" — often the shortest
  or most natural. It can be one of the input names or a new name.
- Every merge needs a one-sentence rationale. No empty rationales.
- Do NOT touch names containing "documentation", "shared-infra",
  or "examples" — those are synthetic.
- Do NOT merge features whose descriptions describe different
  responsibilities even if names overlap.
- Sprint 14 — DIFFERENT ACTORS or DIFFERENT EVENTS are NEVER
  duplicates. ``Onboard Employee`` vs ``Onboard Contractor`` use
  similar tooling but represent different business processes:
  separate compliance checks, different approval flows, different
  data models. Same for ``Renew Subscription`` vs ``Cancel
  Subscription`` — same actor (account owner) but different events
  with different downstream effects. When the input describes a
  flow's actor (employee, contractor, admin, public viewer) or
  event (signup, renewal, cancellation, refund, dispute) and those
  differ across the candidates, leave them separate.
- Each "from" group must contain at least 2 distinct features.
- Cap total merges at 12 groups per pass.
- It is OK to return {"merges": []} when nothing genuinely
  duplicates.
"""


# ── Data shapes ───────────────────────────────────────────────────────


@dataclass
class Merge:
    """One proposed merge group."""

    into: str
    sources: list[str]  # the original feature names being collapsed
    description: str = ""
    rationale: str = ""

    def is_valid(self) -> bool:
        return (
            bool(self.into)
            and len(self.sources) >= 2
            and bool(self.rationale.strip())
        )


# ── Anthropic client protocol ─────────────────────────────────────────


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...


# ── Helpers ───────────────────────────────────────────────────────────


def _build_summaries(
    features: dict[str, list[str]],
    descriptions: dict[str, str],
    sample_paths: int = DEFAULT_SAMPLE_PATHS,
) -> list[dict[str, Any]]:
    """Compress the feature map into the LLM input shape.

    Sample paths are deterministic (sorted prefix) so the same
    feature gets the same summary across runs — consistency improves
    cache hit rate on prompt caching.
    """
    out: list[dict[str, Any]] = []
    for name in sorted(features):
        if name in _PROTECTED_NAMES:
            continue
        paths = features[name]
        out.append({
            "name": name,
            "description": descriptions.get(name, ""),
            "file_count": len(paths),
            "sample_paths": sorted(paths)[:sample_paths],
        })
    return out


def _parse_merges_payload(text: str) -> dict | None:
    """Same JSON-extraction fallback chain as tool_use_scan."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _coerce_merges(payload: dict | None) -> list[Merge]:
    """Turn raw JSON into validated Merge objects.

    Drops malformed entries (missing fields, fewer than 2 sources,
    empty rationale) silently — the LLM is allowed to be sloppy on
    individual entries; we just skip those.
    """
    if not isinstance(payload, dict):
        return []
    raw = payload.get("merges")
    if not isinstance(raw, list):
        return []

    out: list[Merge] = []
    for entry in raw[:MAX_MERGES_PER_PASS]:
        if not isinstance(entry, dict):
            continue
        into = (entry.get("into") or "").strip()
        sources = entry.get("from") or []
        if not isinstance(sources, list):
            continue
        sources = [s for s in (str(x).strip() for x in sources) if s]
        # Dedup sources, preserve order
        seen: set[str] = set()
        unique_sources: list[str] = []
        for s in sources:
            if s not in seen:
                seen.add(s)
                unique_sources.append(s)
        merge = Merge(
            into=into,
            sources=unique_sources,
            description=(entry.get("description") or "").strip(),
            rationale=(entry.get("rationale") or "").strip(),
        )
        if merge.is_valid():
            out.append(merge)
    return out


def _apply_merges(
    result: "DeepScanResult",
    merges: list[Merge],
) -> tuple["DeepScanResult", list[Merge]]:
    """Apply merge ops to the feature map, return (new_result, applied).

    For each merge:
      - Sources missing from ``result.features`` are dropped from the
        merge (and the merge skipped if fewer than 2 valid sources
        remain). Stale model output should not corrupt the map.
      - Files from every source unioned + sorted into ``into``.
      - Descriptions: prefer the model-supplied merge description;
        fall back to the longest source description; finally append
        ``" (merged: A, B, ...)"`` so the trail survives in the
        output JSON without needing schema changes.
      - Flows: union across sources; flow_descriptions union per flow.
      - Source feature entries removed.
      - Protected names (``documentation``, ``shared-infra``,
        ``examples``) are never touched as either source or target.

    Returns the **same DeepScanResult instance** (mutated) plus the
    list of merges that actually applied (so callers can log).
    """
    applied: list[Merge] = []

    for merge in merges:
        if merge.into in _PROTECTED_NAMES:
            logger.warning(
                "dedup: refusing to merge into protected name %r — skipping",
                merge.into,
            )
            continue

        valid_sources = [
            s for s in merge.sources
            if s in result.features and s not in _PROTECTED_NAMES
        ]
        if len(valid_sources) < 2:
            logger.info(
                "dedup: merge %r has fewer than 2 live sources — skipping",
                merge.into,
            )
            continue

        # Union files from every source. dict to preserve insertion
        # order while deduping.
        unioned_files: dict[str, None] = {}
        for src in valid_sources:
            for f in result.features.get(src, []):
                unioned_files[f] = None
        # If `into` already exists (e.g. it was one of the sources or
        # already a feature), include its files too.
        if merge.into in result.features:
            for f in result.features[merge.into]:
                unioned_files[f] = None

        # Determine final description.
        if merge.description:
            final_desc = merge.description
        else:
            # longest source description, else empty
            cands = [result.descriptions.get(s, "") for s in valid_sources]
            final_desc = max(cands, key=len, default="")
        # Append merge trail. Keeps the rationale visible in the
        # JSON output without changing the DeepScanResult schema.
        sources_for_trail = [s for s in valid_sources if s != merge.into]
        if sources_for_trail:
            trail = f" (merged: {', '.join(sources_for_trail)})"
            if not final_desc.endswith(trail):
                final_desc = (final_desc + trail).strip()

        # Union flows + flow_descriptions across sources.
        merged_flows: list[str] = []
        seen_flows: set[str] = set()
        merged_flow_descs: dict[str, str] = {}
        for src in valid_sources + ([merge.into] if merge.into in result.features else []):
            for fl in result.flows.get(src, []):
                if fl not in seen_flows:
                    seen_flows.add(fl)
                    merged_flows.append(fl)
            for fl, desc in result.flow_descriptions.get(src, {}).items():
                # First writer wins; later sources do not overwrite.
                merged_flow_descs.setdefault(fl, desc)

        # Remove source features (skip merge.into so we can write it
        # below in one go).
        for src in valid_sources:
            if src == merge.into:
                continue
            result.features.pop(src, None)
            result.descriptions.pop(src, None)
            result.flows.pop(src, None)
            result.flow_descriptions.pop(src, None)

        # Write the merged feature.
        result.features[merge.into] = sorted(unioned_files)
        if final_desc:
            result.descriptions[merge.into] = final_desc
        if merged_flows:
            result.flows[merge.into] = merged_flows
        if merged_flow_descs:
            result.flow_descriptions[merge.into] = merged_flow_descs

        applied.append(merge)
        logger.info(
            "dedup: merged %d → %r (%d files): %s",
            len(valid_sources), merge.into,
            len(result.features[merge.into]),
            merge.rationale[:120],
        )

    return result, applied


# ── Public entry point ────────────────────────────────────────────────


def dedup_features(
    result: "DeepScanResult",
    *,
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: "CostTracker | None" = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> "DeepScanResult":
    """Run a single Sonnet pass to merge cross-cluster duplicate features.

    Args:
        result: ``DeepScanResult`` from ``deep_scan_workspace`` /
            ``deep_scan``. Mutated in place AND returned.
        client: Optional Anthropic client. If omitted, one is built
            from ``api_key`` (or the ``ANTHROPIC_API_KEY`` env var).
        api_key: Override / fallback for the Anthropic key.
        model: Override Sonnet model id.
        tracker: ``CostTracker`` for budget + reporting.
        max_tokens: Output token cap. 8K is plenty: the response is
            merge ops, not file lists.

    Returns:
        The same ``DeepScanResult`` (so callers can chain) with
        merges applied. On any error (no API key, network failure,
        unparseable response) the original result is returned
        unchanged — dedup is opportunistic.
    """
    if not result or not result.features:
        return result

    summaries = _build_summaries(result.features, result.descriptions)
    if len(summaries) < 2:
        # Nothing to dedup against.
        return result

    if client is None:
        import os
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("dedup: no API key — skipping pass")
            return result
        client = anthropic.Anthropic(api_key=key)

    resolved_model = model or DEFAULT_MODEL
    user_payload = json.dumps(summaries, indent=2, ensure_ascii=False)

    try:
        response = client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            system=_DEDUP_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
    except Exception as exc:  # noqa: BLE001 - opportunistic
        logger.warning("dedup: Anthropic call failed (%s) — skipping", exc)
        return result

    if tracker is not None:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        if input_tokens or output_tokens:
            tracker.record(
                provider="anthropic",
                model=resolved_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                label="dedup",
            )

    content = list(getattr(response, "content", []) or [])
    text_parts = [
        getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", "")
        for b in content
        if (getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else "")) == "text"
    ]
    text = "\n".join(p for p in text_parts if p)

    payload = _parse_merges_payload(text)
    merges = _coerce_merges(payload)
    if not merges:
        logger.info("dedup: model proposed 0 valid merges — no-op")
        return result

    # Diagnostic: log every proposed merge BEFORE applying so we can
    # see what the model intended even when post-apply state is
    # ambiguous (e.g. overlapping `from` lists across merges).
    for i, m in enumerate(merges):
        logger.info(
            "dedup: proposed merge[%d] into=%r from=%s",
            i, m.into, m.sources,
        )

    result, applied = _apply_merges(result, merges)
    logger.info(
        "dedup: applied %d merge(s); feature count → %d",
        len(applied), len(result.features),
    )
    return result
