"""Self-critique loop (Sprint 5).

After Sprints 1-4 the engine produces well-named features, dedup'd
across packages, sized appropriately, with grounded flows. A small
handful of weak names still slip through — generic placeholders the
model picks when a cluster is genuinely miscellaneous, or names that
re-emerge after dedup creates a new feature.

This module adds a final critique stage. Two passes:

    1. Critique pass — single Sonnet call, no tools, sees the whole
       feature map + flow names. Outputs a list of flagged items
       with reasons.

    2. Re-investigation pass — for each flagged item (capped at 5),
       run a focused tool-augmented rename via the Sprint 1
       dispatcher. Apply the new name only if it is materially
       better than the original.

Both passes are opt-in via ``pipeline.run(critique=True)``. Failure
mode is opportunistic: any error returns the original
``DeepScanResult`` unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dedup import _PROTECTED_NAMES, _parse_merges_payload  # reuse parser
from .tool_use_scan import tool_use_scan


# Critique is a classification + rename task; Sonnet's reasoning isn't
# required. Haiku 4.5 handles the small JSON outputs and the cheap
# rename calls fine and cuts ~$0.60-1.00 per scan.
DEFAULT_MODEL = "claude-haiku-4-5"

if TYPE_CHECKING:  # pragma: no cover
    from .cost import CostTracker
    from .sonnet_scanner import DeepScanResult


logger = logging.getLogger(__name__)


DEFAULT_MAX_ITEMS = 5
DEFAULT_MAX_TOKENS = 4_096
DEFAULT_TOOL_BUDGET = 5  # tighter than Sprint 1+3+4 — focused rename


# ── Prompts ────────────────────────────────────────────────────────────


_CRITIQUE_SYSTEM_PROMPT = """\
You are reviewing a feature map for quality. The map already passed
through naming, dedup, and sub-decomposition. Flag the ≤ 5 weakest
names — entries whose name is too generic, vague, or fails to
convey what the code does.

INPUT
A JSON array of items:
  [{"kind": "feature", "name": "...", "description": "...",
    "file_count": N},
   {"kind": "flow", "feature": "...", "name": "...",
    "description": "..."},
   ...]

OUTPUT (JSON only, no prose, no markdown fences)
{
  "weak": [
    {
      "kind": "feature",
      "name": "lib/platform-infrastructure",
      "reason": "Generic name; 166 files suggests this should split."
    }
  ]
}

RULES
- 0-5 entries. Quality over quantity. Empty list is acceptable.
- Skip names containing 'documentation', 'shared-infra', 'examples'.
- Skip names that are already specific even if short
  (auth, billing, signing — these are real domain words).
- For flows, the "feature" field tells the rename pass which bucket
  to update.
- Each "reason" is a single sentence.
"""


_RENAME_SYSTEM_PROMPT = """\
You are renaming a single feature whose current name is too generic
or vague. The feature already has a coherent file set; your job is
to inspect a few files and propose a better business-readable name.

WORKFLOW
1. Run read_file_head on 3-5 representative files.
2. Optionally grep_pattern to confirm a hypothesis.
3. Output the new name + a one-sentence description.

OUTPUT (JSON only)
{
  "new_name": "pdf-rendering",
  "description": "Server-side PDF generation for completed signed documents.",
  "reason": "Why the new name is better than the old one."
}

RULES
- Two-word business names preferred. No "lib", "utils", "shared",
  "core", "general", "misc", "main", "common", "base", "helpers".
- If you cannot find a clearly better name, return
  {"new_name": ""} and we will keep the original.
- Do NOT propose a different file split — only a rename.
"""


# ── Helpers ────────────────────────────────────────────────────────────


def _build_critique_summaries(result: "DeepScanResult") -> list[dict]:
    """Pack the feature map into the critique-pass input shape.

    Synthetic buckets are skipped. Flow entries are flattened so the
    reviewer can spot bad flow names alongside bad feature names.
    """
    out: list[dict] = []
    for name in sorted(result.features):
        if name in _PROTECTED_NAMES:
            continue
        out.append({
            "kind": "feature",
            "name": name,
            "description": result.descriptions.get(name, ""),
            "file_count": len(result.features[name]),
        })
        for flow in result.flows.get(name, []):
            flow_desc = result.flow_descriptions.get(name, {}).get(flow, "")
            out.append({
                "kind": "flow",
                "feature": name,
                "name": flow,
                "description": flow_desc,
            })
    return out


def _coerce_critique(payload: dict | None, max_items: int) -> list[dict]:
    """Validate and clean the critique pass output."""
    if not isinstance(payload, dict):
        return []
    raw = payload.get("weak")
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    for entry in raw[:max_items]:
        if not isinstance(entry, dict):
            continue
        kind = (entry.get("kind") or "").strip().lower()
        if kind not in {"feature", "flow"}:
            continue
        name = (entry.get("name") or "").strip()
        if not name or name in _PROTECTED_NAMES:
            continue
        item: dict[str, Any] = {
            "kind": kind,
            "name": name,
            "reason": (entry.get("reason") or "").strip(),
        }
        if kind == "flow":
            feat = (entry.get("feature") or "").strip()
            if not feat:
                continue
            item["feature"] = feat
        out.append(item)
    return out


_GENERIC_TOKENS: frozenset[str] = frozenset({
    "lib", "utils", "shared", "core", "general", "misc", "main",
    "common", "base", "helpers", "things", "stuff", "data",
    # Structural suffixes that add no domain meaning when tacked
    # onto a name; "billing" → "billing-service" should be rejected.
    "service", "services", "module", "modules", "manager",
    "system", "handler", "handlers", "logic", "feature",
})


def _tokens(name: str) -> set[str]:
    """Lowercase tokens of a feature name, split on /, -, _."""
    return {t for t in re.split(r"[/\-_]", name.lower()) if t}


def _is_materially_better(old: str, new: str) -> bool:
    """Heuristic: is ``new`` worth swapping in over ``old``?

    Yes when:
      - new is non-empty AND not equal to old (case-insensitive).
      - new contains at least one token that is NOT in
        :data:`_GENERIC_TOKENS` and not in old's tokens.
      - new does not REGRESS into a generic token.

    Used to filter "rename" responses that just shuffle synonyms.
    """
    if not new:
        return False
    if new.strip().lower() == old.strip().lower():
        return False

    new_tokens = _tokens(new)
    old_tokens = _tokens(old)

    # New must not be ENTIRELY made of generic words.
    if new_tokens and new_tokens.issubset(_GENERIC_TOKENS):
        return False

    # New introduces at least one specific (non-generic, non-old)
    # token.
    novelty = (new_tokens - old_tokens) - _GENERIC_TOKENS
    return bool(novelty)


def _rewrite_feature_name(
    result: "DeepScanResult",
    old: str,
    new: str,
    new_description: str = "",
) -> bool:
    """Atomically rename ``old → new`` across features / flows /
    descriptions / flow_descriptions.

    No-op (returns False) when ``old`` is missing or ``new`` already
    exists. Returns True on success.
    """
    if old not in result.features or new in result.features:
        return False

    result.features[new] = result.features.pop(old)
    if old in result.descriptions:
        result.descriptions[new] = (
            new_description or result.descriptions.pop(old)
        )
        result.descriptions.pop(old, None)
    elif new_description:
        result.descriptions[new] = new_description
    if old in result.flows:
        result.flows[new] = result.flows.pop(old)
    if old in result.flow_descriptions:
        result.flow_descriptions[new] = result.flow_descriptions.pop(old)
    return True


def _rewrite_flow_name(
    result: "DeepScanResult",
    feature: str,
    old: str,
    new: str,
    new_description: str = "",
) -> bool:
    """Rename a single flow inside a feature."""
    flows = result.flows.get(feature)
    if flows is None or old not in flows:
        return False
    if new in flows:
        return False
    flows[:] = [new if f == old else f for f in flows]
    flow_descs = result.flow_descriptions.get(feature)
    if flow_descs is not None and old in flow_descs:
        flow_descs[new] = new_description or flow_descs.pop(old)
        flow_descs.pop(old, None)
    elif new_description and flow_descs is not None:
        flow_descs[new] = new_description
    return True


# ── Re-investigation: feature rename ───────────────────────────────────


def _rename_feature_with_tools(
    *,
    feature_name: str,
    files: list[str],
    repo_root: Path,
    client: Any,
    model: str,
    tracker: "CostTracker | None",
    tool_budget: int,
) -> dict | None:
    """Tool-augmented rename for a single weak feature.

    Returns ``{"new_name": str, "description": str, "reason": str}``
    when the model proposes a non-empty name. Returns ``None``
    otherwise. Validation against ``_is_materially_better`` happens
    in the caller.
    """
    try:
        parsed = tool_use_scan(
            package_name=feature_name,
            files=files,
            repo_root=repo_root,
            client=client,
            model=model,
            tool_budget=tool_budget,
            tracker=tracker,
            cost_label=f"critique-rename:{feature_name}",
            system_prompt=_RENAME_SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("critique(%s): rename scan failed (%s)", feature_name, exc)
        return None

    if not isinstance(parsed, dict):
        return None
    new = (parsed.get("new_name") or "").strip()
    if not new:
        return None
    return {
        "new_name": new,
        "description": (parsed.get("description") or "").strip(),
        "reason": (parsed.get("reason") or "").strip(),
    }


# ── Public entry point ────────────────────────────────────────────────


def critique_and_refine(
    result: "DeepScanResult",
    *,
    repo_root: Path | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: "CostTracker | None" = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    locked_names: frozenset[str] | None = None,
) -> "DeepScanResult":
    """Run a critique pass and apply tool-augmented renames.

    Mutates ``result`` in place AND returns it. Each rejected
    proposal is logged so the user can see what the engine
    considered but did not change.

    ``locked_names`` is the optional set of feature names that
    critique must NOT rename — typically the canonical names from
    the user's ``.faultline.yaml``. Locking these stabilizes scan
    output across runs (Sprint 5 critique is otherwise non-
    deterministic and would re-rename the same feature differently
    on consecutive scans).
    """
    if not result or not result.features:
        return result

    summaries = _build_critique_summaries(result)
    if not summaries:
        return result

    locked: frozenset[str] = locked_names or frozenset()
    if locked:
        # Strip locked features from the input so the model never
        # sees them as candidates for renaming.
        summaries = [s for s in summaries if s.get("name") not in locked]
        if not summaries:
            logger.info(
                "critique: every feature is locked by user config — "
                "no-op",
            )
            return result

    if client is None:
        import os
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("critique: no API key — skipping pass")
            return result
        client = anthropic.Anthropic(api_key=key)

    resolved_model = model or DEFAULT_MODEL
    user_payload = json.dumps(summaries, indent=2, ensure_ascii=False)

    try:
        response = client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            system=_CRITIQUE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_payload}],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("critique: Anthropic call failed (%s)", exc)
        return result

    if tracker is not None:
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        if in_tok or out_tok:
            tracker.record(
                provider="anthropic", model=resolved_model,
                input_tokens=in_tok, output_tokens=out_tok,
                label="critique",
            )

    content = list(getattr(response, "content", []) or [])
    text_parts: list[str] = []
    for b in content:
        btype = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else "")
        if btype == "text":
            text_parts.append(getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", ""))
    text = "\n".join(p for p in text_parts if p)

    payload = _parse_merges_payload(text)
    weak_items = _coerce_critique(payload, max_items=max_items)
    if not weak_items:
        logger.info("critique: 0 weak items proposed — no-op")
        return result

    for i, item in enumerate(weak_items):
        logger.info(
            "critique: weak[%d] kind=%s name=%r reason=%r",
            i, item["kind"], item["name"], item["reason"],
        )

    if repo_root is None:
        logger.warning("critique: repo_root not set — cannot run renames; skipping apply")
        return result

    for item in weak_items:
        kind = item["kind"]
        name = item["name"]
        if kind == "feature":
            files = result.features.get(name)
            if not files:
                continue
            proposal = _rename_feature_with_tools(
                feature_name=name,
                files=files,
                repo_root=repo_root,
                client=client,
                model=resolved_model,
                tracker=tracker,
                tool_budget=tool_budget,
            )
            if proposal is None:
                logger.info("critique: %r — model declined to rename", name)
                continue
            new = proposal["new_name"]
            if not _is_materially_better(name, new):
                logger.info(
                    "critique: %r → %r rejected (not materially better)",
                    name, new,
                )
                continue
            ok = _rewrite_feature_name(
                result, name, new, proposal.get("description", ""),
            )
            if ok:
                logger.info("critique: renamed feature %r → %r", name, new)
        else:  # kind == "flow"
            feat = item.get("feature", "")
            # Flow rename via single Sonnet call (no tools) — cheaper
            # and sufficient for short names. Reuse the same client.
            try:
                resp = client.messages.create(
                    model=resolved_model,
                    max_tokens=512,
                    system=(
                        "Rename one user-flow that has a vague name. "
                        "Return ONLY JSON: {\"new_name\": \"...\"}. "
                        "Avoid generic words. Empty new_name keeps "
                        "the original."
                    ),
                    messages=[{"role": "user", "content": json.dumps({
                        "feature": feat,
                        "current_name": name,
                        "description": item.get("reason", ""),
                    })}],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("critique: flow rename failed for %r (%s)", name, exc)
                continue
            if tracker is not None:
                u = getattr(resp, "usage", None)
                ti = int(getattr(u, "input_tokens", 0) or 0)
                to = int(getattr(u, "output_tokens", 0) or 0)
                if ti or to:
                    tracker.record(
                        provider="anthropic", model=resolved_model,
                        input_tokens=ti, output_tokens=to,
                        label="critique-flow-rename",
                    )
            txt = ""
            for b in getattr(resp, "content", []) or []:
                bt = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else "")
                if bt == "text":
                    txt += getattr(b, "text", "") if not isinstance(b, dict) else b.get("text", "")
            data = _parse_merges_payload(txt) or {}
            new = (data.get("new_name") or "").strip() if isinstance(data, dict) else ""
            if not new or not _is_materially_better(name, new):
                logger.info("critique: flow %r → %r rejected", name, new)
                continue
            ok = _rewrite_flow_name(result, feat, name, new)
            if ok:
                logger.info("critique: renamed flow %r → %r in %r", name, new, feat)

    return result
