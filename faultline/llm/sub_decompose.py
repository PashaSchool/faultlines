"""Sub-decomposition of oversized features (Sprint 3).

After Sprint 1 (per-package tool-use) + Sprint 2 (cross-cluster dedup),
some features land too coarse — e.g. ``document-signing 328 files`` on
documenso, ``web/surveys 982 files`` on formbricks. A 982-file feature
isn't actionable on a dashboard: a regression in one sub-area gets
diluted by stable code in another.

This module walks the feature map after dedup, picks features above a
size threshold (default 200 files), and runs a focused
tool-augmented split pass on each. The Sprint 1 tool dispatcher and
message loop are reused via :func:`tool_use_scan` with a different
system prompt.

Public entry points:

    sub_decompose_feature(name, files, ...)
        per-feature pass — returns a list of sub-feature dicts or
        None when no clean split is found.

    sub_decompose_oversized(result, ...)
        top-level — walks ``result.features``, applies the per-feature
        pass where appropriate, rewrites flows / descriptions /
        flow_descriptions to the new sub-feature keys.

Behaviour is opt-in via ``pipeline.run(sub_decompose=True)`` (or the
``--sub-decompose`` CLI flag). On any failure the original feature is
kept unchanged — same opportunistic pattern as Sprint 2 dedup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dedup import _PROTECTED_NAMES
from .tool_use_scan import (
    DEFAULT_MODEL,
    tool_use_scan,
)

if TYPE_CHECKING:  # pragma: no cover
    from .cost import CostTracker
    from .sonnet_scanner import DeepScanResult


logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD = 200
# When ``threshold`` would skip the entire repo because every
# feature is well under 200 files (small single-package repos like
# faultline-self), scale the threshold down. Formula:
# ``min(threshold, max(MIN_DYNAMIC, total_source_files / 4))``.
# A 60-file repo gets a 15-file threshold, so a 47-file ``Analysis
# Core`` feature still gets sub-decomposed. A 2000-file monorepo
# stays at 200, unchanged.
MIN_DYNAMIC_THRESHOLD = 15
DEFAULT_MAX_SUB_FEATURES = 6
# Smaller than Sprint 1's per-package 15 — the input is one cohesive
# feature, not a whole package, so fewer probing reads suffice.
DEFAULT_TOOL_BUDGET = 8


# ── Prompt ─────────────────────────────────────────────────────────────


_SUB_DECOMPOSE_SYSTEM_PROMPT = """\
You are a senior engineer subdividing a single feature into its real
internal sub-domains. The feature already has a coherent overall
purpose and a business name. Your job is to discover the 2-6 distinct
sub-areas that live inside it.

WORKFLOW
1. Look at the feature name and the file list.
2. For files whose role is ambiguous, USE TOOLS:
   - read_file_head to see imports and top-level declarations
   - grep_pattern to trace concepts across the feature
   - get_file_commits to see what work the file is associated with
3. Group files into 2-6 sub-features. Every input file must belong
   to exactly one sub-feature.
4. Return final JSON. Do not narrate.

NAMING RULES (HARD)
- Sub-feature names must NOT include the parent feature's name.
  We add the parent prefix on our side. So write "field-validation",
  not "document-signing/field-validation".
- No generic names: lib, utils, shared, general, core, misc, common,
  base, helpers, types, ui, api, main.
- Each sub-feature: 1-3 word business name a PM would recognize.

OUTPUT (final message, after tool use is done)
Respond with ONLY a JSON object, no prose:

{
  "features": [
    {
      "name": "field-validation",
      "paths": ["lib/validation/field-rules.ts", "..."],
      "description": "Validation rules for individual signing fields."
    }
  ]
}

CONSTRAINTS
- 2 to 6 features. If no clean split exists, return {"features": []}
  and we will keep the parent intact.
- Cover EVERY input file in exactly one sub-feature.
- Keep descriptions to one sentence.
"""


# ── Helpers ────────────────────────────────────────────────────────────


def _validate_split(
    subfeatures: list[dict],
    parent_files: list[str],
    max_sub: int = DEFAULT_MAX_SUB_FEATURES,
) -> list[dict] | None:
    """Return ``subfeatures`` if it is a clean split of ``parent_files``.

    A clean split means:
      - Between 2 and ``max_sub`` entries.
      - Every entry has a non-empty name and at least one path.
      - The union of all paths equals the parent file set exactly.
        No leftovers, no extra paths.
      - No two sub-features share the same name (after stripping).

    Returns ``None`` if any check fails. The caller leaves the parent
    feature intact.
    """
    if not isinstance(subfeatures, list):
        return None
    if not 2 <= len(subfeatures) <= max_sub:
        return None

    parent_set = set(parent_files)
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    cleaned: list[dict] = []

    for entry in subfeatures:
        if not isinstance(entry, dict):
            return None
        name = (entry.get("name") or "").strip()
        if not name or name.lower() in {"core", "lib", "utils", "shared",
                                          "general", "misc", "common",
                                          "base", "helpers", "main"}:
            return None
        if name in seen_names:
            return None
        seen_names.add(name)

        paths = entry.get("paths") or []
        if not isinstance(paths, list) or not paths:
            return None
        # Normalize and dedupe path list
        clean_paths: list[str] = []
        for p in paths:
            if not isinstance(p, str):
                return None
            p = p.strip()
            if not p:
                continue
            if p in seen_paths:
                # File claimed twice across sub-features — invalid split
                return None
            seen_paths.add(p)
            clean_paths.append(p)
        if not clean_paths:
            return None

        cleaned.append({
            "name": name,
            "paths": sorted(clean_paths),
            "description": (entry.get("description") or "").strip(),
        })

    # Must cover the parent set exactly.
    if seen_paths != parent_set:
        return None

    return cleaned


# ── Per-feature pass ───────────────────────────────────────────────────


def sub_decompose_feature(
    *,
    name: str,
    files: list[str],
    repo_root: Path,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: "CostTracker | None" = None,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    max_sub: int = DEFAULT_MAX_SUB_FEATURES,
) -> list[dict] | None:
    """Split a single oversized feature into 2-``max_sub`` sub-features.

    Returns the validated list of sub-feature dicts on success, or
    ``None`` when the model declines to split, returns garbage, or
    proposes a split that does not cover every parent file.
    """
    if not files:
        return None

    if client is None:
        import os
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("sub_decompose(%s): no API key — skipping", name)
            return None
        client = anthropic.Anthropic(api_key=key)

    resolved_model = model or DEFAULT_MODEL

    try:
        parsed = tool_use_scan(
            package_name=name,
            files=files,
            repo_root=repo_root,
            client=client,
            model=resolved_model,
            tool_budget=tool_budget,
            tracker=tracker,
            cost_label=f"sub-decompose:{name}",
            system_prompt=_SUB_DECOMPOSE_SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001 - opportunistic
        logger.warning("sub_decompose(%s): scan failed (%s) — keeping parent", name, exc)
        return None

    if not parsed:
        return None

    raw_subs = parsed.get("features") or []
    if not raw_subs:
        # Model declined to split — keep parent.
        logger.info("sub_decompose(%s): model returned 0 sub-features", name)
        return None

    validated = _validate_split(raw_subs, files, max_sub=max_sub)
    if validated is None:
        logger.info(
            "sub_decompose(%s): %d sub-features failed validation — keeping parent",
            name, len(raw_subs),
        )
        return None

    return validated


# ── Top-level pass ─────────────────────────────────────────────────────


def sub_decompose_oversized(
    result: "DeepScanResult",
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_sub: int = DEFAULT_MAX_SUB_FEATURES,
    repo_root: Path | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: "CostTracker | None" = None,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    locked_names: frozenset[str] | None = None,
) -> "DeepScanResult":
    """Walk ``result.features`` and split anything above ``threshold``.

    Mutates ``result`` in place AND returns it. Skips:
      - Features at or below ``threshold`` (cheap path: no LLM call).
      - Protected synthetic buckets (``documentation``, ``shared-infra``,
        ``examples``).

    For each oversized real feature:
      - Call :func:`sub_decompose_feature`.
      - If a valid split comes back, replace the feature with
        ``{name}/{sub}`` keys, sort by file count desc, and rewrite
        any flows / descriptions / flow_descriptions to follow the
        new keys.
      - Otherwise leave the parent unchanged.
    """
    if not result or not result.features:
        return result

    # Dynamic threshold: scale down on small repos so Sprint 3
    # actually fires. A 60-file repo's largest feature might be
    # 15-30 files — well under the 200 default. Without scaling,
    # sub-decompose never runs on small projects (faultline-self
    # showed this clearly: 7 features, biggest 47 files, 0 splits).
    total_source_files = sum(
        len(files) for name, files in result.features.items()
        if name not in _PROTECTED_NAMES
    )
    effective_threshold = min(
        threshold,
        max(MIN_DYNAMIC_THRESHOLD, total_source_files // 4),
    )
    if effective_threshold < threshold:
        logger.info(
            "sub_decompose: small repo (%d source files) — using "
            "dynamic threshold %d (default %d)",
            total_source_files, effective_threshold, threshold,
        )

    # Stability lock: when a name is in ``locked_names`` (typically
    # the user's ``.faultline.yaml`` features + auto_aliases), skip
    # sub-decomposition entirely. The user (or a previous run)
    # decided this feature is canonical at this granularity;
    # re-splitting it on every scan generates churn and destroys
    # cross-run feature-name stability.
    locked: frozenset[str] = locked_names or frozenset()

    # Catch-all suffixes that almost always indicate a leftover bucket
    # rather than a real feature. Trigger sub-decompose at a much lower
    # threshold (50 files) so an aggressive ``web/shell`` 680-file
    # bucket gets split into proper sub-features instead of riding
    # along under the default ``threshold`` budget.
    catchall_suffixes = (
        "/shell", "/core", "/common", "/main", "/app", "/misc",
        "/utils", "/lib", "/base", "/general", "/leftover", "/residual",
    )
    catchall_threshold = max(50, MIN_DYNAMIC_THRESHOLD)

    # Snapshot keys so we can mutate ``result.features`` while iterating.
    candidates = []
    skipped_locked = 0
    for name, files in result.features.items():
        if name in _PROTECTED_NAMES:
            continue
        is_catchall = any(name.endswith(suf) for suf in catchall_suffixes)
        size_floor = catchall_threshold if is_catchall else effective_threshold
        if len(files) <= size_floor:
            continue
        if name in locked:
            skipped_locked += 1
            continue
        candidates.append((name, list(files)))

    if skipped_locked:
        logger.info(
            "sub_decompose: %d feature(s) skipped because they are "
            "locked by user config or auto_aliases",
            skipped_locked,
        )
    if not candidates:
        return result

    if repo_root is None:
        logger.warning(
            "sub_decompose: repo_root not provided — cannot read files; skipping pass",
        )
        return result

    logger.info(
        "sub_decompose: %d feature(s) over threshold=%d",
        len(candidates), effective_threshold,
    )

    for name, files in candidates:
        sub_features = sub_decompose_feature(
            name=name,
            files=files,
            repo_root=repo_root,
            client=client,
            api_key=api_key,
            model=model,
            tracker=tracker,
            tool_budget=tool_budget,
            max_sub=max_sub,
        )
        if sub_features is None:
            continue

        # Apply the split. Remove the parent entry, write each sub-
        # feature under ``{name}/{sub_name}``.
        parent_desc = result.descriptions.get(name, "")
        parent_flows = list(result.flows.get(name, []))
        parent_flow_descs = dict(result.flow_descriptions.get(name, {}))

        result.features.pop(name, None)
        result.descriptions.pop(name, None)
        result.flows.pop(name, None)
        result.flow_descriptions.pop(name, None)

        # Largest sub-feature inherits the parent's flows + flow_descs
        # so we don't lose downstream attribution. Subsequent sub-
        # features get only their own (none, since this is a new pass).
        sub_features.sort(key=lambda s: -len(s["paths"]))
        primary_key: str | None = None

        for i, sub in enumerate(sub_features):
            sub_name = sub["name"]
            full_key = f"{name}/{sub_name}"
            result.features[full_key] = sub["paths"]
            if sub.get("description"):
                result.descriptions[full_key] = sub["description"]
            if i == 0:
                primary_key = full_key

        if primary_key is not None:
            if parent_flows and primary_key not in result.flows:
                result.flows[primary_key] = parent_flows
            if parent_flow_descs and primary_key not in result.flow_descriptions:
                result.flow_descriptions[primary_key] = parent_flow_descs
            # Preserve any parent-feature description by attaching it
            # to the largest sub-feature when the model didn't supply
            # its own.
            if parent_desc and primary_key not in result.descriptions:
                result.descriptions[primary_key] = parent_desc

        logger.info(
            "sub_decompose: %s (%d files) → %d sub-features",
            name, len(files), len(sub_features),
        )

    return result
