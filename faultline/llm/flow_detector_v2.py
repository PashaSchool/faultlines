"""Tool-augmented flow detection (Sprint 4).

Replaces the legacy Haiku per-feature call (``flow_detector.detect_flows_llm``)
with a tool-augmented Sonnet pass that grounds every flow in a real
route handler / API endpoint / event subscription.

The two new Sprint 4 tools (``find_route_handlers``,
``find_event_handlers``) are exposed automatically via
``faultline.llm.tools.TOOL_SCHEMAS`` — they ship to the LLM through
the same dispatcher as Sprint 1's tools.

Public entry points:

    detect_flows_for_feature(name, files, ...)
        per-feature pass — returns a list of validated flow dicts or
        None when the model returns nothing usable.

    detect_flows_with_tools(result, ...)
        top-level — walks ``result.features`` and writes flows +
        flow_descriptions for each non-protected, non-library feature.

Behaviour is opt-in via ``pipeline.run(tool_flows=True)``. Library
mode (``is_library=True``) skips the entire pass — same suppression
that Sprint 1's deep_scan_workspace honours.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dedup import _PROTECTED_NAMES
from .tool_use_scan import DEFAULT_MODEL, tool_use_scan

if TYPE_CHECKING:  # pragma: no cover
    from .cost import CostTracker
    from .sonnet_scanner import DeepScanResult


logger = logging.getLogger(__name__)


DEFAULT_TOOL_BUDGET = 6
DEFAULT_MAX_FLOWS_PER_FEATURE = 8
# Generic flow names that almost always indicate the model failed to
# ground in a real entry point. Reject and keep the parent feature's
# flow set untouched.
_GENERIC_FLOW_NAMES: frozenset[str] = frozenset({
    "process-data-flow", "manage-things", "view-things",
    "handle-request", "handle-event", "do-thing", "main-flow",
    "default-flow",
})


# ── Prompt ─────────────────────────────────────────────────────────────


_FLOW_SYSTEM_PROMPT = """\
You are a senior engineer mapping a feature's user-facing flows.
The feature already has a business name and a coherent file list.
Your job is to discover real entry points and group them into named
user flows.

WORKFLOW
1. Run find_route_handlers and find_event_handlers, scoped to this
   feature's path prefix when possible.
2. For ambiguous matches, read_file_head to confirm purpose.
3. Group entry points into 1-8 named flows. A flow is a user
   journey — multiple endpoints CAN share one flow, but a flow with
   no real entry point should be dropped.
4. Return final JSON. Do not narrate.

NAMING RULES (HARD)
- Flow names use imperative business actions ("create-document",
  "cancel-subscription", "import-csv-contacts").
- No generic placeholders: "process-data-flow", "manage-things",
  "main-flow", "default-flow", "handle-request".
- 1-8 flows per feature. Cap at 8.
- If you cannot find any real entry point in the feature's files,
  return {"flows": []} and we will skip flow detection for this
  feature.

OUTPUT (final message, after tool use is done)
Respond with ONLY a JSON object, no prose:

{
  "flows": [
    {
      "name": "create-document",
      "description": "User uploads a PDF and configures recipients.",
      "entry_point_file": "apps/web/app/routes/documents.new.tsx",
      "entry_point_line": 24
    }
  ]
}

CONSTRAINTS
- Every flow MUST have entry_point_file. entry_point_line may be 0
  if the entry is a whole-file convention (e.g. Next.js page.tsx).
- Each description: one sentence, business-readable.
"""


# ── Helpers ────────────────────────────────────────────────────────────


def _validate_flows(
    raw_flows: list[Any],
    *,
    max_flows: int = DEFAULT_MAX_FLOWS_PER_FEATURE,
) -> list[dict] | None:
    """Return a cleaned list of flow dicts, or ``None`` when none are valid.

    Each flow needs a non-generic name and an entry_point_file string.
    Duplicate names within one feature are dropped (first wins).
    """
    if not isinstance(raw_flows, list):
        return None

    out: list[dict] = []
    seen: set[str] = set()
    for entry in raw_flows[:max_flows]:
        if not isinstance(entry, dict):
            continue
        name = (entry.get("name") or "").strip()
        if not name or name in seen or name in _GENERIC_FLOW_NAMES:
            continue
        entry_file = (entry.get("entry_point_file") or "").strip()
        if not entry_file:
            continue
        seen.add(name)
        out.append({
            "name": name,
            "description": (entry.get("description") or "").strip(),
            "entry_point_file": entry_file,
            "entry_point_line": int(entry.get("entry_point_line") or 0),
        })

    return out or None


# ── Per-feature pass ───────────────────────────────────────────────────


def detect_flows_for_feature(
    *,
    name: str,
    files: list[str],
    repo_root: Path,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: "CostTracker | None" = None,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    max_flows: int = DEFAULT_MAX_FLOWS_PER_FEATURE,
) -> list[dict] | None:
    """Run a tool-augmented flow detection pass for one feature.

    Returns a validated list of flow dicts on success, or ``None``
    when no usable flows were proposed (model declined, JSON
    unparseable, all flows lacked entry points, etc.).
    """
    if not files:
        return None

    if client is None:
        import os
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            logger.warning("flow_v2(%s): no API key — skipping", name)
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
            cost_label=f"flow-v2:{name}",
            system_prompt=_FLOW_SYSTEM_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001 - opportunistic
        logger.warning("flow_v2(%s): scan failed (%s) — keeping feature flowless", name, exc)
        return None

    if not parsed:
        return None

    raw = parsed.get("flows") or []
    return _validate_flows(raw, max_flows=max_flows)


# ── Top-level pass ─────────────────────────────────────────────────────


def detect_flows_with_tools(
    result: "DeepScanResult",
    *,
    repo_root: Path | None = None,
    is_library: bool = False,
    client: Any | None = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: "CostTracker | None" = None,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    max_flows: int = DEFAULT_MAX_FLOWS_PER_FEATURE,
) -> "DeepScanResult":
    """Walk ``result.features`` and attach grounded flows to each.

    Skips:
      - Library repos (``is_library=True``) — flows are noise on
        bibliotheques; same suppression as Sprint 1.
      - Protected synthetic buckets (``documentation``, ``shared-infra``,
        ``examples``).
      - Features with no source files.

    Mutates ``result.flows`` and ``result.flow_descriptions`` in place
    AND returns ``result``. Existing entries are overwritten only when
    a non-empty validated list comes back from the LLM; otherwise they
    are left untouched (so the legacy Haiku path can be a fallback).
    """
    if not result or not result.features:
        return result
    if is_library:
        logger.info("flow_v2: library mode → skipping pass")
        return result
    if repo_root is None:
        logger.warning("flow_v2: repo_root not provided — skipping pass")
        return result

    candidates = [
        (name, list(files))
        for name, files in result.features.items()
        if name not in _PROTECTED_NAMES and files
    ]
    if not candidates:
        return result

    logger.info("flow_v2: running on %d feature(s)", len(candidates))

    for name, files in candidates:
        flows = detect_flows_for_feature(
            name=name,
            files=files,
            repo_root=repo_root,
            client=client,
            api_key=api_key,
            model=model,
            tracker=tracker,
            tool_budget=tool_budget,
            max_flows=max_flows,
        )
        if not flows:
            continue

        flow_names = [f["name"] for f in flows]
        result.flows[name] = flow_names

        # Stash per-flow description with an "(entry: file:line)"
        # suffix so the entry point survives end-to-end through the
        # existing string-only flow_descriptions carrier. The
        # downstream cli helper :func:`_inject_new_pipeline_flows`
        # parses the suffix back out into the first-class
        # Flow.entry_point_file / Flow.entry_point_line fields and
        # strips it from the visible description.
        per_flow_descs: dict[str, str] = {}
        for f in flows:
            desc = f.get("description") or ""
            entry = f.get("entry_point_file") or ""
            line = f.get("entry_point_line") or 0
            tag = f" (entry: {entry}:{line})" if entry else ""
            per_flow_descs[f["name"]] = (desc + tag).strip()
        result.flow_descriptions[name] = per_flow_descs

        logger.info(
            "flow_v2: %s → %d flow(s)", name, len(flow_names),
        )

    return result
