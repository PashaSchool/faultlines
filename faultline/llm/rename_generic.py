"""Fix #4 from Fixable-accuracy work: rename generic feature names.

The eval flagged ~12 features per n8n scan with names like ``Utils``,
``Constants``, ``Decorators``, ``Dto``, ``Backend Common`` — names
that describe what something is *technically* but not what it does
*for the product*. Such names tank "naming accuracy" because an EM
opening the dashboard cannot tell what a "Constants" feature owns.

This module's job is mechanical: identify features whose normalized
name appears in :data:`_GENERIC_NAMES`, batch them into one Haiku
call with their sample paths, and accept the model's specific
2–4 word business-language rename. The original name is preserved
as an alias so existing references keep resolving.

Runs late in the pipeline (after dedup + sub-decompose so the input
is stable) and is opt-out via ``rename_generic=False``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, TYPE_CHECKING

from faultline.llm.cost import CostTracker, deterministic_params

if TYPE_CHECKING:
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_SAMPLE_PATHS = 6
MAX_RENAMES_PER_PASS = 30  # cap so a misbehaving call can't rewrite an entire scan

# Names that score "bad" on the eval's auto-naming-accuracy rule
# UNLESS they literally match the package path (handled below). These
# are picked from the actual eval output — every name listed here
# appeared as a "bad name" on at least one of the 6 scanned repos.
_GENERIC_NAMES: frozenset[str] = frozenset({
    "api",
    "backend common",
    "backend test utils",
    "commands",
    "components",
    "config",
    "constants",
    "decorators",
    "developer platform",
    "di",
    "documentation",  # only when paths aren't actually under docs/
    "dto",
    "entities",
    "future",
    "hooks",
    "http client foundation",
    "maintenance",
    "platform infrastructure",
    "platform primitives",
    "scenarios",
    "scripts front",
    "search analytics & platform",
    "tournament",
    "ui",
    "utils",
    "web",
})

_PROTECTED_NAMES: frozenset[str] = frozenset({
    "shared-infra", "documentation", "examples",
})


_SYSTEM_PROMPT = """\
You rename generic-named features in a code-analysis tool to specific
business-language names that an engineering manager would actually use.

The user gives you a list of features, each with its current generic
name and 4-6 sample file paths. For each, propose a 2–4 word name in
title case that describes what the feature does FOR THE PRODUCT, not
what it is TECHNICALLY. Examples:

  GOOD: "Workflow Constants" instead of "Constants"
  GOOD: "API Schema Types" instead of "Dto"
  GOOD: "Backend Helpers" instead of "Backend Common"
  GOOD: "Feature Flags" instead of "Future"

  BAD:  "Constants" (still generic)
  BAD:  "TypeScript Types" (technical, not product)
  BAD:  "Various Shared Code" (vague)

If the paths really do contain only undifferentiated technical
plumbing and a specific name would be misleading, reply with
``KEEP`` for that feature — but use this sparingly; most generic
names CAN be specialized given the paths.

Reply with a single JSON object: ``{"renames": [{"name": "<original>",
"new": "<proposed>"}, ...]}``. Use ``"new": "KEEP"`` to skip a feature.
No prose outside the JSON.
"""


def _format_user_message(
    candidates: list[tuple[str, list[str]]],
) -> str:
    """Format the candidate list into a single user prompt."""
    lines = ["Features to rename:\n"]
    for name, paths in candidates:
        lines.append(f"\n## {name}")
        for p in paths[:DEFAULT_SAMPLE_PATHS]:
            lines.append(f"  - {p}")
    lines.append(
        "\nReply with one JSON object:\n"
        '{"renames": [{"name": "<original>", "new": "<proposed or KEEP>"}, ...]}'
    )
    return "\n".join(lines)


def _name_matches_dominant_path(name: str, paths: list[str]) -> bool:
    """A generic name like ``Utils`` is OK when paths really are
    ``packages/utils/*`` — the package literally is named utils."""
    if not paths:
        return False
    n = name.strip().lower()
    prefixes: dict[str, int] = {}
    for p in paths:
        parts = p.split("/")
        if len(parts) >= 2:
            key = parts[1].lower()
            prefixes[key] = prefixes.get(key, 0) + 1
    if not prefixes:
        return False
    top = max(prefixes, key=prefixes.get)
    return n == top or n.replace(" ", "-") == top or n.replace(" ", "") == top


def _select_candidates(
    result: "DeepScanResult",
) -> list[tuple[str, list[str]]]:
    """Find features that need renaming. Returns (name, sample_paths)."""
    out: list[tuple[str, list[str]]] = []
    for name, paths in result.features.items():
        if name in _PROTECTED_NAMES:
            continue
        norm = name.strip().lower()
        if norm not in _GENERIC_NAMES:
            continue
        # ``Utils`` covering ``packages/utils/*`` is OK — skip
        if _name_matches_dominant_path(name, paths):
            continue
        # ``documentation`` is only generic when paths aren't actually docs
        if norm == "documentation" and any("/docs/" in p or p.startswith("docs/") for p in paths):
            continue
        out.append((name, list(paths)))
        if len(out) >= MAX_RENAMES_PER_PASS:
            break
    return out


def _parse_response(text: str) -> dict[str, str]:
    """Extract {original_name: new_name} from the model response."""
    # Find the first JSON object in the response
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    renames = data.get("renames", [])
    if not isinstance(renames, list):
        return {}
    out: dict[str, str] = {}
    for entry in renames:
        if not isinstance(entry, dict):
            continue
        old = entry.get("name", "").strip()
        new = entry.get("new", "").strip()
        if old and new and new.upper() != "KEEP":
            out[old] = new
    return out


def rename_generic_features(
    result: "DeepScanResult",
    *,
    api_key: str | None = None,
    model: str | None = None,
    tracker: CostTracker | None = None,
) -> "DeepScanResult":
    """Identify and rename features with generic names via one Haiku call.

    No-op when no candidates qualify or when the API call fails — the
    pipeline must keep running. Returns the (possibly mutated) input
    so callers can chain.
    """
    candidates = _select_candidates(result)
    if not candidates:
        return result

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("rename_generic: no API key — skipping")
        return result

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("rename_generic: anthropic package missing — skipping")
        return result

    chosen_model = model or DEFAULT_MODEL
    client = Anthropic(api_key=api_key)
    params = deterministic_params(chosen_model)

    try:
        response = client.messages.create(
            model=chosen_model,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _format_user_message(candidates)}],
            max_tokens=2048,
            **params,
        )
    except Exception as exc:  # noqa: BLE001 — opportunistic, never block the pipeline
        logger.warning("rename_generic: API call failed (%s) — skipping", exc)
        return result

    if tracker is not None:
        tracker.record_usage(
            model=chosen_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stage="rename_generic",
        )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    renames = _parse_response(text)
    if not renames:
        logger.info("rename_generic: model returned no renames")
        return result

    # Apply renames atomically (skip ones that collide with existing names)
    from faultline.llm.critique import _rewrite_feature_name
    applied = 0
    for old, new in renames.items():
        if _rewrite_feature_name(result, old, new):
            applied += 1
        else:
            logger.debug("rename_generic: skip %r → %r (collision or missing)", old, new)

    if applied:
        logger.info("rename_generic: renamed %d generic feature(s)", applied)
    return result
