"""Sprint 12 Day 4-5 — per-flow symbol picker (Layer B).

For each flow, ask Haiku which exported symbols from the flow's
candidate files actually participate in the user journey. The output
is a list of ``(file, symbol)`` pairs that the deterministic resolver
turns into ``SymbolRange`` entries with line_start / line_end.

Architecture
============

    pick_flow_symbols(flow, candidate_files, source_loader, client)
        1. For each candidate file, call ``list_exported_symbols`` to
           get every top-level symbol with its (start, end) range.
        2. Build one Haiku prompt with flow name + description +
           per-file symbol list.
        3. Send to Haiku; parse JSON response.
        4. Filter response against the actual symbol map (drop
           hallucinated names) and resolve to ``SymbolRange``.

Cache: per-flow, keyed by (flow_name, hash(candidate_files+symbols)).
Cache file: ``~/.faultline/flow-symbols-<repo_slug>.json``. Invalidates
when the candidate-set hash changes — adding a file or renaming a
symbol re-asks Haiku, but a pure flow-name re-attribution does not.

Public surface
==============

    pick_flow_symbols(flow, candidates, ...) -> list[PickedSymbol]
        One-flow entry point. Mocked in unit tests.

    resolve_flow_symbols(result, repo_root, ...) -> int
        Pipeline-level entry point. Walks every flow in the
        DeepScanResult; populates ``flow_participants`` with
        symbol-level data. Returns the count of flows updated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

from faultline.llm.cost import CostTracker, deterministic_params

if TYPE_CHECKING:
    from faultline.llm.sonnet_scanner import DeepScanResult
    from faultline.models.types import SymbolRange

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4_096
DEFAULT_CACHE_DIR = Path.home() / ".faultline"
_CACHE_VERSION = 1

# Token budget guards. A flow with 20 candidate files × 30 symbols each
# would blow the prompt; cap before hitting Haiku.
MAX_CANDIDATE_FILES = 15
MAX_SYMBOLS_PER_FILE = 25
CONFIDENCE_FLOOR = 3   # accept symbols with confidence ≥ 3


@dataclass
class CandidateSymbol:
    """One exported symbol shown to Haiku for selection."""

    file: str
    name: str
    kind: str        # "function" | "class" | "const" | "type" | "enum"
    start_line: int
    end_line: int


@dataclass
class PickedSymbol:
    """Haiku's verdict for one symbol — kept or dropped."""

    file: str
    name: str
    confidence: int  # 1..5
    reasoning: str = ""


@dataclass
class FlowSymbolBundle:
    """Resolved symbols for one flow, ready to attach to a Flow."""

    flow_name: str
    feature_owner: str
    picked: list[PickedSymbol] = field(default_factory=list)
    # The resolved (file, SymbolRange) pairs — built by
    # ``resolve_picked`` from a candidate map.
    ranges: dict[str, list["SymbolRange"]] = field(default_factory=dict)


_SYSTEM_PROMPT = """\
You select code symbols (functions, classes, constants) that
participate in a user-flow's journey.

For each flow you receive a description and a list of CANDIDATE
symbols across one or more files. Return ONLY the symbols that the
user-flow actually exercises end-to-end — the route handler / event
binding that triggers it, the controller / service that orchestrates
it, the data layer that persists it, the validators it runs through.

EXCLUDE:
  - Pure type / interface re-exports
  - Helper utilities that the flow merely imports for unrelated
    reasons (date formatters, generic `clsx` helpers, logging)
  - Tests
  - UI primitives that are reused everywhere (Button, Modal) UNLESS
    the flow's UX depends on this specific component

When in doubt about a borderline symbol, prefer SMALLER over LARGER —
this set drives test-coverage attribution downstream, so noise
inflates artificial coverage.

OUTPUT (JSON only, no prose, no markdown fences)
{
  "selected": [
    {"file": "<rel/path.ts>",
     "symbol": "<exact_name>",
     "confidence": 1..5,
     "reasoning": "<one short sentence>"}
  ]
}

Confidence:
  5 — clearly the entry point or core handler of this flow
  4 — definitely participates (validator, persistence, render)
  3 — plausibly participates; include if the flow's name explicitly
      mentions it
  1-2 — do not include

Cover only what the flow actually exercises. Empty ``selected`` is
acceptable when none of the candidates fit.
"""


# ── Anthropic client protocol ─────────────────────────────────────────


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...  # pragma: no cover


# ── Pure helpers ──────────────────────────────────────────────────────


def _slugify_repo(repo_path: str) -> str:
    if not repo_path:
        return "unknown"
    name = Path(repo_path).name or "unknown"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"


def _candidates_hash(candidates: list[CandidateSymbol]) -> str:
    """Stable hash over (file, symbol, start_line). Detects renames /
    additions but ignores reordering."""
    payload = sorted([(c.file, c.name, c.start_line) for c in candidates])
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def _trim_candidates(
    candidates: list[CandidateSymbol],
) -> list[CandidateSymbol]:
    """Cap to MAX_CANDIDATE_FILES files, MAX_SYMBOLS_PER_FILE per file."""
    by_file: dict[str, list[CandidateSymbol]] = {}
    for c in candidates:
        by_file.setdefault(c.file, []).append(c)
    files = list(by_file.keys())[:MAX_CANDIDATE_FILES]
    out: list[CandidateSymbol] = []
    for f in files:
        out.extend(by_file[f][:MAX_SYMBOLS_PER_FILE])
    return out


def _build_prompt(
    flow_name: str,
    flow_description: str,
    candidates: list[CandidateSymbol],
) -> str:
    by_file: dict[str, list[CandidateSymbol]] = {}
    for c in candidates:
        by_file.setdefault(c.file, []).append(c)
    payload = {
        "flow": {
            "name": flow_name,
            "description": flow_description or "",
        },
        "candidates": [
            {
                "file": f,
                "symbols": [
                    {"name": c.name, "kind": c.kind}
                    for c in by_file[f]
                ],
            }
            for f in by_file
        ],
    }
    return (
        "Pick the symbols that participate in this user-flow.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def _parse_response(text: str) -> list[dict] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    selected = data.get("selected")
    return selected if isinstance(selected, list) else None


def _coerce_picked(entry: dict) -> PickedSymbol | None:
    file = (entry.get("file") or "").strip()
    name = (entry.get("symbol") or entry.get("name") or "").strip()
    if not file or not name:
        return None
    try:
        confidence = int(entry.get("confidence", 3))
    except (TypeError, ValueError):
        confidence = 3
    confidence = max(1, min(5, confidence))
    reasoning = (entry.get("reasoning") or "").strip()
    return PickedSymbol(
        file=file, name=name, confidence=confidence, reasoning=reasoning,
    )


def _filter_by_candidates(
    picked: list[PickedSymbol],
    candidates: list[CandidateSymbol],
) -> list[PickedSymbol]:
    """Drop hallucinated picks — keep only (file, name) pairs that
    actually exist in the candidate list."""
    valid = {(c.file, c.name) for c in candidates}
    return [p for p in picked if (p.file, p.name) in valid]


def resolve_picked(
    picked: list[PickedSymbol],
    candidates: list[CandidateSymbol],
) -> dict[str, list["SymbolRange"]]:
    """Map picked (file, name) pairs → file → list[SymbolRange]."""
    from faultline.models.types import SymbolRange  # local: avoid cycle

    by_key: dict[tuple[str, str], CandidateSymbol] = {
        (c.file, c.name): c for c in candidates
    }
    out: dict[str, list[SymbolRange]] = {}
    for p in picked:
        c = by_key.get((p.file, p.name))
        if c is None:
            continue
        out.setdefault(p.file, []).append(SymbolRange(
            name=c.name,
            start_line=c.start_line,
            end_line=c.end_line,
            kind=c.kind,
        ))
    return out


# ── Cache ─────────────────────────────────────────────────────────────


def _cache_path(cache_dir: Path, repo_slug: str) -> Path:
    return cache_dir / f"flow-symbols-{repo_slug}.json"


def _load_cache(
    cache_dir: Path | None,
    repo_slug: str,
) -> dict[str, dict[str, Any]]:
    if cache_dir is None:
        return {}
    path = _cache_path(cache_dir, repo_slug)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if data.get("version") != _CACHE_VERSION:
        return {}
    raw = data.get("flows", {})
    return raw if isinstance(raw, dict) else {}


def _save_cache(
    cache_dir: Path | None,
    repo_slug: str,
    flow_map: dict[str, dict[str, Any]],
) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CACHE_VERSION,
            "flows": flow_map,
        }
        _cache_path(cache_dir, repo_slug).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        )
    except OSError as exc:
        logger.warning("flow_symbols: cache save failed (%s)", exc)


# ── Single-flow API ───────────────────────────────────────────────────


def pick_flow_symbols(
    flow_name: str,
    flow_description: str,
    candidates: list[CandidateSymbol],
    *,
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    tracker: CostTracker | None = None,
    confidence_floor: int = CONFIDENCE_FLOOR,
) -> list[PickedSymbol]:
    """Ask Haiku which symbols participate in this flow.

    Returns the filtered, confidence-floored list. Empty list when no
    API key, no candidates, or call fails — never raises.
    """
    if not candidates:
        return []

    candidates = _trim_candidates(candidates)

    if client is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("flow_symbols: no API key — skipping pick")
            return []
        try:
            from anthropic import Anthropic
        except ImportError:
            logger.warning("flow_symbols: anthropic package missing")
            return []
        client = Anthropic(api_key=api_key)

    params = deterministic_params(model)
    try:
        response = client.messages.create(
            model=model,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _build_prompt(flow_name, flow_description, candidates),
            }],
            max_tokens=DEFAULT_MAX_TOKENS,
            **params,
        )
    except Exception as exc:  # noqa: BLE001 — opportunistic
        logger.warning("flow_symbols: API call failed (%s)", exc)
        return []

    if tracker is not None:
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                tracker.record(
                    model=model,
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                )
        except Exception:  # noqa: BLE001 — opportunistic
            pass

    text = ""
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")
    raw = _parse_response(text) or []
    picked = [
        coerced
        for entry in raw
        if (coerced := _coerce_picked(entry)) is not None
    ]
    picked = [p for p in picked if p.confidence >= confidence_floor]
    picked = _filter_by_candidates(picked, candidates)
    return picked


# ── Pipeline-level driver ─────────────────────────────────────────────


SourceLoader = Callable[[str], str | None]
"""``(rel_path) -> source_text | None`` — caller supplies; ``None``
when the file cannot be read (deleted, binary, unreadable)."""

CandidateBuilder = Callable[
    [str, str, list[str]],
    list[CandidateSymbol],
]
"""``(flow_name, feature_owner, candidate_files) -> CandidateSymbols``.
Default builder reads each file, runs ``list_exported_symbols``."""


def default_candidate_builder(
    source_loader: SourceLoader,
) -> CandidateBuilder:
    """Standard builder using ast_extractor.list_exported_symbols."""
    from faultline.analyzer.ast_extractor import list_exported_symbols

    def _build(
        flow_name: str,
        feature_owner: str,
        candidate_files: list[str],
    ) -> list[CandidateSymbol]:
        out: list[CandidateSymbol] = []
        for rel in candidate_files[:MAX_CANDIDATE_FILES]:
            source = source_loader(rel)
            if not source:
                continue
            for sr in list_exported_symbols(rel, source):
                out.append(CandidateSymbol(
                    file=rel,
                    name=sr.name,
                    kind=sr.kind,
                    start_line=sr.start_line,
                    end_line=sr.end_line,
                ))
        return out

    return _build


def _candidate_files_for_flow(
    flow_name: str,
    feature_owner: str,
    result: "DeepScanResult",
) -> list[str]:
    """Pick the most plausible files for a flow.

    Priority:
        1. Existing flow_participants entries (Sprint 7 trace_flows).
        2. Owner feature's paths whose name contains a flow-token.
        3. First N owner paths (fallback).
    """
    parts = result.flow_participants.get(feature_owner, {}).get(flow_name, [])
    if parts:
        seen: list[str] = []
        for p in parts:
            f = getattr(p, "path", None) or (
                p.get("path") if isinstance(p, dict) else None
            )
            if f and f not in seen:
                seen.append(f)
        if seen:
            return seen[:MAX_CANDIDATE_FILES]

    owner_paths = result.features.get(feature_owner, [])
    flow_tokens = re.split(r"[\s\-_/]+", flow_name.lower())
    flow_tokens = [t for t in flow_tokens if t and t != "flow"]
    if flow_tokens:
        scored: list[tuple[int, str]] = []
        for p in owner_paths:
            pl = p.lower()
            score = sum(1 for t in flow_tokens if t in pl)
            if score:
                scored.append((score, p))
        if scored:
            scored.sort(key=lambda x: (-x[0], x[1]))
            picked = [p for _, p in scored[:MAX_CANDIDATE_FILES]]
            if picked:
                return picked
    return owner_paths[:MAX_CANDIDATE_FILES]


def resolve_flow_symbols(
    result: "DeepScanResult",
    *,
    source_loader: SourceLoader,
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    tracker: CostTracker | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    repo_slug: str | None = None,
    candidate_builder: CandidateBuilder | None = None,
) -> int:
    """Walk every flow, ask Haiku which symbols participate, write
    results into ``result.flow_participants``.

    Returns the count of flows whose symbol list was populated /
    refreshed (counts cache hits too).

    Cache lives at ``cache_dir/flow-symbols-<repo_slug>.json``.
    Per-flow entries record the candidate-set hash so a flow whose
    candidate files change re-asks Haiku, but a stable flow replays
    instantly.
    """
    if repo_slug is None:
        repo_slug = _slugify_repo(getattr(result, "repo_path", "") or "")

    if candidate_builder is None:
        candidate_builder = default_candidate_builder(source_loader)

    cache = _load_cache(cache_dir, repo_slug)
    new_cache: dict[str, dict[str, Any]] = dict(cache)

    populated = 0
    for owner, flow_names in list(result.flows.items()):
        for flow_name in flow_names:
            description = (
                result.flow_descriptions.get(owner, {}).get(flow_name, "")
                if isinstance(result.flow_descriptions.get(owner), dict)
                else ""
            )
            candidate_files = _candidate_files_for_flow(
                flow_name, owner, result,
            )
            if not candidate_files:
                continue
            candidates = candidate_builder(
                flow_name, owner, candidate_files,
            )
            if not candidates:
                continue
            cand_hash = _candidates_hash(candidates)
            cache_key = f"{owner}::{flow_name}"

            cached_entry = cache.get(cache_key)
            if (
                cached_entry is not None
                and cached_entry.get("candidates_hash") == cand_hash
            ):
                picked_raw = cached_entry.get("picked") or []
                picked = [
                    PickedSymbol(**p) for p in picked_raw
                    if isinstance(p, dict) and "file" in p and "name" in p
                ]
            else:
                picked = pick_flow_symbols(
                    flow_name=flow_name,
                    flow_description=description,
                    candidates=candidates,
                    client=client,
                    api_key=api_key,
                    model=model,
                    tracker=tracker,
                )
                new_cache[cache_key] = {
                    "candidates_hash": cand_hash,
                    "picked": [asdict(p) for p in picked],
                }

            if not picked:
                continue
            ranges_by_file = resolve_picked(picked, candidates)
            _attach_to_participants(
                result, owner, flow_name, ranges_by_file,
            )
            populated += 1

    _save_cache(cache_dir, repo_slug, new_cache)
    return populated


def _attach_to_participants(
    result: "DeepScanResult",
    owner: str,
    flow_name: str,
    ranges_by_file: dict[str, list["SymbolRange"]],
) -> None:
    """Merge picked ranges into ``result.flow_participants``.

    The participant entry is a dict (so it stays serialisable across
    the pipeline); CLI's _inject_new_pipeline_flows turns it into a
    FlowParticipant Pydantic later. When a participant for the same
    file already exists, we replace its ``symbols`` with the
    Haiku-picked subset (Layer B is more precise than Sprint 7's
    BFS-based symbol list).
    """
    feat_parts = result.flow_participants.setdefault(owner, {})
    existing = feat_parts.get(flow_name, [])
    by_path: dict[str, dict[str, Any]] = {}
    for p in existing:
        path = getattr(p, "path", None) or (
            p.get("path") if isinstance(p, dict) else None
        )
        if path:
            by_path[path] = (
                p if isinstance(p, dict)
                else {
                    "path": path,
                    "layer": getattr(p, "layer", "support"),
                    "depth": getattr(p, "depth", 0),
                    "side_effect_only": getattr(p, "side_effect_only", False),
                    "symbols": list(getattr(p, "symbols", []) or []),
                }
            )

    for file, ranges in ranges_by_file.items():
        entry = by_path.get(file) or {
            "path": file,
            "layer": "support",
            "depth": 0,
            "side_effect_only": False,
            "symbols": [],
        }
        entry["symbols"] = ranges
        by_path[file] = entry

    feat_parts[flow_name] = list(by_path.values())
