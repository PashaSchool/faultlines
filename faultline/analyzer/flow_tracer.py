"""Per-flow BFS walker (Sprint 7 Day 2).

Given a Sprint 4 flow's entry point (file + line), walk the symbol
import graph forward up to ``depth`` hops and return every file that
participates in the flow, with the specific symbols touched in each.

The walker is the core of the call-graph trace. Layer classification
(Day 3) and Pydantic wiring (Day 4) build on its output. No LLM
calls — pure graph traversal.

Entry-point semantics
- An entry is ``(file, line)``.
- We resolve the line to the smallest enclosing
  :class:`SymbolRange` via ``SymbolGraph.find_enclosing_symbol``.
  When no symbol contains the line, we still walk the file's
  outgoing edges (some entries are top-level module side effects).

BFS shape
- Forward only — from entry downward through callees. Walking
  callers up the tree pulls in unrelated parents.
- Visited set is keyed on (file) — a file reached twice via
  different edges merges its imported symbols into one
  participant entry.
- Depth caps the hop count from the entry. Default 3 — beyond
  that the trace becomes too noisy on monorepos.

Special edge symbols
- ``"*"``  (namespace import) — pulls in every export of the
  target file as touched.
- ``"@import"``  (side-effect import) — records the file as
  participating but with no specific symbols.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

from .symbol_graph import ImportEdge, SymbolGraph
from ..models.types import SymbolRange


logger = logging.getLogger(__name__)


DEFAULT_DEPTH = 3
DEFAULT_MAX_PARTICIPANTS = 200


# ── Output shapes ───────────────────────────────────────────────────


@dataclass
class TracedParticipant:
    """One file that participates in a flow's call-graph reach."""

    file: str
    symbols: list[SymbolRange] = field(default_factory=list)
    depth: int = 0  # BFS depth from entry (0 = entry file itself)
    # Whether the file was reached via a side-effect-only import
    # (``import './x'``) — usually polyfills or registration code,
    # included for completeness but visually distinct.
    side_effect_only: bool = False


@dataclass
class TracedFlow:
    """Full BFS result for one flow."""

    entry_file: str
    entry_symbol: SymbolRange | None
    participants: list[TracedParticipant] = field(default_factory=list)
    # Set of file paths visited; convenience for callers.
    visited_files: set[str] = field(default_factory=set)


# ── Walker ──────────────────────────────────────────────────────────


def trace_flow(
    graph: SymbolGraph,
    entry_file: str,
    entry_line: int = 0,
    *,
    depth: int = DEFAULT_DEPTH,
    max_participants: int = DEFAULT_MAX_PARTICIPANTS,
) -> TracedFlow:
    """Walk forward from an entry point and return the reachable set.

    Args:
        graph: Pre-built :class:`SymbolGraph` for the repo.
        entry_file: Repo-relative path of the file where the flow
            starts. Must exist in the graph (have at least an
            exports or symbol_ranges entry); otherwise the result
            has only the entry participant with no symbols.
        entry_line: Optional 1-based line number. When > 0, the
            walker resolves the enclosing symbol; when 0 it walks
            the whole file's exports.
        depth: BFS hop cap. Default 3.
        max_participants: Hard cap on participants — prevents
            runaway traversal on highly-connected utility files.

    Returns:
        :class:`TracedFlow`. Always returns a value (never raises).
        Empty trace = entry file unknown to the graph; participants
        list will hold one entry-file participant with no symbols.
    """
    entry_symbol = (
        graph.find_enclosing_symbol(entry_file, entry_line)
        if entry_line > 0
        else None
    )

    out = TracedFlow(entry_file=entry_file, entry_symbol=entry_symbol)
    by_file: dict[str, TracedParticipant] = {}

    def _add(file: str, symbols: Iterable[SymbolRange],
             d: int, side_effect: bool) -> TracedParticipant:
        existing = by_file.get(file)
        if existing is None:
            existing = TracedParticipant(
                file=file, depth=d, side_effect_only=side_effect,
            )
            by_file[file] = existing
            out.visited_files.add(file)
        # Merge symbols (dedup on name+line range).
        seen = {(s.name, s.start_line, s.end_line) for s in existing.symbols}
        for s in symbols:
            key = (s.name, s.start_line, s.end_line)
            if key not in seen:
                seen.add(key)
                existing.symbols.append(s)
        # Side-effect-only flag flips off as soon as we get a real
        # symbol from this file — any other edge upgrades it.
        if not side_effect:
            existing.side_effect_only = False
        # Keep the smallest depth seen (a file might be reached via
        # multiple paths).
        if d < existing.depth:
            existing.depth = d
        return existing

    # Seed: entry file with the enclosing symbol (if any).
    seed_symbols = [entry_symbol] if entry_symbol else []
    _add(entry_file, seed_symbols, 0, side_effect=False)

    # BFS queue of (file, depth_from_entry).
    queue: deque[tuple[str, int]] = deque([(entry_file, 0)])

    capped = False
    while queue and not capped:
        current, d = queue.popleft()
        if d >= depth:
            continue
        for edge in graph.imports_from(current):
            if len(by_file) >= max_participants:
                logger.info(
                    "flow_tracer: hit max_participants=%d on %s — stopping BFS",
                    max_participants, entry_file,
                )
                capped = True
                break
            target = edge.target_file
            if target == current:
                # Self-import — would loop forever.
                continue
            target_symbols = _resolve_target_symbols(graph, edge)
            side_effect = (edge.target_symbol == "@import"
                           and not target_symbols)
            already_visited = target in out.visited_files
            _add(target, target_symbols, d + 1, side_effect=side_effect)
            # Enqueue only on first visit so we don't re-explore
            # the same file's outgoing edges.
            if not already_visited:
                queue.append((target, d + 1))

    out.participants = sorted(
        by_file.values(),
        key=lambda p: (p.depth, p.file),
    )
    logger.info(
        "flow_tracer: %s → %d participants (depth ≤ %d, %d total files)",
        entry_file, len(out.participants), depth, len(out.visited_files),
    )
    return out


def _resolve_target_symbols(
    graph: SymbolGraph,
    edge: ImportEdge,
) -> list[SymbolRange]:
    """Map an :class:`ImportEdge` to the SymbolRange entries in the
    target file that it actually touches.

    - ``edge.target_symbol == "*"`` (namespace) → every export of
      the target file.
    - ``edge.target_symbol == "@import"`` (side-effect) → empty list;
      caller flags the participant as side-effect-only.
    - Specific name → the matching SymbolRange (or empty if the
      target's exports don't list it).
    """
    if edge.target_symbol == "@import":
        return []
    if edge.target_symbol == "*":
        return list(graph.exports.get(edge.target_file, []))
    matches = [
        r for r in graph.exports.get(edge.target_file, [])
        if r.name == edge.target_symbol
    ]
    return matches
