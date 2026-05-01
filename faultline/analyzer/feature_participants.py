"""Per-feature participants — BFS from a feature's whole file set.

Mirrors :mod:`faultline.analyzer.flow_tracer` but seeded from every
file the feature owns, not from a single flow entry-point. The output
fills :pyattr:`Feature.participants` so symbol-scoped scoring
(``_compute_line_scoped_health``, ``_apply_feature_coverage``) reads
from the same primary surface as flows do.

Why not reuse `shared_attributions`?
  ``shared_attributions`` only emits entries for files that live in
  2+ features. Workspace monorepos and Django apps put every file in
  exactly one feature, so the cross-feature shape is empty by design
  even when there is plenty of intra-feature import structure. This
  module attaches participants per feature regardless of cross-
  feature sharing.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable

from .layer_classifier import classify_files
from .symbol_graph import ImportEdge, SymbolGraph
from ..models.types import FlowParticipant, SymbolRange

logger = logging.getLogger(__name__)

DEFAULT_DEPTH = 3
DEFAULT_MAX_PARTICIPANTS_PER_FEATURE = 500


def build_feature_participants(
    feature_paths: dict[str, list[str]],
    graph: SymbolGraph,
    *,
    repo_root: str | None = None,
    depth: int = DEFAULT_DEPTH,
    max_per_feature: int = DEFAULT_MAX_PARTICIPANTS_PER_FEATURE,
) -> dict[str, list[FlowParticipant]]:
    """For each feature, BFS forward through the symbol graph.

    Seeds the queue with every file in the feature at depth 0 and
    walks ``graph.imports_from(file)`` up to ``depth`` hops. The
    resulting participants are de-duplicated by file path and
    annotated with the layer classifier output.

    Args:
        feature_paths: ``{feature_name: [file_path, ...]}`` — the
            same shape pipeline.run produces.
        graph: Pre-built :class:`SymbolGraph` for the repo.
        depth: BFS hop cap. Default 3 (matches flow_tracer).
        max_per_feature: Hard cap so a highly-connected utility
            file doesn't pull in half the repo.

    Returns:
        ``{feature_name: [FlowParticipant, ...]}``. Each participant
        carries ``path``, ``layer``, ``depth``, ``side_effect_only``,
        and ``symbols`` (with line ranges). Cross-language: works
        wherever the graph has resolved imports — TS / JS / Python /
        Go / Rust per Sprint 1 Day 4-5.
    """
    if not feature_paths:
        return {}

    out: dict[str, list[FlowParticipant]] = {}

    for feature_name, paths in feature_paths.items():
        by_file: dict[str, FlowParticipant] = {}
        seed_set = set(paths)
        queue: deque[tuple[str, int]] = deque()

        # Seed every owned file at depth 0 with its full export list.
        for owned_file in paths:
            seed_symbols = list(graph.exports.get(owned_file, []))
            _add(by_file, owned_file, seed_symbols, 0, side_effect=False)
            queue.append((owned_file, 0))

        capped = False
        while queue and not capped:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for edge in graph.imports_from(current):
                if len(by_file) >= max_per_feature:
                    logger.info(
                        "feature_participants: feature %s hit max=%d — stopping BFS",
                        feature_name, max_per_feature,
                    )
                    capped = True
                    break
                target = edge.target_file
                if target == current:
                    continue
                target_symbols = _resolve_target_symbols(graph, edge)
                side_effect = (
                    edge.target_symbol == "@import"
                    and not target_symbols
                )
                already_visited = target in by_file
                _add(by_file, target, target_symbols, d + 1, side_effect)
                if not already_visited:
                    queue.append((target, d + 1))

        # Layer classification — same step flow_tracer does post-BFS.
        # Skip when repo_root isn't available (test fixtures); leave
        # default 'support' on each participant.
        if repo_root:
            layers = classify_files(repo_root, list(by_file.keys()), read_content=False)
            for f, p in by_file.items():
                p.layer = str(layers.get(f, "support") or "support")

        out[feature_name] = sorted(
            by_file.values(),
            key=lambda p: (p.depth, p.path),
        )
        logger.debug(
            "feature_participants: %s → %d participants (seeded from %d files)",
            feature_name, len(out[feature_name]), len(seed_set),
        )

    total = sum(len(v) for v in out.values())
    logger.info(
        "feature_participants: %d participants across %d features (avg %.0f/feature)",
        total, len(out), total / max(len(out), 1),
    )
    return out


# ── Internals ────────────────────────────────────────────────────


def _add(
    by_file: dict[str, FlowParticipant],
    file: str,
    symbols: Iterable[SymbolRange],
    depth: int,
    side_effect: bool,
) -> FlowParticipant:
    existing = by_file.get(file)
    if existing is None:
        existing = FlowParticipant(
            path=file,
            layer="support",
            depth=depth,
            side_effect_only=side_effect,
            symbols=[],
        )
        by_file[file] = existing
    seen = {(s.name, s.start_line, s.end_line) for s in existing.symbols}
    for s in symbols:
        key = (s.name, s.start_line, s.end_line)
        if key not in seen:
            seen.add(key)
            existing.symbols.append(s)
    if not side_effect:
        existing.side_effect_only = False
    if depth < existing.depth:
        existing.depth = depth
    return existing


def _resolve_target_symbols(
    graph: SymbolGraph, edge: ImportEdge,
) -> list[SymbolRange]:
    """Resolve the imported name to the target file's actual ranges."""
    sym = edge.target_symbol
    target_ranges = graph.symbol_ranges.get(edge.target_file, [])
    if not target_ranges:
        return []
    if sym == "*":
        # Namespace import — return all exports.
        return list(graph.exports.get(edge.target_file, []))
    if sym == "@import":
        return []
    # Named import — find the matching range.
    matches = [r for r in target_ranges if r.name == sym]
    return matches
