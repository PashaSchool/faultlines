"""Sprint 12 Day 3.5 — deterministic signals for flow re-attribution.

The Sprint 11 flow_judge sees only flow-name + feature-menu and decides
purely from semantics. That misses cases where the flow's actual file
distribution screams "this is feature X" — e.g. 80 % of the flow's
participant files live under ``Feature.paths`` of feature X but the
flow's name has a misleading prefix.

This module computes two cheap deterministic scores per (flow,
feature) pair so the judge can take them as evidence:

  file_ownership_score(flow_paths, feature_paths) -> float in [0,1]
      Fraction of flow's participant paths owned by this feature.

  call_graph_centrality(flow, feature_paths, signatures) -> float
      Approximate fan-in: how many of the flow's entry-point's
      symbols are imported back into this feature's paths.

These are **inputs** to flow_judge — not gatekeepers. The Haiku still
makes the final call; the signals just feed the prompt.
"""

from __future__ import annotations

from typing import Any


def file_ownership_score(
    flow_paths: list[str],
    feature_paths: list[str],
) -> float:
    """Fraction of ``flow_paths`` that appear in ``feature_paths``.

    Returns 0.0 when the flow has no paths (legacy detector — judge
    falls back to pure semantics in that case).
    """
    if not flow_paths:
        return 0.0
    feature_set = set(feature_paths)
    matched = sum(1 for p in flow_paths if p in feature_set)
    return matched / len(flow_paths)


def file_ownership_distribution(
    flow_paths: list[str],
    features: dict[str, list[str]],
) -> dict[str, float]:
    """Per-feature ownership scores. Used as evidence in flow_judge."""
    return {
        name: file_ownership_score(flow_paths, paths)
        for name, paths in features.items()
    }


def call_graph_centrality(
    entry_point_file: str | None,
    feature_paths: list[str],
    import_graph: dict[str, set[str]] | None = None,
) -> float:
    """How many feature paths import the flow's entry-point file.

    ``import_graph`` is ``{importer_path: set(imported_paths)}``. When
    None or missing the entry point, the centrality is 0. Cheap fan-in
    proxy — the more paths in this feature import the flow's entry
    point, the more likely the flow's centre of gravity is here.

    Returns a normalised value in [0, 1]: ``importers / |feature|``.
    """
    if entry_point_file is None or not feature_paths:
        return 0.0
    if not import_graph:
        return 0.0
    feature_set = set(feature_paths)
    importers = sum(
        1 for p in feature_set
        if p in import_graph and entry_point_file in import_graph[p]
    )
    return importers / len(feature_set)


def call_graph_centrality_distribution(
    entry_point_file: str | None,
    features: dict[str, list[str]],
    import_graph: dict[str, set[str]] | None = None,
) -> dict[str, float]:
    return {
        name: call_graph_centrality(entry_point_file, paths, import_graph)
        for name, paths in features.items()
    }


def format_signals_for_prompt(
    ownership: dict[str, float],
    centrality: dict[str, float] | None = None,
    top_n: int = 5,
) -> str:
    """Compact human-readable summary for prompt injection.

    Output only the top ``top_n`` features by ownership — the rest are
    noise. Falls back to "no signals" when both maps are empty.
    """
    if not ownership and not centrality:
        return "(no deterministic signals available)"

    centrality = centrality or {}
    feats = sorted(
        ownership.keys(),
        key=lambda f: ownership.get(f, 0.0),
        reverse=True,
    )[:top_n]
    if not feats:
        return "(no deterministic signals available)"

    lines = []
    for f in feats:
        ow = ownership.get(f, 0.0) * 100
        cn = centrality.get(f, 0.0) * 100
        if ow == 0.0 and cn == 0.0:
            continue
        if cn > 0:
            lines.append(f"  - {f}: {ow:.0f}% files owned, {cn:.0f}% fan-in")
        else:
            lines.append(f"  - {f}: {ow:.0f}% files owned")
    return "\n".join(lines) if lines else "(no deterministic signals available)"


def aggregate_signals(
    flow_paths: list[str],
    entry_point_file: str | None,
    features: dict[str, list[str]],
    import_graph: dict[str, set[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute ownership+centrality for a flow against every feature.

    Returns ``{feature_name: {"ownership": x, "centrality": y}}``.
    """
    ownership = file_ownership_distribution(flow_paths, features)
    centrality = call_graph_centrality_distribution(
        entry_point_file, features, import_graph,
    )
    return {
        name: {
            "ownership": ownership.get(name, 0.0),
            "centrality": centrality.get(name, 0.0),
        }
        for name in features
    }
