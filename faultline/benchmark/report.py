"""Markdown report renderer for benchmark scoring."""

from __future__ import annotations

from dataclasses import dataclass

from .loader import BenchmarkSpec
from .metrics import (
    attribution_accuracy,
    feature_precision,
    feature_recall,
    flow_recall,
    generic_name_rate,
)


@dataclass
class BenchmarkScore:
    repo: str
    feature_recall: float
    feature_precision: float
    flow_recall: float | None  # None for libraries (skipped)
    attribution_accuracy: float | None  # None when no samples provided
    generic_name_rate: float
    feature_count_expected: int
    feature_count_detected: int


def score(
    spec: BenchmarkSpec,
    detected_features: dict[str, list[str]],
    detected_flows: dict[str, list[str]] | None = None,
    *,
    is_library: bool = False,
) -> BenchmarkScore:
    """Compute all 5 metrics for one repo's scan output."""
    fr = feature_recall(spec.features, detected_features)
    fp = feature_precision(spec.features, detected_features)
    if is_library or not spec.flows:
        flow_r: float | None = None
    else:
        flow_r = flow_recall(spec.flows, detected_flows or {})
    if spec.attribution:
        attr = attribution_accuracy(
            spec.attribution, detected_features,
            expected_features=spec.features,
        )
    else:
        attr = None
    gnr = generic_name_rate(detected_features)

    return BenchmarkScore(
        repo=spec.repo,
        feature_recall=fr,
        feature_precision=fp,
        flow_recall=flow_r,
        attribution_accuracy=attr,
        generic_name_rate=gnr,
        feature_count_expected=len(spec.features),
        feature_count_detected=len(detected_features),
    )


def _pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def render_markdown(scores: list[BenchmarkScore]) -> str:
    """Render a single Markdown page for one or more repos."""
    if not scores:
        return "# Benchmark report\n\n_No scores._\n"

    lines: list[str] = ["# Benchmark report", ""]
    lines.append(
        "| repo | feature recall | feature precision | flow recall | "
        "attribution | generic-name rate | features detected/expected |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in scores:
        lines.append(
            f"| {s.repo} | {_pct(s.feature_recall)} | "
            f"{_pct(s.feature_precision)} | {_pct(s.flow_recall)} | "
            f"{_pct(s.attribution_accuracy)} | "
            f"{_pct(s.generic_name_rate)} | "
            f"{s.feature_count_detected}/{s.feature_count_expected} |"
        )
    lines.append("")

    if len(scores) > 1:
        # Weighted average across repos that have a value for each
        # metric (skipping None entries).
        def avg(getter):
            vals = [getter(s) for s in scores if getter(s) is not None]
            return (sum(vals) / len(vals)) if vals else None

        lines.append("## Overall scorecard")
        lines.append("")
        lines.append(f"- Avg feature recall: **{_pct(avg(lambda s: s.feature_recall))}**")
        lines.append(f"- Avg feature precision: **{_pct(avg(lambda s: s.feature_precision))}**")
        lines.append(f"- Avg flow recall: **{_pct(avg(lambda s: s.flow_recall))}**")
        lines.append(f"- Avg attribution accuracy: **{_pct(avg(lambda s: s.attribution_accuracy))}**")
        lines.append(f"- Avg generic-name rate: **{_pct(avg(lambda s: s.generic_name_rate))}** (lower is better)")
        lines.append("")

    return "\n".join(lines) + "\n"
