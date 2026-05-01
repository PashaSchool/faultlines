"""Tests for line-scoped coverage application (Sprint 2 Day 9)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from faultline.analyzer.coverage import FileCoverage
from faultline.cli import _apply_feature_coverage, _coverage_for_attributions
from faultline.models.types import (
    Feature,
    FeatureMap,
    Flow,
    SymbolAttribution,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _attr(
    file_path: str,
    symbols: list[str],
    line_ranges: list[tuple[int, int]],
) -> SymbolAttribution:
    total = sum(end - start + 1 for start, end in line_ranges)
    return SymbolAttribution(
        file_path=file_path,
        symbols=symbols,
        line_ranges=line_ranges,
        attributed_lines=total,
        total_file_lines=100,
    )


def _feature(
    name: str,
    paths: list[str],
    *,
    flows: list[Flow] | None = None,
    attributions: list[SymbolAttribution] | None = None,
) -> Feature:
    return Feature(
        name=name,
        paths=paths,
        authors=[],
        total_commits=10,
        bug_fixes=2,
        bug_fix_ratio=0.2,
        last_modified=_now(),
        health_score=80.0,
        flows=flows or [],
        shared_attributions=attributions or [],
    )


def _flow(
    name: str,
    paths: list[str],
    *,
    attributions: list[SymbolAttribution] | None = None,
) -> Flow:
    return Flow(
        name=name,
        paths=paths,
        authors=[],
        total_commits=5,
        bug_fixes=1,
        bug_fix_ratio=0.2,
        last_modified=_now(),
        health_score=80.0,
        symbol_attributions=attributions or [],
    )


def _fm(features: list[Feature]) -> FeatureMap:
    return FeatureMap(
        repo_path=".",
        analyzed_at=_now(),
        total_commits=10,
        date_range_days=30,
        features=features,
    )


class TestCoverageForAttributions:
    def test_returns_none_without_detailed(self):
        attr = _attr("a.ts", ["x"], [(1, 10)])
        assert _coverage_for_attributions([attr], None, "") is None

    def test_returns_none_without_attributions(self):
        detailed = {"a.ts": FileCoverage(pct=80.0, line_hits={1: 1})}
        assert _coverage_for_attributions(None, detailed, "") is None
        assert _coverage_for_attributions([], detailed, "") is None

    def test_averages_per_range(self):
        # formatDate at 1-10 is 100% covered
        # parseDate at 11-20 is 50% covered
        # Average: 75%
        line_hits = {**{i: 1 for i in range(1, 11)},
                     **{i: 1 for i in range(11, 16)},
                     **{i: 0 for i in range(16, 21)}}
        detailed = {"utils.ts": FileCoverage(pct=75.0, line_hits=line_hits)}
        attrs = [
            _attr("utils.ts", ["formatDate"], [(1, 10)]),
            _attr("utils.ts", ["parseDate"], [(11, 20)]),
        ]
        result = _coverage_for_attributions(attrs, detailed, "")
        assert result == 75.0

    def test_skips_files_without_detailed_coverage(self):
        # One attribution has data, one doesn't
        detailed = {"a.ts": FileCoverage(pct=100.0, line_hits={i: 1 for i in range(1, 11)})}
        attrs = [
            _attr("a.ts", ["x"], [(1, 10)]),  # 100% covered
            _attr("b.ts", ["y"], [(1, 10)]),  # not in detailed
        ]
        result = _coverage_for_attributions(attrs, detailed, "")
        assert result == 100.0  # b.ts skipped, only a.ts counts

    def test_path_prefix_resolution(self):
        # Coverage data uses absolute paths; attributions use relative
        detailed = {"prefix/utils.ts": FileCoverage(pct=80.0, line_hits={1: 1})}
        attrs = [_attr("utils.ts", ["x"], [(1, 1)])]
        result = _coverage_for_attributions(attrs, detailed, "prefix/")
        assert result is not None and result == 100.0  # the single line is covered


class TestApplyFeatureCoverageTier1:
    def test_uses_line_scoped_when_attributions_present(self):
        # File-level: 50% (only half the file covered)
        # Symbol-scoped: only the FIRST function which is 100% covered
        line_hits = {**{i: 1 for i in range(1, 11)},   # function A: covered
                     **{i: 0 for i in range(11, 21)}}  # function B: missed
        detailed = {"utils.ts": FileCoverage(pct=50.0, line_hits=line_hits)}
        feature = _feature(
            "utils-feat",
            ["utils.ts"],
            attributions=[_attr("utils.ts", ["formatDate"], [(1, 10)])],
        )
        fm = _fm([feature])
        _apply_feature_coverage(
            fm, coverage_data={"utils.ts": 50.0}, path_prefix="",
            detailed=detailed,
        )
        # Tier 1 should win — 100% (the symbol's specific lines)
        assert feature.coverage_pct == 100.0

    def test_falls_back_to_file_level_when_no_attributions(self):
        line_hits = {1: 1, 2: 0}
        detailed = {"utils.ts": FileCoverage(pct=50.0, line_hits=line_hits)}
        feature = _feature("utils-feat", ["utils.ts"])  # no attributions
        fm = _fm([feature])
        _apply_feature_coverage(
            fm, coverage_data={"utils.ts": 50.0}, path_prefix="",
            detailed=detailed,
        )
        assert feature.coverage_pct == 50.0  # file-level

    def test_falls_back_when_detailed_missing(self):
        feature = _feature(
            "utils-feat",
            ["utils.ts"],
            attributions=[_attr("utils.ts", ["x"], [(1, 10)])],
        )
        fm = _fm([feature])
        _apply_feature_coverage(
            fm, coverage_data={"utils.ts": 50.0}, path_prefix="",
            detailed=None,
        )
        # No detailed → file-level
        assert feature.coverage_pct == 50.0


class TestFlowCoverage:
    def test_flow_uses_symbol_attributions(self):
        line_hits = {**{i: 1 for i in range(1, 11)},
                     **{i: 0 for i in range(11, 21)}}
        detailed = {"utils.ts": FileCoverage(pct=50.0, line_hits=line_hits)}
        flow = _flow(
            "format-flow",
            ["utils.ts"],
            attributions=[_attr("utils.ts", ["formatDate"], [(1, 10)])],
        )
        feature = _feature("utils", ["utils.ts"], flows=[flow])
        fm = _fm([feature])
        _apply_feature_coverage(
            fm, coverage_data={}, path_prefix="", detailed=detailed,
        )
        # Flow should get 100% (only its symbol's lines)
        assert flow.coverage_pct == 100.0

    def test_two_flows_diverge(self):
        # Same file, two flows pulling different symbols → different coverage
        line_hits = {**{i: 1 for i in range(1, 11)},
                     **{i: 0 for i in range(11, 21)}}
        detailed = {"utils.ts": FileCoverage(pct=50.0, line_hits=line_hits)}
        flow_a = _flow(
            "format-flow",
            ["utils.ts"],
            attributions=[_attr("utils.ts", ["formatDate"], [(1, 10)])],
        )
        flow_b = _flow(
            "parse-flow",
            ["utils.ts"],
            attributions=[_attr("utils.ts", ["parseDate"], [(11, 20)])],
        )
        feature = _feature("utils", ["utils.ts"], flows=[flow_a, flow_b])
        fm = _fm([feature])
        _apply_feature_coverage(
            fm, coverage_data={}, path_prefix="", detailed=detailed,
        )
        assert flow_a.coverage_pct == 100.0  # formatDate fully covered
        assert flow_b.coverage_pct == 0.0    # parseDate uncovered
