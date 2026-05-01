"""Tests for apply_test_attribution + shared_with_flows (Sprint 2 Day 10)."""

from __future__ import annotations

from datetime import datetime, timezone

from faultline.analyzer.test_mapper import TestMap, apply_test_attribution
from faultline.models.types import (
    Feature,
    FeatureMap,
    Flow,
    SymbolAttribution,
)


def _now():
    return datetime.now(tz=timezone.utc)


def _attr(
    file_path: str,
    symbols: list[str],
    line_ranges: list[tuple[int, int]] | None = None,
) -> SymbolAttribution:
    if line_ranges is None:
        line_ranges = [(1, 10)]
    total = sum(end - start + 1 for start, end in line_ranges)
    return SymbolAttribution(
        file_path=file_path,
        symbols=symbols,
        line_ranges=line_ranges,
        attributed_lines=total,
        total_file_lines=100,
    )


def _flow(name: str, attrs: list[SymbolAttribution]) -> Flow:
    return Flow(
        name=name,
        paths=[a.file_path for a in attrs],
        authors=[],
        total_commits=1,
        bug_fixes=0,
        bug_fix_ratio=0,
        last_modified=_now(),
        health_score=100.0,
        symbol_attributions=attrs,
    )


def _feature(
    name: str,
    flows: list[Flow] | None = None,
    shared: list[SymbolAttribution] | None = None,
) -> Feature:
    return Feature(
        name=name,
        paths=[],
        authors=[],
        total_commits=1,
        bug_fixes=0,
        bug_fix_ratio=0,
        last_modified=_now(),
        health_score=100.0,
        flows=flows or [],
        shared_attributions=shared or [],
    )


def _fm(features: list[Feature]) -> FeatureMap:
    return FeatureMap(
        repo_path=".",
        analyzed_at=_now(),
        total_commits=1,
        date_range_days=1,
        features=features,
    )


class TestApplyTestAttribution:
    def test_populates_flow_test_files_via_symbol(self):
        flow = _flow("login-flow", [_attr("auth.ts", ["login"])])
        feature = _feature("auth", flows=[flow])
        fm = _fm([feature])
        tm = TestMap(
            by_symbol={("auth.ts", "login"): ["auth.test.ts"]},
        )
        apply_test_attribution(fm, tm)
        assert flow.test_files == ["auth.test.ts"]
        assert flow.test_file_count == 1

    def test_populates_via_file_fallback_when_symbol_absent(self):
        flow = _flow("login-flow", [_attr("auth.ts", ["login"])])
        feature = _feature("auth", flows=[flow])
        fm = _fm([feature])
        tm = TestMap(by_file={"auth.ts": ["auth.test.ts"]})
        apply_test_attribution(fm, tm)
        assert flow.test_files == ["auth.test.ts"]

    def test_empty_when_no_tests_match(self):
        flow = _flow("login-flow", [_attr("auth.ts", ["login"])])
        feature = _feature("auth", flows=[flow])
        fm = _fm([feature])
        tm = TestMap()
        apply_test_attribution(fm, tm)
        assert flow.test_files == []
        assert flow.test_file_count == 0

    def test_dedupes_test_files_across_symbols(self):
        # Two symbols share the same test file
        flow = _flow("auth-flow", [
            _attr("auth.ts", ["login", "logout"]),
        ])
        feature = _feature("auth", flows=[flow])
        fm = _fm([feature])
        tm = TestMap(
            by_symbol={
                ("auth.ts", "login"): ["auth.test.ts"],
                ("auth.ts", "logout"): ["auth.test.ts"],
            },
        )
        apply_test_attribution(fm, tm)
        assert flow.test_files == ["auth.test.ts"]


class TestSharedWithFlowsBadge:
    def test_marks_symbol_shared_across_flows(self):
        # Both flows pull formatDate from utils.ts
        flow_a = _flow("display-flow", [_attr("utils.ts", ["formatDate"])])
        flow_b = _flow("export-flow", [_attr("utils.ts", ["formatDate"])])
        feature = _feature("utils", flows=[flow_a, flow_b])
        fm = _fm([feature])
        apply_test_attribution(fm, TestMap())
        # Each flow's attribution sees the OTHER flow as shared
        attr_a = flow_a.symbol_attributions[0]
        attr_b = flow_b.symbol_attributions[0]
        assert attr_a.shared_with_flows == ["export-flow"]
        assert attr_b.shared_with_flows == ["display-flow"]

    def test_unique_symbol_has_empty_shared_list(self):
        flow_a = _flow("a", [_attr("utils.ts", ["formatDate"])])
        flow_b = _flow("b", [_attr("utils.ts", ["parseDate"])])
        feature = _feature("utils", flows=[flow_a, flow_b])
        fm = _fm([feature])
        apply_test_attribution(fm, TestMap())
        # Different symbols → no overlap
        assert flow_a.symbol_attributions[0].shared_with_flows == []
        assert flow_b.symbol_attributions[0].shared_with_flows == []

    def test_three_way_share(self):
        flow_a = _flow("a", [_attr("utils.ts", ["formatDate"])])
        flow_b = _flow("b", [_attr("utils.ts", ["formatDate"])])
        flow_c = _flow("c", [_attr("utils.ts", ["formatDate"])])
        feature = _feature("utils", flows=[flow_a, flow_b, flow_c])
        fm = _fm([feature])
        apply_test_attribution(fm, TestMap())
        for flow in (flow_a, flow_b, flow_c):
            shared = flow.symbol_attributions[0].shared_with_flows
            assert flow.name not in shared
            assert len(shared) == 2

    def test_feature_shared_attributions_get_badge_data(self):
        # The feature itself has a shared_attributions entry; it should
        # list every flow in the feature that uses the symbol.
        flow_a = _flow("display", [_attr("utils.ts", ["formatDate"])])
        flow_b = _flow("export", [_attr("utils.ts", ["formatDate"])])
        feature_attr = _attr("utils.ts", ["formatDate"])
        feature = _feature("utils", flows=[flow_a, flow_b], shared=[feature_attr])
        fm = _fm([feature])
        apply_test_attribution(fm, TestMap())
        assert set(feature_attr.shared_with_flows) == {"display", "export"}
