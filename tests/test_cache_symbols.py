"""Tests for faultline/cache/symbols.py — incremental symbol refresh."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from faultline.cache.symbols import (
    SymbolDelta,
    _clean_attribution_list,
    _cleanup_stale_attributions,
    _compute_delta,
    refresh_symbol_attributions,
)
from faultline.models.types import (
    Feature,
    FeatureMap,
    Flow,
    SymbolAttribution,
)


def _flow(name: str, attrs: list[SymbolAttribution]) -> Flow:
    return Flow(
        name=name,
        paths=[a.file_path for a in attrs],
        authors=[],
        total_commits=1,
        bug_fixes=0,
        bug_fix_ratio=0,
        last_modified=datetime.now(tz=timezone.utc),
        health_score=100,
        symbol_attributions=attrs,
    )


def _map_with_symbols() -> FeatureMap:
    flow = _flow("checkout", [
        SymbolAttribution(
            file_path="src/stripe.ts",
            symbols=["charge", "refund"],
            line_ranges=[],
            attributed_lines=0,
            total_file_lines=0,
        ),
    ])
    feature = Feature(
        name="payments",
        paths=["src/stripe.ts"],
        authors=[],
        total_commits=5,
        bug_fixes=1,
        bug_fix_ratio=0.2,
        last_modified=datetime.now(tz=timezone.utc),
        health_score=80,
        flows=[flow],
        shared_attributions=[
            SymbolAttribution(
                file_path="src/stripe.ts",
                symbols=["PaymentMethod", "Money"],
                line_ranges=[],
                attributed_lines=0,
                total_file_lines=0,
            ),
        ],
    )
    return FeatureMap(
        repo_path="/tmp/x",
        analyzed_at=datetime.now(tz=timezone.utc),
        total_commits=5,
        date_range_days=30,
        features=[feature],
        symbol_hashes={
            "src/stripe.ts": {
                "charge": "h_charge_v1",
                "refund": "h_refund_v1",
                "PaymentMethod": "h_pm_v1",
                "Money": "h_money_v1",
            },
        },
    )


class TestComputeDelta:
    def test_detects_added(self) -> None:
        delta = _compute_delta("a.ts", {"x": "h1"}, {"x": "h1", "y": "h2"})
        assert delta.added == ["y"]
        assert delta.removed == []
        assert delta.body_modified == []
        assert delta.unchanged == ["x"]

    def test_detects_removed(self) -> None:
        delta = _compute_delta("a.ts", {"x": "h1", "y": "h2"}, {"x": "h1"})
        assert delta.removed == ["y"]
        assert delta.added == []

    def test_detects_body_modified(self) -> None:
        delta = _compute_delta("a.ts", {"x": "h1"}, {"x": "h2"})
        assert delta.body_modified == ["x"]
        assert delta.added == []
        assert delta.removed == []

    def test_is_structurally_changed_flag(self) -> None:
        only_body = _compute_delta("a.ts", {"x": "h1"}, {"x": "h2"})
        assert only_body.is_structurally_changed is False

        added = _compute_delta("a.ts", {"x": "h1"}, {"x": "h1", "y": "h2"})
        assert added.is_structurally_changed is True

        removed = _compute_delta("a.ts", {"x": "h1", "y": "h2"}, {"x": "h1"})
        assert removed.is_structurally_changed is True


class TestCleanupStaleAttributions:
    def test_removes_symbols_from_flow(self) -> None:
        fm = _map_with_symbols()
        deltas = {
            "src/stripe.ts": SymbolDelta(
                file_path="src/stripe.ts",
                removed=["refund"],
            ),
        }
        count = _cleanup_stale_attributions(fm, deltas)
        assert count == 1
        flow = fm.features[0].flows[0]
        assert flow.symbol_attributions[0].symbols == ["charge"]

    def test_removes_entire_attribution_when_all_symbols_gone(self) -> None:
        fm = _map_with_symbols()
        deltas = {
            "src/stripe.ts": SymbolDelta(
                file_path="src/stripe.ts",
                removed=["charge", "refund"],
            ),
        }
        count = _cleanup_stale_attributions(fm, deltas)
        assert count == 2
        assert fm.features[0].flows[0].symbol_attributions == []

    def test_cleans_shared_attributions_too(self) -> None:
        fm = _map_with_symbols()
        deltas = {
            "src/stripe.ts": SymbolDelta(
                file_path="src/stripe.ts",
                removed=["PaymentMethod"],
            ),
        }
        count = _cleanup_stale_attributions(fm, deltas)
        assert count == 1
        assert fm.features[0].shared_attributions[0].symbols == ["Money"]

    def test_no_op_for_unrelated_changes(self) -> None:
        fm = _map_with_symbols()
        deltas = {
            "src/stripe.ts": SymbolDelta(
                file_path="src/stripe.ts",
                body_modified=["charge"],
            ),
        }
        count = _cleanup_stale_attributions(fm, deltas)
        assert count == 0
        # Nothing removed — body-only changes keep attribution intact
        assert fm.features[0].flows[0].symbol_attributions[0].symbols == ["charge", "refund"]


class TestCleanAttributionList:
    def test_preserves_order_and_other_files(self) -> None:
        attrs = [
            SymbolAttribution(
                file_path="a.ts", symbols=["x", "y"],
                line_ranges=[], attributed_lines=0, total_file_lines=0,
            ),
            SymbolAttribution(
                file_path="b.ts", symbols=["z"],
                line_ranges=[], attributed_lines=0, total_file_lines=0,
            ),
        ]
        count = _clean_attribution_list(attrs, {"a.ts": {"y"}})
        assert count == 1
        assert attrs[0].symbols == ["x"]
        assert attrs[1].symbols == ["z"]


class TestRefreshSymbolAttributions:
    def test_body_modified_preserves_attribution(self, tmp_path: Path) -> None:
        """Critical case: function body changed but name same → keep attribution."""
        fm = _map_with_symbols()
        # Stub extract_signatures so we don't need real files
        from faultline.analyzer.ast_extractor import FileSignature
        from faultline.models.types import SymbolRange

        sig = FileSignature(path="src/stripe.ts")
        sig.symbol_ranges = [
            SymbolRange(name="charge", start_line=1, end_line=10, kind="function"),
            SymbolRange(name="refund", start_line=12, end_line=20, kind="function"),
            SymbolRange(name="PaymentMethod", start_line=22, end_line=25, kind="type"),
            SymbolRange(name="Money", start_line=27, end_line=30, kind="type"),
        ]
        sig.source = "def charge():\n    pass  # modified implementation\n" * 20

        with patch("faultline.cache.symbols.extract_signatures") as mock_extract:
            mock_extract.return_value = {"src/stripe.ts": sig}
            with patch("faultline.cache.symbols._hash_current_symbols") as mock_hash:
                # Body hashes changed but symbol names the same
                mock_hash.return_value = {
                    "src/stripe.ts": {
                        "charge": "h_charge_v2",   # body changed
                        "refund": "h_refund_v1",   # unchanged
                        "PaymentMethod": "h_pm_v1",
                        "Money": "h_money_v1",
                    },
                }
                report = refresh_symbol_attributions(fm, str(tmp_path))

        # Attribution stays fully intact — only body changed
        assert report.symbols_body_modified == 1
        assert report.symbols_added == 0
        assert report.symbols_removed == 0
        assert report.attributions_removed == 0
        assert fm.features[0].flows[0].symbol_attributions[0].symbols == ["charge", "refund"]

    def test_removed_symbol_cleans_attribution(self, tmp_path: Path) -> None:
        fm = _map_with_symbols()
        from faultline.analyzer.ast_extractor import FileSignature
        from faultline.models.types import SymbolRange

        sig = FileSignature(path="src/stripe.ts")
        sig.symbol_ranges = [
            SymbolRange(name="charge", start_line=1, end_line=10, kind="function"),
        ]
        sig.source = "def charge():\n    pass\n"

        with patch("faultline.cache.symbols.extract_signatures") as mock_extract:
            mock_extract.return_value = {"src/stripe.ts": sig}
            with patch("faultline.cache.symbols._hash_current_symbols") as mock_hash:
                mock_hash.return_value = {
                    "src/stripe.ts": {"charge": "h_charge_v1"},
                }
                report = refresh_symbol_attributions(fm, str(tmp_path))

        assert report.symbols_removed == 3  # refund, PaymentMethod, Money
        assert report.attributions_removed == 3
        assert fm.features[0].flows[0].symbol_attributions[0].symbols == ["charge"]

    def test_added_symbol_without_api_key_preserves_existing(self, tmp_path: Path) -> None:
        fm = _map_with_symbols()
        from faultline.analyzer.ast_extractor import FileSignature
        from faultline.models.types import SymbolRange

        sig = FileSignature(path="src/stripe.ts")
        sig.symbol_ranges = [
            SymbolRange(name="charge", start_line=1, end_line=10, kind="function"),
            SymbolRange(name="refund", start_line=12, end_line=20, kind="function"),
            SymbolRange(name="PaymentMethod", start_line=22, end_line=25, kind="type"),
            SymbolRange(name="Money", start_line=27, end_line=30, kind="type"),
            SymbolRange(name="dispute", start_line=32, end_line=40, kind="function"),
        ]
        sig.source = "x" * 1000

        with patch("faultline.cache.symbols.extract_signatures") as mock_extract:
            mock_extract.return_value = {"src/stripe.ts": sig}
            with patch("faultline.cache.symbols._hash_current_symbols") as mock_hash:
                mock_hash.return_value = {
                    "src/stripe.ts": {
                        "charge": "h_charge_v1",
                        "refund": "h_refund_v1",
                        "PaymentMethod": "h_pm_v1",
                        "Money": "h_money_v1",
                        "dispute": "h_dispute_v1",  # new
                    },
                }
                # No api_key → re-attribution skipped
                report = refresh_symbol_attributions(fm, str(tmp_path), api_key=None)

        assert report.symbols_added == 1
        assert report.features_reattributed == 0
        # Existing attribution preserved
        assert set(fm.features[0].flows[0].symbol_attributions[0].symbols) == {"charge", "refund"}

    def test_updates_symbol_hashes_on_map(self, tmp_path: Path) -> None:
        fm = _map_with_symbols()
        from faultline.analyzer.ast_extractor import FileSignature
        from faultline.models.types import SymbolRange

        sig = FileSignature(path="src/stripe.ts")
        sig.symbol_ranges = [
            SymbolRange(name="charge", start_line=1, end_line=10, kind="function"),
        ]
        sig.source = "def charge(): pass\n"

        with patch("faultline.cache.symbols.extract_signatures") as mock_extract:
            mock_extract.return_value = {"src/stripe.ts": sig}
            with patch("faultline.cache.symbols._hash_current_symbols") as mock_hash:
                mock_hash.return_value = {"src/stripe.ts": {"charge": "NEW_HASH"}}
                refresh_symbol_attributions(fm, str(tmp_path))

        assert fm.symbol_hashes["src/stripe.ts"] == {"charge": "NEW_HASH"}
