"""Tests for faultline/symbols/."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from faultline.analyzer.ast_extractor import FileSignature
from faultline.models.types import Feature, Flow, SymbolRange
from faultline.symbols.attribution import (
    _apply_mapping_to_flows,
    _parse_response,
    attribute_symbols_to_flows,
)
from faultline.symbols.extractor import (
    FileSymbols,
    extract_file_symbols,
)
from faultline.symbols.pipeline import enrich_with_symbols


def _sig(path: str, ranges: list[SymbolRange]) -> FileSignature:
    sig = FileSignature(path=path)
    sig.symbol_ranges = ranges
    sig.exports = [r.name for r in ranges]
    return sig


def _flow(name: str, paths: list[str]) -> Flow:
    return Flow(
        name=name,
        paths=paths,
        authors=[],
        total_commits=10,
        bug_fixes=2,
        bug_fix_ratio=0.2,
        last_modified=datetime.now(tz=timezone.utc),
        health_score=80.0,
    )


def _feature_with_flows() -> Feature:
    return Feature(
        name="payments",
        description="Stripe payment processing",
        paths=["src/stripe.ts", "src/types.ts"],
        authors=["alice"],
        total_commits=50,
        bug_fixes=10,
        bug_fix_ratio=0.2,
        last_modified=datetime.now(tz=timezone.utc),
        health_score=80.0,
        flows=[
            _flow("checkout-flow", ["src/stripe.ts"]),
            _flow("refund-flow", ["src/stripe.ts"]),
        ],
    )


class TestExtractFileSymbols:
    def test_splits_flow_eligible_and_feature_only(self) -> None:
        sigs = {
            "src/stripe.ts": _sig("src/stripe.ts", [
                SymbolRange(name="charge", start_line=10, end_line=45, kind="function"),
                SymbolRange(name="refund", start_line=50, end_line=80, kind="function"),
                SymbolRange(name="PaymentMethod", start_line=5, end_line=8, kind="type"),
                SymbolRange(name="StripeClient", start_line=90, end_line=120, kind="class"),
            ]),
        }
        result = extract_file_symbols(sigs)
        assert "src/stripe.ts" in result
        fs = result["src/stripe.ts"]
        assert set(fs.flow_symbols) == {"charge", "refund", "StripeClient"}
        assert fs.feature_symbols == ["PaymentMethod"]

    def test_python_fallback_all_to_flow(self) -> None:
        sig = FileSignature(path="app.py")
        sig.exports = ["handler", "validate"]
        sigs = {"app.py": sig}
        result = extract_file_symbols(sigs)
        assert result["app.py"].flow_symbols == ["handler", "validate"]
        assert result["app.py"].feature_symbols == []


class TestParseResponse:
    def test_parses_plain_json(self) -> None:
        text = '{"checkout-flow": [{"file": "a.ts", "symbols": ["charge"]}]}'
        result = _parse_response(text)
        assert result == {"checkout-flow": [{"file": "a.ts", "symbols": ["charge"]}]}

    def test_parses_fenced_json(self) -> None:
        text = '```json\n{"flow": [{"file": "a.ts", "symbols": ["x"]}]}\n```'
        result = _parse_response(text)
        assert result is not None
        assert "flow" in result

    def test_returns_none_for_invalid(self) -> None:
        assert _parse_response("not json at all") is None

    def test_returns_none_for_non_dict(self) -> None:
        assert _parse_response("[1, 2, 3]") is None


class TestApplyMappingToFlows:
    def test_attaches_valid_symbols(self) -> None:
        feature = _feature_with_flows()
        relevant = {"src/stripe.ts": ["charge", "refund", "StripeClient"]}
        mapping = {
            "checkout-flow": [{"file": "src/stripe.ts", "symbols": ["charge"]}],
            "refund-flow": [{"file": "src/stripe.ts", "symbols": ["refund"]}],
        }
        _apply_mapping_to_flows(feature, relevant, mapping)

        checkout = feature.flows[0]
        refund = feature.flows[1]
        assert len(checkout.symbol_attributions) == 1
        assert checkout.symbol_attributions[0].symbols == ["charge"]
        assert len(refund.symbol_attributions) == 1
        assert refund.symbol_attributions[0].symbols == ["refund"]

    def test_rejects_hallucinated_symbols(self) -> None:
        feature = _feature_with_flows()
        relevant = {"src/stripe.ts": ["charge"]}
        mapping = {
            "checkout-flow": [{"file": "src/stripe.ts", "symbols": ["fake_symbol"]}],
        }
        _apply_mapping_to_flows(feature, relevant, mapping)
        assert len(feature.flows[0].symbol_attributions) == 0

    def test_same_symbol_can_attribute_to_multiple_flows(self) -> None:
        feature = _feature_with_flows()
        relevant = {"src/stripe.ts": ["validatePayment"]}
        mapping = {
            "checkout-flow": [{"file": "src/stripe.ts", "symbols": ["validatePayment"]}],
            "refund-flow": [{"file": "src/stripe.ts", "symbols": ["validatePayment"]}],
        }
        _apply_mapping_to_flows(feature, relevant, mapping)
        assert feature.flows[0].symbol_attributions[0].symbols == ["validatePayment"]
        assert feature.flows[1].symbol_attributions[0].symbols == ["validatePayment"]

    def test_skips_unknown_flow(self) -> None:
        feature = _feature_with_flows()
        relevant = {"src/stripe.ts": ["charge"]}
        mapping = {
            "nonexistent-flow": [{"file": "src/stripe.ts", "symbols": ["charge"]}],
        }
        _apply_mapping_to_flows(feature, relevant, mapping)
        assert all(len(fl.symbol_attributions) == 0 for fl in feature.flows)


class TestAttributeSymbolsToFlows:
    def test_skips_feature_without_flows(self) -> None:
        feature = Feature(
            name="lib",
            paths=["src/lib.ts"],
            authors=[],
            total_commits=1,
            bug_fixes=0,
            bug_fix_ratio=0,
            last_modified=datetime.now(tz=timezone.utc),
            health_score=100,
            flows=[],
        )
        attribute_symbols_to_flows(feature, {})
        assert feature.shared_attributions == []

    def test_attaches_types_to_feature_level(self) -> None:
        feature = _feature_with_flows()
        file_syms = {
            "src/types.ts": FileSymbols(
                path="src/types.ts",
                flow_symbols=[],
                feature_symbols=["PaymentMethod", "Money"],
            ),
        }
        with patch("faultline.symbols.attribution._ask_llm") as mock_llm:
            mock_llm.return_value = None
            attribute_symbols_to_flows(feature, file_syms)

        assert len(feature.shared_attributions) == 1
        assert feature.shared_attributions[0].symbols == ["PaymentMethod", "Money"]

    def test_full_flow_with_mocked_llm(self) -> None:
        feature = _feature_with_flows()
        file_syms = {
            "src/stripe.ts": FileSymbols(
                path="src/stripe.ts",
                flow_symbols=["charge", "refund"],
                feature_symbols=[],
            ),
        }
        with patch("faultline.symbols.attribution._ask_llm") as mock_llm:
            mock_llm.return_value = {
                "checkout-flow": [{"file": "src/stripe.ts", "symbols": ["charge"]}],
                "refund-flow": [{"file": "src/stripe.ts", "symbols": ["refund"]}],
            }
            attribute_symbols_to_flows(feature, file_syms)

        assert feature.flows[0].symbol_attributions[0].symbols == ["charge"]
        assert feature.flows[1].symbol_attributions[0].symbols == ["refund"]


class TestPipeline:
    def test_enrich_with_symbols_skips_empty_map(self) -> None:
        from faultline.models.types import FeatureMap
        fm = FeatureMap(
            repo_path="/tmp/x",
            analyzed_at=datetime.now(tz=timezone.utc),
            total_commits=0,
            date_range_days=0,
            features=[],
        )
        # Should not raise
        enrich_with_symbols(fm, {})

    def test_enrich_runs_per_feature(self) -> None:
        from faultline.models.types import FeatureMap
        fm = FeatureMap(
            repo_path="/tmp/x",
            analyzed_at=datetime.now(tz=timezone.utc),
            total_commits=10,
            date_range_days=30,
            features=[_feature_with_flows()],
        )
        sigs = {
            "src/stripe.ts": _sig("src/stripe.ts", [
                SymbolRange(name="charge", start_line=10, end_line=45, kind="function"),
            ]),
        }
        with patch("faultline.symbols.pipeline.attribute_symbols_to_flows") as mock_attr:
            enrich_with_symbols(fm, sigs)
            mock_attr.assert_called_once()
