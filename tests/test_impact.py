"""Tests for faultline/impact/risk.py."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from faultline.impact.risk import (
    predict_impact,
    _find_affected_features,
    _find_missing_cochanges,
    _build_risk_signals,
    _calculate_risk_level,
    _estimate_regression_probability,
)


def _sample_features() -> list[dict]:
    return [
        {
            "name": "payments",
            "description": "Stripe payment processing",
            "paths": ["src/payments/stripe.ts", "src/payments/webhook.ts", "src/payments/checkout.ts"],
            "authors": ["alice"],
            "total_commits": 80,
            "bug_fixes": 50,
            "bug_fix_ratio": 0.625,
            "health_score": 20.0,
            "coverage_pct": 15.0,
            "flows": [
                {
                    "name": "checkout-flow",
                    "paths": ["src/payments/checkout.ts"],
                    "bus_factor": 1,
                    "health_score": 12.0,
                },
            ],
        },
        {
            "name": "auth",
            "description": "Authentication",
            "paths": ["src/auth/login.ts", "src/auth/session.ts"],
            "authors": ["alice", "bob", "charlie"],
            "total_commits": 40,
            "bug_fixes": 5,
            "bug_fix_ratio": 0.125,
            "health_score": 90.0,
            "coverage_pct": 85.0,
            "flows": [
                {
                    "name": "login-flow",
                    "paths": ["src/auth/login.ts"],
                    "bus_factor": 3,
                    "health_score": 88.0,
                },
            ],
        },
    ]


def _sample_map() -> dict:
    return {
        "repo_path": "/tmp/sample",
        "analyzed_at": "2026-04-12T10:00:00Z",
        "total_commits": 500,
        "features": _sample_features(),
    }


class TestFindAffectedFeatures:
    def test_finds_matching_feature(self) -> None:
        features = _sample_features()
        affected = _find_affected_features(["src/payments/stripe.ts"], features)
        assert len(affected) == 1
        assert affected[0]["name"] == "payments"

    def test_finds_multiple_features(self) -> None:
        features = _sample_features()
        affected = _find_affected_features(
            ["src/payments/stripe.ts", "src/auth/login.ts"],
            features,
        )
        assert len(affected) == 2

    def test_returns_empty_for_unknown_files(self) -> None:
        features = _sample_features()
        affected = _find_affected_features(["src/unknown.ts"], features)
        assert affected == []


class TestFindMissingCochanges:
    def test_finds_missing_partner(self) -> None:
        pairs = [
            ("src/payments/stripe.ts", "src/payments/webhook.ts", 0.8),
        ]
        missing = _find_missing_cochanges(["src/payments/stripe.ts"], pairs)
        assert len(missing) == 1
        assert missing[0]["file"] == "src/payments/webhook.ts"
        assert missing[0]["cochange_score"] == 0.8
        assert missing[0]["confidence"] == "high"

    def test_no_missing_when_all_included(self) -> None:
        pairs = [
            ("src/payments/stripe.ts", "src/payments/webhook.ts", 0.8),
        ]
        missing = _find_missing_cochanges(
            ["src/payments/stripe.ts", "src/payments/webhook.ts"],
            pairs,
        )
        assert missing == []

    def test_confidence_labels(self) -> None:
        pairs = [
            ("a.ts", "b.ts", 0.6),
            ("a.ts", "c.ts", 0.3),
            ("a.ts", "d.ts", 0.1),
        ]
        missing = _find_missing_cochanges(["a.ts"], pairs)
        assert missing[0]["confidence"] == "high"
        assert missing[1]["confidence"] == "medium"


class TestBuildRiskSignals:
    def test_critical_health_signal(self) -> None:
        affected = [{"name": "payments", "health_score": 20, "coverage_pct": 15, "authors": ["alice"], "flows": [{"bus_factor": 1}]}]
        signals = _build_risk_signals(affected, [], [])
        assert any("critical health" in s for s in signals)

    def test_bus_factor_signal(self) -> None:
        affected = [{"name": "payments", "health_score": 80, "coverage_pct": 90, "authors": ["alice"], "flows": [{"bus_factor": 1}]}]
        signals = _build_risk_signals(affected, [], [])
        assert any("bus factor 1" in s for s in signals)

    def test_coverage_signal(self) -> None:
        affected = [{"name": "payments", "health_score": 80, "coverage_pct": 10, "authors": ["alice", "bob"], "flows": []}]
        signals = _build_risk_signals(affected, [], [])
        assert any("10% test coverage" in s for s in signals)

    def test_missing_cochange_signal(self) -> None:
        missing = [{"file": "webhook.ts", "cochange_score": 0.8, "confidence": "high"}]
        signals = _build_risk_signals([], missing, [])
        assert any("co-change are missing" in s for s in signals)


class TestCalculateRiskLevel:
    def test_critical_with_multiple_severe_signals(self) -> None:
        signals = [
            "Feature 'x' has critical health (15).",
            "Feature 'x' has bus factor 1.",
        ]
        assert _calculate_risk_level(signals) == "critical"

    def test_high_with_one_severe_signal(self) -> None:
        signals = ["Feature 'x' has critical health (15)."]
        assert _calculate_risk_level(signals) == "high"

    def test_medium_with_minor_signal(self) -> None:
        signals = ["Feature 'x' is at risk (health 45)."]
        assert _calculate_risk_level(signals) == "medium"

    def test_low_with_no_signals(self) -> None:
        assert _calculate_risk_level([]) == "low"


class TestEstimateRegressionProbability:
    def test_high_ratio_for_buggy_feature(self) -> None:
        features = _sample_features()
        prob = _estimate_regression_probability(
            ["src/payments/stripe.ts"], features, ".", 365,
        )
        assert prob >= 0.5

    def test_low_ratio_for_healthy_feature(self) -> None:
        features = _sample_features()
        prob = _estimate_regression_probability(
            ["src/auth/login.ts"], features, ".", 365,
        )
        assert prob <= 0.2

    def test_zero_for_unknown_files(self) -> None:
        features = _sample_features()
        prob = _estimate_regression_probability(
            ["src/unknown.ts"], features, ".", 365,
        )
        assert prob == 0.0


class TestPredictImpact:
    def test_full_report(self) -> None:
        fm = _sample_map()
        with patch("faultline.impact.risk._compute_cochange_pairs") as mock_cc:
            mock_cc.return_value = [
                ("src/payments/stripe.ts", "src/payments/webhook.ts", 0.85),
            ]
            result = predict_impact(
                changed_files=["src/payments/stripe.ts"],
                feature_map=fm,
                repo_path=".",
            )

        assert result["risk_level"] in ("critical", "high", "medium", "low")
        assert len(result["affected_features"]) == 1
        assert result["affected_features"][0]["name"] == "payments"
        assert result["affected_features"][0]["health"] == 20
        assert result["regression_probability"] >= 0.5
        assert len(result["missing_cochanges"]) == 1
        assert result["missing_cochanges"][0]["file"] == "src/payments/webhook.ts"
        assert len(result["risk_signals"]) > 0

    def test_safe_change(self) -> None:
        fm = _sample_map()
        with patch("faultline.impact.risk._compute_cochange_pairs") as mock_cc:
            mock_cc.return_value = []
            result = predict_impact(
                changed_files=["src/auth/login.ts"],
                feature_map=fm,
                repo_path=".",
            )

        assert result["risk_level"] in ("low", "medium")
        assert result["regression_probability"] <= 0.2
