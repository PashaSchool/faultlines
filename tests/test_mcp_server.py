"""Tests for faultline/mcp_server.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from faultline.mcp_server import (
    _load_map,
    _savings_metadata,
    find_feature,
    get_feature_files,
    get_feature_owners,
    get_flow_files,
    get_hotspots,
    get_repo_summary,
    list_features,
)


def _sample_map() -> dict:
    return {
        "repo_path": "/tmp/sample",
        "remote_url": "https://github.com/org/sample",
        "analyzed_at": "2026-04-11T10:00:00Z",
        "total_commits": 500,
        "date_range_days": 365,
        "features": [
            {
                "name": "payments",
                "description": "Stripe payment processing",
                "paths": ["src/payments/charge.ts", "src/payments/webhook.ts"],
                "authors": ["alice", "bob"],
                "total_commits": 80,
                "bug_fixes": 40,
                "bug_fix_ratio": 0.5,
                "health_score": 25.0,
                "coverage_pct": 60.0,
                "flows": [
                    {
                        "name": "checkout-flow",
                        "paths": ["src/payments/checkout.ts"],
                        "total_commits": 30,
                        "bug_fixes": 20,
                        "bug_fix_ratio": 0.67,
                        "health_score": 18.0,
                        "bus_factor": 1,
                        "hotspot_files": ["src/payments/charge.ts"],
                    },
                ],
            },
            {
                "name": "auth",
                "description": "User authentication",
                "paths": ["src/auth/login.ts"],
                "authors": ["alice", "bob", "charlie"],
                "total_commits": 40,
                "bug_fixes": 5,
                "bug_fix_ratio": 0.125,
                "health_score": 85.0,
                "coverage_pct": 92.0,
                "flows": [],
            },
        ],
    }


@pytest.fixture
def fake_map(tmp_path: Path) -> Path:
    """Write a sample feature map and point the loader at it."""
    p = tmp_path / "feature-map-sample.json"
    p.write_text(json.dumps(_sample_map()))
    with patch.dict("os.environ", {"FAULTLINE_MAP_PATH": str(p)}):
        yield p


class TestLoadMap:
    def test_loads_from_env_path(self, fake_map: Path) -> None:
        data = _load_map()
        assert data["repo_path"] == "/tmp/sample"
        assert len(data["features"]) == 2

    def test_raises_when_env_path_missing(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"FAULTLINE_MAP_PATH": str(tmp_path / "missing.json")}):
            with pytest.raises(RuntimeError, match="does not exist"):
                _load_map()


class TestSavingsMetadata:
    def test_returns_positive_savings_for_small_response(self) -> None:
        m = _savings_metadata(files_returned=3)
        assert m["estimated_tokens_saved"] > 0
        assert m["files_returned"] == 3
        assert m["baseline_tokens"] > 0

    def test_clamps_savings_to_zero_when_overshooting(self) -> None:
        m = _savings_metadata(files_returned=999)
        assert m["estimated_tokens_saved"] == 0


class TestListFeatures:
    def test_returns_sorted_by_health(self, fake_map: Path) -> None:
        # fn.fn form — bypass the MCP wrapper to call the underlying callable
        result = list_features()
        assert result["total_features"] == 2
        # Sorted by health ascending — payments (25) before auth (85)
        assert result["features"][0]["name"] == "payments"
        assert result["features"][1]["name"] == "auth"

    def test_includes_savings_metadata(self, fake_map: Path) -> None:
        result = list_features()
        assert "_savings_metadata" in result


class TestFindFeature:
    def test_matches_by_name(self, fake_map: Path) -> None:
        result = find_feature(query="payments")
        assert result is not None
        assert result["name"] == "payments"
        assert len(result["files"]) == 2
        assert "alice" in result["owners"]

    def test_matches_by_description(self, fake_map: Path) -> None:
        result = find_feature(query="stripe")
        assert result is not None
        assert result["name"] == "payments"

    def test_case_insensitive(self, fake_map: Path) -> None:
        result = find_feature(query="AUTH")
        assert result is not None
        assert result["name"] == "auth"

    def test_returns_none_for_unknown(self, fake_map: Path) -> None:
        assert find_feature(query="nonexistent") is None


class TestGetFeatureFiles:
    def test_returns_files_for_known_feature(self, fake_map: Path) -> None:
        result = get_feature_files(feature_name="payments")
        assert result["feature"] == "payments"
        assert len(result["files"]) == 2
        assert "src/payments/charge.ts" in result["files"]

    def test_returns_hotspot_files(self, fake_map: Path) -> None:
        result = get_feature_files(feature_name="payments")
        assert "src/payments/charge.ts" in result["hotspot_files"]

    def test_error_for_unknown_feature(self, fake_map: Path) -> None:
        result = get_feature_files(feature_name="unknown")
        assert "error" in result
        assert "payments" in result["available"]


class TestGetHotspots:
    def test_returns_riskiest_first(self, fake_map: Path) -> None:
        result = get_hotspots(limit=2)
        assert len(result["hotspots"]) == 2
        assert result["hotspots"][0]["name"] == "payments"
        assert result["hotspots"][0]["health"] < result["hotspots"][1]["health"]

    def test_respects_limit(self, fake_map: Path) -> None:
        result = get_hotspots(limit=1)
        assert len(result["hotspots"]) == 1


class TestGetFeatureOwners:
    def test_returns_authors_and_bus_factor(self, fake_map: Path) -> None:
        result = get_feature_owners(feature_name="payments")
        assert result["feature"] == "payments"
        assert result["owners"] == ["alice", "bob"]
        assert result["bus_factor"] == 1
        assert result["at_risk"] is True

    def test_not_at_risk_when_bus_factor_higher(self, fake_map: Path) -> None:
        result = get_feature_owners(feature_name="auth")
        assert result["at_risk"] is False

    def test_error_for_unknown_feature(self, fake_map: Path) -> None:
        result = get_feature_owners(feature_name="unknown")
        assert "error" in result


class TestGetFlowFiles:
    def test_returns_flow_files(self, fake_map: Path) -> None:
        result = get_flow_files(feature_name="payments", flow_name="checkout-flow")
        assert result["flow"] == "checkout-flow"
        assert result["files"] == ["src/payments/checkout.ts"]
        assert result["hotspot_files"] == ["src/payments/charge.ts"]

    def test_error_for_unknown_flow(self, fake_map: Path) -> None:
        result = get_flow_files(feature_name="payments", flow_name="missing")
        assert "error" in result


class TestGetRepoSummary:
    def test_returns_aggregated_stats(self, fake_map: Path) -> None:
        result = get_repo_summary()
        assert result["total_features"] == 2
        assert result["total_commits"] == 500
        assert result["total_bug_fixes"] == 45
        assert result["features_at_risk"] == 1  # payments health < 50
        assert result["avg_coverage_pct"] == 76.0  # (60 + 92) / 2

    def test_avg_health_computed_correctly(self, fake_map: Path) -> None:
        result = get_repo_summary()
        assert result["avg_health_score"] == 55.0  # (25 + 85) / 2
