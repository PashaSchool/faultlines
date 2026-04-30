"""Tests for the prior-scan loader (Stage 1 of incremental pipeline)."""

from __future__ import annotations

import json

import pytest

from faultline.llm.scan_loader import (
    PriorScan,
    find_prior_scan_for,
    load_scan_as_seed,
)


def _write_scan(tmp_path, **overrides):
    body = {
        "analyzed_at": "2026-04-29T11:41:31.397036Z",
        "repo_path": "/tmp/documenso",
        "remote_url": None,
        "total_commits": 200,
        "date_range_days": 365,
        "last_scanned_sha": "8bad62cc923539d73a6ccc94b1a8740bab7d0da9",
        "file_hashes": {},
        "symbol_hashes": {},
        "features": [
            {
                "name": "user-authentication",
                "display_name": "User Authentication",
                "description": "Login, signup, sessions.",
                "paths": ["src/auth/login.ts", "src/auth/session.ts"],
                "total_commits": 23,
                "bug_fix_ratio": 0.6,
                "health_score": 25.5,
                "flows": [
                    {
                        "name": "sign-in-flow",
                        "display_name": "Sign In",
                        "description": "User enters creds and logs in.",
                        "total_commits": 7,
                        "bug_fix_ratio": 0.4,
                        "entry_point_file": "src/auth/login.ts",
                        "entry_point_line": 14,
                        "participants": [
                            {"path": "src/auth/login.ts", "layer": "ui"},
                        ],
                    },
                ],
            },
            {
                "name": "shared-infra",
                "description": "Configs and tooling.",
                "paths": ["package.json", "tsconfig.json"],
                "total_commits": 50,
                "flows": [],
            },
        ],
    }
    body.update(overrides)
    p = tmp_path / "scan.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


def test_load_scan_returns_prior_scan(tmp_path):
    path = _write_scan(tmp_path)
    prior = load_scan_as_seed(path)
    assert isinstance(prior, PriorScan)
    assert prior.last_sha == "8bad62cc923539d73a6ccc94b1a8740bab7d0da9"
    assert prior.scan_meta["total_commits"] == 200


def test_load_scan_rehydrates_features(tmp_path):
    path = _write_scan(tmp_path)
    prior = load_scan_as_seed(path)
    assert set(prior.features) == {"user-authentication", "shared-infra"}
    assert prior.features["user-authentication"] == [
        "src/auth/login.ts", "src/auth/session.ts",
    ]
    assert prior.result.descriptions["user-authentication"].startswith("Login")


def test_load_scan_rehydrates_flows(tmp_path):
    path = _write_scan(tmp_path)
    prior = load_scan_as_seed(path)
    assert prior.result.flows["user-authentication"] == ["sign-in-flow"]
    assert "sign-in-flow" in prior.result.flow_descriptions["user-authentication"]
    # Participants come back as raw dicts (rebuild only if needed).
    flow_participants = prior.result.flow_participants["user-authentication"]
    # Rehydrated as TracedParticipant — the intermediate dataclass
    # downstream injectors expect (with ``file`` attribute, not ``path``).
    p0 = flow_participants["sign-in-flow"][0]
    assert p0.file == "src/auth/login.ts"
    assert p0.layer == "ui"


def test_load_scan_carries_feature_stats(tmp_path):
    path = _write_scan(tmp_path)
    prior = load_scan_as_seed(path)
    stats = prior.stats_for("user-authentication")
    assert stats["total_commits"] == 23
    assert stats["bug_fix_ratio"] == 0.6
    assert stats["health_score"] == 25.5


def test_load_scan_carries_flow_stats(tmp_path):
    path = _write_scan(tmp_path)
    prior = load_scan_as_seed(path)
    fs = prior.flow_stats_for("user-authentication", "sign-in-flow")
    assert fs["total_commits"] == 7
    assert fs["entry_point_file"] == "src/auth/login.ts"
    assert fs["entry_point_line"] == 14


def test_load_scan_handles_feature_without_flows(tmp_path):
    path = _write_scan(tmp_path)
    prior = load_scan_as_seed(path)
    # shared-infra had empty flows list — should NOT create entries.
    assert "shared-infra" not in prior.result.flows
    assert "shared-infra" not in prior.result.flow_descriptions


def test_load_scan_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scan_as_seed(tmp_path / "does-not-exist.json")


def test_load_scan_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_scan_as_seed(p)


def test_load_scan_wrong_shape_raises(tmp_path):
    p = tmp_path / "wrong.json"
    p.write_text(json.dumps({"features": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="features"):
        load_scan_as_seed(p)


def test_load_scan_missing_optional_fields(tmp_path):
    minimal = {
        "features": [
            {"name": "x", "paths": ["a.ts"]},
        ],
    }
    p = tmp_path / "min.json"
    p.write_text(json.dumps(minimal), encoding="utf-8")
    prior = load_scan_as_seed(p)
    assert prior.last_sha is None
    assert prior.features == {"x": ["a.ts"]}
    assert prior.scan_meta["total_commits"] is None


def test_find_prior_scan_returns_most_recent(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    older = cache / "feature-map-acme-20260101-000000.json"
    newer = cache / "feature-map-acme-20260201-000000.json"
    older.write_text("{}")
    newer.write_text("{}")
    # Force mtime ordering even though the filesystem usually orders
    # files in creation order.
    import os
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_750_000_000, 1_750_000_000))

    repo = tmp_path / "Acme"
    repo.mkdir()
    found = find_prior_scan_for(repo, cache_dir=cache)
    assert found is not None
    assert found.name.endswith("20260201-000000.json")


def test_find_prior_scan_returns_none_when_missing(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    repo = tmp_path / "nothing"
    repo.mkdir()
    assert find_prior_scan_for(repo, cache_dir=cache) is None
