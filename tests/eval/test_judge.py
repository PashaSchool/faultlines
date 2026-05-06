"""Sprint 16 Day 2 — judge.py unit tests with mocked Anthropic."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tests.eval.judge import (
    JudgeResult,
    Match,
    _build_metrics,
    _cache_key,
    _coerce_match,
    _parse_response,
    judge_run,
)


# ── Fake Anthropic client ─────────────────────────────────────────────


@dataclass
class _Block:
    text: str = ""

    @property
    def type(self) -> str:
        return "text"


@dataclass
class _Response:
    content: list


class _FakeMessages:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(content=[_Block(text=json.dumps(self.payload))])


class _FakeClient:
    def __init__(self, payload: dict):
        self.messages = _FakeMessages(payload)


# ── Pure helpers ──────────────────────────────────────────────────────


def test_parse_response_extracts_envelope():
    text = '{"matches": [{"expected": "x", "detected": "y", "quality": "exact"}]}'
    out = _parse_response(text)
    assert out["matches"][0]["expected"] == "x"


def test_parse_response_handles_prose():
    text = "Here we go:\n" + '{"matches": []}\n\nDone.'
    assert _parse_response(text) == {"matches": []}


def test_parse_response_returns_none_on_garbage():
    assert _parse_response("nope") is None


def test_coerce_match_valid():
    m = _coerce_match(
        {"expected": "auth", "detected": "user-auth", "quality": "exact"},
        {"auth"},
    )
    assert m is not None
    assert m.is_hit


def test_coerce_match_normalises_NONE():
    m = _coerce_match(
        {"expected": "auth", "detected": "NONE", "quality": "none"},
        {"auth"},
    )
    assert m.detected is None


def test_coerce_match_drops_unknown_expected():
    m = _coerce_match(
        {"expected": "ghost", "detected": "x", "quality": "exact"},
        {"auth"},  # ghost not expected
    )
    assert m is None


def test_coerce_match_drops_invalid_quality():
    m = _coerce_match(
        {"expected": "auth", "detected": "x", "quality": "bogus"},
        {"auth"},
    )
    assert m is None


def test_build_metrics_empty():
    assert _build_metrics([], []) == (0.0, 0.0, 0.0)


def test_build_metrics_perfect():
    matches = [
        Match("a", "a", "exact"),
        Match("b", "b", "exact"),
    ]
    cov, prec, f1 = _build_metrics(matches, ["a", "b"])
    assert cov == 1.0
    assert prec == 1.0
    assert f1 == 1.0


def test_build_metrics_partial_coverage():
    matches = [
        Match("a", "a", "exact"),
        Match("b", None, "none"),
    ]
    cov, prec, f1 = _build_metrics(matches, ["a"])
    assert cov == 0.5
    assert prec == 1.0
    assert pytest.approx(f1, abs=0.01) == 2 * 0.5 * 1.0 / (0.5 + 1.0)


def test_build_metrics_dedups_detected():
    """Duplicate detected names don't inflate precision denominator."""
    matches = [Match("a", "a", "exact")]
    cov, prec, _ = _build_metrics(matches, ["a", "a", "a"])
    assert prec == 1.0  # |set(detected)| = 1


def test_cache_key_stable_under_reorder():
    a = _cache_key(["x", "y"], ["a", "b"])
    b = _cache_key(["y", "x"], ["b", "a"])
    assert a == b


def test_cache_key_changes_with_added_name():
    a = _cache_key(["x"], ["a"])
    b = _cache_key(["x", "y"], ["a"])
    assert a != b


# ── judge_run with mocked client ──────────────────────────────────────


def test_judge_run_happy_path(tmp_path):
    client = _FakeClient({
        "matches": [
            {"expected": "auth", "detected": "user-auth", "quality": "exact"},
            {"expected": "billing", "detected": "subscriptions", "quality": "partial"},
            {"expected": "search", "detected": "NONE", "quality": "none"},
        ],
        "extras": ["debug-tools"],
    })
    result = judge_run(
        expected=["auth", "billing", "search"],
        detected=["user-auth", "subscriptions", "debug-tools"],
        repo="testrepo",
        client=client,
        cache_dir=tmp_path,
    )
    assert result.coverage == pytest.approx(2 / 3)
    assert result.precision == pytest.approx(2 / 3)
    assert result.hits() == 2
    assert "debug-tools" in result.extras


def test_judge_run_fills_missing_expected(tmp_path):
    """Judge silently dropped 'billing' — judge_run must add a none entry."""
    client = _FakeClient({
        "matches": [
            {"expected": "auth", "detected": "auth", "quality": "exact"},
        ],
    })
    result = judge_run(
        expected=["auth", "billing"],
        detected=["auth"],
        repo="testrepo",
        client=client,
        cache_dir=tmp_path,
    )
    assert len(result.matches) == 2
    billing_match = next(m for m in result.matches if m.expected == "billing")
    assert billing_match.quality == "none"
    assert billing_match.detected is None


def test_judge_run_uses_cache(tmp_path):
    client = _FakeClient({
        "matches": [{"expected": "x", "detected": "x", "quality": "exact"}],
    })
    judge_run(
        expected=["x"], detected=["x"], repo="r1", client=client, cache_dir=tmp_path,
    )
    assert len(client.messages.calls) == 1

    # Second run with same inputs → cache hit, no API call
    client2 = _FakeClient({"matches": []})
    result = judge_run(
        expected=["x"], detected=["x"], repo="r1", client=client2, cache_dir=tmp_path,
    )
    assert result.cache_hit
    assert client2.messages.calls == []
    assert result.coverage == 1.0


def test_judge_run_no_api_key_returns_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = judge_run(
        expected=["auth"], detected=["auth"], repo="r2",
        cache_dir=tmp_path,
    )
    assert result.coverage == 0.0
    assert result.precision == 0.0


def test_judge_run_empty_expected_returns_zero(tmp_path):
    client = _FakeClient({"matches": []})
    result = judge_run(
        expected=[], detected=["x"], repo="r3",
        client=client, cache_dir=tmp_path,
    )
    assert result.coverage == 0.0
    # Empty expected is degenerate — judge isn't even called
    assert client.messages.calls == []


def test_judge_run_filters_hallucinated_expected(tmp_path):
    """Judge invented an expected name not in our list — filter it."""
    client = _FakeClient({
        "matches": [
            {"expected": "auth", "detected": "auth", "quality": "exact"},
            {"expected": "ghost", "detected": "x", "quality": "exact"},
        ],
    })
    result = judge_run(
        expected=["auth"], detected=["auth"],
        repo="r4", client=client, cache_dir=tmp_path,
    )
    expected_names = {m.expected for m in result.matches}
    assert expected_names == {"auth"}
