"""Sprint 16 Day 4 — failure_modes tests."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tests.eval.failure_modes import (
    ClassifiedMiss,
    FailureMode,
    _coerce_mode,
    _parse_response,
    classify_all,
    classify_miss,
    summarise,
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
    def __init__(self, scripted: list[dict]):
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._scripted:
            raise RuntimeError("scripted exhausted")
        payload = self._scripted.pop(0)
        return _Response(content=[_Block(text=json.dumps(payload))])


class _FakeClient:
    def __init__(self, scripted: list[dict]):
        self.messages = _FakeMessages(scripted)


# ── Helpers ───────────────────────────────────────────────────────────


def test_parse_response_valid():
    text = '{"mode": "OVER_CLUSTERING", "reasoning": "merged"}'
    assert _parse_response(text) == {"mode": "OVER_CLUSTERING", "reasoning": "merged"}


def test_parse_response_handles_prose():
    text = "Sure!\n" + '{"mode": "MISSING_KEY", "reasoning": "x"}\n\nDone.'
    assert _parse_response(text)["mode"] == "MISSING_KEY"


def test_parse_response_returns_none_on_garbage():
    assert _parse_response("nope") is None


def test_coerce_mode_valid():
    mode, reasoning = _coerce_mode({"mode": "GENERIC_NAMING", "reasoning": "utils"})
    assert mode == FailureMode.GENERIC_NAMING
    assert reasoning == "utils"


def test_coerce_mode_falls_back_to_uncategorised_on_invalid():
    mode, reasoning = _coerce_mode({"mode": "BOGUS_MODE", "reasoning": ""})
    assert mode == FailureMode.UNCATEGORISED


def test_coerce_mode_lowercase_normalised():
    mode, _ = _coerce_mode({"mode": "over_clustering"})
    assert mode == FailureMode.OVER_CLUSTERING


# ── classify_miss ──────────────────────────────────────────────────────


def test_classify_miss_happy_path(tmp_path):
    client = _FakeClient([
        {"mode": "OVER_CLUSTERING", "reasoning": "auth + signin merged"},
    ])
    result = classify_miss(
        expected="auth", detected="i18n",
        detected_features=["i18n", "workflow"],
        client=client, cache_dir=tmp_path,
    )
    assert result.mode == FailureMode.OVER_CLUSTERING
    assert "auth" in result.reasoning


def test_classify_miss_no_api_key_uncategorised(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = classify_miss(
        expected="auth", detected=None,
        detected_features=[], cache_dir=tmp_path,
    )
    assert result.mode == FailureMode.UNCATEGORISED


def test_classify_miss_uses_cache(tmp_path):
    client = _FakeClient([
        {"mode": "MISSING_KEY", "reasoning": "no candidate"},
    ])
    classify_miss(
        expected="auth", detected=None,
        detected_features=["x"], client=client, cache_dir=tmp_path,
    )
    assert len(client.messages.calls) == 1

    client2 = _FakeClient([])  # empty — would raise if called
    result = classify_miss(
        expected="auth", detected=None,
        detected_features=["x"], client=client2, cache_dir=tmp_path,
    )
    assert result.mode == FailureMode.MISSING_KEY
    assert client2.messages.calls == []


def test_classify_miss_invalid_mode_falls_to_uncategorised(tmp_path):
    client = _FakeClient([
        {"mode": "MADE_UP_MODE", "reasoning": "nonsense"},
    ])
    result = classify_miss(
        expected="x", detected=None,
        detected_features=[], client=client, cache_dir=tmp_path,
    )
    assert result.mode == FailureMode.UNCATEGORISED


def test_classify_miss_unparseable_response_uncategorised(tmp_path):
    @dataclass
    class _BadBlock:
        text: str = "this is not json"

        @property
        def type(self) -> str:
            return "text"

    class _BadMessages:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _Response(content=[_BadBlock()])

    class _BadClient:
        def __init__(self):
            self.messages = _BadMessages()

    result = classify_miss(
        expected="x", detected=None,
        detected_features=[], client=_BadClient(), cache_dir=tmp_path,
    )
    assert result.mode == FailureMode.UNCATEGORISED


# ── classify_all + summarise ───────────────────────────────────────────


def test_classify_all_preserves_order(tmp_path):
    client = _FakeClient([
        {"mode": "OVER_CLUSTERING", "reasoning": "a"},
        {"mode": "MISSING_KEY", "reasoning": "b"},
        {"mode": "GENERIC_NAMING", "reasoning": "c"},
    ])
    results = classify_all(
        misses=[("auth", "i18n"), ("billing", None), ("search", "utils")],
        detected_features=["i18n", "utils"],
        client=client, cache_dir=tmp_path,
    )
    assert [r.expected for r in results] == ["auth", "billing", "search"]
    assert [r.mode for r in results] == [
        FailureMode.OVER_CLUSTERING,
        FailureMode.MISSING_KEY,
        FailureMode.GENERIC_NAMING,
    ]


def test_summarise_counts_modes():
    classified = [
        ClassifiedMiss("a", None, FailureMode.OVER_CLUSTERING),
        ClassifiedMiss("b", None, FailureMode.OVER_CLUSTERING),
        ClassifiedMiss("c", None, FailureMode.MISSING_KEY),
    ]
    counts = summarise(classified)
    assert counts[FailureMode.OVER_CLUSTERING] == 2
    assert counts[FailureMode.MISSING_KEY] == 1
