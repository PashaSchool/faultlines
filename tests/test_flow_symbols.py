"""Sprint 12 Day 4-5 — flow_symbols (Layer B) tests.

Mocked client throughout — no real Haiku calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from faultline.llm.flow_symbols import (
    CandidateSymbol,
    PickedSymbol,
    _build_prompt,
    _candidate_files_for_flow,
    _candidates_hash,
    _coerce_picked,
    _filter_by_candidates,
    _parse_response,
    _trim_candidates,
    pick_flow_symbols,
    resolve_flow_symbols,
    resolve_picked,
)
from faultline.llm.sonnet_scanner import DeepScanResult


# ── Fake Anthropic client ─────────────────────────────────────────────


@dataclass
class _Block:
    text: str = ""

    @property
    def type(self) -> str:
        return "text"


@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class _Response:
    content: list
    usage: _Usage = None


class _FakeMessages:
    def __init__(self, response_payload: dict):
        self._payload = response_payload
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(
            content=[_Block(text=json.dumps(self._payload))],
            usage=_Usage(),
        )


class _FakeClient:
    def __init__(self, response_payload: dict):
        self.messages = _FakeMessages(response_payload)


# ── Pure helper tests ─────────────────────────────────────────────────


def test_parse_response_extracts_selected_array():
    text = '{"selected": [{"file": "a.ts", "symbol": "x", "confidence": 5}]}'
    out = _parse_response(text)
    assert out == [{"file": "a.ts", "symbol": "x", "confidence": 5}]


def test_parse_response_handles_prose_around_json():
    text = "Sure! Here you go:\n" + '{"selected": []}\n\nDone.'
    assert _parse_response(text) == []


def test_parse_response_returns_none_on_invalid():
    assert _parse_response("nope") is None
    assert _parse_response('{"selected": "not a list"}') is None


def test_coerce_picked_accepts_minimal_entry():
    p = _coerce_picked({"file": "a.ts", "symbol": "x", "confidence": 5})
    assert p is not None
    assert p.file == "a.ts" and p.name == "x" and p.confidence == 5


def test_coerce_picked_rejects_missing_fields():
    assert _coerce_picked({"symbol": "x"}) is None
    assert _coerce_picked({"file": "a.ts"}) is None


def test_coerce_picked_clamps_confidence():
    p = _coerce_picked({"file": "a.ts", "symbol": "x", "confidence": 99})
    assert p.confidence == 5
    p = _coerce_picked({"file": "a.ts", "symbol": "x", "confidence": -3})
    assert p.confidence == 1


def test_filter_drops_hallucinated_picks():
    cands = [
        CandidateSymbol("a.ts", "real", "function", 1, 5),
    ]
    picks = [
        PickedSymbol(file="a.ts", name="real", confidence=5),
        PickedSymbol(file="a.ts", name="halluc", confidence=5),
        PickedSymbol(file="b.ts", name="real", confidence=5),
    ]
    out = _filter_by_candidates(picks, cands)
    assert len(out) == 1
    assert out[0].name == "real" and out[0].file == "a.ts"


def test_resolve_picked_builds_symbol_ranges():
    cands = [
        CandidateSymbol("a.ts", "login", "function", 10, 25),
        CandidateSymbol("a.ts", "logout", "function", 27, 40),
        CandidateSymbol("b.ts", "AuthService", "class", 1, 80),
    ]
    picks = [
        PickedSymbol(file="a.ts", name="login", confidence=5),
        PickedSymbol(file="b.ts", name="AuthService", confidence=4),
    ]
    out = resolve_picked(picks, cands)
    assert "a.ts" in out and "b.ts" in out
    assert out["a.ts"][0].name == "login"
    assert out["a.ts"][0].start_line == 10
    assert out["a.ts"][0].end_line == 25
    assert out["b.ts"][0].kind == "class"


def test_trim_candidates_caps_files_and_symbols():
    cands: list[CandidateSymbol] = []
    for fi in range(20):
        for si in range(40):
            cands.append(CandidateSymbol(
                file=f"f{fi}.ts", name=f"s{si}",
                kind="function", start_line=si * 10, end_line=si * 10 + 5,
            ))
    out = _trim_candidates(cands)
    files = {c.file for c in out}
    assert len(files) <= 15
    # Per-file cap
    by_file: dict[str, int] = {}
    for c in out:
        by_file[c.file] = by_file.get(c.file, 0) + 1
    assert all(n <= 25 for n in by_file.values())


def test_candidates_hash_stable_under_reorder():
    a = [
        CandidateSymbol("a.ts", "x", "function", 1, 10),
        CandidateSymbol("b.ts", "y", "class", 1, 5),
    ]
    b = [
        CandidateSymbol("b.ts", "y", "class", 1, 5),
        CandidateSymbol("a.ts", "x", "function", 1, 10),
    ]
    assert _candidates_hash(a) == _candidates_hash(b)


def test_candidates_hash_changes_with_new_symbol():
    a = [CandidateSymbol("a.ts", "x", "function", 1, 10)]
    b = a + [CandidateSymbol("a.ts", "y", "function", 12, 20)]
    assert _candidates_hash(a) != _candidates_hash(b)


def test_build_prompt_includes_flow_metadata():
    cands = [CandidateSymbol("a.ts", "login", "function", 1, 10)]
    out = _build_prompt("login-flow", "User signs in", cands)
    assert "login-flow" in out
    assert "User signs in" in out
    assert "login" in out
    assert "a.ts" in out


# ── Single-flow API ───────────────────────────────────────────────────


def test_pick_flow_symbols_happy_path():
    cands = [
        CandidateSymbol("a.ts", "login", "function", 1, 20),
        CandidateSymbol("a.ts", "noise", "const", 25, 27),
    ]
    client = _FakeClient({
        "selected": [
            {"file": "a.ts", "symbol": "login", "confidence": 5},
        ],
    })
    out = pick_flow_symbols("login-flow", "user signs in", cands, client=client)
    assert len(out) == 1
    assert out[0].name == "login"


def test_pick_flow_symbols_empty_candidates_no_call():
    client = _FakeClient({"selected": []})
    out = pick_flow_symbols("x", "", [], client=client)
    assert out == []
    assert client.messages.calls == []  # no API call


def test_pick_flow_symbols_drops_low_confidence():
    cands = [CandidateSymbol("a.ts", "x", "function", 1, 5)]
    client = _FakeClient({
        "selected": [{"file": "a.ts", "symbol": "x", "confidence": 1}],
    })
    out = pick_flow_symbols("x", "", cands, client=client)
    assert out == []


def test_pick_flow_symbols_filters_hallucinated():
    cands = [CandidateSymbol("a.ts", "real", "function", 1, 5)]
    client = _FakeClient({
        "selected": [
            {"file": "a.ts", "symbol": "real", "confidence": 5},
            {"file": "ghost.ts", "symbol": "made-up", "confidence": 5},
        ],
    })
    out = pick_flow_symbols("x", "", cands, client=client)
    assert len(out) == 1
    assert out[0].name == "real"


# ── Pipeline driver tests ─────────────────────────────────────────────


_TS_AUTH = """\
export function login(email, password) {
  return api.post('/login', { email, password });
}

export function logout() {
  return api.post('/logout');
}

export const NOISE = 5000;
"""


def _result_with_one_flow() -> DeepScanResult:
    return DeepScanResult(
        features={
            "auth": ["web/auth/api.ts"],
            "ui": ["web/ui/button.ts"],
        },
        flows={"auth": ["login-flow"]},
        descriptions={"auth": "user authentication"},
        flow_descriptions={"auth": {"login-flow": "user signs in"}},
    )


def test_resolve_flow_symbols_populates_participants(tmp_path: Path):
    result = _result_with_one_flow()

    def loader(rel: str) -> str | None:
        if rel.endswith("api.ts"):
            return _TS_AUTH
        return None

    client = _FakeClient({
        "selected": [
            {"file": "web/auth/api.ts", "symbol": "login", "confidence": 5},
        ],
    })
    n = resolve_flow_symbols(
        result,
        source_loader=loader,
        client=client,
        cache_dir=tmp_path,
        repo_slug="test",
    )
    assert n == 1
    parts = result.flow_participants["auth"]["login-flow"]
    assert len(parts) == 1
    assert parts[0]["path"] == "web/auth/api.ts"
    syms = parts[0]["symbols"]
    assert len(syms) == 1
    assert syms[0].name == "login"
    assert syms[0].start_line >= 1


def test_resolve_flow_symbols_uses_cache(tmp_path: Path):
    result = _result_with_one_flow()

    def loader(rel: str) -> str | None:
        if rel.endswith("api.ts"):
            return _TS_AUTH
        return None

    client = _FakeClient({
        "selected": [
            {"file": "web/auth/api.ts", "symbol": "login", "confidence": 5},
        ],
    })
    resolve_flow_symbols(
        result, source_loader=loader, client=client,
        cache_dir=tmp_path, repo_slug="test",
    )
    assert len(client.messages.calls) == 1

    # Second pass on a fresh result with same candidates → cache hit
    result2 = _result_with_one_flow()
    client2 = _FakeClient({"selected": []})  # would return empty if called
    n = resolve_flow_symbols(
        result2, source_loader=loader, client=client2,
        cache_dir=tmp_path, repo_slug="test",
    )
    assert n == 1
    assert client2.messages.calls == []  # NO fresh call


def test_candidate_files_for_flow_prefers_token_match():
    result = DeepScanResult(
        features={
            "ui": [
                "web/ui/button.ts",
                "web/ui/login_form.ts",
                "web/ui/modal.ts",
            ],
        },
        flows={"ui": ["login-flow"]},
    )
    files = _candidate_files_for_flow("login-flow", "ui", result)
    assert "web/ui/login_form.ts" == files[0]


def test_candidate_files_uses_participants_when_present():
    """If flow_participants already has paths, use those."""
    result = DeepScanResult(
        features={"auth": ["a.ts", "b.ts", "c.ts"]},
        flows={"auth": ["x-flow"]},
    )
    result.flow_participants["auth"] = {
        "x-flow": [{"path": "b.ts"}, {"path": "c.ts"}],
    }
    files = _candidate_files_for_flow("x-flow", "auth", result)
    assert files == ["b.ts", "c.ts"]


# ── Sprint 13 Day 1b: smarter candidate ranker ───────────────────────


def test_candidate_ranker_prefers_handler_paths_over_random():
    """When token overlap fails, fall back to handler-density not alphabetical."""
    result = DeepScanResult(
        features={
            "billing": [
                "billing/aaa-config.json",
                "billing/bbb-icons.svg",
                "billing/zzz-utils.ts",
                "billing/api/checkout/route.ts",
                "billing/services/stripe.ts",
            ],
        },
        flows={"billing": ["unrelated-flow"]},
    )
    files = _candidate_files_for_flow("unrelated-flow", "billing", result)
    # Top of list should be handler-density files, not aaa-config.json
    assert files[0] in {"billing/api/checkout/route.ts", "billing/services/stripe.ts"}


def test_candidate_ranker_drops_test_and_locale_noise():
    result = DeepScanResult(
        features={
            "auth": [
                "auth/__tests__/login.test.ts",
                "auth/i18n/en.json",
                "auth/login.ts",
            ],
        },
        flows={"auth": ["login-flow"]},
    )
    files = _candidate_files_for_flow("login-flow", "auth", result)
    assert "auth/login.ts" in files
    assert all(".test." not in f for f in files)
    assert all("/i18n/" not in f for f in files)


def test_candidate_ranker_clusters_around_top_dir():
    """Outlier files sharing a token but in different dir get dropped."""
    result = DeepScanResult(
        features={
            "ui": [
                "ui/random/button.ts",
                "ui/auth/login.ts",     # shares 'login' token
                "ui/auth/signup.ts",
                "ui/auth/oauth.ts",
                "ui/random/login-helper.ts",  # shares 'login' but different dir
            ],
        },
        flows={"ui": ["login-flow"]},
    )
    files = _candidate_files_for_flow("login-flow", "ui", result)
    assert "ui/auth/login.ts" in files
    # Cluster around ui/auth/ — random/login-helper.ts dropped or last
    auth_count = sum(1 for f in files if f.startswith("ui/auth/"))
    assert auth_count >= 2


def test_path_is_noise_classifier():
    from faultline.llm.flow_symbols import _path_is_noise
    assert _path_is_noise("a/b/foo.test.ts")
    assert _path_is_noise("a/i18n/en.json")
    assert _path_is_noise("types/user.d.ts")
    assert not _path_is_noise("api/auth/login.ts")


def test_path_handler_score_recognises_api_paths():
    from faultline.llm.flow_symbols import _path_handler_score
    assert _path_handler_score("api/users/route.ts") >= 2
    assert _path_handler_score("services/stripe.ts") >= 1
    assert _path_handler_score("utils/math.ts") == 0
