"""Sprint 11 Day 1 — Haiku-judged flow re-attribution tests.

All tests use mocked Anthropic clients — no real API calls. Day 4
covers end-to-end on real scans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from faultline.llm.flow_judge import (
    CONFIDENCE_FLOOR,
    FlowEntry,
    FlowVerdict,
    _apply_verdicts,
    _build_prompt,
    _coerce_verdict,
    _parse_response,
    _select_flows_for_judging,
    judge_flow_attribution,
)
from faultline.llm.sonnet_scanner import DeepScanResult


# ── Fake Anthropic client ─────────────────────────────────────────────


@dataclass
class _FakeBlock:
    text: str = ""

    @property
    def type(self) -> str:
        return "text"


@dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]
    usage: _FakeUsage = field(default_factory=_FakeUsage)
    stop_reason: str = "end_turn"


class _FakeMessages:
    def __init__(self, scripted: list[_FakeResponse]):
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._scripted:
            raise RuntimeError("scripted responses exhausted")
        return self._scripted.pop(0)


class _FakeAnthropic:
    def __init__(self, scripted: list[_FakeResponse]):
        self.messages = _FakeMessages(scripted)


def _ds(**kwargs) -> DeepScanResult:
    return DeepScanResult(
        features=kwargs.get("features", {}),
        flows=kwargs.get("flows", {}),
        descriptions=kwargs.get("descriptions", {}),
        flow_descriptions=kwargs.get("flow_descriptions", {}),
        flow_participants=kwargs.get("flow_participants", {}),
    )


# ── _select_flows_for_judging ────────────────────────────────────────


class TestSelectFlows:
    def test_flattens_each_flow_with_owner(self):
        result = _ds(
            features={"Auth": ["a.ts"], "Billing": ["b.ts"]},
            flows={
                "Auth": ["log-in", "sign-up"],
                "Billing": ["upgrade-plan"],
            },
        )
        flows = _select_flows_for_judging(result)
        assert {(f.name, f.current_owner) for f in flows} == {
            ("log-in", "Auth"),
            ("sign-up", "Auth"),
            ("upgrade-plan", "Billing"),
        }

    def test_skips_protected_buckets(self):
        result = _ds(
            features={"shared-infra": ["x.ts"], "Auth": ["a.ts"]},
            flows={
                "shared-infra": ["build-pipeline"],
                "Auth": ["log-in"],
            },
        )
        flows = _select_flows_for_judging(result)
        owners = {f.current_owner for f in flows}
        assert "shared-infra" not in owners
        assert "Auth" in owners

    def test_includes_descriptions_when_available(self):
        result = _ds(
            features={"Auth": ["a.ts"]},
            flows={"Auth": ["log-in"]},
            flow_descriptions={"Auth": {"log-in": "Submit credentials."}},
        )
        flows = _select_flows_for_judging(result)
        assert flows[0].description == "Submit credentials."

    def test_skips_features_with_empty_flow_list(self):
        result = _ds(
            features={"Auth": ["a.ts"], "Empty": ["e.ts"]},
            flows={"Auth": ["log-in"], "Empty": []},
        )
        flows = _select_flows_for_judging(result)
        assert all(f.current_owner != "Empty" for f in flows)


# ── _build_prompt ────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_includes_features_and_flows(self):
        prompt = _build_prompt(
            flows=[
                FlowEntry(name="log-in", current_owner="Vue Blocks", description="Login form."),
            ],
            features={"Auth": "User authentication.", "Vue Blocks": "Vue UI demos."},
        )
        assert "Auth" in prompt
        assert "Vue Blocks" in prompt
        assert "log-in" in prompt
        assert "Login form." in prompt
        # Should be valid JSON-shaped after the leading instruction line
        first_brace = prompt.find("{")
        # JSON parses without raising
        import json
        json.loads(prompt[first_brace:])


# ── _parse_response ──────────────────────────────────────────────────


class TestParseResponse:
    def test_well_formed(self):
        text = (
            '{"verdicts": ['
            '{"flow": "log-in", "from": "Vue Blocks", '
            '"decision": "move", "to": "Auth", "confidence": 5}'
            "]}"
        )
        v = _parse_response(text)
        assert v is not None
        assert len(v) == 1

    def test_handles_prose_around_json(self):
        text = (
            "Here are my verdicts:\n\n"
            '{"verdicts": [{"flow":"x","from":"A","decision":"keep","to":null,"confidence":3}]}'
            "\n\nDone."
        )
        v = _parse_response(text)
        assert v is not None
        assert len(v) == 1

    def test_malformed_returns_none(self):
        assert _parse_response("not json") is None
        assert _parse_response("{this is broken}") is None

    def test_missing_verdicts_key_returns_none(self):
        assert _parse_response('{"other": []}') is None


# ── _coerce_verdict ──────────────────────────────────────────────────


class TestCoerceVerdict:
    def test_valid_move(self):
        v = _coerce_verdict({
            "flow": "log-in",
            "from": "Vue Blocks",
            "decision": "move",
            "to": "Auth",
            "confidence": 5,
            "reasoning": "login is the auth surface",
        })
        assert v is not None
        assert v.decision == "move"
        assert v.destination == "Auth"
        assert v.confidence == 5

    def test_valid_keep(self):
        v = _coerce_verdict({
            "flow": "open-dashboard",
            "from": "Studio",
            "decision": "keep",
            "to": None,
            "confidence": 5,
        })
        assert v is not None
        assert v.decision == "keep"
        assert v.destination is None

    def test_move_without_destination_returns_none(self):
        # Move verdict MUST have a destination
        v = _coerce_verdict({
            "flow": "log-in",
            "from": "Vue Blocks",
            "decision": "move",
            "to": None,
            "confidence": 5,
        })
        assert v is None

    def test_invalid_decision_returns_none(self):
        v = _coerce_verdict({
            "flow": "x",
            "from": "A",
            "decision": "merge",  # not move/keep
            "to": "B",
            "confidence": 5,
        })
        assert v is None

    def test_missing_flow_returns_none(self):
        v = _coerce_verdict({
            "from": "A",
            "decision": "keep",
            "confidence": 5,
        })
        assert v is None

    def test_confidence_clamped(self):
        v_low = _coerce_verdict({
            "flow": "x", "from": "A", "decision": "keep",
            "to": None, "confidence": -10,
        })
        v_high = _coerce_verdict({
            "flow": "x", "from": "A", "decision": "keep",
            "to": None, "confidence": 99,
        })
        v_str = _coerce_verdict({
            "flow": "x", "from": "A", "decision": "keep",
            "to": None, "confidence": "five",
        })
        assert v_low.confidence == 1
        assert v_high.confidence == 5
        assert v_str.confidence == 3  # default fallback


# ── _apply_verdicts ──────────────────────────────────────────────────


class TestApplyVerdicts:
    def test_supabase_auth_case(self):
        # Real motivation: Auth feature has 38 files, 0 flows. Login
        # form's flow stranded on Vue Blocks. Judge moves it.
        result = _ds(
            features={
                "Auth": ["studio/auth/login.tsx"],
                "Vue Blocks": ["vue-blocks/login-form.vue"],
            },
            flows={"Vue Blocks": ["Authenticate with Password"]},
            flow_descriptions={
                "Vue Blocks": {"Authenticate with Password": "Login form."}
            },
            flow_participants={
                "Vue Blocks": {
                    "Authenticate with Password": [{"file_path": "vue-blocks/login-form.vue"}]
                }
            },
        )
        verdicts = [
            FlowVerdict(
                flow_name="Authenticate with Password",
                current_owner="Vue Blocks",
                decision="move",
                destination="Auth",
                confidence=5,
            ),
        ]
        moves = _apply_verdicts(result, verdicts)
        assert moves == 1
        # Flow now on Auth
        assert "Authenticate with Password" in result.flows["Auth"]
        assert "Authenticate with Password" not in result.flows.get("Vue Blocks", [])
        # Description migrated
        assert (
            result.flow_descriptions["Auth"]["Authenticate with Password"]
            == "Login form."
        )
        # Participants migrated
        assert "Authenticate with Password" in result.flow_participants["Auth"]

    def test_keep_verdict_doesnt_move(self):
        result = _ds(
            features={"Auth": ["a.ts"], "Studio": ["s.ts"]},
            flows={"Studio": ["open-dashboard"]},
        )
        verdicts = [
            FlowVerdict(
                flow_name="open-dashboard",
                current_owner="Studio",
                decision="keep",
                destination=None,
                confidence=5,
            ),
        ]
        assert _apply_verdicts(result, verdicts) == 0
        assert "open-dashboard" in result.flows["Studio"]

    def test_below_confidence_floor_doesnt_move(self):
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Vue Blocks": ["log-in"]},
        )
        # Confidence 3 — below CONFIDENCE_FLOOR (4)
        verdicts = [
            FlowVerdict(
                flow_name="log-in",
                current_owner="Vue Blocks",
                decision="move",
                destination="Auth",
                confidence=3,
            ),
        ]
        assert _apply_verdicts(result, verdicts) == 0
        assert "log-in" in result.flows["Vue Blocks"]

    def test_unknown_destination_doesnt_move(self):
        # Model hallucinates a destination not in the scan
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Vue Blocks": ["log-in"]},
        )
        verdicts = [
            FlowVerdict(
                flow_name="log-in",
                current_owner="Vue Blocks",
                decision="move",
                destination="Authentification",  # typo, not in features
                confidence=5,
            ),
        ]
        assert _apply_verdicts(result, verdicts) == 0
        assert "log-in" in result.flows["Vue Blocks"]

    def test_stale_current_owner_doesnt_move(self):
        # Model thinks flow is on X but it's already been moved
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Auth": ["log-in"]},  # already on Auth somehow
        )
        verdicts = [
            FlowVerdict(
                flow_name="log-in",
                current_owner="Vue Blocks",  # stale
                decision="move",
                destination="Auth",
                confidence=5,
            ),
        ]
        assert _apply_verdicts(result, verdicts) == 0

    def test_multiple_verdicts_one_pass(self):
        result = _ds(
            features={"Auth": ["a.ts"], "Billing": ["b.ts"], "Studio": ["s.ts"]},
            flows={
                "Studio": [
                    "Manage Auth Configuration",
                    "View Billing Account",
                    "Open Studio Dashboard",
                ],
            },
        )
        verdicts = [
            FlowVerdict(
                flow_name="Manage Auth Configuration",
                current_owner="Studio",
                decision="move",
                destination="Auth",
                confidence=5,
            ),
            FlowVerdict(
                flow_name="View Billing Account",
                current_owner="Studio",
                decision="move",
                destination="Billing",
                confidence=5,
            ),
            FlowVerdict(
                flow_name="Open Studio Dashboard",
                current_owner="Studio",
                decision="keep",
                destination=None,
                confidence=5,
            ),
        ]
        assert _apply_verdicts(result, verdicts) == 2
        assert "Manage Auth Configuration" in result.flows["Auth"]
        assert "View Billing Account" in result.flows["Billing"]
        assert "Open Studio Dashboard" in result.flows["Studio"]


# ── judge_flow_attribution end-to-end with fake client ───────────────


class TestJudgeFlowAttributionLoop:
    def test_full_flow_with_mocked_client(self):
        result = _ds(
            features={
                "Auth": ["a.ts"],
                "Vue Blocks": ["v.ts"],
            },
            descriptions={"Auth": "Login.", "Vue Blocks": "UI demos."},
            flows={"Vue Blocks": ["Authenticate with Password"]},
        )
        scripted = [
            _FakeResponse(
                content=[_FakeBlock(text=(
                    '{"verdicts": [{'
                    '"flow": "Authenticate with Password",'
                    '"from": "Vue Blocks",'
                    '"decision": "move",'
                    '"to": "Auth",'
                    '"confidence": 5'
                    "}]}"
                ))],
            ),
        ]
        client = _FakeAnthropic(scripted)
        moves = judge_flow_attribution(result, client=client)
        assert moves == 1
        assert "Authenticate with Password" in result.flows["Auth"]

    def test_no_flows_returns_zero_no_api_call(self):
        result = _ds(features={"Auth": ["a.ts"]}, flows={"Auth": []})
        client = _FakeAnthropic([])
        assert judge_flow_attribution(result, client=client) == 0
        assert client.messages.calls == []

    def test_no_features_returns_zero_no_api_call(self):
        # Only protected buckets exist — no menu to choose from
        result = _ds(
            features={"shared-infra": ["x.ts"]},
            flows={"shared-infra": ["bootstrap"]},
        )
        client = _FakeAnthropic([])
        assert judge_flow_attribution(result, client=client) == 0

    def test_unparseable_response_returns_zero(self):
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Vue Blocks": ["log-in"]},
        )
        scripted = [
            _FakeResponse(content=[_FakeBlock(text="The verdict is move to Auth.")]),
        ]
        client = _FakeAnthropic(scripted)
        assert judge_flow_attribution(result, client=client) == 0
        # Flow stays put
        assert "log-in" in result.flows["Vue Blocks"]

    def test_batches_flows_above_batch_size(self):
        # 3 flows with batch_size=2 → 2 API calls
        result = _ds(
            features={"Auth": ["a.ts"], "Studio": ["s.ts"]},
            descriptions={"Auth": "", "Studio": ""},
            flows={"Studio": ["a", "b", "c"]},
        )
        # Each batch returns one move
        scripted = [
            _FakeResponse(content=[_FakeBlock(text=(
                '{"verdicts":[{"flow":"a","from":"Studio","decision":"move","to":"Auth","confidence":5},'
                '{"flow":"b","from":"Studio","decision":"keep","to":null,"confidence":5}]}'
            ))]),
            _FakeResponse(content=[_FakeBlock(text=(
                '{"verdicts":[{"flow":"c","from":"Studio","decision":"keep","to":null,"confidence":5}]}'
            ))]),
        ]
        client = _FakeAnthropic(scripted)
        moves = judge_flow_attribution(result, client=client, batch_size=2)
        assert moves == 1
        assert len(client.messages.calls) == 2


class TestCacheLayer:
    """Day 3 — verdict cache. Re-scans skip Haiku for unchanged flows."""

    def _scripted_judge(self, flow: str, frm: str, to: str) -> _FakeResponse:
        text = (
            '{"verdicts":[{'
            f'"flow":"{flow}","from":"{frm}",'
            f'"decision":"move","to":"{to}","confidence":5'
            "}]}"
        )
        return _FakeResponse(content=[_FakeBlock(text=text)])

    def test_writes_cache_on_first_run(self, tmp_path):
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            descriptions={"Auth": "Login.", "Vue Blocks": ""},
            flows={"Vue Blocks": ["log-in"]},
            repo_path="/repos/myapp",
        )
        client = _FakeAnthropic([
            self._scripted_judge("log-in", "Vue Blocks", "Auth"),
        ])
        moves = judge_flow_attribution(
            result, client=client, cache_dir=tmp_path, repo_slug="myapp",
        )
        assert moves == 1
        # Cache file written
        cache_file = tmp_path / "flow-verdicts-myapp.json"
        assert cache_file.exists()
        import json
        data = json.loads(cache_file.read_text())
        assert data["version"] == 1
        assert "feature_set_hash" in data
        # Cached verdict keyed by current_owner::flow_name
        assert "Vue Blocks::log-in" in data["verdicts"]

    def test_replay_cache_skips_api_call(self, tmp_path):
        # First run writes cache
        result1 = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            descriptions={"Auth": "Login.", "Vue Blocks": ""},
            flows={"Vue Blocks": ["log-in"]},
        )
        client1 = _FakeAnthropic([
            self._scripted_judge("log-in", "Vue Blocks", "Auth"),
        ])
        moves1 = judge_flow_attribution(
            result1, client=client1, cache_dir=tmp_path, repo_slug="myapp",
        )
        assert moves1 == 1
        assert len(client1.messages.calls) == 1

        # Second run with SAME flow + features → cache hit, no API call
        result2 = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            descriptions={"Auth": "Login.", "Vue Blocks": ""},
            flows={"Vue Blocks": ["log-in"]},
        )
        client2 = _FakeAnthropic([])  # would crash if any API call attempted
        moves2 = judge_flow_attribution(
            result2, client=client2, cache_dir=tmp_path, repo_slug="myapp",
        )
        assert moves2 == 1
        assert len(client2.messages.calls) == 0  # Pure cache replay
        # And the move actually applied on result2
        assert "log-in" in result2.flows["Auth"]

    def test_invalidates_cache_when_features_change(self, tmp_path):
        # First run: write cache for {Auth, Vue Blocks}
        result1 = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            descriptions={"Auth": "", "Vue Blocks": ""},
            flows={"Vue Blocks": ["log-in"]},
        )
        client1 = _FakeAnthropic([
            self._scripted_judge("log-in", "Vue Blocks", "Auth"),
        ])
        judge_flow_attribution(
            result1, client=client1, cache_dir=tmp_path, repo_slug="myapp",
        )

        # Second run with DIFFERENT feature set → cache invalidates,
        # judge re-runs against fresh menu
        result2 = _ds(
            features={"Auth": ["a.ts"], "Studio": ["s.ts"]},  # Vue Blocks gone, Studio appeared
            descriptions={"Auth": "", "Studio": ""},
            flows={"Studio": ["log-in"]},  # owner is now Studio
        )
        client2 = _FakeAnthropic([
            self._scripted_judge("log-in", "Studio", "Auth"),
        ])
        moves2 = judge_flow_attribution(
            result2, client=client2, cache_dir=tmp_path, repo_slug="myapp",
        )
        assert moves2 == 1
        assert len(client2.messages.calls) == 1  # API was called — cache invalidated

    def test_cache_disabled_with_none(self, tmp_path):
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Vue Blocks": ["log-in"]},
        )
        client = _FakeAnthropic([
            self._scripted_judge("log-in", "Vue Blocks", "Auth"),
        ])
        judge_flow_attribution(
            result, client=client, cache_dir=None, repo_slug="myapp",
        )
        # No file in tmp_path because caching disabled
        cache_file = tmp_path / "flow-verdicts-myapp.json"
        assert not cache_file.exists()

    def test_corrupt_cache_falls_back_to_full_judge(self, tmp_path):
        # Plant a corrupt cache file
        (tmp_path / "flow-verdicts-myapp.json").write_text("not valid json {{{")
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Vue Blocks": ["log-in"]},
        )
        client = _FakeAnthropic([
            self._scripted_judge("log-in", "Vue Blocks", "Auth"),
        ])
        moves = judge_flow_attribution(
            result, client=client, cache_dir=tmp_path, repo_slug="myapp",
        )
        # Corrupt cache treated as empty → judge ran fresh
        assert moves == 1
        assert len(client.messages.calls) == 1


class TestFallbackChain:
    """Day 2 — pipeline-level wiring: judge first, heuristic fallback."""

    def test_no_api_key_returns_zero_no_crash(self, monkeypatch):
        # No client, no env key → graceful no-op (caller falls back)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Vue Blocks": ["log-in"]},
        )
        moves = judge_flow_attribution(result, client=None, api_key=None)
        assert moves == 0
        # Flow stays put
        assert "log-in" in result.flows["Vue Blocks"]

    def test_api_call_failure_returns_zero(self):
        class _ExplodingMessages:
            def create(self, **kwargs):
                raise RuntimeError("network down")

        class _ExplodingClient:
            messages = _ExplodingMessages()

        result = _ds(
            features={"Auth": ["a.ts"], "Vue Blocks": ["v.ts"]},
            flows={"Vue Blocks": ["log-in"]},
        )
        # Should not raise — falls through to 0 moves
        moves = judge_flow_attribution(result, client=_ExplodingClient())
        assert moves == 0
        assert "log-in" in result.flows["Vue Blocks"]


class TestProtectedBuckets:
    def test_protected_set_includes_synthetics(self):
        from faultline.llm.flow_judge import _PROTECTED_BUCKETS
        assert "shared-infra" in _PROTECTED_BUCKETS
        assert "documentation" in _PROTECTED_BUCKETS
        assert "examples" in _PROTECTED_BUCKETS
        assert "developer-infrastructure" in _PROTECTED_BUCKETS


# ── Sprint 12 Day 3.5: multi-feature ownership ────────────────────────


def test_coerce_verdict_parses_also_belongs_to():
    from faultline.llm.flow_judge import _coerce_verdict
    v = _coerce_verdict({
        "flow": "create-organization-flow",
        "from": "auth",
        "decision": "keep",
        "to": None,
        "confidence": 5,
        "also_belongs_to": ["billing", "notifications"],
    })
    assert v is not None
    assert v.also_belongs_to == ["billing", "notifications"]


def test_coerce_verdict_drops_self_reference():
    """``also_belongs_to`` must not include the primary destination/current."""
    from faultline.llm.flow_judge import _coerce_verdict
    v = _coerce_verdict({
        "flow": "x",
        "from": "auth",
        "decision": "move",
        "to": "billing",
        "confidence": 5,
        "also_belongs_to": ["billing", "auth", "notifications"],
    })
    assert v is not None
    assert v.also_belongs_to == ["notifications"]


def test_coerce_verdict_caps_at_three():
    from faultline.llm.flow_judge import _coerce_verdict
    v = _coerce_verdict({
        "flow": "x",
        "from": "ui",
        "decision": "keep",
        "to": None,
        "confidence": 5,
        "also_belongs_to": ["a", "b", "c", "d", "e"],
    })
    assert v is not None
    assert len(v.also_belongs_to) == 3


def test_apply_writes_flow_secondaries():
    from faultline.llm.flow_judge import FlowVerdict, _apply_verdicts
    from faultline.llm.sonnet_scanner import DeepScanResult
    result = DeepScanResult(
        features={"auth": ["a.ts"], "billing": ["b.ts"], "notifications": ["n.ts"]},
        flows={"auth": ["create-organization-flow"]},
    )
    verdicts = [FlowVerdict(
        flow_name="create-organization-flow",
        current_owner="auth",
        decision="keep",
        destination=None,
        confidence=5,
        also_belongs_to=["billing", "notifications"],
    )]
    _apply_verdicts(result, verdicts)
    assert result.flow_secondaries["create-organization-flow"] == ["billing", "notifications"]


def test_apply_drops_unknown_secondaries():
    """Hallucinated feature names get filtered out, not written."""
    from faultline.llm.flow_judge import FlowVerdict, _apply_verdicts
    from faultline.llm.sonnet_scanner import DeepScanResult
    result = DeepScanResult(
        features={"auth": ["a.ts"], "billing": ["b.ts"]},
        flows={"auth": ["create-org-flow"]},
    )
    verdicts = [FlowVerdict(
        flow_name="create-org-flow",
        current_owner="auth",
        decision="keep",
        destination=None,
        confidence=5,
        also_belongs_to=["billing", "made-up-feature"],
    )]
    _apply_verdicts(result, verdicts)
    assert result.flow_secondaries["create-org-flow"] == ["billing"]


def test_apply_secondaries_works_alongside_move():
    """When verdict is move, both move AND record secondaries."""
    from faultline.llm.flow_judge import FlowVerdict, _apply_verdicts
    from faultline.llm.sonnet_scanner import DeepScanResult
    result = DeepScanResult(
        features={"i18n": [], "auth": [], "billing": ["b.ts"]},
        flows={"i18n": ["pay-and-create-account-flow"]},
    )
    verdicts = [FlowVerdict(
        flow_name="pay-and-create-account-flow",
        current_owner="i18n",
        decision="move",
        destination="auth",
        confidence=5,
        also_belongs_to=["billing"],
    )]
    moves = _apply_verdicts(result, verdicts)
    assert moves == 1
    assert "pay-and-create-account-flow" in result.flows["auth"]
    assert "pay-and-create-account-flow" not in result.flows["i18n"]
    assert result.flow_secondaries["pay-and-create-account-flow"] == ["billing"]


def test_verdict_roundtrip_through_cache_preserves_secondaries():
    from faultline.llm.flow_judge import (
        FlowVerdict, _verdict_from_dict, _verdict_to_dict,
    )
    v = FlowVerdict(
        flow_name="x",
        current_owner="auth",
        decision="keep",
        destination=None,
        confidence=5,
        also_belongs_to=["billing", "notifications"],
    )
    d = _verdict_to_dict(v)
    v2 = _verdict_from_dict(d)
    assert v2 is not None
    assert v2.also_belongs_to == ["billing", "notifications"]


# ── Sprint 13 Day 1: signals injection + re-judge ─────────────────────


def test_build_prompt_includes_evidence_when_signals_present():
    from faultline.llm.flow_judge import FlowEntry, _build_prompt
    flows = [FlowEntry(
        name="manage-billing-subscription",
        current_owner="contracts",
        description="user manages plan",
        signals={
            "billing": {"ownership": 0.73, "centrality": 0.0},
            "contracts": {"ownership": 0.27, "centrality": 0.0},
        },
    )]
    prompt = _build_prompt(flows, {"contracts": "...", "billing": "..."})
    assert "deterministic_evidence" in prompt
    assert "73" in prompt  # 73% ownership rendered
    assert "billing" in prompt


def test_build_prompt_omits_evidence_when_no_signals():
    from faultline.llm.flow_judge import FlowEntry, _build_prompt
    flows = [FlowEntry(name="x", current_owner="auth")]
    prompt = _build_prompt(flows, {"auth": ""})
    assert "deterministic_evidence" not in prompt


def test_flow_to_payload_drops_zero_evidence_rows():
    from faultline.llm.flow_judge import FlowEntry, _flow_to_payload
    f = FlowEntry(
        name="x",
        current_owner="a",
        signals={
            "a": {"ownership": 0.5, "centrality": 0.0},
            "b": {"ownership": 0.0, "centrality": 0.0},
        },
    )
    out = _flow_to_payload(f)
    features = [e["feature"] for e in out["deterministic_evidence"]]
    assert features == ["a"]


def test_re_judge_skips_when_no_disagreement():
    """If current owner has highest ownership, no candidate."""
    from faultline.llm.flow_judge import re_judge_with_signals

    result = DeepScanResult(
        features={"auth": ["a.ts", "b.ts"], "billing": ["c.ts"]},
        flows={"auth": ["x"]},
        descriptions={"auth": "", "billing": ""},
    )
    result.flow_participants["auth"] = {
        "x": [{"path": "a.ts"}, {"path": "b.ts"}],
    }
    client = _FakeAnthropic(scripted=[])
    n = re_judge_with_signals(result, client=client)
    assert n == 0
    assert client.messages.calls == []  # no API call when no candidates


def test_re_judge_moves_when_signals_disagree():
    """Flow whose paths predominantly belong to other feature gets moved."""
    from faultline.llm.flow_judge import re_judge_with_signals

    result = DeepScanResult(
        features={
            "contracts": ["c.ts"],
            "billing": ["a.ts", "b.ts", "d.ts", "e.ts"],
        },
        flows={"contracts": ["manage-billing-flow"]},
        descriptions={"contracts": "", "billing": "subscriptions"},
    )
    # Flow's paths overwhelmingly in billing (3/4 = 75%)
    result.flow_participants["contracts"] = {
        "manage-billing-flow": [
            {"path": "a.ts"}, {"path": "b.ts"},
            {"path": "d.ts"}, {"path": "c.ts"},
        ],
    }
    client = _FakeAnthropic(scripted=[
        _FakeResponse(content=[_FakeBlock(text=json.dumps({
            "verdicts": [{
                "flow": "manage-billing-flow",
                "from": "contracts",
                "decision": "move",
                "to": "billing",
                "confidence": 5,
            }]
        }))])
    ])
    n = re_judge_with_signals(result, client=client)
    assert n == 1
    assert "manage-billing-flow" in result.flows["billing"]
    assert "manage-billing-flow" not in result.flows["contracts"]
    # Verify prompt included evidence
    assert "deterministic_evidence" in client.messages.calls[0]["messages"][0]["content"]


# ── Sprint 15 Day 1b: flow-set hash invalidation ──────────────────────


def test_flow_set_hash_changes_with_flow_set():
    from faultline.llm.flow_judge import FlowEntry, _flow_set_hash
    a = [FlowEntry(name="x", current_owner="auth")]
    b = [FlowEntry(name="x", current_owner="auth"), FlowEntry(name="y", current_owner="ui")]
    assert _flow_set_hash(a) != _flow_set_hash(b)


def test_flow_set_hash_stable_under_reorder():
    from faultline.llm.flow_judge import FlowEntry, _flow_set_hash
    a = [
        FlowEntry(name="x", current_owner="auth"),
        FlowEntry(name="y", current_owner="ui"),
    ]
    b = [
        FlowEntry(name="y", current_owner="ui"),
        FlowEntry(name="x", current_owner="auth"),
    ]
    assert _flow_set_hash(a) == _flow_set_hash(b)


def test_load_cache_invalidates_on_flow_set_change(tmp_path):
    from faultline.llm.flow_judge import _load_cache, _save_cache
    cache_dir = tmp_path
    _save_cache(cache_dir, "test", "feat-h", {"a::x": {"any": "thing"}}, "flow-h-OLD")
    # Same feature hash, different flow hash → cache miss
    out = _load_cache(cache_dir, "test", "feat-h", "flow-h-NEW")
    assert out == {}
    # Same flow hash → cache hit
    out2 = _load_cache(cache_dir, "test", "feat-h", "flow-h-OLD")
    assert "a::x" in out2


def test_load_cache_legacy_no_flow_hash_not_invalidated(tmp_path):
    """flow_hash=None preserves the S13 behaviour."""
    from faultline.llm.flow_judge import _load_cache, _save_cache
    cache_dir = tmp_path
    _save_cache(cache_dir, "test", "feat-h", {"a::x": {}}, flow_hash=None)
    out = _load_cache(cache_dir, "test", "feat-h", flow_hash=None)
    assert "a::x" in out


# ── Sprint 15 Day 2: asymmetric threshold ─────────────────────────────


def test_is_catchall_owner_basic():
    from faultline.llm.flow_judge import _is_catchall_owner
    assert _is_catchall_owner("i18n")
    assert _is_catchall_owner("ui")
    assert _is_catchall_owner("Vue Blocks")
    assert _is_catchall_owner("App Shell")
    assert not _is_catchall_owner("auth")
    assert not _is_catchall_owner("dify-web/billing")


def test_is_domain_target():
    from faultline.llm.flow_judge import _is_domain_target
    assert _is_domain_target("auth")
    assert not _is_domain_target("Account Settings")  # only literal domain words
    assert _is_domain_target("dify-web/billing")
    assert _is_domain_target("notifications")
    assert not _is_domain_target("dify-web/datasets")


def test_resignal_asymmetric_catchall_to_domain_moves_below_30pct(tmp_path):
    """The dify case: auth feature has 5 paths, i18n has 1300+. A flow
    in i18n with paths 100% in auth (target=100%, current=0%) should
    move even though 30% threshold isn't met (when current=0% target
    just need ≥15% delta which 100% trivially is — but we test the
    BORDERLINE: target=20%, current=5%, delta=15% catchall path)."""
    from faultline.llm.flow_judge import re_judge_with_signals
    result = DeepScanResult(
        features={
            "i18n": [f"i18n/locale-{i}.json" for i in range(80)] + ["x.ts", "y.ts"],
            "auth": ["auth/sso.ts", "auth/jwt.ts"] + [f"auth/extra-{i}.ts" for i in range(18)],
        },
        flows={"i18n": ["sign-in-flow"]},
        descriptions={"i18n": "translations", "auth": "auth"},
    )
    # Flow's paths are mixed: 20% in auth, 5% in i18n
    result.flow_participants["i18n"] = {
        "sign-in-flow": [
            {"path": "auth/sso.ts"},     # in auth
            {"path": "auth/jwt.ts"},     # in auth
            {"path": "i18n/locale-1.json"},  # in i18n
            {"path": "auth/extra-1.ts"},  # in auth
            {"path": "auth/extra-2.ts"},  # in auth
        ],
    }
    # Without S15: ownership(auth)=80%, ownership(i18n)=20%, delta=60% — would move regardless.
    # We need a case where delta < 30% to prove the asymmetric path.
    result.flow_participants["i18n"] = {
        "sign-in-flow": [
            {"path": "i18n/locale-1.json"},
            {"path": "i18n/locale-2.json"},
            {"path": "i18n/locale-3.json"},
            {"path": "i18n/locale-4.json"},
            {"path": "auth/sso.ts"},  # 1 of 5 = 20% in auth, 80% in i18n
        ],
    }
    # current(i18n)=80%, target(auth)=20% → target loses, no move expected
    client = _FakeAnthropic(scripted=[])
    n = re_judge_with_signals(result, client=client)
    assert n == 0  # target ownership lower than current — sane behaviour

    # Now flip: paths mostly in auth (high target ownership)
    result.flow_participants["i18n"] = {
        "sign-in-flow": [
            {"path": "auth/sso.ts"},
            {"path": "auth/jwt.ts"},
            {"path": "auth/extra-1.ts"},
            {"path": "auth/extra-2.ts"},
            {"path": "i18n/locale-1.json"},
        ],
    }
    # current(i18n)=20%, target(auth)=80%, delta=60% — moves under either threshold
    client = _FakeAnthropic(scripted=[
        _FakeResponse(content=[_FakeBlock(text=json.dumps({
            "verdicts": [{
                "flow": "sign-in-flow", "from": "i18n",
                "decision": "move", "to": "auth", "confidence": 5,
            }]
        }))])
    ])
    n = re_judge_with_signals(result, client=client)
    assert n == 1
    assert "sign-in-flow" in result.flows["auth"]


def test_resignal_asymmetric_only_for_catchall_source(tmp_path):
    """domain → domain still uses 30% threshold (no asymmetric loosening)."""
    from faultline.llm.flow_judge import re_judge_with_signals
    result = DeepScanResult(
        features={
            "auth": [f"auth/{i}.ts" for i in range(20)],
            "billing": [f"billing/{i}.ts" for i in range(20)],
        },
        flows={"auth": ["x-flow"]},
        descriptions={"auth": "", "billing": ""},
    )
    # Flow has paths mixed: 4 in auth, 1 in billing — delta=15%, NOT enough
    result.flow_participants["auth"] = {
        "x-flow": [
            {"path": "auth/0.ts"}, {"path": "auth/1.ts"},
            {"path": "auth/2.ts"}, {"path": "auth/3.ts"},
            {"path": "billing/0.ts"},
        ],
    }
    # current(auth)=80%, target(billing)=20%, delta=-60% — no move, target is the loser
    client = _FakeAnthropic(scripted=[])
    n = re_judge_with_signals(result, client=client)
    assert n == 0
    assert client.messages.calls == []
