"""Sprint 9 Day 2 — agentic classifier tests with a fake Anthropic client.

The agent exercises the same tool_use_scan loop pattern as Sprint 1.
We test it the same way: stub out the Anthropic client with a script
of pre-baked responses, then verify the agent dispatches tools
correctly, parses the final JSON, and shapes the output to the
Sprint 8 FeatureClassification dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from faultline.llm.aggregator_agent import (
    _build_user_prompt,
    _coerce_classification,
    _parse_classifications,
    _select_features,
    agentic_classify_features,
)
from faultline.llm.aggregator_detector import FeatureClassification
from faultline.llm.sonnet_scanner import DeepScanResult


# ── Fake Anthropic client ─────────────────────────────────────────────


@dataclass
class _FakeBlock:
    """A single content block. ``type`` is 'text' or 'tool_use'."""
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]
    stop_reason: str
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _FakeAnthropicMessages:
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
        self.messages = _FakeAnthropicMessages(scripted)


# ── Pure-function tests ───────────────────────────────────────────────


class TestSelectFeatures:
    def test_excludes_protected_buckets(self):
        result = DeepScanResult(features={
            "Auth": ["a.ts"],
            "shared-infra": ["x.ts"],
            "documentation": ["docs.md"],
            "examples": ["ex.ts"],
            "developer-infrastructure": ["dev.ts"],
        })
        assert _select_features(result) == ["Auth"]

    def test_returns_empty_when_only_protected(self):
        result = DeepScanResult(features={
            "shared-infra": ["x.ts"],
            "documentation": ["d.md"],
        })
        assert _select_features(result) == []


class TestBuildUserPrompt:
    def test_lists_each_feature(self):
        prompt = _build_user_prompt(["Auth", "Billing", "Workflow"])
        assert "Auth" in prompt
        assert "Billing" in prompt
        assert "Workflow" in prompt
        assert "3 total" in prompt
        assert "feature_summary" in prompt


class TestParseClassifications:
    def test_extracts_array(self):
        text = (
            '{"classifications": ['
            '{"name": "Auth", "class": "product-feature", "confidence": 5}'
            ']}'
        )
        entries = _parse_classifications(text)
        assert entries is not None
        assert len(entries) == 1
        assert entries[0]["name"] == "Auth"

    def test_handles_prose_around_json(self):
        text = (
            "Done investigating. Here's the JSON:\n\n"
            '{"classifications": ['
            '{"name": "Dto", "class": "shared-aggregator", "confidence": 5}'
            ']}\n\nLet me know if you need more.'
        )
        entries = _parse_classifications(text)
        assert entries is not None
        assert entries[0]["class"] == "shared-aggregator"

    def test_returns_none_on_malformed(self):
        assert _parse_classifications("not json") is None
        assert _parse_classifications("{this is broken}") is None

    def test_returns_none_when_array_missing(self):
        assert _parse_classifications('{"other_field": []}') is None


class TestCoerceClassification:
    def test_minimal_product_feature(self):
        v = _coerce_classification({
            "name": "Auth",
            "class": "product-feature",
            "confidence": 5,
            "reasoning": "Login UI surface.",
        })
        assert v is not None
        assert isinstance(v, FeatureClassification)
        assert v.feature_name == "Auth"
        assert v.classification == "product-feature"
        assert v.consumer_features is None
        assert v.proposed_name is None

    def test_aggregator_with_consumers(self):
        v = _coerce_classification({
            "name": "Dto",
            "class": "shared-aggregator",
            "confidence": 5,
            "consumer_features": ["Auth", "Billing", "Workflow"],
            "reasoning": "Sub-folders match Auth/Billing/Workflow features.",
        })
        assert v.consumer_features == ["Auth", "Billing", "Workflow"]

    def test_invalid_class_returns_none(self):
        assert _coerce_classification({
            "name": "X", "class": "not-a-class", "confidence": 5,
        }) is None


# ── Agent loop integration tests (with mocked client) ────────────────


class TestAgentLoopNoToolUse:
    """When the agent emits a final JSON on the first response (no
    tool calls needed), the loop should parse and return without
    looping further.
    """

    def test_returns_classifications_directly(self, tmp_path: Path):
        result = DeepScanResult(features={
            "Auth": ["src/auth/login.tsx"],
            "Billing": ["src/billing/plan.tsx"],
        })
        scripted = [
            _FakeResponse(
                content=[_FakeBlock(
                    type="text",
                    text=(
                        '{"classifications": ['
                        '{"name":"Auth","class":"product-feature","confidence":5,"reasoning":"login UI."},'
                        '{"name":"Billing","class":"product-feature","confidence":5,"reasoning":"stripe checkout."}'
                        ']}'
                    ),
                )],
                stop_reason="end_turn",
            ),
        ]
        client = _FakeAnthropic(scripted)
        out = agentic_classify_features(
            result,
            repo_root=tmp_path,
            symbol_graph=None,
            client=client,
        )
        assert "Auth" in out
        assert "Billing" in out
        assert out["Auth"].classification == "product-feature"
        assert len(client.messages.calls) == 1


class TestAgentLoopWithToolUse:
    """When the agent needs to investigate, it should dispatch tools
    via the wrapped dispatcher, append tool_result back to the
    message list, and continue until it emits a final JSON."""

    def test_loops_through_tool_call_then_returns(self, tmp_path: Path):
        # Make a real file for read_file_head to read
        (tmp_path / "a.ts").write_text("export const X = 1;\n")
        result = DeepScanResult(features={"Auth": ["a.ts"]})

        scripted = [
            # Round 1: agent calls read_file_head
            _FakeResponse(
                content=[_FakeBlock(
                    type="tool_use",
                    id="t1",
                    name="read_file_head",
                    input={"path": "a.ts", "lines": 5},
                )],
                stop_reason="tool_use",
            ),
            # Round 2: agent emits final JSON
            _FakeResponse(
                content=[_FakeBlock(
                    type="text",
                    text=(
                        '{"classifications":[{"name":"Auth",'
                        '"class":"product-feature","confidence":5,'
                        '"reasoning":"checked file content."}]}'
                    ),
                )],
                stop_reason="end_turn",
            ),
        ]
        client = _FakeAnthropic(scripted)

        captured_calls: list = []

        def on_tool_call(tn, ti, rs):
            captured_calls.append((tn, ti, rs))

        out = agentic_classify_features(
            result,
            repo_root=tmp_path,
            symbol_graph=None,
            client=client,
            on_tool_call=on_tool_call,
        )

        assert len(client.messages.calls) == 2
        assert captured_calls and captured_calls[0][0] == "read_file_head"
        assert "Auth" in out
        assert out["Auth"].classification == "product-feature"

    def test_consumers_of_dispatch_threads_symbol_graph(self, tmp_path: Path):
        from faultline.analyzer.symbol_graph import ImportEdge, SymbolGraph
        graph = SymbolGraph()
        graph.reverse["packages/dto/login.dto.ts"] = [
            ImportEdge(target_file="src/auth/login.tsx", target_symbol="*"),
        ]
        result = DeepScanResult(features={
            "Dto": ["packages/dto/login.dto.ts"],
            "Auth": ["src/auth/login.tsx"],
        })

        scripted = [
            _FakeResponse(
                content=[_FakeBlock(
                    type="tool_use",
                    id="t1",
                    name="consumers_of",
                    input={"path": "packages/dto/login.dto.ts"},
                )],
                stop_reason="tool_use",
            ),
            _FakeResponse(
                content=[_FakeBlock(
                    type="text",
                    text=(
                        '{"classifications":[{"name":"Dto",'
                        '"class":"shared-aggregator","confidence":5,'
                        '"consumer_features":["Auth"],'
                        '"reasoning":"login.dto consumed by Auth."}]}'
                    ),
                )],
                stop_reason="end_turn",
            ),
        ]
        client = _FakeAnthropic(scripted)

        captured: list = []

        def on_tool_call(tn, ti, rs):
            captured.append((tn, ti, rs))

        out = agentic_classify_features(
            result,
            repo_root=tmp_path,
            symbol_graph=graph,
            client=client,
            on_tool_call=on_tool_call,
        )
        # Agent saw the consumer; classification reflects it
        assert "Dto" in out
        assert out["Dto"].classification == "shared-aggregator"
        assert out["Dto"].consumer_features == ["Auth"]
        # Tool actually returned the right consumer (graph injected)
        assert "src/auth/login.tsx" in captured[0][2]


class TestAgentLoopNoFeatures:
    def test_returns_empty_dict_with_no_input(self, tmp_path: Path):
        result = DeepScanResult(features={})
        client = _FakeAnthropic([])
        out = agentic_classify_features(
            result,
            repo_root=tmp_path,
            symbol_graph=None,
            client=client,
        )
        assert out == {}
        # No API call was made — the agent short-circuited
        assert client.messages.calls == []


class TestAgentLoopBudgetExhaustion:
    """When the tool budget is exhausted, the next request should drop
    ``tools=`` from the call and append a wrap-up nudge so the model
    emits the final JSON instead of looping forever."""

    def test_no_more_tools_after_budget(self, tmp_path: Path):
        (tmp_path / "a.ts").write_text("x")
        result = DeepScanResult(features={"Auth": ["a.ts"]})
        scripted = [
            # Round 1: agent calls a tool (uses 1 of 1 budget)
            _FakeResponse(
                content=[_FakeBlock(
                    type="tool_use",
                    id="t1",
                    name="read_file_head",
                    input={"path": "a.ts"},
                )],
                stop_reason="tool_use",
            ),
            # Round 2: agent might want to call another tool, but
            # budget is exhausted (1/1) — it must return final JSON
            # because we drop tools= and add a nudge.
            _FakeResponse(
                content=[_FakeBlock(
                    type="text",
                    text=(
                        '{"classifications":[{"name":"Auth",'
                        '"class":"product-feature","confidence":3,'
                        '"reasoning":"budget exhausted."}]}'
                    ),
                )],
                stop_reason="end_turn",
            ),
        ]
        client = _FakeAnthropic(scripted)
        out = agentic_classify_features(
            result,
            repo_root=tmp_path,
            symbol_graph=None,
            client=client,
            tool_budget=1,
        )
        # Round 2's request should NOT have a tools key
        round_2 = client.messages.calls[1]
        assert "tools" not in round_2
        # And the message list at that point should include a nudge
        # somewhere (NOTE: messages is passed by reference and gets
        # mutated after this call too, so we just check the nudge
        # made it in by the time round 2 was sent).
        round_2_messages = round_2["messages"]
        nudges = [
            m for m in round_2_messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and "exhausted" in m["content"].lower()
        ]
        assert nudges, "expected a budget-exhausted nudge in round 2 messages"
        assert "Auth" in out


class TestAgentLoopParseFailure:
    def test_returns_empty_dict_on_unparseable_final(self, tmp_path: Path):
        result = DeepScanResult(features={"Auth": ["a.ts"]})
        scripted = [
            _FakeResponse(
                content=[_FakeBlock(
                    type="text",
                    text="The features are auth, billing, and so on.",
                )],
                stop_reason="end_turn",
            ),
        ]
        client = _FakeAnthropic(scripted)
        out = agentic_classify_features(
            result,
            repo_root=tmp_path,
            symbol_graph=None,
            client=client,
        )
        # Empty dict — caller leaves the result untouched
        assert out == {}
