"""Tests for ``faultline.llm.tool_use_scan``.

Day 2 of Sprint 1. The Anthropic client is mocked — these tests cover
the message loop, tool dispatch, budget cap, and JSON parsing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from faultline.llm.tool_use_scan import (
    DEFAULT_TOOL_BUDGET,
    _parse_features_json,
    tool_use_scan,
)


# ── Fake Anthropic client ──────────────────────────────────────────────


def _make_response(blocks: list[dict], stop_reason: str = "end_turn"):
    """Build a SimpleNamespace mimicking an Anthropic Message response."""
    objs = []
    for b in blocks:
        if b["type"] == "text":
            objs.append(SimpleNamespace(type="text", text=b["text"]))
        elif b["type"] == "tool_use":
            objs.append(SimpleNamespace(
                type="tool_use",
                id=b["id"],
                name=b["name"],
                input=b.get("input", {}),
            ))
    return SimpleNamespace(content=objs, stop_reason=stop_reason)


class FakeMessages:
    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        import copy
        self.calls.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise RuntimeError("FakeMessages: no scripted response left")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, scripted_responses):
        self.messages = FakeMessages(scripted_responses)


# ── Fixture repo ───────────────────────────────────────────────────────


def _write(root: Path, rel: str, body: str = "x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _write(tmp_path, "pkg/billing.ts", "import Stripe from 'stripe';\n")
    _write(tmp_path, "pkg/auth.ts", "export function login() {}\n")
    return tmp_path


# ── _parse_features_json ───────────────────────────────────────────────


class TestParse:
    def test_raw_json(self):
        assert _parse_features_json('{"features": []}') == {"features": []}

    def test_fenced_block(self):
        text = "Here:\n```json\n{\"features\": [1]}\n```\nthanks"
        assert _parse_features_json(text) == {"features": [1]}

    def test_balanced_braces_in_prose(self):
        text = "intro {\"features\": [{\"name\": \"x\"}]} trailer"
        out = _parse_features_json(text)
        assert out == {"features": [{"name": "x"}]}

    def test_garbage_returns_none(self):
        assert _parse_features_json("not json at all") is None


# ── End-to-end loop behaviour ──────────────────────────────────────────


class TestToolUseScan:
    def test_empty_files_short_circuits(self, repo: Path):
        client = FakeClient([])
        out = tool_use_scan(
            package_name="empty", files=[], repo_root=repo, client=client,
        )
        assert out == {"features": []}
        assert client.messages.calls == []

    def test_single_text_response_parsed(self, repo: Path):
        client = FakeClient([
            _make_response(
                [{"type": "text", "text": '{"features": [{"name": "billing", "paths": ["pkg/billing.ts"], "description": "stripe"}]}'}],
                stop_reason="end_turn",
            ),
        ])
        out = tool_use_scan(
            package_name="pkg",
            files=["pkg/billing.ts", "pkg/auth.ts"],
            repo_root=repo,
            client=client,
        )
        assert out is not None
        assert out["features"][0]["name"] == "billing"
        assert len(client.messages.calls) == 1

    def test_tool_use_then_final(self, repo: Path):
        client = FakeClient([
            _make_response(
                [{"type": "tool_use", "id": "t1", "name": "read_file_head",
                  "input": {"path": "pkg/billing.ts"}}],
                stop_reason="tool_use",
            ),
            _make_response(
                [{"type": "text", "text": '{"features": [{"name": "billing", "paths": ["pkg/billing.ts"], "description": "x"}]}'}],
                stop_reason="end_turn",
            ),
        ])
        observed: list[tuple] = []
        out = tool_use_scan(
            package_name="pkg",
            files=["pkg/billing.ts"],
            repo_root=repo,
            client=client,
            on_tool_call=lambda n, i, r: observed.append((n, i, r)),
        )
        assert out is not None
        assert out["features"][0]["name"] == "billing"
        assert len(client.messages.calls) == 2
        # tool_results were appended to the second call's messages
        second_msgs = client.messages.calls[1]["messages"]
        assert second_msgs[-1]["role"] == "user"
        assert second_msgs[-1]["content"][0]["type"] == "tool_result"
        assert second_msgs[-1]["content"][0]["tool_use_id"] == "t1"
        # on_tool_call was invoked
        assert len(observed) == 1
        assert observed[0][0] == "read_file_head"
        # tool actually ran against the fixture file
        assert "Stripe" in observed[0][2]

    def test_multiple_tool_uses_in_one_turn(self, repo: Path):
        client = FakeClient([
            _make_response(
                [
                    {"type": "tool_use", "id": "a", "name": "read_file_head",
                     "input": {"path": "pkg/billing.ts"}},
                    {"type": "tool_use", "id": "b", "name": "list_directory",
                     "input": {"dirpath": "pkg"}},
                ],
                stop_reason="tool_use",
            ),
            _make_response(
                [{"type": "text", "text": '{"features": []}'}],
                stop_reason="end_turn",
            ),
        ])
        out = tool_use_scan(
            package_name="pkg", files=["pkg/billing.ts"],
            repo_root=repo, client=client,
        )
        assert out == {"features": []}
        # Both tool_results in one user turn
        second_msgs = client.messages.calls[1]["messages"]
        last_user = second_msgs[-1]
        assert last_user["role"] == "user"
        ids = [b["tool_use_id"] for b in last_user["content"]]
        assert ids == ["a", "b"]

    def test_tool_error_marked_is_error(self, repo: Path):
        client = FakeClient([
            _make_response(
                [{"type": "tool_use", "id": "x", "name": "read_file_head",
                  "input": {"path": "../../escape.txt"}}],
                stop_reason="tool_use",
            ),
            _make_response(
                [{"type": "text", "text": '{"features": []}'}],
                stop_reason="end_turn",
            ),
        ])
        tool_use_scan(
            package_name="pkg", files=["pkg/x.ts"],
            repo_root=repo, client=client,
        )
        second_msgs = client.messages.calls[1]["messages"]
        tr = second_msgs[-1]["content"][0]
        assert tr["is_error"] is True
        assert tr["content"].startswith("ERROR:")

    def test_budget_cap_terminates_loop(self, repo: Path):
        # Always-tool-use responses; loop must stop after budget
        always_tool = _make_response(
            [{"type": "tool_use", "id": "loop", "name": "list_directory",
              "input": {"dirpath": "pkg"}}],
            stop_reason="tool_use",
        )
        # After budget exhausted, we send a no-tools call and the model
        # should now return text. Provide that as the terminal response.
        terminal = _make_response(
            [{"type": "text", "text": '{"features": []}'}],
            stop_reason="end_turn",
        )
        # Need: budget tool-use turns, then 1 terminal turn.
        budget = 3
        scripted = [always_tool] * budget + [terminal]
        client = FakeClient(scripted)

        out = tool_use_scan(
            package_name="pkg", files=["pkg/billing.ts"],
            repo_root=repo, client=client, tool_budget=budget,
        )
        assert out == {"features": []}
        # budget+1 calls total
        assert len(client.messages.calls) == budget + 1
        # Final call had no tools
        assert "tools" not in client.messages.calls[-1]

    def test_unparseable_final_returns_none(self, repo: Path):
        client = FakeClient([
            _make_response(
                [{"type": "text", "text": "I refuse to JSON"}],
                stop_reason="end_turn",
            ),
        ])
        out = tool_use_scan(
            package_name="pkg", files=["pkg/x.ts"],
            repo_root=repo, client=client,
        )
        assert out is None

    def test_tool_use_stop_with_no_blocks_returns_none(self, repo: Path):
        client = FakeClient([
            _make_response([{"type": "text", "text": "weird"}], stop_reason="tool_use"),
        ])
        out = tool_use_scan(
            package_name="pkg", files=["pkg/x.ts"],
            repo_root=repo, client=client,
        )
        assert out is None

    def test_unknown_tool_returns_error_string_not_crash(self, repo: Path):
        client = FakeClient([
            _make_response(
                [{"type": "tool_use", "id": "z", "name": "rm_rf_slash",
                  "input": {}}],
                stop_reason="tool_use",
            ),
            _make_response(
                [{"type": "text", "text": '{"features": []}'}],
                stop_reason="end_turn",
            ),
        ])
        out = tool_use_scan(
            package_name="pkg", files=["pkg/x.ts"],
            repo_root=repo, client=client,
        )
        assert out == {"features": []}
        tr = client.messages.calls[1]["messages"][-1]["content"][0]
        assert tr["is_error"] is True
        assert "unknown tool" in tr["content"]

    def test_default_budget_constant(self):
        assert DEFAULT_TOOL_BUDGET == 15

    def test_first_call_includes_tools_and_system_prompt(self, repo: Path):
        client = FakeClient([
            _make_response(
                [{"type": "text", "text": '{"features": []}'}],
                stop_reason="end_turn",
            ),
        ])
        tool_use_scan(
            package_name="pkg", files=["pkg/x.ts"],
            repo_root=repo, client=client,
        )
        call = client.messages.calls[0]
        assert "tools" in call
        assert isinstance(call["tools"], list) and len(call["tools"]) >= 4
        assert "system" in call and "business-readable" in call["system"]
        # User prompt mentions package name + file list
        assert call["messages"][0]["role"] == "user"
        assert "Package: pkg" in call["messages"][0]["content"]
        assert "pkg/x.ts" in call["messages"][0]["content"]
