"""Unit tests for ``faultline.llm.flow_detector_v2`` (Sprint 4 Day 1)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from faultline.llm.flow_detector_v2 import (
    DEFAULT_MAX_FLOWS_PER_FEATURE,
    DEFAULT_TOOL_BUDGET,
    _validate_flows,
    detect_flows_for_feature,
    detect_flows_with_tools,
)
from faultline.llm.sonnet_scanner import DeepScanResult


def _resp(text: str, *, usage_in=200, usage_out=80):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=usage_in, output_tokens=usage_out),
    )


class FakeMessages:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self._scripted:
            raise RuntimeError("no scripted response left")
        nxt = self._scripted.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class FakeClient:
    def __init__(self, scripted):
        self.messages = FakeMessages(scripted)


# ── _validate_flows ──────────────────────────────────────────────────


class TestValidateFlows:
    def test_clean_flow(self):
        out = _validate_flows([
            {"name": "create-doc",
             "description": "x",
             "entry_point_file": "a.ts",
             "entry_point_line": 12},
        ])
        assert out and out[0]["name"] == "create-doc"
        assert out[0]["entry_point_line"] == 12

    def test_drops_generic_names(self):
        out = _validate_flows([
            {"name": "process-data-flow",
             "entry_point_file": "x.ts"},
            {"name": "real-action",
             "entry_point_file": "y.ts"},
        ])
        assert out is not None
        assert {f["name"] for f in out} == {"real-action"}

    def test_drops_missing_entry_point(self):
        out = _validate_flows([
            {"name": "no-entry"},  # missing entry_point_file
            {"name": "with-entry", "entry_point_file": "y.ts"},
        ])
        assert out and len(out) == 1
        assert out[0]["name"] == "with-entry"

    def test_drops_duplicate_names(self):
        out = _validate_flows([
            {"name": "x", "entry_point_file": "a.ts"},
            {"name": "x", "entry_point_file": "b.ts"},
        ])
        assert out and len(out) == 1

    def test_caps_at_max(self):
        many = [
            {"name": f"f{i}", "entry_point_file": "a.ts"}
            for i in range(DEFAULT_MAX_FLOWS_PER_FEATURE + 5)
        ]
        out = _validate_flows(many)
        assert out and len(out) == DEFAULT_MAX_FLOWS_PER_FEATURE

    def test_garbage_returns_none(self):
        assert _validate_flows(None) is None  # type: ignore[arg-type]
        assert _validate_flows([]) is None
        assert _validate_flows([{"name": "x"}]) is None  # no entry, all dropped


# ── detect_flows_for_feature ─────────────────────────────────────────


class TestDetectFlowsForFeature:
    def test_returns_none_when_no_files(self, tmp_path: Path):
        client = FakeClient([])
        out = detect_flows_for_feature(
            name="x", files=[], repo_root=tmp_path, client=client,
        )
        assert out is None

    def test_clean_response_validated(self, tmp_path: Path):
        client = FakeClient([_resp(json.dumps({
            "flows": [
                {"name": "sign-doc", "description": "user signs",
                 "entry_point_file": "sign.tsx", "entry_point_line": 5},
            ]
        }))])
        out = detect_flows_for_feature(
            name="signing", files=["sign.tsx"], repo_root=tmp_path, client=client,
        )
        assert out and out[0]["name"] == "sign-doc"
        assert out[0]["entry_point_line"] == 5

    def test_empty_flows_means_no_grounding(self, tmp_path: Path):
        client = FakeClient([_resp('{"flows": []}')])
        out = detect_flows_for_feature(
            name="x", files=["a.ts"], repo_root=tmp_path, client=client,
        )
        assert out is None

    def test_unparseable_returns_none(self, tmp_path: Path):
        client = FakeClient([_resp("nope")])
        out = detect_flows_for_feature(
            name="x", files=["a.ts"], repo_root=tmp_path, client=client,
        )
        assert out is None

    def test_anthropic_exception_returns_none(self, tmp_path: Path):
        client = FakeClient([RuntimeError("net")])
        out = detect_flows_for_feature(
            name="x", files=["a.ts"], repo_root=tmp_path, client=client,
        )
        assert out is None

    def test_tracker_records_usage(self, tmp_path: Path):
        from faultline.llm.cost import CostTracker
        tracker = CostTracker()
        client = FakeClient([_resp(json.dumps({
            "flows": [{"name": "x-flow", "entry_point_file": "a.ts"}]
        }), usage_in=400, usage_out=90)])
        detect_flows_for_feature(
            name="x", files=["a.ts"], repo_root=tmp_path,
            client=client, tracker=tracker,
        )
        s = tracker.summary()
        assert s["total_calls"] == 1
        assert s["total_input_tokens"] == 400
        assert s["total_output_tokens"] == 90


# ── detect_flows_with_tools (top level) ──────────────────────────────


def _result(features, **kw) -> DeepScanResult:
    return DeepScanResult(features=features, **kw)


class TestDetectFlowsWithTools:
    def test_skips_when_library(self, tmp_path: Path):
        result = _result({"x": ["a.ts"]})
        client = FakeClient([])
        out = detect_flows_with_tools(
            result, repo_root=tmp_path, is_library=True, client=client,
        )
        assert client.messages.calls == []
        assert "x" not in out.flows

    def test_skips_protected(self, tmp_path: Path):
        result = _result({
            "documentation": ["docs/a.md"],
            "shared-infra": ["package.json"],
            "examples": ["examples/x.ts"],
            "real-feature": ["a.ts"],
        })
        client = FakeClient([_resp(json.dumps({
            "flows": [{"name": "create-thing", "entry_point_file": "a.ts"}]
        }))])
        out = detect_flows_with_tools(
            result, repo_root=tmp_path, client=client,
        )
        # Only one feature reached the LLM
        assert len(client.messages.calls) == 1
        assert "real-feature" in out.flows
        assert "documentation" not in out.flows

    def test_writes_flows_and_descriptions(self, tmp_path: Path):
        result = _result({"signing": ["sign.tsx"]})
        client = FakeClient([_resp(json.dumps({
            "flows": [
                {"name": "sign-doc", "description": "user signs",
                 "entry_point_file": "apps/sign.tsx",
                 "entry_point_line": 12},
            ]
        }))])
        out = detect_flows_with_tools(
            result, repo_root=tmp_path, client=client,
        )
        assert out.flows["signing"] == ["sign-doc"]
        desc = out.flow_descriptions["signing"]["sign-doc"]
        assert "user signs" in desc
        assert "(entry: apps/sign.tsx:12)" in desc

    def test_no_repo_root_skips(self, tmp_path: Path):
        result = _result({"x": ["a.ts"]})
        client = FakeClient([])
        out = detect_flows_with_tools(
            result, repo_root=None, client=client,
        )
        assert client.messages.calls == []
        assert "x" not in out.flows

    def test_empty_response_keeps_existing(self, tmp_path: Path):
        result = _result(
            {"x": ["a.ts"]},
            flows={"x": ["legacy-flow"]},
            flow_descriptions={"x": {"legacy-flow": "old"}},
        )
        client = FakeClient([_resp('{"flows": []}')])
        out = detect_flows_with_tools(
            result, repo_root=tmp_path, client=client,
        )
        assert out.flows["x"] == ["legacy-flow"]
        assert out.flow_descriptions["x"]["legacy-flow"] == "old"

    def test_default_budget(self):
        assert DEFAULT_TOOL_BUDGET == 6
        assert DEFAULT_MAX_FLOWS_PER_FEATURE == 8
