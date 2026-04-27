"""Unit tests for ``faultline.llm.sub_decompose`` (Sprint 3 Day 1)."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from faultline.llm.sub_decompose import (
    DEFAULT_MAX_SUB_FEATURES,
    DEFAULT_THRESHOLD,
    _validate_split,
    sub_decompose_feature,
    sub_decompose_oversized,
)
from faultline.llm.sonnet_scanner import DeepScanResult


# ── Fake Anthropic client ─────────────────────────────────────────────


def _resp(text: str, *, stop="end_turn", usage_in=100, usage_out=50):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
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


# ── _validate_split ───────────────────────────────────────────────────


class TestValidateSplit:
    def test_clean_split_returned(self):
        out = _validate_split(
            [
                {"name": "a", "paths": ["x.ts", "y.ts"]},
                {"name": "b", "paths": ["z.ts"]},
            ],
            parent_files=["x.ts", "y.ts", "z.ts"],
        )
        assert out is not None
        assert {s["name"] for s in out} == {"a", "b"}

    def test_under_two_subfeatures_rejected(self):
        out = _validate_split(
            [{"name": "a", "paths": ["x.ts"]}],
            parent_files=["x.ts"],
        )
        assert out is None

    def test_over_max_subfeatures_rejected(self):
        subs = [
            {"name": f"s{i}", "paths": [f"f{i}.ts"]}
            for i in range(DEFAULT_MAX_SUB_FEATURES + 1)
        ]
        parent = [f"f{i}.ts" for i in range(DEFAULT_MAX_SUB_FEATURES + 1)]
        out = _validate_split(subs, parent_files=parent)
        assert out is None

    def test_missing_files_rejected(self):
        out = _validate_split(
            [
                {"name": "a", "paths": ["x.ts"]},
                {"name": "b", "paths": ["y.ts"]},
            ],
            parent_files=["x.ts", "y.ts", "z.ts"],
        )
        assert out is None

    def test_extra_files_rejected(self):
        out = _validate_split(
            [
                {"name": "a", "paths": ["x.ts"]},
                {"name": "b", "paths": ["y.ts", "extra.ts"]},
            ],
            parent_files=["x.ts", "y.ts"],
        )
        assert out is None

    def test_duplicate_path_across_subs_rejected(self):
        out = _validate_split(
            [
                {"name": "a", "paths": ["x.ts", "shared.ts"]},
                {"name": "b", "paths": ["shared.ts", "y.ts"]},
            ],
            parent_files=["x.ts", "y.ts", "shared.ts"],
        )
        assert out is None

    def test_duplicate_subfeature_name_rejected(self):
        out = _validate_split(
            [
                {"name": "a", "paths": ["x.ts"]},
                {"name": "a", "paths": ["y.ts"]},
            ],
            parent_files=["x.ts", "y.ts"],
        )
        assert out is None

    def test_empty_paths_rejected(self):
        out = _validate_split(
            [
                {"name": "a", "paths": []},
                {"name": "b", "paths": ["y.ts"]},
            ],
            parent_files=["y.ts"],
        )
        assert out is None

    def test_generic_name_rejected(self):
        for bad in ("core", "lib", "utils", "shared", "general", "main"):
            out = _validate_split(
                [
                    {"name": bad, "paths": ["x.ts"]},
                    {"name": "real", "paths": ["y.ts"]},
                ],
                parent_files=["x.ts", "y.ts"],
            )
            assert out is None, f"generic '{bad}' should be rejected"

    def test_empty_name_rejected(self):
        out = _validate_split(
            [
                {"name": "", "paths": ["x.ts"]},
                {"name": "b", "paths": ["y.ts"]},
            ],
            parent_files=["x.ts", "y.ts"],
        )
        assert out is None

    def test_garbage_input(self):
        assert _validate_split(None, ["x.ts"]) is None  # type: ignore[arg-type]
        assert _validate_split([], ["x.ts"]) is None
        assert _validate_split("nope", ["x.ts"]) is None  # type: ignore[arg-type]


# ── sub_decompose_feature with fake client ────────────────────────────


class TestSubDecomposeFeature:
    def test_returns_none_when_files_empty(self, tmp_path):
        client = FakeClient([])
        out = sub_decompose_feature(
            name="x", files=[], repo_root=tmp_path, client=client,
        )
        assert out is None
        assert client.messages.calls == []

    def test_clean_split_returns_validated_subs(self, tmp_path):
        files = ["a.ts", "b.ts", "c.ts", "d.ts"]
        for f in files:
            (tmp_path / f).write_text("x", encoding="utf-8")
        client = FakeClient([
            _resp(
                '{"features": ['
                '{"name": "alpha", "paths": ["a.ts","b.ts"], "description": "A"},'
                '{"name": "beta",  "paths": ["c.ts","d.ts"], "description": "B"}'
                ']}'
            ),
        ])
        out = sub_decompose_feature(
            name="parent", files=files, repo_root=tmp_path, client=client,
        )
        assert out is not None
        assert {s["name"] for s in out} == {"alpha", "beta"}

    def test_empty_subfeatures_means_no_split(self, tmp_path):
        client = FakeClient([_resp('{"features": []}')])
        out = sub_decompose_feature(
            name="x", files=["a.ts", "b.ts"], repo_root=tmp_path, client=client,
        )
        assert out is None

    def test_invalid_split_returns_none(self, tmp_path):
        # missing file 'b.ts'
        client = FakeClient([_resp(
            '{"features": ['
            '{"name": "alpha", "paths": ["a.ts"]},'
            '{"name": "beta",  "paths": ["c.ts"]}'
            ']}'
        )])
        out = sub_decompose_feature(
            name="x",
            files=["a.ts", "b.ts", "c.ts"],
            repo_root=tmp_path,
            client=client,
        )
        assert out is None

    def test_unparseable_response_returns_none(self, tmp_path):
        client = FakeClient([_resp("not json at all")])
        out = sub_decompose_feature(
            name="x", files=["a.ts", "b.ts"], repo_root=tmp_path, client=client,
        )
        assert out is None

    def test_anthropic_exception_returns_none(self, tmp_path):
        client = FakeClient([RuntimeError("network kaput")])
        out = sub_decompose_feature(
            name="x", files=["a.ts", "b.ts"], repo_root=tmp_path, client=client,
        )
        assert out is None

    def test_tracker_records_token_usage(self, tmp_path):
        from faultline.llm.cost import CostTracker

        files = ["a.ts", "b.ts"]
        for f in files:
            (tmp_path / f).write_text("x", encoding="utf-8")
        tracker = CostTracker()
        client = FakeClient([_resp(
            '{"features": ['
            '{"name": "alpha", "paths": ["a.ts"]},'
            '{"name": "beta", "paths": ["b.ts"]}'
            ']}',
            usage_in=300, usage_out=80,
        )])
        sub_decompose_feature(
            name="x", files=files, repo_root=tmp_path,
            client=client, tracker=tracker,
        )
        s = tracker.summary()
        assert s["total_calls"] == 1
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 80


# ── sub_decompose_oversized (top level) ───────────────────────────────


def _result(features: dict, descriptions: dict | None = None,
            flows: dict | None = None,
            flow_descriptions: dict | None = None) -> DeepScanResult:
    return DeepScanResult(
        features=features,
        descriptions=descriptions or {},
        flows=flows or {},
        flow_descriptions=flow_descriptions or {},
    )


class TestSubDecomposeOversized:
    def test_skips_when_no_features_oversized(self, tmp_path):
        result = _result({"small": [f"{i}.ts" for i in range(50)]})
        client = FakeClient([])
        out = sub_decompose_oversized(
            result, threshold=200, repo_root=tmp_path, client=client,
        )
        assert "small" in out.features
        assert client.messages.calls == []

    def test_skips_protected_names(self, tmp_path):
        big_docs = [f"docs/{i}.md" for i in range(500)]
        result = _result({"documentation": big_docs})
        client = FakeClient([])
        out = sub_decompose_oversized(
            result, threshold=200, repo_root=tmp_path, client=client,
        )
        assert "documentation" in out.features
        assert client.messages.calls == []

    def test_replaces_oversized_with_subfeatures(self, tmp_path):
        files = [f"f{i}.ts" for i in range(250)]
        result = _result({"big-feature": files})
        # Build a valid 2-way split response.
        a = files[:100]
        b = files[100:]
        import json
        client = FakeClient([_resp(json.dumps({
            "features": [
                {"name": "alpha", "paths": a, "description": "A"},
                {"name": "beta",  "paths": b, "description": "B"},
            ]
        }))])
        out = sub_decompose_oversized(
            result, threshold=200, repo_root=tmp_path, client=client,
        )
        assert "big-feature" not in out.features
        assert "big-feature/alpha" in out.features
        assert "big-feature/beta" in out.features
        assert sorted(out.features["big-feature/alpha"]) == sorted(a)
        assert out.descriptions.get("big-feature/alpha") == "A"

    def test_keeps_parent_when_split_invalid(self, tmp_path):
        files = [f"f{i}.ts" for i in range(250)]
        result = _result({"big": files})
        # Returns subs that don't cover all files (missing some).
        client = FakeClient([_resp(
            '{"features": ['
            '{"name": "x", "paths": ["f1.ts"]},'
            '{"name": "y", "paths": ["f2.ts"]}'
            ']}'
        )])
        out = sub_decompose_oversized(
            result, threshold=200, repo_root=tmp_path, client=client,
        )
        assert "big" in out.features
        assert len(out.features["big"]) == 250

    def test_keeps_parent_when_model_declines(self, tmp_path):
        files = [f"f{i}.ts" for i in range(250)]
        result = _result({"big": files})
        client = FakeClient([_resp('{"features": []}')])
        out = sub_decompose_oversized(
            result, threshold=200, repo_root=tmp_path, client=client,
        )
        assert "big" in out.features

    def test_flows_carry_to_largest_subfeature(self, tmp_path):
        files = [f"f{i}.ts" for i in range(250)]
        result = _result(
            {"big": files},
            flows={"big": ["create-thing", "delete-thing"]},
            flow_descriptions={"big": {"create-thing": "make stuff"}},
            descriptions={"big": "parent desc"},
        )
        a = files[:200]  # bigger
        b = files[200:]
        import json
        client = FakeClient([_resp(json.dumps({
            "features": [
                {"name": "alpha", "paths": a},
                {"name": "beta",  "paths": b},
            ]
        }))])
        out = sub_decompose_oversized(
            result, threshold=200, repo_root=tmp_path, client=client,
        )
        # alpha is largest → inherits flows + flow_descs + parent desc
        assert "big/alpha" in out.flows
        assert out.flows["big/alpha"] == ["create-thing", "delete-thing"]
        assert out.flow_descriptions["big/alpha"]["create-thing"] == "make stuff"
        assert out.descriptions["big/alpha"] == "parent desc"
        # beta has neither
        assert "big/beta" not in out.flows
        assert "big/beta" not in out.flow_descriptions

    def test_no_repo_root_skips_pass(self, tmp_path):
        files = [f"f{i}.ts" for i in range(250)]
        result = _result({"big": files})
        client = FakeClient([])
        out = sub_decompose_oversized(
            result, threshold=200, repo_root=None, client=client,
        )
        assert "big" in out.features
        assert client.messages.calls == []

    def test_threshold_default_200(self):
        assert DEFAULT_THRESHOLD == 200

    def test_max_subfeatures_default_6(self):
        assert DEFAULT_MAX_SUB_FEATURES == 6
