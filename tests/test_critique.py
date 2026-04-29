"""Unit tests for ``faultline.llm.critique`` (Sprint 5 Day 1)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from faultline.llm.critique import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_TOOL_BUDGET,
    _build_critique_summaries,
    _coerce_critique,
    _is_materially_better,
    _rewrite_feature_name,
    _rewrite_flow_name,
    critique_and_refine,
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


# ── _is_materially_better ─────────────────────────────────────────────


class TestMateriallyBetter:
    def test_specific_replaces_generic(self):
        assert _is_materially_better(
            "lib/platform-infrastructure", "pdf-rendering"
        )

    def test_synonym_only_rejected(self):
        # Both essentially the same domain — no novelty
        assert not _is_materially_better("billing", "billing-service")

    def test_case_only_diff_rejected(self):
        assert not _is_materially_better("auth", "AUTH")

    def test_empty_new_rejected(self):
        assert not _is_materially_better("auth", "")

    def test_all_generic_new_rejected(self):
        assert not _is_materially_better("billing", "lib/utils")

    def test_completely_different_accepted(self):
        assert _is_materially_better("misc", "stripe-checkout")

    def test_adds_specific_token(self):
        assert _is_materially_better("data", "user-events-data")


# ── _build_critique_summaries ─────────────────────────────────────────


class TestBuildSummaries:
    def test_skips_protected(self):
        r = DeepScanResult(
            features={
                "auth": ["a.ts"],
                "documentation": ["d.md"],
                "shared-infra": ["package.json"],
                "examples": ["e.ts"],
            },
            descriptions={"auth": "User auth"},
        )
        out = _build_critique_summaries(r)
        names = [s["name"] for s in out]
        assert "auth" in names
        assert "documentation" not in names
        assert "shared-infra" not in names
        assert "examples" not in names

    def test_flattens_flows(self):
        r = DeepScanResult(
            features={"auth": ["a.ts"]},
            flows={"auth": ["sign-in", "sign-out"]},
            flow_descriptions={"auth": {"sign-in": "user signs in"}},
        )
        out = _build_critique_summaries(r)
        kinds = [s["kind"] for s in out]
        assert kinds.count("feature") == 1
        assert kinds.count("flow") == 2
        # Flow desc carried through
        signin = next(s for s in out if s.get("name") == "sign-in")
        assert signin["description"] == "user signs in"


# ── _coerce_critique ──────────────────────────────────────────────────


class TestCoerceCritique:
    def test_valid_feature_item(self):
        out = _coerce_critique({"weak": [
            {"kind": "feature", "name": "lib", "reason": "generic"}
        ]}, max_items=5)
        assert out == [{"kind": "feature", "name": "lib", "reason": "generic"}]

    def test_valid_flow_item(self):
        out = _coerce_critique({"weak": [
            {"kind": "flow", "feature": "billing", "name": "manage-things",
             "reason": "vague"}
        ]}, max_items=5)
        assert out and out[0]["feature"] == "billing"

    def test_drops_unknown_kind(self):
        out = _coerce_critique({"weak": [
            {"kind": "package", "name": "x", "reason": "y"}
        ]}, max_items=5)
        assert out == []

    def test_drops_protected(self):
        out = _coerce_critique({"weak": [
            {"kind": "feature", "name": "documentation", "reason": "x"}
        ]}, max_items=5)
        assert out == []

    def test_drops_flow_without_feature(self):
        out = _coerce_critique({"weak": [
            {"kind": "flow", "name": "x", "reason": "y"}
        ]}, max_items=5)
        assert out == []

    def test_caps_at_max_items(self):
        many = {"weak": [
            {"kind": "feature", "name": f"f{i}", "reason": "y"}
            for i in range(10)
        ]}
        out = _coerce_critique(many, max_items=3)
        assert len(out) == 3

    def test_garbage(self):
        assert _coerce_critique(None, max_items=5) == []
        assert _coerce_critique({"weak": "not a list"}, max_items=5) == []


# ── rename helpers ────────────────────────────────────────────────────


class TestRewriteFeature:
    def test_renames_features_descriptions_flows(self):
        r = DeepScanResult(
            features={"old": ["a.ts"]},
            descriptions={"old": "old desc"},
            flows={"old": ["x-flow"]},
            flow_descriptions={"old": {"x-flow": "x"}},
        )
        ok = _rewrite_feature_name(r, "old", "new")
        assert ok is True
        assert "new" in r.features and "old" not in r.features
        assert "new" in r.descriptions
        assert "new" in r.flows
        assert "new" in r.flow_descriptions

    def test_uses_supplied_description(self):
        r = DeepScanResult(features={"old": ["a.ts"]})
        _rewrite_feature_name(r, "old", "new", new_description="fresh")
        assert r.descriptions["new"] == "fresh"

    def test_noop_when_old_missing(self):
        r = DeepScanResult(features={"a": ["a.ts"]})
        ok = _rewrite_feature_name(r, "missing", "new")
        assert ok is False

    def test_noop_when_new_already_exists(self):
        r = DeepScanResult(features={"a": ["a.ts"], "b": ["b.ts"]})
        ok = _rewrite_feature_name(r, "a", "b")
        assert ok is False
        assert "a" in r.features


class TestRewriteFlow:
    def test_renames_flow(self):
        r = DeepScanResult(
            features={"x": ["a.ts"]},
            flows={"x": ["old-flow", "other"]},
            flow_descriptions={"x": {"old-flow": "old desc"}},
        )
        ok = _rewrite_flow_name(r, "x", "old-flow", "new-flow")
        assert ok is True
        assert "new-flow" in r.flows["x"]
        assert "old-flow" not in r.flows["x"]
        assert r.flow_descriptions["x"]["new-flow"] == "old desc"

    def test_noop_when_feature_missing(self):
        r = DeepScanResult(features={"x": ["a.ts"]})
        assert _rewrite_flow_name(r, "missing", "a", "b") is False

    def test_noop_when_new_collides(self):
        r = DeepScanResult(features={"x": ["a"]}, flows={"x": ["a", "b"]})
        assert _rewrite_flow_name(r, "x", "a", "b") is False


# ── critique_and_refine end-to-end ────────────────────────────────────


class TestCritiqueAndRefine:
    def test_skips_when_no_features(self, tmp_path: Path):
        r = DeepScanResult(features={})
        client = FakeClient([])
        out = critique_and_refine(r, repo_root=tmp_path, client=client)
        assert client.messages.calls == []
        assert out is r

    def test_no_weak_items_is_noop(self, tmp_path: Path):
        r = DeepScanResult(features={"auth": ["a.ts"]})
        client = FakeClient([_resp('{"weak": []}')])
        out = critique_and_refine(r, repo_root=tmp_path, client=client)
        assert "auth" in out.features

    def test_unparseable_returns_original(self, tmp_path: Path):
        r = DeepScanResult(features={"auth": ["a.ts"]})
        client = FakeClient([_resp("totally not json")])
        out = critique_and_refine(r, repo_root=tmp_path, client=client)
        assert "auth" in out.features

    def test_anthropic_exception_returns_original(self, tmp_path: Path):
        r = DeepScanResult(features={"auth": ["a.ts"]})
        client = FakeClient([RuntimeError("net")])
        out = critique_and_refine(r, repo_root=tmp_path, client=client)
        assert "auth" in out.features

    def test_feature_rename_applied_when_better(self, tmp_path: Path):
        (tmp_path / "x.ts").write_text("x", encoding="utf-8")
        # NOTE: critique now skips features whose name contains a
        # slash (Sub-features from Sprint 3 are protected). Use a
        # flat name so the rename path is exercised.
        r = DeepScanResult(
            features={"platform-infrastructure": ["x.ts"]},
            descriptions={"platform-infrastructure": "stuff"},
        )
        client = FakeClient([
            _resp('{"weak": [{"kind": "feature", "name": "platform-infrastructure", "reason": "generic"}]}'),
            # tool_use_scan flow: model returns final text, no tool calls
            _resp(json.dumps({
                "new_name": "pdf-rendering",
                "description": "PDF rendering for signed docs.",
                "reason": "70% are PDF.",
            })),
        ])
        out = critique_and_refine(
            r, repo_root=tmp_path, client=client, max_items=5,
        )
        assert "pdf-rendering" in out.features
        assert "platform-infrastructure" not in out.features
        assert out.descriptions["pdf-rendering"] == "PDF rendering for signed docs."

    def test_feature_rename_rejected_when_not_better(self, tmp_path: Path):
        (tmp_path / "x.ts").write_text("x", encoding="utf-8")
        r = DeepScanResult(features={"billing": ["x.ts"]})
        client = FakeClient([
            _resp('{"weak": [{"kind": "feature", "name": "billing", "reason": "vague"}]}'),
            _resp(json.dumps({"new_name": "billing-service", "description": "x"})),
        ])
        out = critique_and_refine(
            r, repo_root=tmp_path, client=client, max_items=5,
        )
        # Synonymous rename rejected — original survives
        assert "billing" in out.features
        assert "billing-service" not in out.features

    def test_flow_rename_applied(self, tmp_path: Path):
        (tmp_path / "x.ts").write_text("x", encoding="utf-8")
        r = DeepScanResult(
            features={"billing": ["x.ts"]},
            flows={"billing": ["manage-things"]},
            flow_descriptions={"billing": {"manage-things": "x"}},
        )
        client = FakeClient([
            _resp('{"weak": [{"kind": "flow", "feature": "billing", "name": "manage-things", "reason": "vague"}]}'),
            _resp('{"new_name": "cancel-subscription"}'),
        ])
        out = critique_and_refine(
            r, repo_root=tmp_path, client=client, max_items=5,
        )
        assert "cancel-subscription" in out.flows["billing"]
        assert "manage-things" not in out.flows["billing"]

    def test_no_repo_root_skips_apply(self, tmp_path: Path):
        # Flat feature name (slashed names are now skipped entirely
        # by the slash-guard in critique).
        r = DeepScanResult(features={"infra-shared": ["a.ts"]})
        client = FakeClient([
            _resp('{"weak": [{"kind": "feature", "name": "infra-shared", "reason": "generic"}]}'),
        ])
        out = critique_and_refine(r, repo_root=None, client=client)
        # critique pass ran (1 call) but no rename apply → only 1 call total
        assert len(client.messages.calls) == 1
        assert "infra-shared" in out.features

    def test_tracker_records_critique_call(self, tmp_path: Path):
        from faultline.llm.cost import CostTracker
        tracker = CostTracker()
        r = DeepScanResult(features={"auth": ["a.ts"]})
        client = FakeClient([_resp('{"weak": []}', usage_in=300, usage_out=40)])
        critique_and_refine(
            r, repo_root=tmp_path, client=client, tracker=tracker,
        )
        s = tracker.summary()
        assert s["total_calls"] == 1
        assert s["total_input_tokens"] == 300

    def test_default_constants(self):
        # Stability tuning lowered the cap from 5 → 2 (Step E).
        # See SPRINT_5_STABILITY analysis in the commit message.
        assert DEFAULT_MAX_ITEMS == 2
        assert DEFAULT_TOOL_BUDGET == 5

    def test_locked_names_skip_critique(self, tmp_path):
        # Locked features get stripped from the critique input so
        # the model never sees them as candidates for renaming.
        # NOTE: features with slash names are also skipped (Sprint 3
        # sub-features are protected by the slash-guard) — using
        # flat names so the test exercises lock specifically.
        r = DeepScanResult(features={
            "billing": ["a.ts"],
            "platform": ["b.ts"],
        })
        client = FakeClient([_resp('{"weak": []}')])
        out = critique_and_refine(
            r, repo_root=tmp_path, client=client,
            locked_names=frozenset({"billing"}),
        )
        assert "billing" in out.features
        user_msg = client.messages.calls[0]["messages"][0]["content"]
        assert "billing" not in user_msg
        assert "platform" in user_msg

    def test_locked_names_partial_filter(self, tmp_path):
        r = DeepScanResult(features={
            "billing": ["a.ts"],
            "platform": ["b.ts"],
        })
        client = FakeClient([_resp('{"weak": []}')])
        critique_and_refine(
            r, repo_root=tmp_path, client=client,
            locked_names=frozenset({"billing"}),
        )
        user_msg = client.messages.calls[0]["messages"][0]["content"]
        assert "billing" not in user_msg
        assert "platform" in user_msg

    def test_locked_names_token_match(self, tmp_path):
        # Stability fix: token-set match in lock. ``auth`` locked
        # blocks ``user-authentication`` from being flagged in a
        # later run.
        r = DeepScanResult(features={
            "user-authentication": ["a.ts"],
            "billing-and-subscriptions": ["b.ts"],
        })
        client = FakeClient([_resp('{"weak": []}')])
        critique_and_refine(
            r, repo_root=tmp_path, client=client,
            locked_names=frozenset({"authentication"}),
        )
        user_msg = client.messages.calls[0]["messages"][0]["content"]
        # token-set match collapses ``user-authentication`` ≡
        # ``authentication``  → filtered
        assert "user-authentication" not in user_msg
        assert "billing-and-subscriptions" in user_msg

    def test_subfeature_names_skipped_from_critique(self, tmp_path):
        # Sprint 3 sub-features (path/sub) never get critiqued —
        # they're already named carefully and re-naming them
        # destabilizes the trace.
        r = DeepScanResult(features={
            "auth/login-handler": ["a.ts"],
            "billing": ["b.ts"],
        })
        client = FakeClient([_resp('{"weak": []}')])
        critique_and_refine(
            r, repo_root=tmp_path, client=client,
        )
        user_msg = client.messages.calls[0]["messages"][0]["content"]
        # Slashed name not in critique input
        assert "auth/login-handler" not in user_msg
        assert "billing" in user_msg

    def test_all_locked_short_circuits(self, tmp_path):
        r = DeepScanResult(features={"billing": ["a.ts"], "auth": ["b.ts"]})
        client = FakeClient([])  # no calls expected
        out = critique_and_refine(
            r, repo_root=tmp_path, client=client,
            locked_names=frozenset({"billing", "auth"}),
        )
        assert "billing" in out.features
        assert "auth" in out.features
        assert client.messages.calls == []

    def test_weak_items_processed_in_size_order(self, tmp_path):
        # Critique flags 3 features. We expect dispatch in file-count
        # desc order, then alphabetical within ties.
        (tmp_path / "x.ts").write_text("x", encoding="utf-8")
        r = DeepScanResult(features={
            "small": ["a.ts"],
            "biggest-feature": [f"f{i}.ts" for i in range(50)],
            "medium-thing": [f"m{i}.ts" for i in range(20)],
        })
        # Model returns in random order to simulate non-determinism.
        weak_response = _resp('{"weak": ['
            '{"kind": "feature", "name": "small", "reason": "x"},'
            '{"kind": "feature", "name": "biggest-feature", "reason": "x"},'
            '{"kind": "feature", "name": "medium-thing", "reason": "x"}'
        ']}')
        # Each rename returns empty new_name → rejected, but log
        # order shows the dispatch order.
        decline = _resp('{"new_name": ""}')
        client = FakeClient([weak_response, decline, decline, decline])
        critique_and_refine(
            r, repo_root=tmp_path, client=client,
            # Override the default 2 cap so all 3 items dispatch
            # — the test is about ORDER, not the new cap.
            max_items=10,
        )
        # Inspect the order rename calls were dispatched in: each
        # tool_use_scan call sends the feature name in the user
        # prompt. Calls 1, 2, 3 (after the critique pass at index 0).
        call_names = [
            c["messages"][0]["content"].split("\n")[0].replace("Package: ", "")
            for c in client.messages.calls[1:]
            if "Package:" in c["messages"][0]["content"]
        ]
        # biggest-feature first (50 files), medium-thing next (20),
        # small last (1)
        assert call_names == ["biggest-feature", "medium-thing", "small"]
