"""Unit tests for ``faultline.llm.dedup`` (Sprint 2 Day 1).

No network. The Anthropic client is mocked with a hand-rolled fake;
the same pattern as ``tests/test_tool_use_scan.py``.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from faultline.llm.dedup import (
    MAX_MERGES_PER_PASS,
    Merge,
    _apply_merges,
    _build_summaries,
    _coerce_merges,
    _parse_merges_payload,
    dedup_features,
)
from faultline.llm.sonnet_scanner import DeepScanResult


# ── Fake Anthropic client ─────────────────────────────────────────────


def _resp(text: str, *, usage_in: int = 100, usage_out: int = 50):
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


# ── Fixtures ──────────────────────────────────────────────────────────


def _result_with_signing():
    """Mimics documenso post-Sprint-1 with 3 signing siblings."""
    return DeepScanResult(
        features={
            "lib/document-signing": ["lib/sign1.ts", "lib/sign2.ts", "lib/sign3.ts"],
            "remix/document-signing": ["remix/route1.tsx", "remix/route2.tsx"],
            "trpc/document-signing": ["trpc/proc1.ts", "trpc/proc2.ts"],
            "ui/design-system-primitives": ["ui/button.tsx", "ui/card.tsx"],
            "documentation": ["docs/intro.md"],
        },
        descriptions={
            "lib/document-signing": "Signing core logic",
            "remix/document-signing": "Signing routes",
            "trpc/document-signing": "Signing API procs",
            "ui/design-system-primitives": "Design system",
        },
        flows={
            "lib/document-signing": ["sign-document"],
            "remix/document-signing": ["open-signing-page"],
        },
        flow_descriptions={
            "lib/document-signing": {"sign-document": "user signs"},
        },
    )


# ── _build_summaries ──────────────────────────────────────────────────


class TestBuildSummaries:
    def test_skips_protected(self):
        feats = {"docs": ["a.md"], "documentation": ["x.md"], "shared-infra": ["y"]}
        out = _build_summaries(feats, {})
        names = [s["name"] for s in out]
        assert "documentation" not in names
        assert "shared-infra" not in names
        assert "docs" in names

    def test_carries_description_and_count(self):
        out = _build_summaries({"a": ["1.ts", "2.ts"]}, {"a": "thing"})
        assert out == [{
            "name": "a",
            "description": "thing",
            "file_count": 2,
            "sample_paths": ["1.ts", "2.ts"],
        }]

    def test_caps_sample_paths_default_5(self):
        feats = {"a": [f"{i}.ts" for i in range(20)]}
        out = _build_summaries(feats, {})
        assert len(out[0]["sample_paths"]) == 5

    def test_deterministic_ordering(self):
        feats = {"b": ["b.ts"], "a": ["a.ts"], "c": ["c.ts"]}
        names = [s["name"] for s in _build_summaries(feats, {})]
        assert names == ["a", "b", "c"]

    def test_empty_input(self):
        assert _build_summaries({}, {}) == []


# ── _parse_merges_payload ─────────────────────────────────────────────


class TestParse:
    def test_raw_json(self):
        assert _parse_merges_payload('{"merges": []}') == {"merges": []}

    def test_fenced(self):
        text = "ok ```json\n{\"merges\":[1]}\n```"
        assert _parse_merges_payload(text) == {"merges": [1]}

    def test_balanced_braces(self):
        text = "intro {\"merges\": [{\"into\":\"x\"}]} outro"
        out = _parse_merges_payload(text)
        assert out == {"merges": [{"into": "x"}]}

    def test_garbage(self):
        assert _parse_merges_payload("nope") is None


# ── _coerce_merges ────────────────────────────────────────────────────


class TestCoerce:
    def test_valid_merge(self):
        ms = _coerce_merges({"merges": [{
            "into": "signing", "from": ["a", "b"],
            "rationale": "same domain"
        }]})
        assert len(ms) == 1
        assert ms[0].into == "signing"
        assert ms[0].sources == ["a", "b"]
        assert ms[0].rationale == "same domain"

    def test_drops_short_source_list(self):
        ms = _coerce_merges({"merges": [{
            "into": "x", "from": ["only-one"], "rationale": "y"
        }]})
        assert ms == []

    def test_drops_empty_rationale(self):
        ms = _coerce_merges({"merges": [{
            "into": "x", "from": ["a", "b"], "rationale": "  "
        }]})
        assert ms == []

    def test_drops_empty_into(self):
        ms = _coerce_merges({"merges": [{
            "into": "", "from": ["a", "b"], "rationale": "y"
        }]})
        assert ms == []

    def test_dedups_sources_preserves_order(self):
        ms = _coerce_merges({"merges": [{
            "into": "x", "from": ["a", "b", "a", "c"], "rationale": "y"
        }]})
        assert ms[0].sources == ["a", "b", "c"]

    def test_caps_at_max_merges(self):
        many = {"merges": [
            {"into": f"x{i}", "from": ["a", "b"], "rationale": "y"}
            for i in range(MAX_MERGES_PER_PASS + 5)
        ]}
        ms = _coerce_merges(many)
        assert len(ms) == MAX_MERGES_PER_PASS

    def test_garbage_input_returns_empty(self):
        assert _coerce_merges(None) == []
        assert _coerce_merges({"merges": "not a list"}) == []
        assert _coerce_merges({}) == []


# ── _apply_merges ─────────────────────────────────────────────────────


class TestApplyMerges:
    def test_simple_two_feature_merge(self):
        r = DeepScanResult(features={
            "a": ["x.ts"], "b": ["y.ts"],
        })
        merge = Merge(into="ab", sources=["a", "b"],
                      description="combined", rationale="same")
        out, applied = _apply_merges(r, [merge])
        assert "a" not in out.features
        assert "b" not in out.features
        assert sorted(out.features["ab"]) == ["x.ts", "y.ts"]
        assert len(applied) == 1
        assert "merged" in out.descriptions["ab"]

    def test_into_can_be_one_of_sources(self):
        r = DeepScanResult(features={
            "lib/signing": ["a.ts"],
            "remix/signing": ["b.ts"],
        })
        merge = Merge(
            into="lib/signing",
            sources=["lib/signing", "remix/signing"],
            rationale="x",
        )
        out, _ = _apply_merges(r, [merge])
        assert "remix/signing" not in out.features
        assert sorted(out.features["lib/signing"]) == ["a.ts", "b.ts"]

    def test_five_feature_merge(self):
        r = _result_with_signing()
        # Pretend lib/remix/trpc all merge into "signing"
        merge = Merge(
            into="signing",
            sources=["lib/document-signing", "remix/document-signing", "trpc/document-signing"],
            description="Document signing across layers.",
            rationale="three packages all implement signing",
        )
        out, applied = _apply_merges(r, [merge])
        assert "signing" in out.features
        assert len(out.features["signing"]) == 7
        # ui and documentation untouched
        assert "ui/design-system-primitives" in out.features
        assert "documentation" in out.features
        assert len(applied) == 1

    def test_missing_source_silently_skipped(self):
        r = DeepScanResult(features={"a": ["x.ts"], "b": ["y.ts"]})
        merge = Merge(
            into="ab",
            sources=["a", "b", "c-doesnt-exist"],
            rationale="same",
        )
        out, applied = _apply_merges(r, [merge])
        assert sorted(out.features["ab"]) == ["x.ts", "y.ts"]
        assert len(applied) == 1

    def test_skipped_when_under_2_valid_sources(self):
        r = DeepScanResult(features={"a": ["x.ts"]})
        merge = Merge(
            into="ab",
            sources=["a", "b-missing"],
            rationale="same",
        )
        out, applied = _apply_merges(r, [merge])
        assert "ab" not in out.features
        assert "a" in out.features
        assert applied == []

    def test_protected_target_refused(self):
        r = DeepScanResult(features={
            "a": ["x.ts"], "b": ["y.ts"], "documentation": ["d.md"],
        })
        merge = Merge(
            into="documentation",
            sources=["a", "b"],
            rationale="bad",
        )
        out, applied = _apply_merges(r, [merge])
        assert applied == []
        assert "a" in out.features and "b" in out.features
        assert out.features["documentation"] == ["d.md"]

    def test_protected_source_skipped(self):
        r = DeepScanResult(features={
            "a": ["x.ts"], "documentation": ["d.md"],
        })
        merge = Merge(
            into="ax",
            sources=["a", "documentation"],
            rationale="bad",
        )
        out, applied = _apply_merges(r, [merge])
        # documentation never participates → only "a" remains as source
        # → fewer than 2 valid → merge skipped
        assert applied == []
        assert "documentation" in out.features

    def test_flows_unioned_across_sources(self):
        r = _result_with_signing()
        merge = Merge(
            into="signing",
            sources=["lib/document-signing", "remix/document-signing"],
            rationale="x",
        )
        out, _ = _apply_merges(r, [merge])
        assert "sign-document" in out.flows["signing"]
        assert "open-signing-page" in out.flows["signing"]
        # flow_descriptions carried
        assert out.flow_descriptions["signing"]["sign-document"] == "user signs"

    def test_description_uses_model_supplied_when_present(self):
        r = DeepScanResult(
            features={"a": ["x.ts"], "b": ["y.ts"]},
            descriptions={"a": "old long description here", "b": "short"},
        )
        merge = Merge(
            into="ab", sources=["a", "b"],
            description="The new combined description.",
            rationale="y",
        )
        out, _ = _apply_merges(r, [merge])
        assert "The new combined description." in out.descriptions["ab"]

    def test_description_falls_back_to_longest_source(self):
        r = DeepScanResult(
            features={"a": ["x.ts"], "b": ["y.ts"]},
            descriptions={"a": "tiny", "b": "much longer description text"},
        )
        merge = Merge(into="ab", sources=["a", "b"], rationale="y")
        out, _ = _apply_merges(r, [merge])
        assert "much longer description text" in out.descriptions["ab"]

    def test_merge_trail_preserved_in_description(self):
        r = DeepScanResult(features={"a": ["x"], "b": ["y"]})
        merge = Merge(into="c", sources=["a", "b"], rationale="z")
        out, _ = _apply_merges(r, [merge])
        assert "(merged: a, b)" in out.descriptions["c"]

    def test_no_double_trail_when_into_is_a_source(self):
        r = DeepScanResult(features={"a": ["x"], "b": ["y"]})
        merge = Merge(into="a", sources=["a", "b"], rationale="z")
        out, _ = _apply_merges(r, [merge])
        # a is target, only b in trail
        assert "(merged: b)" in out.descriptions["a"]
        assert "(merged: a" not in out.descriptions["a"]

    def test_empty_merges_list_is_noop(self):
        r = DeepScanResult(features={"a": ["x.ts"]})
        out, applied = _apply_merges(r, [])
        assert applied == []
        assert "a" in out.features


# ── dedup_features (with fake client) ─────────────────────────────────


class TestDedupFeatures:
    def test_skips_when_no_features(self):
        r = DeepScanResult(features={})
        client = FakeClient([])
        out = dedup_features(r, client=client)
        assert out is r
        assert client.messages.calls == []

    def test_skips_when_only_one_feature_after_protection_filter(self):
        r = DeepScanResult(features={"a": ["x.ts"], "documentation": ["d.md"]})
        client = FakeClient([])
        out = dedup_features(r, client=client)
        assert client.messages.calls == []
        assert "a" in out.features

    def test_applies_one_merge(self):
        r = _result_with_signing()
        client = FakeClient([_resp(
            '{"merges": [{"into": "signing", '
            '"from": ["lib/document-signing", "remix/document-signing", "trpc/document-signing"], '
            '"description": "Combined.", "rationale": "all signing"}]}'
        )])
        out = dedup_features(r, client=client)
        assert "signing" in out.features
        assert "lib/document-signing" not in out.features
        assert "remix/document-signing" not in out.features
        assert "trpc/document-signing" not in out.features
        # protected and unrelated untouched
        assert "documentation" in out.features
        assert "ui/design-system-primitives" in out.features

    def test_empty_merges_response_is_noop(self):
        r = _result_with_signing()
        before = sorted(r.features.keys())
        client = FakeClient([_resp('{"merges": []}')])
        out = dedup_features(r, client=client)
        assert sorted(out.features.keys()) == before

    def test_unparseable_response_returns_original(self):
        r = _result_with_signing()
        before = sorted(r.features.keys())
        client = FakeClient([_resp("not json at all")])
        out = dedup_features(r, client=client)
        assert sorted(out.features.keys()) == before

    def test_anthropic_exception_returns_original(self):
        r = _result_with_signing()
        before = sorted(r.features.keys())
        client = FakeClient([RuntimeError("boom")])
        out = dedup_features(r, client=client)
        assert sorted(out.features.keys()) == before

    def test_tracker_records_call(self):
        from faultline.llm.cost import CostTracker

        r = _result_with_signing()
        tracker = CostTracker()
        client = FakeClient([_resp('{"merges": []}', usage_in=500, usage_out=120)])
        dedup_features(r, client=client, tracker=tracker)
        s = tracker.summary()
        assert s["total_calls"] == 1
        assert s["total_input_tokens"] == 500
        assert s["total_output_tokens"] == 120

    def test_summary_input_excludes_protected(self):
        r = _result_with_signing()
        client = FakeClient([_resp('{"merges": []}')])
        dedup_features(r, client=client)
        # Inspect what was sent to the LLM
        user_msg = client.messages.calls[0]["messages"][0]["content"]
        assert "documentation" not in user_msg
        assert "lib/document-signing" in user_msg
