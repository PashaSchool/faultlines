"""Unit tests for the benchmark harness (Sprint 6 Day 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.benchmark.loader import (
    ExpectedAttribution,
    ExpectedFeature,
    ExpectedFlow,
    load_benchmark,
    load_expected_attribution,
    load_expected_features,
    load_expected_flows,
)
from faultline.benchmark.metrics import (
    GENERIC_NAME_BLOCKLIST,
    attribution_accuracy,
    feature_precision,
    feature_recall,
    flow_recall,
    generic_name_rate,
)
from faultline.benchmark.report import render_markdown, score


# ── Loader: features ─────────────────────────────────────────────────


class TestLoadFeatures:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "f.yaml"
        p.write_text(
            "features:\n"
            "  - name: auth\n"
            "    description: User auth\n"
            "    aliases: [api/auth, user-auth]\n"
            "    must_include: [a.ts, b.ts]\n",
            encoding="utf-8",
        )
        out = load_expected_features(p)
        assert len(out) == 1
        assert out[0].name == "auth"
        assert out[0].aliases == ("api/auth", "user-auth")
        assert out[0].must_include == ("a.ts", "b.ts")
        assert "auth" in out[0].all_names

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_expected_features(tmp_path / "nope.yaml")

    def test_top_level_must_be_mapping(self, tmp_path: Path):
        p = tmp_path / "f.yaml"
        p.write_text("- just a list\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_expected_features(p)

    def test_missing_name_rejected(self, tmp_path: Path):
        p = tmp_path / "f.yaml"
        p.write_text("features:\n  - description: no name\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'name'"):
            load_expected_features(p)

    def test_duplicate_name_rejected(self, tmp_path: Path):
        p = tmp_path / "f.yaml"
        p.write_text(
            "features:\n  - name: x\n  - name: x\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_expected_features(p)

    def test_empty_yaml_returns_empty_list(self, tmp_path: Path):
        p = tmp_path / "f.yaml"
        p.write_text("", encoding="utf-8")
        assert load_expected_features(p) == []


class TestLoadFlows:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "fl.yaml"
        p.write_text(
            "flows:\n"
            "  - name: create-document\n"
            "    feature: signing\n"
            "    description: User uploads\n",
            encoding="utf-8",
        )
        out = load_expected_flows(p)
        assert len(out) == 1
        assert out[0].name == "create-document"
        assert out[0].feature == "signing"


class TestLoadAttribution:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "a.yaml"
        p.write_text(
            "samples:\n"
            "  - path: src/a.ts\n"
            "    expected: auth\n",
            encoding="utf-8",
        )
        out = load_expected_attribution(p)
        assert out == [ExpectedAttribution(path="src/a.ts", expected="auth")]

    def test_missing_field_rejected(self, tmp_path: Path):
        p = tmp_path / "a.yaml"
        p.write_text("samples:\n  - path: src/a.ts\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_expected_attribution(p)

    def test_duplicate_path_rejected(self, tmp_path: Path):
        p = tmp_path / "a.yaml"
        p.write_text(
            "samples:\n"
            "  - {path: x.ts, expected: a}\n"
            "  - {path: x.ts, expected: b}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_expected_attribution(p)


class TestLoadBenchmark:
    def test_partial_files_tolerated(self, tmp_path: Path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        (repo_dir / "expected_features.yaml").write_text(
            "features:\n  - name: auth\n", encoding="utf-8",
        )
        # No flows or attribution yaml
        spec = load_benchmark("myrepo", tmp_path)
        assert len(spec.features) == 1
        assert spec.flows == []
        assert spec.attribution == []


# ── Metrics: feature recall ──────────────────────────────────────────


class TestFeatureRecall:
    def test_perfect(self):
        e = [ExpectedFeature(name="auth"), ExpectedFeature(name="billing")]
        assert feature_recall(e, {"auth": [], "billing": []}) == 1.0

    def test_half(self):
        e = [ExpectedFeature(name="auth"), ExpectedFeature(name="billing")]
        assert feature_recall(e, {"auth": []}) == 0.5

    def test_zero(self):
        e = [ExpectedFeature(name="auth")]
        assert feature_recall(e, {"phantom": []}) == 0.0

    def test_alias_match(self):
        e = [ExpectedFeature(name="auth", aliases=("api/auth",))]
        assert feature_recall(e, {"api/auth": []}) == 1.0

    def test_empty_expected(self):
        assert feature_recall([], {"x": []}) == 0.0

    def test_prefix_match_subdecomposition(self):
        # Sprint 3 splits document-signing into 3 sub-features —
        # the canonical name should still count as found.
        e = [ExpectedFeature(name="document-signing")]
        detected = {
            "document-signing/recipient-signing-experience": [],
            "document-signing/pdf-sealing-certification": [],
        }
        assert feature_recall(e, detected) == 1.0

    def test_token_set_word_order(self):
        # team-and-organisation-management ≡ organisation-and-team-management
        e = [ExpectedFeature(name="organisation-and-team-management")]
        assert feature_recall(e, {"team-and-organisation-management": []}) == 1.0

    def test_token_set_stop_words(self):
        # "user-authentication" (expected) vs "authentication" (detected)
        e = [ExpectedFeature(name="user-authentication")]
        assert feature_recall(e, {"authentication": []}) == 1.0

    def test_no_match_genuinely_different(self):
        # Different domain words → not a match
        e = [ExpectedFeature(name="billing")]
        assert feature_recall(e, {"signing": []}) == 0.0

    def test_token_subset_two_meaningful_tokens(self):
        # expected tokens after stop-word filter: {envelope, document}
        # detected last-segment tokens: {envelope, document, billing}
        # subset matches (≥2 expected tokens)
        e = [ExpectedFeature(name="envelope-document-management")]
        assert feature_recall(e, {"envelope-document-billing": []}) == 1.0

    def test_subset_single_token_expected_does_NOT_match(self):
        # Expected "email" has only one non-stop token; subset
        # match could falsely include any feature containing "email".
        # User must add an explicit alias instead.
        e = [ExpectedFeature(name="email")]
        assert feature_recall(e, {"organisation-email-domains": []}) == 0.0

    def test_subset_one_way_detected_coarser_does_not_match(self):
        # User expected user-payment-billing (tokens: {payment, billing}),
        # engine produced just "billing" (tokens: {billing}).
        # Detected ⊊ expected — should NOT match (recall would
        # over-credit a partial detection).
        e = [ExpectedFeature(name="user-payment-billing")]
        assert feature_recall(e, {"billing": []}) == 0.0

    def test_subset_two_token_envelope_management(self):
        # When the user gives a 2+-token canonical, subset works.
        # expected: envelope-management → tokens {envelope}
        # Hmm — "management" is a stop word, leaving {envelope}.
        # That's only 1 token, so this should NOT match by subset.
        # User would need an explicit alias for envelope-management.
        e = [ExpectedFeature(name="envelope-management")]
        assert feature_recall(e, {"document-and-envelopes": []}) == 0.0


# ── Metrics: feature precision ───────────────────────────────────────


class TestFeaturePrecision:
    def test_perfect(self):
        e = [ExpectedFeature(name="auth"), ExpectedFeature(name="billing")]
        assert feature_precision(e, {"auth": [], "billing": []}) == 1.0

    def test_phantom_drops_precision(self):
        e = [ExpectedFeature(name="auth")]
        assert feature_precision(e, {"auth": [], "phantom": []}) == 0.5

    def test_excludes_synthetic(self):
        e = [ExpectedFeature(name="auth")]
        # documentation/shared-infra excluded → just auth counts
        result = feature_precision(
            e, {"auth": [], "documentation": [], "shared-infra": []},
        )
        assert result == 1.0

    def test_alias_counts(self):
        e = [ExpectedFeature(name="auth", aliases=("api/auth",))]
        assert feature_precision(e, {"api/auth": []}) == 1.0

    def test_empty_detected(self):
        assert feature_precision([ExpectedFeature(name="x")], {}) == 0.0


# ── Metrics: flow recall ─────────────────────────────────────────────


class TestFlowRecall:
    def test_basic(self):
        e = [ExpectedFlow(name="create-document"),
             ExpectedFlow(name="cancel-subscription")]
        assert flow_recall(e, {"f": ["create-document"]}) == 0.5

    def test_normalize_dehyphen(self):
        e = [ExpectedFlow(name="create-document")]
        assert flow_recall(e, ["create_document"]) == 1.0
        assert flow_recall(e, ["Create Document"]) == 1.0

    def test_empty_expected(self):
        assert flow_recall([], {"f": ["x"]}) == 0.0


# ── Metrics: attribution accuracy ────────────────────────────────────


class TestAttribution:
    def test_perfect(self):
        samples = [
            ExpectedAttribution(path="a.ts", expected="auth"),
            ExpectedAttribution(path="b.ts", expected="billing"),
        ]
        assert attribution_accuracy(
            samples, {"auth": ["a.ts"], "billing": ["b.ts"]}
        ) == 1.0

    def test_via_alias(self):
        samples = [ExpectedAttribution(path="a.ts", expected="auth")]
        feats = [ExpectedFeature(name="auth", aliases=("api/auth",))]
        # detected uses alias, but expected uses canonical
        assert attribution_accuracy(
            samples, {"api/auth": ["a.ts"]}, expected_features=feats,
        ) == 1.0

    def test_missing_file_drops(self):
        samples = [ExpectedAttribution(path="missing.ts", expected="auth")]
        assert attribution_accuracy(samples, {"auth": ["a.ts"]}) == 0.0

    def test_empty_samples_zero(self):
        assert attribution_accuracy([], {"x": ["a.ts"]}) == 0.0


# ── Metrics: generic-name rate ───────────────────────────────────────


class TestGenericNameRate:
    def test_blocklist_entries_count(self):
        rate = generic_name_rate({"auth": [], "lib": [], "billing": []})
        assert 0.32 < rate < 0.34

    def test_excludes_synthetic(self):
        # documentation / shared-infra / examples are not counted
        rate = generic_name_rate({
            "auth": [], "billing": [],
            "documentation": [], "shared-infra": [], "examples": [],
        })
        assert rate == 0.0

    def test_path_segment_match(self):
        # last segment "lib" is generic
        rate = generic_name_rate({"app/lib": [], "app/auth": []})
        assert rate == 0.5

    def test_empty(self):
        assert generic_name_rate({}) == 0.0

    def test_all_clean(self):
        assert generic_name_rate({"auth": [], "billing": []}) == 0.0


# ── Report rendering ─────────────────────────────────────────────────


class TestReport:
    def test_single_repo(self):
        from faultline.benchmark.loader import BenchmarkSpec
        spec = BenchmarkSpec(
            repo="myrepo",
            features=[ExpectedFeature(name="auth"),
                      ExpectedFeature(name="billing")],
        )
        s = score(spec, {"auth": ["a.ts"], "billing": ["b.ts"]})
        md = render_markdown([s])
        assert "myrepo" in md
        assert "100.0%" in md
        assert "Overall scorecard" not in md  # only one repo

    def test_multi_repo_overall(self):
        from faultline.benchmark.loader import BenchmarkSpec
        s1 = score(
            BenchmarkSpec(repo="r1", features=[ExpectedFeature(name="x")]),
            {"x": []},
        )
        s2 = score(
            BenchmarkSpec(repo="r2", features=[ExpectedFeature(name="y")]),
            {"phantom": []},
        )
        md = render_markdown([s1, s2])
        assert "Overall scorecard" in md
        assert "r1" in md and "r2" in md

    def test_empty(self):
        assert "No scores" in render_markdown([])
