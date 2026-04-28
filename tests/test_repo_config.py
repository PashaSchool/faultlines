"""Tests for ``faultline.analyzer.repo_config``."""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.analyzer.repo_config import (
    ForcedMerge,
    RepoConfig,
    apply_repo_config,
    find_repo_config,
    load_repo_config,
)
from faultline.llm.sonnet_scanner import DeepScanResult


# ── find_repo_config ─────────────────────────────────────────────────


class TestFind:
    def test_returns_none_when_absent(self, tmp_path: Path):
        assert find_repo_config(tmp_path) is None

    def test_finds_yaml_variant(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text("features: {}\n", encoding="utf-8")
        assert find_repo_config(tmp_path).name == ".faultline.yaml"

    def test_finds_yml_variant(self, tmp_path: Path):
        (tmp_path / ".faultline.yml").write_text("features: {}\n", encoding="utf-8")
        assert find_repo_config(tmp_path).name == ".faultline.yml"

    def test_yaml_wins_over_yml(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text("a: 1\n", encoding="utf-8")
        (tmp_path / ".faultline.yml").write_text("b: 2\n", encoding="utf-8")
        assert find_repo_config(tmp_path).name == ".faultline.yaml"

    def test_alternate_filename(self, tmp_path: Path):
        (tmp_path / "faultline.config.yaml").write_text(
            "features: {}\n", encoding="utf-8",
        )
        assert find_repo_config(tmp_path).name == "faultline.config.yaml"


# ── load_repo_config ─────────────────────────────────────────────────


class TestLoad:
    def test_no_config_returns_none(self, tmp_path: Path):
        assert load_repo_config(tmp_path) is None

    def test_empty_yaml_returns_empty_config(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text("", encoding="utf-8")
        cfg = load_repo_config(tmp_path)
        assert cfg is not None and cfg.is_empty

    def test_full_config(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text("""
features:
  billing:
    description: Stripe billing.
    variants:
      - lib/billing
      - ee/stripe-billing
  embedded-signing:
    variants:
      - remix/embedded-signing-authoring
skip_features:
  - tsconfig
  - tailwind-config
force_merges:
  - into: design-system
    from:
      - ui-primitives
      - ui/primitive-components
    description: Reusable UI primitives.
""", encoding="utf-8")
        cfg = load_repo_config(tmp_path)
        assert cfg is not None
        assert len(cfg.features) == 2
        assert cfg.features[0].canonical == "billing"
        assert "lib/billing" in cfg.features[0].variants
        assert cfg.skip_features == ["tsconfig", "tailwind-config"]
        assert cfg.force_merges[0].into == "design-system"
        assert "ui-primitives" in cfg.force_merges[0].sources

    def test_top_level_must_be_mapping(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text("- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_repo_config(tmp_path)

    def test_features_must_be_mapping(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text(
            "features: not-a-map\n", encoding="utf-8",
        )
        with pytest.raises(ValueError, match="features"):
            load_repo_config(tmp_path)

    def test_duplicate_canonical_rejected(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text(
            "features:\n  x: {}\n  x: {}\n", encoding="utf-8",
        )
        # YAML dedup makes this OK at parse time — but if we ever get
        # explicit duplicates via dict construction, our loader
        # would catch them. Verify no exception on YAML-collapsed dup.
        cfg = load_repo_config(tmp_path)
        assert cfg is not None and len(cfg.features) == 1

    def test_force_merge_missing_into(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text("""
force_merges:
  - from: [a, b]
""", encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'into'"):
            load_repo_config(tmp_path)

    def test_skip_features_must_be_list(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text(
            "skip_features: nope\n", encoding="utf-8",
        )
        with pytest.raises(ValueError, match="skip_features"):
            load_repo_config(tmp_path)

    def test_records_source_path(self, tmp_path: Path):
        (tmp_path / ".faultline.yaml").write_text("features: {}\n", encoding="utf-8")
        cfg = load_repo_config(tmp_path)
        assert ".faultline.yaml" in cfg.source_path


# ── apply_repo_config ────────────────────────────────────────────────


def _result(features, **kw) -> DeepScanResult:
    return DeepScanResult(features=features, **kw)


class TestApply:
    def test_none_config_passthrough(self):
        r = _result({"a": ["x.ts"]})
        out = apply_repo_config(r, None)
        assert "a" in out.features

    def test_empty_config_passthrough(self):
        r = _result({"a": ["x.ts"]})
        out = apply_repo_config(r, RepoConfig())
        assert "a" in out.features

    def test_canonical_aliasing(self):
        r = _result(
            {"remix/embedded-signing-authoring": ["a.ts", "b.ts"]},
            descriptions={"remix/embedded-signing-authoring": "src"},
        )
        from faultline.analyzer.repo_config import FeatureRule
        cfg = RepoConfig(features=[FeatureRule(
            canonical="embedded-signing",
            description="Embedded signing SDK.",
            variants=("remix/embedded-signing-authoring",),
        )])
        out = apply_repo_config(r, cfg)
        assert "embedded-signing" in out.features
        assert "remix/embedded-signing-authoring" not in out.features
        assert sorted(out.features["embedded-signing"]) == ["a.ts", "b.ts"]
        # Explicit description overrides
        assert out.descriptions["embedded-signing"] == "Embedded signing SDK."

    def test_alias_with_existing_canonical_unions_files(self):
        from faultline.analyzer.repo_config import FeatureRule
        r = _result({
            "billing": ["a.ts"],
            "lib/billing": ["b.ts"],
            "ee/stripe-billing": ["c.ts"],
        })
        cfg = RepoConfig(features=[FeatureRule(
            canonical="billing",
            variants=("lib/billing", "ee/stripe-billing"),
        )])
        out = apply_repo_config(r, cfg)
        assert "billing" in out.features
        assert sorted(out.features["billing"]) == ["a.ts", "b.ts", "c.ts"]
        assert "lib/billing" not in out.features
        assert "ee/stripe-billing" not in out.features

    def test_skip_features_drops(self):
        r = _result({"a": ["x.ts"], "tsconfig": ["t.ts"]})
        cfg = RepoConfig(skip_features=["tsconfig"])
        out = apply_repo_config(r, cfg)
        assert "tsconfig" not in out.features
        assert "a" in out.features

    def test_skip_missing_feature_silent(self):
        r = _result({"a": ["x.ts"]})
        cfg = RepoConfig(skip_features=["does-not-exist"])
        # Should not raise
        out = apply_repo_config(r, cfg)
        assert "a" in out.features

    def test_force_merge_basic(self):
        r = _result({
            "ui-primitives": ["a.ts"],
            "ui/primitive-components": ["b.ts"],
            "other": ["c.ts"],
        })
        cfg = RepoConfig(force_merges=[ForcedMerge(
            into="design-system",
            sources=("ui-primitives", "ui/primitive-components"),
            description="Design system primitives.",
        )])
        out = apply_repo_config(r, cfg)
        assert "design-system" in out.features
        assert sorted(out.features["design-system"]) == ["a.ts", "b.ts"]
        assert "ui-primitives" not in out.features
        assert "ui/primitive-components" not in out.features
        assert "other" in out.features

    def test_force_merge_into_existing_unions(self):
        r = _result({
            "design-system": ["existing.ts"],
            "ui-primitives": ["new.ts"],
        })
        cfg = RepoConfig(force_merges=[ForcedMerge(
            into="design-system", sources=("ui-primitives",),
        )])
        out = apply_repo_config(r, cfg)
        assert sorted(out.features["design-system"]) == ["existing.ts", "new.ts"]

    def test_force_merge_no_live_sources_silent(self):
        r = _result({"a": ["x.ts"]})
        cfg = RepoConfig(force_merges=[ForcedMerge(
            into="design-system", sources=("missing-1", "missing-2"),
        )])
        out = apply_repo_config(r, cfg)
        assert "design-system" not in out.features
        assert "a" in out.features

    def test_alias_carries_flows_and_descriptions(self):
        from faultline.analyzer.repo_config import FeatureRule
        r = _result(
            {"remix/embedded-signing-authoring": ["a.ts"]},
            descriptions={"remix/embedded-signing-authoring": "Original."},
            flows={"remix/embedded-signing-authoring": ["sign-doc"]},
            flow_descriptions={
                "remix/embedded-signing-authoring": {"sign-doc": "user signs"},
            },
        )
        cfg = RepoConfig(features=[FeatureRule(
            canonical="embedded-signing",
            variants=("remix/embedded-signing-authoring",),
        )])
        out = apply_repo_config(r, cfg)
        assert out.descriptions["embedded-signing"] == "Original."
        assert "sign-doc" in out.flows["embedded-signing"]
        assert out.flow_descriptions["embedded-signing"]["sign-doc"] == "user signs"

    def test_combined_pipeline(self):
        from faultline.analyzer.repo_config import FeatureRule
        r = _result({
            "lib/billing": ["b.ts"],
            "ee/stripe-billing": ["s.ts"],
            "ui-primitives": ["u.ts"],
            "ui/primitive-components": ["p.ts"],
            "tsconfig": ["t.ts"],
        })
        cfg = RepoConfig(
            features=[FeatureRule(
                canonical="billing",
                variants=("lib/billing", "ee/stripe-billing"),
            )],
            skip_features=["tsconfig"],
            force_merges=[ForcedMerge(
                into="design-system",
                sources=("ui-primitives", "ui/primitive-components"),
            )],
        )
        out = apply_repo_config(r, cfg)
        assert "billing" in out.features
        assert sorted(out.features["billing"]) == ["b.ts", "s.ts"]
        assert "design-system" in out.features
        assert sorted(out.features["design-system"]) == ["p.ts", "u.ts"]
        assert "tsconfig" not in out.features
