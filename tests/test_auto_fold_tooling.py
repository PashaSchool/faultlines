"""Tests for the built-in auto-fold of universally-tooling packages.

The pipeline folds packages whose name matches
``TOOLING_PACKAGE_NAMES`` into ``shared-infra`` automatically — no
user config required. This file pins the behaviour with focused
unit tests around ``_auto_fold_tooling``.
"""

from __future__ import annotations

from faultline.llm.pipeline import _auto_fold_tooling
from faultline.llm.sonnet_scanner import DeepScanResult


def _r(features, **kw) -> DeepScanResult:
    return DeepScanResult(features=features, **kw)


class TestAutoFoldTooling:
    def test_folds_tsconfig_into_shared_infra(self):
        r = _r({
            "tsconfig": ["packages/tsconfig/base.json"],
            "auth": ["a.ts"],
        })
        _auto_fold_tooling(r)
        assert "tsconfig" not in r.features
        assert "auth" in r.features
        assert "packages/tsconfig/base.json" in r.features["shared-infra"]

    def test_folds_multiple_tooling_packages(self):
        r = _r({
            "tsconfig": ["a.json"],
            "eslint-config": ["b.js"],
            "prettier-config": ["c.json"],
            "tailwind-config": ["d.ts"],
            "auth": ["x.ts"],
        })
        _auto_fold_tooling(r)
        assert set(r.features.keys()) == {"shared-infra", "auth"}
        assert sorted(r.features["shared-infra"]) == [
            "a.json", "b.js", "c.json", "d.ts",
        ]

    def test_folds_alternate_naming_config_typescript(self):
        # turborepo / next-forge use this shape
        r = _r({"config-typescript": ["a.json"], "auth": ["x.ts"]})
        _auto_fold_tooling(r)
        assert "config-typescript" not in r.features
        assert "a.json" in r.features["shared-infra"]

    def test_unions_with_existing_shared_infra(self):
        r = _r({
            "shared-infra": ["existing-config.yml"],
            "tsconfig": ["new.json"],
        })
        _auto_fold_tooling(r)
        assert sorted(r.features["shared-infra"]) == [
            "existing-config.yml", "new.json",
        ]

    def test_dedups_path_list(self):
        r = _r({
            "shared-infra": ["x.json", "x.json"],
            "tsconfig": ["x.json"],  # duplicate of existing
        })
        _auto_fold_tooling(r)
        assert r.features["shared-infra"] == ["x.json"]

    def test_no_match_is_noop(self):
        r = _r({"auth": ["a.ts"], "billing": ["b.ts"]})
        _auto_fold_tooling(r)
        assert "auth" in r.features
        assert "billing" in r.features
        assert "shared-infra" not in r.features

    def test_drops_descriptions_and_flows_for_folded(self):
        r = _r(
            {"tsconfig": ["a.json"]},
            descriptions={"tsconfig": "TS config"},
            flows={"tsconfig": ["build-flow"]},
            flow_descriptions={"tsconfig": {"build-flow": "x"}},
        )
        _auto_fold_tooling(r)
        assert "tsconfig" not in r.descriptions
        assert "tsconfig" not in r.flows
        assert "tsconfig" not in r.flow_descriptions

    def test_path_segment_match_only(self):
        # A feature named "lib/tsconfig" — last segment "tsconfig"
        # is tooling, so we still fold it.
        r = _r({"lib/tsconfig": ["a.json"]})
        _auto_fold_tooling(r)
        assert "lib/tsconfig" not in r.features
        assert "shared-infra" in r.features

    def test_substring_does_not_match(self):
        # "tsconfig-shared" is a different name, not folded
        r = _r({"tsconfig-shared": ["a.ts"]})
        _auto_fold_tooling(r)
        assert "tsconfig-shared" in r.features
        assert "shared-infra" not in r.features

    def test_empty_input_is_noop(self):
        r = _r({})
        _auto_fold_tooling(r)
        assert r.features == {}
