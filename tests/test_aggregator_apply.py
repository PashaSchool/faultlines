"""Sprint 8 Day 4 — apply classifications to a scan result."""

from __future__ import annotations

from faultline.llm.aggregator_apply import (
    DEV_INFRA_BUCKET,
    _delete_feature,
    _fold_to_bucket,
    _rename_feature,
    apply_classifications,
)
from faultline.llm.aggregator_detector import FeatureClassification
from faultline.llm.sonnet_scanner import DeepScanResult


def _ds(**kwargs) -> DeepScanResult:
    return DeepScanResult(
        features=kwargs.get("features", {}),
        flows=kwargs.get("flows", {}),
        descriptions=kwargs.get("descriptions", {}),
        flow_descriptions=kwargs.get("flow_descriptions", {}),
        flow_participants=kwargs.get("flow_participants", {}),
    )


def _cls(**kwargs) -> FeatureClassification:
    return FeatureClassification(
        feature_name=kwargs["feature_name"],
        classification=kwargs["classification"],
        confidence=kwargs.get("confidence", 5),
        reasoning=kwargs.get("reasoning", ""),
        consumer_features=kwargs.get("consumer_features"),
        proposed_name=kwargs.get("proposed_name"),
    )


# ── _delete_feature ──────────────────────────────────────────────────


class TestDeleteFeature:
    def test_removes_from_every_side_channel(self):
        result = _ds(
            features={"Dto": ["a.ts"], "Auth": ["b.ts"]},
            flows={"Dto": ["x-flow"], "Auth": ["y-flow"]},
            descriptions={"Dto": "DTO desc", "Auth": "Auth desc"},
            flow_descriptions={"Dto": {"x-flow": "..."}},
            flow_participants={"Dto": {"x-flow": []}},
        )
        _delete_feature(result, "Dto")
        assert "Dto" not in result.features
        assert "Dto" not in result.flows
        assert "Dto" not in result.descriptions
        assert "Dto" not in result.flow_descriptions
        assert "Dto" not in result.flow_participants
        # Untouched features remain
        assert "Auth" in result.features


# ── _rename_feature ──────────────────────────────────────────────────


class TestRenameFeature:
    def test_renames_across_all_channels(self):
        result = _ds(
            features={"i18n": ["locales/en.json"]},
            flows={"i18n": []},
            descriptions={"i18n": "Locale files"},
        )
        ok = _rename_feature(result, "i18n", "Translations")
        assert ok is True
        assert "Translations" in result.features
        assert "i18n" not in result.features
        assert result.descriptions["Translations"] == "Locale files"

    def test_collision_returns_false_no_data_loss(self):
        result = _ds(
            features={"i18n": ["a.json"], "Translations": ["b.json"]},
        )
        ok = _rename_feature(result, "i18n", "Translations")
        assert ok is False
        # Both still present — caller will decide what to do
        assert "i18n" in result.features
        assert "Translations" in result.features

    def test_missing_source_returns_false(self):
        result = _ds(features={"Auth": ["a.ts"]})
        assert _rename_feature(result, "MissingFeature", "Whatever") is False

    def test_no_op_on_same_name(self):
        result = _ds(features={"Auth": ["a.ts"]})
        assert _rename_feature(result, "Auth", "Auth") is True
        assert result.features["Auth"] == ["a.ts"]


# ── _fold_to_bucket ──────────────────────────────────────────────────


class TestFoldToBucket:
    def test_creates_bucket_and_dedupes(self):
        result = _ds(features={
            "Tooling Configs": ["a.config.ts", "b.config.ts"],
            "shared-infra": ["existing.ts"],
        })
        _fold_to_bucket(result, "Tooling Configs", "shared-infra")
        assert "Tooling Configs" not in result.features
        assert sorted(result.features["shared-infra"]) == [
            "a.config.ts", "b.config.ts", "existing.ts",
        ]


# ── apply_classifications: aggregator path ───────────────────────────


class TestAggregatorRedistribution:
    def test_dto_redistributes_to_consumers_and_disappears(self):
        result = _ds(
            features={
                "Dto": [
                    "packages/api-types/dto/auth/login.dto.ts",
                    "packages/api-types/dto/billing/plan.dto.ts",
                ],
                "Authentication": ["src/auth/login.tsx"],
                "Billing": ["src/billing/plan.tsx"],
            },
            flows={"Dto": []},
            descriptions={"Dto": "Multi-domain DTOs"},
        )
        classifications = {
            "Dto": _cls(
                feature_name="Dto",
                classification="shared-aggregator",
                confidence=5,
                consumer_features=["Authentication", "Billing"],
            ),
        }
        consumer_maps = {
            "Dto": {
                "packages/api-types/dto/auth/login.dto.ts": ["Authentication"],
                "packages/api-types/dto/billing/plan.dto.ts": ["Billing"],
            },
        }

        apply_classifications(result, classifications, consumer_maps)

        # Dto bucket gone
        assert "Dto" not in result.features
        # Each consumer received its participant
        auth_paths = [
            sp.file_path for sp in result.shared_participants_map["Authentication"]
        ]
        bill_paths = [
            sp.file_path for sp in result.shared_participants_map["Billing"]
        ]
        assert "packages/api-types/dto/auth/login.dto.ts" in auth_paths
        assert "packages/api-types/dto/billing/plan.dto.ts" in bill_paths
        # Provenance preserved
        for participants in result.shared_participants_map.values():
            for sp in participants:
                assert sp.origin_feature == "Dto"

    def test_orphan_files_fall_back_to_shared_infra(self):
        result = _ds(features={
            "Dto": ["packages/dto/orphan.ts"],
            "Auth": ["a.tsx"],
        })
        classifications = {
            "Dto": _cls(
                feature_name="Dto",
                classification="shared-aggregator",
                confidence=5,
            ),
        }
        consumer_maps = {"Dto": {"packages/dto/orphan.ts": []}}  # no consumer

        apply_classifications(result, classifications, consumer_maps)

        assert "Dto" not in result.features
        assert "packages/dto/orphan.ts" in result.features["shared-infra"]

    def test_low_confidence_aggregator_skipped(self):
        # Confidence 3 → don't act; feature stays untouched
        result = _ds(features={
            "Dto": ["packages/dto/x.ts"],
            "Auth": ["a.tsx"],
        })
        classifications = {
            "Dto": _cls(
                feature_name="Dto",
                classification="shared-aggregator",
                confidence=3,
            ),
        }
        apply_classifications(result, classifications, {})
        # Dto unchanged
        assert "Dto" in result.features

    def test_co_owner_role_when_multiple_consumers(self):
        result = _ds(features={
            "Shared UI": ["packages/shared-ui/Button.tsx"],
            "Auth": ["src/auth/login.tsx"],
            "Billing": ["src/billing/plan.tsx"],
            "Settings": ["src/settings/profile.tsx"],
        })
        classifications = {
            "Shared UI": _cls(
                feature_name="Shared UI",
                classification="shared-aggregator",
                confidence=5,
            ),
        }
        consumer_maps = {
            "Shared UI": {
                "packages/shared-ui/Button.tsx": [
                    "Auth", "Billing", "Settings",
                ],
            },
        }
        apply_classifications(result, classifications, consumer_maps)
        for feat in ["Auth", "Billing", "Settings"]:
            sp_list = result.shared_participants_map[feat]
            assert sp_list[0].role == "co-owner"
            assert sp_list[0].file_path == "packages/shared-ui/Button.tsx"


# ── apply_classifications: flow re-attribution ──────────────────────


class TestFlowReattribution:
    def test_flow_moves_to_dominant_consumer(self):
        # Dto owns embed-login-flow but the flow's participants live
        # mostly in Auth. The flow should move to Auth.
        result = _ds(
            features={
                "Dto": ["packages/dto/auth/login.dto.ts"],
                "Authentication": [
                    "apps/cli/auth/login.controller.ts",
                    "apps/web/auth/login.tsx",
                ],
            },
            flows={"Dto": ["embed-login-flow"]},
            flow_descriptions={
                "Dto": {"embed-login-flow": "Embed login journey"},
            },
            flow_participants={
                "Dto": {
                    "embed-login-flow": [
                        {"file_path": "apps/cli/auth/login.controller.ts"},
                        {"file_path": "apps/web/auth/login.tsx"},
                        {"file_path": "packages/dto/auth/login.dto.ts"},
                    ],
                },
            },
        )
        classifications = {
            "Dto": _cls(
                feature_name="Dto",
                classification="shared-aggregator",
                confidence=5,
            ),
        }
        consumer_maps = {
            "Dto": {"packages/dto/auth/login.dto.ts": ["Authentication"]},
        }

        apply_classifications(result, classifications, consumer_maps)

        # Flow moved
        assert "embed-login-flow" in result.flows.get("Authentication", [])
        # Description carried over
        assert (
            result.flow_descriptions["Authentication"]["embed-login-flow"]
            == "Embed login journey"
        )
        # Participants migrated
        assert (
            result.flow_participants["Authentication"]["embed-login-flow"]
            is not None
        )
        # Dto and its flows are gone
        assert "Dto" not in result.flows

    def test_flow_dropped_when_no_majority(self):
        # Flow with participants split 50/50 across two consumers —
        # no clear winner, so drop rather than guess.
        result = _ds(
            features={
                "Dto": ["packages/dto/x.dto.ts"],
                "Auth": ["a.tsx"],
                "Billing": ["b.tsx"],
            },
            flows={"Dto": ["ambiguous-flow"]},
            flow_participants={
                "Dto": {
                    "ambiguous-flow": [
                        {"file_path": "a.tsx"},
                        {"file_path": "b.tsx"},
                    ],
                },
            },
        )
        classifications = {
            "Dto": _cls(
                feature_name="Dto",
                classification="shared-aggregator",
                confidence=5,
            ),
        }
        apply_classifications(result, classifications, {})
        assert "ambiguous-flow" not in result.flows.get("Auth", [])
        assert "ambiguous-flow" not in result.flows.get("Billing", [])

    def test_flow_dropped_when_no_callgraph_data(self):
        result = _ds(
            features={
                "Dto": ["packages/dto/x.dto.ts"],
                "Auth": ["a.tsx"],
            },
            flows={"Dto": ["mystery-flow"]},
            # No flow_participants for mystery-flow
        )
        classifications = {
            "Dto": _cls(
                feature_name="Dto",
                classification="shared-aggregator",
                confidence=5,
            ),
        }
        apply_classifications(result, classifications, {})
        assert "mystery-flow" not in result.flows.get("Auth", [])


# ── apply_classifications: developer-internal ───────────────────────


class TestDeveloperInternal:
    def test_renamed_when_proposed_name_high_confidence(self):
        result = _ds(features={
            "i18n": ["locales/en.json", "locales/fr.json"],
        })
        classifications = {
            "i18n": _cls(
                feature_name="i18n",
                classification="developer-internal",
                confidence=5,
                proposed_name="Translations",
            ),
        }
        apply_classifications(result, classifications, {})
        assert "Translations" in result.features
        assert "i18n" not in result.features

    def test_folded_when_no_proposed_name(self):
        result = _ds(features={
            "Tournament": ["benchmarks/x.ts", "benchmarks/y.ts"],
        })
        classifications = {
            "Tournament": _cls(
                feature_name="Tournament",
                classification="developer-internal",
                confidence=5,
                proposed_name=None,
            ),
        }
        apply_classifications(result, classifications, {})
        assert "Tournament" not in result.features
        assert DEV_INFRA_BUCKET in result.features
        assert "benchmarks/x.ts" in result.features[DEV_INFRA_BUCKET]

    def test_folded_when_low_confidence_rename(self):
        # Confidence 2 — even though there's a proposed name, we don't
        # trust it. Fall back to fold so the feature is not silently
        # mislabelled.
        result = _ds(features={
            "i18n": ["locales/en.json"],
        })
        classifications = {
            "i18n": _cls(
                feature_name="i18n",
                classification="developer-internal",
                confidence=2,
                proposed_name="Translations",
            ),
        }
        apply_classifications(result, classifications, {})
        assert "i18n" not in result.features
        assert "Translations" not in result.features
        assert "locales/en.json" in result.features[DEV_INFRA_BUCKET]


# ── apply_classifications: product-feature inline rename ────────────


class TestProductFeatureRename:
    def test_high_confidence_rename_applies(self):
        result = _ds(features={
            "auth-stuff": ["src/auth/login.tsx"],
        })
        classifications = {
            "auth-stuff": _cls(
                feature_name="auth-stuff",
                classification="product-feature",
                confidence=5,
                proposed_name="Authentication",
            ),
        }
        apply_classifications(result, classifications, {})
        assert "Authentication" in result.features
        assert "auth-stuff" not in result.features

    def test_no_rename_without_proposed_name(self):
        result = _ds(features={
            "Authentication": ["src/auth/login.tsx"],
        })
        classifications = {
            "Authentication": _cls(
                feature_name="Authentication",
                classification="product-feature",
                confidence=5,
                proposed_name=None,
            ),
        }
        apply_classifications(result, classifications, {})
        assert result.features == {"Authentication": ["src/auth/login.tsx"]}


# ── apply_classifications: tooling-infra ────────────────────────────


class TestToolingInfra:
    def test_folds_into_shared_infra(self):
        result = _ds(features={
            "Tsconfig": ["packages/tsconfig/index.json"],
            "Auth": ["a.tsx"],
        })
        classifications = {
            "Tsconfig": _cls(
                feature_name="Tsconfig",
                classification="tooling-infra",
                confidence=5,
            ),
        }
        apply_classifications(result, classifications, {})
        assert "Tsconfig" not in result.features
        assert "packages/tsconfig/index.json" in result.features["shared-infra"]


# ── No-op when nothing to do ────────────────────────────────────────


class TestStructuralGuards:
    """Day 5 regression: excalidraw's main 484-file library package
    was classified as developer-internal and folded into a
    Developer Infrastructure bucket. The model genuinely thought
    the main codebase was infrastructure. These tests pin the
    structural safeguards that catch this no matter what the
    model says.
    """

    def test_largest_feature_never_folded_as_dev_internal(self):
        # Excalidraw scenario: main library package is BIGGEST. Even
        # with a high-confidence developer-internal verdict + a
        # plausible rename, the largest feature must survive.
        result = _ds(features={
            "Excalidraw": [f"packages/excalidraw/file_{i}.ts" for i in range(484)],
            "Math": ["packages/math/index.ts"],
            "Documentation": ["docs/readme.md"],
        })
        classifications = {
            "Excalidraw": _cls(
                feature_name="Excalidraw",
                classification="developer-internal",
                confidence=5,
                proposed_name="Excalidraw Drawing Engine",
            ),
        }
        apply_classifications(result, classifications, {})
        # Largest feature still there with its name
        assert "Excalidraw" in result.features
        assert len(result.features["Excalidraw"]) == 484

    def test_largest_feature_refused_as_aggregator_via_lock(self):
        # The 484-file Excalidraw library is the LARGEST feature
        # → locked → aggregator redistribution refused. The main
        # product code is never scattered across other features.
        # (Pure size-cap was removed for aggregators — real shared-
        # schema packages can be 100+ files. The lock catches the
        # largest-feature-as-aggregator misclassification specifically.)
        result = _ds(features={
            "Excalidraw": [f"packages/excalidraw/file_{i}.ts" for i in range(484)],
            "Math": ["packages/math/index.ts"],
        })
        classifications = {
            "Excalidraw": _cls(
                feature_name="Excalidraw",
                classification="shared-aggregator",
                confidence=5,
                consumer_features=["Math"],
            ),
        }
        apply_classifications(result, classifications, {
            "Excalidraw": {
                f"packages/excalidraw/file_{i}.ts": ["Math"]
                for i in range(484)
            },
        })
        assert "Excalidraw" in result.features
        # No participants accidentally added to Math
        assert result.shared_participants_map.get("Math", []) == []

    def test_medium_aggregator_redistributes_normally(self):
        # 142-file Contracts (canonical aggregator size — n8n's Dto
        # at 78f, dify's Contracts at 142f). Pre-fix the size cap
        # blocked this; now only the largest-feature lock applies
        # and Contracts isn't largest, so redistribution fires.
        result = _ds(features={
            "MainApp": [f"src/app/file_{i}.ts" for i in range(800)],  # largest
            "Contracts": [f"packages/contracts/dto_{i}.ts" for i in range(142)],
            "Authentication": ["src/auth/login.tsx"],
        })
        classifications = {
            "Contracts": _cls(
                feature_name="Contracts",
                classification="shared-aggregator",
                confidence=5,
                consumer_features=["Authentication"],
            ),
        }
        consumer_maps = {
            "Contracts": {
                f"packages/contracts/dto_{i}.ts": ["Authentication"]
                for i in range(142)
            },
        }
        apply_classifications(result, classifications, consumer_maps)
        # Contracts redistributed
        assert "Contracts" not in result.features
        # Authentication received participants
        assert len(result.shared_participants_map["Authentication"]) == 142
        # MainApp untouched
        assert "MainApp" in result.features

    def test_high_commit_feature_refused_as_dev_internal(self):
        # 200+ commits = active maintenance area = real feature
        result = _ds(features={
            "Auth": ["src/auth/login.tsx"],
            "Billing": ["src/billing/plan.tsx"],
        })
        classifications = {
            "Auth": _cls(
                feature_name="Auth",
                classification="developer-internal",
                confidence=5,
                proposed_name=None,
            ),
        }
        commit_counts = {"Auth": 350, "Billing": 12}
        apply_classifications(
            result, classifications, {},
            commit_counts=commit_counts,
        )
        assert "Auth" in result.features

    def test_small_quiet_feature_still_foldable(self):
        # Tiny + cold feature properly folds when classifier says so
        result = _ds(features={
            "Auth": ["src/auth/login.tsx"],
            "Locales": ["locales/en.json", "locales/fr.json"],
        })
        classifications = {
            "Locales": _cls(
                feature_name="Locales",
                classification="developer-internal",
                confidence=5,
                proposed_name="Translations",
            ),
        }
        apply_classifications(result, classifications, {})
        # Auth (largest) is now locked, Locales gets renamed normally
        assert "Translations" in result.features
        assert "Locales" not in result.features

    def test_explicit_locked_features_protected(self):
        # Caller-supplied locks (e.g. library main package from
        # repo_classifier) override any classifier decision.
        result = _ds(features={
            "MyLibrary": ["packages/mylib/a.ts", "packages/mylib/b.ts"],
            "Other": ["other/x.ts"],
        })
        classifications = {
            "MyLibrary": _cls(
                feature_name="MyLibrary",
                classification="developer-internal",
                confidence=5,
            ),
        }
        apply_classifications(
            result, classifications, {},
            locked_features=frozenset({"MyLibrary"}),
        )
        assert "MyLibrary" in result.features


class TestNoOp:
    def test_only_product_features_passes_through(self):
        # Every feature is a product-feature with no rename → identity.
        result = _ds(features={
            "Auth": ["src/auth/login.tsx"],
            "Billing": ["src/billing/plan.tsx"],
        })
        classifications = {
            "Auth": _cls(
                feature_name="Auth",
                classification="product-feature",
                confidence=5,
            ),
            "Billing": _cls(
                feature_name="Billing",
                classification="product-feature",
                confidence=5,
            ),
        }
        apply_classifications(result, classifications, {})
        assert result.features == {
            "Auth": ["src/auth/login.tsx"],
            "Billing": ["src/billing/plan.tsx"],
        }
        assert result.shared_participants_map == {}
