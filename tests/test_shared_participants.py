"""Sprint 8 Day 3 — multi-owner file model + relaxed orphan validation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from faultline.llm.pipeline import _validate_source_coverage
from faultline.llm.sonnet_scanner import DeepScanResult
from faultline.models.types import Feature, SharedParticipant


# ── SharedParticipant Pydantic model ─────────────────────────────────


class TestSharedParticipantModel:
    def test_minimal_construction(self):
        sp = SharedParticipant(file_path="packages/api-types/dto/auth.ts")
        assert sp.file_path == "packages/api-types/dto/auth.ts"
        assert sp.role == "consumer"
        assert sp.line_weight == 1.0
        assert sp.origin_feature is None

    def test_full_construction(self):
        sp = SharedParticipant(
            file_path="packages/shared-ui/Button.tsx",
            role="co-owner",
            line_weight=0.35,
            origin_feature="Shared UI",
        )
        assert sp.role == "co-owner"
        assert sp.line_weight == 0.35
        assert sp.origin_feature == "Shared UI"

    def test_serializes_round_trip(self):
        sp = SharedParticipant(
            file_path="x.ts",
            line_weight=0.5,
            origin_feature="Dto",
        )
        revived = SharedParticipant(**sp.model_dump())
        assert revived == sp


# ── Feature.shared_participants integration ──────────────────────────


class TestFeatureSharedParticipants:
    def _feat(self, **overrides) -> Feature:
        defaults = dict(
            name="Authentication",
            paths=["src/auth/login.tsx"],
            authors=["alice@example.com"],
            total_commits=10,
            bug_fixes=2,
            bug_fix_ratio=0.2,
            last_modified=datetime.now(tz=timezone.utc),
            health_score=80.0,
        )
        defaults.update(overrides)
        return Feature(**defaults)

    def test_default_empty_list(self):
        feat = self._feat()
        assert feat.shared_participants == []

    def test_attaches_participants(self):
        feat = self._feat(
            shared_participants=[
                SharedParticipant(
                    file_path="packages/api-types/dto/auth/login.dto.ts",
                    origin_feature="Dto",
                ),
                SharedParticipant(
                    file_path="packages/shared-ui/LoginForm.tsx",
                    origin_feature="Shared UI",
                ),
            ],
        )
        assert len(feat.shared_participants) == 2
        assert feat.shared_participants[0].origin_feature == "Dto"

    def test_owned_paths_independent_from_participants(self):
        # ``paths`` (owned) and ``shared_participants`` (consumed) are
        # additive surfaces — neither replaces the other.
        feat = self._feat(
            paths=["src/auth/login.tsx", "src/auth/signup.tsx"],
            shared_participants=[
                SharedParticipant(file_path="packages/dto/auth.ts"),
            ],
        )
        assert "src/auth/login.tsx" in feat.paths
        assert feat.shared_participants[0].file_path == "packages/dto/auth.ts"


# ── DeepScanResult.shared_participants_map side channel ──────────────


class TestDeepScanResultSideChannel:
    def test_default_empty_dict(self):
        result = DeepScanResult(features={"Auth": ["a.ts"]})
        assert result.shared_participants_map == {}

    def test_can_attach_participant_lists(self):
        result = DeepScanResult(features={"Auth": ["a.ts"]})
        result.shared_participants_map["Auth"] = [
            SharedParticipant(
                file_path="packages/dto/auth.ts",
                origin_feature="Dto",
            ),
        ]
        assert len(result.shared_participants_map["Auth"]) == 1


# ── _validate_source_coverage with shared_participants ───────────────


class TestValidateCoverageWithSharedParticipants:
    def test_orphan_warning_unchanged_without_participants(self, caplog):
        # Legacy behaviour: file in nobody's paths → warning + fold.
        features: dict[str, list[str]] = {"Auth": ["a.ts"]}
        with caplog.at_level(logging.WARNING):
            _validate_source_coverage(
                ["a.ts", "stray.ts"],
                features,
                shared_participants_map=None,
            )
        assert "stray.ts" in caplog.text
        assert "stray.ts" in features["shared-infra"]

    def test_file_only_in_participants_is_covered(self, caplog):
        # The redistributed Sprint 8 case: file lives only as a
        # shared_participant of some feature (its old aggregator
        # owner was deleted). Validation must NOT flag it as orphan.
        features: dict[str, list[str]] = {"Auth": ["a.ts"]}
        participants_map = {
            "Auth": [
                SharedParticipant(
                    file_path="packages/dto/auth.dto.ts",
                    origin_feature="Dto",
                ),
            ],
        }
        with caplog.at_level(logging.WARNING):
            _validate_source_coverage(
                ["a.ts", "packages/dto/auth.dto.ts"],
                features,
                shared_participants_map=participants_map,
            )
        # No warning, no shared-infra fold for the dto file
        assert "packages/dto/auth.dto.ts" not in caplog.text
        assert "packages/dto/auth.dto.ts" not in features.get("shared-infra", [])

    def test_genuine_orphan_still_flagged_when_participants_present(self, caplog):
        features: dict[str, list[str]] = {"Auth": ["a.ts"]}
        participants_map = {
            "Auth": [
                SharedParticipant(file_path="packages/dto/auth.ts"),
            ],
        }
        with caplog.at_level(logging.WARNING):
            _validate_source_coverage(
                ["a.ts", "packages/dto/auth.ts", "truly_orphaned.ts"],
                features,
                shared_participants_map=participants_map,
            )
        # Participant covered ✓ but truly_orphaned still flagged
        assert "truly_orphaned.ts" in caplog.text
        assert "truly_orphaned.ts" in features["shared-infra"]
        assert "packages/dto/auth.ts" not in features.get("shared-infra", [])

    def test_dict_form_participant_supported_for_legacy(self):
        # Defensive: if Day 4 stashes participants as plain dicts
        # rather than SharedParticipant objects, validation still works.
        features: dict[str, list[str]] = {"Auth": ["a.ts"]}
        participants_map = {
            "Auth": [
                {"file_path": "packages/dto/auth.ts"},
            ],
        }
        # No warning expected for the dto path; it's covered.
        _validate_source_coverage(
            ["a.ts", "packages/dto/auth.ts"],
            features,
            shared_participants_map=participants_map,
        )
        assert "packages/dto/auth.ts" not in features.get("shared-infra", [])

    def test_no_participants_no_orphans_clean_run(self):
        # All source files attributed to features; nothing to do.
        features: dict[str, list[str]] = {
            "Auth": ["a.ts"],
            "Billing": ["b.ts"],
        }
        _validate_source_coverage(
            ["a.ts", "b.ts"],
            features,
            shared_participants_map=None,
        )
        # No shared-infra synthetic bucket created
        assert "shared-infra" not in features
