"""Sprint 8 Day 1 — unit tests for the aggregator detector.

Tests the pure functions (candidate selection, response parsing,
coercion) without making a real API call. The full classify_features
end-to-end is exercised in Day 5 against real scans.
"""

from __future__ import annotations

from faultline.llm.aggregator_detector import (
    FeatureClassification,
    _coerce_classification,
    _format_user_message,
    _parse_response,
    _select_candidates,
)
from faultline.llm.sonnet_scanner import DeepScanResult


def _ds(
    features: dict[str, list[str]],
    flows: dict[str, list[str]] | None = None,
) -> DeepScanResult:
    return DeepScanResult(
        features=features,
        flows=flows or {},
    )


# ── _select_candidates ───────────────────────────────────────────────


class TestSelectCandidates:
    def test_picks_every_non_protected_feature(self):
        result = _ds(
            {
                "Authentication": ["src/auth/login.tsx"],
                "Dto": ["packages/api-types/dto/auth.dto.ts"],
                "shared-infra": ["package.json"],         # protected
                "documentation": ["docs/intro.md"],        # protected
            },
            flows={
                "Authentication": ["log-in-flow", "sign-up-flow"],
            },
        )

        candidates = _select_candidates(result)

        names = {n for n, _, _ in candidates}
        assert names == {"Authentication", "Dto"}

    def test_carries_paths_and_flows_through(self):
        result = _ds(
            {"Editor": ["src/editor/canvas.tsx", "src/editor/toolbar.tsx"]},
            flows={"Editor": ["draw-shape", "save-canvas"]},
        )
        candidates = _select_candidates(result)
        assert len(candidates) == 1
        name, paths, flows = candidates[0]
        assert name == "Editor"
        assert paths == ["src/editor/canvas.tsx", "src/editor/toolbar.tsx"]
        assert flows == ["draw-shape", "save-canvas"]

    def test_features_without_flows_get_empty_flow_list(self):
        result = _ds({"Mailer": ["src/mailer/send.ts"]})
        candidates = _select_candidates(result)
        assert candidates[0][2] == []


# ── _format_user_message ──────────────────────────────────────────────


class TestFormatUserMessage:
    def test_renders_each_feature_with_paths_and_flows(self):
        msg = _format_user_message([
            ("Authentication",
             ["src/auth/login.tsx", "src/auth/signup.tsx"],
             ["log-in-flow", "sign-up-flow"]),
            ("Dto", ["packages/api-types/dto/auth.dto.ts"], []),
        ])
        # Section headers
        assert "## Authentication" in msg
        assert "## Dto" in msg
        # Path samples
        assert "src/auth/login.tsx" in msg
        assert "packages/api-types/dto/auth.dto.ts" in msg
        # Flow samples on the feature that has them
        assert "log-in-flow" in msg
        # Closing instruction is present
        assert "JSON" in msg

    def test_caps_path_samples_at_default(self):
        from faultline.llm.aggregator_detector import DEFAULT_SAMPLE_PATHS
        many_paths = [f"src/file_{i}.ts" for i in range(20)]
        msg = _format_user_message([("Big", many_paths, [])])
        # Only first DEFAULT_SAMPLE_PATHS appear
        for i in range(DEFAULT_SAMPLE_PATHS):
            assert f"src/file_{i}.ts" in msg
        # Beyond cap is dropped
        assert "src/file_19.ts" not in msg


# ── _parse_response ───────────────────────────────────────────────────


class TestParseResponse:
    def test_well_formed_response(self):
        text = """
        {
          "classifications": [
            {"name": "Authentication", "class": "product-feature", "confidence": 5, "reasoning": "User-facing login surface."}
          ]
        }
        """
        entries = _parse_response(text)
        assert len(entries) == 1
        assert entries[0]["name"] == "Authentication"
        assert entries[0]["class"] == "product-feature"

    def test_response_with_prose_around_json(self):
        text = (
            "Here's my analysis:\n\n"
            '{"classifications": [{"name": "Dto", "class": "shared-aggregator", '
            '"confidence": 5, "reasoning": "Multi-domain DTOs."}]}\n\n'
            "Done."
        )
        entries = _parse_response(text)
        assert len(entries) == 1
        assert entries[0]["class"] == "shared-aggregator"

    def test_malformed_json_returns_empty(self):
        assert _parse_response("not json at all") == []
        assert _parse_response("{this is broken}") == []

    def test_missing_classifications_key(self):
        text = '{"other_field": []}'
        assert _parse_response(text) == []


# ── _coerce_classification ────────────────────────────────────────────


class TestCoerceClassification:
    def test_product_feature_minimal_entry(self):
        verdict = _coerce_classification({
            "name": "Authentication",
            "class": "product-feature",
            "confidence": 5,
            "reasoning": "User-facing login surface.",
        })
        assert verdict is not None
        assert isinstance(verdict, FeatureClassification)
        assert verdict.feature_name == "Authentication"
        assert verdict.classification == "product-feature"
        assert verdict.confidence == 5
        assert verdict.consumer_features is None
        assert verdict.proposed_name is None

    def test_shared_aggregator_with_consumers(self):
        verdict = _coerce_classification({
            "name": "Dto",
            "class": "shared-aggregator",
            "confidence": 5,
            "reasoning": "DTOs span auth, workflows, billing.",
            "consumer_features": ["Authentication", "Workflows", "Billing"],
        })
        assert verdict.classification == "shared-aggregator"
        assert verdict.consumer_features == [
            "Authentication", "Workflows", "Billing",
        ]

    def test_developer_internal_with_proposed_name(self):
        verdict = _coerce_classification({
            "name": "i18n",
            "class": "developer-internal",
            "confidence": 4,
            "reasoning": "Locale JSON files, no admin UI.",
            "proposed_name": "Translations",
        })
        assert verdict.classification == "developer-internal"
        assert verdict.proposed_name == "Translations"

    def test_developer_internal_without_proposed_name(self):
        verdict = _coerce_classification({
            "name": "Misc",
            "class": "developer-internal",
            "confidence": 3,
            "reasoning": "Mixed bag with no clear product label.",
            "proposed_name": None,
        })
        assert verdict.proposed_name is None

    def test_invalid_class_returns_none(self):
        assert _coerce_classification({
            "name": "Foo",
            "class": "made-up-class",
            "confidence": 5,
        }) is None

    def test_missing_name_returns_none(self):
        assert _coerce_classification({
            "class": "product-feature",
            "confidence": 5,
        }) is None

    def test_confidence_clamped_to_range(self):
        v_low = _coerce_classification({
            "name": "X", "class": "product-feature", "confidence": -10,
        })
        v_high = _coerce_classification({
            "name": "Y", "class": "product-feature", "confidence": 99,
        })
        v_string = _coerce_classification({
            "name": "Z", "class": "product-feature", "confidence": "five",
        })
        assert v_low.confidence == 1
        assert v_high.confidence == 5
        assert v_string.confidence == 3  # default fallback

    def test_consumer_features_filters_non_strings(self):
        verdict = _coerce_classification({
            "name": "Dto",
            "class": "shared-aggregator",
            "confidence": 4,
            "reasoning": "...",
            "consumer_features": ["Auth", 42, None, "Workflows", ""],
        })
        assert verdict.consumer_features == ["Auth", "Workflows"]

    def test_consumer_features_only_for_aggregator(self):
        # Even if model accidentally provides consumer_features on a
        # product-feature, the field is None for non-aggregator classes.
        verdict = _coerce_classification({
            "name": "Auth",
            "class": "product-feature",
            "confidence": 5,
            "reasoning": "...",
            "consumer_features": ["Other"],
        })
        assert verdict.consumer_features is None


# ── _PROTECTED_NAMES coverage (regression guard) ──────────────────────


class TestPromptTestInfraFoldRule:
    """Pinning the rule that test infrastructure always folds —
    never renames. Day 5 found ``E2E Auth Server`` was being
    promoted to ``E2E Test Auth Server`` (a rename) instead of
    folded into developer-infrastructure. Renames keep test scaffolds
    visible on the dashboard; folding hides them under one drawer.
    The system prompt now explicitly enumerates the always-fold
    sub-cases. This test pins the prompt content so a regression
    that drops the rule shows up in CI.
    """

    def test_prompt_lists_test_infra_as_always_fold(self):
        from faultline.llm.aggregator_detector import _SYSTEM_PROMPT
        assert "ALWAYS FOLD" in _SYSTEM_PROMPT
        assert "E2E" in _SYSTEM_PROMPT
        assert "test scaffolding" in _SYSTEM_PROMPT.lower()
        assert "ci/cd" in _SYSTEM_PROMPT.lower()
        # The rule must be unambiguous: proposed_name MUST be null
        assert "proposed_name MUST be null" in _SYSTEM_PROMPT

    def test_prompt_keeps_translations_as_rename_example(self):
        # The rename branch still allows i18n → Translations
        from faultline.llm.aggregator_detector import _SYSTEM_PROMPT
        assert "Translations" in _SYSTEM_PROMPT


class TestProtectedNames:
    def test_developer_infrastructure_protected(self):
        # Day 4 will materialize this synthetic bucket; classifier must
        # not see it on subsequent runs (idempotency).
        from faultline.llm.aggregator_detector import _PROTECTED_NAMES
        assert "developer-infrastructure" in _PROTECTED_NAMES
        assert "shared-infra" in _PROTECTED_NAMES
        assert "documentation" in _PROTECTED_NAMES
        assert "examples" in _PROTECTED_NAMES
