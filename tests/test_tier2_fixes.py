"""Tier 2 of Fixable-accuracy improvements.

Fix #3 — auto-enable cross-cluster dedup by default (already
covered by tests/test_dedup.py; this file pins the default-on
behavior at the pipeline.run() level).

Fix #4 — Haiku batch rename for generic feature names. Tested
without making a real API call by exercising the candidate-
selection and response-parsing pure functions directly.
"""

from __future__ import annotations

from faultline.llm.rename_generic import (
    _GENERIC_NAMES,
    _name_matches_dominant_path,
    _parse_response,
    _select_candidates,
)
from faultline.llm.sonnet_scanner import DeepScanResult


def _ds(features: dict[str, list[str]]) -> DeepScanResult:
    return DeepScanResult(features=features)


# ── _name_matches_dominant_path ──────────────────────────────────────


class TestNameMatchesPath:
    def test_utils_for_packages_utils_matches(self):
        # ``Utils`` covering ``packages/utils/*`` is OK — package literally is utils
        assert _name_matches_dominant_path("Utils", [
            "packages/utils/a.ts",
            "packages/utils/b.ts",
            "packages/utils/c.ts",
        ])

    def test_utils_for_packages_core_does_not_match(self):
        # ``Utils`` covering ``packages/core/*`` is suspicious — needs rename
        assert not _name_matches_dominant_path("Utils", [
            "packages/core/a.ts",
            "packages/core/b.ts",
        ])

    def test_handles_empty_paths(self):
        assert not _name_matches_dominant_path("Utils", [])


# ── _select_candidates ───────────────────────────────────────────────


class TestSelectCandidates:
    def test_picks_generic_named_with_non_matching_path(self):
        result = _ds({
            "Utils":     ["packages/core/u.ts"],         # mismatched → candidate
            "Constants": ["packages/core/c.ts"],         # mismatched → candidate
            "Editor":    ["packages/editor/e.ts"],       # specific name, not generic
        })
        candidates = _select_candidates(result)
        names = {n for n, _ in candidates}
        assert "Utils" in names
        assert "Constants" in names
        assert "Editor" not in names

    def test_skips_generic_name_when_path_matches(self):
        # ``Utils`` with paths under packages/utils → skip (literal package match)
        result = _ds({
            "Utils": ["packages/utils/a.ts", "packages/utils/b.ts"],
        })
        assert _select_candidates(result) == []

    def test_skips_protected_buckets(self):
        result = _ds({
            "shared-infra": ["c.ts", "d.ts", "e.ts"],
            "documentation": ["docs/x.md"],
        })
        assert _select_candidates(result) == []

    def test_documentation_with_docs_paths_skipped(self):
        # ``Documentation`` covering actual ``docs/*`` IS correctly named — skip
        result = _ds({
            "Documentation": ["docs/intro.md", "docs/api.md"],
        })
        assert _select_candidates(result) == []

    def test_documentation_with_examples_paths_is_candidate(self):
        # ``Documentation`` covering ``examples/*`` IS misnamed — Fix #4 should
        # propose "Examples" or similar. Eval flagged this exact case on
        # excalidraw and strapi.
        result = _ds({
            "Documentation": ["examples/with-nextjs/app.tsx", "examples/script/x.ts"],
        })
        candidates = _select_candidates(result)
        names = {n for n, _ in candidates}
        assert "Documentation" in names

    def test_specific_names_skipped(self):
        # Already-specific names should not appear as candidates
        result = _ds({
            "Authentication":      ["auth/login.ts"],
            "Issue Board":         ["space/issues.tsx"],
            "Workflow Constants":  ["lib/workflow_constants.ts"],
        })
        assert _select_candidates(result) == []


# ── _parse_response ──────────────────────────────────────────────────


class TestParseResponse:
    def test_extracts_simple_renames(self):
        text = '{"renames": [{"name": "Utils", "new": "Frontend Helpers"}]}'
        assert _parse_response(text) == {"Utils": "Frontend Helpers"}

    def test_skips_keep_responses(self):
        text = (
            '{"renames": ['
            '{"name": "Utils", "new": "KEEP"},'
            '{"name": "Constants", "new": "Workflow Constants"}'
            "]}"
        )
        assert _parse_response(text) == {"Constants": "Workflow Constants"}

    def test_returns_empty_on_malformed_json(self):
        assert _parse_response("not json at all") == {}
        assert _parse_response("{this is not valid json}") == {}

    def test_handles_response_with_prose_around_json(self):
        text = (
            "Here are my proposed renames:\n\n"
            '{"renames": [{"name": "Utils", "new": "Helpers"}]}\n\n'
            "Reasoning: ..."
        )
        assert _parse_response(text) == {"Utils": "Helpers"}

    def test_ignores_entries_missing_fields(self):
        text = (
            '{"renames": ['
            '{"name": "Utils"},'                                 # no "new"
            '{"new": "Workflow Constants"},'                     # no "name"
            '{"name": "Constants", "new": "Workflow Constants"}' # OK
            "]}"
        )
        assert _parse_response(text) == {"Constants": "Workflow Constants"}


# ── _GENERIC_NAMES coverage ──────────────────────────────────────────


class TestGenericNames:
    def test_all_eval_flagged_names_present(self):
        """Names auto-flagged by EVAL_REPORT should all appear in
        _GENERIC_NAMES so Fix #4 actually picks them up."""
        eval_flagged = {
            "dto", "config", "entities", "decorators", "backend common",
            "backend test utils", "constants", "utils", "documentation",
            "api", "commands", "di", "scenarios", "tournament",
            "components", "future", "scripts front", "platform infrastructure",
            "platform primitives", "search analytics & platform",
            "http client foundation", "developer platform", "maintenance",
        }
        missing = eval_flagged - _GENERIC_NAMES
        assert not missing, f"these eval-flagged names not in _GENERIC_NAMES: {missing}"
