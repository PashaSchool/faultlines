"""Regression tests for ``cli._inject_new_pipeline_descriptions``.

Sprint 2 surfaced this bug: the substring-based matcher leaked
descriptions across unrelated features once dedup introduced longer
multi-word names. Caught on a real formbricks scan where:

  - ``ai`` (15-file feature) absorbed the description of
    ``email/auth-emails`` because the substring 'ai' appears inside
    'auth-emails'.
  - ``config-typescript`` absorbed the description of ``types``
    because 'types' is a substring of 'typescript'.

Fix: tighten the matcher so substring containment must align with a
slash-segment boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone

from faultline.cli import _inject_new_pipeline_descriptions
from faultline.models.types import Feature, FeatureMap


def _f(name: str, desc: str | None = None) -> Feature:
    return Feature(
        name=name,
        description=desc,
        paths=["x.ts"],
        authors=[],
        total_commits=0,
        bug_fixes=0,
        bug_fix_ratio=0.0,
        last_modified=datetime.now(tz=timezone.utc),
        health_score=80.0,
    )


def _map(feats: list[Feature]) -> FeatureMap:
    return FeatureMap(
        repo_path="/tmp/x",
        analyzed_at=datetime.now(tz=timezone.utc),
        total_commits=0,
        date_range_days=30,
        features=feats,
    )


class TestNoSubstringLeak:
    def test_ai_does_not_inherit_email_auth_emails(self):
        fm = _map([_f("ai"), _f("email/auth-emails")])
        _inject_new_pipeline_descriptions(
            fm, {"email/auth-emails": "About auth emails."},
        )
        feats = {f.name: f.description for f in fm.features}
        assert feats["ai"] is None
        assert feats["email/auth-emails"] == "About auth emails."

    def test_config_typescript_does_not_inherit_types(self):
        fm = _map([_f("config-typescript"), _f("types")])
        _inject_new_pipeline_descriptions(fm, {"types": "Type defs."})
        feats = {f.name: f.description for f in fm.features}
        assert feats["config-typescript"] is None
        assert feats["types"] == "Type defs."

    def test_unrelated_long_name_does_not_match_short(self):
        fm = _map([_f("api"), _f("user-api-tokens")])
        _inject_new_pipeline_descriptions(
            fm, {"user-api-tokens": "API token CRUD."},
        )
        feats = {f.name: f.description for f in fm.features}
        # "api" must not silently grab user-api-tokens' description.
        assert feats["api"] is None

    def test_path_segment_does_not_leak_across_packages(self):
        # Real Sprint 2 regression: feat 'web/surveys' (web package's
        # survey code) inherited the description of 'surveys' (the
        # surveys package) because both ended in 'surveys' after slash
        # split.
        fm = _map([_f("web/surveys"), _f("surveys")])
        _inject_new_pipeline_descriptions(
            fm, {"surveys": "The surveys runtime package."},
        )
        feats = {f.name: f.description for f in fm.features}
        assert feats["web/surveys"] is None
        assert feats["surveys"] == "The surveys runtime package."


class TestStillMatches:
    def test_exact_name(self):
        fm = _map([_f("auth")])
        _inject_new_pipeline_descriptions(fm, {"auth": "User auth."})
        assert fm.features[0].description == "User auth."

    def test_path_segment_match_dropped_for_safety(self):
        # The pre-Sprint-2 matcher fuzzed across slash boundaries; the
        # current contract is exact-name + singular/plural only. The
        # new pipeline keeps descriptions keyed by the same name that
        # reaches feature_map, so fuzzy fallback is unnecessary AND
        # demonstrably leaks (see TestNoSubstringLeak).
        fm = _map([_f("auth")])
        _inject_new_pipeline_descriptions(fm, {"api/auth": "User auth."})
        assert fm.features[0].description is None

    def test_singular_plural_same_level(self):
        # Singular/plural fuzz only kicks in at the same identifier
        # level — not across slash boundaries — so feat "issue" still
        # picks up desc "issues" but feat "api/issue" does not pick
        # up plain "issues".
        fm = _map([_f("issue")])
        _inject_new_pipeline_descriptions(fm, {"issues": "Issues."})
        assert fm.features[0].description == "Issues."

    def test_existing_description_preserved(self):
        fm = _map([_f("auth", desc="kept")])
        _inject_new_pipeline_descriptions(fm, {"auth": "overwritten"})
        assert fm.features[0].description == "kept"

    def test_empty_descriptions_is_noop(self):
        fm = _map([_f("auth")])
        _inject_new_pipeline_descriptions(fm, {})
        assert fm.features[0].description is None
