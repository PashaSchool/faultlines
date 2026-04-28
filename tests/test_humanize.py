"""Tests for ``faultline.analyzer.humanize``."""

from __future__ import annotations

import pytest

from faultline.analyzer.humanize import humanize_feature_name, humanize_flow_name


class TestFeatureName:
    def test_simple_kebab(self):
        assert humanize_feature_name("user-authentication") == "User Authentication"

    def test_underscore(self):
        assert humanize_feature_name("user_authentication") == "User Authentication"

    def test_slash_takes_last_segment(self):
        assert (
            humanize_feature_name("team-and-organisation-management/team-lifecycle")
            == "Team Lifecycle"
        )

    def test_and_becomes_ampersand(self):
        assert (
            humanize_feature_name("billing-and-subscriptions")
            == "Billing & Subscriptions"
        )

    def test_acronym_uppercased(self):
        assert humanize_feature_name("api-tokens") == "API Tokens"
        assert humanize_feature_name("sdk-init") == "SDK Init"
        assert humanize_feature_name("oauth-flow") == "OAuth Flow"
        assert humanize_feature_name("trpc-router") == "tRPC Router"

    def test_acronym_in_middle_or_end(self):
        assert humanize_feature_name("user-api-tokens") == "User API Tokens"
        assert humanize_feature_name("public-rest-api") == "Public REST API"

    def test_inner_lowercase_words(self):
        assert (
            humanize_feature_name("acceptance-of-terms")
            == "Acceptance of Terms"
        )
        assert humanize_feature_name("login-by-passkey") == "Login by Passkey"

    def test_first_word_always_capital(self):
        # "of" appears first — should still be capitalized
        assert humanize_feature_name("of-life-and-death") == "Of Life & Death"

    def test_camelcase_passthrough(self):
        # Token with internal caps (OAuth, GraphQL) preserved
        assert humanize_feature_name("OAuth-callback") == "OAuth Callback"

    def test_idempotent(self):
        assert (
            humanize_feature_name("Already Humanized Name")
            == "Already Humanized Name"
        )

    def test_empty_input(self):
        assert humanize_feature_name("") == ""

    def test_protected_synthetic_buckets(self):
        # documentation / shared-infra rendered cleanly too
        assert humanize_feature_name("documentation") == "Documentation"
        assert humanize_feature_name("shared-infra") == "Shared Infra"


class TestFlowName:
    def test_simple(self):
        assert humanize_flow_name("create-organisation") == "Create Organisation"

    def test_long_imperative(self):
        assert (
            humanize_flow_name("accept-or-decline-organisation-invitation")
            == "Accept or Decline Organisation Invitation"
        )

    def test_acronym_in_flow(self):
        assert humanize_flow_name("revoke-api-token") == "Revoke API Token"
