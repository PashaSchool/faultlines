"""Tests for ``faultline.analyzer.auto_alias_discoverer``
(Improvement #8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.analyzer.auto_alias_discoverer import (
    _team_to_feature_name,
    discover_aliases,
    discover_from_codeowners,
    discover_from_workspace,
)


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# ── _team_to_feature_name ──────────────────────────────────────


class TestTeamToFeatureName:
    def test_strips_org_prefix(self):
        assert _team_to_feature_name("@documenso/billing-team") == "billing"

    def test_strips_at_only(self):
        assert _team_to_feature_name("@billing-team") == "billing"

    def test_no_team_suffix(self):
        assert _team_to_feature_name("@auth") == "auth"

    def test_drops_engineering_suffix(self):
        assert _team_to_feature_name("@auth-engineering") == "auth"

    def test_drops_squad(self):
        assert _team_to_feature_name("@auth-squad") == "auth"


# ── discover_from_codeowners ───────────────────────────────────


class TestCodeowners:
    def test_basic_parsing(self, tmp_path: Path):
        _write(tmp_path, "CODEOWNERS",
               "/apps/web/billing/  @documenso/billing-team\n"
               "/packages/lib/billing/  @documenso/billing-team\n"
               "/apps/auth/  @documenso/auth-team\n")
        rules = discover_from_codeowners(tmp_path)
        names = {r.canonical for r in rules}
        assert names == {"billing", "auth"}
        billing = next(r for r in rules if r.canonical == "billing")
        assert "/apps/web/billing/" in billing.variants
        assert "/packages/lib/billing/" in billing.variants

    def test_handles_dot_github_location(self, tmp_path: Path):
        _write(tmp_path, ".github/CODEOWNERS",
               "/lib/  @auth\n")
        rules = discover_from_codeowners(tmp_path)
        assert any(r.canonical == "auth" for r in rules)

    def test_skips_comments_and_blanks(self, tmp_path: Path):
        _write(tmp_path, "CODEOWNERS",
               "# Top comment\n"
               "\n"
               "/lib/  @real-team\n"
               "# trailing comment\n")
        rules = discover_from_codeowners(tmp_path)
        assert len(rules) == 1
        assert rules[0].canonical == "real"

    def test_skips_protected_names(self, tmp_path: Path):
        _write(tmp_path, "CODEOWNERS",
               "/docs/  @documentation\n"
               "/lib/   @real-team\n")
        rules = discover_from_codeowners(tmp_path)
        names = {r.canonical for r in rules}
        # ``documentation`` is protected — never a real feature
        assert "documentation" not in names
        assert "real" in names

    def test_no_file_returns_empty(self, tmp_path: Path):
        assert discover_from_codeowners(tmp_path) == []

    def test_multiple_owners_uses_first(self, tmp_path: Path):
        _write(tmp_path, "CODEOWNERS",
               "/lib/  @primary-team @secondary-team\n")
        rules = discover_from_codeowners(tmp_path)
        names = {r.canonical for r in rules}
        # Only the first owner counts
        assert names == {"primary"}

    def test_handles_inline_comments(self, tmp_path: Path):
        _write(tmp_path, "CODEOWNERS",
               "/lib/  @billing-team  # owns billing\n")
        rules = discover_from_codeowners(tmp_path)
        # Inline comment shouldn't break parsing
        assert any(r.canonical == "billing" for r in rules)


# ── discover_from_workspace ────────────────────────────────────


class TestWorkspace:
    def test_pnpm_workspace(self, tmp_path: Path):
        _write(tmp_path, "pnpm-workspace.yaml",
               "packages:\n  - 'packages/*'\n")
        _write(tmp_path, "packages/billing/package.json",
               '{"name": "@org/billing"}\n')
        _write(tmp_path, "packages/auth/package.json",
               '{"name": "@org/auth"}\n')
        rules = discover_from_workspace(
            tmp_path,
            ["packages/billing/index.ts", "packages/auth/index.ts"],
        )
        names = {r.canonical for r in rules}
        assert "billing" in names
        assert "auth" in names

    def test_no_workspace_returns_empty(self, tmp_path: Path):
        # Plain repo with no workspace manifest
        assert discover_from_workspace(tmp_path, []) == []


# ── discover_aliases (combined) ────────────────────────────────


class TestCombined:
    def test_merges_codeowners_and_workspace(self, tmp_path: Path):
        _write(tmp_path, "pnpm-workspace.yaml",
               "packages:\n  - 'packages/*'\n")
        _write(tmp_path, "packages/billing/package.json",
               '{"name": "@org/billing"}\n')
        _write(tmp_path, "CODEOWNERS",
               "/packages/billing/  @org/billing-team\n"
               "/apps/web/auth/  @org/auth-team\n")

        rules = discover_aliases(
            tmp_path,
            ["packages/billing/index.ts"],
        )
        names = {r.canonical for r in rules}
        # billing comes from BOTH sources — merged into one
        assert "billing" in names
        # auth from codeowners only
        assert "auth" in names

    def test_workspace_description_wins_for_overlap(self, tmp_path: Path):
        _write(tmp_path, "pnpm-workspace.yaml",
               "packages:\n  - 'packages/*'\n")
        _write(tmp_path, "packages/billing/package.json",
               '{"name": "@org/billing"}\n')
        _write(tmp_path, "CODEOWNERS",
               "/packages/billing/  @org/billing-team\n")

        rules = discover_aliases(
            tmp_path,
            ["packages/billing/index.ts"],
        )
        billing = next(r for r in rules if r.canonical == "billing")
        # Workspace description wins (it's more concrete)
        assert "workspace package" in billing.description
        # CODEOWNERS path still merged into variants
        assert "/packages/billing/" in billing.variants

    def test_empty_repo(self, tmp_path: Path):
        assert discover_aliases(tmp_path, []) == []
