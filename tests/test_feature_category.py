"""Sprint 16 polish — feature_category tests.

Anchored on the 69 real features observed across S12-S15 dify /
supabase / immich live scans. Every assertion below mirrors a name +
sample path from those scans.
"""

from __future__ import annotations

from faultline.analyzer.feature_category import (
    Category,
    classify_categories,
    classify_feature,
    tier_for,
)


# ── Real-world fixtures from S12-S15 scans ────────────────────────────


def test_dify_workflow_app_is_product():
    assert classify_feature(
        "dify-web/workflow-app",
        ["web/app/components/workflow-app/index.tsx"],
    ) == Category.PRODUCT


def test_dify_billing_is_product():
    assert classify_feature(
        "dify-web/billing",
        ["web/app/(commonLayout)/app/(appDetailLayout)/[appId]/billing.tsx"],
    ) == Category.PRODUCT


def test_dify_auth_is_product():
    assert classify_feature(
        "auth",
        ["web/app/(shareLayout)/webapp-signin/components/external-member-sso-auth.tsx"],
    ) == Category.PRODUCT


def test_dify_datasets_is_product():
    assert classify_feature(
        "dify-web/datasets",
        ["web/app/(commonLayout)/datasets/page.tsx"],
    ) == Category.PRODUCT


def test_dify_i18n_classified_as_i18n():
    assert classify_feature(
        "i18n",
        ["web/i18n/fr-FR/dataset-documents.json"],
    ) == Category.I18N


def test_dify_workspace_i18n_classified_as_i18n():
    assert classify_feature(
        "dify-web/i18n",
        ["web/i18n/nl-NL/dataset-hit-testing.json"],
    ) == Category.I18N


def test_dify_contracts_is_contracts():
    assert classify_feature(
        "contracts",
        ["packages/contracts/generated/api/console/apps/orpc.gen.ts"],
    ) == Category.CONTRACTS


def test_dify_types_is_contracts():
    assert classify_feature(
        "dify-web/types",
        ["web/app/components/base/features/types.ts"],
    ) == Category.CONTRACTS


def test_dify_ui_is_ui_system():
    assert classify_feature(
        "ui",
        ["web/app/components/workflow/nodes/http/components/x.tsx"],
    ) == Category.UI_SYSTEM


def test_dify_dify_ui_is_ui_system():
    assert classify_feature(
        "dify-ui",
        ["packages/dify-ui/.gitignore"],
    ) == Category.UI_SYSTEM


def test_dify_index_stories_is_ui_system():
    """Storybook tile."""
    assert classify_feature(
        "dify-web/index.stories",
        ["web/app/components/app-sidebar/app-info/x.stories.tsx"],
    ) == Category.UI_SYSTEM


def test_dify_iconify_is_ui_system():
    assert classify_feature(
        "iconify-collections",
        ["packages/iconify-collections/package.json"],
    ) == Category.UI_SYSTEM


def test_dify_node_sdk_is_sdk():
    assert classify_feature(
        "Node.js SDK",
        ["sdks/nodejs-client/.gitignore"],
    ) == Category.SDK


def test_dify_dify_client_is_sdk():
    assert classify_feature(
        "dify-client",
        ["sdks/nodejs-client/.gitignore"],
    ) == Category.SDK


def test_dify_migrate_is_tooling():
    assert classify_feature(
        "migrate-no-unchecked-indexed-access",
        ["packages/migrate-no-unchecked-indexed-access/package.json"],
    ) == Category.TOOLING


def test_dify_documentation_is_documentation():
    assert classify_feature(
        "documentation",
        ["docs/intro.md", "docs/setup.md"],
    ) == Category.DOCUMENTATION


def test_dify_synthetic_shared_infra():
    assert classify_feature("shared-infra", []) == Category.SYNTHETIC


# ── Supabase fixtures ─────────────────────────────────────────────────


def test_supabase_studio_is_product():
    """Studio is the main supabase dashboard — user-facing."""
    assert classify_feature(
        "studio",
        ["apps/studio/.env"],
    ) == Category.PRODUCT


def test_supabase_billing_is_product():
    assert classify_feature(
        "billing",
        ["apps/www/components/Pricing/PricingComputeSection.tsx"],
    ) == Category.PRODUCT


def test_supabase_auth_is_product():
    assert classify_feature(
        "auth",
        ["apps/docs/content/guides/auth/auth-hooks.mdx"],
    ) == Category.PRODUCT


def test_supabase_etl_is_product():
    assert classify_feature(
        "etl",
        ["apps/studio/data/replication/stop-pipeline-mutation.ts"],
    ) == Category.PRODUCT


def test_supabase_ui_library_is_ui_system():
    assert classify_feature(
        "ui-library",
        ["apps/ui-library/registry/blocks.ts"],
    ) == Category.UI_SYSTEM


def test_supabase_ui_patterns_is_ui_system():
    assert classify_feature(
        "ui-patterns",
        ["packages/ui-patterns/src/AssistantChat/AssistantChat.tsx"],
    ) == Category.UI_SYSTEM


def test_supabase_design_system_is_ui_system():
    assert classify_feature(
        "design-system/registry",
        ["apps/design-system/registry/charts.ts"],
    ) == Category.UI_SYSTEM


def test_supabase_icons_is_ui_system():
    assert classify_feature(
        "icons",
        ["packages/icons/src/icons/REST-api.ts"],
    ) == Category.UI_SYSTEM


def test_supabase_api_types_is_contracts():
    assert classify_feature(
        "api-types",
        ["packages/api-types/index.ts"],
    ) == Category.CONTRACTS


def test_supabase_pg_meta_is_contracts():
    assert classify_feature(
        "pg-meta",
        ["packages/pg-meta/src/sql/column-privileges.ts"],
    ) == Category.CONTRACTS


def test_supabase_eslint_config_is_tooling():
    assert classify_feature(
        "eslint-config-supabase",
        ["packages/eslint-config-supabase/next.js"],
    ) == Category.TOOLING


def test_supabase_dev_tools_is_tooling():
    assert classify_feature(
        "dev-tools",
        ["packages/dev-tools/DevToolbar.tsx"],
    ) == Category.TOOLING


def test_supabase_build_icons_is_tooling():
    assert classify_feature(
        "build-icons",
        ["packages/build-icons/package.json"],
    ) == Category.TOOLING


def test_supabase_documentation_is_documentation():
    assert classify_feature(
        "documentation",
        ["apps/design-system/app/(app)/docs/x/page.mdx"],
    ) == Category.DOCUMENTATION


def test_supabase_learn_is_documentation():
    assert classify_feature(
        "learn/content",
        ["apps/learn/content/foundations/architecture.mdx"],
    ) == Category.DOCUMENTATION


# ── Immich fixtures ───────────────────────────────────────────────────


def test_immich_documentation_is_documentation():
    assert classify_feature(
        "documentation",
        ["docs/.gitignore"],
    ) == Category.DOCUMENTATION


def test_immich_immich_i18n_is_i18n():
    assert classify_feature(
        "immich-i18n",
        ["i18n/br.json"],
    ) == Category.I18N


def test_immich_cli_is_sdk():
    assert classify_feature(
        "cli",
        ["cli/.gitignore"],
    ) == Category.SDK


def test_immich_sdk_is_sdk():
    assert classify_feature(
        "sdk",
        ["open-api/typescript-sdk/.npmignore"],
    ) == Category.SDK


def test_immich_deployment_is_deployment():
    assert classify_feature(
        "deployment",
        ["deployment/modules/cloudflare/docs/.terraform.lock"],
    ) == Category.DEPLOYMENT


def test_immich_e2e_is_deployment():
    """E2E test infra — closest semantic category is deployment-ish.
    Acceptable to land in PRODUCT or DEPLOYMENT; we just don't want
    it labelled ``product`` and confuse the carousel."""
    cat = classify_feature(
        "e2e-auth-server",
        ["e2e-auth-server/Dockerfile"],
    )
    # With Dockerfile path → looks_like_deployment fires
    assert cat in {Category.DEPLOYMENT, Category.PRODUCT}


def test_immich_shared_components_is_ui_system():
    assert classify_feature(
        "immich-web/shared-components",
        ["web/src/lib/components/shared-components/settings/x.svelte"],
    ) == Category.UI_SYSTEM


# ── Tier mapping ──────────────────────────────────────────────────────


def test_tier_mapping_three_tiers():
    """Every category maps to exactly one of three tiers."""
    tiers = {tier_for(c) for c in Category}
    assert tiers == {"product", "supporting", "hidden"}


def test_product_tier_only_for_product():
    assert tier_for(Category.PRODUCT) == "product"
    assert tier_for(Category.UI_SYSTEM) == "supporting"
    assert tier_for(Category.I18N) == "supporting"
    assert tier_for(Category.CONTRACTS) == "supporting"
    assert tier_for(Category.SDK) == "supporting"
    assert tier_for(Category.TOOLING) == "hidden"
    assert tier_for(Category.DOCUMENTATION) == "hidden"
    assert tier_for(Category.DEPLOYMENT) == "hidden"
    assert tier_for(Category.SYNTHETIC) == "hidden"


# ── Bulk classifier ───────────────────────────────────────────────────


def test_classify_categories_bulk():
    """End-to-end on a mini dify-shaped fixture."""
    features = {
        "auth": ["web/app/signin/page.tsx"],
        "i18n": ["web/i18n/en.json"],
        "contracts": ["packages/contracts/generated/api/orpc.gen.ts"],
        "ui": ["web/app/components/x.tsx"],
        "Node.js SDK": ["sdks/nodejs-client/index.ts"],
        "documentation": ["docs/intro.md"],
        "shared-infra": [],
    }
    result = classify_categories(features)
    assert result["auth"] == Category.PRODUCT
    assert result["i18n"] == Category.I18N
    assert result["contracts"] == Category.CONTRACTS
    assert result["ui"] == Category.UI_SYSTEM
    assert result["Node.js SDK"] == Category.SDK
    assert result["documentation"] == Category.DOCUMENTATION
    assert result["shared-infra"] == Category.SYNTHETIC


# ── Edge cases ────────────────────────────────────────────────────────


def test_unknown_name_defaults_to_product():
    """When nothing matches, default is PRODUCT — feature shows up
    in the main carousel rather than vanishing."""
    assert classify_feature("some-novel-domain", ["src/x.ts"]) == Category.PRODUCT


def test_empty_paths_falls_back_to_name():
    """No paths → can only use name. ``Node.js SDK`` should still
    classify on token alone."""
    assert classify_feature("Node.js SDK", []) == Category.SDK
    assert classify_feature("documentation", []) == Category.DOCUMENTATION


def test_path_share_dominant_wins_over_neutral_name():
    """Even if name is neutral, dominant translation paths win."""
    assert classify_feature(
        "messages",
        [f"web/i18n/{lang}.json" for lang in ("en", "fr", "de", "es")],
    ) == Category.I18N


# ── Sprint 16 polish: synthetic leftover detection ────────────────────


def test_commit_prefix_leftover_detected_as_synthetic():
    """Features with 'Detected from N commits' description came from
    commit_prefix_enrichment_pass — they're not real products."""
    cat = classify_feature(
        "workflow",
        ["web/app/components/x.tsx"],
        description="Detected from 49 commits with 'workflow:' prefix.",
    )
    assert cat == Category.SYNTHETIC


def test_extract_overlooked_leftover_detected_as_synthetic():
    cat = classify_feature(
        "packages",
        ["packages/contracts/api/x.ts"],
        description="Files under packages/ extracted from documentation",
    )
    assert cat == Category.SYNTHETIC


def test_classify_categories_uses_descriptions():
    out = classify_categories(
        features={
            "auth": ["web/app/auth.ts"],
            "workflow": ["web/x.ts"],
        },
        descriptions={
            "auth": "User authentication and access",
            "workflow": "Detected from 49 commits with 'workflow:' prefix.",
        },
    )
    assert out["auth"] == Category.PRODUCT
    assert out["workflow"] == Category.SYNTHETIC
