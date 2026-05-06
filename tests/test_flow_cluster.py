"""Sprint 12 Day 2 — virtual cluster promotion tests."""

from __future__ import annotations

from faultline.llm.flow_cluster import (
    DOMAIN_TOKENS,
    MIN_FLOWS,
    Promotion,
    apply_promotions,
    classify_flow_domain,
    plan_promotions,
    promote_virtual_clusters,
)
from faultline.llm.sonnet_scanner import DeepScanResult


# ── classify_flow_domain ──────────────────────────────────────────────


def test_classify_auth_tokens():
    assert classify_flow_domain("reset-password-with-email-flow") == "auth"
    assert classify_flow_domain("user-signin-flow") == "auth"
    assert classify_flow_domain("oauth-callback-flow") == "auth"
    assert classify_flow_domain("signup-email-verification-flow") == "auth"


def test_classify_billing_tokens():
    assert classify_flow_domain("manage-subscription-flow") == "billing"
    assert classify_flow_domain("invoice-download-flow") == "billing"
    assert classify_flow_domain("stripe-checkout-flow") == "billing"


def test_classify_notifications():
    assert classify_flow_domain("display-notification-flow") == "notifications"


def test_classify_no_match():
    assert classify_flow_domain("create-workflow-flow") is None
    assert classify_flow_domain("delete-dataset-flow") is None


def test_classify_underscore_separator():
    """Tokens normalised so signin matches sign_in_user_flow."""
    assert classify_flow_domain("sign_in_user_flow") == "auth"


# ── plan_promotions ───────────────────────────────────────────────────


def _result_dify_minimal() -> DeepScanResult:
    """Minimal dify-shaped fixture: auth flows scattered across i18n/ui."""
    return DeepScanResult(
        features={
            "i18n": [
                "web/i18n/en-US/translation.ts",
                "web/i18n/zh-CN/translation.ts",
                "web/app/signin/page.tsx",
                "web/app/reset-password/page.tsx",
            ],
            "ui": [
                "web/components/Button.tsx",
                "web/app/forgot-password/page.tsx",
                "web/app/signup-set-password/page.tsx",
            ],
            "contracts": [
                "web/types/api.ts",
                "web/types/auth/login-types.ts",
            ],
            "workflow": [
                "web/app/workflow/editor.tsx",
            ],
        },
        flows={
            "i18n": [
                "change-language-flow",
                "reset-password-with-email-flow",
                "webapp-email-signin-flow",
                "signup-email-verification-flow",
            ],
            "ui": [
                "forgot-password-flow",
                "initial-password-setup-flow",
            ],
            "contracts": [
                "login-flow",
                "logout-flow",
            ],
            "workflow": [
                "create-workflow-flow",
            ],
        },
        descriptions={"i18n": "...", "ui": "...", "contracts": "...", "workflow": "..."},
    )


def test_plan_promotes_auth_when_no_auth_feature():
    result = _result_dify_minimal()
    promos = plan_promotions(result)
    auth_promos = [p for p in promos if p.domain == "auth"]
    assert len(auth_promos) == 1
    promo = auth_promos[0]
    # 3 from i18n + 2 from ui + 2 from contracts = 7
    assert len(promo.flows_to_move) == 7
    # Paths to move are auth-named files only (not Button.tsx, not translation.ts).
    moved_paths = {p for _, p in promo.paths_to_move}
    assert "web/app/signin/page.tsx" in moved_paths
    assert "web/app/reset-password/page.tsx" in moved_paths
    assert "web/app/forgot-password/page.tsx" in moved_paths
    assert "web/types/auth/login-types.ts" in moved_paths
    # Non-domain paths stay put.
    assert "web/i18n/en-US/translation.ts" not in moved_paths
    assert "web/components/Button.tsx" not in moved_paths


def test_plan_skips_when_auth_feature_exists():
    """If feature menu already has 'auth', don't promote."""
    result = _result_dify_minimal()
    result.features["auth"] = ["server/auth/handler.ts"]
    promos = plan_promotions(result)
    assert all(p.domain != "auth" for p in promos)


def test_plan_skips_when_below_min_flows():
    """Need ≥ MIN_FLOWS auth flows to trigger promotion."""
    result = DeepScanResult(
        features={"i18n": ["web/app/signin/page.tsx"]},
        flows={"i18n": ["signin-flow"]},  # only 1 flow
        descriptions={"i18n": "..."},
    )
    promos = plan_promotions(result)
    assert promos == []


def test_plan_skips_when_no_matching_paths():
    """If owner has no auth-named paths, skip — we'd have nothing to move."""
    result = DeepScanResult(
        features={"i18n": ["web/i18n/en.ts", "web/components/x.tsx"]},
        flows={
            "i18n": [
                "signin-flow",
                "signup-flow",
                "logout-flow",
            ],
        },
        descriptions={"i18n": "..."},
    )
    promos = plan_promotions(result)
    assert promos == []


# ── apply_promotions ──────────────────────────────────────────────────


def test_apply_creates_auth_feature_and_moves_paths():
    result = _result_dify_minimal()
    n = promote_virtual_clusters(result)
    assert n == 1
    assert "auth" in result.features
    auth_paths = result.features["auth"]
    assert "web/app/signin/page.tsx" in auth_paths
    assert "web/app/reset-password/page.tsx" in auth_paths
    assert "web/types/auth/login-types.ts" in auth_paths
    # Original owners have non-auth paths intact.
    assert "web/i18n/en-US/translation.ts" in result.features["i18n"]
    assert "web/components/Button.tsx" in result.features["ui"]
    assert "web/types/api.ts" in result.features["contracts"]
    # And their auth paths are gone.
    assert "web/app/signin/page.tsx" not in result.features["i18n"]
    assert "web/app/forgot-password/page.tsx" not in result.features["ui"]


def test_apply_moves_flows_to_new_feature():
    result = _result_dify_minimal()
    promote_virtual_clusters(result)
    auth_flows = result.flows["auth"]
    assert "reset-password-with-email-flow" in auth_flows
    assert "login-flow" in auth_flows
    assert "forgot-password-flow" in auth_flows
    # Non-auth flows stay where they were.
    assert "change-language-flow" in result.flows["i18n"]
    assert "create-workflow-flow" in result.flows["workflow"]
    # Auth flows removed from original owners.
    assert "reset-password-with-email-flow" not in result.flows["i18n"]
    assert "login-flow" not in result.flows["contracts"]


def test_apply_migrates_flow_descriptions():
    result = _result_dify_minimal()
    result.flow_descriptions["i18n"] = {
        "reset-password-with-email-flow": "user resets password via email",
    }
    promote_virtual_clusters(result)
    assert "auth" in result.flow_descriptions
    assert (
        result.flow_descriptions["auth"]["reset-password-with-email-flow"]
        == "user resets password via email"
    )
    assert (
        "reset-password-with-email-flow" not in result.flow_descriptions.get("i18n", {})
    )


def test_apply_migrates_flow_participants():
    result = _result_dify_minimal()
    result.flow_participants["i18n"] = {
        "reset-password-with-email-flow": [{"path": "web/app/reset-password/page.tsx"}],
    }
    promote_virtual_clusters(result)
    assert "auth" in result.flow_participants
    assert (
        result.flow_participants["auth"]["reset-password-with-email-flow"]
        == [{"path": "web/app/reset-password/page.tsx"}]
    )


def test_apply_synthesises_description():
    result = _result_dify_minimal()
    promote_virtual_clusters(result)
    assert "auth" in result.descriptions
    assert "uthentication" in result.descriptions["auth"]


def test_apply_idempotent():
    """Running twice does not duplicate paths / flows."""
    result = _result_dify_minimal()
    promote_virtual_clusters(result)
    paths_first = list(result.features["auth"])
    flows_first = list(result.flows["auth"])
    promote_virtual_clusters(result)  # second time should be no-op
    # Auth feature now exists in menu, so promotion is skipped.
    assert result.features["auth"] == paths_first
    assert result.flows["auth"] == flows_first


def test_promote_returns_zero_when_nothing_to_do():
    result = DeepScanResult(
        features={"workflow": ["web/app/workflow/editor.tsx"]},
        flows={"workflow": ["create-workflow-flow"]},
        descriptions={"workflow": "..."},
    )
    assert promote_virtual_clusters(result) == 0


# ── DOMAIN_TOKENS sanity ──────────────────────────────────────────────


def test_domain_tokens_have_no_overlap():
    """Each token belongs to exactly one domain."""
    seen: dict[str, str] = {}
    for domain, tokens in DOMAIN_TOKENS.items():
        for tok in tokens:
            assert tok not in seen, (
                f"token {tok!r} duplicated in domains "
                f"{seen.get(tok)} and {domain}"
            )
            seen[tok] = domain
