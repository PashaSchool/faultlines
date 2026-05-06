"""Sprint 12 Day 2 — Virtual cluster promotion (Layer A).

Some repos lack a domain-named feature for a coherent cluster of flows.
Example (dify): there is no ``auth`` feature, so 20 auth flows
(``reset-password-with-email-flow``, ``signup-email-verification-flow``,
``webapp-email-signin-flow`` ...) get attached to the catch-all bucket
that happens to own their entry-point file (``i18n``, ``ui``,
``contracts``). Sprint 11's ``flow_judge`` cannot fix this — it only
moves flows between *existing* features, never creates one.

This module synthesises a feature when a critical mass of flows share
a domain. The synthetic feature steals only the auth-named files from
each catch-all owner, never touches non-domain files, and never
deletes/renames a domain-specific feature.

Pure-function shape:

    DOMAIN_TOKENS                — registry of domain → trigger tokens
    classify_flow_domain(name)   — name → domain | None
    plan_promotions(result)      — dry-run, returns Promotion list
    apply_promotions(result, ps) — mutates DeepScanResult in place

Trigger rule for a promotion (all must hold):
    1. ≥ MIN_FLOWS flows match a domain
    2. NO existing feature name contains the domain token
    3. ≥ 1 path in current owners contains a domain token
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

MIN_FLOWS = 3  # at least this many flows in a domain to promote

# Sprint 15 Day 1 — backfill mode trigger. If an existing domain-named
# feature has fewer paths than this threshold AND ≥ MIN_FLOWS catch-all
# flows match the domain, we promote into the existing feature instead
# of skipping. Catches the dify case where Sonnet emitted a 5-path
# ``auth`` feature from a tiny package and Layer A wrongly considered
# the domain "covered".
MIN_DOMAIN_PATHS = 15

# Tokens are matched case-insensitively against normalised
# (``-``-joined) names and paths. Token must be substring (e.g.
# ``signin`` matches ``user-signin-flow`` and ``app/signin/page.tsx``).
DOMAIN_TOKENS: dict[str, tuple[str, ...]] = {
    "auth": (
        # Sprint 13 — broadened set after live dify scan showed
        # `webapp-auth-flow`, `authenticate-webapp-user`,
        # `verify-webapp-email-code` slipping through the original
        # signin/login/password vocabulary.
        "auth", "authenticate", "signin", "sign-in", "signup",
        "sign-up", "login", "logout", "password", "oauth", "sso",
        "saml", "oidc", "register", "verify-email",
        "email-verification", "reset-password", "forgot-password",
        "two-factor", "2fa", "mfa", "magic-link",
        "activate-invited", "activate-account", "session",
    ),
    "billing": (
        "billing", "subscription", "invoice", "payment",
        "checkout", "pricing", "stripe", "refund", "plan-upgrade",
        "downgrade-plan", "manage-plan",
    ),
    "notifications": (
        "notification", "alert-rule", "push-notification",
    ),
}

# Sprint 13 — features whose names don't literally contain a domain
# token but ARE the domain feature. Without this lookup, Layer A would
# wrongly create a synthetic ``auth`` feature next to an existing
# ``Account Settings`` / ``Identity Management`` / ``Access Control``.
_DOMAIN_FEATURE_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "auth": (
        "account", "identity", "access control", "access-control",
        "user management", "user-management", "session",
    ),
    "billing": (
        "subscription", "payment", "plan", "checkout",
    ),
    "notifications": (
        "alert", "alerting",
    ),
}


@dataclass
class Promotion:
    """One synthetic-feature creation order.

    ``flows_to_move`` is a list of ``(source_owner, flow_name)``
    pairs — needed because the same flow name may exist under
    multiple owners. ``paths_to_move`` is similarly a list of
    ``(source_owner, path)`` pairs.
    """

    domain: str
    flows_to_move: list[tuple[str, str]] = field(default_factory=list)
    paths_to_move: list[tuple[str, str]] = field(default_factory=list)
    # Sprint 15 — when set, backfill into THIS existing feature
    # instead of creating a new one named after the domain. Used
    # when Sonnet emitted a small domain-named feature and Layer A
    # needs to grow it from catch-all owners.
    target_feature: str | None = None

    @property
    def feature_name(self) -> str:
        return self.target_feature or self.domain  # "auth", "billing", etc.


def _normalise(s: str) -> str:
    """Lowercase + replace separators with ``-`` so token search is uniform."""
    return re.sub(r"[\s_/]+", "-", s.lower())


def classify_flow_domain(flow_name: str) -> str | None:
    norm = _normalise(flow_name)
    for domain, tokens in DOMAIN_TOKENS.items():
        for tok in tokens:
            if tok in norm:
                return domain
    return None


def _path_matches_domain(path: str, domain: str) -> bool:
    norm = _normalise(path)
    return any(tok in norm for tok in DOMAIN_TOKENS[domain])


def _menu_has_domain(features: dict[str, list[str]], domain: str) -> bool:
    """True if any existing feature name covers the domain.

    Direct hit: ``domain in name.lower()``.
    Hint hit:   any of ``_DOMAIN_FEATURE_NAME_HINTS[domain]`` in name.

    Sprint 13 — added hint match so ``Account Settings`` /
    ``Identity Management`` / ``Subscription Plans`` correctly count
    as the domain feature, blocking a redundant synthetic promotion.
    """
    hints = _DOMAIN_FEATURE_NAME_HINTS.get(domain, ())
    for name in features:
        nl = name.lower()
        if domain in nl:
            return True
        if any(h in nl for h in hints):
            return True
    return False


def _is_protected_owner(owner: str) -> bool:
    """Synthetic / catch-all buckets we never read paths from.

    These buckets are managed by the pipeline itself; carving paths out
    of them would corrupt downstream stages.
    """
    return owner in {
        "shared-infra",
        "documentation",
        "examples",
        "developer-infrastructure",
    }


def plan_promotions(result: DeepScanResult) -> list[Promotion]:
    """Inspect the scan result and decide which synthetic features to create.

    Returns one ``Promotion`` per eligible domain. No mutation; the
    caller decides whether to apply.
    """
    promotions: list[Promotion] = []

    # 1. Collect all flow occurrences grouped by domain.
    by_domain: dict[str, list[tuple[str, str]]] = {}
    for owner, flow_names in result.flows.items():
        for flow_name in flow_names:
            domain = classify_flow_domain(flow_name)
            if domain is not None:
                by_domain.setdefault(domain, []).append((owner, flow_name))

    for domain, occurrences in by_domain.items():
        # 3. Skip if not enough flows.
        unique_owners = {o for o, _ in occurrences}
        if len(occurrences) < MIN_FLOWS:
            continue

        # 2. Decide whether to create / backfill / skip based on what's
        # already in the menu (Sprint 15 — was just "skip if exists").
        existing = _find_domain_feature(result.features, domain)
        target_feature: str | None = None
        if existing is not None:
            existing_path_count = len(result.features.get(existing, []))
            # Catch-all flows are stranded if their owner is NOT the
            # domain feature itself.
            stranded = [o for o, _ in occurrences if o != existing]
            if existing_path_count >= MIN_DOMAIN_PATHS or len(stranded) < MIN_FLOWS:
                logger.debug(
                    "flow_cluster: skip domain=%s — feature %r already "
                    "covered (%d paths, %d stranded flows)",
                    domain, existing, existing_path_count, len(stranded),
                )
                continue
            # Backfill mode: existing feature is undersized AND there
            # are enough stranded flows to justify growing it.
            target_feature = existing
            logger.info(
                "flow_cluster: backfill mode for domain=%s into %r "
                "(%d existing paths, %d stranded flows)",
                domain, existing, existing_path_count, len(stranded),
            )

        # 4. For each owner of these flows, collect paths that match the
        #    domain. Skip protected (synthetic) owners AND the existing
        #    domain feature itself (we already own those paths).
        paths_to_move: list[tuple[str, str]] = []
        for owner in unique_owners:
            if _is_protected_owner(owner) or owner == target_feature:
                continue
            owner_paths = result.features.get(owner, [])
            for p in owner_paths:
                if _path_matches_domain(p, domain):
                    paths_to_move.append((owner, p))

        if not paths_to_move:
            logger.info(
                "flow_cluster: domain=%s has %d flows but no matching paths "
                "in current owners — skipping promotion",
                domain, len(occurrences),
            )
            continue

        # Filter flows: never move flows whose current owner IS the
        # backfill target (they're already there).
        flows_filtered = [
            (o, fn) for o, fn in occurrences if o != target_feature
        ]
        promotions.append(Promotion(
            domain=domain,
            flows_to_move=flows_filtered,
            paths_to_move=paths_to_move,
            target_feature=target_feature,
        ))
        logger.info(
            "flow_cluster: planning %s domain=%s, flows=%d, paths=%d",
            "backfill" if target_feature else "promotion",
            domain, len(flows_filtered), len(paths_to_move),
        )

    return promotions


def _find_domain_feature(
    features: dict[str, list[str]],
    domain: str,
) -> str | None:
    """Return the name of the existing feature that covers ``domain``,
    or ``None`` if no such feature exists in the menu.

    Sprint 15 — extracts the lookup logic from ``_menu_has_domain``
    so callers can also inspect the matching feature's path count.
    """
    hints = _DOMAIN_FEATURE_NAME_HINTS.get(domain, ())
    for name in features:
        nl = name.lower()
        if domain in nl:
            return name
        if any(h in nl for h in hints):
            return name
    return None


def apply_promotions(
    result: DeepScanResult,
    promotions: list[Promotion],
) -> int:
    """Mutate ``result`` in place.

    Returns count of features created (excluding backfills into
    existing features). For tests that count "did anything happen",
    use ``len(promotions)`` instead.
    """
    created = 0
    for promo in promotions:
        feat = promo.feature_name
        if feat in result.features:
            # Sprint 15 — backfill or defensive append. For backfills
            # this is the expected path. For accidental duplicates
            # (target_feature unset) we still merge silently.
            if promo.target_feature:
                logger.info(
                    "flow_cluster: backfilling %d paths + %d flows into "
                    "existing %r",
                    len(promo.paths_to_move),
                    len(promo.flows_to_move),
                    feat,
                )
            else:
                logger.warning(
                    "flow_cluster: feature %r already exists; merging",
                    feat,
                )
        else:
            result.features[feat] = []
            result.flows.setdefault(feat, [])
            created += 1

        # Move paths.
        for owner, path in promo.paths_to_move:
            owner_paths = result.features.get(owner, [])
            if path in owner_paths:
                owner_paths.remove(path)
            if path not in result.features[feat]:
                result.features[feat].append(path)

        # Move flows.
        moved_flows: dict[str, list[str]] = {}  # owner → flows removed
        for owner, flow_name in promo.flows_to_move:
            owner_flows = result.flows.get(owner, [])
            if flow_name in owner_flows:
                owner_flows.remove(flow_name)
                moved_flows.setdefault(owner, []).append(flow_name)
            if flow_name not in result.flows[feat]:
                result.flows[feat].append(flow_name)

        # Migrate flow_descriptions.
        for owner, flow_names in moved_flows.items():
            src_descs = result.flow_descriptions.get(owner, {})
            for fn in flow_names:
                desc = src_descs.pop(fn, None)
                if desc:
                    result.flow_descriptions.setdefault(feat, {})[fn] = desc

        # Migrate flow_participants.
        for owner, flow_names in moved_flows.items():
            src_parts = result.flow_participants.get(owner, {})
            for fn in flow_names:
                parts = src_parts.pop(fn, None)
                if parts is not None:
                    result.flow_participants.setdefault(feat, {})[fn] = parts

        # Synthesize a description if missing.
        if feat not in result.descriptions:
            result.descriptions[feat] = _DEFAULT_DESCRIPTIONS.get(
                promo.domain,
                f"{promo.domain.title()} feature (auto-promoted from flow cluster).",
            )

        logger.info(
            "flow_cluster: created %r with %d paths, %d flows",
            feat, len(promo.paths_to_move), len(promo.flows_to_move),
        )

    return created


_DEFAULT_DESCRIPTIONS: dict[str, str] = {
    "auth": "Authentication and account access — sign-in, sign-up, password recovery, email verification, session management.",
    "billing": "Billing and subscription management — plans, invoices, payments, refunds, checkout.",
    "notifications": "User notifications — in-app alerts, email notifications, notification preferences.",
}


def promote_virtual_clusters(result: DeepScanResult) -> int:
    """Convenience wrapper: plan + apply. Returns features created.

    This is the function the pipeline calls.
    """
    promotions = plan_promotions(result)
    if not promotions:
        return 0
    return apply_promotions(result, promotions)
