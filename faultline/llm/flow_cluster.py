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

# Tokens are matched case-insensitively against normalised
# (``-``-joined) names and paths. Token must be substring (e.g.
# ``signin`` matches ``user-signin-flow`` and ``app/signin/page.tsx``).
DOMAIN_TOKENS: dict[str, tuple[str, ...]] = {
    "auth": (
        "signin", "sign-in", "signup", "sign-up", "login", "logout",
        "password", "oauth", "register", "verify-email",
        "email-verification", "reset-password", "forgot-password",
        "two-factor", "activate-invited", "activate-account",
    ),
    "billing": (
        "billing", "subscription", "invoice", "payment",
        "checkout", "pricing", "stripe", "refund",
    ),
    "notifications": (
        "notification", "alert-rule",
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

    @property
    def feature_name(self) -> str:
        return self.domain  # "auth", "billing", etc.


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
    """True if any existing feature name contains the domain token."""
    for name in features:
        if domain in name.lower():
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
        # 2. Skip if a domain-named feature already exists.
        if _menu_has_domain(result.features, domain):
            logger.debug(
                "flow_cluster: skip domain=%s (feature already in menu)",
                domain,
            )
            continue
        # 3. Skip if not enough flows.
        unique_owners = {o for o, _ in occurrences}
        if len(occurrences) < MIN_FLOWS:
            continue

        # 4. For each owner of these flows, collect paths that match the
        #    domain. Skip protected (synthetic) owners.
        paths_to_move: list[tuple[str, str]] = []
        for owner in unique_owners:
            if _is_protected_owner(owner):
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

        promotions.append(Promotion(
            domain=domain,
            flows_to_move=list(occurrences),
            paths_to_move=paths_to_move,
        ))
        logger.info(
            "flow_cluster: planning promotion domain=%s, flows=%d, paths=%d",
            domain, len(occurrences), len(paths_to_move),
        )

    return promotions


def apply_promotions(
    result: DeepScanResult,
    promotions: list[Promotion],
) -> int:
    """Mutate ``result`` in place. Returns count of features created."""
    created = 0
    for promo in promotions:
        feat = promo.feature_name
        if feat in result.features:
            # Defensive — _menu_has_domain should have filtered this out,
            # but if a prior promotion created the feature, append to it.
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
