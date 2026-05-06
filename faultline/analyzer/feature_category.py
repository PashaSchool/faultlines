"""Sprint 16 polish — classify features into UX-friendly categories.

Drives the dashboard tiering: ``product`` features land in the main
carousel, ``supporting`` (ui-system / i18n / contracts / sdk) collapse
into a secondary section, ``hidden`` (tooling / documentation /
deployment / synthetic) only show when the user explicitly drills in.

Taxonomy derived from analysing 69 unique features across dify,
supabase, immich live scans (S12-S15). Rule-based, deterministic,
no LLM cost. Backwards-compatible — features without an explicit
category default to ``"product"``.

Public surface
==============

    Category               — Enum of 9 values
    classify_feature(name, paths) -> Category
    classify_categories(features) -> dict[name, Category]
    tier_for(category)     -> "product" | "supporting" | "hidden"
"""

from __future__ import annotations

import enum
import re
from typing import Iterable


class Category(str, enum.Enum):
    """Concrete taxonomy emerged from S12-S15 live scans."""

    PRODUCT = "product"             # user-facing business feature
    UI_SYSTEM = "ui-system"         # design system, primitives, icons, Storybook
    I18N = "i18n"                   # translations / locales
    CONTRACTS = "contracts"         # types, schemas, generated API code
    SDK = "sdk"                     # client libraries (Node SDK, CLI)
    TOOLING = "tooling"             # build configs, codemods, eslint, dev tools
    DOCUMENTATION = "documentation"  # markdown content
    DEPLOYMENT = "deployment"       # Docker, terraform, k8s manifests
    SYNTHETIC = "synthetic"         # pipeline-managed buckets (shared-infra, etc.)


_TIER_MAP: dict[Category, str] = {
    Category.PRODUCT: "product",
    Category.UI_SYSTEM: "supporting",
    Category.I18N: "supporting",
    Category.CONTRACTS: "supporting",
    Category.SDK: "supporting",
    Category.TOOLING: "hidden",
    Category.DOCUMENTATION: "hidden",
    Category.DEPLOYMENT: "hidden",
    Category.SYNTHETIC: "hidden",
}


def tier_for(category: Category) -> str:
    """``product`` / ``supporting`` / ``hidden`` for UI rendering."""
    return _TIER_MAP.get(category, "product")


# ── Rule data tables ──────────────────────────────────────────────────


# Synthetic buckets are pipeline-managed — never re-classify them.
_SYNTHETIC_NAMES: frozenset[str] = frozenset({
    "shared-infra", "examples", "developer-infrastructure",
})

# Strong product domain names. When the feature's last name segment
# matches one of these (or has it as a clear suffix/prefix), we treat
# it as PRODUCT regardless of path content. Reason: ``supabase auth``
# scoped to docs guides is still semantically the auth product, and
# users expect it in the main carousel.
_STRONG_PRODUCT_NAMES: frozenset[str] = frozenset({
    "auth", "authentication", "authn", "authz",
    "billing", "subscriptions", "subscription", "payments", "payment",
    "checkout", "signup", "signin", "login", "logout",
    "search", "dashboard", "notifications", "alerts", "profile",
})

# Names that signal documentation when matched whole or as last segment.
_DOC_NAME_PATTERNS = (
    "documentation", "docs", "www", "learn", "marketing",
)

# Names that signal i18n.
_I18N_NAME_PATTERNS = (
    "i18n", "i10n", "intl", "locales", "translations",
)

# Names that signal contracts / types / generated code.
_CONTRACTS_NAME_PATTERNS = (
    "contracts", "types", "api-types", "schemas", "pg-meta",
    "openapi", "graphql-codegen",
)

# Names that signal a UI library / design system.
_UI_NAME_PATTERNS = (
    "ui-library", "ui-patterns", "ui-kit", "design-system",
    "icons", "iconify", "stories",  # Storybook
)
_UI_NAME_TOKENS = (
    "ui",  # standalone
)

# Names that signal SDK / client.
_SDK_NAME_PATTERNS = (
    "sdk", "client", "cli", "rest-client", "websocket-client",
)

# Tooling — STRICT match. ``config`` and ``generator`` are too easy
# to substring-match against legitimate product names like
# ``plugin-configuration`` or ``code-generator-feature``, so we use
# token-precise matching only (last segment must equal or end with
# the pattern; substring is intentionally NOT enough).
_TOOLING_LAST_SEG_EQ = frozenset({
    "config", "eslint", "prettier", "tsconfig", "generator",
    "codemod", "dev-tools", "shared-data", "ai-commands", "scripts",
    "build-icons",
})
_TOOLING_LAST_SEG_SUFFIXES = (
    "-config", "-eslint", "-codemod", "-generator", "-prettier", "-tsconfig",
)
_TOOLING_LAST_SEG_PREFIXES = (
    "build-", "migrate-", "codegen-",
)

# Names that signal deployment / infra-as-code.
_DEPLOYMENT_NAME_PATTERNS = (
    "deployment", "infra", "docker", "k8s", "kubernetes",
    "terraform", "helm", "charts", "ansible",
)


# Path-share thresholds. We tested them against the 69 real features:
# 0.7 catches the obvious cases without misclassifying mixed buckets.
_PATH_SHARE_THRESHOLD = 0.7


def _matches_token(name: str, patterns: Iterable[str]) -> bool:
    """Substring + last-segment + token check.

    A pattern matches if:
      - it equals the lowercased name, OR
      - it appears as a substring, OR
      - it appears as any /-, /-, or _-separated token of the name.
    """
    nl = name.lower()
    if any(p == nl for p in patterns):
        return True
    if any(p in nl for p in patterns):
        return True
    tokens = re.split(r"[/\-_]", nl)
    return any(p in tokens for p in patterns)


def _path_share(paths: list[str], substrings: Iterable[str]) -> float:
    if not paths:
        return 0.0
    hits = sum(
        1 for p in paths
        if any(s in p.lower() for s in substrings)
    )
    return hits / len(paths)


def _looks_like_generated(paths: list[str]) -> bool:
    """≥70% paths are *.gen.ts, *.d.ts, /__generated__/, /generated/."""
    return _path_share(paths, (
        ".gen.ts", ".gen.js", ".d.ts", "/__generated__/", "/generated/",
        "/orpc.gen", ".pb.go", "_pb2.py",
    )) >= _PATH_SHARE_THRESHOLD


def _looks_like_translations(paths: list[str]) -> bool:
    return _path_share(paths, (
        "/i18n/", "/locales/", "/translations/",
    )) >= _PATH_SHARE_THRESHOLD


def _looks_like_docs(paths: list[str]) -> bool:
    if not paths:
        return False
    md_share = sum(
        1 for p in paths
        if p.lower().endswith((".md", ".mdx", ".rst"))
    ) / len(paths)
    return md_share >= _PATH_SHARE_THRESHOLD


def _looks_like_deployment(paths: list[str]) -> bool:
    return _path_share(paths, (
        "/dockerfile", ".dockerfile", "docker-compose",
        ".tf", "/terraform/", "/helm/", "/charts/", "/k8s/",
        "/kubernetes/", "/ansible/",
    )) >= 0.5  # deployment is mixed by nature, looser threshold


def _looks_like_storybook(name: str, paths: list[str]) -> bool:
    if "stories" in name.lower():
        return True
    if not paths:
        return False
    story_share = sum(
        1 for p in paths
        if ".stories." in p or "/stories/" in p
    ) / len(paths)
    return story_share >= 0.4


# ── Classification ────────────────────────────────────────────────────


# Sprint 16 polish — features auto-promoted by post_process'
# commit_prefix_enrichment_pass / extract_overlooked_top_dirs carry a
# tell-tale description prefix. They're not real product features —
# they're pipeline-managed leftovers we don't want in the main
# carousel.
_SYNTHETIC_DESCRIPTION_PREFIXES = (
    "Detected from ",      # commit_prefix_enrichment_pass
    "Files under ",         # extract_overlooked_top_dirs
)


def classify_feature(
    name: str,
    paths: list[str] | None = None,
    description: str | None = None,
) -> Category:
    """Pick the most specific category that matches.

    Order matters: synthetic > docs > i18n > contracts > tooling >
    deployment > sdk > ui-system > product (default).

    Earlier rules are STRICTER — only fire when patterns clearly hold.
    Anything ambiguous lands as ``product`` so it shows up where the
    user expects.
    """
    paths = paths or []
    nl = name.lower()
    last_seg = nl.rsplit("/", 1)[-1]

    # 1. Synthetic — exact membership only (hardcoded pipeline buckets).
    # Description-based synthetic detection is checked AFTER strong-
    # product preempt so commit-mined ``auth`` / ``billing`` features
    # still surface as products.
    if name in _SYNTHETIC_NAMES:
        return Category.SYNTHETIC

    # 1b. Strong product preemption — feature names that are obvious
    # product domains override every other rule (path-based docs /
    # i18n detection AND commit-mined-leftover detection). Real case
    # from supabase: ``auth`` (41 files, all .mdx guides, mined from
    # ``auth:`` commit prefix) is STILL the auth product. The user
    # expects it in the carousel.
    is_strong_product = (
        last_seg in _STRONG_PRODUCT_NAMES
        or any(
            last_seg == t or last_seg.endswith("-" + t) or last_seg.startswith(t + "-")
            for t in _STRONG_PRODUCT_NAMES
        )
    )
    if is_strong_product:
        return Category.PRODUCT

    # 1c. Pipeline-leftover detection — only after strong-product
    # preempt. Commit_prefix_enrichment_pass and
    # extract_overlooked_top_dirs both write tell-tale prefixes.
    if description and description.startswith(_SYNTHETIC_DESCRIPTION_PREFIXES):
        return Category.SYNTHETIC

    # 2. Documentation
    if _matches_token(nl, _DOC_NAME_PATTERNS) or _looks_like_docs(paths):
        return Category.DOCUMENTATION

    # 3. i18n / translations
    if _matches_token(nl, _I18N_NAME_PATTERNS) or _looks_like_translations(paths):
        return Category.I18N

    # 4. Contracts / types / generated
    if _matches_token(nl, _CONTRACTS_NAME_PATTERNS) or _looks_like_generated(paths):
        return Category.CONTRACTS

    # 5. Tooling — codemods, configs, generators, dev-tools.
    # STRICT: token-equality on last-segment splits. Substring would
    # mis-classify ``plugin-detail-and-configuration`` (a product
    # feature) as tooling because it contains ``config``.
    last_seg_tokens = re.split(r"[/\-_.]", last_seg)
    _TOOLING_TOKEN_SET = {
        "eslint", "config", "tsconfig", "prettier", "codemod",
    }
    if (
        last_seg in _TOOLING_LAST_SEG_EQ
        or last_seg.endswith(_TOOLING_LAST_SEG_SUFFIXES)
        or last_seg.startswith(_TOOLING_LAST_SEG_PREFIXES)
        or any(t in _TOOLING_TOKEN_SET for t in last_seg_tokens)
    ):
        return Category.TOOLING

    # 6. Deployment / IaC
    if _matches_token(nl, _DEPLOYMENT_NAME_PATTERNS) or _looks_like_deployment(paths):
        return Category.DEPLOYMENT

    # 7. SDK / client libraries
    if _matches_token(nl, _SDK_NAME_PATTERNS):
        return Category.SDK
    # Common SDK location: ``sdks/`` or ``clients/`` top-level dir
    if paths and _path_share(paths, ("sdks/", "clients/")) >= 0.5:
        return Category.SDK

    # 8. UI system — design tokens, primitives, Storybook, icons
    if _matches_token(nl, _UI_NAME_PATTERNS):
        return Category.UI_SYSTEM
    # Standalone "ui" or "*-ui" / "ui-*" suffix as a feature name
    if last_seg in {"ui", "ui-kit", "components"}:
        return Category.UI_SYSTEM
    if last_seg.endswith("-ui") or last_seg.startswith("ui-"):
        return Category.UI_SYSTEM
    if last_seg.endswith(("-components", "-component")):
        return Category.UI_SYSTEM
    if _looks_like_storybook(nl, paths):
        return Category.UI_SYSTEM

    # 9. Default
    return Category.PRODUCT


def classify_categories(
    features: dict[str, list[str]],
    descriptions: dict[str, str] | None = None,
) -> dict[str, Category]:
    """Bulk classify every feature. ``descriptions`` is optional —
    when supplied, a description starting with ``Detected from `` /
    ``Files under `` (commit_prefix_enrichment / extract_top_dirs
    markers) marks the feature SYNTHETIC."""
    descriptions = descriptions or {}
    return {
        name: classify_feature(name, paths, descriptions.get(name))
        for name, paths in features.items()
    }
