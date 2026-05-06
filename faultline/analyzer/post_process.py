"""Post-process pipeline for cleaning up the FeatureMap before write/return.

This is the integrated version of ``scripts/cleanup_feature_map.py``,
operating directly on Pydantic ``FeatureMap`` and ``Feature`` objects
instead of raw JSON dicts. Hooked into ``cli.py`` after
``build_feature_map`` returns the fully-populated map.

Stages (in order):
  1. ``merge_sub_features`` — collapse slash/hyphen sub-feature artifacts
  2. ``reattribute_noise_files`` — move real code out of shared-infra /
     uncategorized buckets back into the matching real feature
  3. ``refine_by_path_signal`` — drop foreign files (mis-clustered files
     that live in another feature's path domain)
  4. ``extract_overlooked_top_dirs`` — re-extract Go-style top-level
     packages that the LLM merged into the wrong bucket
  5. ``commit_prefix_enrichment_pass`` — mine ``feat(scope):`` /
     ``<domain>:`` prefixes from local git log and create new features
     for well-attested vocabulary words missing from detection
  6. Drop noise / vendored / phantom / mega-bucket / triple-slug
  7. Filter marketing flows out of documentation

All stages are pure transformations and safe to run repeatedly.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from faultline.models.types import Feature, FeatureMap, Flow

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

_VENDOR_PREFIXES = ("external-crates/", "vendor/", "third_party/", "node_modules/")
_NOISE_NAMES = frozenset({"shared-infra", "Shared Infra", "uncategorized"})
_DOC_PROTECTED = frozenset({"documentation"})

_VAGUE_NAMES = frozenset({
    "catalog", "common", "generic", "system", "root", "init", "main",
    "base", "misc", "stuff", "platform", "backend", "frontend", "api", "lib",
})

_MARKETING_FLOW_PATTERNS = [
    r"book-meeting", r"subscribe-to-newsletter", r"apply-to-",
    r"view-go-page", r"view-pricing", r"sign-up-for-", r"contact-sales",
]

_COMMIT_TYPE_WORDS = frozenset({
    "fix", "feat", "chore", "docs", "doc", "refactor", "test", "tests",
    "build", "ci", "perf", "style", "revert", "wip", "deps", "dep",
    "security", "release", "merge", "bump", "update", "rename", "remove",
    "feature", "ref", "hotfix", "patch", "minor", "major", "tick",
    "readme", "license", "changelog", "deps-dev",
})

_GO_GENERIC_PREFIXES = frozenset({
    "src", "lib", "app", "cmd", "test", "tests", "internal", "pkg",
    "vendor", "third_party", "external", "build", "scripts", "docs",
    "examples",
})

_LAYER_DIR_PREFIXES = frozenset({
    "routers", "routes", "services", "service", "models", "modules",
    "controllers", "views", "pages", "components", "contrib", "store",
    "stores", "schemas", "serializers", "middleware", "handlers",
    "web_src", "frontend", "backend", "providers", "utils", "helpers",
    "types", "core", "common", "shared", "config", "configs",
    "ml", "llama", "llm", "kvcache", "fs", "harmony", "tokenizer",
    "format", "progress", "version", "manifest", "readline", "wintray",
    "parser", "runner", "thinking",
})


# Sprint 16 — domain prefixes owned by flow_cluster's Layer A. Commit-
# prefix mining must NOT create a parallel feature with these names
# because Layer A already groups domain-correct paths under the
# canonical domain feature (auth/billing/notifications) or the
# workspace-named carrier (dify-web/i18n).
_DOMAIN_PROTECTED_PREFIXES = frozenset({
    "auth", "authentication", "authn", "authz",
    "billing", "subscription", "subscriptions", "payment", "payments",
    "notification", "notifications", "alert", "alerts",
    "i18n", "i10n", "intl", "locale", "locales", "translation", "translations",
})


# ── Helpers ──────────────────────────────────────────────────────────

def _top_prefix(path: str) -> str:
    return path.split("/")[0] if "/" in path else path


def _two_level_prefix(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else ""


def _collapse_triple_slug(name: str) -> str:
    parts = name.split("/")
    out: list[str] = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return "/".join(out)


def _is_uncategorized(name: str) -> bool:
    return name == "uncategorized" or name.endswith("/uncategorized")


def _is_vendored(paths: list[str]) -> bool:
    if not paths:
        return False
    return all(
        any(p.startswith(v) or f"/{v}" in p for v in _VENDOR_PREFIXES)
        for p in paths
    )


def _is_marketing_flow(flow_name: str) -> bool:
    n = flow_name.lower()
    return any(re.search(p, n) for p in _MARKETING_FLOW_PATTERNS)


def _empty_feature(name: str) -> Feature:
    return Feature(
        name=name,
        paths=[],
        authors=[],
        total_commits=0,
        bug_fixes=0,
        bug_fix_ratio=0.0,
        last_modified=datetime.now(timezone.utc),
        health_score=80.0,
        flows=[],
    )


# ── Stage 1: merge sub-features ──────────────────────────────────────

def merge_sub_features(features: list[Feature]) -> list[Feature]:
    """Merge same-domain sub-features into one parent feature.

    Two patterns:
      - Slash-prefix workspace over-split (``milli/update`` + ``milli/search``
        + ``milli/heed_codec`` → ``milli`` when ≥3 share prefix and combined
        size <500). Skipped for big monorepo apps (supabase studio).
      - Hyphen-prefix sub-decompose artifacts (``order-order`` +
        ``order-graphql`` → ``order``) when bare prefix existed in the
        original feature list.
    """
    if not features:
        return features
    by_name: dict[str, Feature] = {f.name: f for f in features}
    original_names = set(by_name.keys())

    # Pattern 2: slash-prefix groups
    slash_groups: dict[str, list[Feature]] = {}
    for f in features:
        if "/" in f.name:
            prefix = f.name.split("/")[0]
            slash_groups.setdefault(prefix, []).append(f)

    slash_merged: set[str] = set()
    slash_synth_prefixes: set[str] = set()
    slash_results: list[Feature] = []
    for prefix, group in slash_groups.items():
        if len(group) < 3:
            continue
        combined = sum(len(f.paths) for f in group)
        if combined >= 500:
            continue
        if prefix in by_name:
            continue
        biggest = max(group, key=lambda x: len(x.paths))
        all_paths: list[str] = []
        all_authors: list[str] = []
        all_flows: list[Flow] = []
        total_commits = 0
        bug_fixes = 0
        for f in group:
            all_paths.extend(f.paths)
            all_authors.extend(f.authors)
            all_flows.extend(f.flows)
            total_commits += f.total_commits
            bug_fixes += f.bug_fixes
            slash_merged.add(f.name)

        merged = biggest.model_copy(deep=False)
        merged.name = prefix
        merged.display_name = prefix.replace("-", " ").title()
        merged.paths = list(dict.fromkeys(all_paths))
        merged.authors = list(dict.fromkeys(all_authors))
        merged.flows = all_flows
        merged.total_commits = total_commits
        merged.bug_fixes = bug_fixes
        merged.bug_fix_ratio = bug_fixes / max(total_commits, 1)
        slash_results.append(merged)
        slash_synth_prefixes.add(prefix)

    features = [f for f in features if f.name not in slash_merged] + slash_results
    by_name = {f.name: f for f in features}

    # Pattern 1: hyphen-prefix groups
    groups: dict[str, list[Feature]] = {}
    for f in features:
        if f.name in slash_synth_prefixes:
            groups.setdefault(f.name, []).append(f)
            continue
        if "-" in f.name and "/" not in f.name:
            prefix = f.name.split("-")[0]
            if prefix in slash_synth_prefixes:
                groups.setdefault(f.name, []).append(f)
                continue
            groups.setdefault(prefix, []).append(f)
        else:
            groups.setdefault(f.name, []).append(f)

    merged: list[Feature] = []
    for prefix, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        if prefix not in original_names:
            for f in group:
                merged.append(f)
            continue
        canonical = by_name.get(prefix)
        if canonical is None:
            for f in group:
                merged.append(f)
            continue

        all_paths: list[str] = []
        all_authors: list[str] = []
        all_flows: list[Flow] = []
        total_commits = 0
        bug_fixes = 0
        for f in group:
            all_paths.extend(f.paths)
            all_authors.extend(f.authors)
            all_flows.extend(f.flows)
            total_commits += f.total_commits
            bug_fixes += f.bug_fixes
        out = canonical.model_copy(deep=False)
        out.name = prefix
        out.display_name = prefix.replace("-", " ").title()
        out.paths = list(dict.fromkeys(all_paths))
        out.authors = list(dict.fromkeys(all_authors))
        out.flows = all_flows
        out.total_commits = total_commits
        out.bug_fixes = bug_fixes
        out.bug_fix_ratio = bug_fixes / max(total_commits, 1)
        merged.append(out)

    return merged


# ── Stage 2: re-attribute noise files ────────────────────────────────

def reattribute_noise_files(features: list[Feature]) -> tuple[list[Feature], int, int]:
    noise = [
        f for f in features
        if f.name in _NOISE_NAMES or _is_uncategorized(f.name)
    ]
    if not noise:
        return features, 0, 0
    kept = [f for f in features if f not in noise]
    if not kept:
        return features, 0, 0

    prefix_to_features: dict[str, dict[str, int]] = {}
    for f in kept:
        for p in f.paths:
            d = _two_level_prefix(p)
            if not d:
                continue
            bucket = prefix_to_features.setdefault(d, {})
            bucket[f.name] = bucket.get(f.name, 0) + 1

    feature_by_name = {f.name: f for f in kept}
    reattributed = 0
    truly_dropped = 0

    for nf in noise:
        for path in list(nf.paths):
            d = _two_level_prefix(path)
            scores = prefix_to_features.get(d, {})
            if not scores:
                truly_dropped += 1
                continue
            winner_name, winner_count = max(scores.items(), key=lambda x: x[1])
            if winner_count < 3:
                truly_dropped += 1
                continue
            target = feature_by_name[winner_name]
            target.paths.append(path)
            reattributed += 1

    return kept, reattributed, truly_dropped


# ── Stage 3: foreign-file removal (Rule 1 only) ──────────────────────

def refine_by_path_signal(features: list[Feature]) -> tuple[list[Feature], int]:
    prefix_owner: dict[str, str] = {}
    prefix_counts: dict[str, dict[str, int]] = {}
    for f in features:
        for p in f.paths:
            pre = _top_prefix(p)
            bucket = prefix_counts.setdefault(pre, {})
            bucket[f.name] = bucket.get(f.name, 0) + 1
    for pre, owners in prefix_counts.items():
        winner, count = max(owners.items(), key=lambda x: x[1])
        total = sum(owners.values())
        if count / total >= 0.6:
            prefix_owner[pre] = winner

    removed_total = 0
    for f in features:
        if len(f.paths) < 4:
            continue
        prefix_dist = Counter(_top_prefix(p) for p in f.paths)
        total = len(f.paths)
        kept_paths: list[str] = []
        for p in f.paths:
            pre = _top_prefix(p)
            owner = prefix_owner.get(pre)
            count_here = prefix_dist[pre]
            if owner and owner != f.name and count_here / total < 0.1:
                removed_total += 1
                continue
            kept_paths.append(p)
        f.paths = kept_paths
    return features, removed_total


# ── Stage 4: extract overlooked top-dirs ─────────────────────────────

def _mine_commit_prefixes(repo_path: str, min_count: int = 5) -> dict[str, int]:
    """Extract domain prefixes from local git log subjects."""
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "log", "--pretty=%s", "--since=2 years ago"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return {}

    pat_bare = re.compile(r"^([a-z][a-z0-9_/-]+):")
    pat_scope = re.compile(r"^[a-z]+\(([a-z][a-z0-9_/-]+)\):", re.IGNORECASE)
    counts: dict[str, int] = {}
    for line in r.stdout.splitlines():
        m = pat_scope.match(line)
        if m:
            prefix = m.group(1).lower()
            if prefix not in _COMMIT_TYPE_WORDS:
                counts[prefix] = counts.get(prefix, 0) + 1
            continue
        m = pat_bare.match(line)
        if m:
            prefix = m.group(1)
            if prefix in _COMMIT_TYPE_WORDS:
                continue
            counts[prefix] = counts.get(prefix, 0) + 1

    return {p: c for p, c in counts.items() if c >= min_count}


def extract_overlooked_top_dirs(
    features: list[Feature],
    commit_prefixes: dict[str, int] | None = None,
) -> tuple[list[Feature], list[str]]:
    feature_names = {f.name for f in features}
    log_lines: list[str] = []
    new_features: list[Feature] = []

    for f in features:
        if len(f.paths) < 10:
            continue
        prefix_dist = Counter(_top_prefix(p) for p in f.paths)
        if len(prefix_dist) < 2:
            continue
        own_name = f.name

        for prefix, count in list(prefix_dist.items()):
            if count < 3:
                continue
            if prefix == own_name or prefix in feature_names:
                continue
            if prefix in _GO_GENERIC_PREFIXES or prefix in _LAYER_DIR_PREFIXES:
                continue
            if prefix.lower() in own_name.lower() or own_name.lower() in prefix.lower():
                continue
            commit_count = (commit_prefixes or {}).get(prefix, 0)
            if count / len(f.paths) >= 0.7 and commit_count < 10:
                continue

            extracted_paths = [p for p in f.paths if _top_prefix(p) == prefix]
            ratio = count / len(f.paths)
            tc = f.total_commits
            bf = f.bug_fixes

            new_f = f.model_copy(deep=False)
            new_f.name = prefix
            new_f.display_name = prefix.replace("-", " ").title()
            new_f.description = (
                f"Files under {prefix}/ extracted from {own_name} "
                "(auto-split for granularity)"
            )
            new_f.paths = extracted_paths
            new_f.total_commits = int(tc * ratio)
            new_f.bug_fixes = int(bf * ratio)
            new_f.bug_fix_ratio = (bf / max(tc, 1)) if tc else 0.0
            new_f.flows = []
            new_features.append(new_f)
            feature_names.add(prefix)
            log_lines.append(
                f"extracted '{prefix}' ({count} files) out of '{own_name}'"
            )

            f.paths = [p for p in f.paths if _top_prefix(p) != prefix]
            f.total_commits = tc - new_f.total_commits
            f.bug_fixes = bf - new_f.bug_fixes
            f.bug_fix_ratio = (
                f.bug_fixes / max(f.total_commits, 1) if f.total_commits else 0.0
            )
            if len(f.paths) < 10:
                break

    return features + new_features, log_lines


# ── Stage 5: commit-prefix enrichment ────────────────────────────────

def _collect_files_for_commit_prefix(repo_path: str, prefix: str) -> set[str]:
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "log",
             "--pretty=format:COMMIT:%s", "--name-only", "--no-renames",
             "--since=2 years ago"],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return set()

    pat = re.compile(
        rf"^([a-z]+\({re.escape(prefix)}\)|{re.escape(prefix)}):", re.IGNORECASE,
    )
    files: set[str] = set()
    in_match = False
    for line in r.stdout.splitlines():
        if line.startswith("COMMIT:"):
            in_match = bool(pat.match(line[7:]))
        elif in_match and line.strip():
            files.add(line)
    return files


def commit_prefix_enrichment_pass(
    features: list[Feature],
    repo_path: str,
    commit_prefixes: dict[str, int],
) -> tuple[list[Feature], list[str]]:
    if not commit_prefixes or not repo_path:
        return features, []

    log_lines: list[str] = []
    feature_names = {f.name for f in features}
    # Sprint 16 — token lookup over feature names. Splits each name on
    # ``/`` and ``-`` so ``dify-web/i18n`` covers ``i18n``, ``dify-ui``
    # covers ``ui``, ``packages/auth`` covers ``auth``. Prevents
    # commit_prefix_enrichment_pass from creating a parallel feature
    # that cannibalises Layer A's path attribution.
    feature_tokens: set[str] = set()
    for _n in feature_names:
        for tok in re.split(r"[/\-]", _n.lower()):
            if tok:
                feature_tokens.add(tok)
    path_to_feature: dict[str, Feature | None] = {}
    for f in features:
        for p in f.paths:
            path_to_feature[p] = f

    new_features: list[Feature] = []
    for prefix, ccount in commit_prefixes.items():
        if ccount < 10 or "/" in prefix:
            continue
        if prefix in feature_names:
            continue
        if any(n.startswith(prefix + "/") for n in feature_names):
            continue
        # Sprint 16 — domain prefixes (auth, billing, notifications,
        # i18n) are owned by Layer A's flow_cluster; let it manage
        # them, don't create a parallel commit-mined twin that
        # cannibalises Layer A's path attribution.
        if prefix.lower() in _DOMAIN_PROTECTED_PREFIXES:
            continue
        # Sprint 16 — also block when the prefix matches any token of
        # an existing feature name (workspace monorepos: ``dify-web/i18n``
        # blocks ``i18n``, ``dify-ui`` blocks ``ui``).
        if prefix.lower() in feature_tokens:
            continue
        if prefix in _COMMIT_TYPE_WORDS:
            continue
        if prefix in _LAYER_DIR_PREFIXES or prefix in _GO_GENERIC_PREFIXES:
            continue

        touched = _collect_files_for_commit_prefix(repo_path, prefix)
        present = touched & path_to_feature.keys()
        if len(present) < 10:
            continue

        path3 = Counter("/".join(p.split("/")[:3]) for p in present)
        top_pre, top_cnt = path3.most_common(1)[0]
        if top_cnt / len(present) < 0.4:
            continue
        if "docs" in top_pre.lower() or top_pre.startswith("apps/docs"):
            continue

        feature_dist = Counter(
            (path_to_feature[p].name if path_to_feature[p] else "")
            for p in present
        )
        primary_origin, primary_count = feature_dist.most_common(1)[0]
        if primary_count / len(present) > 0.85:
            continue

        # Transfer files
        transferred_paths: list[str] = []
        origins_touched: set[str] = set()
        for p in list(present):
            origin = path_to_feature.get(p)
            if origin is None:
                continue
            if p in origin.paths:
                origin.paths.remove(p)
            transferred_paths.append(p)
            origins_touched.add(origin.name)
            path_to_feature[p] = None

        # Transfer flows
        transferred_set = set(transferred_paths)
        transferred_flows: list[Flow] = []
        for f in features:
            if f.name not in origins_touched:
                continue
            kept_flows: list[Flow] = []
            for flow in f.flows:
                if not flow.paths:
                    kept_flows.append(flow)
                    continue
                overlap = sum(1 for p in flow.paths if p in transferred_set)
                if overlap / len(flow.paths) >= 0.6:
                    transferred_flows.append(flow)
                else:
                    kept_flows.append(flow)
            f.flows = kept_flows

        new_f = _empty_feature(prefix)
        new_f.display_name = prefix.replace("-", " ").title()
        new_f.description = f"Detected from {ccount} commits with '{prefix}:' prefix."
        new_f.paths = transferred_paths
        new_f.total_commits = ccount
        new_f.flows = transferred_flows
        new_features.append(new_f)
        feature_names.add(prefix)
        for p in transferred_paths:
            path_to_feature[p] = new_f
        log_lines.append(
            f"new feature '{prefix}' from commit prefix ({ccount} commits, "
            f"{len(transferred_paths)} files concentrated in {top_pre}/)"
        )

    return features + new_features, log_lines


# ── Stage 6: drop noise / vendored / phantom / mega-bucket ──────────

def drop_noise_features(features: list[Feature]) -> tuple[list[Feature], list[tuple[str, str, int]]]:
    cleaned: list[Feature] = []
    dropped: list[tuple[str, str, int]] = []
    total_files = sum(len(f.paths) for f in features)

    for f in features:
        name = f.name
        path_count = len(f.paths)

        # 1. shared-infra / Shared Infra
        if name in _NOISE_NAMES:
            dropped.append((name, "shared-infra/noise", path_count))
            continue

        # 2. uncategorized slug-leak
        if _is_uncategorized(name):
            dropped.append((name, "uncategorized catch-all", path_count))
            continue

        # 3. vendored
        if _is_vendored(f.paths):
            dropped.append((name, "vendored third-party", path_count))
            continue

        # 4. phantom (≤2 files AND ≤1 commit)
        if path_count <= 2 and f.total_commits <= 1:
            dropped.append((name, "phantom (no real activity)", path_count))
            continue

        # 5. mega-bucket — only when name is vague
        if (
            name in _VAGUE_NAMES
            and total_files > 0
            and path_count > 0.25 * total_files
            and path_count > 100
        ):
            dropped.append(
                (name, f"mega-bucket {path_count}/{total_files} files", path_count)
            )
            continue

        # 6. triple-slug collapse
        new_name = _collapse_triple_slug(name)
        if new_name != name:
            f.name = new_name
            if f.display_name:
                f.display_name = _collapse_triple_slug(f.display_name)

        # 7. filter marketing flows out of documentation
        if f.name == "documentation" and f.flows:
            f.flows = [fl for fl in f.flows if not _is_marketing_flow(fl.name)]

        cleaned.append(f)

    cleaned = [f for f in cleaned if f.paths]
    cleaned.sort(key=lambda x: -x.total_commits)
    return cleaned, dropped


# ── Public entry point ──────────────────────────────────────────────

def run(
    feature_map: FeatureMap,
    repo_path: str | None = None,
) -> FeatureMap:
    """Run all post-process stages on a FeatureMap. Returns a new map.

    Idempotent — safe to run multiple times.
    """
    features = list(feature_map.features)

    features = merge_sub_features(features)

    features, reattributed, truly_dropped = reattribute_noise_files(features)
    if reattributed:
        logger.info(
            "post_process: re-attributed %d files from noise buckets (%d truly dropped)",
            reattributed, truly_dropped,
        )

    features, removed = refine_by_path_signal(features)
    if removed:
        logger.info("post_process: removed %d foreign files", removed)

    commit_prefixes: dict[str, int] = {}
    if repo_path and Path(repo_path).is_dir():
        commit_prefixes = _mine_commit_prefixes(repo_path)

    features, extract_log = extract_overlooked_top_dirs(features, commit_prefixes)
    for line in extract_log:
        logger.info("post_process: %s", line)

    if commit_prefixes and repo_path:
        features, enrich_log = commit_prefix_enrichment_pass(
            features, repo_path, commit_prefixes,
        )
        for line in enrich_log:
            logger.info("post_process: %s", line)

    features, dropped = drop_noise_features(features)
    for name, reason, n in dropped:
        logger.info("post_process: dropped %s (%s, %d files)", name, reason, n)

    feature_map.features = features
    return feature_map
