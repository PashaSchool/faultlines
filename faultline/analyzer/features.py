import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from faultline.models.types import Commit, Feature, FeatureMap, Flow, PullRequest, SymbolAttribution, TimelinePoint

_MAX_FILES_PER_BULK_COMMIT = 30  # commits touching more files than this are excluded (bulk ops)
_MIN_COCHANGE_COMMITS = 2        # minimum shared commits for a pair to count
_MIN_COCHANGE_SCORE = 0.25       # minimum Jaccard coupling score
_MAX_COCHANGE_PAIRS = 40         # max pairs to return (sorted by score desc)


def compute_cochange(commits: list[Commit]) -> list[tuple[str, str, float]]:
    """
    Computes file co-change coupling from commit history.

    Returns top file pairs sorted by coupling score (descending).
    Coupling score (Jaccard) = commits_touching_both / commits_touching_either.
    Scores >= 0.25 are strong signals that files belong to the same feature.

    Commits touching more than _MAX_FILES_PER_BULK_COMMIT files are excluded
    to avoid noise from bulk operations (formatting, large refactors).
    """
    filtered = [c for c in commits if len(c.files_changed) <= _MAX_FILES_PER_BULK_COMMIT]

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    file_counts: dict[str, int] = defaultdict(int)

    for commit in filtered:
        files = commit.files_changed
        for f in files:
            file_counts[f] += 1
        for f1, f2 in combinations(sorted(files), 2):
            pair_counts[(f1, f2)] += 1

    results: list[tuple[str, str, float]] = []
    for (f1, f2), count in pair_counts.items():
        if count < _MIN_COCHANGE_COMMITS:
            continue
        union = file_counts[f1] + file_counts[f2] - count
        score = count / union if union > 0 else 0.0
        if score >= _MIN_COCHANGE_SCORE:
            results.append((f1, f2, round(score, 2)))

    return sorted(results, key=lambda x: x[2], reverse=True)[:_MAX_COCHANGE_PAIRS]


_MAX_FEATURE_FILES = 40  # features larger than this get split

# ── Candidate-based feature detection ──────────────────────────────────────
# Instead of asking the LLM to group all files from scratch, we first extract
# "feature candidates" using structural heuristics (anchor directories, naming
# conventions), then ask the LLM to verify / merge / split them.

# Directories whose *children* are individual feature modules (anchor dirs).
# e.g. backend/routers/chat.py → candidate "chat"
_ANCHOR_DIRS = {
    "routers", "routes", "endpoints", "controllers", "views",
    "pages", "screens",
    "features",
    "apps",  # Django apps
    "blueprints",  # Flask
    "handlers",
}

# "Deep anchor" dirs — their *subdirectories* are feature candidates.
# e.g. components/graphql/QueryEditor.tsx → candidate "graphql"
# Unlike _ANCHOR_DIRS (which look at filename stem), these look at the
# first subdirectory after the deep-anchor dir.
_DEEP_ANCHOR_DIRS = {
    "components", "containers", "modules",
    "store", "stores", "composables",
    "services",
}

# Directories that contain supporting files — matched to candidates by stem.
# e.g. backend/services/chat_executor.py → stem "chat" → candidate "chat"
_SUPPORT_DIRS = {
    "services", "service",
    "models", "schemas", "serializers",
    "utils", "helpers", "lib",
    "agent", "agents", "skills",
    "tasks", "jobs", "workers",
    "middleware",
    "hooks", "stores", "context", "contexts", "providers",
    "components",
    "queries", "mutations",
}

# Files in these dirs are infra, not features
_INFRA_DIRS = {
    "tests", "test", "__tests__", "migrations", "fixtures",
    "scripts", "ci", ".github", "docs", "static", "assets",
    "templates", "locale", "management",
}

# Technical layer dirs that should NOT become features themselves.
# Files in these get redistributed to business features or go to "shared-infra".
_TECH_LAYER_DIRS = {
    "prisma", "drizzle", "graphql-schema",
    ".yarn", ".changeset", ".claude", ".agents", ".opencode",
    ".devcontainer", ".snaplet", ".storybook",
    "docker", "deploy", "infra", "terraform",
}

# Dirs that look technical but contain business logic via subdirectories.
# trpc/server/auth-router/ → "auth", trpc/server/document-router/ → "document"
# These are treated as deep anchors when they have *-router subdirs.
_ROUTER_SUFFIX_DIRS = {"trpc", "server"}

# Config/infra files at root level — skip entirely
_INFRA_EXTENSIONS = {".toml", ".lock", ".cfg", ".ini", ".yml", ".yaml", ".json", ".md", ".txt", ".rst"}
_INFRA_FILENAMES = {
    "dockerfile", "docker-compose.yml", "makefile", "caddyfile",
    ".gitignore", ".env", ".env.example", "readme.md",
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "package.json", "package-lock.json", "tsconfig.json",
    "vite.config.ts", "next.config.js", "tailwind.config.js",
}


def detect_candidates(files: list[str]) -> dict[str, list[str]]:
    """
    Extracts feature candidates from file paths using structural heuristics.

    Strategy:
    1. Find "anchor" files — files in routers/, pages/, features/ etc.
       Each anchor file's stem becomes a candidate name.
    2. Find "support" files — files in services/, models/, schemas/ etc.
       Match them to candidates by stem overlap.
    3. Remaining files — group by first meaningful directory (existing heuristic).

    Returns:
        dict mapping candidate names to lists of file paths.
        Candidates are prefixed with their top-level dir for disambiguation
        (e.g. "backend:chat", "frontend:integrations").
    """
    candidates: dict[str, list[str]] = defaultdict(list)
    unmatched: list[str] = []

    # Track which anchor types each candidate was found through.
    # A real business feature appears in multiple anchor types (pages + components + stores).
    # A UI primitive appears only in deep-anchor dirs (only components/).
    candidate_anchor_types: dict[str, set[str]] = defaultdict(set)

    # Collect all anchor stems for matching support files later
    anchor_stems: dict[str, str] = {}  # stem → candidate_name

    # Phase 1: extract anchors
    for fp in files:
        if _is_infra_file(fp):
            continue

        parts = Path(fp).parts
        anchor_dir, stem = _find_anchor(parts)
        if anchor_dir and stem:
            candidate_name = stem
            candidates[candidate_name].append(fp)
            candidate_anchor_types[candidate_name].add(anchor_dir)
            for s in _stem_variants(stem):
                anchor_stems[s] = candidate_name
        else:
            unmatched.append(fp)

    # Phase 2: match support files to candidates by stem
    still_unmatched: list[str] = []
    for fp in unmatched:
        parts = Path(fp).parts
        matched_candidate = _match_support_file(parts, anchor_stems)
        if matched_candidate:
            candidates[matched_candidate].append(fp)
            # Track the support dir as an additional source layer
            for part in parts[:-1]:
                lower = part.lower()
                if lower in _SUPPORT_DIRS or lower in _DEEP_ANCHOR_DIRS:
                    candidate_anchor_types[matched_candidate].add(lower)
        else:
            still_unmatched.append(fp)

    # Phase 3: deeper matching for remaining files
    # Try filename stem, parent dir name, or hook pattern (useXxx → xxx)
    phase4_unmatched: list[str] = []
    for fp in still_unmatched:
        parts = Path(fp).parts
        matched = _deep_match_file(parts, anchor_stems)
        if matched:
            candidates[matched].append(fp)
            # Track ALL parent dirs as source layers — if ANY dir besides
            # components/ contains this candidate's files, it's multi-layer
            for part in parts[:-1]:
                lower = part.lower()
                if lower in _SUPPORT_DIRS or lower in _DEEP_ANCHOR_DIRS:
                    candidate_anchor_types[matched].add(lower)
                elif _normalize_stem(lower) == matched:
                    # Dir name matches candidate (e.g. stores/issues/ → "issues")
                    candidate_anchor_types[matched].add(lower)
        else:
            phase4_unmatched.append(fp)

    # Phase 4: group remaining files by meaningful directory
    for fp in phase4_unmatched:
        parts = Path(fp).parts
        feature_name = _extract_feature_name(parts)
        if feature_name == "root":
            stem = Path(fp).stem.lower().replace("-", "_")
            matched = _best_stem_match(stem, anchor_stems)
            if matched:
                candidates[matched].append(fp)
                continue
        candidates[feature_name].append(fp)

    # Phase 5: collapse UI-only candidates into "shared-ui"
    # A candidate that was found ONLY through deep-anchor dirs (components/, stores/)
    # and never through regular anchors (pages/, routers/, features/) is likely a
    # UI primitive (dropdowns, empty-state, breadcrumbs) — not a business feature.
    result = _collapse_ui_primitives(dict(candidates), candidate_anchor_types)

    # Phase 6: merge small candidates into parent candidates
    result = _merge_candidate_siblings(result)

    return result


def _collapse_ui_primitives(
    candidates: dict[str, list[str]],
    anchor_types: dict[str, set[str]],
) -> dict[str, list[str]]:
    """
    Moves candidates that exist only in deep-anchor dirs to "shared-ui".

    A real business feature like "issues" appears in routers/, pages/, AND
    components/. A UI primitive like "dropdowns" appears ONLY in components/.

    The structural signal: if a candidate was discovered exclusively through
    deep-anchor dirs (components/, stores/, containers/) and never through
    regular anchors (routers/, pages/, features/, handlers/) or support dirs,
    it's likely generic UI — not a business feature.
    """
    result: dict[str, list[str]] = {}
    shared_ui: list[str] = []

    for name, paths in candidates.items():
        sources = anchor_types.get(name, set())

        # A real business feature spans multiple architectural layers:
        # components/ + stores/, or pages/ + services/, etc.
        # A UI primitive lives in only ONE source dir (just components/).
        is_ui_primitive = (
            len(sources) == 1
            and sources.issubset({"components", "containers"})
        )

        if is_ui_primitive:
            shared_ui.extend(paths)
        else:
            result[name] = paths

    if shared_ui:
        result.setdefault("shared-ui", []).extend(shared_ui)

    return result


def _deep_match_file(
    parts: tuple[str, ...],
    anchor_stems: dict[str, str],
) -> str | None:
    """
    Deeper matching for files not caught by anchor/support matching.

    Strategies:
    1. Hook pattern: useIssues.ts → "issues", useCycles.ts → "cycles"
    2. Store subdirs: stores/issues/store.ts → "issues"
    3. Any meaningful parent dir that matches a candidate
    4. Filename stem match against candidates
    """
    filename = parts[-1]
    stem = Path(filename).stem

    # Strategy 1: hook pattern (useXxx, use-xxx)
    hook_match = re.match(r"^use[-_]?([A-Z][a-zA-Z]+|[a-z][-a-z]+)", stem)
    if hook_match:
        hook_name = _normalize_stem(hook_match.group(1))
        matched = _best_stem_match(hook_name, anchor_stems)
        if matched:
            return matched

    # Strategy 2: parent directory matches a candidate
    # Walk up from deepest meaningful dir to shallowest.
    # Skip generic dirs UNLESS the dir name matches an existing candidate
    # (e.g. "views" is a generic dir name but also a real feature in Plane).
    _SKIP_PARENT = {
        "src", "app", "core", "lib", "web", "api", "ce", "ee",
        "packages", "apps", "internal", "public", "types", "constants",
    }
    for part in reversed(parts[:-1]):
        lower = part.lower()
        normalized = _normalize_stem(lower)

        # Always check if this dir name directly matches a candidate
        if normalized in anchor_stems:
            return anchor_stems[normalized]

        # Skip generic/structural dirs
        if lower in _SKIP_PARENT or lower in _SUPPORT_DIRS or lower in _DEEP_ANCHOR_DIRS:
            continue
        if lower in _INFRA_DIRS or lower in _TECH_LAYER_DIRS:
            continue

        matched = _best_stem_match(normalized, anchor_stems)
        if matched:
            return matched

    # Strategy 3: filename stem
    file_stem = _normalize_stem(stem)
    if file_stem and len(file_stem) >= 3:
        return _best_stem_match(file_stem, anchor_stems)

    return None


def _merge_plural_duplicates(
    candidates: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Merges singular/plural duplicate candidates.

    "issue" + "issues" → "issues" (larger wins)
    "notification" + "notifications" → "notifications"
    "sticky" + "stickies" → "stickies"
    "analytic" + "analytics" → "analytics"
    """
    result: dict[str, list[str]] = {}
    merged_into: dict[str, str] = {}  # name → target

    # Sort by size descending — larger candidate becomes the target
    sorted_items = sorted(candidates.items(), key=lambda x: -len(x[1]))

    for name, paths in sorted_items:
        if name in merged_into:
            continue

        # Check if a plural/singular variant already exists in result
        variants = _plural_variants(name)
        target = None
        for v in variants:
            if v in result and v != name:
                target = v
                break

        if target:
            result[target].extend(paths)
            merged_into[name] = target
        else:
            result[name] = list(paths)

    return result


def _plural_variants(name: str) -> list[str]:
    """Returns singular/plural variants of a name."""
    variants = []
    # issues → issue, issue → issues
    if name.endswith("ies"):
        variants.append(name[:-3] + "y")  # stickies → sticky
    if name.endswith("s") and not name.endswith("ss"):
        variants.append(name[:-1])  # issues → issue, analytics → analytic
    if not name.endswith("s"):
        variants.append(name + "s")  # issue → issues, analytic → analytics
    if name.endswith("y") and not name.endswith("ey"):
        variants.append(name[:-1] + "ies")  # sticky → stickies
    return variants


def _merge_candidate_siblings(
    candidates: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Merges related candidates:
    1. Singular/plural duplicates: "issue" + "issues" → "issues" (larger wins)
    2. Small candidates into parent by prefix: "workspace-notifications" → "workspace"
    """
    # Phase 1: merge singular/plural duplicates
    result = _merge_plural_duplicates(candidates)

    # Phase 2: merge small candidates into parent by prefix
    _MIN_FILES_TO_KEEP = 5
    merge_targets: dict[str, str] = {}
    sorted_candidates = sorted(result.items(), key=lambda x: -len(x[1]))

    for name, paths in sorted_candidates:
        if len(paths) >= _MIN_FILES_TO_KEEP:
            continue

        name_parts = re.split(r"[-_]", name)
        for prefix_len in range(len(name_parts) - 1, 0, -1):
            prefix = "-".join(name_parts[:prefix_len])
            prefix_us = "_".join(name_parts[:prefix_len])
            parent = None
            if prefix in result and prefix != name:
                parent = prefix
            elif prefix_us in result and prefix_us != name:
                parent = prefix_us

            if parent and len(result[parent]) > len(paths):
                merge_targets[name] = parent
                break

    # Execute merges
    for child, parent in merge_targets.items():
        if child in result and parent in result:
            result[parent].extend(result.pop(child))

    return result


def _is_infra_file(fp: str) -> bool:
    """Returns True if this file should be skipped entirely (config/infra)."""
    parts = Path(fp).parts
    filename = parts[-1].lower()

    # Skip files in infra directories
    for part in parts[:-1]:
        lower = part.lower()
        if lower in _INFRA_DIRS:
            return True
        # Skip tech layer dirs (prisma/, .yarn/, .changeset/, etc.)
        if lower in _TECH_LAYER_DIRS:
            return True
        # Skip hidden dirs (but not .github which is already in _INFRA_DIRS)
        if lower.startswith(".") and lower not in {".github"}:
            return True

    # Skip known infra files
    if filename in _INFRA_FILENAMES:
        return True

    # Skip config files at root or shallow depth (< 3 levels)
    if len(parts) <= 2 and Path(filename).suffix in _INFRA_EXTENSIONS:
        return True

    return False


def _find_anchor(parts: tuple[str, ...]) -> tuple[str | None, str | None]:
    """
    Checks if a file is in an anchor directory and extracts the feature stem.

    Supports two types of anchors:
    - Regular anchors (routers/, pages/): filename stem = feature
    - Deep anchors (components/, stores/): subdirectory name = feature

    Examples:
        backend/routers/chat.py → ("routers", "chat")
        frontend/src/pages/ChatPage.tsx → ("pages", "chat")
        frontend/src/features/integrations/IntegrationsPage.tsx → ("features", "integrations")
        components/graphql/QueryEditor.tsx → ("components", "graphql")
        components/collections/CollectionList.tsx → ("components", "collections")
        components/ui/Button.tsx → (None, None) — "ui" is generic, skip
        backend/agent/skills/base_persona.py → (None, None) — agent is support, not anchor
    """
    # Generic subdirs inside deep-anchor dirs that should NOT become candidates
    _GENERIC_SUBDIRS = {
        "ui", "common", "shared", "layout", "layouts", "base",
        "icons", "primitives", "core", "general", "app",
    }

    # First pass: check for deep anchors (higher priority for monorepos
    # where components/X/ is the primary feature signal)
    for i, part in enumerate(parts[:-1]):
        lower = part.lower()
        if lower in _DEEP_ANCHOR_DIRS and i + 1 < len(parts) - 1:
            subdir = parts[i + 1].lower()
            if subdir not in _GENERIC_SUBDIRS:
                return (lower, _normalize_stem(subdir))

    # Pass 1.5: detect *-router pattern (tRPC, etc.)
    # trpc/server/auth-router/handler.ts → ("router", "auth")
    for i, part in enumerate(parts[:-1]):
        lower = part.lower()
        if lower.endswith("-router") and len(lower) > 7:
            stem = lower[:-7]  # "auth-router" → "auth"
            return ("router", _normalize_stem(stem))

    # Second pass: regular anchor dirs
    # Common wrapper names that should NOT be treated as features
    _WRAPPER_NAMES = {
        "web", "api", "admin", "server", "client", "frontend", "backend",
        "mobile", "desktop", "docs", "cli", "common", "shared", "main",
        "app", "core", "src",
    }

    for i, part in enumerate(parts[:-1]):
        lower = part.lower()
        if lower in _ANCHOR_DIRS:
            if i + 1 < len(parts) - 1:
                next_part = parts[i + 1].lower()
                if next_part.startswith("(") and next_part.endswith(")"):
                    next_part = next_part[1:-1]
                # Skip if next part is a common wrapper (apps/web/ is not a feature)
                if next_part in _WRAPPER_NAMES:
                    continue
                return (lower, _normalize_stem(next_part))
            else:
                stem = _normalize_stem(Path(parts[-1]).stem)
                if stem and stem not in {"__init__", "index", "base", "main"}:
                    return (lower, stem)

    return (None, None)


def _normalize_stem(raw: str) -> str:
    """
    Normalizes a stem for matching.
    ChatPage → chat, chat_executor → chat, v1_messages → messages
    """
    # Remove common suffixes
    s = raw.lower()
    for suffix in ("page", "router", "controller", "handler", "view",
                    "service", "executor", "worker", "model", "schema",
                    "serializer", "blueprint", "container", "dialog"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[:len(s) - len(suffix)]
            break

    # Remove common prefixes
    for prefix in ("v1_", "v2_", "api_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break

    # Convert camelCase/PascalCase to snake_case for matching
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()

    # Remove trailing underscores and hyphens
    s = s.strip("_- ")

    return s


def _stem_variants(stem: str) -> list[str]:
    """
    Returns stem variants for fuzzy matching.
    "chat" → ["chat", "chat_"]
    "shared_conversation" → ["shared_conversation", "shared_conversation_", "shared", "conversation"]
    """
    variants = [stem]
    # Add individual parts for compound names
    parts = re.split(r"[_-]", stem)
    if len(parts) > 1:
        variants.extend(parts)
    return variants


def _match_support_file(
    parts: tuple[str, ...],
    anchor_stems: dict[str, str],
) -> str | None:
    """
    Tries to match a support file to an existing candidate.

    Matching strategies (in order):
    1. Subdirectory after support dir matches a candidate
       e.g. components/chat/widget-block.tsx → "chat"
       e.g. services/edr/crowdstrike.py → "edr" or "integrations"
    2. Filename stem matches a candidate
       e.g. services/chat_executor.py → "chat"
       e.g. schemas/chat.py → "chat"
    """
    # Find the support directory and what comes after it
    support_idx = None
    for i, part in enumerate(parts[:-1]):
        if part.lower() in _SUPPORT_DIRS:
            support_idx = i
            break

    if support_idx is None:
        return None

    # Strategy 1: subdirectory after support dir
    if support_idx + 1 < len(parts) - 1:
        subdir = _normalize_stem(parts[support_idx + 1])
        match = _best_stem_match(subdir, anchor_stems)
        if match:
            return match

    # Strategy 2: filename stem
    file_stem = _normalize_stem(Path(parts[-1]).stem)
    return _best_stem_match(file_stem, anchor_stems)


def _best_stem_match(
    file_stem: str,
    anchor_stems: dict[str, str],
) -> str | None:
    """Finds the best matching candidate for a file stem."""
    if not file_stem:
        return None

    # Exact match
    if file_stem in anchor_stems:
        return anchor_stems[file_stem]

    # Check if file_stem starts with or contains an anchor stem
    best_match = None
    best_len = 0
    for stem, candidate in anchor_stems.items():
        if len(stem) < 3:
            continue
        if file_stem.startswith(stem) or stem.startswith(file_stem):
            if len(stem) > best_len:
                best_match = candidate
                best_len = len(stem)
        # Also check if any part of the file stem matches
        file_parts = re.split(r"[_-]", file_stem)
        for fp in file_parts:
            if fp == stem and len(fp) >= 3:
                if len(stem) > best_len:
                    best_match = candidate
                    best_len = len(stem)

    return best_match


def detect_features_from_structure(files: list[str]) -> dict[str, list[str]]:
    """
    Detects features based on directory structure.
    This is a heuristic fallback — LLM analysis provides richer results.

    Examples:
        src/auth/login.py      → feature: "auth"
        src/payments/stripe.py → feature: "payments"
        src/api/users.py       → feature: "api"
    """
    features: dict[str, list[str]] = defaultdict(list)

    for file_path in files:
        parts = Path(file_path).parts
        feature_name = _extract_feature_name(parts)
        features[feature_name].append(file_path)

    # Split oversized features — e.g. Django apps where all code lives in one dir
    return split_large_features(dict(features))


_SKIP_STEMS = {"__init__", "apps", "urls", "admin", "conftest", "setup", "config"}
_TEST_PREFIXES = ("test_", "tests_")


def split_large_features(
    features: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Splits features with too many files into sub-features.

    Strategy 1: find common directory prefix, split by next-level subdirectory.
    Strategy 2: if most files are flat (no subdir), split by filename stem
    to extract sub-domains (e.g. barcodes.py, classifier.py → separate features).
    """
    result: dict[str, list[str]] = {}

    for name, paths in features.items():
        if len(paths) <= _MAX_FEATURE_FILES:
            result[name] = paths
            continue

        # Find common directory prefix among all paths
        all_parts = [Path(p).parts for p in paths]
        prefix_len = 0
        if all_parts:
            for i in range(min(len(p) for p in all_parts)):
                if len({p[i] for p in all_parts}) == 1:
                    prefix_len = i + 1
                else:
                    break

        # Strategy 1: split by next directory level after common prefix
        # Skip technical dirs (tests, migrations) — they'll be handled by Strategy 2
        _SKIP_SUBDIRS = {"tests", "test", "__tests__", "migrations", "templates", "static"}
        sub_groups: dict[str, list[str]] = defaultdict(list)
        flat_count = 0
        for p in paths:
            parts = Path(p).parts
            if prefix_len < len(parts) - 1:
                sub_name = parts[prefix_len].lower()
                if sub_name in _SKIP_SUBDIRS:
                    flat_count += 1
                    sub_groups[name].append(p)
                else:
                    sub_groups[f"{name}-{sub_name}"].append(p)
            else:
                flat_count += 1
                sub_groups[name].append(p)

        max_sub = max(len(v) for v in sub_groups.values()) if sub_groups else 0
        if len(sub_groups) > 2 and max_sub < len(paths) * 0.7:
            result.update(sub_groups)
            continue

        # Strategy 2: split source files by stem, group tests/infra into parent
        # Handles Django apps where features are individual .py modules
        _INFRA_DIRS = {"tests", "test", "__tests__", "migrations", "templates", "static",
                       "management", "templatetags", "locale"}
        source_stems: dict[str, list[str]] = defaultdict(list)
        infra_files: list[str] = []

        for p in paths:
            parent_dir = Path(p).parent.name.lower()
            stem = Path(p).stem.lower()

            if parent_dir in _INFRA_DIRS or stem in _SKIP_STEMS:
                infra_files.append(p)
            elif any(stem.startswith(tp) for tp in _TEST_PREFIXES):
                infra_files.append(p)
            else:
                source_stems[f"{name}-{stem}"].append(p)

        # Merge single-file groups into parent
        merged: dict[str, list[str]] = {name: list(infra_files)}
        for k, v in source_stems.items():
            if len(v) <= 1:
                merged[name].extend(v)
            else:
                merged[k] = v

        max_merged = max(len(v) for v in merged.values()) if merged else 0
        if len(merged) > 3 and max_merged < len(paths) * 0.7:
            result.update(merged)
        else:
            result[name] = paths

    return result


def _extract_feature_name(parts: tuple[str, ...]) -> str:
    """
    Extracts a feature name from a file path.

    Logic:
    - src/auth/login.py       → "auth"
    - app/api/payments/...    → "payments"
    - lib/utils/helpers.py    → "utils"
    - app/(dashboard)/settings/page.tsx → "settings"
    - index.py                → "root"
    """
    # Skip common top-level wrapper directories (not feature names themselves)
    skip_prefixes = {
        # Generic source roots
        "src", "app", "lib", "pkg", "internal", "core",
        # Frontend structural directories — not business features
        "views", "pages", "screens", "routes", "containers",
        "components", "layouts", "features",
        # Next.js/Remix structural dirs
        "api", "actions", "hooks", "providers", "contexts", "stores",
        "services", "helpers", "utils", "types", "models", "schemas",
        "middleware", "config", "constants", "assets", "styles",
        "public", "static",
    }

    # Also skip Next.js route groups like (dashboard), (auth), etc.
    def _is_route_group(part: str) -> bool:
        return part.startswith("(") and part.endswith(")")

    for part in parts[:-1]:  # Exclude the filename
        normalized = part.lower()
        if normalized not in skip_prefixes and not _is_route_group(part):
            return normalized

    # File is at the repo root or only in structural dirs
    return "root"


_TEST_FILE_RE = re.compile(
    r"\.(test|spec)\.(ts|tsx|js|jsx|py)$"
    r"|/__tests__/"
    r"|/tests?/"
    r"|/test_[^/]+\.py$"
    r"|/[^/]+_test\.py$",
    re.IGNORECASE,
)


def _is_test_file(path: str) -> bool:
    """Returns True if the file path looks like a test file."""
    normalized = "/" + path.replace("\\", "/")
    return bool(_TEST_FILE_RE.search(normalized))


def _is_test_commit(commit: Commit, flow_paths: set[str]) -> bool:
    """
    Returns True if the commit adds/modifies test files for this flow's directory tree.
    Checks whether any changed file is a test file alongside the flow's source files.
    """
    # Derive the directories that contain flow source files
    flow_dirs = {str(Path(p).parent) for p in flow_paths}

    for f in commit.files_changed:
        if not _is_test_file(f):
            continue
        # Counts as a test commit if the test file is in or near a flow directory
        test_dir = str(Path(f).parent)
        if test_dir in flow_dirs or any(test_dir.startswith(d) for d in flow_dirs):
            return True
    return False


def _build_weekly_timeline(
    commits: list[Commit],
    flow_paths: set[str],
) -> list[TimelinePoint]:
    """
    Groups commits by ISO week and counts total, bug-fix, and test commits.
    Returns points sorted chronologically (oldest first).
    """
    weekly: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0, "bug_fix": 0, "test": 0,
    })

    for c in commits:
        iso = c.date.isocalendar()
        label = f"{iso.year}-W{iso.week:02d}"
        weekly[label]["total"] += 1
        if c.is_bug_fix:
            weekly[label]["bug_fix"] += 1
        if _is_test_commit(c, flow_paths):
            weekly[label]["test"] += 1

    return [
        TimelinePoint(
            date=label,
            total_commits=v["total"],
            bug_fix_commits=v["bug_fix"],
            test_commits=v["test"],
        )
        for label, v in sorted(weekly.items())
    ]


def _collect_prs(commits: list[Commit], remote_url: str) -> list[PullRequest]:
    """Returns deduplicated PullRequest objects from bug-fix commits that have a PR number."""
    seen: set[int] = set()
    prs: list[PullRequest] = []
    for c in commits:
        if not c.is_bug_fix or c.pr_number is None or c.pr_number in seen:
            continue
        seen.add(c.pr_number)
        url = f"{remote_url}/pull/{c.pr_number}" if remote_url else ""
        prs.append(PullRequest(
            number=c.pr_number,
            url=url,
            title=c.message.split("\n")[0][:120],
            author=c.author,
            date=c.date,
        ))
    return sorted(prs, key=lambda p: p.date, reverse=True)


_MIN_FEATURE_FILES = 3     # features with fewer files get merged into a parent
_MIN_FEATURE_COMMITS = 2   # features with fewer commits get merged into a parent


def _merge_small_features(
    feature_paths: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Merges small/granular features into larger related ones.

    Strategy: features with < _MIN_FEATURE_FILES files get absorbed into the
    closest parent-directory feature. If no parent exists, they merge into
    the largest feature sharing the same top-level directory.
    """
    if len(feature_paths) <= 15:
        return feature_paths

    # Sort: large features first (they are merge targets)
    sorted_features = sorted(feature_paths.items(), key=lambda x: len(x[1]), reverse=True)

    # Build a dir→feature index for parent-matching
    dir_to_feature: dict[str, str] = {}
    for name, paths in sorted_features:
        for p in paths:
            parts = Path(p).parts
            for depth in range(1, len(parts)):
                d = str(Path(*parts[:depth]))
                dir_to_feature.setdefault(d, name)

    merged: dict[str, list[str]] = {}
    merge_target: dict[str, str] = {}  # small_feature → target_feature

    for name, paths in sorted_features:
        if len(paths) >= _MIN_FEATURE_FILES:
            merged[name] = list(paths)
        else:
            # Find closest parent feature
            target = _find_merge_target(name, paths, merged, dir_to_feature)
            if target and target != name:
                merge_target[name] = target
                merged.setdefault(target, []).extend(paths)
            else:
                merged[name] = list(paths)

    return merged


def _find_merge_target(
    name: str,
    paths: list[str],
    existing: dict[str, list[str]],
    dir_to_feature: dict[str, str],
) -> str | None:
    """Finds the best feature to absorb a small feature into."""
    # Strategy 1: parent directory match
    for p in paths:
        parent = str(Path(p).parent)
        while parent and parent != ".":
            candidate = dir_to_feature.get(parent)
            if candidate and candidate != name and candidate in existing:
                return candidate
            parent = str(Path(parent).parent)

    # Strategy 2: same top-level directory → merge into largest
    top_dirs = {Path(p).parts[0] if len(Path(p).parts) > 1 else "root" for p in paths}
    best, best_size = None, 0
    for feat_name, feat_paths in existing.items():
        if feat_name == name:
            continue
        feat_tops = {Path(p).parts[0] if len(Path(p).parts) > 1 else "root" for p in feat_paths}
        if top_dirs & feat_tops and len(feat_paths) > best_size:
            best = feat_name
            best_size = len(feat_paths)

    return best


def build_feature_map(
    repo_path: str,
    commits: list[Commit],
    feature_paths: dict[str, list[str]],
    days: int,
    remote_url: str = "",
    shared_attributions: dict[str, list[SymbolAttribution]] | None = None,
    skip_small_feature_merge: bool = False,
) -> FeatureMap:
    """Builds a FeatureMap by joining commits with detected features.

    ``skip_small_feature_merge`` (Day 14 library-mode fix): when True,
    do not collapse features that have fewer than ``_MIN_FEATURE_FILES``
    files into their parent directories. Libraries routinely keep one
    ``.go`` or ``.py`` file per public module, and collapsing those
    into a single shared-infra bucket destroys the whole point of
    per-module detection. The caller (``cli.py``) sets this True when
    ``repo_structure.is_library`` is True.
    """

    # Merge small/granular features before computing metrics.
    if not skip_small_feature_merge:
        feature_paths = _merge_small_features(feature_paths)

    # Deduplicate paths within each feature
    feature_paths = {name: list(dict.fromkeys(paths)) for name, paths in feature_paths.items()}

    # Build file→feature index for O(1) lookup + dir-based fallback for deleted/renamed files
    file_to_feature: dict[str, str] = {}
    dir_to_feature: dict[str, str] = {}
    for feature_name, paths in feature_paths.items():
        for p in paths:
            file_to_feature[p] = feature_name
            parent = str(Path(p).parent)
            if parent != ".":
                dir_to_feature.setdefault(parent, feature_name)

    # Pre-compute symbol weights for shared files:
    # symbol_weights[feature_name][file_path] = weight (0.0-1.0)
    symbol_weights: dict[str, dict[str, float]] = {}
    if shared_attributions:
        for feat_name, attributions in shared_attributions.items():
            for attr in attributions:
                weight = attr.attributed_lines / attr.total_file_lines if attr.total_file_lines > 0 else 1.0
                symbol_weights.setdefault(feat_name, {})[attr.file_path] = min(weight, 1.0)

    feature_commits: dict[str, list[Commit]] = defaultdict(list)
    feature_commit_weights: dict[str, dict[str, float]] = defaultdict(dict)
    feature_authors: dict[str, set[str]] = defaultdict(set)
    feature_last_modified: dict[str, datetime] = {}

    for commit in commits:
        touched_features = set()

        for file_path in commit.files_changed:
            feat = file_to_feature.get(file_path)
            if not feat:
                parent = str(Path(file_path).parent)
                feat = dir_to_feature.get(parent)
            if feat:
                touched_features.add(feat)

        for feature_name in touched_features:
            feature_commits[feature_name].append(commit)
            feature_authors[feature_name].add(commit.author)

            # Compute commit weight: max weight across all files touched in this commit
            # that belong to this feature (1.0 for non-shared files)
            max_weight = 0.0
            feat_weights = symbol_weights.get(feature_name, {})
            for fp in commit.files_changed:
                if file_to_feature.get(fp) == feature_name or dir_to_feature.get(str(Path(fp).parent)) == feature_name:
                    max_weight = max(max_weight, feat_weights.get(fp, 1.0))
            feature_commit_weights[feature_name][commit.sha] = max_weight or 1.0

            if feature_name not in feature_last_modified or \
               commit.date > feature_last_modified[feature_name]:
                feature_last_modified[feature_name] = commit.date

    features = []
    for feature_name, paths in feature_paths.items():
        commits_for_feature = feature_commits.get(feature_name, [])
        total = len(commits_for_feature)

        bug_fixes = sum(1 for c in commits_for_feature if c.is_bug_fix)
        bug_fix_ratio = bug_fixes / total if total > 0 else 0.0

        # Symbol-weighted health score
        sym_health = None
        commit_weights = feature_commit_weights.get(feature_name, {})
        if shared_attributions and feature_name in shared_attributions and total > 0:
            sym_health = _calculate_weighted_health(commits_for_feature, commit_weights)

        feat_attributions = shared_attributions.get(feature_name, []) if shared_attributions else []

        features.append(Feature(
            name=feature_name,
            paths=paths,
            authors=sorted(feature_authors.get(feature_name, set())),
            total_commits=total,
            bug_fixes=bug_fixes,
            bug_fix_ratio=round(bug_fix_ratio, 3),
            last_modified=feature_last_modified.get(
                feature_name,
                datetime.now(tz=timezone.utc)
            ),
            health_score=_calculate_health(bug_fix_ratio, total, commits_for_feature) if total > 0 else 100.0,
            bug_fix_prs=_collect_prs(commits_for_feature, remote_url),
            shared_attributions=feat_attributions,
            symbol_health_score=round(sym_health, 1) if sym_health is not None else None,
        ))

    return FeatureMap(
        repo_path=repo_path,
        remote_url=remote_url,
        analyzed_at=datetime.now(tz=timezone.utc),
        total_commits=len(commits),
        date_range_days=days,
        features=features,
    )


def build_flows_metrics(
    commits: list[Commit],
    flow_file_mappings: dict[str, list[str]],
    remote_url: str = "",
    coverage_data: dict[str, float] | None = None,
) -> list[Flow]:
    """
    Builds Flow objects with commit metrics, mirroring build_feature_map logic.

    Args:
        commits: All commits for the parent feature.
        flow_file_mappings: Dict of flow_name → list of file paths.

    Returns:
        List of Flow objects with health scores and bug fix metrics.
    """
    # Build file→flow index for O(1) lookup + dir-based fallback
    file_to_flow: dict[str, str] = {}
    dir_to_flow: dict[str, str] = {}
    for flow_name, paths in flow_file_mappings.items():
        for p in paths:
            file_to_flow[p] = flow_name
            parent = str(Path(p).parent)
            if parent != ".":
                dir_to_flow.setdefault(parent, flow_name)

    flow_commits: dict[str, list[Commit]] = defaultdict(list)
    flow_authors: dict[str, set[str]] = defaultdict(set)
    flow_last_modified: dict[str, datetime] = {}

    for commit in commits:
        touched_flows: set[str] = set()
        for file_path in commit.files_changed:
            flow = file_to_flow.get(file_path)
            if not flow:
                parent = str(Path(file_path).parent)
                flow = dir_to_flow.get(parent)
            if flow:
                touched_flows.add(flow)
        for flow_name in touched_flows:
            flow_commits[flow_name].append(commit)
            flow_authors[flow_name].add(commit.author)
            if flow_name not in flow_last_modified or \
               commit.date > flow_last_modified[flow_name]:
                flow_last_modified[flow_name] = commit.date

    # Pre-compute flow directories for test-commit detection
    flow_dirs: dict[str, set[str]] = {
        flow_name: {str(Path(p).parent) for p in paths}
        for flow_name, paths in flow_file_mappings.items()
    }

    # Collect test-only commits per flow (commits that touch adjacent test files
    # but may not touch any flow source file — e.g. standalone test additions)
    flow_test_only_commits: dict[str, list[Commit]] = defaultdict(list)
    seen_shas: dict[str, set[str]] = defaultdict(set)  # avoid duplicates with flow_commits

    for commit in commits:
        for flow_name, dirs in flow_dirs.items():
            if commit.sha in seen_shas[flow_name]:
                continue
            for f in commit.files_changed:
                if _is_test_file(f) and str(Path(f).parent) in dirs:
                    flow_test_only_commits[flow_name].append(commit)
                    seen_shas[flow_name].add(commit.sha)
                    break

    flows = []
    for flow_name, paths in flow_file_mappings.items():
        commits_for_flow = flow_commits.get(flow_name, [])
        total = len(commits_for_flow)
        bug_fixes = sum(1 for c in commits_for_flow if c.is_bug_fix)
        bug_fix_ratio = bug_fixes / total if total > 0 else 0.0

        paths_set = set(paths)
        # Count test files: those explicitly in flow paths + adjacent test files
        # touched by any commit (from test-only commits collected above)
        adjacent_test_files: set[str] = set()
        for c in flow_test_only_commits.get(flow_name, []):
            for f in c.files_changed:
                if _is_test_file(f):
                    adjacent_test_files.add(f)
        test_file_count = (
            sum(1 for p in paths if _is_test_file(p)) + len(adjacent_test_files)
        )

        # Merge source-file commits + test-only commits for the timeline
        seen_for_timeline = {c.sha for c in commits_for_flow}
        timeline_commits = list(commits_for_flow) + [
            c for c in flow_test_only_commits.get(flow_name, [])
            if c.sha not in seen_for_timeline
        ]
        weekly_points = _build_weekly_timeline(timeline_commits, paths_set)

        # Bus factor: authors with ≥20% of flow commits
        threshold = max(1, total * 0.2)
        author_counts: dict[str, int] = {}
        for c in commits_for_flow:
            author_counts[c.author] = author_counts.get(c.author, 0) + 1
        bus_factor = max(1, sum(1 for cnt in author_counts.values() if cnt >= threshold))

        # Health trend: first-half vs second-half bug_fix_ratio delta (positive = improving)
        health_trend: float | None = None
        if len(weekly_points) >= 4:
            mid = len(weekly_points) // 2

            def _bug_ratio(pts: list[TimelinePoint]) -> float:
                total_c = sum(p.total_commits for p in pts)
                return sum(p.bug_fix_commits for p in pts) / total_c if total_c > 0 else 0.0

            health_trend = round(_bug_ratio(weekly_points[:mid]) - _bug_ratio(weekly_points[mid:]), 3)

        # Hotspot files: source files with >40% bug_fix_ratio and ≥3 commits
        file_total: dict[str, int] = {}
        file_bugs: dict[str, int] = {}
        for c in commits_for_flow:
            for f in c.files_changed:
                if f in paths_set and not _is_test_file(f):
                    file_total[f] = file_total.get(f, 0) + 1
                    if c.is_bug_fix:
                        file_bugs[f] = file_bugs.get(f, 0) + 1

        hotspot_files = sorted(
            [f for f, t in file_total.items() if t >= 3 and file_bugs.get(f, 0) / t > 0.4],
            key=lambda f: -(file_bugs.get(f, 0) / file_total[f]),
        )[:5]

        # Coverage: avg line coverage % across non-test source files
        coverage_pct: float | None = None
        if coverage_data:
            coverages = []
            for p in paths:
                if _is_test_file(p):
                    continue
                for cov_path, pct in coverage_data.items():
                    if cov_path.endswith(p) or p.endswith(cov_path.lstrip("/")):
                        coverages.append(pct)
                        break
            if coverages:
                coverage_pct = round(sum(coverages) / len(coverages), 1)

        flows.append(Flow(
            name=flow_name,
            paths=paths,
            authors=sorted(flow_authors.get(flow_name, set())),
            total_commits=total,
            bug_fixes=bug_fixes,
            bug_fix_ratio=round(bug_fix_ratio, 3),
            last_modified=flow_last_modified.get(
                flow_name,
                datetime.now(tz=timezone.utc),
            ),
            health_score=_calculate_health(bug_fix_ratio, total, commits_for_flow),
            bug_fix_prs=_collect_prs(commits_for_flow, remote_url),
            test_file_count=test_file_count,
            weekly_points=weekly_points,
            bus_factor=bus_factor,
            health_trend=health_trend,
            hotspot_files=hotspot_files,
            coverage_pct=coverage_pct,
        ))

    return flows


def _calculate_health(
    bug_fix_ratio: float,
    total_commits: int,
    commits: list[Commit] | None = None,
) -> float:
    """
    Calculates a health score from 0 to 100.
    100 = healthy, 0 = high technical debt.

    Formula (revised Day 15):
    - Base score decreases with bug fix ratio using a sigmoid-like
      curve centered at 35% bug-fix ratio. This means:
        - 0-15% bug-fix → 85-100 (healthy — mostly new feature work)
        - 15-35% → 55-85 (normal active development)
        - 35-55% → 25-55 (elevated — worth watching)
        - 55-75% → 10-25 (high debt — maintenance-dominant)
        - 75%+   → 0-10 (critical — almost all commits are bug fixes)
      The old formula (ratio * 200) mapped anything ≥50% to flat zero,
      which made 90%+ of features in real OSS repos (outline, ghost,
      cal.com) "critical" — that's misleading for stable products where
      40-60% bug-fix ratio is normal active maintenance.
    - Activity factor scales confidence: a feature with 2 commits and
      50% bug ratio is less concerning than one with 200 commits and
      50%.
    - Age decay: recent bug fixes (last 30 days) weigh 2x more than
      older ones, so features getting healthier over time show it.
    """
    if total_commits == 0:
        return 100.0

    # Apply age-weighted bug fix ratio when commits are available
    effective_ratio = bug_fix_ratio
    if commits:
        now = datetime.now(tz=timezone.utc)
        weighted_bugs = 0.0
        weighted_total = 0.0
        for c in commits:
            age_days = (now - c.date).days
            # Recent commits (< 30 days) get weight 2.0, older decay to 0.5
            weight = 2.0 if age_days < 30 else max(0.5, 1.0 - (age_days - 30) / 365)
            weighted_total += weight
            if c.is_bug_fix:
                weighted_bugs += weight
        if weighted_total > 0:
            effective_ratio = weighted_bugs / weighted_total

    # Sigmoid-like curve centered at 55% bug-fix ratio, which is the
    # empirical median for mature OSS products (outline 56%, cal.com 67%,
    # excalidraw 49%). The old center (35%) made everything above 50% a
    # flat "critical=0", which was misleading for active codebases.
    #
    #   ratio  →  health
    #     0%   →  ~99  (healthy — pure feature work)
    #    20%   →  ~94  (healthy)
    #    35%   →  ~83  (healthy — more fixes than average)
    #    45%   →  ~69  (normal active development)
    #    55%   →   50  (midpoint — balanced maintenance)
    #    65%   →  ~31  (elevated — worth watching)
    #    75%   →  ~17  (high — maintenance-dominant)
    #    85%   →   ~8  (critical — almost all fixes)
    #    95%   →   ~4  (critical)
    import math
    # Logistic centered at 0.55 with steepness 8
    x = effective_ratio
    base_score = 100.0 / (1.0 + math.exp(8.0 * (x - 0.55)))

    activity_factor = min(1.0, total_commits / 50)

    return round(base_score * activity_factor + base_score * (1 - activity_factor) * 0.8, 1)


def _calculate_weighted_health(
    commits: list[Commit],
    commit_weights: dict[str, float],
) -> float:
    """Calculates health score with symbol-level line-range weights.

    Same formula as _calculate_health but each commit's contribution is
    scaled by its weight (fraction of shared file lines attributed to
    this feature). Non-shared file commits have weight 1.0.
    """
    if not commits:
        return 100.0

    now = datetime.now(tz=timezone.utc)
    weighted_bugs = 0.0
    weighted_total = 0.0

    for c in commits:
        age_days = (now - c.date).days
        age_weight = 2.0 if age_days < 30 else max(0.5, 1.0 - (age_days - 30) / 365)
        sym_weight = commit_weights.get(c.sha, 1.0)
        combined = age_weight * sym_weight

        weighted_total += combined
        if c.is_bug_fix:
            weighted_bugs += combined

    if weighted_total == 0:
        return 100.0

    effective_ratio = weighted_bugs / weighted_total
    base_score = max(0.0, 100.0 - (effective_ratio * 200))
    total = sum(commit_weights.get(c.sha, 1.0) for c in commits)
    activity_factor = min(1.0, total / 50)

    return round(base_score * activity_factor + base_score * (1 - activity_factor) * 0.8, 1)
