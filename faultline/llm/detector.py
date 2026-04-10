import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds, doubles each attempt

from faultline.analyzer.ast_extractor import FileSignature
from faultline.models.types import Commit, Feature

_MODEL = "claude-haiku-4-5-20251001"
_MAX_SAMPLE_PATHS = 5
_MAX_FEATURES_PER_CALL = 50
_MAX_FILES_FOR_DETECTION = 500

# Token budgets for LLM responses.
# The Anthropic SDK requires streaming for max_tokens > ~21,333 when using
# messages.parse() (non-streaming). Stay well below that limit.
# Dir-collapse responses list directory paths (~4–6K tokens in practice),
# so 16,384 is more than sufficient even for repos with 500+ unique dirs.
_MAX_TOKENS_FILE = 16_384
_MAX_TOKENS_DIR  = 16_384

_DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# When file count exceeds this, collapse to unique directories to save tokens
_DIR_COLLAPSE_THRESHOLD = 500
# Max route-anchor entries injected into the LLM prompt
_MAX_ROUTE_ANCHOR_FILES = 25
_MAX_ROUTES_PER_ENTRY   = 3
# Max sample filenames shown per directory in the enriched dir tree
_DIR_SAMPLE_FILES = 4
# Top co-change pairs to include in the prompt
_MAX_COCHANGE_IN_PROMPT = 20
# Max clusters per single merge LLM call (avoids prompt/response truncation)
_MERGE_CHUNK_SIZE = 40

_COMMIT_STOP_WORDS = {
    # conventional commit types
    "feat", "fix", "chore", "docs", "test", "refactor", "style", "perf", "ci",
    # generic coding verbs
    "add", "update", "remove", "change", "move", "delete", "create", "handle",
    "support", "use", "implement", "improve", "cleanup", "clean", "minor", "wip",
    # English stop words
    "the", "a", "an", "and", "or", "in", "to", "for", "of", "with", "is",
    "it", "this", "that", "not", "be", "from", "by", "on", "at", "as", "are",
}

_DETECTION_SYSTEM_PROMPT = """\
You are a senior software architect analyzing a codebase's file tree to identify semantic business features.

## Task

Given a list of file paths from a git repository, group them into business features. A feature is a user-facing capability or business domain area, not a technical layer.

## Rules

1. Group files by the business domain they serve, not by technical role. Files from different directories (components, stores, API routes, tests) that serve the same business purpose belong to the same feature.
2. Use business domain terminology for feature names. Prefer "user-auth" over "authentication-module", "order-checkout" over "stripe-integration", "content-search" over "elasticsearch-wrapper".
3. Feature names must be lowercase, hyphen-separated, 1-3 words. Examples: "user-auth", "payment-processing", "dashboard", "notifications", "team-management".
4. Each file must appear in exactly one feature. No duplicates, no omissions.
5. Group by business domain, not by directory. Each distinct business domain should be its own feature. Merge only when two groups serve the exact same domain.
6. Every feature must contain at least 2 files. Single files must be merged into the closest related feature.
7. Test files belong to the same feature as the code they test. Match by naming convention (test_auth.py belongs with auth.py, UserService.test.ts belongs with UserService.ts).
8. Skip infrastructure and tooling files entirely: package.json, pyproject.toml, setup.py, .gitignore, Makefile, *.lock, *.toml, Dockerfile, docker-compose.yml, CI configs.
9. Shared utility files go into the most closely related business feature, or into "shared-utilities" only if they truly cross all feature boundaries.
10. For monorepo structures, group by business feature across packages when the same domain spans multiple packages.
11. If a <route-anchors> section is provided, treat those files as strong feature anchors.
    Files that define API routes (GET/POST/PUT/DELETE) are entry points to a feature — group
    other files in the same directory tree with the file that shares their route prefix.

## Shared / utility files — CRITICAL

Files in shared directories (hooks/, utils/, helpers/, lib/, components/ui/) are NOT automatically \
"shared" features. Most of them serve a SPECIFIC business domain:
- `hooks/useEntitySearch.tsx` → belongs to entity-management feature (NOT "shared-hooks")
- `hooks/useIPEnrichment.ts` → belongs to enrichment feature (NOT "shared-hooks")
- `hooks/useOrganizationMembers.tsx` → belongs to organization feature (NOT "shared-hooks")
- `utils/exportCSV.ts` → belongs to export feature (NOT "shared-utils")

Only truly generic, domain-agnostic files belong to "shared-*" features:
- `hooks/useHover.ts` → shared (generic UI behavior)
- `hooks/useDebounce.ts` → shared (generic utility)
- `utils/formatDate.ts` → shared (generic formatter)
- `components/ui/Button.tsx` → shared-ui (generic component)

RULE: If a file name contains a business domain keyword (entity, auth, payment, export, etc.), \
it belongs to that business feature, regardless of which directory it's in.

## Size limits — CRITICAL

- No feature may contain more than 40 files. If a group grows larger, split it into \
  distinct sub-features by business capability.
- API routes that serve different domains (e.g. /api/organizations/*, /api/cost/*, \
  /api/health/*) MUST be separate features, not one "backend-api" bucket.
- NEVER create a catch-all feature like "backend-api", "api-routes", "core", or "shared-backend". \
  Every API route file belongs to the business feature it serves.

## Page ↔ API route matching — CRITICAL

In Next.js / app-router projects, match page directories to their API routes by shared path segment:
- `app/organizations/` (page) + `app/api/organizations/` (API) → same feature
- `app/dashboard/` (page) + `app/api/dashboard/` (API) → same feature
- `app/cost/` (page) + `app/api/cost/` (API) → same feature
- `app/health/` (page) + `app/api/health/` (API) → same feature

Also include related hooks, libs, components, and types in the same feature:
- `hooks/use-organizations.ts` → belongs to the organizations feature
- `lib/integrations/firestore.ts` → belongs to the feature that uses Firestore most
- `components/org-sync-status.tsx` → belongs to the organizations feature (org- prefix)

## Anti-patterns

BAD — splitting page from its API routes into separate features:
  "organizations": [app/organizations/page.tsx, app/organizations/list-page.tsx]
  "backend-api": [app/api/organizations/route.ts, app/api/organizations/[id]/route.ts, ...]
  ← the API routes serve organizations, they belong together

GOOD — page + API routes + hooks unified by business domain:
  "organizations": [app/organizations/page.tsx, app/organizations/list-page.tsx,
    app/api/organizations/route.ts, app/api/organizations/[id]/route.ts,
    hooks/use-organizations.ts, components/org-sync-status.tsx]

BAD — catch-all "backend" or "api" feature:
  "backend-api": [api/auth/route.ts, api/payments/route.ts, api/health/route.ts, ...]
  ← these are distinct business domains forced into one bucket

BAD — grouping by technical layer:
  "components": [LoginForm.tsx, CheckoutForm.tsx, Dashboard.tsx]
  "api": [auth.ts, payments.ts, analytics.ts]

GOOD — grouping by business domain:
  "user-auth": [LoginForm.tsx, auth.ts, middleware.ts]
  "checkout": [CheckoutForm.tsx, payments.ts]
  "analytics": [Dashboard.tsx, analytics.ts]

## Example (Next.js app router)

Files:
  app/organizations/page.tsx
  app/organizations/org-list-page.tsx
  app/organizations/[id]/page.tsx
  app/organizations/[id]/org-detail-page.tsx
  app/api/organizations/route.ts
  app/api/organizations/[id]/route.ts
  app/api/organizations/[id]/sync-status/route.ts
  app/dashboard/page.tsx
  app/dashboard/dashboard-page.tsx
  app/api/dashboard/stats/route.ts
  app/sign-in/page.tsx
  hooks/use-organizations.ts
  hooks/use-dashboard-data.ts
  lib/auth.ts
  middleware.ts
  components/ui/button.tsx
  components/ui/modal.tsx
  components/org-sync-status.tsx

Result:
  "organizations": [app/organizations/page.tsx, app/organizations/org-list-page.tsx,
    app/organizations/[id]/page.tsx, app/organizations/[id]/org-detail-page.tsx,
    app/api/organizations/route.ts, app/api/organizations/[id]/route.ts,
    app/api/organizations/[id]/sync-status/route.ts,
    hooks/use-organizations.ts, components/org-sync-status.tsx]
  "dashboard": [app/dashboard/page.tsx, app/dashboard/dashboard-page.tsx,
    app/api/dashboard/stats/route.ts, hooks/use-dashboard-data.ts]
  "user-auth": [app/sign-in/page.tsx, lib/auth.ts, middleware.ts]
  "shared-ui": [components/ui/button.tsx, components/ui/modal.tsx]\
"""

_DETECTION_USER_PROMPT = """\
Analyze these repository files and group them into semantic business features.
{feature_hint}
<file_list>
{file_tree}
</file_list>{extra_context}
Return the JSON mapping of features to files. Skip infrastructure/config files. Each file in exactly one feature. Use business domain names.\
"""

# ── Dir-collapse variants (used when file count > _DIR_COLLAPSE_THRESHOLD) ──
# The input is DIRECTORIES (with sample filenames for context), not individual
# files. The LLM must return directory paths in the `files` field, not filenames.

_DIR_DETECTION_SYSTEM_PROMPT = """\
You are a senior software architect grouping a large codebase's directories into semantic business features.

## Task

You will receive a list of DIRECTORIES. Each line shows a directory path, optionally followed \
by → and a few sample filenames to illustrate what that directory contains. \
Indented lines are subdirectories of the line above them — use this nesting to understand \
how the codebase is structured.

Group these directories into business features. A feature is a user-facing capability or \
business domain area, not a technical layer.

## Rules

1. Group directories by the business domain they serve, not by technical role.
2. Feature names: lowercase, hyphen-separated, 1-3 words. \
   Examples: "user-auth", "app-router", "build-pipeline", "image-optimization".
3. Every directory must appear in exactly one feature. No omissions.
4. The `files` field in your response must contain DIRECTORY PATHS exactly as shown in the \
   input — not the sample filenames after →, not expanded sub-paths, not invented paths.
5. Balance granularity: each distinct business capability gets its own feature. \
   Do NOT lump unrelated capabilities into a single feature just because they share a parent directory. \
   For example, "billing", "webhooks", "templates", "auth" are SEPARATE features, not one "core-platform".
6. Deeply nested subdirectories almost always belong to the same feature as their parent. \
   Only split siblings when they serve clearly different business domains (e.g. "payments" vs "auth").
11. SIZE LIMIT: No feature may contain more than 40 directories. If a group grows larger, \
    split it into sub-features by business capability.
12. In Next.js / app-router projects, each `app/<page>/` directory is usually its own feature. \
    The corresponding `app/api/<page>/` routes belong to the SAME feature as the page, not to \
    a separate "api" feature. Never create a catch-all "backend-api" or "api-routes" feature.
10. IMPORTANT: Look at the sample filenames after → to detect MULTIPLE business domains within \
   a single directory. In Django/Rails/Flask apps, one directory often contains many business \
   modules: e.g. if sample files show barcodes.py, classifier.py, bulk_edit.py, mail.py, \
   signals.py → this directory spans barcode-detection, classification, bulk-operations, etc. \
   When a directory has many sample files suggesting different business domains, assign that \
   directory to the MOST DOMINANT domain. The other domains will be in sibling directories or \
   identified elsewhere.
7. Technical directories (utils, helpers, hooks, providers, components, types, models, schemas, \
   middleware, config, constants) are NOT features — absorb them into the business feature they support.
8. Skip pure infrastructure directories: .storybook, __mocks__, .github, ci/, scripts/, etc.
9. If a <route-anchors> section is provided, directories with routes are strong anchors.
   Assign nearby sibling directories to the same feature as the directory that shares
   the same route prefix (e.g. /api/payments/* dirs belong to the payments feature).

## Anti-patterns

BAD — too many tiny features (one dir = one feature):
  "login-form":  ["src/auth/login"]
  "signup-form": ["src/auth/signup"]
  "auth-utils":  ["src/auth/utils"]   ← these are all one feature

GOOD — grouped by business domain:
  "user-auth":   ["src/auth/login", "src/auth/signup", "src/auth/utils"]

BAD — too few features (everything lumped into one):
  "core-platform": ["src/auth", "src/billing", "src/webhooks", "src/templates", "src/api"]
  ← these are 5 distinct business capabilities, not one feature

GOOD — each capability is its own feature:
  "user-auth":  ["src/auth"]
  "billing":    ["src/billing"]
  "webhooks":   ["src/webhooks"]
  "templates":  ["src/templates"]
  "api":        ["src/api"]

BAD — splitting page dirs from their API route dirs:
  "organizations": ["app/organizations", "app/organizations/[id]"]
  "backend-api": ["app/api/organizations", "app/api/organizations/[id]", "app/api/cost", ...]
  ← API routes belong to the same feature as the page they serve

GOOD — page + API route dirs unified:
  "organizations": ["app/organizations", "app/organizations/[id]",
    "app/api/organizations", "app/api/organizations/[id]"]

BAD — putting individual filenames in `files`:
  "auth": ["LoginForm.tsx", "useAuth.ts"]  ← WRONG, these are filenames not directories

GOOD — putting directory paths exactly as listed:
  "auth": ["src/auth", "src/hooks/auth", "src/api/auth"]  ← correct

## Example

Input:
  app/organizations → page.tsx, org-list-page.tsx
    app/organizations/[id] → page.tsx, org-detail-page.tsx
  app/api/organizations → route.ts
    app/api/organizations/[id] → route.ts
      app/api/organizations/[id]/sync-status → route.ts
  app/dashboard → page.tsx, dashboard-page.tsx
  app/api/dashboard/stats → route.ts
  app/sign-in → page.tsx
  hooks → use-organizations.ts, use-dashboard.ts
  lib → auth.ts, config.ts
  components/ui → button.tsx, modal.tsx

Result:
  "organizations": ["app/organizations", "app/organizations/[id]",
    "app/api/organizations", "app/api/organizations/[id]",
    "app/api/organizations/[id]/sync-status"]
  "dashboard": ["app/dashboard", "app/api/dashboard/stats"]
  "user-auth": ["app/sign-in", "lib"]
  "shared-ui": ["hooks", "components/ui"]\
"""

_DIR_DETECTION_USER_PROMPT = """\
Group these directories into semantic business features.
{feature_hint}
<directories>
{file_tree}
</directories>{extra_context}
Return directory paths exactly as listed above in the `files` field (not individual filenames). \
Every directory in exactly one feature. Use business domain names.\
"""


class _FeatureFileMapping(BaseModel):
    feature_name: str
    files: list[str]


class _FeatureDetectionResponse(BaseModel):
    features: list[_FeatureFileMapping]


class _FeatureEnrichment(BaseModel):
    original_name: str
    description: str


class _EnrichmentResponse(BaseModel):
    features: list[_FeatureEnrichment]


def detect_features_llm(
    files: list[str],
    api_key: str | None = None,
    commits: list[Commit] | None = None,
    path_prefix: str = "",
    signatures: dict[str, FileSignature] | None = None,
    layer_context: str = "",
    model: str | None = None,
) -> dict[str, list[str]]:
    """
    Sends the repository file tree to Claude and returns a semantic feature mapping.
    Returns {} on any error (caller falls back to heuristic detection).

    When commits are provided, enriches the prompt with:
    - Co-change pairs (files that frequently change together)
    - Commit message keywords per directory

    For large repos (>_DIR_COLLAPSE_THRESHOLD files), collapses to unique directories
    before sending to the LLM — saves tokens and improves accuracy. The returned
    feature→files mapping is then expanded back to full file paths.

    Args:
        files: List of file paths (relative, with path_prefix already stripped).
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        commits: Optional commit history for co-change and keyword enrichment.
        path_prefix: Prefix stripped from files (e.g. "src/"). Used to normalize
            commit paths so they match the stripped file paths.

    Returns:
        dict mapping feature names to lists of file paths.
        Empty dict if LLM call fails.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not files:
        return {}

    effective_model = model or _MODEL
    client = anthropic.Anthropic(api_key=key)

    # Normalize commit file paths to match analysis_files (which have path_prefix stripped)
    norm_commits = _normalize_commit_files(commits, path_prefix) if commits and path_prefix else commits
    cochange_pairs = _compute_cochange(norm_commits) if norm_commits else []

    if len(files) > _DIR_COLLAPSE_THRESHOLD:
        dirs = _unique_dirs(files)
        samples = _dir_to_sample_files(dirs, files)
        dir_keywords = _extract_dir_keywords(dirs, files, norm_commits) if norm_commits else {}
        file_tree = _format_dir_tree(dirs, samples)
        route_anchors = _format_route_anchors(signatures, dirs=dirs) if signatures else ""
        entity_anchors = _format_entity_anchors(signatures, dirs=dirs) if signatures else ""
        extra_context = _format_extra_context(cochange_pairs, dir_keywords) + route_anchors + entity_anchors
        response = _call_dir_detection(client, file_tree, n_dirs=len(dirs), extra_context=extra_context, layer_context=layer_context, model=effective_model)
        if not response:
            return {}
        result = _expand_dir_mapping(response, files)
    else:
        file_tree = "\n".join(files[:_MAX_FILES_FOR_DETECTION])
        route_anchors = _format_route_anchors(signatures) if signatures else ""
        entity_anchors = _format_entity_anchors(signatures) if signatures else ""
        package_anchors = _format_package_anchors(files, signatures)
        extra_context = _format_extra_context(cochange_pairs, {}) + route_anchors + entity_anchors + package_anchors
        response = _call_feature_detection(client, file_tree, extra_context, n_files=len(files), layer_context=layer_context, model=effective_model)
        if not response:
            return {}
        result = _build_feature_dict(response, set(files))

    # Post-process: collapse plugins, extract shared UI, re-split, then redistribute
    result = _collapse_plugin_features(result)
    result = _extract_shared_ui(result)
    result = _resplit_oversized_features(client, result)
    result = _redistribute_oversized_features(result)
    result = _redistribute_infra_features(result)
    return _final_cleanup(result)


# ── Candidate-based detection ──────────────────────────────────────────────
# Uses pre-computed candidates from heuristics and asks LLM to verify/merge/
# split them and assign unmatched files.

_CANDIDATE_SYSTEM_PROMPT = """\
You are a senior software architect reviewing pre-detected feature candidates.

## Task

You will receive a list of feature candidates with file counts. Return ONLY merge/rename operations — \
do NOT return individual file paths.

## Output format

Return a JSON with these operations:
- **merge**: groups of candidate names that should become one feature
- **rename**: candidates that need a better business name
- **remove**: candidates that are infrastructure, not business features (they go to "shared-infra")

## Rules

1. Merge candidates that serve the same business domain (e.g. "investigations" + "investigationdetail" → "investigations")
2. Merge UI-only candidates into their business feature (e.g. "dropdowns", "empty-state" → remove or merge into parent)
3. Rename candidates to use business domain names (lowercase, hyphen-separated, 1-3 words)
4. Remove pure infrastructure candidates: "base-layouts", "navigation", "empty-state", "dropdowns", "icons"
5. Do NOT merge unrelated features — "issues" and "cycles" are separate even if they share code
6. Keep candidates that represent real business capabilities

## Example

Input candidates: issues (447 files), cycles (69), modules (66), workspace-notifications (35), \
dropdowns (26), empty-state (57), base-layouts (17), gantt-chart (76), work-item-filters (7)

Output:
  merge: [["workspace-notifications", "workspace"], ["work-item-filters", "issues"]]
  rename: [{"from": "gantt-chart", "to": "timeline"}]
  remove: ["dropdowns", "empty-state", "base-layouts"]\
"""

_CANDIDATE_USER_PROMPT = """\
Review these feature candidates. Return merge/rename/remove operations ONLY.

<candidates>
{candidates_text}
</candidates>

<unmatched_directories>
{unmatched_text}
</unmatched_directories>
{extra_context}
Return JSON with merge, rename, and remove arrays. Keep it minimal — only change what clearly needs changing.\
"""


class _MergeOperation(BaseModel):
    """List of candidate names to merge (first name becomes the target)."""
    candidates: list[str]


class _RenameOperation(BaseModel):
    from_name: str
    to_name: str


class _CandidateOperations(BaseModel):
    merge: list[list[str]] = []
    rename: list[_RenameOperation] = []
    remove: list[str] = []


def detect_features_with_candidates(
    files: list[str],
    candidates: dict[str, list[str]],
    api_key: str | None = None,
    commits: list[Commit] | None = None,
    path_prefix: str = "",
) -> dict[str, list[str]]:
    """
    Uses pre-computed candidates + LLM verification for feature detection.

    Instead of asking the LLM to return all files, we ask it to return
    OPERATIONS (merge, rename, remove) on the candidate list. This keeps
    the LLM output tiny and avoids truncation on large repos.

    Returns:
        dict mapping feature names to lists of file paths.
        Falls back to candidates as-is if LLM call fails.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return candidates

    client = anthropic.Anthropic(api_key=key)

    # Separate real candidates from unmatched buckets
    real_candidates: dict[str, list[str]] = {}
    unmatched_files: list[str] = []

    _CATCHALL_BUCKETS = {"backend", "frontend", "root", "init", "packages", "web", "api", "lib"}

    for name, paths in candidates.items():
        if name in _CATCHALL_BUCKETS:
            unmatched_files.extend(paths)
        elif len(paths) < 2:
            unmatched_files.extend(paths)
        else:
            real_candidates[name] = paths

    # Format candidates — just names + file counts + sample paths
    candidates_lines = []
    for name, paths in sorted(real_candidates.items(), key=lambda x: -len(x[1])):
        samples = ", ".join(Path(p).name for p in paths[:5])
        candidates_lines.append(f"- {name} ({len(paths)} files): {samples}")
    candidates_text = "\n".join(candidates_lines)

    # Format unmatched as directory summary
    from collections import defaultdict as _ddict
    dir_files: dict[str, list[str]] = _ddict(list)
    for f in unmatched_files:
        d = str(Path(f).parent) if "/" in f else "."
        dir_files[d].append(Path(f).name)
    unmatched_lines = []
    for d in sorted(dir_files.keys()):
        samples = dir_files[d][:3]
        unmatched_lines.append(f"- {d}/ ({len(dir_files[d])} files): {', '.join(samples)}")
    unmatched_text = "\n".join(unmatched_lines) if unmatched_lines else "(none)"

    # Build extra context from commits
    norm_commits = _normalize_commit_files(commits, path_prefix) if commits and path_prefix else commits
    cochange_pairs = _compute_cochange(norm_commits) if norm_commits else []
    extra_context = _format_extra_context(cochange_pairs, {})

    prompt = _CANDIDATE_USER_PROMPT.format(
        candidates_text=candidates_text,
        unmatched_text=unmatched_text,
        extra_context=extra_context,
    )

    system = _CANDIDATE_SYSTEM_PROMPT

    # Call LLM for operations
    ops = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.parse(
                model=_MODEL,
                max_tokens=4096,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=_CandidateOperations,
            )
            if response.parsed_output:
                ops = response.parsed_output
                break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("Candidate LLM call failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
        except ValidationError as e:
            logger.warning("Candidate LLM response validation failed: %s", str(e)[:200])
            break
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError):
            break
        except anthropic.APIStatusError as e:
            logger.warning("Candidate LLM API error: %s", str(e)[:200])
            break

    if not ops:
        logger.warning("Candidate-based LLM call failed — returning raw candidates")
        return candidates

    # Apply operations to candidates
    result = dict(real_candidates)

    # Apply merges — target is the candidate with most files
    for group in ops.merge:
        if len(group) < 2:
            continue
        existing = [(name, len(result.get(name, []))) for name in group if name in result]
        if len(existing) < 2:
            continue
        # Largest candidate becomes the target
        target = max(existing, key=lambda x: x[1])[0]
        for name, _ in existing:
            if name != target:
                result[target].extend(result.pop(name))
                logger.info("Merged '%s' into '%s'", name, target)

    # Apply renames
    for rename in ops.rename:
        if rename.from_name in result:
            result[rename.to_name] = result.pop(rename.from_name)
            logger.info("Renamed '%s' → '%s'", rename.from_name, rename.to_name)

    # Apply removes — move to shared-infra
    infra_files: list[str] = []
    for name in ops.remove:
        if name in result:
            infra_files.extend(result.pop(name))
            logger.info("Removed '%s' → shared-infra", name)
    if infra_files:
        result["shared-infra"] = infra_files

    # Add unmatched files as catch-all buckets (keep their original grouping)
    for name, paths in candidates.items():
        if name in _CATCHALL_BUCKETS and paths:
            result.setdefault(name, []).extend(paths)

    return result


_RESPLIT_FILE_THRESHOLD = 40
_RESPLIT_CONCENTRATION_PCT = 0.70  # re-split if >70% of files in one dir

# Patterns that indicate infrastructure-only features (not business domains)
_INFRA_FEATURE_PATTERNS = {
    "database-migrations", "migrations", "test-fixtures", "fixtures",
    "management-commands", "commands", "config-files", "infrastructure",
    "scripts", "build-scripts", "ci-cd", "tooling",
    "dom-fixtures", "test-fixtures", "storybook",
}

# Partial patterns — features containing these words are likely infrastructure
_INFRA_PARTIAL_PATTERNS = {"fixtures", "demo", "scripts", "examples", "samples"}


_NEXTJS_API_RE = re.compile(r"^(.*?)app/api/([^/]+)")
_NEXTJS_PAGE_RE = re.compile(r"^(.*?)app/([^/]+)")
_HOOK_DOMAIN_RE = re.compile(r"use-([a-z]+)")
_COMPONENT_DOMAIN_RE = re.compile(r"^([a-z]+-)")


def _redistribute_oversized_features(
    result: dict[str, list[str]],
    max_files: int = 40,
) -> dict[str, list[str]]:
    """Deterministic post-processing: move files from oversized features to matching smaller ones.

    For each file in an oversized feature, tries to find a better home:
    1. API route files (app/api/X/) → feature that owns app/X/ pages
    2. Hook files (use-X.ts) → feature with "X" in its name
    3. Component files (org-*.tsx) → feature with matching domain prefix
    4. Remaining files → stay in a residual feature or "shared-ui"/"app-shell"
    """
    # Only redistribute CATCH-ALL features, not legitimate business features
    # A catch-all feature has files spanning many unrelated directories
    _CATCHALL_NAMES = {"app-shell", "shared", "services", "core", "platform",
                       "shared-utilities", "shared-backend", "misc", "other"}
    oversized: dict[str, list[str]] = {}
    for n, fs in result.items():
        if len(fs) <= max_files:
            continue
        # Always redistribute known catch-all names
        if n in _CATCHALL_NAMES or n.startswith("shared-"):
            oversized[n] = fs
            continue
        # Check if it's a catch-all by directory diversity
        biz_dirs: set[str] = set()
        for f in fs[:100]:  # sample first 100 files for speed
            parts = Path(f).parts
            for part in parts[:-1]:
                if part.lower() not in _SKIP_DIRS and part.lower() not in _SHARED_DIR_PATTERNS:
                    biz_dirs.add(part.lower())
                    break
        # If files span many unrelated dirs → catch-all, redistribute
        if len(biz_dirs) > 8:
            oversized[n] = fs

    if not oversized:
        return result

    small = {n: list(fs) for n, fs in result.items() if n not in oversized}

    # Discover page directories in oversized features and create new features for them
    # if they don't already exist (e.g. dashboard pages stuck in a catch-all).
    # Only create features for dirs that have a page.tsx (actual Next.js pages).
    _SKIP_PAGE_DIRS = {"api", "actions"}
    all_oversized_files = [f for fs in oversized.values() for f in fs]
    page_dir_candidates: set[str] = set()
    for f in all_oversized_files:
        m = _NEXTJS_PAGE_RE.match(f)
        if not m or m.group(2) in _SKIP_PAGE_DIRS:
            continue
        page_dir = m.group(2)
        # Skip files directly in app/ root (page.tsx, home-page.tsx, layout.tsx)
        parts_after_app = f[m.end():]
        if "/" not in f[len(m.group(1)) + len("app/") + len(page_dir):].lstrip("/"):
            # This is a direct child of app/<page_dir>/ — it's a real page subdir
            pass
        # Only count if this is a page.tsx or *-page.tsx (not layout.tsx, globals.css, etc.)
        fname = Path(f).name
        if (fname == "page.tsx" or fname.endswith("-page.tsx")) and page_dir not in ("", ):
            # Verify it's a named subdirectory, not root app files
            # app/dashboard/page.tsx → page_dir="dashboard" ✓
            # app/page.tsx → would match page_dir from previous dir — skip by checking
            # the file actually lives inside app/<page_dir>/
            if f"app/{page_dir}/" in f:
                page_dir_candidates.add(page_dir)

    for page_dir in page_dir_candidates:
        # Check if any small feature already owns this page domain
        already_owned = any(
            any(_NEXTJS_PAGE_RE.match(ff) and _NEXTJS_PAGE_RE.match(ff).group(2) == page_dir
                for ff in feat_files)
            for feat_files in small.values()
        )
        if not already_owned and page_dir not in small:
            small[page_dir] = []
            logger.info("Created new feature '%s' from page directory in oversized feature", page_dir)

    # Build page-domain index: "organizations" → feature name
    page_domain_to_feature: dict[str, str] = {}
    for feat_name, feat_files in small.items():
        for f in feat_files:
            m = _NEXTJS_PAGE_RE.match(f)
            if m and m.group(2) not in ("api",):
                page_domain_to_feature[m.group(2)] = feat_name
        # Also register empty features created above (page_dir == feat_name)
        if not feat_files and feat_name not in page_domain_to_feature:
            page_domain_to_feature[feat_name] = feat_name

    # Also index by feature name keywords
    feature_keywords: dict[str, str] = {}
    for feat_name in small:
        for part in feat_name.split("-"):
            if len(part) >= 3:
                feature_keywords[part] = feat_name

    for oversized_name, files in oversized.items():
        remaining = []
        for f in files:
            target = _match_file_to_feature(f, page_domain_to_feature, feature_keywords, small)
            if target:
                small[target].append(f)
            else:
                remaining.append(f)

        if remaining:
            # Extract API-only features: API route dirs with no matching page feature
            api_only: dict[str, list[str]] = {}
            non_api_remaining = []
            for f in remaining:
                api_m = _NEXTJS_API_RE.match(f)
                if api_m:
                    api_domain = api_m.group(2)
                    if api_domain not in page_domain_to_feature:
                        api_only.setdefault(api_domain, []).append(f)
                        continue
                non_api_remaining.append(f)

            for api_domain, api_files in api_only.items():
                small.setdefault(api_domain, []).extend(api_files)
                # Register new feature in keyword index so second pass can match hooks/libs
                for part in api_domain.split("-"):
                    if len(part) >= 3:
                        feature_keywords[part] = api_domain
                logger.info("Created API-only feature '%s' (%d files)", api_domain, len(api_files))

            # Second pass: try matching remaining files against newly created features
            still_remaining = []
            for f in non_api_remaining:
                target = _match_file_to_feature(f, page_domain_to_feature, feature_keywords, small)
                if target:
                    small[target].append(f)
                else:
                    still_remaining.append(f)

            # Group remaining files by meaningful directory into features
            grouped = _group_by_directory(still_remaining, small)
            for group_name, group_files in grouped.items():
                small.setdefault(group_name, []).extend(group_files)

        logger.info(
            "Redistributed '%s' (%d files): %d moved to existing features, %d remaining",
            oversized_name, len(files), len(files) - len(remaining), len(remaining),
        )

    # Remove empty features that didn't get any files
    small = {n: fs for n, fs in small.items() if fs}

    return small


_RE_NUMERIC_SUFFIX = re.compile(r"-\d+$")


def _fix_numeric_feature_names(
    features: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Eliminates features with numeric suffixes (LLM naming bugs).

    For each numeric-suffix feature (e.g. "ndr-2", "ui-4"):
    1. If files span multiple directories → split by directory, name each sub-group
    2. If files are in one directory → derive name from that directory
    3. Try to merge into existing features with matching names
    4. If no match → create properly named new features

    This function GUARANTEES no numeric-suffix features in the output.
    """
    numeric = {}
    result: dict[str, list[str]] = {}

    for name, files in features.items():
        if _RE_NUMERIC_SUFFIX.search(name) and files:
            numeric[name] = files
        else:
            result[name] = files

    if not numeric:
        return result

    for name, files in numeric.items():
        prefix = _RE_NUMERIC_SUFFIX.sub("", name)
        prefix_lower = prefix.lower()

        # Group files by their most specific meaningful directory
        dir_groups: dict[str, list[str]] = {}
        for f in files:
            parts = Path(f).parts
            best_part = None
            for part in parts[:-1]:
                lower = part.lower()
                if lower in _SKIP_DIRS or lower in _SHARED_DIR_PATTERNS:
                    continue
                norm = re.sub(r"([a-z])([A-Z])", r"\1-\2", part).lower()
                if norm == prefix_lower:
                    continue
                best_part = part
            group_key = best_part or "__root__"
            dir_groups.setdefault(group_key, []).append(f)

        # For each sub-group, derive a proper name and merge/create
        for dir_key, group_files in dir_groups.items():
            if dir_key == "__root__":
                new_name = prefix if prefix else name.rstrip("-0123456789")
            else:
                dir_name = re.sub(r"([a-z])([A-Z])", r"\1-\2", dir_key).lower()
                new_name = f"{prefix}-{dir_name}" if prefix else dir_name

            # Try to merge into an existing feature
            merged = False
            # Exact match
            if new_name in result:
                result[new_name].extend(group_files)
                merged = True
            else:
                # Try keyword match against existing features
                keywords = set(new_name.split("-"))
                keywords.discard("")
                for feat_name in list(result.keys()):
                    feat_keywords = set(feat_name.split("-"))
                    # If the new name shares ≥2 keywords with an existing feature, merge
                    overlap = keywords & feat_keywords
                    if len(overlap) >= 2 or (len(overlap) == 1 and len(keywords) == 1):
                        result[feat_name].extend(group_files)
                        merged = True
                        logger.info("Merged numeric '%s' (%s) into existing '%s'", name, dir_key, feat_name)
                        break

            if not merged:
                result.setdefault(new_name, []).extend(group_files)
                logger.info("Renamed numeric '%s' → '%s' (%d files)", name, new_name, len(group_files))

    return result


def _merge_prefix_siblings(
    features: dict[str, list[str]],
    min_siblings: int = 2,
) -> dict[str, list[str]]:
    """Merges features that share a common prefix into one feature.

    Multi-level: tries longest prefix first, then shorter.
    E.g. issues-issue-detail-widgets, issues-sub-issues, issues-peek-overview
    → all merge into "issues" (the shortest common prefix that has siblings).

    Only skips merging when the prefix is a known intentional split (shared-*, ndr-*).
    """
    _SKIP_PREFIXES = {"app", "shared", "custom"}

    from collections import defaultdict

    result = dict(features)
    merged_in_round = True

    # Iterate until no more merges happen (multi-level prefixes need multiple passes)
    while merged_in_round:
        merged_in_round = False
        prefix_groups: dict[str, list[str]] = defaultdict(list)

        for name in result:
            parts = name.split("-")
            # Generate all possible prefixes: "issues-sub" and "issues" for "issues-sub-issues"
            for depth in range(len(parts) - 1, 0, -1):
                prefix = "-".join(parts[:depth])
                if prefix.lower() not in _SKIP_PREFIXES and len(prefix) >= 2:
                    prefix_groups[prefix].append(name)

        # Sort prefixes shortest first — merge at the broadest level
        for prefix in sorted(prefix_groups.keys(), key=len):
            # Only consider siblings that still exist in result
            siblings = [s for s in prefix_groups[prefix] if s in result and s != prefix]
            if len(siblings) < min_siblings:
                continue

            base_name = prefix
            merged_files: list[str] = []

            # Grab the base feature if it exists
            if base_name in result:
                merged_files.extend(result.pop(base_name))

            for s in siblings:
                if s in result:
                    merged_files.extend(result.pop(s))

            if merged_files:
                result[base_name] = merged_files
                merged_in_round = True
                logger.info("Merged %d siblings into '%s' (%d files)", len(siblings), base_name, len(merged_files))

    return result


def _merge_by_directory_ancestry(
    features: dict[str, list[str]],
    overlap_threshold: float = 0.5,
) -> dict[str, list[str]]:
    """Merges features whose files live in the same business directory.

    For each feature, computes the dominant business directory (the top-level
    non-generic dir where most files live). If 2+ features share the same
    dominant directory with >overlap_threshold of their files there, merge them.

    Example: survey-list (80% in app/surveys/), survey-responses (90% in app/surveys/)
    → merged into "surveys" (or the larger feature's name).
    """
    from collections import defaultdict

    def _dominant_dir(files: list[str]) -> tuple[str | None, float]:
        """Returns (dir_name, fraction) of the most common business directory."""
        if not files:
            return None, 0.0
        dir_counts: dict[str, int] = defaultdict(int)
        for f in files:
            parts = Path(f).parts
            for part in parts[:-1]:
                lower = part.lower()
                if lower in _SKIP_DIRS or lower in _SHARED_DIR_PATTERNS:
                    continue
                dir_counts[part] += 1
                break
        if not dir_counts:
            return None, 0.0
        top_dir = max(dir_counts, key=dir_counts.get)
        return top_dir, dir_counts[top_dir] / len(files)

    # Step 1: compute dominant dir for each feature
    feat_dirs: dict[str, tuple[str, float]] = {}
    for name, files in features.items():
        dom_dir, fraction = _dominant_dir(files)
        if dom_dir and fraction >= overlap_threshold:
            feat_dirs[name] = (dom_dir, fraction)

    # Step 2: group features by their dominant directory
    dir_to_features: dict[str, list[str]] = defaultdict(list)
    for name, (dom_dir, _) in feat_dirs.items():
        dir_to_features[dom_dir].append(name)

    # Step 3: merge groups with 2+ features
    result = dict(features)
    for dom_dir, group in dir_to_features.items():
        if len(group) < 2:
            continue

        # Pick the best name: prefer the shortest, or the one matching the directory
        dir_kebab = re.sub(r"([a-z])([A-Z])", r"\1-\2", dom_dir).lower()
        # If one feature name matches the dir name, use it
        best_name = None
        for name in group:
            if name == dir_kebab or name == dom_dir.lower():
                best_name = name
                break
        if not best_name:
            # Use the feature with the most files
            best_name = max(group, key=lambda n: len(result.get(n, [])))

        merged_files: list[str] = []
        for name in group:
            if name in result:
                merged_files.extend(result.pop(name))

        result[best_name] = merged_files
        if len(group) > 1:
            logger.info(
                "Directory merge: %d features in '%s/' → '%s' (%d files)",
                len(group), dom_dir, best_name, len(merged_files),
            )

    return result


_CONSOLIDATION_SYSTEM_PROMPT = """\
You are reviewing a feature list generated by automated code analysis. \
Your job is to consolidate features that are fragments of the same business domain.

## Rules

1. If two or more features clearly serve the SAME business domain, merge them. \
   Use the most descriptive name for the merged feature.
   Example: "survey-list", "survey-responses", "survey-analysis" → "surveys"
   Example: "project-states", "project-sidebar", "project-management" → "project-management"

2. Do NOT merge features that represent genuinely different business capabilities. \
   "billing" and "user-auth" are separate even though both are in settings.

3. Keep features that are distinct product areas even if they have few files. \
   "webhooks" (8 files) is a real feature, not a fragment.

4. The result should be what an Engineering Manager would recognize as their product's feature list.

5. Return ALL features — both merged and untouched ones.
"""

_CONSOLIDATION_USER_PROMPT = """\
Here are {n_features} features detected from code analysis. \
An engineering manager expects roughly {expected_min}–{expected_max} business features.

Features:
{feature_list}

For each group of features that should merge, provide:
- merged_name: the consolidated feature name
- original_names: list of feature names being merged

Also list features that should stay as-is (keep_names).
Every feature must appear exactly once — either in a merge group or in keep_names.\
"""


class _ConsolidationMerge(BaseModel):
    merged_name: str
    original_names: list[str]


class _ConsolidationResponse(BaseModel):
    merges: list[_ConsolidationMerge]
    keep_names: list[str]


def _consolidate_features_llm(
    features: dict[str, list[str]],
    client: anthropic.Anthropic,
) -> dict[str, list[str]]:
    """Second-pass LLM call: consolidates fragmented features.

    Sends only feature names + file counts (cheap, fast).
    Returns merged feature dict.
    """
    if len(features) <= 15:
        return features  # already compact

    # Estimate expected range from file count
    total_files = sum(len(fs) for fs in features.values())
    expected_min = max(8, total_files // 120)
    expected_max = max(15, total_files // 50)
    expected_max = min(expected_max, len(features))

    feature_lines = []
    for name, files in sorted(features.items(), key=lambda x: -len(x[1])):
        # Show sample dirs for context
        dirs = set()
        for f in files[:20]:
            parts = Path(f).parts
            for part in parts[:-1]:
                if part.lower() not in _SKIP_DIRS:
                    dirs.add(part)
                    break
        dir_hint = f" (dirs: {', '.join(sorted(dirs)[:3])})" if dirs else ""
        feature_lines.append(f"  {name}: {len(files)} files{dir_hint}")

    prompt = _CONSOLIDATION_USER_PROMPT.format(
        n_features=len(features),
        expected_min=expected_min,
        expected_max=expected_max,
        feature_list="\n".join(feature_lines),
    )

    try:
        response = client.messages.parse(
            model=_MODEL,
            max_tokens=2048,
            temperature=0,
            system=_CONSOLIDATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=_ConsolidationResponse,
        )
        parsed = response.parsed_output

        result: dict[str, list[str]] = {}

        # Apply merges
        for merge in parsed.merges:
            merged_files: list[str] = []
            for orig_name in merge.original_names:
                if orig_name in features:
                    merged_files.extend(features[orig_name])
            if merged_files:
                result[merge.merged_name] = merged_files
                logger.info(
                    "Consolidated %d features → '%s' (%d files)",
                    len(merge.original_names), merge.merged_name, len(merged_files),
                )

        # Keep untouched features
        for name in parsed.keep_names:
            if name in features and name not in result:
                result[name] = features[name]

        # Safety: any features not mentioned by LLM stay as-is
        all_mentioned = set()
        for merge in parsed.merges:
            all_mentioned.update(merge.original_names)
            all_mentioned.add(merge.merged_name)
        all_mentioned.update(parsed.keep_names)

        for name, files in features.items():
            if name not in all_mentioned and name not in result:
                result[name] = files

        if result:
            # Guardrail: don't consolidate below minimum
            if len(result) < expected_min:
                logger.warning(
                    "Consolidation too aggressive: %d → %d (min %d) — keeping original",
                    len(features), len(result), expected_min,
                )
                return features
            logger.info("Consolidation: %d → %d features", len(features), len(result))
            return result

    except Exception as e:
        logger.warning("Consolidation pass failed: %s — keeping original features", e)

    return features


def _final_cleanup(result: dict[str, list[str]]) -> dict[str, list[str]]:
    """Guaranteed last step in every pipeline path.

    1. Collapse tech-layer features (api, prisma, trpc, lib, .yarn, etc.) into shared-infra
    2. Fix numeric feature names (ndr-2 → ndr-auto-segmentation)
    3. Merge prefix siblings (complex-query-hooks + complex-query-utils → complex-query)
    4. Absorb tiny features into related larger ones
    5. Collapse infra-only features into shared-infra
    6. Remove empty features
    """
    result = {n: fs for n, fs in result.items() if fs}
    result = _collapse_tech_layer_features(result)
    result = _fix_numeric_feature_names(result)
    result = _merge_prefix_siblings(result)
    result = _absorb_tiny_features(result)
    result = _collapse_infra_only_features(result)
    return {n: fs for n, fs in result.items() if fs}


def _collapse_tech_layer_features(
    features: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Collapses features named after technical layers into shared-infra.

    Features like "api", "prisma", "trpc", "lib", ".yarn" are technical
    infrastructure — not business features. Their files get merged into
    shared-infra so they don't pollute the feature map.
    """
    # Only collapse features that are UNAMBIGUOUSLY technical — never business domains.
    # "api", "lib", "features" are excluded because they CAN be business features.
    _TECH_LAYER_NAMES = {
        # ORM / database layers
        "prisma", "drizzle", "typeorm", "knex", "sequelize",
        # API framework layers
        "trpc", "graphql-schema",
        # Build / tooling (hidden dirs)
        ".yarn", ".changeset", ".claude", ".agents", ".opencode",
        ".devcontainer", ".snaplet", ".storybook", ".husky",
        # Non-business
        "specs", "testing", "example-apps", "examples",
        "dayjs",
    }

    result: dict[str, list[str]] = {}
    infra_files: list[str] = []

    for name, paths in features.items():
        if name.lower() in _TECH_LAYER_NAMES:
            infra_files.extend(paths)
        else:
            result[name] = paths

    if infra_files:
        result.setdefault("shared-infra", []).extend(infra_files)

    return result


def _absorb_tiny_features(
    features: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Absorbs very small features into the most related larger feature.

    Uses a relative threshold: the median feature size. Features significantly
    below median are fragments — they get absorbed into the feature with the
    most keyword overlap in its name.

    The intuition: in a healthy codebase, real features are roughly similar
    in size. A 3-file "feature" in a repo where median is 50 files is noise.
    """
    _PROTECTED = {"shared-ui", "shared-infra", "project-config"}

    sizes = sorted([len(p) for n, p in features.items() if n not in _PROTECTED and p])
    if len(sizes) < 5:
        return features

    # Threshold: features below 10% of median size are fragments
    median = sizes[len(sizes) // 2]
    threshold = max(2, median // 10)

    result: dict[str, list[str]] = {}
    tiny: dict[str, list[str]] = {}

    for name, paths in features.items():
        if len(paths) <= threshold and name not in _PROTECTED:
            tiny[name] = paths
        else:
            result[name] = paths

    if not tiny or not result:
        result.update(tiny)
        return result

    for name, paths in tiny.items():
        target = _find_keyword_match(name, result)
        if target:
            result[target].extend(paths)
        else:
            # No match — put in shared-infra
            result.setdefault("shared-infra", []).extend(paths)

    return result


def _collapse_infra_only_features(
    features: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Collapses features whose files are ALL in infrastructure directories.

    If every file in a feature lives in a technical/shared directory
    (hooks/, utils/, lib/, types/, config/), it's infrastructure — not a
    business feature. Merge into shared-infra.
    """
    _INFRA_DIR_NAMES = {
        "hooks", "utils", "helpers", "lib", "types", "config", "configs",
        "constants", "middleware", "providers", "contexts", "assets",
        "styles", "icons", "svg", "public", "static",
    }

    result: dict[str, list[str]] = {}
    infra_files: list[str] = []

    for name, paths in features.items():
        if name in {"shared-ui", "shared-infra", "project-config"}:
            result[name] = paths
            continue

        # Check if ALL files are in infra dirs
        all_infra = True
        for fp in paths:
            parts = Path(fp).parts
            # Check if any meaningful part is NOT an infra dir
            has_business_dir = False
            for part in parts[:-1]:
                lower = part.lower()
                if lower not in _INFRA_DIR_NAMES and lower not in {
                    "src", "app", "core", "packages", "apps", "web", "api",
                }:
                    has_business_dir = True
                    break
            if has_business_dir:
                all_infra = False
                break

        if all_infra and len(paths) > 0:
            infra_files.extend(paths)
        else:
            result[name] = paths

    if infra_files:
        result.setdefault("shared-infra", []).extend(infra_files)

    return result


def _find_keyword_match(
    name: str,
    features: dict[str, list[str]],
) -> str | None:
    """Finds the best feature to absorb a small feature by keyword overlap."""
    name_parts = set(re.split(r"[-_]", name))
    name_parts.discard("")

    best_match = None
    best_score = 0

    for feat_name, feat_paths in features.items():
        if feat_name in {"shared-ui", "shared-infra", "project-config"}:
            continue
        feat_parts = set(re.split(r"[-_]", feat_name))
        overlap = name_parts & feat_parts
        # Score: keyword overlap weighted by target size (prefer larger features)
        score = len(overlap) * 10 + min(len(feat_paths), 50)
        if overlap and score > best_score:
            best_match = feat_name
            best_score = score

    return best_match


# Directories that are shared infrastructure, not business features
_SHARED_DIR_PATTERNS = {
    "components", "hooks", "lib", "utils", "helpers", "types", "models",
    "schemas", "constants", "config", "configs", "middleware", "providers",
    "contexts", "store", "stores", "assets", "styles", "icons", "svg",
    "stories", "storybook", "__tests__", "__mocks__", "fixtures", "test",
}

# Skip these as feature-directory candidates (too generic) — go deeper
_SKIP_DIRS = {
    "src", "app", "pages", "api", "public", "static", "dist", "build",
    "views", "features", "modules", "domains", "routes", "screens",
}


def _extract_domain_keywords(filename: str) -> list[str]:
    """Extracts potential business-domain keywords from a filename.

    'useEntitySearch.tsx' → ['entity', 'search']
    'useHover.ts' → ['hover']
    'exportCSV.ts' → ['export']
    """
    stem = Path(filename).stem
    # Strip common prefixes
    for prefix in ("use", "with", "create", "get", "set", "fetch", "load"):
        if stem.lower().startswith(prefix) and len(stem) > len(prefix):
            stem = stem[len(prefix):]
            break
    # Split camelCase/PascalCase into words
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem).lower().split()
    # Filter out very short or generic words
    generic = {"index", "main", "app", "page", "view", "base", "common", "default",
               "test", "spec", "mock", "stub", "type", "types", "interface", "model"}
    return [w for w in words if len(w) >= 3 and w not in generic]


# Words that indicate a truly generic/shared utility (no business domain)
_GENERIC_HOOK_KEYWORDS = {
    "hover", "click", "outside", "debounce", "throttle", "mount", "unmount",
    "previous", "prev", "props", "update", "effect", "deep", "compare",
    "stable", "callback", "interval", "periodic", "animation", "count",
    "resize", "scroll", "window", "media", "query", "local", "storage",
    "session", "ref", "focus", "keyboard", "shortcut", "clipboard", "copy",
    "toggle", "boolean", "array", "object", "memo", "lazy", "async",
    "timeout", "raf", "intersection", "observer", "mutation",
}


def _group_by_directory(
    files: list[str],
    existing_features: dict[str, list[str]],
    min_group_size: int = 3,
) -> dict[str, list[str]]:
    """Groups remaining files into features by business domain.

    First tries to match each file to an existing feature by domain keywords
    in the filename. Only files that can't be matched go into directory-based
    groups or shared buckets.
    """
    # Build keyword → feature index from existing features
    feat_keyword_index: dict[str, str] = {}
    for feat_name in existing_features:
        for part in feat_name.split("-"):
            if len(part) >= 3:
                feat_keyword_index[part.lower()] = feat_name

    dir_groups: dict[str, list[str]] = {}
    shared_ui: list[str] = []
    shared_generic: list[str] = []

    for f in files:
        parts = Path(f).parts
        filename = parts[-1] if parts else f

        # Step 1: Try to match by filename keywords to existing features
        keywords = _extract_domain_keywords(filename)
        matched_feature = None
        for kw in keywords:
            if kw in feat_keyword_index:
                matched_feature = feat_keyword_index[kw]
                break

        if matched_feature:
            existing_features[matched_feature].append(f)
            continue

        # Step 2: Check if it's in a shared dir
        in_shared_dir = False
        for part in parts[:-1]:
            if part.lower() in _SHARED_DIR_PATTERNS:
                in_shared_dir = True
                break

        if in_shared_dir:
            # Check if filename suggests a specific business domain
            # by looking at directory structure for sub-features
            has_domain_sub = False
            for i, part in enumerate(parts[:-1]):
                if part.lower() in _SHARED_DIR_PATTERNS and i + 1 < len(parts) - 1:
                    next_part = parts[i + 1]
                    if next_part.lower() not in _SHARED_DIR_PATTERNS and not next_part.startswith("_"):
                        name = re.sub(r"([a-z])([A-Z])", r"\1-\2", next_part).lower()
                        dir_groups.setdefault(name, []).append(f)
                        has_domain_sub = True
                        break

            if not has_domain_sub:
                # Check if filename keywords are all generic
                is_generic = not keywords or all(kw in _GENERIC_HOOK_KEYWORDS for kw in keywords)
                if "components/ui" in f or "components/UI" in f:
                    shared_ui.append(f)
                elif is_generic:
                    shared_generic.append(f)
                else:
                    # Has domain keywords but no matching feature — group by first keyword
                    group_name = keywords[0] if keywords else "app-shell"
                    dir_groups.setdefault(group_name, []).append(f)
            continue

        # Step 3: Find meaningful directory for non-shared files
        meaningful_dir = None
        for part in parts[:-1]:
            lower = part.lower()
            if lower in _SKIP_DIRS:
                continue
            meaningful_dir = part
            break

        if meaningful_dir:
            name = re.sub(r"([a-z])([A-Z])", r"\1-\2", meaningful_dir).lower()
            if name in existing_features:
                existing_features[name].append(f)
            else:
                dir_groups.setdefault(name, []).append(f)
        else:
            shared_generic.append(f)

    result: dict[str, list[str]] = {}

    for name, group_files in dir_groups.items():
        if len(group_files) < min_group_size:
            # Try to merge small groups into existing features
            merged = False
            for kw in name.split("-"):
                if kw in feat_keyword_index:
                    existing_features[feat_keyword_index[kw]].extend(group_files)
                    merged = True
                    break
            if not merged:
                shared_generic.extend(group_files)
        else:
            result[name] = group_files

    if shared_ui:
        result["shared-ui"] = shared_ui

    if shared_generic:
        result["app-shell"] = shared_generic

    return result


def _split_by_subdir(
    parent_name: str,
    files: list[str],
    min_group_size: int,
    max_size: int = 50,
) -> dict[str, list[str]]:
    """Splits a large group by the deepest common directory, recursively.

    Finds the longest common path prefix across all files, then groups by the
    next directory level after that prefix. Recurses on groups still > max_size.
    """
    if len(files) <= max_size:
        return {parent_name: files}

    # Find common path prefix (directory level)
    split_parts = [Path(f).parts[:-1] for f in files]  # exclude filename
    if not split_parts:
        return {parent_name: files}

    # Common prefix length
    min_depth = min(len(p) for p in split_parts) if split_parts else 0
    common_len = 0
    for i in range(min_depth):
        vals = {p[i] for p in split_parts}
        if len(vals) == 1:
            common_len = i + 1
        else:
            break

    # Group by the part right after common prefix
    sub_groups: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for f in files:
        parts = Path(f).parts
        if common_len < len(parts) - 1:
            sub_dir = parts[common_len]
            sub_name = re.sub(r"([a-z])([A-Z])", r"\1-\2", sub_dir).lower()
            full_name = f"{parent_name}-{sub_name}" if sub_name != parent_name else sub_name
            sub_groups.setdefault(full_name, []).append(f)
        else:
            ungrouped.append(f)

    if len(sub_groups) <= 1:
        # Can't split further at this level — try next level
        if common_len + 1 < min_depth:
            deeper_groups: dict[str, list[str]] = {}
            for f in files:
                parts = Path(f).parts
                if common_len + 1 < len(parts) - 1:
                    sub_dir = parts[common_len + 1]
                    sub_name = re.sub(r"([a-z])([A-Z])", r"\1-\2", sub_dir).lower()
                    full_name = f"{parent_name}-{sub_name}"
                    deeper_groups.setdefault(full_name, []).append(f)
                else:
                    ungrouped.append(f)
            if len(deeper_groups) > 1:
                sub_groups = deeper_groups
            else:
                return {parent_name: files}
        else:
            return {parent_name: files}

    result: dict[str, list[str]] = {}
    for name, group_files in sub_groups.items():
        if len(group_files) < min_group_size:
            ungrouped.extend(group_files)
        elif len(group_files) > max_size:
            # Recurse
            sub_result = _split_by_subdir(name, group_files, min_group_size, max_size)
            result.update(sub_result)
        else:
            result[name] = group_files

    if ungrouped:
        result[parent_name] = ungrouped

    return result


def _match_file_to_feature(
    file_path: str,
    page_domain_to_feature: dict[str, str],
    feature_keywords: dict[str, str],
    features: dict[str, list[str]],
) -> str | None:
    """Tries to match a single file to an existing feature by path patterns."""
    # 1. API route → page feature: app/api/organizations/... → "organizations" page domain
    api_match = _NEXTJS_API_RE.match(file_path)
    if api_match:
        api_domain = api_match.group(2)
        if api_domain in page_domain_to_feature:
            return page_domain_to_feature[api_domain]

    # 2. Hook file → feature by domain: use-organizations.ts → "organizations"
    filename = Path(file_path).stem
    hook_match = _HOOK_DOMAIN_RE.match(filename)
    if hook_match:
        hook_domain = hook_match.group(1)
        # Try plural, singular, and abbreviation expansion
        for variant in (hook_domain, hook_domain + "s", hook_domain.rstrip("s")):
            if variant in feature_keywords:
                return feature_keywords[variant]
        # Also try matching hook domain as prefix of feature keywords
        for kw, feat in feature_keywords.items():
            if kw.startswith(hook_domain) or hook_domain.startswith(kw):
                return feat

    # 3. Component with domain prefix: org-sync-status.tsx → "organization*"
    if "components/" in file_path and "components/ui/" not in file_path:
        comp_match = _COMPONENT_DOMAIN_RE.match(filename)
        if comp_match:
            prefix = comp_match.group(1).rstrip("-")
            for variant in (prefix, prefix + "s", prefix.rstrip("s")):
                if variant in feature_keywords:
                    return feature_keywords[variant]
            # Expand abbreviations: "org" → match "organization*"
            for kw, feat in feature_keywords.items():
                if kw.startswith(prefix) or prefix.startswith(kw):
                    return feat

    # 4. lib/auth.ts → feature with "auth" in name
    if "lib/" in file_path:
        for part in Path(file_path).stem.split("-"):
            if len(part) >= 3 and part in feature_keywords:
                return feature_keywords[part]

    # 5. File in a page directory: app/dashboard/layout.tsx → "dashboard"
    page_match = _NEXTJS_PAGE_RE.match(file_path)
    if page_match and page_match.group(2) != "api":
        page_dir = page_match.group(2)
        if page_dir in page_domain_to_feature:
            return page_domain_to_feature[page_dir]

    # 6. Any file with feature keyword in filename: dashboard-layout.tsx → "dashboard"
    for part in filename.split("-"):
        if len(part) >= 4 and part in feature_keywords:
            return feature_keywords[part]

    return None


def _collapse_plugin_features(
    features: dict[str, list[str]],
    min_siblings: int = 8,
    max_files_per_sibling: int = 8,
) -> dict[str, list[str]]:
    """Collapses plugin/integration directory patterns into single features.

    Detects when many small features share the same parent directory structure
    (e.g. 152 app-store integrations in cal.com). These are a plugin registry,
    not 152 separate business features.

    Detection criteria:
    - N+ features whose files share the same parent directory pattern
    - Each feature has ≤max_files_per_sibling files
    - The parent directory has a recognizable plugin pattern

    Only triggers for large repos with many small similarly-structured features.
    """
    from collections import defaultdict

    if len(features) < 20:
        return features  # small repos don't have this problem

    # For each feature, find the common parent directory of its files
    feat_parent: dict[str, str] = {}
    for name, files in features.items():
        if not files or len(files) > max_files_per_sibling:
            continue
        # Find common parent: go 2 levels up from files
        parents = set()
        for f in files:
            parts = Path(f).parts
            # Use grandparent dir (parent of the plugin dir)
            if len(parts) >= 3:
                parents.add("/".join(parts[:-2]))
            elif len(parts) >= 2:
                parents.add(parts[0])
        if len(parents) == 1:
            feat_parent[name] = parents.pop()

    # Group features by their common parent
    parent_groups: dict[str, list[str]] = defaultdict(list)
    for name, parent in feat_parent.items():
        parent_groups[parent].append(name)

    # Collapse groups with enough siblings
    result = dict(features)
    for parent, group in parent_groups.items():
        if len(group) < min_siblings:
            continue

        # Derive a meaningful name from the parent directory
        parent_parts = parent.split("/")
        # Find last meaningful part
        name_parts = []
        for part in reversed(parent_parts):
            lower = part.lower()
            if lower not in _SKIP_DIRS and lower not in _SHARED_DIR_PATTERNS:
                name_parts.insert(0, part)
                break

        if not name_parts:
            collapse_name = "integrations"
        else:
            collapse_name = re.sub(r"([a-z])([A-Z])", r"\1-\2", name_parts[0]).lower()
            # Common patterns
            if any(kw in collapse_name for kw in ("app", "store", "plugin", "connector", "integration")):
                collapse_name = "integrations"

        # Merge all sibling features into one
        merged_files: list[str] = []
        for name in group:
            if name in result:
                merged_files.extend(result.pop(name))

        # Add to existing feature or create new
        if collapse_name in result:
            result[collapse_name].extend(merged_files)
        else:
            result[collapse_name] = merged_files

        logger.info(
            "Plugin collapse: %d features under '%s/' → '%s' (%d files)",
            len(group), parent, collapse_name, len(merged_files),
        )

    return result


def _extract_shared_ui(
    result: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Extracts shared UI library files from business features into a dedicated shared-ui feature.

    Files matching shared UI patterns (components/ui/, common UI directories) are
    generic reusable components that happen to be imported by business features.
    The import graph clusters them together, but they should be in their own bucket.

    Only extracts from features where the UI files are clearly a shared library
    (living in a generic ui/ directory), not domain-specific UI (e.g. NDR/components/ui/).
    """
    # Patterns that indicate a shared UI library (not domain-specific UI)
    _SHARED_UI_PATTERNS = (
        "/components/ui/",
        "/ui/hooks/",
        "/ui/@types/",
    )
    # Domain-specific UI dirs that should stay in their business feature
    # e.g. src/views/NDR/Home/components/ui/ is NDR-specific
    _DOMAIN_UI_INDICATORS = ("/views/", "/features/", "/pages/", "/screens/")

    shared_ui_files: list[str] = []
    cleaned: dict[str, list[str]] = {}

    for name, files in result.items():
        kept: list[str] = []
        for f in files:
            is_shared_ui = (
                any(p in f for p in _SHARED_UI_PATTERNS)
                and not any(d in f.split("/components/ui/")[0] if "/components/ui/" in f else "" for d in _DOMAIN_UI_INDICATORS)
            )
            if is_shared_ui:
                # Check it's truly top-level shared, not nested under a domain view
                prefix = f.split("/components/ui/")[0] if "/components/ui/" in f else f.split("/ui/hooks/")[0] if "/ui/hooks/" in f else ""
                is_domain = any(indicator in prefix for indicator in _DOMAIN_UI_INDICATORS)
                if is_domain:
                    kept.append(f)
                else:
                    shared_ui_files.append(f)
            else:
                kept.append(f)
        if kept:
            cleaned[name] = kept

    if shared_ui_files:
        # Merge into existing shared-ui or create new
        existing = cleaned.get("shared-ui", [])
        existing.extend(shared_ui_files)
        cleaned["shared-ui"] = list(dict.fromkeys(existing))  # dedup
        logger.info("Extracted %d shared UI files from business features into 'shared-ui'", len(shared_ui_files))

    return cleaned


def _redistribute_infra_features(
    result: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Redistribute files from infrastructure-only features to their nearest business feature."""
    infra_names = [
        name for name in result
        if name in _INFRA_FEATURE_PATTERNS
        or any(name.endswith(f"-{p}") or name.startswith(f"{p}-") for p in ("migrations", "fixtures", "commands"))
        or any(p in name for p in _INFRA_PARTIAL_PATTERNS)
    ]
    if not infra_names:
        return result

    business = {n: fs for n, fs in result.items() if n not in infra_names}
    if not business:
        return result

    for infra_name in infra_names:
        infra_files = result[infra_name]
        for f in infra_files:
            target = _find_best_merge_target([f], business)
            if target:
                business[target].append(f)
            else:
                business.setdefault("shared-utilities", []).append(f)
        logger.info("Redistributed %d files from '%s' to business features", len(infra_files), infra_name)

    return business


def _resplit_oversized_features(
    client: anthropic.Anthropic,
    result: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Re-splits oversized or directory-concentrated features.

    Triggers when:
    - A feature has >_RESPLIT_FILE_THRESHOLD files (too many files), OR
    - >70% of a feature's files are in a single directory AND the feature
      has >= 20 files (indicates dir-collapse couldn't split properly).
    """
    resplit_needed = {}
    for name, feat_files in result.items():
        if len(feat_files) > _RESPLIT_FILE_THRESHOLD:
            resplit_needed[name] = feat_files
        elif len(feat_files) >= 20:
            dir_counts: dict[str, int] = {}
            for f in feat_files:
                d = str(Path(f).parent)
                dir_counts[d] = dir_counts.get(d, 0) + 1
            max_dir_pct = max(dir_counts.values()) / len(feat_files) if dir_counts else 0
            if max_dir_pct >= _RESPLIT_CONCENTRATION_PCT:
                resplit_needed[name] = feat_files
    if not resplit_needed:
        return result

    for feat_name, feat_files in resplit_needed.items():
        logger.info("Re-splitting oversized feature '%s' (%d files) with Sonnet", feat_name, len(feat_files))
        sub_tree = "\n".join(feat_files)
        min_sub = max(2, len(feat_files) // 40)
        max_sub = max(3, len(feat_files) // 15)
        resplit_prompt = (
            f"These {len(feat_files)} files were all grouped into one feature '{feat_name}'. "
            f"This is too coarse. Split them into {min_sub}–{max_sub} distinct sub-features "
            "based on what each file DOES, not which directory it's in.\n\n"
            "KEY PATTERNS to look for:\n"
            "- In Next.js app-router: match page dirs with their API route dirs by shared path segment. "
            "app/organizations/ (page) + app/api/organizations/ (routes) = same feature. "
            "app/dashboard/ + app/api/dashboard/ = same feature. NEVER group all API routes together.\n"
            "- Hooks belong to the feature they serve: use-organizations.ts → organizations feature.\n"
            "- Components with domain prefixes belong to that domain: org-sync-status.tsx → organizations.\n"
            "- In Django/Rails/Flask apps, filenames reveal business domains: "
            "tags.py, permissions.py, bulk_edit.py, workflows.py, custom_fields.py each "
            "represent a SEPARATE business capability, even if they're in the same directory.\n"
            "- Files named models.py, views.py, serializers.py, admin.py are shared across "
            "ALL features in that app — assign them to the LARGEST or most core sub-feature.\n"
            "- Match test files (test_tags.py) with their source (tags.py).\n"
            "- Shared UI components (button.tsx, modal.tsx, card.tsx) → 'shared-ui' feature.\n"
            "- lib/ utility files → assign to the business feature that uses them most.\n"
            "- Do NOT split a feature that is already cohesive (e.g. 'email-ingestion' with "
            "mail fetching, parsing, rules all serving the same business domain). Only split "
            "when filenames clearly indicate DIFFERENT user-facing capabilities.\n\n"
            f"<file_list>\n{sub_tree}\n</file_list>\n"
            "Return a JSON mapping of sub-feature names to file lists. "
            "Each file in exactly one feature. Use business domain names."
        )
        try:
            # Use Haiku with messages.parse (Sonnet 4 doesn't support output_format)
            resp = client.messages.parse(
                model=_MODEL,
                max_tokens=_MAX_TOKENS_FILE,
                temperature=0,
                system=_DETECTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": resplit_prompt}],
                output_format=_FeatureDetectionResponse,
            )
            if resp.parsed_output:
                sub_features = _build_feature_dict(resp.parsed_output, set(feat_files))
                if len(sub_features) > 1:
                    del result[feat_name]
                    result.update(sub_features)
                    logger.info("Re-split '%s' → %d sub-features", feat_name, len(sub_features))
        except Exception as e:
            logger.warning("Re-split failed for '%s': %s", feat_name, e)

    return result



def _call_feature_detection(
    client: anthropic.Anthropic,
    file_tree: str,
    extra_context: str = "",
    n_files: int = 0,
    layer_context: str = "",
    model: str | None = None,
) -> _FeatureDetectionResponse | None:
    """Calls Claude API for feature detection (file-path mode). Returns None on any failure."""
    hint = _file_feature_count_hint(n_files) if n_files else ""
    prompt = _DETECTION_USER_PROMPT.format(
        file_tree=file_tree, extra_context=extra_context, feature_hint=hint,
    )

    # Inject minimum feature count into system prompt for large repos
    system = _DETECTION_SYSTEM_PROMPT + layer_context
    if n_files >= 100:
        min_f = min(max(8, n_files // 30), 15)
        system += (
            f"\n\n## CRITICAL REQUIREMENT\n"
            f"This codebase has {n_files} files. You MUST return at least {min_f} features. "
            f"Producing fewer than {min_f} means you are over-merging distinct business capabilities. "
            f"Each of these should be separate: auth, billing, webhooks, templates, integrations, "
            f"api, settings, notifications, admin, teams, etc."
        )

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.parse(
                model=model or _MODEL,
                max_tokens=_MAX_TOKENS_FILE,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=_FeatureDetectionResponse,
            )
            return response.parsed_output
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...", attempt + 1, _MAX_RETRIES, e, delay)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            ValidationError,
        ):
            return None
        except anthropic.APIStatusError:
            return None
    return None


def _call_dir_detection(
    client: anthropic.Anthropic,
    file_tree: str,
    n_dirs: int,
    extra_context: str = "",
    layer_context: str = "",
    model: str | None = None,
) -> _FeatureDetectionResponse | None:
    """
    Calls Claude API for dir-collapse feature detection.
    Uses dir-specific prompts and a larger token budget to accommodate
    responses that list hundreds of directory paths.
    Returns None on any failure.
    """
    prompt = _DIR_DETECTION_USER_PROMPT.format(
        file_tree=file_tree,
        feature_hint=_feature_count_hint(n_dirs),
        extra_context=extra_context,
    )

    # Inject minimum feature count into system prompt
    system = _DIR_DETECTION_SYSTEM_PROMPT + layer_context
    min_f = min(max(8, n_dirs // 15), 15)
    system += (
        f"\n\n## CRITICAL REQUIREMENT\n"
        f"This codebase has {n_dirs} directories. You MUST return at least {min_f} features. "
        f"Producing fewer than {min_f} means you are over-merging distinct business capabilities. "
        f"Each of these should be separate: auth, billing, webhooks, templates, integrations, "
        f"api, settings, notifications, admin, teams, etc."
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.parse(
                model=model or _MODEL,
                max_tokens=_MAX_TOKENS_DIR,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=_FeatureDetectionResponse,
            )
            return response.parsed_output
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("LLM dir-detection failed (attempt %d/%d): %s. Retrying in %.1fs...", attempt + 1, _MAX_RETRIES, e, delay)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            ValidationError,
        ):
            return None
        except anthropic.APIStatusError:
            return None
    return None


def _build_feature_dict(
    response: _FeatureDetectionResponse,
    allowed_files: set[str],
) -> dict[str, list[str]]:
    """Converts the LLM response into a dict, filtering out unknown file paths."""
    result: dict[str, list[str]] = {}
    for mapping in response.features:
        valid_files = [f for f in mapping.files if f in allowed_files]
        if valid_files:
            result[mapping.feature_name] = valid_files
    return result


def _unique_dirs(files: list[str]) -> list[str]:
    """
    Extracts unique directory paths from a file list, sorted.
    Skips the root (files with no parent directory).
    """
    dirs: set[str] = set()
    for f in files:
        parent = str(Path(f).parent)
        if parent != ".":
            dirs.add(parent)
    return sorted(dirs)


def _dir_to_sample_files(dirs: list[str], all_files: list[str]) -> dict[str, list[str]]:
    """For each directory, returns sample file names.

    Large directories (>20 files) get up to 20 samples to give the LLM enough
    context to detect multiple business domains within a single directory
    (common in Django/Rails monoliths).
    """
    # Count files per directory first
    dir_file_count: dict[str, int] = {d: 0 for d in dirs}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent in dir_file_count:
            dir_file_count[parent] += 1

    samples: dict[str, list[str]] = {d: [] for d in dirs}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent not in samples:
            continue
        limit = 20 if dir_file_count[parent] > 20 else _DIR_SAMPLE_FILES
        if len(samples[parent]) < limit:
            samples[parent].append(Path(f).name)
    return samples


def _format_dir_tree(dirs: list[str], samples: dict[str, list[str]]) -> str:
    """
    Formats directories with sample file names and hierarchical indentation.
    Child directories (whose parent also appears in the list) are indented to
    visually communicate nesting depth to the LLM.
    """
    dir_set = set(dirs)
    lines = []
    for d in dirs:
        depth = _dir_nesting_depth(d, dir_set)
        indent = "  " * (depth + 1)
        s = samples.get(d, [])
        suffix = f" → {', '.join(s)}" if s else ""
        lines.append(f"{indent}{d}{suffix}")
    return "\n".join(lines)


def _dir_nesting_depth(d: str, dir_set: set[str]) -> int:
    """
    Returns how many ancestor directories of `d` also appear in `dir_set`.
    Used to compute indentation depth for the dir tree.
    """
    depth = 0
    current = d
    while True:
        parent = str(Path(current).parent)
        if parent == "." or parent == current:
            break
        if parent in dir_set:
            depth += 1
        current = parent
    return depth


def _feature_count_hint(n_dirs: int) -> str:
    """Generates a feature-count guidance line for the dir-collapse prompt."""
    min_f = min(max(8, n_dirs // 15), 15)
    max_f = min(max(15, n_dirs // 7), 30)
    return (
        f"\nYou have {n_dirs} directories. "
        f"You MUST produce at least {min_f} features (aim for {min_f}–{max_f}). "
        "Each distinct business capability (auth, billing, webhooks, templates, "
        "integrations, etc.) MUST be its own feature. "
        "Do NOT merge unrelated capabilities into 'core-platform' or 'app-features'.\n"
    )


def _file_feature_count_hint(n_files: int) -> str:
    """Generates a feature-count guidance line for the file-path mode prompt."""
    if n_files < 30:
        return ""
    min_f = min(max(5, n_files // 30), 15)
    max_f = min(max(10, n_files // 12), 30)
    return (
        f"\nYou have {n_files} files. "
        f"You MUST produce at least {min_f} features (aim for {min_f}–{max_f}).\n"
        "IMPORTANT: Do NOT create one giant feature for an entire directory. "
        "Each distinct business capability (auth, billing, webhooks, templates, "
        "integrations, search, etc.) MUST be its own feature. "
        "Do NOT merge unrelated capabilities into 'core-platform' or 'app-features'.\n"
    )


def _extract_dir_keywords(
    dirs: list[str],
    all_files: list[str],
    commits: list[Commit],
) -> dict[str, list[str]]:
    """Extracts top commit message keywords per directory from git history."""
    from collections import Counter

    dir_set = set(dirs)
    dir_counters: dict[str, Counter] = {d: Counter() for d in dirs}

    file_to_dir: dict[str, str] = {}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent in dir_set:
            file_to_dir[f] = parent

    word_pattern = re.compile(r"[a-z]{3,}")
    for commit in commits:
        words = {
            w for w in word_pattern.findall(commit.message.lower())
            if w not in _COMMIT_STOP_WORDS
        }
        dirs_touched: set[str] = set()
        for f in commit.files_changed:
            if f in file_to_dir:
                dirs_touched.add(file_to_dir[f])
        for d in dirs_touched:
            dir_counters[d].update(words)

    return {
        d: [w for w, _ in counter.most_common(4)]
        for d, counter in dir_counters.items()
        if counter
    }


def _normalize_commit_files(commits: list[Commit], path_prefix: str) -> list[Commit]:
    """
    Returns commits with path_prefix stripped from each file path.
    Needed when --src is used: commits retain full paths (src/auth/...)
    but analysis_files have the prefix stripped (auth/...).
    """
    result = []
    for c in commits:
        stripped = [
            f[len(path_prefix):] if f.startswith(path_prefix) else f
            for f in c.files_changed
        ]
        result.append(c.model_copy(update={"files_changed": stripped}))
    return result


def _compute_cochange(commits: list[Commit]) -> list[tuple[str, str, float]]:
    """Delegates co-change computation to the features module."""
    from faultline.analyzer.features import compute_cochange
    return compute_cochange(commits)


def _format_extra_context(
    cochange_pairs: list[tuple[str, str, float]],
    dir_keywords: dict[str, list[str]],
) -> str:
    """Builds an extra context block to append to the LLM feature detection prompt."""
    parts: list[str] = []

    if cochange_pairs:
        lines = [
            f"  {f1} ↔ {f2} ({int(s * 100)}%)"
            for f1, f2, s in cochange_pairs[:_MAX_COCHANGE_IN_PROMPT]
        ]
        parts.append(
            "<co-changes>\n"
            "Files changed together frequently — strong signal they belong to the same feature:\n"
            + "\n".join(lines)
            + "\n</co-changes>"
        )

    kw_lines = [
        f"  {d} → {', '.join(sorted(kws))}"
        for d, kws in sorted(dir_keywords.items())
        if kws
    ]
    if kw_lines:
        parts.append(
            "<commit-topics>\n"
            "Top commit message topics per directory:\n"
            + "\n".join(kw_lines)
            + "\n</commit-topics>"
        )

    return ("\n\n" + "\n\n".join(parts) + "\n") if parts else ""


def _format_route_anchors(
    signatures: dict[str, FileSignature],
    dirs: list[str] | None = None,
) -> str:
    """
    Builds a <route-anchors> section for the LLM prompt.

    File mode (dirs=None): one line per file that has routes.
    Dir mode (dirs provided): one line per directory, routes aggregated from direct children.

    Returns empty string if no routes found in signatures.
    """
    if not signatures:
        return ""

    if dirs is None:
        lines = []
        for path, sig in sorted(signatures.items()):
            if not sig.routes:
                continue
            routes_str = ", ".join(sorted(sig.routes)[:_MAX_ROUTES_PER_ENTRY])
            lines.append(f"  {path} → {routes_str}")
            if len(lines) >= _MAX_ROUTE_ANCHOR_FILES:
                break

        if not lines:
            return ""

        return (
            "\n\n<route-anchors>\n"
            "Files with API routes — use as starting anchors for feature grouping:\n"
            + "\n".join(lines)
            + "\n</route-anchors>"
        )
    else:
        dirs_set = set(dirs)
        dir_routes: dict[str, list[str]] = {}
        for path, sig in sorted(signatures.items()):
            if not sig.routes:
                continue
            parent = str(Path(path).parent)
            if parent in dirs_set:
                dir_routes.setdefault(parent, []).extend(sig.routes)

        if not dir_routes:
            return ""

        lines = []
        for d in dirs:
            if d not in dir_routes:
                continue
            routes_str = ", ".join(dir_routes[d][:_MAX_ROUTES_PER_ENTRY])
            lines.append(f"  {d} → {routes_str}")
            if len(lines) >= _MAX_ROUTE_ANCHOR_FILES:
                break

        if not lines:
            return ""

        return (
            "\n\n<route-anchors>\n"
            "Directories with API routes — strong feature boundary anchors:\n"
            + "\n".join(lines)
            + "\n</route-anchors>"
        )


# ── Package anchors (Python subdirectories with __init__.py) ────────────────

_MAX_PACKAGE_ANCHORS = 20


def _format_package_anchors(
    files: list[str],
    signatures: dict[str, FileSignature] | None = None,
) -> str:
    """Build a <package-anchors> section listing Python sub-packages.

    Each subdirectory with __init__.py is a distinct Python package and is a
    strong signal for a separate feature. Shows key exports from the package's
    files to help LLM understand what the package does.
    """
    # Find directories that contain __init__.py
    packages: dict[str, list[str]] = {}
    for f in files:
        p = Path(f)
        if p.name == "__init__.py" and str(p.parent) != ".":
            packages[str(p.parent)] = []

    if not packages:
        return ""

    # Collect key exports per package
    if signatures:
        for path, sig in signatures.items():
            if not sig.exports:
                continue
            parent = str(Path(path).parent)
            if parent in packages:
                packages[parent].extend(sig.exports[:6])

    # Also count files per package
    pkg_file_counts: dict[str, int] = {pkg: 0 for pkg in packages}
    for f in files:
        parent = str(Path(f).parent)
        if parent in pkg_file_counts:
            pkg_file_counts[parent] += 1

    lines: list[str] = []
    for pkg in sorted(packages.keys()):
        exports = list(dict.fromkeys(packages[pkg]))[:8]
        count = pkg_file_counts.get(pkg, 0)
        exports_str = f" → {', '.join(exports)}" if exports else ""
        lines.append(f"  {pkg}/ ({count} files){exports_str}")
        if len(lines) >= _MAX_PACKAGE_ANCHORS:
            break

    if not lines:
        return ""

    return (
        "\n\n<package-anchors>\n"
        "Python packages (directories with __init__.py) — each is a distinct module "
        "and should be treated as a SEPARATE feature or strong feature boundary. "
        "Do NOT merge these into a generic 'core-utilities' or 'shared' feature:\n"
        + "\n".join(lines)
        + "\n</package-anchors>"
    )


# ── Entity anchors ──────────────────────────────────────────────────────────

_MAX_ENTITY_ANCHOR_FILES = 30
_MAX_ENTITIES_PER_FILE = 12

# File patterns that typically define business entities/models
_ENTITY_FILE_PATTERNS = {
    "models.py", "model.py", "schemas.py", "schema.py", "types.py",
    "entities.py", "entity.py", "forms.py", "serializers.py",
    "admin.py", "views.py", "urls.py", "routes.py", "handlers.py",
}
# Directory name patterns that suggest entity-defining files
_ENTITY_DIR_PATTERNS = {"models", "schemas", "entities", "types"}


def _format_entity_anchors(
    signatures: dict[str, FileSignature],
    dirs: list[str] | None = None,
) -> str:
    """Build an <entity-anchors> section listing class/export names from key files.

    This helps the LLM detect features that exist as classes inside shared files
    (e.g. Django models.py with Tag, Correspondent, SavedView classes).

    File mode (dirs=None): shows exports from model/schema/entity files.
    Dir mode (dirs provided): aggregates exports by directory.
    """
    if not signatures:
        return ""

    # Filter to Python entity-defining files only.
    # TS/JS models/types/schemas are technical layers, not business entities —
    # including them causes over-merging in non-Python repos.
    entity_sigs: list[FileSignature] = []
    for path, sig in signatures.items():
        if not sig.exports:
            continue
        if not path.endswith(".py"):
            continue
        filename = Path(path).name.lower()
        parent_name = Path(path).parent.name.lower()
        if filename in _ENTITY_FILE_PATTERNS or parent_name in _ENTITY_DIR_PATTERNS:
            entity_sigs.append(sig)

    if not entity_sigs:
        return ""

    if dirs is None:
        # File mode: one line per entity-defining file
        lines: list[str] = []
        for sig in sorted(entity_sigs, key=lambda s: s.path)[
            :_MAX_ENTITY_ANCHOR_FILES
        ]:
            exports_str = ", ".join(sig.exports[:_MAX_ENTITIES_PER_FILE])
            more = (
                f" (+{len(sig.exports) - _MAX_ENTITIES_PER_FILE} more)"
                if len(sig.exports) > _MAX_ENTITIES_PER_FILE
                else ""
            )
            lines.append(f"  {sig.path} → {exports_str}{more}")

        if not lines:
            return ""

        return (
            "\n\n<entity-anchors>\n"
            "Business entity definitions found in model/schema files — "
            "each entity name often maps to a distinct feature or sub-feature:\n"
            + "\n".join(lines)
            + "\n</entity-anchors>"
        )

    # Dir mode: aggregate exports by directory
    dir_set = set(dirs)
    dir_entities: dict[str, list[str]] = {}
    for sig in entity_sigs:
        parent = str(Path(sig.path).parent)
        if parent in dir_set:
            dir_entities.setdefault(parent, []).extend(
                sig.exports[:_MAX_ENTITIES_PER_FILE]
            )

    if not dir_entities:
        return ""

    lines = []
    for d in dirs:
        if d not in dir_entities:
            continue
        # Deduplicate and limit
        entities = list(dict.fromkeys(dir_entities[d]))[
            :_MAX_ENTITIES_PER_FILE
        ]
        entities_str = ", ".join(entities)
        lines.append(f"  {d} → {entities_str}")
        if len(lines) >= _MAX_ENTITY_ANCHOR_FILES:
            break

    if not lines:
        return ""

    return (
        "\n\n<entity-anchors>\n"
        "Business entity definitions (models/schemas) found in these directories — "
        "entity names reveal distinct business domains within a single directory:\n"
        + "\n".join(lines)
        + "\n</entity-anchors>"
    )


def _expand_dir_mapping(
    response: _FeatureDetectionResponse,
    all_files: list[str],
) -> dict[str, list[str]]:
    """
    Expands a directory-level feature mapping to file-level.
    LLM returned directories → we assign all files under those dirs to the feature.
    """
    dir_to_files: dict[str, list[str]] = {}
    for f in all_files:
        parts = Path(f).parts
        for i in range(1, len(parts)):
            d = str(Path(*parts[:i]))
            dir_to_files.setdefault(d, []).append(f)

    result: dict[str, list[str]] = {}
    assigned: set[str] = set()

    for mapping in response.features:
        feature_files: list[str] = []
        for d in mapping.files:
            d_clean = d.rstrip("/").strip()  # normalize trailing slashes from LLM
            for f in dir_to_files.get(d_clean, []):
                if f not in assigned:
                    feature_files.append(f)
                    assigned.add(f)
        if feature_files:
            result[mapping.feature_name] = feature_files

    return result


def validate_ollama(
    model: str = _DEFAULT_OLLAMA_MODEL,
    host: str = _DEFAULT_OLLAMA_HOST,
) -> tuple[bool, str]:
    """
    Checks if Ollama is reachable and the requested model is available.
    Returns (is_valid, error_message).
    """
    try:
        import ollama as _ollama
    except ImportError:
        return False, (
            "ollama package not installed. Run: pip install 'faultline[ollama]' "
            "or: pip install ollama"
        )

    try:
        client = _ollama.Client(host=host)
        available = [m.model for m in client.list().models]
        model_base = model.split(":")[0]
        if not any(m.startswith(model_base) for m in available):
            available_str = ", ".join(available) if available else "none pulled yet"
            return False, (
                f"Model '{model}' not found in Ollama. "
                f"Available: {available_str}. "
                f"Run: ollama pull {model}"
            )
        return True, ""
    except Exception:
        return False, (
            f"Cannot connect to Ollama at {host}. "
            "Make sure Ollama is running: ollama serve"
        )


def detect_features_ollama(
    files: list[str],
    model: str = _DEFAULT_OLLAMA_MODEL,
    host: str = _DEFAULT_OLLAMA_HOST,
    commits: list[Commit] | None = None,
    path_prefix: str = "",
    signatures: dict[str, FileSignature] | None = None,
    layer_context: str = "",
) -> dict[str, list[str]]:
    """
    Sends the repository file tree to a local Ollama model and returns a semantic feature mapping.
    Returns {} on any error (caller falls back to heuristic detection).

    Args:
        files: List of file paths (relative, with path_prefix already stripped).
        model: Ollama model name (e.g. 'qwen2.5-coder:7b', 'llama3.2').
        host: Ollama server URL.
        commits: Optional commit history for co-change enrichment.
        path_prefix: Prefix stripped from files (e.g. "src/"). Used to normalize
            commit paths so they match the stripped file paths.

    Returns:
        dict mapping feature names to lists of file paths.
        Empty dict if the call fails.
    """
    if not files:
        return {}

    norm_commits = _normalize_commit_files(commits, path_prefix) if commits and path_prefix else commits
    cochange_pairs = _compute_cochange(norm_commits) if norm_commits else []

    if len(files) > _DIR_COLLAPSE_THRESHOLD:
        dirs = _unique_dirs(files)
        samples = _dir_to_sample_files(dirs, files)
        dir_keywords = _extract_dir_keywords(dirs, files, norm_commits) if norm_commits else {}
        file_tree = _format_dir_tree(dirs, samples)
        route_anchors = _format_route_anchors(signatures, dirs=dirs) if signatures else ""
        entity_anchors = _format_entity_anchors(signatures, dirs=dirs) if signatures else ""
        extra_context = _format_extra_context(cochange_pairs, dir_keywords) + route_anchors + entity_anchors
        response = _call_dir_detection_ollama(file_tree, model, host, n_dirs=len(dirs), extra_context=extra_context, layer_context=layer_context)
        if not response:
            return {}
        return _expand_dir_mapping(response, files)
    else:
        file_tree = "\n".join(files[:_MAX_FILES_FOR_DETECTION])
        route_anchors = _format_route_anchors(signatures) if signatures else ""
        entity_anchors = _format_entity_anchors(signatures) if signatures else ""
        package_anchors = _format_package_anchors(files, signatures)
        extra_context = _format_extra_context(cochange_pairs, {}) + route_anchors + entity_anchors + package_anchors
        response = _call_feature_detection_ollama(file_tree, model, host, extra_context, n_files=len(files), layer_context=layer_context)
        if not response:
            return {}
        return _build_feature_dict(response, set(files))


def _call_feature_detection_ollama(
    file_tree: str,
    model: str,
    host: str,
    extra_context: str = "",
    n_files: int = 0,
    layer_context: str = "",
) -> _FeatureDetectionResponse | None:
    """Calls Ollama API for feature detection (file-path mode). Returns None on any failure."""
    try:
        import ollama as _ollama
    except ImportError:
        return None

    hint = _file_feature_count_hint(n_files) if n_files else ""
    prompt = _DETECTION_USER_PROMPT.format(
        file_tree=file_tree, extra_context=extra_context, feature_hint=hint,
    )

    system = _DETECTION_SYSTEM_PROMPT + layer_context
    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            format=_FeatureDetectionResponse.model_json_schema(),
        )
        return _FeatureDetectionResponse.model_validate_json(response.message.content)
    except (ValidationError, Exception):
        return None


def _call_dir_detection_ollama(
    file_tree: str,
    model: str,
    host: str,
    n_dirs: int,
    extra_context: str = "",
    layer_context: str = "",
) -> _FeatureDetectionResponse | None:
    """
    Calls Ollama API for dir-collapse feature detection.
    Uses dir-specific prompts so the model returns directory paths, not filenames.
    Returns None on any failure.
    """
    try:
        import ollama as _ollama
    except ImportError:
        return None

    prompt = _DIR_DETECTION_USER_PROMPT.format(
        file_tree=file_tree,
        feature_hint=_feature_count_hint(n_dirs),
        extra_context=extra_context,
    )

    system = _DIR_DETECTION_SYSTEM_PROMPT + layer_context
    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            format=_FeatureDetectionResponse.model_json_schema(),
        )
        return _FeatureDetectionResponse.model_validate_json(response.message.content)
    except (ValidationError, Exception):
        return None


# ── DeepSeek provider ─────────────────────────────────────────────────────────

_DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def detect_features_deepseek(
    files: list[str],
    api_key: str | None = None,
    model: str = _DEFAULT_DEEPSEEK_MODEL,
    base_url: str | None = None,
    commits: list[Commit] | None = None,
    path_prefix: str = "",
    signatures: dict[str, FileSignature] | None = None,
    layer_context: str = "",
) -> dict[str, list[str]]:
    """Sends file tree to DeepSeek API for feature detection. Returns {} on failure."""
    from faultline.llm.deepseek_client import call_deepseek_parsed

    if not files:
        return {}

    norm_commits = _normalize_commit_files(commits, path_prefix) if commits and path_prefix else commits
    cochange_pairs = _compute_cochange(norm_commits) if norm_commits else []

    if len(files) > _DIR_COLLAPSE_THRESHOLD:
        dirs = _unique_dirs(files)
        samples = _dir_to_sample_files(dirs, files)
        dir_keywords = _extract_dir_keywords(dirs, files, norm_commits) if norm_commits else {}
        file_tree = _format_dir_tree(dirs, samples)
        route_anchors = _format_route_anchors(signatures, dirs=dirs) if signatures else ""
        entity_anchors = _format_entity_anchors(signatures, dirs=dirs) if signatures else ""
        extra_context = _format_extra_context(cochange_pairs, dir_keywords) + route_anchors + entity_anchors

        system = _DIR_DETECTION_SYSTEM_PROMPT + layer_context
        min_f = min(max(8, len(dirs) // 15), 15)
        system += (
            f"\n\n## CRITICAL REQUIREMENT\n"
            f"This codebase has {len(dirs)} directories. You MUST return at least {min_f} features."
        )
        prompt = _DIR_DETECTION_USER_PROMPT.format(
            file_tree=file_tree,
            feature_hint=_feature_count_hint(len(dirs)),
            extra_context=extra_context,
        )

        response = call_deepseek_parsed(
            system, prompt, _FeatureDetectionResponse,
            api_key=api_key, model=model, base_url=base_url, max_tokens=_MAX_TOKENS_DIR,
        )
        if not response:
            return {}
        result = _expand_dir_mapping(response, files)
    else:
        file_tree = "\n".join(files[:_MAX_FILES_FOR_DETECTION])
        route_anchors = _format_route_anchors(signatures) if signatures else ""
        entity_anchors = _format_entity_anchors(signatures) if signatures else ""
        package_anchors = _format_package_anchors(files, signatures)
        extra_context = _format_extra_context(cochange_pairs, {}) + route_anchors + entity_anchors + package_anchors

        system = _DETECTION_SYSTEM_PROMPT + layer_context
        hint = _file_feature_count_hint(len(files)) if files else ""
        prompt = _DETECTION_USER_PROMPT.format(
            file_tree=file_tree, extra_context=extra_context, feature_hint=hint,
        )

        response = call_deepseek_parsed(
            system, prompt, _FeatureDetectionResponse,
            api_key=api_key, model=model, base_url=base_url, max_tokens=_MAX_TOKENS_FILE,
        )
        if not response:
            return {}
        result = _build_feature_dict(response, set(files))

    result = _collapse_plugin_features(result)
    result = _extract_shared_ui(result)
    result = _redistribute_oversized_features(result)
    result = _redistribute_infra_features(result)
    return _final_cleanup(result)


def merge_and_name_clusters_deepseek(
    cluster_mapping: dict[str, list[str]],
    api_key: str | None = None,
    model: str = _DEFAULT_DEEPSEEK_MODEL,
    base_url: str | None = None,
    commits: list[Commit] | None = None,
    layer_context: str = "",
    cluster_edges: dict[str, dict[str, int]] | None = None,
) -> dict[str, list[str]]:
    """Uses DeepSeek to merge import-graph clusters into business features."""
    from faultline.llm.deepseek_client import call_deepseek_parsed

    if not cluster_mapping:
        return cluster_mapping

    working_clusters = _consolidate_domain_clusters(cluster_mapping)
    working_clusters = _pre_merge_tiny_clusters(working_clusters)

    cache_key = _merge_cache_key(working_clusters, model)
    cached = _read_name_cache(cache_key)
    if cached is not None:
        if isinstance(next(iter(cached.values()), None), list):
            return _final_cleanup(_redistribute_oversized_features(
                _redistribute_infra_features(
                    _extract_shared_ui(_collapse_plugin_features(cached))
                )
            ))

    keywords_per_cluster = _extract_cluster_keywords(working_clusters, commits) if commits else None

    prompt = _MERGE_USER_PROMPT.format(
        clusters=_format_clusters_for_merge_prompt(working_clusters, keywords_per_cluster, cluster_edges=cluster_edges),
        feature_hint=_merge_feature_count_hint(len(working_clusters)),
    )
    system = _MERGE_SYSTEM_PROMPT + layer_context

    merge_response = call_deepseek_parsed(
        system, prompt, _ClusterMergeResponse,
        api_key=api_key, model=model, base_url=base_url, max_tokens=_MAX_TOKENS_FILE,
    )

    if merge_response:
        merged = _filter_technical_features(
            _apply_cluster_merge(working_clusters, merge_response),
        )
        merged = _validate_merge_cohesion(merged, working_clusters, merge_response)
        _write_name_cache(cache_key, merged)
        merged = _collapse_plugin_features(merged)
        merged = _extract_shared_ui(merged)
        merged = _redistribute_infra_features(merged)
        merged = _redistribute_oversized_features(merged)
        return _final_cleanup(merged)

    return _final_cleanup(working_clusters)


# ── Cluster merge + name (import graph → semantic features) ──────────────────
# Used when import-graph clusters need to be MERGED across business boundaries
# and given semantic names in a single LLM call.
# Unlike naming-only, LLM can merge N clusters → M features (M ≤ N).
# Results are cached: same cluster structure → same output on every run.

_MERGE_SYSTEM_PROMPT = """\
You are a software architect merging import-dependency clusters into business features.

## Context
Files have been pre-grouped into clusters by analyzing import statements.
Files inside a cluster directly import each other.
Your job is to identify which clusters serve the same business feature and merge them,
even when they have no direct import relationship (e.g. a Redux slice and the component
that uses it via a hook; an API service and the page that renders its data).

## Task
1. Merge clusters that serve the same user-facing capability.
2. Give each merged group a semantic business domain name.

## Rules
- Feature names: lowercase, hyphen-separated, 1–3 words.
- Use business domain terminology, not technical layers.
  Examples: "user-auth", "checkout", "analytics-dashboard", "notifications", "team-management".
- Merge clusters that serve the same business domain into one feature.
  BAD: merge all hooks clusters into "hooks-feature".
  GOOD: merge auth-hooks + auth-components + auth-api into "user-auth".
- Keep features separate when they represent distinct business capabilities \
  (e.g. "payments" vs "auth" vs "labels" vs "notifications").
- IMPORTANT: small clusters (2-5 files) that represent a clear business domain \
  must stay as their own feature. Do NOT absorb them into larger features. \
  Example: a "system-status" page with 3 files is a real feature, not part of "dashboard".
- When in doubt, keep features separate rather than merging them.
- NEVER create features for technical layers or utilities. Clusters containing only \
  shared utilities, hooks, helpers, icons, assets, UI components, stories (Storybook), \
  theme/locale files, or general-purpose code must be merged into the business feature \
  that uses them. If no clear business owner exists, merge them into the largest \
  feature that imports them.
  BAD: "hooks-utils", "shared-components", "icons-assets", "stories", "general-utils".
  GOOD: merge these into the business features they support.
- Every cluster must appear in exactly one feature — no omissions.
- cluster_indices contains 1-based indices from the list provided.

## Import connections — USE THIS DATA
Each cluster may show "Imports from: cluster N (count)" lines. This tells you how \
many file-level import statements connect the two clusters.
- High connection count (5+) between two clusters = strong signal they serve the same \
  business feature → MERGE them.
- Zero or low connections = clusters are independent → keep them as SEPARATE features.
- Do NOT merge clusters with zero import connections just because their names look similar. \
  Imports are the ground truth for code relationships.

## Shared / utility clusters — CRITICAL
- Clusters containing hooks, utils, helpers, or lib files that have DOMAIN-SPECIFIC names \
  (useEntitySearch, useIPEnrichment, useOrganizationMembers, exportCSV) must be merged into \
  the business feature they serve, NOT into "shared-hooks" or "shared-utils".
- Only truly generic, domain-agnostic clusters (useHover, useDebounce, formatDate, Button) \
  belong in "shared-*" features.

## Size limits — CRITICAL
- No feature may contain more than 40 files. If merging clusters would create a feature \
  with >40 files, split it into distinct sub-features by business capability instead.
- NEVER create a catch-all feature like "backend-api", "api-routes", "core", \
  "shared-backend", or "platform". Every API route cluster belongs to the business \
  feature it serves.

## Page ↔ API route matching — CRITICAL
In Next.js / app-router projects, merge clusters by shared path segment:
- Cluster with `app/organizations/` files + cluster with `app/api/organizations/` files \
  → SAME feature "organizations" (not separate features).
- Cluster with `app/dashboard/` + cluster with `app/api/dashboard/` → SAME feature.
- Cluster with `hooks/use-organizations.ts` → merge into "organizations" feature.
- Cluster with `components/org-*.tsx` → merge into "organizations" feature.
- API route clusters MUST join the page feature they serve, never a separate "api" feature.

BAD — API routes separated from their pages:
  "organizations": [clusters with app/organizations/* files]
  "backend-api": [clusters with app/api/organizations/*, app/api/cost/*, app/api/health/*]

GOOD — API routes merged with their pages:
  "organizations": [clusters with app/organizations/* AND app/api/organizations/* files]
  "cost-analysis": [clusters with app/cost/* AND app/api/cost/* files]
  "health-monitoring": [clusters with app/health/* AND app/api/health/* files]\
"""

_MERGE_USER_PROMPT = """\
Below are code clusters formed from import dependency analysis.
Group related clusters into business features and name each feature.
{feature_hint}
{clusters}

For each feature: provide a feature_name and the list of cluster_indices (1-based) it contains.
Every cluster index must appear in exactly one feature.\
"""


def _merge_feature_count_hint(n_clusters: int) -> str:
    """Generates a scoped feature-count guidance line for the merge prompt.

    Scale: ~1 feature per 12 clusters, clamped to 6–40.
    Small repo (20 clusters) → 6–10 features.
    Medium repo (100 clusters) → 8–16 features.
    Large repo (300 clusters) → 20–40 features.
    """
    min_f = max(6, n_clusters // 16)
    max_f = max(10, n_clusters // 8)
    max_f = min(max_f, 40)
    return (
        f"\nYou have {n_clusters} clusters. "
        f"Aim for {min_f}–{max_f} business features. "
        f"Merge clusters that clearly serve the same domain, "
        f"but keep distinct business capabilities as separate features.\n"
    )


class _ClusterMergeItem(BaseModel):
    feature_name: str
    cluster_indices: list[int]


class _ClusterMergeResponse(BaseModel):
    features: list[_ClusterMergeItem]


def _merge_cache_key(cluster_mapping: dict[str, list[str]], model: str) -> str:
    """Stable cache key that includes both file membership and cluster structure."""
    # Sort clusters by their sorted file list for a stable canonical form
    clusters_repr = sorted([sorted(files) for files in cluster_mapping.values()])
    content = json.dumps(clusters_repr) + model
    return "merge_" + hashlib.sha256(content.encode()).hexdigest()[:24]


def _extract_cluster_keywords(
    cluster_mapping: dict[str, list[str]],
    commits: list[Commit],
) -> dict[str, list[str]]:
    """Extracts top commit message keywords per cluster from git history.

    For each cluster, collects commit messages of commits that touched any file
    in that cluster, then returns the top 4 non-stop words. These keywords give
    the LLM semantic hints about the business domain (e.g. "payment", "checkout",
    "billing") even when file paths alone are ambiguous.

    Bulk commits (>30 files) are excluded — they're refactors, not feature signals.
    """
    from collections import Counter

    _MAX_FILES_BULK = 30

    file_to_cluster: dict[str, str] = {
        f: cluster_id
        for cluster_id, files in cluster_mapping.items()
        for f in files
    }

    cluster_counters: dict[str, Counter] = {c: Counter() for c in cluster_mapping}
    word_pattern = re.compile(r"[a-z]{3,}")

    for commit in commits:
        if len(commit.files_changed) > _MAX_FILES_BULK:
            continue
        words = {
            w for w in word_pattern.findall(commit.message.lower())
            if w not in _COMMIT_STOP_WORDS
        }
        clusters_touched: set[str] = set()
        for f in commit.files_changed:
            if f in file_to_cluster:
                clusters_touched.add(file_to_cluster[f])
        for cluster_id in clusters_touched:
            cluster_counters[cluster_id].update(words)

    return {
        cluster_id: [w for w, _ in counter.most_common(4)]
        for cluster_id, counter in cluster_counters.items()
        if counter
    }


def _format_clusters_for_merge_prompt(
    cluster_mapping: dict[str, list[str]],
    keywords_per_cluster: dict[str, list[str]] | None = None,
    cluster_edges: dict[str, dict[str, int]] | None = None,
) -> str:
    """Formats clusters as a numbered list for the LLM merge prompt.

    When keywords_per_cluster is provided, each cluster entry includes its
    top commit message topics as a hint for semantic business domain naming.
    When cluster_edges is provided, shows import connections to other clusters.
    """
    # Build cluster_id → index mapping for edge display
    cluster_ids = list(cluster_mapping.keys())
    id_to_idx = {cid: i + 1 for i, cid in enumerate(cluster_ids)}

    lines = []
    for i, (cluster_id, files) in enumerate(cluster_mapping.items(), start=1):
        sample = files[:8]
        file_lines = "\n".join(f"  {f}" for f in sample)
        suffix = f"\n  … ({len(files) - 8} more)" if len(files) > 8 else ""
        keywords = (keywords_per_cluster or {}).get(cluster_id, [])
        kw_line = f"\n  Commit topics: {', '.join(keywords)}" if keywords else ""

        # Format import connections to other clusters
        edge_line = ""
        if cluster_edges and cluster_id in cluster_edges:
            connections = cluster_edges[cluster_id]
            # Sort by connection count, show top 5
            top = sorted(connections.items(), key=lambda x: -x[1])[:5]
            if top:
                parts = []
                for target_id, count in top:
                    if target_id in id_to_idx:
                        parts.append(f"cluster {id_to_idx[target_id]} ({count})")
                if parts:
                    edge_line = f"\n  Imports from: {', '.join(parts)}"

        lines.append(f"Cluster {i} ({cluster_id}):{kw_line}{edge_line}\n{file_lines}{suffix}")
    return "\n\n".join(lines)


_GENERIC_DIR_NAMES = {
    "src", "lib", "app", "core", "internal", "views", "pages",
    "components", "shared", "common", "utils", "features",
    "hooks", "services", "modules",
}


def _derive_feature_name(files: list[str]) -> str:
    """Derives a descriptive feature name from a list of file paths.

    Uses the most common package/directory as the name. Handles monorepo
    patterns (packages/react-dom/src/...) and flat structures.
    """
    from collections import Counter
    counts: Counter = Counter()

    for f in files:
        parts = Path(f).parts
        # For monorepo: packages/<name>/... → use <name>
        if len(parts) >= 2 and parts[0] in ("packages", "apps", "modules", "services"):
            counts[parts[1]] += 1
        # For flat: find first non-generic directory
        elif len(parts) >= 2:
            for part in parts[:-1]:
                if part.lower() not in _GENERIC_DIR_NAMES:
                    counts[part] += 1
                    break

    if counts:
        name = counts.most_common(1)[0][0]
        return name.lower().replace("_", "-")
    return "overflow"


# Maximum files per feature after merge — prevents "junk drawer" features
# where LLM over-merges all related clusters into one giant group.
_MAX_FEATURE_SIZE_AFTER_MERGE = 150


def _apply_cluster_merge(
    cluster_mapping: dict[str, list[str]],
    merge_response: _ClusterMergeResponse,
) -> dict[str, list[str]]:
    """Builds the merged feature mapping from the LLM response.

    Handles duplicate names, out-of-range indices, and unassigned clusters
    (any cluster not referenced falls back to its original directory-derived name).
    Enforces a hard cap: if merging would create a feature > _MAX_FEATURE_SIZE_AFTER_MERGE
    files, excess clusters are kept as separate features.
    """
    cluster_ids = list(cluster_mapping.keys())
    assigned: set[int] = set()
    result: dict[str, list[str]] = {}
    used_names: set[str] = set()

    for item in merge_response.features:
        merged_files: list[str] = []
        overflow_clusters: list[tuple[str, list[str]]] = []

        for idx in item.cluster_indices:
            if 1 <= idx <= len(cluster_ids) and idx not in assigned:
                cluster_id = cluster_ids[idx - 1]
                cluster_files = cluster_mapping[cluster_id]

                if len(merged_files) + len(cluster_files) <= _MAX_FEATURE_SIZE_AFTER_MERGE:
                    merged_files.extend(cluster_files)
                    assigned.add(idx)
                else:
                    # Would exceed cap — keep this cluster separate
                    overflow_clusters.append((cluster_id, cluster_files))
                    assigned.add(idx)

        if not merged_files:
            continue

        name = item.feature_name
        if name in used_names:
            suffix = 2
            while f"{name}-{suffix}" in used_names:
                suffix += 1
            name = f"{name}-{suffix}"
        used_names.add(name)
        result[name] = sorted(merged_files)

        # Overflow clusters become their own features with descriptive names
        for cluster_id, cluster_files in overflow_clusters:
            overflow_name = _derive_feature_name(cluster_files)
            if overflow_name in used_names:
                suffix = 2
                while f"{overflow_name}-{suffix}" in used_names:
                    suffix += 1
                overflow_name = f"{overflow_name}-{suffix}"
            used_names.add(overflow_name)
            result[overflow_name] = sorted(cluster_files)
            logger.info(
                "Cluster '%s' (%d files) overflowed feature '%s' cap (%d) — kept as separate feature '%s'",
                cluster_id, len(cluster_files), name, _MAX_FEATURE_SIZE_AFTER_MERGE, overflow_name,
            )

    # Unassigned clusters: merge into nearest feature by directory overlap.
    # Clusters with a distinct business directory stay standalone.
    for i, cluster_id in enumerate(cluster_ids, start=1):
        if i not in assigned:
            orphan_files = cluster_mapping[cluster_id]
            # Try to merge into an existing feature with directory overlap
            target = _find_best_merge_target(orphan_files, result)
            if target:
                result[target].extend(orphan_files)
            elif _cluster_has_distinct_dir(orphan_files):
                # Distinct business dir — keep as standalone
                name = cluster_id
                if name in used_names:
                    suffix = 2
                    while f"{name}-{suffix}" in used_names:
                        suffix += 1
                    name = f"{name}-{suffix}"
                used_names.add(name)
                result[name] = orphan_files
            else:
                # No overlap, no distinct dir — merge into largest feature
                largest = max(result, key=lambda k: len(result[k])) if result else None
                if largest:
                    result[largest].extend(orphan_files)
                else:
                    result[cluster_id] = orphan_files

    return result


def _find_best_merge_target(
    orphan_files: list[str],
    existing_features: dict[str, list[str]],
) -> str | None:
    """Finds the existing feature with the most directory overlap to absorb orphan files."""
    if not existing_features:
        return None

    orphan_dirs = {str(Path(f).parent) for f in orphan_files}

    best_name, best_overlap = None, 0
    for feat_name, feat_files in existing_features.items():
        feat_dirs = {str(Path(f).parent) for f in feat_files}
        overlap = len(orphan_dirs & feat_dirs)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = feat_name

    # Fallback: match by top-level directory
    if not best_name:
        orphan_tops = {Path(f).parts[0] if len(Path(f).parts) > 1 else "" for f in orphan_files}
        for feat_name, feat_files in existing_features.items():
            feat_tops = {Path(f).parts[0] if len(Path(f).parts) > 1 else "" for f in feat_files}
            if orphan_tops & feat_tops:
                if len(feat_files) > best_overlap:
                    best_overlap = len(feat_files)
                    best_name = feat_name

    return best_name


def _call_cluster_merge(
    client: anthropic.Anthropic,
    cluster_mapping: dict[str, list[str]],
    keywords_per_cluster: dict[str, list[str]] | None = None,
    layer_context: str = "",
    cluster_edges: dict[str, dict[str, int]] | None = None,
) -> _ClusterMergeResponse | None:
    """Sends all clusters to Claude for merge+name. Returns None on any failure."""
    prompt = _MERGE_USER_PROMPT.format(
        clusters=_format_clusters_for_merge_prompt(cluster_mapping, keywords_per_cluster, cluster_edges),
        feature_hint=_merge_feature_count_hint(len(cluster_mapping)),
    )
    system = _MERGE_SYSTEM_PROMPT + layer_context
    n_clusters = len(cluster_mapping)
    logger.info("Cluster merge: %d clusters, prompt length ~%d chars", n_clusters, len(prompt))
    for attempt in range(_MAX_RETRIES):
        try:
            max_tokens = 2048 if n_clusters < 50 else 4096 if n_clusters < 150 else 8192
            response = client.messages.parse(
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=_ClusterMergeResponse,
            )
            logger.info("Cluster merge success: %d features", len(response.parsed_output.features))
            return response.parsed_output
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("LLM cluster merge failed (attempt %d/%d): %s. Retrying in %.1fs...", attempt + 1, _MAX_RETRIES, e, delay)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
        except ValidationError as e:
            logger.warning("LLM cluster merge ValidationError (clusters=%d): %s", n_clusters, e)
            return None
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
        ) as e:
            logger.warning("LLM cluster merge auth error: %s", e)
            return None
        except anthropic.APIStatusError as e:
            logger.warning("LLM cluster merge APIStatusError: %s", e)
            return None
    return None


# ── Domain-keyword consolidation ─────────────────────────────────────────────
# Clusters whose files share a distinctive business keyword (e.g. "labels",
# "payments") are merged before LLM sees them.  This fixes the common case
# where one feature's files are split across many import-graph clusters because
# they don't import each other directly (hooks, views, services, schemas).

_DOMAIN_TECH_WORDS = {
    "components", "component", "hooks", "hook", "utils", "util", "helpers",
    "helper", "views", "view", "pages", "page", "features", "feature",
    "shared", "common", "lib", "libs", "src", "app", "modules", "module",
    "services", "service", "schemas", "schema", "types", "models", "model",
    "queries", "query", "mutations", "actions", "reducers", "slices",
    "middleware", "guards", "interceptors", "pipes", "decorators",
    "providers", "context", "contexts", "stores", "store", "state",
    "api", "rest", "graphql", "grpc",
    "ui", "assets", "icons", "images", "fonts", "styles", "theme", "themes",
    "stories", "storybook", "tests", "test", "spec", "specs", "mocks",
    "fixtures", "config", "configs", "constants", "enums",
    "index", "main", "root", "base", "core", "internal",
    "ndr", "hunterx", "easm", "edr",  # product-specific prefixes (too broad)
}

# Minimum fraction of a cluster's files that must contain a keyword
# for it to count as a "signature keyword" for that cluster.
_DOMAIN_KEYWORD_MIN_RATIO = 0.3
# Minimum number of clusters sharing a keyword to trigger consolidation.
_DOMAIN_KEYWORD_MIN_CLUSTERS = 2


def _extract_domain_keywords(files: list[str]) -> set[str]:
    """Extracts distinctive business-domain keywords from file paths.

    Looks at directory names and file stems, filtering out technical terms.
    Returns lowercase keywords that appear in at least 30% of the files.
    """
    from collections import Counter

    keyword_counts: Counter[str] = Counter()
    for f in files:
        seen: set[str] = set()
        parts = list(Path(f).parts)
        # Include directory names + file stem (without extension)
        stem = Path(f).stem.lower()
        for part in parts[:-1]:
            token = part.lower()
            if token not in seen and len(token) > 2:
                seen.add(token)
        # Also check camelCase/PascalCase splitting of stem
        import re
        tokens = re.findall(r"[a-z]+", re.sub(r"([A-Z])", r" \1", stem).lower())
        for t in tokens:
            if len(t) > 2 and t not in seen:
                seen.add(t)
        for s in seen:
            keyword_counts[s] += 1

    threshold = max(1, int(len(files) * _DOMAIN_KEYWORD_MIN_RATIO))
    return {
        kw for kw, count in keyword_counts.items()
        if count >= threshold and kw not in _DOMAIN_TECH_WORDS
    }


def _consolidate_domain_clusters(
    cluster_mapping: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merges clusters that share distinctive business-domain keywords.

    Example: clusters containing files in `*/labels/*`, `*/Labels/*`,
    `*LabelManager*` all share the keyword "label" and get merged into one
    cluster before LLM merge sees them.
    """
    from collections import Counter, defaultdict as _defaultdict

    if len(cluster_mapping) <= 20:
        return cluster_mapping

    # Step 1: Extract signature keywords per cluster
    cluster_keywords: dict[str, set[str]] = {}
    keyword_to_clusters: dict[str, list[str]] = _defaultdict(list)

    for cname, files in cluster_mapping.items():
        kws = _extract_domain_keywords(files)
        cluster_keywords[cname] = kws
        for kw in kws:
            keyword_to_clusters[kw].append(cname)

    # Step 2: Find keywords shared by multiple clusters (domain signals)
    # Only use keywords that aren't too common (shared by <15% of clusters)
    # to avoid creating mega-clusters through transitive chains.
    max_cluster_share = max(3, len(cluster_mapping) // 7)
    merge_groups: dict[str, list[str]] = {}  # keyword → list of cluster names
    for kw, cnames in keyword_to_clusters.items():
        if _DOMAIN_KEYWORD_MIN_CLUSTERS <= len(cnames) <= max_cluster_share:
            merge_groups[kw] = cnames

    if not merge_groups:
        return cluster_mapping

    # Step 3: Direct merge — for each keyword, merge its clusters into one.
    # Unlike Union-Find, this doesn't create transitive chains across keywords.
    # Each keyword group is merged independently.
    already_merged: set[str] = set()
    merged_clusters: dict[str, list[str]] = {}

    # Sort keywords by specificity (fewer clusters = more specific = merge first)
    for kw in sorted(merge_groups, key=lambda k: len(merge_groups[k])):
        cnames = [c for c in merge_groups[kw] if c not in already_merged]
        if len(cnames) < 2:
            continue

        # Calculate total files — don't create mega-clusters
        total_files = sum(len(cluster_mapping[c]) for c in cnames)
        total_all_files = sum(len(v) for v in cluster_mapping.values())
        max_merged_size = max(80, total_all_files // 5)
        if total_files > max_merged_size:
            continue

        # Merge all into one cluster
        combined: list[str] = []
        for c in cnames:
            combined.extend(cluster_mapping[c])
            already_merged.add(c)
        merged_clusters[kw] = sorted(set(combined))

    # Step 4: Build result — merged clusters + untouched originals
    result: dict[str, list[str]] = {}
    used_names: set[str] = set()
    # Add merged clusters first (named by keyword)
    for kw_name, files in merged_clusters.items():
        name = kw_name
        if name in used_names:
            suffix = 2
            while f"{name}-{suffix}" in used_names:
                suffix += 1
            name = f"{name}-{suffix}"
        used_names.add(name)
        result[name] = files
    # Add untouched originals
    for cname, cfiles in cluster_mapping.items():
        if cname not in already_merged:
            name = cname
            if name in used_names:
                suffix = 2
                while f"{name}-{suffix}" in used_names:
                    suffix += 1
                name = f"{name}-{suffix}"
            used_names.add(name)
            result[name] = cfiles

    consolidated = len(cluster_mapping) - len(result)
    if consolidated > 0:
        logger.info(
            "Domain consolidation: %d → %d clusters (merged %d by shared keywords)",
            len(cluster_mapping), len(result), consolidated,
        )
    return result


_PRE_MERGE_MAX_FILES = 3  # clusters with this many files or fewer get pre-merged
_PRE_MERGE_THRESHOLD = 150  # only pre-merge when total clusters exceed this

# Directories that are technical layers, not business domains.
# Clusters rooted in these are safe to absorb.
_TECHNICAL_DIR_NAMES = {
    "utils", "util", "helpers", "helper", "lib", "libs", "common", "shared",
    "core", "base", "config", "configs", "constants", "types", "interfaces",
    "hooks", "hoc", "providers", "context", "contexts", "middleware",
    "middlewares", "decorators", "guards", "interceptors", "pipes",
    "styles", "assets", "icons", "images", "fonts", "theme", "themes",
    "__tests__", "__mocks__", "test", "tests", "spec", "specs",
    "fixtures", "storybook", "stories",
}


def _cluster_has_distinct_dir(files: list[str]) -> bool:
    """Returns True if the cluster's files live in a unique, non-technical directory.

    Such clusters likely represent a distinct business domain and should not
    be absorbed into larger clusters during pre-merge.
    """
    dirs = set()
    for f in files:
        parts = Path(f).parts
        if len(parts) >= 2:
            dirs.add(parts[0])
    if len(dirs) != 1:
        return False
    dir_name = next(iter(dirs)).lower().rstrip("s")
    return dir_name not in {d.lower().rstrip("s") for d in _TECHNICAL_DIR_NAMES}


def _pre_merge_tiny_clusters(
    cluster_mapping: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Merges tiny clusters (≤_PRE_MERGE_MAX_FILES files) into the nearest large cluster.

    This reduces the number of clusters sent to the LLM merge step, making the
    prompt manageable for repos with 300+ import graph clusters.

    Clusters in a unique business-domain directory are protected from absorption.
    """
    if len(cluster_mapping) <= _PRE_MERGE_THRESHOLD:
        return cluster_mapping

    large: dict[str, list[str]] = {}
    tiny: dict[str, list[str]] = {}
    protected: dict[str, list[str]] = {}
    for name, files in cluster_mapping.items():
        if len(files) <= _PRE_MERGE_MAX_FILES:
            if _cluster_has_distinct_dir(files):
                protected[name] = files
            else:
                tiny[name] = files
        else:
            large[name] = files

    if not large:
        return cluster_mapping

    result = {name: list(files) for name, files in large.items()}

    for _name, files in tiny.items():
        target = _find_best_merge_target(files, result)
        if target:
            result[target].extend(files)
        else:
            result[_name] = files

    # Add protected clusters back — they stay as-is for LLM to see
    for name, files in protected.items():
        result[name] = files

    absorbed = len(cluster_mapping) - len(result)
    logger.info("Pre-merge: %d → %d clusters (absorbed %d tiny, protected %d)",
                len(cluster_mapping), len(result), absorbed, len(protected))
    return result


def _dedup_chunk_names(merged: dict[str, list[str]]) -> dict[str, list[str]]:
    """Merge features that only differ by a chunk-collision suffix (e.g. auth-2 → auth).

    When chunked merging names the same feature in two chunks, the second gets a
    '-2' suffix.  This merges them back together.
    """
    import re as _re

    suffix_re = _re.compile(r"^(.+)-(\d+)$")
    result: dict[str, list[str]] = {}
    for name, files in merged.items():
        m = suffix_re.match(name)
        base = m.group(1) if m and m.group(1) in merged else name
        if base in result:
            result[base].extend(files)
        else:
            result[base] = list(files)
    return result


def _chunked_cluster_merge(
    client: anthropic.Anthropic,
    cluster_mapping: dict[str, list[str]],
    keywords_per_cluster: dict[str, list[str]] | None = None,
    layer_context: str = "",
) -> dict[str, list[str]] | None:
    """Merges clusters in chunks to avoid prompt/response truncation.

    For small cluster sets (<=_MERGE_CHUNK_SIZE), does a single merge call.
    For larger sets, splits into chunks, merges each separately, then
    does a final merge pass to consolidate cross-chunk duplicates.

    Returns the final feature->files mapping, or None if merge failed.
    """
    if len(cluster_mapping) <= _MERGE_CHUNK_SIZE:
        response = _call_cluster_merge(
            client, cluster_mapping, keywords_per_cluster, layer_context=layer_context,
        )
        if response:
            return _apply_cluster_merge(cluster_mapping, response)
        return None

    cluster_items = list(cluster_mapping.items())
    chunks = [
        dict(cluster_items[i : i + _MERGE_CHUNK_SIZE])
        for i in range(0, len(cluster_items), _MERGE_CHUNK_SIZE)
    ]
    logger.info(
        "Chunked merge: %d clusters -> %d chunks of <=%d",
        len(cluster_mapping),
        len(chunks),
        _MERGE_CHUNK_SIZE,
    )

    all_merged: dict[str, list[str]] = {}
    for idx, chunk in enumerate(chunks):
        chunk_kw = (
            {k: v for k, v in keywords_per_cluster.items() if k in chunk}
            if keywords_per_cluster
            else None
        )
        response = _call_cluster_merge(client, chunk, chunk_kw or None, layer_context=layer_context)
        if response:
            chunk_result = _apply_cluster_merge(chunk, response)
            for name, files in chunk_result.items():
                unique_name = name
                suffix = 2
                while unique_name in all_merged:
                    unique_name = f"{name}-{suffix}"
                    suffix += 1
                all_merged[unique_name] = files
            logger.info(
                "Chunk %d/%d: %d clusters -> %d features",
                idx + 1,
                len(chunks),
                len(chunk),
                len(chunk_result),
            )
        else:
            for cid, files in chunk.items():
                all_merged[cid] = files
            logger.warning(
                "Chunk %d/%d failed, keeping %d raw clusters",
                idx + 1,
                len(chunks),
                len(chunk),
            )

    if not all_merged:
        return None

    # Final merge pass if intermediate result is still too fragmented
    if len(all_merged) > _MERGE_CHUNK_SIZE:
        logger.info(
            "Final merge pass: %d intermediate features",
            len(all_merged),
        )
        final_response = _call_cluster_merge(client, all_merged, None)
        if final_response:
            return _apply_cluster_merge(all_merged, final_response)
        logger.warning(
            "Final merge pass failed, returning %d intermediate features",
            len(all_merged),
        )

    # Deduplicate features with collision suffixes (e.g. "auth-2" → merge into "auth")
    return _dedup_chunk_names(all_merged)


# Single words: feature is technical if ALL its words are in this set.
_TECHNICAL_FEATURE_WORDS = {
    "utils", "util", "utilities", "helpers", "helper", "hooks", "hoc",
    "shared", "common", "general", "misc", "core", "base", "lib",
    "icons", "assets", "images", "fonts", "theme", "themes", "locale",
    "stories", "storybook", "mocks", "fixtures", "test", "tests",
    "styles", "css", "scss", "components", "ui",
    "workers", "worker",
    # Technical artifacts from validation
    "stores", "store", "dropdowns", "dropdown", "loaders", "loader",
    "constants", "config", "configs", "providers", "context", "contexts",
    "types", "models", "schemas", "middleware",
    "empty", "states", "state",  # "empty-states" is not a feature
    "list", "readonly", "fields",  # "list-components", "readonly-fields"
    "web", "manifest",  # "web-manifest"
    "preferences", "appearance", "sidebar",
}

# Multi-word patterns: feature names matching these exactly are technical.
_TECHNICAL_FEATURE_NAMES = {
    "ui-library", "ui-components", "shared-components", "custom-hooks",
    "hooks-utils", "general-utils", "data-schemas", "state-management",
    "api-services", "export-utilities", "app-shell", "routing",
    "icons-assets", "locale-theme", "custom-components", "custom-utils",
    "dashboard-utils", "chart-components", "input-component", "table-cells",
    "common-store", "template-components", "data-prefetch",
    "filter-processor", "entity-processors",
    # From validation on real repos
    "list-components", "rich-filters", "readonly-fields",
    "sidebar-stats", "web-manifest", "empty-states",
}


def _validate_merge_cohesion(
    merged: dict[str, list[str]],
    original_clusters: dict[str, list[str]],
    merge_response: _ClusterMergeResponse,
    min_cohesion: float = 0.005,
) -> dict[str, list[str]]:
    """Validates merged features by checking if multi-cluster merges are cohesive.

    For each merged feature that combined 3+ clusters: check if the files
    share a common directory structure. If not (low cohesion), revert the
    merge — keep original clusters as separate features.

    Uses directory overlap as a fast cohesion proxy (no import re-analysis needed).
    A merge is considered low-cohesion if the files span many unrelated directories
    with little overlap.
    """
    cluster_ids = list(original_clusters.keys())
    result = dict(merged)

    for item in merge_response.features:
        if len(item.cluster_indices) < 3:
            continue  # small merges are usually fine

        feat_name = item.feature_name
        if feat_name not in result:
            continue

        feat_files = result[feat_name]
        if len(feat_files) <= 15:
            continue  # small features don't need validation

        # Check directory cohesion: how many distinct top-level business dirs?
        biz_dirs: set[str] = set()
        for f in feat_files:
            parts = Path(f).parts
            for part in parts[:-1]:
                if part.lower() not in {"src", "app", "lib", "components", "hooks",
                                         "utils", "types", "shared", "common",
                                         "pages", "views", "features", "modules"}:
                    biz_dirs.add(part.lower())
                    break

        # If files span many unrelated directories, this is a bad merge
        dir_ratio = len(biz_dirs) / len(feat_files) if feat_files else 0

        if dir_ratio > 0.3 and len(biz_dirs) > 5:
            # Too fragmented — revert to original clusters
            logger.info(
                "Reverting merge of '%s' (%d files, %d dirs, ratio=%.2f) — low cohesion",
                feat_name, len(feat_files), len(biz_dirs), dir_ratio,
            )
            del result[feat_name]
            for idx in item.cluster_indices:
                if 1 <= idx <= len(cluster_ids):
                    cid = cluster_ids[idx - 1]
                    if cid in original_clusters:
                        result[cid] = original_clusters[cid]

    return result


def _filter_technical_features(
    merged: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Absorbs technical/utility features into the business feature with the most overlap.

    Features named after technical layers (hooks-utils, shared-components, stories)
    are not useful to engineering managers. This redistributes their files.
    """
    technical: dict[str, list[str]] = {}
    business: dict[str, list[str]] = {}

    # Prefixes that are always technical regardless of suffix
    _TECHNICAL_PREFIXES = ("assets-",)

    for name, files in merged.items():
        normalized = name.lower().replace("_", "-")
        parts = set(normalized.split("-"))
        is_technical = (
            normalized in _TECHNICAL_FEATURE_NAMES
            or (parts & _TECHNICAL_FEATURE_WORDS and not (parts - _TECHNICAL_FEATURE_WORDS - {""}))
            or any(normalized.startswith(p) for p in _TECHNICAL_PREFIXES)
        )
        if is_technical:
            technical[name] = files
        else:
            business[name] = files

    if not technical or not business:
        return merged

    for tech_name, tech_files in technical.items():
        target = _find_best_merge_target(tech_files, business)
        if target:
            business[target].extend(tech_files)
        else:
            # No single overlap target — try per-file matching
            unmatched: list[str] = []
            for f in tech_files:
                file_target = _find_best_merge_target([f], business)
                if file_target:
                    business[file_target].append(f)
                else:
                    unmatched.append(f)
            if unmatched:
                # Keep as separate feature with original name rather than
                # dumping into the largest feature
                business[tech_name] = unmatched

    logger.info("Filtered %d technical features, redistributed files into business features",
                len(technical))
    return business


def merge_and_name_clusters_llm(
    cluster_mapping: dict[str, list[str]],
    api_key: str | None = None,
    commits: list[Commit] | None = None,
    layer_context: str = "",
    cluster_edges: dict[str, dict[str, int]] | None = None,
) -> dict[str, list[str]]:
    """Uses Claude to merge import-graph clusters into business features and name them.

    Unlike name_clusters_llm(), this can merge multiple clusters into one feature —
    essential when related files don't import each other directly (e.g. Redux slices,
    separate services, cross-cutting utilities).

    When commits are provided, extracts top commit message keywords per cluster
    and includes them in the prompt as semantic naming hints.

    When cluster_edges is provided, each cluster in the prompt shows its import
    connections to other clusters, helping the LLM make informed merge decisions.

    Results are cached by cluster structure hash — same codebase → same result.
    Falls back to the original cluster_mapping on any error.

    Args:
        cluster_mapping: Output of build_import_clusters().
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        commits: Optional commit history for keyword extraction.
        cluster_edges: Inter-cluster import connections from compute_cluster_edges().
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not cluster_mapping:
        return cluster_mapping

    # Consolidate clusters sharing business-domain keywords (e.g. "labels")
    working_clusters = _consolidate_domain_clusters(cluster_mapping)
    # Pre-merge tiny clusters to keep prompt size manageable for large repos
    working_clusters = _pre_merge_tiny_clusters(working_clusters)

    cache_key = _merge_cache_key(working_clusters, _MODEL)
    cached = _read_name_cache(cache_key)
    if cached is not None:
        if isinstance(next(iter(cached.values()), None), list):
            # Cache stores LLM merge result before redistribute — apply redistribute now
            return _final_cleanup(_redistribute_oversized_features(
                _redistribute_infra_features(
                    _extract_shared_ui(
                        _collapse_plugin_features(cached)  # type: ignore[arg-type]
                    )
                )
            ))

    keywords_per_cluster = _extract_cluster_keywords(working_clusters, commits) if commits else None

    client = anthropic.Anthropic(api_key=key)

    # Recompute edges for working_clusters if edges were provided for original clusters
    # (consolidation/pre-merge may have changed cluster names)
    working_edges = cluster_edges  # pass through — edge keys may not match perfectly but that's OK

    # Try single-shot merge first (preserves previous behavior).
    # Fall back to chunked merge only when single-shot fails (large repos).
    merge_response = _call_cluster_merge(
        client, working_clusters, keywords_per_cluster,
        layer_context=layer_context, cluster_edges=working_edges,
    )
    if merge_response:
        merged = _filter_technical_features(
            _apply_cluster_merge(working_clusters, merge_response),
        )
        # Validate: revert merges with low internal cohesion
        merged = _validate_merge_cohesion(merged, working_clusters, merge_response)

        _write_name_cache(cache_key, merged)  # type: ignore[arg-type]
        merged = _collapse_plugin_features(merged)
        merged = _extract_shared_ui(merged)
        merged = _redistribute_infra_features(merged)
        merged = _redistribute_oversized_features(merged)
        return _final_cleanup(merged)

    # Single-shot failed (likely truncation) — try chunked merge
    if len(working_clusters) > _MERGE_CHUNK_SIZE:
        logger.info("Single-shot merge failed, trying chunked merge")
        merged = _chunked_cluster_merge(
            client, working_clusters, keywords_per_cluster, layer_context=layer_context,
        )
        if merged:
            merged = _filter_technical_features(merged)
            _write_name_cache(cache_key, merged)  # type: ignore[arg-type]
            merged = _redistribute_infra_features(merged)
            merged = _redistribute_oversized_features(merged)
            return _final_cleanup(merged)

    return _final_cleanup(working_clusters)


def _call_cluster_merge_ollama(
    cluster_mapping: dict[str, list[str]],
    model: str,
    host: str,
    keywords_per_cluster: dict[str, list[str]] | None = None,
    layer_context: str = "",
) -> _ClusterMergeResponse | None:
    """Calls Ollama for cluster merge+name. Returns None on any failure."""
    try:
        import ollama as _ollama
    except ImportError:
        return None

    prompt = _MERGE_USER_PROMPT.format(
        clusters=_format_clusters_for_merge_prompt(cluster_mapping, keywords_per_cluster),
        feature_hint=_merge_feature_count_hint(len(cluster_mapping)),
    )
    system = _MERGE_SYSTEM_PROMPT + layer_context
    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            format=_ClusterMergeResponse.model_json_schema(),
        )
        return _ClusterMergeResponse.model_validate_json(response.message.content)
    except (ValidationError, Exception):
        return None


def merge_and_name_clusters_ollama(
    cluster_mapping: dict[str, list[str]],
    model: str = _DEFAULT_OLLAMA_MODEL,
    host: str = _DEFAULT_OLLAMA_HOST,
    commits: list[Commit] | None = None,
    layer_context: str = "",
) -> dict[str, list[str]]:
    """Ollama version of merge_and_name_clusters_llm. See that function for full docs."""
    if not cluster_mapping:
        return cluster_mapping

    working_clusters = _consolidate_domain_clusters(cluster_mapping)
    working_clusters = _pre_merge_tiny_clusters(working_clusters)

    cache_key = _merge_cache_key(working_clusters, model)
    cached = _read_name_cache(cache_key)
    if cached is not None:
        if isinstance(next(iter(cached.values()), None), list):
            return _final_cleanup(_redistribute_oversized_features(
                _redistribute_infra_features(
                    _extract_shared_ui(
                        _collapse_plugin_features(cached)  # type: ignore[arg-type]
                    )
                )
            ))

    keywords_per_cluster = _extract_cluster_keywords(working_clusters, commits) if commits else None

    merge_response = _call_cluster_merge_ollama(working_clusters, model, host, keywords_per_cluster, layer_context=layer_context)
    if merge_response:
        merged = _filter_technical_features(
            _apply_cluster_merge(working_clusters, merge_response),
        )
        _write_name_cache(cache_key, merged)  # type: ignore[arg-type]
        merged = _collapse_plugin_features(merged)
        merged = _extract_shared_ui(merged)
        merged = _redistribute_infra_features(merged)
        merged = _redistribute_oversized_features(merged)
        return _final_cleanup(merged)

    return _final_cleanup(working_clusters)


# ── Cluster naming (co-change grouping → semantic names) ─────────────────────
# Used when co-change detection produced the clusters and LLM only needs to name them.
# Results are cached: same file set → same names on every run.

_NAME_CACHE_DIR = Path.home() / ".faultline" / "llm-cache"
_NAME_CACHE_TTL_DAYS = 90

_NAMING_SYSTEM_PROMPT = """\
You are a software architect assigning business domain names to feature clusters.
Each cluster is a group of files that frequently change together in git history —
they belong to the same business feature even if spread across multiple directories.

Rules:
- Feature names must be lowercase, hyphen-separated, 1–3 words.
- Use business domain terminology, not technical layer names.
- Examples: "user-auth", "payment-processing", "dashboard", "notifications", "team-management".
- Every cluster must receive a unique name.
- Return exactly one name per cluster index — no skipping.\
"""

_NAMING_USER_PROMPT = """\
Name each feature cluster below. Each cluster contains files that change together in git.

{clusters}

Return a feature_name for each cluster by its index.\
"""


class _ClusterNamingItem(BaseModel):
    index: int
    feature_name: str


class _ClusterNamingResponse(BaseModel):
    features: list[_ClusterNamingItem]


def _cluster_cache_key(cluster_mapping: dict[str, list[str]], model: str) -> str:
    """Stable SHA256 cache key based on all files across all clusters."""
    all_files = sorted(f for files in cluster_mapping.values() for f in files)
    content = json.dumps(all_files) + model
    return hashlib.sha256(content.encode()).hexdigest()[:24]


def _read_name_cache(key: str) -> dict[str, str] | None:
    """Returns cached {cluster_id: feature_name} mapping or None if missing/expired."""
    path = _NAME_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    age_days = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
    if age_days > _NAME_CACHE_TTL_DAYS:
        path.unlink()
        return None
    return json.loads(path.read_text())


def _write_name_cache(key: str, names: dict[str, str]) -> None:
    _NAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_NAME_CACHE_DIR / f"{key}.json").write_text(json.dumps(names))


def _format_clusters_for_prompt(cluster_mapping: dict[str, list[str]]) -> str:
    """Formats cluster mapping as a numbered list for the naming prompt."""
    lines = []
    for i, (cluster_id, files) in enumerate(cluster_mapping.items(), start=1):
        sample = files[:8]
        file_lines = "\n".join(f"  {f}" for f in sample)
        suffix = f" … ({len(files) - 8} more)" if len(files) > 8 else ""
        lines.append(f"Cluster {i}:\n{file_lines}{suffix}")
    return "\n\n".join(lines)


def _apply_cluster_names(
    cluster_mapping: dict[str, list[str]],
    names: dict[str, str],
) -> dict[str, list[str]]:
    """Replaces cluster IDs with LLM-generated names, deduplicating collisions."""
    result: dict[str, list[str]] = {}
    used: set[str] = set()
    for cluster_id, files in cluster_mapping.items():
        name = names.get(cluster_id, cluster_id)
        if name in used:
            suffix = 2
            while f"{name}-{suffix}" in used:
                suffix += 1
            name = f"{name}-{suffix}"
        used.add(name)
        result[name] = files
    return result


def _call_cluster_naming(
    client: anthropic.Anthropic,
    cluster_mapping: dict[str, list[str]],
) -> dict[str, str] | None:
    """Sends all clusters to Claude in one call. Returns {cluster_id: name} or None."""
    cluster_ids = list(cluster_mapping.keys())
    prompt = _NAMING_USER_PROMPT.format(
        clusters=_format_clusters_for_prompt(cluster_mapping),
    )

    try:
        response = client.messages.parse(
            model=_MODEL,
            max_tokens=512,
            temperature=0,
            system=_NAMING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=_ClusterNamingResponse,
        )
        items = response.parsed_output.features
        # Map 1-based index back to cluster_id
        return {
            cluster_ids[item.index - 1]: item.feature_name
            for item in items
            if 1 <= item.index <= len(cluster_ids)
        }
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        ValidationError,
        IndexError,
    ):
        return None


def name_clusters_llm(
    cluster_mapping: dict[str, list[str]],
    api_key: str | None = None,
) -> dict[str, list[str]]:
    """Uses Claude to assign semantic names to co-change clusters.

    Results are cached by a hash of the file set — same repo state always
    returns the same names without making another API call.

    Falls back to the original cluster_mapping (directory-derived names)
    on any error or missing API key.

    Args:
        cluster_mapping: Output of detect_features_from_cochange().
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not cluster_mapping:
        return cluster_mapping

    cache_key = _cluster_cache_key(cluster_mapping, _MODEL)
    cached = _read_name_cache(cache_key)
    if cached is not None:
        return _apply_cluster_names(cluster_mapping, cached)

    client = anthropic.Anthropic(api_key=key)
    names = _call_cluster_naming(client, cluster_mapping)
    if names:
        _write_name_cache(cache_key, names)
        return _apply_cluster_names(cluster_mapping, names)

    return cluster_mapping


def _call_cluster_naming_ollama(
    cluster_mapping: dict[str, list[str]],
    model: str,
    host: str,
) -> dict[str, str] | None:
    """Calls Ollama to name clusters. Returns {cluster_id: name} or None."""
    try:
        import ollama as _ollama
    except ImportError:
        return None

    cluster_ids = list(cluster_mapping.keys())
    prompt = _NAMING_USER_PROMPT.format(
        clusters=_format_clusters_for_prompt(cluster_mapping),
    )

    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _NAMING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format=_ClusterNamingResponse.model_json_schema(),
        )
        parsed = _ClusterNamingResponse.model_validate_json(response.message.content)
        return {
            cluster_ids[item.index - 1]: item.feature_name
            for item in parsed.features
            if 1 <= item.index <= len(cluster_ids)
        }
    except (ValidationError, IndexError, Exception):
        return None


def name_clusters_ollama(
    cluster_mapping: dict[str, list[str]],
    model: str = _DEFAULT_OLLAMA_MODEL,
    host: str = _DEFAULT_OLLAMA_HOST,
) -> dict[str, list[str]]:
    """Ollama version of name_clusters_llm. See name_clusters_llm() for full docs."""
    if not cluster_mapping:
        return cluster_mapping

    cache_key = _cluster_cache_key(cluster_mapping, model)
    cached = _read_name_cache(cache_key)
    if cached is not None:
        return _apply_cluster_names(cluster_mapping, cached)

    names = _call_cluster_naming_ollama(cluster_mapping, model, host)
    if names:
        _write_name_cache(cache_key, names)
        return _apply_cluster_names(cluster_mapping, names)

    return cluster_mapping


def validate_api_key(api_key: str | None = None) -> tuple[bool, str]:
    """
    Validates the Anthropic API key before running the full analysis.
    Returns (is_valid, error_message).
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return False, "No API key provided. Use --api-key or set ANTHROPIC_API_KEY env var."

    if not key.startswith("sk-ant-"):
        return False, (
            f"Key format looks wrong (got: {key[:10]}...). "
            "Anthropic API keys start with 'sk-ant-'. "
            "Get yours at console.anthropic.com → API Keys."
        )

    client = anthropic.Anthropic(api_key=key)
    try:
        client.messages.create(
            model=_MODEL,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True, ""
    except anthropic.AuthenticationError as e:
        return False, (
            f"API key rejected by Anthropic ({e.status_code}). "
            "The key may be revoked or incorrect. "
            "Check console.anthropic.com → API Keys."
        )
    except anthropic.PermissionDeniedError:
        return False, (
            f"API key has no access to model '{_MODEL}'. "
            "Check your plan at console.anthropic.com."
        )
    except anthropic.APIConnectionError:
        return False, "Cannot reach Anthropic API. Check your internet connection."
    except anthropic.APIStatusError as e:
        if e.status_code == 400 and "credit balance" in str(e.message).lower():
            return False, (
                "Insufficient credits. Add funds at console.anthropic.com → Settings → Billing."
            )
        return False, f"Unexpected API error (HTTP {e.status_code}): {e.message}"


def enrich_features(
    features: list[Feature],
    api_key: str | None = None,
) -> list[Feature]:
    """
    Enriches features with LLM-generated descriptions.
    Returns original features unchanged if the API call fails or no key is provided.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not features:
        return features

    client = anthropic.Anthropic(api_key=key)
    enrichments = _fetch_enrichments(client, features)
    return _apply_enrichments(features, enrichments)


def _fetch_enrichments(
    client: anthropic.Anthropic,
    features: list[Feature],
) -> list[_FeatureEnrichment]:
    """Calls Claude API and returns enrichments. Returns empty list on any failure."""
    feature_data = [
        {
            "name": f.name,
            "sample_paths": f.paths[:_MAX_SAMPLE_PATHS],
        }
        for f in features[:_MAX_FEATURES_PER_CALL]
    ]

    try:
        response = client.messages.parse(
            model=_MODEL,
            max_tokens=1024,
            temperature=0,
            system=(
                "You are a software architecture analyst. "
                "Analyze code modules by their directory names and file paths, "
                "and return structured metadata about each one."
            ),
            messages=[{
                "role": "user",
                "content": (
                    "For each code module below, provide:\n"
                    "- original_name: exactly the same name as given (do not change it)\n"
                    "- description: one sentence describing what this module does\n\n"
                    f"Modules:\n{json.dumps(feature_data, indent=2)}"
                ),
            }],
            output_format=_EnrichmentResponse,
        )
        return response.parsed_output.features
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        ValidationError,
    ):
        return []


def _apply_enrichments(
    features: list[Feature],
    enrichments: list[_FeatureEnrichment],
) -> list[Feature]:
    """Merges LLM enrichment data into existing Feature objects."""
    by_name = {e.original_name: e for e in enrichments}
    return [
        feature.model_copy(update={"description": by_name[feature.name].description})
        if feature.name in by_name
        else feature
        for feature in features
    ]
