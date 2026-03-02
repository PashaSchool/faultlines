import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

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
_DIR_COLLAPSE_THRESHOLD = 300
# Max route-anchor entries injected into the LLM prompt
_MAX_ROUTE_ANCHOR_FILES = 25
_MAX_ROUTES_PER_ENTRY   = 3
# Max sample filenames shown per directory in the enriched dir tree
_DIR_SAMPLE_FILES = 4
# Top co-change pairs to include in the prompt
_MAX_COCHANGE_IN_PROMPT = 20

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
5. Every feature must contain at least 2 files. If a file would be the sole member of a feature, merge it into the most closely related feature.
6. Test files belong to the same feature as the code they test. Match by naming convention (test_auth.py belongs with auth.py, UserService.test.ts belongs with UserService.ts).
7. Skip infrastructure and tooling files entirely: package.json, pyproject.toml, setup.py, .gitignore, Makefile, *.lock, *.toml, Dockerfile, docker-compose.yml, CI configs.
8. Shared utility files go into the most closely related business feature, or into "shared-utilities" only if they truly cross all feature boundaries.
9. For monorepo structures, group by business feature across packages when the same domain spans multiple packages.
10. If a <route-anchors> section is provided, treat those files as strong feature anchors.
    Files that define API routes (GET/POST/PUT/DELETE) are entry points to a feature — group
    other files in the same directory tree with the file that shares their route prefix.

## Anti-patterns

BAD — grouping by technical layer:
  "components": [LoginForm.tsx, CheckoutForm.tsx, Dashboard.tsx]
  "api": [auth.ts, payments.ts, analytics.ts]

GOOD — grouping by business domain:
  "user-auth": [LoginForm.tsx, auth.ts]
  "checkout": [CheckoutForm.tsx, payments.ts]
  "analytics": [Dashboard.tsx, analytics.ts]

## Example

Files:
  components/LoginForm.tsx
  components/CheckoutForm.tsx
  api/auth/login.ts
  api/payments/charge.ts
  hooks/useSession.ts
  utils/currency.ts

Reasoning: LoginForm.tsx, api/auth/login.ts, and hooks/useSession.ts all serve user authentication across different technical layers. CheckoutForm.tsx and api/payments/charge.ts handle payment processing. utils/currency.ts is a shared utility — assign to the feature that uses it most.

Result:
  "user-auth": [components/LoginForm.tsx, api/auth/login.ts, hooks/useSession.ts]
  "checkout": [components/CheckoutForm.tsx, api/payments/charge.ts, utils/currency.ts]\
"""

_DETECTION_USER_PROMPT = """\
Analyze these repository files and group them into semantic business features.

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
5. Be granular — prefer many focused features over a few broad ones. \
   Deeply nested subdirectories with distinct responsibilities often warrant their own feature.
6. Sibling subdirectories that serve different concerns should be separate features even if \
   they share a parent directory.
7. Merge only truly tiny or ambiguous leaf directories into the closest related feature.
8. Skip pure infrastructure directories: .storybook, __mocks__, .github, ci/, scripts/, etc.
9. If a <route-anchors> section is provided, directories with routes are strong anchors.
   Assign nearby sibling directories to the same feature as the directory that shares
   the same route prefix (e.g. /api/payments/* dirs belong to the payments feature).

## Anti-patterns

BAD — grouping too broadly (one feature swallows 30 dirs):
  "server": ["server/app-render", "server/app-render/utils", "server/edge-runtime", ...]  ← TOO BROAD

GOOD — granular, each feature is a distinct concern:
  "app-router":    ["server/app-render", "server/app-render/utils"]
  "edge-runtime":  ["server/edge-runtime", "server/edge-runtime/utils"]

BAD — putting individual filenames in `files`:
  "auth": ["LoginForm.tsx", "useAuth.ts"]  ← WRONG, these are filenames not directories

GOOD — putting directory paths exactly as listed:
  "auth": ["src/auth", "src/hooks/auth", "src/api/auth"]  ← correct

## Example

Input:
  src/auth → LoginForm.tsx, useSession.ts
    src/auth/utils → token.ts
  src/api/auth → login.ts, logout.ts
  src/payments → CheckoutForm.tsx, stripe.ts
    src/payments/hooks → useCheckout.ts
  src/api/payments → charge.ts, refund.ts

Result:
  "user-auth": ["src/auth", "src/auth/utils", "src/api/auth"]
  "checkout":  ["src/payments", "src/payments/hooks", "src/api/payments"]\
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
        extra_context = _format_extra_context(cochange_pairs, dir_keywords) + route_anchors
        response = _call_dir_detection(client, file_tree, n_dirs=len(dirs), extra_context=extra_context)
        if not response:
            return {}
        return _expand_dir_mapping(response, files)
    else:
        file_tree = "\n".join(files[:_MAX_FILES_FOR_DETECTION])
        route_anchors = _format_route_anchors(signatures) if signatures else ""
        extra_context = _format_extra_context(cochange_pairs, {}) + route_anchors
        response = _call_feature_detection(client, file_tree, extra_context)
        if not response:
            return {}
        return _build_feature_dict(response, set(files))


def _call_feature_detection(
    client: anthropic.Anthropic,
    file_tree: str,
    extra_context: str = "",
) -> _FeatureDetectionResponse | None:
    """Calls Claude API for feature detection (file-path mode). Returns None on any failure."""
    prompt = _DETECTION_USER_PROMPT.format(file_tree=file_tree, extra_context=extra_context)

    try:
        response = client.messages.parse(
            model=_MODEL,
            max_tokens=_MAX_TOKENS_FILE,
            temperature=0,
            system=_DETECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=_FeatureDetectionResponse,
        )
        return response.parsed_output
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        ValidationError,
    ):
        return None


def _call_dir_detection(
    client: anthropic.Anthropic,
    file_tree: str,
    n_dirs: int,
    extra_context: str = "",
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

    try:
        response = client.messages.parse(
            model=_MODEL,
            max_tokens=_MAX_TOKENS_DIR,
            temperature=0,
            system=_DIR_DETECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=_FeatureDetectionResponse,
        )
        return response.parsed_output
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        ValidationError,
    ):
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
    """For each directory, returns up to _DIR_SAMPLE_FILES representative file names."""
    samples: dict[str, list[str]] = {d: [] for d in dirs}
    for f in all_files:
        parent = str(Path(f).parent)
        if parent in samples and len(samples[parent]) < _DIR_SAMPLE_FILES:
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
    min_f = max(8, n_dirs // 15)
    max_f = max(15, n_dirs // 7)
    return (
        f"\nYou have {n_dirs} directories. "
        f"Aim for {min_f}–{max_f} focused features. "
        "Be granular — deeply nested subdirectories with distinct responsibilities "
        "should be separate features, not merged into one broad group.\n"
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
        extra_context = _format_extra_context(cochange_pairs, dir_keywords) + route_anchors
        response = _call_dir_detection_ollama(file_tree, model, host, n_dirs=len(dirs), extra_context=extra_context)
        if not response:
            return {}
        return _expand_dir_mapping(response, files)
    else:
        file_tree = "\n".join(files[:_MAX_FILES_FOR_DETECTION])
        route_anchors = _format_route_anchors(signatures) if signatures else ""
        extra_context = _format_extra_context(cochange_pairs, {}) + route_anchors
        response = _call_feature_detection_ollama(file_tree, model, host, extra_context)
        if not response:
            return {}
        return _build_feature_dict(response, set(files))


def _call_feature_detection_ollama(
    file_tree: str,
    model: str,
    host: str,
    extra_context: str = "",
) -> _FeatureDetectionResponse | None:
    """Calls Ollama API for feature detection (file-path mode). Returns None on any failure."""
    try:
        import ollama as _ollama
    except ImportError:
        return None

    prompt = _DETECTION_USER_PROMPT.format(file_tree=file_tree, extra_context=extra_context)

    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _DETECTION_SYSTEM_PROMPT},
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

    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _DIR_DETECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format=_FeatureDetectionResponse.model_json_schema(),
        )
        return _FeatureDetectionResponse.model_validate_json(response.message.content)
    except (ValidationError, Exception):
        return None


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
- Merge clusters by business purpose, not by technical role.
  BAD: merge all hooks clusters into "hooks-feature".
  GOOD: merge auth-hooks + auth-components + auth-api into "user-auth".
- Keep genuinely distinct capabilities as separate features.
- Every cluster must appear in exactly one feature — no omissions.
- cluster_indices contains 1-based indices from the list provided.\
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

    Prevents local models from merging everything into one giant feature.
    """
    min_f = max(5, n_clusters // 6)
    max_f = max(10, n_clusters // 3)
    max_per = max(5, n_clusters // 8)
    return (
        f"\nYou have {n_clusters} clusters. "
        f"Aim for {min_f}–{max_f} focused features. "
        f"Do NOT put more than {max_per} clusters into a single feature "
        "unless they are truly tightly related.\n"
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
) -> str:
    """Formats clusters as a numbered list for the LLM merge prompt.

    When keywords_per_cluster is provided, each cluster entry includes its
    top commit message topics as a hint for semantic business domain naming.
    """
    lines = []
    for i, (cluster_id, files) in enumerate(cluster_mapping.items(), start=1):
        sample = files[:8]
        file_lines = "\n".join(f"  {f}" for f in sample)
        suffix = f"\n  … ({len(files) - 8} more)" if len(files) > 8 else ""
        keywords = (keywords_per_cluster or {}).get(cluster_id, [])
        kw_line = f"\n  Commit topics: {', '.join(keywords)}" if keywords else ""
        lines.append(f"Cluster {i} ({cluster_id}):{kw_line}\n{file_lines}{suffix}")
    return "\n\n".join(lines)


def _apply_cluster_merge(
    cluster_mapping: dict[str, list[str]],
    merge_response: _ClusterMergeResponse,
) -> dict[str, list[str]]:
    """Builds the merged feature mapping from the LLM response.

    Handles duplicate names, out-of-range indices, and unassigned clusters
    (any cluster not referenced falls back to its original directory-derived name).
    """
    cluster_ids = list(cluster_mapping.keys())
    assigned: set[int] = set()
    result: dict[str, list[str]] = {}
    used_names: set[str] = set()

    for item in merge_response.features:
        merged_files: list[str] = []
        for idx in item.cluster_indices:
            if 1 <= idx <= len(cluster_ids) and idx not in assigned:
                cluster_id = cluster_ids[idx - 1]
                merged_files.extend(cluster_mapping[cluster_id])
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

    # Fallback: any cluster not referenced by LLM keeps its original name
    for i, cluster_id in enumerate(cluster_ids, start=1):
        if i not in assigned:
            name = cluster_id
            if name in used_names:
                suffix = 2
                while f"{name}-{suffix}" in used_names:
                    suffix += 1
                name = f"{name}-{suffix}"
            used_names.add(name)
            result[name] = cluster_mapping[cluster_id]

    return result


def _call_cluster_merge(
    client: anthropic.Anthropic,
    cluster_mapping: dict[str, list[str]],
    keywords_per_cluster: dict[str, list[str]] | None = None,
) -> _ClusterMergeResponse | None:
    """Sends all clusters to Claude for merge+name. Returns None on any failure."""
    prompt = _MERGE_USER_PROMPT.format(
        clusters=_format_clusters_for_merge_prompt(cluster_mapping, keywords_per_cluster),
        feature_hint=_merge_feature_count_hint(len(cluster_mapping)),
    )
    try:
        response = client.messages.parse(
            model=_MODEL,
            max_tokens=2048,
            temperature=0,
            system=_MERGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=_ClusterMergeResponse,
        )
        return response.parsed_output
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        ValidationError,
    ):
        return None


def merge_and_name_clusters_llm(
    cluster_mapping: dict[str, list[str]],
    api_key: str | None = None,
    commits: list[Commit] | None = None,
) -> dict[str, list[str]]:
    """Uses Claude to merge import-graph clusters into business features and name them.

    Unlike name_clusters_llm(), this can merge multiple clusters into one feature —
    essential when related files don't import each other directly (e.g. Redux slices,
    separate services, cross-cutting utilities).

    When commits are provided, extracts top commit message keywords per cluster
    and includes them in the prompt as semantic naming hints.

    Results are cached by cluster structure hash — same codebase → same result.
    Falls back to the original cluster_mapping on any error.

    Args:
        cluster_mapping: Output of build_import_clusters().
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        commits: Optional commit history for keyword extraction.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not cluster_mapping:
        return cluster_mapping

    cache_key = _merge_cache_key(cluster_mapping, _MODEL)
    cached = _read_name_cache(cache_key)
    if cached is not None:
        # cached format for merge is the full feature→files mapping
        if isinstance(next(iter(cached.values()), None), list):
            return cached  # type: ignore[return-value]

    keywords_per_cluster = _extract_cluster_keywords(cluster_mapping, commits) if commits else None

    client = anthropic.Anthropic(api_key=key)
    merge_response = _call_cluster_merge(client, cluster_mapping, keywords_per_cluster)
    if merge_response:
        merged = _apply_cluster_merge(cluster_mapping, merge_response)
        _write_name_cache(cache_key, merged)  # type: ignore[arg-type]
        return merged

    return cluster_mapping


def _call_cluster_merge_ollama(
    cluster_mapping: dict[str, list[str]],
    model: str,
    host: str,
    keywords_per_cluster: dict[str, list[str]] | None = None,
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
    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _MERGE_SYSTEM_PROMPT},
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
) -> dict[str, list[str]]:
    """Ollama version of merge_and_name_clusters_llm. See that function for full docs."""
    if not cluster_mapping:
        return cluster_mapping

    cache_key = _merge_cache_key(cluster_mapping, model)
    cached = _read_name_cache(cache_key)
    if cached is not None:
        if isinstance(next(iter(cached.values()), None), list):
            return cached  # type: ignore[return-value]

    keywords_per_cluster = _extract_cluster_keywords(cluster_mapping, commits) if commits else None

    merge_response = _call_cluster_merge_ollama(cluster_mapping, model, host, keywords_per_cluster)
    if merge_response:
        merged = _apply_cluster_merge(cluster_mapping, merge_response)
        _write_name_cache(cache_key, merged)  # type: ignore[arg-type]
        return merged

    return cluster_mapping


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
