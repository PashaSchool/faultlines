"""Centralized validation and filtering primitives for the feature pipeline.

The legacy pipeline had six scattered places that each re-implemented
"is this a test file?", "should this feature be dropped?", and
"what's the canonical name for this bucket?". This module exposes those
decisions as pure functions with explicit inputs and unit-tested behaviour.

This file is consumed by both the legacy detector (gradually, as sites
are migrated) and the new pipeline (from day one). Nothing here performs
I/O or calls LLMs — everything is deterministic given its inputs.

Categories:
  - Test-file / test-package filtering (no test dir should become a feature)
  - Documentation / example / tutorial collapse (new Day 1 criterion)
  - Canonical bucket naming (root/init/main → shared-infra)
  - Phantom feature removal (zero files, duplicates)
"""

from __future__ import annotations

from pathlib import PurePosixPath

# ── Test filtering ────────────────────────────────────────────────────────

# Path segments that mark a file as test code. Matching is on exact
# segment equality (case-insensitive), not substring, to avoid false
# positives like "attestation/" or "contest/".
_TEST_DIR_SEGMENTS = frozenset({
    "test", "tests", "__tests__", "testing", "testutils", "test-utils",
    "e2e", "integration-tests", "unit-tests", "spec", "specs",
    "fixtures", "__fixtures__", "mocks", "__mocks__", "__snapshots__",
    "cypress", "playwright", "vitest-mocks",
})

# Filename patterns that mark individual test files even when they
# live next to production code.
_TEST_FILENAME_SUFFIXES = (
    ".test.ts", ".test.tsx", ".test.js", ".test.jsx", ".test.mjs",
    ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx", ".spec.mjs",
    ".test.py", "_test.py", "_test.go",
    ".e2e.ts", ".e2e.js",
)

# Feature / package names that should never be presented to users as
# business features regardless of size.
_TEST_FEATURE_NAMES = frozenset({
    "test", "tests", "__tests__", "testing", "testutils",
    "e2e", "e2e-tests", "integration-tests", "unit-tests",
    "spec", "specs", "fixtures", "__fixtures__",
    "mocks", "__mocks__", "__snapshots__",
    "vitest-mocks", "jest-mocks", "cypress", "playwright",
})


def is_test_file(path: str) -> bool:
    """Return True if a file path represents test code.

    A file is considered a test file if any of its directory segments
    match a known test directory name OR its filename ends with a known
    test suffix. This is the single source of truth used by both the
    legacy detector and the new pipeline.

    >>> is_test_file("src/auth/login.test.ts")
    True
    >>> is_test_file("packages/core/__tests__/parser.ts")
    True
    >>> is_test_file("src/attestation/handler.ts")
    False
    >>> is_test_file("README.md")
    False
    """
    p = PurePosixPath(path)
    for segment in p.parts[:-1]:
        if segment.lower() in _TEST_DIR_SEGMENTS:
            return True
    name = p.name.lower()
    return any(name.endswith(suffix) for suffix in _TEST_FILENAME_SUFFIXES)


def is_test_feature_name(name: str) -> bool:
    """Return True if a feature name should be filtered as test infrastructure.

    Used to drop heuristic candidates like ``vitest-mocks`` (observed in the
    cal.com baseline) or ``__fixtures__`` before they reach the LLM.

    >>> is_test_feature_name("vitest-mocks")
    True
    >>> is_test_feature_name("authentication")
    False
    """
    return name.lower().strip() in _TEST_FEATURE_NAMES


# ── Documentation / tutorial collapse ─────────────────────────────────────

# Directory segments that mark documentation/tutorial/example content.
# Files under these directories are NOT business features; they collapse
# into a single synthetic "documentation" feature or get dropped entirely
# depending on repo configuration.
_DOCS_DIR_SEGMENTS = frozenset({
    "docs", "doc", "docs_src", "documentation",
    "examples", "example", "demos", "demo",
    "tutorials", "tutorial",
    "www", "website", "site", "landing", "marketing",
    "playground", "playgrounds", "sandbox", "sandboxes",
    "cookbook", "recipes",
})


def is_documentation_file(path: str) -> bool:
    """Return True if a file belongs to documentation, tutorial, or example content.

    Matches when ANY directory segment in the path is a known docs segment.
    This catches both top-level ``docs/``/``examples/`` and nested paths
    like ``apps/docs/``.

    >>> is_documentation_file("docs_src/tutorial001_py310/main.py")
    True
    >>> is_documentation_file("apps/docs/next.config.ts")
    True
    >>> is_documentation_file("src/auth/login.ts")
    False
    >>> is_documentation_file("README.md")
    False
    """
    p = PurePosixPath(path)
    return any(segment.lower() in _DOCS_DIR_SEGMENTS for segment in p.parts[:-1])


# ── Canonical bucket naming ───────────────────────────────────────────────

# Aliases that the legacy pipeline occasionally emits as feature names
# but that should always be canonicalized to ``shared-infra``. Observed in
# the Day 1 baseline on gin (``root``) and various detector paths (``init``).
_SHARED_INFRA_ALIASES = frozenset({
    "root", "init", "__init__", "main", "src", "source", "lib", "libs",
    "utils", "util", "helpers", "helper", "common", "shared", "core",
    "infra", "infrastructure", "config", "configs",
})

CANONICAL_SHARED_INFRA = "shared-infra"


def canonical_bucket_name(name: str) -> str:
    """Return the canonical form of a bucket name.

    Used to merge duplicates like ``root``, ``init``, ``main`` into a
    single ``shared-infra`` feature before display.

    >>> canonical_bucket_name("root")
    'shared-infra'
    >>> canonical_bucket_name("authentication")
    'authentication'
    >>> canonical_bucket_name("SHARED")
    'shared-infra'
    """
    cleaned = name.lower().strip().strip("/")
    if cleaned in _SHARED_INFRA_ALIASES:
        return CANONICAL_SHARED_INFRA
    return name


# ── Phantom feature removal ───────────────────────────────────────────────


def drop_phantom_features(
    features: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Remove features with zero files and test-named features.

    The legacy pipeline sometimes emits features that match no files
    (phantoms from LLM hallucination) or test infrastructure that slipped
    through earlier filtering. This is the final cleanup pass before a
    feature map is written to disk.

    Unlike the legacy ``_final_cleanup``, this function performs NO
    merging or renaming — it only drops entries. Merging is a separate
    concern handled by the post-processing module.

    >>> drop_phantom_features({"auth": ["a.ts"], "ghost": [], "tests": ["t.ts"]})
    {'auth': ['a.ts']}
    """
    return {
        name: paths
        for name, paths in features.items()
        if paths and not is_test_feature_name(name)
    }


# ── Combined helpers ──────────────────────────────────────────────────────


def filter_test_files(files: list[str]) -> list[str]:
    """Return only the files that are not test files.

    Convenience wrapper used by the pipeline before building feature
    candidates. Preserves input order.

    >>> filter_test_files(["src/a.ts", "src/a.test.ts", "README.md"])
    ['src/a.ts', 'README.md']
    """
    return [f for f in files if not is_test_file(f)]


def partition_docs_vs_code(
    files: list[str],
) -> tuple[list[str], list[str]]:
    """Split files into (code_files, docs_files).

    Docs files are anything matched by ``is_documentation_file``. The
    new pipeline uses this to collapse all docs into a single synthetic
    ``documentation`` feature instead of letting the heuristics carve
    them into per-tutorial buckets.

    >>> code, docs = partition_docs_vs_code([
    ...     "src/auth/login.ts",
    ...     "docs_src/tutorial001/main.py",
    ...     "README.md",
    ... ])
    >>> code
    ['src/auth/login.ts', 'README.md']
    >>> docs
    ['docs_src/tutorial001/main.py']
    """
    code_files: list[str] = []
    docs_files: list[str] = []
    for f in files:
        if is_documentation_file(f):
            docs_files.append(f)
        else:
            code_files.append(f)
    return code_files, docs_files
