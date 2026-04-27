"""File bucketizer — single classifier stage for the detection pipeline.

Every file in a repo is assigned to exactly one of five buckets:

  SOURCE          — real business code; fed to LLM feature detection
  DOCUMENTATION   — docs/, examples/, tutorials/, website/, etc.
                    collapsed into one synthetic ``documentation`` feature
  TESTS           — *.test.ts, __tests__/, e2e/, fixtures/ — skipped entirely
  INFRASTRUCTURE  — root configs, CI, helm charts, Docker, workspace manifests
                    collapsed into ``shared-infra`` feature
  GENERATED       — node_modules/, dist/, build/, .next/, lockfiles — skipped

Goals:
  - Single source of truth for file classification (replaces scattered
    filter calls in ``_clean_inputs``, candidate detector, workspace code).
  - Applied once at pipeline entry so downstream LLM calls never see
    non-source files.
  - Deterministic and cheap — no LLM, no I/O beyond path parsing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath

from .validation import is_documentation_file, is_test_file


class Bucket(str, Enum):
    """Five mutually-exclusive categories every file maps to."""

    SOURCE = "source"
    DOCUMENTATION = "documentation"
    TESTS = "tests"
    INFRASTRUCTURE = "infrastructure"
    GENERATED = "generated"


# Generated / vendored directory names. If ANY segment matches, file is
# treated as generated and never reaches LLM.
_GENERATED_DIR_SEGMENTS = frozenset({
    "node_modules", "dist", "build", "out", ".next", ".nuxt",
    ".turbo", ".cache", "coverage", ".coverage", "htmlcov",
    "__pycache__", ".pytest_cache", ".venv", "venv", "env",
    "target",  # Rust/Java build output
    ".gradle", ".idea", ".vscode",
    ".parcel-cache", ".svelte-kit", ".vercel",
    "vendor",  # Go/PHP vendored deps
    "bower_components",
})


# Filename patterns for generated/lock files.
_GENERATED_FILENAME_EXACT = frozenset({
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "bun.lockb", "composer.lock", "poetry.lock", "pipfile.lock",
    "cargo.lock", "gemfile.lock", "go.sum",
})

_GENERATED_FILENAME_SUFFIXES = (
    ".min.js", ".min.css", ".min.map",
    ".generated.ts", ".generated.js", ".gen.go",
    "_pb.go", "_pb.py", "_pb2.py",  # protobuf
)


# Infrastructure directory segments. Covers CI/CD, IaC, deployment,
# monorepo tooling. Matched at any depth so `packages/foo/.github` also
# counts as infra.
_INFRA_DIR_SEGMENTS = frozenset({
    ".github", ".gitlab", ".circleci", ".buildkite", ".woodpecker",
    "helm", "charts", "k8s", "kubernetes", "manifests",
    "terraform", "pulumi", "ansible", "puppet",
    "docker", ".devcontainer", ".gitpod",
    ".changeset", ".husky",
})


# Root-level config files. Only match when at depth 0 (repo root) —
# an `eslintrc` inside `apps/web` belongs to that package, not infra.
_ROOT_CONFIG_FILENAMES = frozenset({
    "package.json", "pnpm-workspace.yaml", "turbo.json", "nx.json",
    "lerna.json", "rush.json", "go.work", "go.work.sum",
    "cargo.toml", "cargo.lock",  # lock also treated as generated; dedup below
    "pyproject.toml", "setup.py", "setup.cfg", "pipfile", "poetry.lock",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "docker-compose.dev.yml", "docker-compose.prod.yml",
    ".dockerignore", ".gitignore", ".gitattributes",
    ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
    ".eslintignore", ".prettierrc", ".prettierrc.js", ".prettierignore",
    ".editorconfig", ".nvmrc", ".npmrc", ".yarnrc", ".yarnrc.yml",
    "tsconfig.json", "tsconfig.base.json", "jsconfig.json",
    "vite.config.ts", "vite.config.js", "vitest.workspace.ts",
    "rollup.config.js", "webpack.config.js", "esbuild.config.mjs",
    "makefile", "justfile", "taskfile.yml",
    "renovate.json", "dependabot.yml", "codecov.yml",
    "lefthook.yml", "commitlint.config.js",
    "sonar-project.properties", ".vercelignore", "vercel.json",
    "netlify.toml", "railway.json",
    "openapi.yml", "openapi.yaml", "openapi.json",
    "readme.md", "license", "license.md", "license.txt",
    "contributing.md", "code_of_conduct.md", "security.md",
    "changelog.md", "authors.md", "agents.md",
    "playwright.config.ts", "playwright.service.config.ts",
    "jest.config.js", "jest.config.ts",
})


# Filename suffixes that indicate config regardless of location.
_CONFIG_FILENAME_SUFFIXES = (
    ".dockerignore", ".gitignore", ".editorconfig", ".nvmrc",
)


def classify_file(path: str) -> Bucket:
    """Assign a single file path to exactly one bucket.

    Rules applied in order (first match wins):
      1. GENERATED   — matches vendored dirs or known lock/minified files
      2. TESTS       — reuses ``is_test_file`` from validation
      3. DOCUMENTATION — reuses ``is_documentation_file`` from validation
      4. INFRASTRUCTURE — matches CI/IaC dirs or root config files
      5. SOURCE      — everything else

    >>> classify_file("apps/web/lib/utils/billing.ts").value
    'source'
    >>> classify_file("docs/images/survey.webp").value
    'documentation'
    >>> classify_file("packages/auth/__tests__/login.test.ts").value
    'tests'
    >>> classify_file("node_modules/react/index.js").value
    'generated'
    >>> classify_file(".github/workflows/ci.yml").value
    'infrastructure'
    >>> classify_file("package.json").value
    'infrastructure'
    >>> classify_file("apps/web/package.json").value
    'source'
    """
    p = PurePosixPath(path)
    parts = p.parts
    filename_lower = p.name.lower()
    segments_lower = [s.lower() for s in parts[:-1]]

    # 1. Generated
    if any(seg in _GENERATED_DIR_SEGMENTS for seg in segments_lower):
        return Bucket.GENERATED
    if filename_lower in _GENERATED_FILENAME_EXACT:
        return Bucket.GENERATED
    if any(filename_lower.endswith(suf) for suf in _GENERATED_FILENAME_SUFFIXES):
        return Bucket.GENERATED

    # 2. Tests (reuse existing validation helper)
    if is_test_file(path):
        return Bucket.TESTS

    # 3. Documentation (reuse existing validation helper)
    if is_documentation_file(path):
        return Bucket.DOCUMENTATION

    # 4. Infrastructure
    if any(seg in _INFRA_DIR_SEGMENTS for seg in segments_lower):
        return Bucket.INFRASTRUCTURE
    # Root-level config: depth 0 AND name in known list
    if len(parts) == 1 and filename_lower in _ROOT_CONFIG_FILENAMES:
        return Bucket.INFRASTRUCTURE

    # 5. Default: source
    return Bucket.SOURCE


def partition_files(files: list[str]) -> dict[Bucket, list[str]]:
    """Partition a flat file list into all five buckets.

    Output dict always contains every Bucket key, even if empty, so
    callers can iterate without KeyError.

    >>> result = partition_files([
    ...     "src/auth.ts",
    ...     "docs/intro.md",
    ...     "src/auth.test.ts",
    ...     "node_modules/x.js",
    ...     ".github/workflows/ci.yml",
    ... ])
    >>> [len(result[b]) for b in Bucket]
    [1, 1, 1, 1, 1]
    """
    out: dict[Bucket, list[str]] = {b: [] for b in Bucket}
    for f in files:
        out[classify_file(f)].append(f)
    return out


def bucket_summary(partition: dict[Bucket, list[str]]) -> dict[str, int]:
    """Human-readable counts for logging."""
    return {b.value: len(paths) for b, paths in partition.items()}
