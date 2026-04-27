"""Unit tests for ``faultline.llm.tools``.

Day 1 of Sprint 1 (tool-use detection). No network, no Anthropic
client — these tests cover the dispatcher and four tool functions
in isolation against a fixture repository built per-test.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from faultline.llm.tools import (
    GREP_MAX_FILES_SCANNED,
    MAX_GREP_RESULTS,
    MAX_LIST_ENTRIES,
    MAX_READ_BYTES,
    MAX_READ_LINES,
    TOOL_SCHEMAS,
    dispatch_tool,
    get_file_commits,
    grep_pattern,
    list_directory,
    read_file_head,
)


# ── Fixtures ───────────────────────────────────────────────────────────


def _write(root: Path, rel: str, body: str = "") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Bare-bones source tree with a few files of each bucket type."""
    _write(tmp_path, "apps/web/lib/billing.ts", "export function chargeStripe() {\n  return 'ok';\n}\n")
    _write(tmp_path, "apps/web/lib/utils.ts", "import Stripe from 'stripe';\nexport const x = 1;\n")
    _write(tmp_path, "apps/web/lib/auth.ts", "export function login() {}\n")
    _write(tmp_path, "apps/web/lib/__tests__/billing.test.ts", "test('x', () => {});\n")
    _write(tmp_path, "docs/intro.md", "# docs\n")
    _write(tmp_path, "node_modules/foo/index.js", "module.exports = 1;\n")
    _write(tmp_path, "package.json", "{}\n")
    return tmp_path


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Real git repo with two commits touching one file."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True, env=env)
    _write(tmp_path, "src/billing.ts", "v1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "feat(billing): add stripe"], cwd=tmp_path, check=True, env=env)
    _write(tmp_path, "src/billing.ts", "v2\n")
    subprocess.run(["git", "commit", "-aq", "-m", "fix(billing): webhook race"], cwd=tmp_path, check=True, env=env)
    return tmp_path


# ── TOOL_SCHEMAS shape ─────────────────────────────────────────────────


class TestSchemas:
    def test_four_tools_exposed(self):
        names = {s["name"] for s in TOOL_SCHEMAS}
        assert names == {"read_file_head", "list_directory", "grep_pattern", "get_file_commits"}

    def test_each_schema_has_required_fields(self):
        for s in TOOL_SCHEMAS:
            assert "name" in s and "description" in s and "input_schema" in s
            assert s["input_schema"]["type"] == "object"
            assert "properties" in s["input_schema"]
            assert "required" in s["input_schema"]


# ── read_file_head ─────────────────────────────────────────────────────


class TestReadFileHead:
    def test_reads_full_short_file(self, repo: Path):
        out = read_file_head(repo, "apps/web/lib/billing.ts", lines=50)
        assert "chargeStripe" in out
        assert "1: export function chargeStripe()" in out

    def test_truncates_at_lines(self, repo: Path):
        body = "\n".join(f"line{i}" for i in range(200))
        _write(repo, "big.ts", body)
        out = read_file_head(repo, "big.ts", lines=10)
        assert "FILE TRUNCATED" in out
        assert "line9" in out
        assert "line10" not in out

    def test_caps_at_max_lines_even_if_caller_asks_more(self, repo: Path):
        body = "\n".join(f"L{i}" for i in range(500))
        _write(repo, "big.ts", body)
        out = read_file_head(repo, "big.ts", lines=10000)
        # Should cap at MAX_READ_LINES
        line_count = sum(1 for ln in out.splitlines() if ln.startswith(("1:", "2:", "9:", "99:", "100:")))
        assert "FILE TRUNCATED" in out

    def test_caps_at_max_bytes(self, repo: Path):
        body = "x" * (MAX_READ_BYTES + 1000)
        _write(repo, "huge.ts", body)
        out = read_file_head(repo, "huge.ts", lines=100)
        assert "FILE TRUNCATED" in out

    def test_missing_file_returns_error(self, repo: Path):
        out = read_file_head(repo, "nope.ts")
        assert out.startswith("ERROR:")
        assert "not found" in out

    def test_directory_returns_error(self, repo: Path):
        out = read_file_head(repo, "apps")
        assert out.startswith("ERROR:")

    def test_path_traversal_rejected(self, repo: Path):
        out = read_file_head(repo, "../../../etc/passwd")
        assert out.startswith("ERROR:")
        assert "escapes" in out

    def test_empty_path_rejected(self, repo: Path):
        out = read_file_head(repo, "")
        assert out.startswith("ERROR:")

    def test_empty_file(self, repo: Path):
        _write(repo, "empty.ts", "")
        out = read_file_head(repo, "empty.ts")
        assert "EMPTY" in out

    def test_lines_below_one_floored(self, repo: Path):
        out = read_file_head(repo, "apps/web/lib/billing.ts", lines=0)
        assert not out.startswith("ERROR:")


# ── list_directory ─────────────────────────────────────────────────────


class TestListDirectory:
    def test_lists_source_files(self, repo: Path):
        out = list_directory(repo, "apps/web/lib")
        assert "apps/web/lib/billing.ts" in out
        assert "apps/web/lib/utils.ts" in out
        assert "apps/web/lib/auth.ts" in out

    def test_excludes_tests_dirs(self, repo: Path):
        out = list_directory(repo, "apps/web/lib")
        # __tests__/ shows up as a directory but no test file leak
        assert "billing.test.ts" not in out

    def test_excludes_generated(self, repo: Path):
        out = list_directory(repo, ".")
        assert "node_modules" not in out

    def test_missing_dir(self, repo: Path):
        out = list_directory(repo, "nope")
        assert out.startswith("ERROR:")

    def test_file_not_dir(self, repo: Path):
        out = list_directory(repo, "apps/web/lib/billing.ts")
        assert out.startswith("ERROR:")

    def test_empty_path_means_root(self, repo: Path):
        out = list_directory(repo, "")
        assert "ERROR" not in out
        assert "apps/" in out

    def test_path_traversal_rejected(self, repo: Path):
        out = list_directory(repo, "../..")
        assert out.startswith("ERROR:")

    def test_caps_at_max_entries(self, repo: Path):
        for i in range(MAX_LIST_ENTRIES + 20):
            _write(repo, f"big/f{i}.ts", "x")
        out = list_directory(repo, "big")
        # Body shouldn't have all of them
        files_listed = sum(1 for ln in out.splitlines() if ln.startswith("big/"))
        assert files_listed <= MAX_LIST_ENTRIES


# ── grep_pattern ───────────────────────────────────────────────────────


class TestGrepPattern:
    def test_finds_matches(self, repo: Path):
        out = grep_pattern(repo, "Stripe")
        assert "apps/web/lib/utils.ts" in out
        assert "apps/web/lib/billing.ts" in out

    def test_case_insensitive(self, repo: Path):
        out = grep_pattern(repo, "stripe")
        assert "apps/web/lib/utils.ts" in out

    def test_no_matches(self, repo: Path):
        out = grep_pattern(repo, "ZZZNOMATCHZZZ")
        assert "NO MATCHES" in out

    def test_invalid_regex(self, repo: Path):
        out = grep_pattern(repo, "[unclosed")
        assert out.startswith("ERROR:")

    def test_empty_pattern(self, repo: Path):
        out = grep_pattern(repo, "")
        assert out.startswith("ERROR:")

    def test_path_glob_limits_scope(self, repo: Path):
        _write(repo, "other/foo.ts", "Stripe is here\n")
        out = grep_pattern(repo, "Stripe", path_glob="apps")
        assert "apps/web/lib" in out
        assert "other/foo.ts" not in out

    def test_skips_non_source(self, repo: Path):
        _write(repo, "node_modules/x.ts", "Stripe found here\n")
        out = grep_pattern(repo, "Stripe")
        assert "node_modules" not in out

    def test_caps_at_max_results(self, repo: Path):
        for i in range(MAX_GREP_RESULTS + 20):
            _write(repo, f"many/f{i}.ts", "needle\n")
        out = grep_pattern(repo, "needle")
        match_lines = [ln for ln in out.splitlines() if ln.startswith("many/")]
        assert len(match_lines) <= MAX_GREP_RESULTS

    def test_traversal_rejected(self, repo: Path):
        out = grep_pattern(repo, "x", path_glob="../..")
        assert out.startswith("ERROR:")


# ── get_file_commits ───────────────────────────────────────────────────


class TestGetFileCommits:
    def test_returns_commit_subjects(self, git_repo: Path):
        out = get_file_commits(git_repo, "src/billing.ts", limit=5)
        assert "feat(billing): add stripe" in out
        assert "fix(billing): webhook race" in out

    def test_limit_applied(self, git_repo: Path):
        out = get_file_commits(git_repo, "src/billing.ts", limit=1)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        assert len(lines) == 1
        # most recent first
        assert "webhook race" in lines[0]

    def test_missing_file(self, git_repo: Path):
        out = get_file_commits(git_repo, "nope.ts")
        assert out.startswith("ERROR:")

    def test_not_a_git_repo(self, tmp_path: Path):
        _write(tmp_path, "x.ts", "y")
        out = get_file_commits(tmp_path, "x.ts")
        assert out.startswith("ERROR:")
        assert "git repository" in out

    def test_traversal_rejected(self, git_repo: Path):
        out = get_file_commits(git_repo, "../../etc/passwd")
        assert out.startswith("ERROR:")

    def test_untracked_file_returns_no_commits(self, git_repo: Path):
        _write(git_repo, "src/new.ts", "x")
        out = get_file_commits(git_repo, "src/new.ts")
        assert "NO COMMITS" in out or out.startswith("ERROR:")


# ── dispatch_tool ──────────────────────────────────────────────────────


class TestDispatch:
    def test_unknown_tool(self, repo: Path):
        out = dispatch_tool("nope", {}, repo)
        assert out.startswith("ERROR:")
        assert "unknown tool" in out

    def test_routes_read(self, repo: Path):
        out = dispatch_tool("read_file_head", {"path": "apps/web/lib/billing.ts"}, repo)
        assert "chargeStripe" in out

    def test_routes_list(self, repo: Path):
        out = dispatch_tool("list_directory", {"dirpath": "apps/web/lib"}, repo)
        assert "billing.ts" in out

    def test_routes_grep(self, repo: Path):
        out = dispatch_tool("grep_pattern", {"pattern": "Stripe"}, repo)
        assert "Stripe" in out or "stripe" in out

    def test_non_dict_input(self, repo: Path):
        out = dispatch_tool("read_file_head", "not a dict", repo)  # type: ignore[arg-type]
        assert out.startswith("ERROR:")

    def test_missing_required_arg(self, repo: Path):
        out = dispatch_tool("read_file_head", {}, repo)
        assert out.startswith("ERROR:")
