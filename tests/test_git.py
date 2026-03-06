"""Unit tests for faultline.analyzer.git module."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import pytest

from faultline.analyzer.git import (
    BUG_FIX_REGEX,
    _FALSE_POSITIVE_REGEX,
    estimate_duration,
    extract_pr_number,
    get_remote_url,
    get_tracked_files,
    is_bug_fix,
    load_repo,
)


# ---------------------------------------------------------------------------
# is_bug_fix
# ---------------------------------------------------------------------------


class TestIsBugFix:
    """Tests for the is_bug_fix function."""

    @pytest.mark.parametrize(
        "message",
        [
            "fix: login redirect loop",
            "Fix crash on empty cart",
            "hotfix: payment timeout",
            "bug: null pointer in user profile",
            "revert: broken auth middleware",
            "fix regression in search results",
            "patch: overflow in pagination",
            "resolve deadlock in queue worker",
            "fix memory leak in websocket handler",
            "handle race condition in checkout",
            "fix NaN in price calculation",
            "fix undefined variable in template",
            "fix timeout on large file upload",
            "fix null pointer exception",
            "fix error in data migration",
            "defect: missing validation on email field",
            "fix: broken image upload",
            "fix issue with date parsing",
        ],
    )
    def test_real_bug_fixes_detected(self, message: str) -> None:
        assert is_bug_fix(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "fix typo in README",
            "fix lint errors",
            "fix formatting issues",
            "fix test assertions",
            "fix docs for API endpoint",
            "fix import order",
            "fix indent in config",
            "fix spacing in template",
            "fix whitespace",
            "fix merge conflicts",
            "fix rebase issues",
            "fix ci pipeline",
            "fix build script",
            "fix deploy configuration",
            "fix style issues in header",
            "fix snapshot tests",
            "fix mock setup",
            "fix spec file imports",
            "fix comment formatting",
            "fix readme links",
            "fix changelog entry",
        ],
    )
    def test_false_positives_excluded(self, message: str) -> None:
        assert is_bug_fix(message) is False

    @pytest.mark.parametrize(
        "message",
        [
            "feat: add user dashboard",
            "refactor: extract payment service",
            "chore: update dependencies",
            "docs: add API documentation",
            "style: run prettier",
            "perf: optimize database queries",
            "test: add unit tests for auth",
            "ci: add GitHub Actions workflow",
            "add new search feature",
            "update README",
            "remove unused imports",
        ],
    )
    def test_non_bug_messages_not_detected(self, message: str) -> None:
        assert is_bug_fix(message) is False

    def test_empty_message(self) -> None:
        assert is_bug_fix("") is False

    def test_case_insensitive(self) -> None:
        assert is_bug_fix("FIX: crash on startup") is True
        assert is_bug_fix("HOTFIX: critical auth bug") is True
        assert is_bug_fix("BUG: memory leak") is True


# ---------------------------------------------------------------------------
# BUG_FIX_REGEX edge cases
# ---------------------------------------------------------------------------


class TestBugFixRegex:
    """Direct tests for BUG_FIX_REGEX pattern matching."""

    def test_word_boundary_prevents_partial_match(self) -> None:
        # "prefix" contains "fix" but should not match due to word boundary
        assert BUG_FIX_REGEX.search("prefix something") is None
        assert BUG_FIX_REGEX.search("suffix handling") is None

    def test_fix_at_word_boundary(self) -> None:
        assert BUG_FIX_REGEX.search("fix something") is not None
        assert BUG_FIX_REGEX.search("a fix for the issue") is not None

    def test_compound_patterns(self) -> None:
        assert BUG_FIX_REGEX.search("null pointer dereference") is not None
        assert BUG_FIX_REGEX.search("null check missing") is not None
        assert BUG_FIX_REGEX.search("null ref error") is not None
        assert BUG_FIX_REGEX.search("race condition in handler") is not None
        assert BUG_FIX_REGEX.search("memory leak detected") is not None


# ---------------------------------------------------------------------------
# _FALSE_POSITIVE_REGEX edge cases
# ---------------------------------------------------------------------------


class TestFalsePositiveRegex:
    """Direct tests for _FALSE_POSITIVE_REGEX pattern matching."""

    def test_fix_typo_variants(self) -> None:
        assert _FALSE_POSITIVE_REGEX.search("fix typo") is not None
        assert _FALSE_POSITIVE_REGEX.search("Fix Typo in header") is not None

    def test_fix_format_variants(self) -> None:
        assert _FALSE_POSITIVE_REGEX.search("fix formatting") is not None
        assert _FALSE_POSITIVE_REGEX.search("fix format issues") is not None

    def test_real_fix_not_matched(self) -> None:
        assert _FALSE_POSITIVE_REGEX.search("fix crash on login") is None
        assert _FALSE_POSITIVE_REGEX.search("fix null pointer") is None

    def test_fix_doc_variants(self) -> None:
        assert _FALSE_POSITIVE_REGEX.search("fix doc typo") is not None
        assert _FALSE_POSITIVE_REGEX.search("fix docs links") is not None


# ---------------------------------------------------------------------------
# extract_pr_number
# ---------------------------------------------------------------------------


class TestExtractPrNumber:
    """Tests for the extract_pr_number function."""

    def test_merge_commit_style(self) -> None:
        msg = "Merge pull request #42 from user/feature-branch"
        assert extract_pr_number(msg) == 42

    def test_merge_commit_case_insensitive(self) -> None:
        msg = "merge pull request #99 from org/fix-auth"
        assert extract_pr_number(msg) == 99

    def test_squash_merge_style(self) -> None:
        msg = "feat: add user dashboard (#123)"
        assert extract_pr_number(msg) == 123

    def test_squash_merge_at_end_of_line(self) -> None:
        msg = "fix: resolve login issue (#456)\n\nSome description here."
        assert extract_pr_number(msg) == 456

    def test_no_pr_number(self) -> None:
        msg = "feat: add user dashboard"
        assert extract_pr_number(msg) is None

    def test_issue_reference_not_extracted(self) -> None:
        # "#123" in the middle of text without parens should not match squash
        # and without "Merge pull request" should not match merge
        msg = "Closes #123 in the auth module"
        assert extract_pr_number(msg) is None

    def test_empty_message(self) -> None:
        assert extract_pr_number("") is None

    def test_large_pr_number(self) -> None:
        msg = "Merge pull request #99999 from org/big-repo"
        assert extract_pr_number(msg) == 99999

    def test_squash_merge_multiline(self) -> None:
        msg = "refactor: clean up auth service (#78)\n\nBreaking change."
        assert extract_pr_number(msg) == 78

    def test_merge_preferred_over_squash(self) -> None:
        # If both patterns exist, merge pattern should match first
        msg = "Merge pull request #10 from branch (#20)"
        assert extract_pr_number(msg) == 10


# ---------------------------------------------------------------------------
# estimate_duration
# ---------------------------------------------------------------------------


class TestEstimateDuration:
    """Tests for the estimate_duration function."""

    def test_small_repo_no_llm(self) -> None:
        # 100 commits * 0.008 = 0.8 sec → < 10 sec
        result = estimate_duration(100)
        assert result == "< 10 sec"

    def test_zero_commits(self) -> None:
        assert estimate_duration(0) == "< 10 sec"

    def test_medium_repo_no_llm(self) -> None:
        # 5000 commits * 0.008 = 40 sec
        result = estimate_duration(5000)
        assert result == "~ 40 sec"

    def test_with_llm_adds_overhead(self) -> None:
        # 100 commits * 0.008 = 0.8 + 5 = 5.8 → < 10 sec
        result = estimate_duration(100, use_llm=True)
        assert result == "< 10 sec"

    def test_with_llm_medium_repo(self) -> None:
        # 5000 * 0.008 = 40 + 5 = 45 sec
        result = estimate_duration(5000, use_llm=True)
        assert result == "~ 45 sec"

    def test_with_flows(self) -> None:
        # 100 * 0.008 = 0.8 + 20*4 = 80.8 → ~ 1.3 min
        result = estimate_duration(100, use_flows=True)
        assert result == "~ 1.3 min"

    def test_with_llm_and_flows(self) -> None:
        # 100 * 0.008 = 0.8 + 5 + 80 = 85.8 → ~ 1.4 min
        result = estimate_duration(100, use_llm=True, use_flows=True)
        assert result == "~ 1.4 min"

    def test_large_repo_with_all_options(self) -> None:
        # 10000 * 0.008 = 80 + 5 + 80 = 165 → ~ 2.8 min
        result = estimate_duration(10000, use_llm=True, use_flows=True)
        assert result == "~ 2.8 min"

    def test_boundary_exactly_ten_seconds(self) -> None:
        # Need commit_count where seconds == 10 exactly: 10/0.008 = 1250
        result = estimate_duration(1250)
        assert result == "~ 10 sec"

    def test_boundary_exactly_sixty_seconds(self) -> None:
        # 60/0.008 = 7500 commits
        result = estimate_duration(7500)
        assert result == "~ 1.0 min"


# ---------------------------------------------------------------------------
# get_remote_url
# ---------------------------------------------------------------------------


class TestGetRemoteUrl:
    """Tests for the get_remote_url function using MagicMock."""

    def _make_repo_mock(self, url: str) -> MagicMock:
        repo = MagicMock()
        repo.remotes.origin.url = url
        return repo

    def test_https_url_unchanged(self) -> None:
        repo = self._make_repo_mock("https://github.com/org/repo")
        assert get_remote_url(repo) == "https://github.com/org/repo"

    def test_https_url_strips_dot_git(self) -> None:
        repo = self._make_repo_mock("https://github.com/org/repo.git")
        assert get_remote_url(repo) == "https://github.com/org/repo"

    def test_ssh_url_converted_to_https(self) -> None:
        repo = self._make_repo_mock("git@github.com:org/repo.git")
        assert get_remote_url(repo) == "https://github.com/org/repo"

    def test_ssh_url_without_dot_git(self) -> None:
        repo = self._make_repo_mock("git@github.com:org/repo")
        assert get_remote_url(repo) == "https://github.com/org/repo"

    def test_trailing_slash_stripped(self) -> None:
        repo = self._make_repo_mock("https://github.com/org/repo/")
        assert get_remote_url(repo) == "https://github.com/org/repo"

    def test_gitlab_ssh_url(self) -> None:
        repo = self._make_repo_mock("git@gitlab.com:org/repo.git")
        assert get_remote_url(repo) == "https://gitlab.com/org/repo"

    def test_no_remotes_returns_empty(self) -> None:
        repo = MagicMock()
        type(repo.remotes).origin = PropertyMock(
            side_effect=AttributeError("no origin")
        )
        assert get_remote_url(repo) == ""

    def test_no_origin_returns_empty(self) -> None:
        repo = MagicMock()
        repo.remotes.origin.url = PropertyMock(
            side_effect=Exception("no url")
        )
        # get_remote_url catches all exceptions
        assert get_remote_url(repo) == ""


# ---------------------------------------------------------------------------
# load_repo (requires tmp_path with real git repo)
# ---------------------------------------------------------------------------


class TestLoadRepo:
    """Tests for load_repo using a temporary git repository."""

    def _init_repo(self, path: str) -> None:
        """Initialize a git repo with one commit."""
        subprocess.run(
            ["git", "init"], cwd=path, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=path, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=path, capture_output=True, check=True,
        )

    def _add_commit(self, path: str, filename: str, content: str, message: str) -> None:
        filepath = f"{path}/{filename}"
        with open(filepath, "w") as f:
            f.write(content)
        subprocess.run(
            ["git", "add", filename], cwd=path, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=path, capture_output=True, check=True,
        )

    def test_load_valid_repo(self, tmp_path: str) -> None:
        path = str(tmp_path)
        self._init_repo(path)
        self._add_commit(path, "hello.txt", "hello", "initial commit")
        repo = load_repo(path)
        assert repo is not None

    def test_load_invalid_path_raises(self, tmp_path: str) -> None:
        path = str(tmp_path)
        with pytest.raises(ValueError, match="is not a git repository"):
            load_repo(path)

    def test_load_empty_repo_raises(self, tmp_path: str) -> None:
        path = str(tmp_path)
        self._init_repo(path)
        # Repo exists but has no commits
        with pytest.raises(ValueError, match="has no commits yet"):
            load_repo(path)


# ---------------------------------------------------------------------------
# get_tracked_files (requires tmp_path with real git repo)
# ---------------------------------------------------------------------------


class TestGetTrackedFiles:
    """Tests for get_tracked_files using a temporary git repository."""

    def _init_repo_with_files(
        self, path: str, files: dict[str, str],
    ) -> None:
        """Create a git repo with the given files committed."""
        subprocess.run(
            ["git", "init"], cwd=path, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=path, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=path, capture_output=True, check=True,
        )
        for filepath, content in files.items():
            full = f"{path}/{filepath}"
            # Ensure parent dirs exist
            subprocess.run(
                ["mkdir", "-p", str(full.rsplit("/", 1)[0])],
                capture_output=True,
            )
            with open(full, "w") as f:
                f.write(content)
        subprocess.run(
            ["git", "add", "-A"], cwd=path, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=path, capture_output=True, check=True,
        )

    def test_returns_tracked_source_files(self, tmp_path: str) -> None:
        from git import Repo

        path = str(tmp_path)
        self._init_repo_with_files(path, {
            "src/app.py": "print('hi')",
            "src/utils.py": "pass",
        })
        repo = Repo(path)
        files = get_tracked_files(repo)
        assert "src/app.py" in files
        assert "src/utils.py" in files

    def test_skips_binary_extensions(self, tmp_path: str) -> None:
        from git import Repo

        path = str(tmp_path)
        self._init_repo_with_files(path, {
            "src/app.py": "code",
            "assets/logo.png": "fake-png",
            "docs/guide.pdf": "fake-pdf",
        })
        repo = Repo(path)
        files = get_tracked_files(repo)
        assert "src/app.py" in files
        assert "assets/logo.png" not in files
        assert "docs/guide.pdf" not in files

    def test_skips_node_modules(self, tmp_path: str) -> None:
        from git import Repo

        path = str(tmp_path)
        self._init_repo_with_files(path, {
            "src/index.ts": "export {}",
            "node_modules/lodash/index.js": "module.exports = {}",
        })
        repo = Repo(path)
        files = get_tracked_files(repo)
        assert "src/index.ts" in files
        assert "node_modules/lodash/index.js" not in files

    def test_skips_lockfiles(self, tmp_path: str) -> None:
        from git import Repo

        path = str(tmp_path)
        self._init_repo_with_files(path, {
            "src/main.py": "pass",
            "package-lock.json": "{}",
            "yarn.lock": "",
        })
        repo = Repo(path)
        files = get_tracked_files(repo)
        assert "src/main.py" in files
        assert "package-lock.json" not in files
        assert "yarn.lock" not in files

    def test_src_filter(self, tmp_path: str) -> None:
        from git import Repo

        path = str(tmp_path)
        self._init_repo_with_files(path, {
            "src/app.py": "code",
            "scripts/deploy.sh": "#!/bin/bash",
            "config.py": "pass",
        })
        repo = Repo(path)
        files = get_tracked_files(repo, src="src")
        assert "src/app.py" in files
        assert "scripts/deploy.sh" not in files
        assert "config.py" not in files

    def test_empty_repo_returns_empty(self, tmp_path: str) -> None:
        from git import Repo

        path = str(tmp_path)
        # Init repo with a single file so we have a valid commit
        self._init_repo_with_files(path, {
            "node_modules/pkg/index.js": "module.exports = {}",
        })
        repo = Repo(path)
        files = get_tracked_files(repo)
        # All files are in skipped dirs
        assert files == []
