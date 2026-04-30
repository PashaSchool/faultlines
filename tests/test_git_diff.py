"""Tests for the git-diff utility (Stage 2 of incremental scan).

Use a real ephemeral git repo so we exercise the actual subprocess
path. Avoiding mocks here is intentional — git's CLI is the contract.
"""

from __future__ import annotations

import subprocess

import pytest

from faultline.analyzer.git_diff import compute_git_diff


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    )


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _commit(repo, msg, **files):
    for relpath, content in files.items():
        p = repo / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_no_base_sha_falls_back(git_repo):
    diff = compute_git_diff(git_repo, base_sha=None)
    assert diff.fallback_full_scan is True
    assert diff.is_empty


def test_unknown_base_sha_falls_back(git_repo):
    _commit(git_repo, "init", **{"a.ts": "x"})
    diff = compute_git_diff(git_repo, base_sha="0" * 40)
    assert diff.fallback_full_scan is True


def test_not_git_repo_falls_back(tmp_path):
    diff = compute_git_diff(tmp_path / "notgit", base_sha="abc")
    assert diff.fallback_full_scan is True


def test_added_modified_deleted(git_repo):
    base = _commit(git_repo, "init", **{
        "src/a.ts": "a", "src/b.ts": "b", "src/c.ts": "c",
    })
    _commit(git_repo, "change", **{
        "src/a.ts": "a-modified",       # modified
        "src/d.ts": "d",                # added
    })
    # Delete c.ts
    _git(git_repo, "rm", "-q", "src/c.ts")
    _git(git_repo, "commit", "-q", "-m", "delete c")

    diff = compute_git_diff(git_repo, base_sha=base)
    assert diff.fallback_full_scan is False
    assert diff.added == {"src/d.ts"}
    assert diff.modified == {"src/a.ts"}
    assert diff.deleted == {"src/c.ts"}
    assert diff.changed_paths == {"src/a.ts", "src/d.ts"}


def test_uncommitted_changes_included(git_repo):
    base = _commit(git_repo, "init", **{"src/a.ts": "a"})
    # Uncommitted edit on top of HEAD.
    (git_repo / "src" / "a.ts").write_text("a-uncommitted", encoding="utf-8")
    diff = compute_git_diff(git_repo, base_sha=base)
    assert "src/a.ts" in diff.modified


def test_empty_diff_returns_no_changes(git_repo):
    base = _commit(git_repo, "init", **{"src/a.ts": "a"})
    diff = compute_git_diff(git_repo, base_sha=base)
    assert diff.fallback_full_scan is False
    assert diff.is_empty


def test_renames_attributed_as_modified_plus_deleted(git_repo):
    base = _commit(git_repo, "init", **{"src/old.ts": "x"})
    _git(git_repo, "mv", "src/old.ts", "src/new.ts")
    _git(git_repo, "commit", "-q", "-m", "rename")
    diff = compute_git_diff(git_repo, base_sha=base)
    # Either git records as rename or as add+delete depending on
    # similarity threshold; both shapes must yield the right
    # incremental decision: new path gets re-analyzed, old path
    # leaves carry-forward.
    assert "src/old.ts" in diff.deleted
    assert "src/new.ts" in (diff.added | diff.modified)
