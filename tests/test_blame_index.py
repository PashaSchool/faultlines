"""Tests for analyzer.blame_index — line-level git blame cache."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from faultline.analyzer.blame_index import BlameIndex, _parse_porcelain


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _commit_file(repo: Path, path: str, content: str, msg: str) -> str:
    fp = repo / path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


class TestParsePorcelain:
    def test_simple_three_lines(self):
        # Each line has sha, then a content line prefixed with tab
        text = (
            "abcdef0123456789abcdef0123456789abcdef01 1 1 3\n"
            "author Alice\n"
            "filename utils.ts\n"
            "\tline one\n"
            "abcdef0123456789abcdef0123456789abcdef01 2 2\n"
            "\tline two\n"
            "abcdef0123456789abcdef0123456789abcdef01 3 3\n"
            "\tline three\n"
        )
        rows = _parse_porcelain(text)
        assert len(rows) == 3
        assert {r.line for r in rows} == {1, 2, 3}
        assert all(r.commit_sha == "abcdef0123456789abcdef0123456789abcdef01" for r in rows)

    def test_multiple_commits(self):
        sha_a = "a" * 40
        sha_b = "b" * 40
        text = (
            f"{sha_a} 1 1 1\n"
            "author Alice\n"
            "\tfirst\n"
            f"{sha_b} 2 2 1\n"
            "author Bob\n"
            "\tsecond\n"
        )
        rows = _parse_porcelain(text)
        assert len(rows) == 2
        line_to_sha = {r.line: r.commit_sha for r in rows}
        assert line_to_sha[1] == sha_a
        assert line_to_sha[2] == sha_b

    def test_empty_input(self):
        assert _parse_porcelain("") == []


class TestBlameIndexBasics:
    def test_index_single_file(self, tmp_path):
        repo = _make_repo(tmp_path)
        sha1 = _commit_file(repo, "src/a.py", "line1\nline2\nline3\n", "initial")
        with BlameIndex(repo) as idx:
            assert idx.index_file("src/a.py") is True
            assert idx.is_indexed("src/a.py") is True
            commits = idx.commits_touching_lines("src/a.py", 1, 3)
            assert commits == {sha1}

    def test_two_commits_split_lines(self, tmp_path):
        repo = _make_repo(tmp_path)
        sha1 = _commit_file(repo, "src/a.py", "line1\nline2\n", "initial")
        sha2 = _commit_file(repo, "src/a.py", "line1\nline2\nline3\n", "add line3")
        with BlameIndex(repo) as idx:
            assert idx.index_file("src/a.py") is True
            commits_line1_2 = idx.commits_touching_lines("src/a.py", 1, 2)
            commits_line3 = idx.commits_touching_lines("src/a.py", 3, 3)
            assert commits_line1_2 == {sha1}
            assert commits_line3 == {sha2}
            commits_all = idx.commits_touching_lines("src/a.py", 1, 3)
            assert commits_all == {sha1, sha2}

    def test_unknown_file_returns_none(self, tmp_path):
        repo = _make_repo(tmp_path)
        _commit_file(repo, "src/a.py", "x\n", "initial")
        with BlameIndex(repo) as idx:
            # Never indexed
            assert idx.commits_touching_lines("src/b.py", 1, 1) is None
            assert idx.is_indexed("src/b.py") is False

    def test_index_file_returns_false_for_untracked(self, tmp_path):
        repo = _make_repo(tmp_path)
        _commit_file(repo, "tracked.py", "x\n", "initial")
        # Create an untracked file on disk
        (repo / "untracked.py").write_text("y\n")
        with BlameIndex(repo) as idx:
            assert idx.index_file("untracked.py") is False


class TestCacheBehavior:
    def test_cache_skips_unchanged_file(self, tmp_path):
        repo = _make_repo(tmp_path)
        _commit_file(repo, "a.py", "x\n", "initial")
        with BlameIndex(repo) as idx:
            # First call: indexes
            stats = idx.index_files(["a.py"])
            assert stats.indexed == 1
            assert stats.cached == 0
            # Second call: should be cached
            stats2 = idx.index_files(["a.py"])
            assert stats2.indexed == 0
            assert stats2.cached == 1

    def test_cache_invalidates_on_new_commit(self, tmp_path):
        repo = _make_repo(tmp_path)
        _commit_file(repo, "a.py", "line1\n", "initial")
        with BlameIndex(repo) as idx:
            idx.index_file("a.py")
        # New commit changes the file
        _commit_file(repo, "a.py", "line1\nline2\n", "add line2")
        with BlameIndex(repo) as idx:
            stats = idx.index_files(["a.py"])
            # Should re-index since head_sha changed
            assert stats.indexed == 1

    def test_cache_persists_across_instances(self, tmp_path):
        repo = _make_repo(tmp_path)
        sha = _commit_file(repo, "a.py", "x\n", "initial")
        with BlameIndex(repo) as idx:
            idx.index_file("a.py")
        # New instance, same cache_dir
        with BlameIndex(repo) as idx2:
            assert idx2.is_indexed("a.py")
            assert idx2.commits_touching_lines("a.py", 1, 1) == {sha}


class TestIndexStats:
    def test_mixed_outcomes(self, tmp_path):
        repo = _make_repo(tmp_path)
        _commit_file(repo, "good.py", "x\n", "initial")
        with BlameIndex(repo) as idx:
            stats = idx.index_files(["good.py", "missing.py"])
            assert stats.indexed == 1
            assert stats.failed == 1
            assert any("missing.py" in f for f in stats.failures)
