"""Tests for faultline/cache/."""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from faultline.cache.freshness import check_freshness
from faultline.cache.hashing import (
    changed_files,
    compute_symbol_hashes,
    hash_file,
    hash_files,
    hash_symbol_body,
)
from faultline.cache.refresh import refresh_feature_map
from faultline.models.types import Feature, FeatureMap, SymbolRange


def _make_repo(tmp_path: Path) -> Path:
    """Init a minimal git repo with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "a.py").write_text("print('a')\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def _feature_map(repo_path: str, sha: str, file_hashes: dict[str, str]) -> FeatureMap:
    return FeatureMap(
        repo_path=repo_path,
        analyzed_at=datetime.now(tz=timezone.utc) - timedelta(days=1),
        total_commits=1,
        date_range_days=30,
        features=[
            Feature(
                name="core",
                paths=["a.py"],
                authors=["T"],
                total_commits=1,
                bug_fixes=0,
                bug_fix_ratio=0.0,
                last_modified=datetime.now(tz=timezone.utc) - timedelta(days=1),
                health_score=100.0,
            ),
        ],
        last_scanned_sha=sha,
        file_hashes=file_hashes,
    )


class TestHashing:
    def test_hash_file_returns_content_digest(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hello")
        h1 = hash_file(f)
        assert h1 is not None and len(h1) == 64  # sha256 hex

        f.write_text("hello world")
        h2 = hash_file(f)
        assert h2 != h1

    def test_hash_file_missing(self, tmp_path: Path) -> None:
        assert hash_file(tmp_path / "nope.txt") is None

    def test_hash_files_batch(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.txt").write_text("B")
        result = hash_files(["a.txt", "b.txt", "missing.txt"], str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result
        assert "missing.txt" not in result
        assert result["a.txt"] != result["b.txt"]

    def test_changed_files_detects_all_cases(self) -> None:
        old = {"a.ts": "h1", "b.ts": "h2", "c.ts": "h3"}
        new = {"a.ts": "h1", "b.ts": "NEW", "d.ts": "h4"}
        modified, added, removed = changed_files(old, new)
        assert modified == {"b.ts"}
        assert added == {"d.ts"}
        assert removed == {"c.ts"}

    def test_symbol_body_hash(self) -> None:
        source = "line 1\nline 2\nline 3\nline 4\n"
        sym = SymbolRange(name="f", start_line=2, end_line=3, kind="function")
        h1 = hash_symbol_body(source, sym)
        h2 = hash_symbol_body(source.replace("line 3", "line X"), sym)
        assert h1 != h2

    def test_compute_symbol_hashes(self) -> None:
        source = "def a():\n    pass\n\ndef b():\n    return 1\n"
        syms = [
            SymbolRange(name="a", start_line=1, end_line=2, kind="function"),
            SymbolRange(name="b", start_line=4, end_line=5, kind="function"),
        ]
        result = compute_symbol_hashes("x.py", source, syms)
        assert set(result.keys()) == {"a", "b"}
        assert result["a"] != result["b"]


class TestFreshness:
    def test_detects_no_staleness_when_sha_matches(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        fm = _feature_map(str(repo), head, {})
        report = check_freshness(fm, str(repo))
        assert report.is_stale is False
        assert report.commits_behind == 0

    def test_detects_staleness_after_new_commit(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        old_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

        (repo / "b.py").write_text("print('b')\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "add b"], cwd=repo, check=True)

        fm = _feature_map(str(repo), old_head, {})
        report = check_freshness(fm, str(repo))
        assert report.is_stale is True
        assert report.commits_behind == 1
        assert report.has_new_files is True

    def test_handles_missing_sha_gracefully(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        fm = _feature_map(str(repo), "", {})
        report = check_freshness(fm, str(repo))
        assert report.is_stale is False


class TestRefresh:
    def test_no_op_when_fresh(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        fm = _feature_map(str(repo), head, {"a.py": "old_hash"})
        result = refresh_feature_map(fm, str(repo), fetch_commits_fn=lambda _: [])
        assert result.freshness_before.is_stale is False
        assert result.files_truly_modified == 0

    def test_incremental_refresh_updates_sha_and_hashes(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        old_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

        # Simulate a modified file in the new commit
        (repo / "a.py").write_text("print('a changed')\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "change a"], cwd=repo, check=True)

        fm = _feature_map(str(repo), old_head, {"a.py": "stale_hash"})

        result = refresh_feature_map(fm, str(repo), fetch_commits_fn=lambda _: [])
        assert result.updated_map.last_scanned_sha != old_head
        assert "a.py" in result.updated_map.file_hashes
        assert result.updated_map.file_hashes["a.py"] != "stale_hash"
