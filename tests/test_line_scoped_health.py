"""Tests for line-scoped health computation (Sprint 2 Day 8)."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from faultline.analyzer.blame_index import BlameIndex
from faultline.analyzer.features import _compute_line_scoped_health, build_feature_map
from faultline.models.types import Commit, SymbolAttribution


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
    _git(repo, "config", "user.email", "t@t.com")
    _git(repo, "config", "user.name", "T")
    return repo


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for path, content in files.items():
        fp = repo / path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _commit_obj(sha: str, message: str, files: list[str]) -> Commit:
    is_fix = any(
        message.lower().startswith(p)
        for p in ("fix", "bug", "hotfix", "patch")
    ) or any(
        kw in message.lower()
        for kw in ("fix:", "bug:", "fixed", "fixes ")
    )
    return Commit(
        sha=sha,
        author="test",
        date=datetime.now(tz=timezone.utc),
        message=message,
        files_changed=files,
        is_bug_fix=is_fix,
        insertions=0,
        deletions=0,
    )


class TestComputeLineScopedHealth:
    def test_returns_none_without_attributions(self):
        result = _compute_line_scoped_health("feat", None, [], None)
        assert result is None

    def test_returns_none_without_blame_index(self):
        attr = SymbolAttribution(
            file_path="x.py", symbols=["foo"], line_ranges=[(1, 10)],
            attributed_lines=10, total_file_lines=20,
        )
        result = _compute_line_scoped_health("feat", [attr], [], None)
        assert result is None

    def test_zero_fixes_when_only_feat_commits_touch_range(self, tmp_path):
        # C1 (feat): create utils.ts with formatDate at lines 1-10
        # C2 (fix): change parseDate at lines 11-20
        # Symbol attribution: formatDate at 1-10
        # Expected: only C1 in scoped commits → 0% bug ratio → high health
        repo = _make_repo(tmp_path)
        v1 = "\n".join(["fmt line"] * 10 + ["parse line"] * 10) + "\n"
        sha1 = _commit(repo, {"utils.ts": v1}, "feat: initial utils")
        v2 = "\n".join(
            ["fmt line"] * 10 + ["parse line CHANGED"] * 10
        ) + "\n"
        sha2 = _commit(repo, {"utils.ts": v2}, "fix: parseDate")

        commits = [
            _commit_obj(sha1, "feat: initial utils", ["utils.ts"]),
            _commit_obj(sha2, "fix: parseDate", ["utils.ts"]),
        ]
        with BlameIndex(repo) as idx:
            idx.index_file("utils.ts")
            attr = SymbolAttribution(
                file_path="utils.ts", symbols=["formatDate"],
                line_ranges=[(1, 10)], attributed_lines=10, total_file_lines=20,
            )
            result = _compute_line_scoped_health("feat", [attr], commits, idx)
        assert result is not None
        health, scoped, fixes = result
        assert scoped == 1
        assert fixes == 0
        # 0% fix ratio → high health, but activity_factor scales down for
        # low commit counts (1 commit gets ~80, not full 100).
        assert health > 70

    def test_high_bug_ratio_when_all_commits_to_range_are_fixes(self, tmp_path):
        repo = _make_repo(tmp_path)
        v1 = "x\n" * 5
        sha1 = _commit(repo, {"a.py": v1}, "fix: bug 1")
        v2 = "y\n" * 5
        sha2 = _commit(repo, {"a.py": v2}, "fix: bug 2")
        v3 = "z\n" * 5
        sha3 = _commit(repo, {"a.py": v3}, "fix: bug 3")

        commits = [
            _commit_obj(sha1, "fix: bug 1", ["a.py"]),
            _commit_obj(sha2, "fix: bug 2", ["a.py"]),
            _commit_obj(sha3, "fix: bug 3", ["a.py"]),
        ]
        with BlameIndex(repo) as idx:
            idx.index_file("a.py")
            attr = SymbolAttribution(
                file_path="a.py", symbols=["fn"], line_ranges=[(1, 5)],
                attributed_lines=5, total_file_lines=5,
            )
            result = _compute_line_scoped_health("feat", [attr], commits, idx)
        assert result is not None
        health, scoped, fixes = result
        # Latest blame shows only sha3 (rewrites overwrite), so 1 commit, 1 fix
        assert scoped >= 1
        assert fixes == scoped  # all are fixes
        assert health < 30  # high debt — every visible commit is a fix

    def test_diverges_from_file_level(self, tmp_path):
        """The whole point — symbol-scoped ratio differs from file-level."""
        repo = _make_repo(tmp_path)
        # Lines 1-10: clean code (only created)
        # Lines 11-20: bug-prone (multiple fixes)
        v1 = "clean\n" * 10 + "buggy initial\n" * 10
        sha1 = _commit(repo, {"a.py": v1}, "feat: initial")
        v2 = "clean\n" * 10 + "buggy v2\n" * 10
        sha2 = _commit(repo, {"a.py": v2}, "fix: rewrite buggy block")
        v3 = "clean\n" * 10 + "buggy v3\n" * 10
        sha3 = _commit(repo, {"a.py": v3}, "fix: another bug in block")

        commits = [
            _commit_obj(sha1, "feat: initial", ["a.py"]),
            _commit_obj(sha2, "fix: rewrite buggy block", ["a.py"]),
            _commit_obj(sha3, "fix: another bug in block", ["a.py"]),
        ]
        # File-level: 2/3 commits are fixes → 67%
        # Symbol scoped to lines 1-10 (clean): 1/1 (only initial) → 0% bug ratio
        # Symbol scoped to lines 11-20 (buggy): commits include sha3 (fix) → high
        with BlameIndex(repo) as idx:
            idx.index_file("a.py")
            attr_clean = SymbolAttribution(
                file_path="a.py", symbols=["clean_fn"],
                line_ranges=[(1, 10)], attributed_lines=10, total_file_lines=20,
            )
            attr_buggy = SymbolAttribution(
                file_path="a.py", symbols=["buggy_fn"],
                line_ranges=[(11, 20)], attributed_lines=10, total_file_lines=20,
            )
            res_clean = _compute_line_scoped_health(
                "clean", [attr_clean], commits, idx,
            )
            res_buggy = _compute_line_scoped_health(
                "buggy", [attr_buggy], commits, idx,
            )
        assert res_clean is not None and res_buggy is not None
        # Clean range health > buggy range health
        assert res_clean[0] > res_buggy[0]
        # Bug fix counts diverge
        assert res_clean[2] == 0
        assert res_buggy[2] >= 1


class TestBuildFeatureMapWithBlameIndex:
    def test_symbol_health_uses_blame_when_available(self, tmp_path):
        repo = _make_repo(tmp_path)
        v1 = "x\n" * 20
        sha1 = _commit(repo, {"a.py": v1}, "feat: initial")
        v2 = "y\n" * 20
        sha2 = _commit(repo, {"a.py": v2}, "fix: critical bug")

        commits = [
            _commit_obj(sha1, "feat: initial", ["a.py"]),
            _commit_obj(sha2, "fix: critical bug", ["a.py"]),
        ]
        attributions = {
            "feat-x": [SymbolAttribution(
                file_path="a.py", symbols=["x"],
                line_ranges=[(1, 5)],  # only first 5 lines — touched by sha2 too
                attributed_lines=5, total_file_lines=20,
            )],
        }
        with BlameIndex(repo) as idx:
            idx.index_file("a.py")
            fm = build_feature_map(
                repo_path=str(repo),
                commits=commits,
                feature_paths={"feat-x": ["a.py"]},
                days=365,
                shared_attributions=attributions,
                blame_index=idx,
            )
        feat = next(f for f in fm.features if f.name == "feat-x")
        assert feat.symbol_health_score is not None
        # File-level fix ratio: 1/2 = 50% → file health around 55-65
        # Symbol scoped to lines 1-5 (also rewritten by sha2): blame shows sha2 only
        # So 1 commit, 1 fix → 100% ratio → very low symbol health
        assert feat.symbol_health_score < feat.health_score

    def test_falls_back_when_blame_index_unavailable(self, tmp_path):
        repo = _make_repo(tmp_path)
        v1 = "x\n" * 20
        sha1 = _commit(repo, {"a.py": v1}, "feat: x")
        commits = [_commit_obj(sha1, "feat: x", ["a.py"])]
        attributions = {
            "feat-x": [SymbolAttribution(
                file_path="a.py", symbols=["x"],
                line_ranges=[(1, 5)],
                attributed_lines=5, total_file_lines=20,
            )],
        }
        # No blame_index → should still produce a valid map (Tier 2 fallback)
        fm = build_feature_map(
            repo_path=str(repo),
            commits=commits,
            feature_paths={"feat-x": ["a.py"]},
            days=365,
            shared_attributions=attributions,
            blame_index=None,
        )
        feat = next(f for f in fm.features if f.name == "feat-x")
        # symbol_health_score still populated via Tier 2 fractional weighting
        assert feat.symbol_health_score is not None
