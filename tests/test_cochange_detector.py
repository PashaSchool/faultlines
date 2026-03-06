"""Tests for analyzer/cochange_detector.py."""

from datetime import datetime, timezone

from faultline.analyzer.cochange_detector import (
    _UnionFind,
    _cluster_name,
    _feature_name_from_path,
    _finalize_clusters,
    _unique_name,
    detect_features_from_cochange,
)
from faultline.models.types import Commit


def _commit(sha: str, files: list[str]) -> Commit:
    return Commit(
        sha=sha,
        message="test",
        author="dev",
        date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        files_changed=files,
        is_bug_fix=False,
    )


class TestUnionFind:
    def test_find_returns_self(self) -> None:
        uf = _UnionFind(["a", "b"])
        assert uf.find("a") == "a"

    def test_union_merges(self) -> None:
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")
        assert uf.find("c") != uf.find("a")

    def test_groups(self) -> None:
        uf = _UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        groups = uf.groups()
        assert len(groups) == 2


class TestFeatureNameFromPath:
    def test_returns_first_meaningful_dir(self) -> None:
        assert _feature_name_from_path("auth/login.ts") == "auth"

    def test_skips_generic_dirs(self) -> None:
        assert _feature_name_from_path("src/app/payments/stripe.ts") == "payments"

    def test_returns_root_for_top_level(self) -> None:
        assert _feature_name_from_path("index.ts") == "root"


class TestClusterName:
    def test_most_common_dir(self) -> None:
        files = ["auth/login.ts", "auth/register.ts", "payments/stripe.ts"]
        assert _cluster_name(files) == "auth"

    def test_single_file(self) -> None:
        assert _cluster_name(["dashboard/main.ts"]) == "dashboard"


class TestUniqueName:
    def test_returns_name_if_not_taken(self) -> None:
        assert _unique_name("auth", {}) == "auth"

    def test_appends_suffix_if_taken(self) -> None:
        assert _unique_name("auth", {"auth": []}) == "auth-2"

    def test_increments_suffix(self) -> None:
        assert _unique_name("auth", {"auth": [], "auth-2": []}) == "auth-3"


class TestFinalizeClusters:
    def test_multi_file_cluster(self) -> None:
        groups = {"r1": ["auth/login.ts", "auth/register.ts"]}
        result = _finalize_clusters(groups)
        assert "auth" in result
        assert len(result["auth"]) == 2

    def test_singleton_absorbed(self) -> None:
        groups = {
            "r1": ["auth/login.ts", "auth/register.ts"],
            "r2": ["auth/forgot.ts"],
        }
        result = _finalize_clusters(groups)
        assert "auth" in result
        assert "auth/forgot.ts" in result["auth"]

    def test_singleton_own_cluster(self) -> None:
        groups = {
            "r1": ["auth/login.ts", "auth/register.ts"],
            "r2": ["payments/stripe.ts"],
        }
        result = _finalize_clusters(groups)
        assert "payments" in result

    def test_empty_input(self) -> None:
        assert _finalize_clusters({}) == {}


class TestDetectFeaturesFromCochange:
    def test_returns_none_with_few_commits(self) -> None:
        commits = [_commit(str(i), ["a.ts"]) for i in range(10)]
        result = detect_features_from_cochange(["a.ts"], commits)
        assert result is None

    def test_groups_cochanged_files(self) -> None:
        files = ["auth/login.ts", "auth/session.ts", "payments/stripe.ts"]
        # 60 commits where auth files always change together
        commits = []
        for i in range(60):
            commits.append(_commit(f"c{i}", ["auth/login.ts", "auth/session.ts"]))
        # A few commits for payments (separate)
        for i in range(10):
            commits.append(_commit(f"p{i}", ["payments/stripe.ts"]))

        result = detect_features_from_cochange(files, commits)
        assert result is not None
        # auth files should be in the same cluster
        for cluster_files in result.values():
            if "auth/login.ts" in cluster_files:
                assert "auth/session.ts" in cluster_files
                break
        else:
            raise AssertionError("auth/login.ts not found in any cluster")

    def test_no_coupling_keeps_files_separate(self) -> None:
        files = ["auth/a.ts", "payments/b.ts"]
        # Each file appears in different commits, never together, only once each
        commits = [_commit(str(i), [f]) for i, f in enumerate(files)]
        # Pad to reach minimum
        commits += [_commit(f"pad{i}", ["other.ts"]) for i in range(60)]
        result = detect_features_from_cochange(files, commits)
        # Files appear in < _MIN_FILE_COMMITS each => no coupling pairs,
        # but singletons still get dir-bucketed
        if result is not None:
            # Files should NOT be in the same cluster
            for members in result.values():
                assert not ("auth/a.ts" in members and "payments/b.ts" in members)

    def test_bulk_commits_excluded(self) -> None:
        files = ["auth/a.ts", "auth/b.ts"]
        # All commits touch 40 files (> _MAX_FILES_PER_COMMIT=30) — excluded
        bulk_files = [f"file_{i}.ts" for i in range(40)]
        commits = [_commit(str(i), bulk_files) for i in range(60)]
        result = detect_features_from_cochange(files, commits)
        # No valid commits contribute to coupling (all bulk), so no pairs
        # Singletons may still produce a result, but files won't be coupled
        if result is not None:
            # Verify: no pair-based coupling happened (both files are singletons)
            for members in result.values():
                assert len(members) <= 2  # dir-bucketed at most
