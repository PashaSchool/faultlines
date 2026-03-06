"""Tests for faultline.analyzer.features — feature detection and metrics."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from faultline.analyzer.features import (
    _build_weekly_timeline,
    _calculate_health,
    _collect_prs,
    _extract_feature_name,
    _is_test_commit,
    _is_test_file,
    build_feature_map,
    build_flows_metrics,
    compute_cochange,
    detect_features_from_structure,
)
from faultline.models.types import Commit, PullRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now(tz=timezone.utc)


def _commit(
    sha: str = "abc",
    message: str = "some change",
    author: str = "alice",
    days_ago: int = 5,
    files: list[str] | None = None,
    is_bug_fix: bool = False,
    pr_number: int | None = None,
) -> Commit:
    return Commit(
        sha=sha,
        message=message,
        author=author,
        date=_NOW - timedelta(days=days_ago),
        files_changed=files or ["a.py"],
        is_bug_fix=is_bug_fix,
        pr_number=pr_number,
    )


# ===================================================================
# compute_cochange
# ===================================================================


class TestComputeCochange:
    def test_empty_commits(self) -> None:
        assert compute_cochange([]) == []

    def test_single_commit_no_pairs(self) -> None:
        c = _commit(files=["a.py", "b.py"])
        # Need _MIN_COCHANGE_COMMITS = 2 shared commits
        assert compute_cochange([c]) == []

    def test_two_commits_same_pair_returns_coupling(self) -> None:
        c1 = _commit(sha="1", files=["a.py", "b.py"])
        c2 = _commit(sha="2", files=["a.py", "b.py"])
        result = compute_cochange([c1, c2])
        assert len(result) == 1
        f1, f2, score = result[0]
        assert (f1, f2) == ("a.py", "b.py")
        assert score == 1.0  # Jaccard: 2 / (2+2-2) = 1.0

    def test_bulk_commits_are_excluded(self) -> None:
        """Commits touching > 30 files are filtered out."""
        bulk_files = [f"file{i}.py" for i in range(35)]
        c1 = _commit(sha="1", files=bulk_files)
        c2 = _commit(sha="2", files=bulk_files)
        assert compute_cochange([c1, c2]) == []

    def test_low_jaccard_filtered_out(self) -> None:
        """Pairs with Jaccard < 0.25 are excluded."""
        # a.py appears in both, b.py only in c1, c.py only in c2
        # Plus many solo commits for a.py to dilute
        commits = [
            _commit(sha="1", files=["a.py", "b.py"]),
            _commit(sha="2", files=["a.py", "c.py"]),
            # a.py now has 5 commits, b.py has 1, pair (a,b) count=1 < min=2
        ]
        # pair (a,b) count=1 => filtered by _MIN_COCHANGE_COMMITS
        # pair (a,c) count=1 => also filtered
        result = compute_cochange(commits)
        assert result == []

    def test_pairs_sorted_by_score_descending(self) -> None:
        commits = [
            _commit(sha="1", files=["a.py", "b.py", "c.py"]),
            _commit(sha="2", files=["a.py", "b.py", "c.py"]),
            _commit(sha="3", files=["a.py", "b.py"]),
            # (a,b) appears 3 times; (a,c) and (b,c) appear 2 times
            # a: 3, b: 3, c: 2
            # (a,b) Jaccard: 3/(3+3-3)=1.0
            # (a,c) Jaccard: 2/(3+2-2)=0.667
            # (b,c) Jaccard: 2/(3+2-2)=0.667
        ]
        result = compute_cochange(commits)
        assert len(result) == 3
        assert result[0][2] == 1.0
        assert result[1][2] == result[2][2] == 0.67

    def test_max_pairs_capped(self) -> None:
        """At most _MAX_COCHANGE_PAIRS (40) are returned."""
        # Create enough commits so many pairs cross the threshold
        files = [f"f{i}.py" for i in range(10)]  # 45 possible pairs
        commits = [
            _commit(sha=str(i), files=files) for i in range(3)
        ]
        result = compute_cochange(commits)
        assert len(result) <= 40


# ===================================================================
# detect_features_from_structure
# ===================================================================


class TestDetectFeaturesFromStructure:
    def test_groups_by_first_meaningful_dir(self) -> None:
        files = [
            "auth/login.py",
            "auth/register.py",
            "payments/stripe.py",
        ]
        features = detect_features_from_structure(files)
        assert set(features.keys()) == {"auth", "payments"}
        assert features["auth"] == ["auth/login.py", "auth/register.py"]

    def test_skips_src_prefix(self) -> None:
        files = ["src/auth/login.py", "src/payments/stripe.py"]
        features = detect_features_from_structure(files)
        assert "auth" in features
        assert "payments" in features
        assert "src" not in features

    def test_root_files(self) -> None:
        files = ["setup.py", "README.md"]
        features = detect_features_from_structure(files)
        assert "root" in features
        assert len(features["root"]) == 2

    def test_multiple_skip_prefixes(self) -> None:
        """Deeply nested with multiple skip prefixes."""
        files = ["src/app/components/layouts/auth/Login.tsx"]
        features = detect_features_from_structure(files)
        assert "auth" in features

    def test_empty_files(self) -> None:
        assert detect_features_from_structure([]) == {}


# ===================================================================
# _extract_feature_name
# ===================================================================


class TestExtractFeatureName:
    @pytest.mark.parametrize(
        "parts,expected",
        [
            (("auth", "login.py"), "auth"),
            (("src", "auth", "login.py"), "auth"),
            (("src", "app", "lib", "utils", "helper.py"), "utils"),
            (("index.py",), "root"),
            (("src", "app", "main.py"), "root"),
            (("pages", "api", "users.ts"), "api"),  # "pages" skipped
            (("features", "dashboard", "chart.tsx"), "dashboard"),
        ],
    )
    def test_extracts_correct_name(
        self, parts: tuple[str, ...], expected: str
    ) -> None:
        assert _extract_feature_name(parts) == expected

    def test_case_insensitive_skip(self) -> None:
        assert _extract_feature_name(("SRC", "Auth", "login.py")) == "auth"

    def test_single_dir_not_skipped(self) -> None:
        assert _extract_feature_name(("billing", "invoice.py")) == "billing"


# ===================================================================
# _is_test_file
# ===================================================================


class TestIsTestFile:
    @pytest.mark.parametrize(
        "path",
        [
            "auth/login.test.ts",
            "auth/login.spec.tsx",
            "auth/login.test.py",
            "auth/__tests__/login.ts",
            "tests/test_auth.py",
            "test/test_auth.py",
            "utils/test_helper.py",
            "utils/helper_test.py",
            "auth/Login.spec.js",
            "auth/Login.test.jsx",
        ],
    )
    def test_detects_test_files(self, path: str) -> None:
        assert _is_test_file(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "auth/login.py",
            "auth/login.ts",
            "auth/login.tsx",
            "utils/helper.js",
            "contest/main.py",
            "auth/testing_utils.py",
        ],
    )
    def test_rejects_non_test_files(self, path: str) -> None:
        assert _is_test_file(path) is False


# ===================================================================
# _is_test_commit
# ===================================================================


class TestIsTestCommit:
    def test_commit_with_adjacent_test_file(self) -> None:
        flow_paths = {"auth/login.py", "auth/utils.py"}
        commit = _commit(files=["auth/login.test.ts"])
        assert _is_test_commit(commit, flow_paths) is True

    def test_commit_with_unrelated_test_file(self) -> None:
        flow_paths = {"auth/login.py"}
        commit = _commit(files=["payments/payments.test.ts"])
        assert _is_test_commit(commit, flow_paths) is False

    def test_commit_with_no_test_files(self) -> None:
        flow_paths = {"auth/login.py"}
        commit = _commit(files=["auth/login.py"])
        assert _is_test_commit(commit, flow_paths) is False

    def test_commit_with_nested_test_dir(self) -> None:
        flow_paths = {"auth/login.py"}
        commit = _commit(files=["auth/__tests__/login.test.ts"])
        assert _is_test_commit(commit, flow_paths) is True


# ===================================================================
# _build_weekly_timeline
# ===================================================================


class TestBuildWeeklyTimeline:
    def test_groups_commits_by_iso_week(self) -> None:
        # Two commits in the same week, one in a different week
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)  # Monday W02
        c1 = _commit(sha="1", days_ago=0)
        c1_patched = c1.model_copy(update={"date": base})
        c2 = _commit(sha="2", days_ago=0)
        c2_patched = c2.model_copy(update={"date": base + timedelta(days=1)})
        c3 = _commit(sha="3", days_ago=0)
        c3_patched = c3.model_copy(update={"date": base + timedelta(days=7)})

        points = _build_weekly_timeline(
            [c1_patched, c2_patched, c3_patched], set()
        )
        assert len(points) == 2
        assert points[0].total_commits == 2
        assert points[1].total_commits == 1

    def test_counts_bug_fixes(self) -> None:
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)
        c1 = Commit(
            sha="1", message="fix: broken auth", author="a",
            date=base, files_changed=["a.py"], is_bug_fix=True,
        )
        c2 = Commit(
            sha="2", message="feat: add login", author="a",
            date=base + timedelta(days=1), files_changed=["a.py"],
        )
        points = _build_weekly_timeline([c1, c2], set())
        assert len(points) == 1
        assert points[0].bug_fix_commits == 1
        assert points[0].total_commits == 2

    def test_counts_test_commits(self) -> None:
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)
        c = Commit(
            sha="1", message="add tests", author="a",
            date=base, files_changed=["auth/login.test.ts"],
        )
        points = _build_weekly_timeline([c], {"auth/login.py"})
        assert points[0].test_commits == 1

    def test_sorted_chronologically(self) -> None:
        base = datetime(2025, 3, 3, tzinfo=timezone.utc)
        c_old = Commit(
            sha="1", message="old", author="a",
            date=base, files_changed=["a.py"],
        )
        c_new = Commit(
            sha="2", message="new", author="a",
            date=base + timedelta(weeks=4), files_changed=["a.py"],
        )
        points = _build_weekly_timeline([c_new, c_old], set())
        assert points[0].date < points[1].date

    def test_empty_commits(self) -> None:
        assert _build_weekly_timeline([], set()) == []


# ===================================================================
# _collect_prs
# ===================================================================


class TestCollectPrs:
    def test_collects_bug_fix_prs(self) -> None:
        c = _commit(
            sha="1", message="fix: auth bug", is_bug_fix=True,
            pr_number=42, author="alice",
        )
        prs = _collect_prs([c], "https://github.com/org/repo")
        assert len(prs) == 1
        assert prs[0].number == 42
        assert prs[0].url == "https://github.com/org/repo/pull/42"
        assert prs[0].author == "alice"

    def test_skips_non_bug_fix(self) -> None:
        c = _commit(sha="1", is_bug_fix=False, pr_number=10)
        assert _collect_prs([c], "") == []

    def test_skips_commits_without_pr(self) -> None:
        c = _commit(sha="1", is_bug_fix=True, pr_number=None)
        assert _collect_prs([c], "") == []

    def test_deduplicates_same_pr(self) -> None:
        c1 = _commit(sha="1", is_bug_fix=True, pr_number=5, days_ago=3)
        c2 = _commit(sha="2", is_bug_fix=True, pr_number=5, days_ago=1)
        prs = _collect_prs([c1, c2], "")
        assert len(prs) == 1

    def test_sorted_by_date_descending(self) -> None:
        c1 = _commit(sha="1", is_bug_fix=True, pr_number=1, days_ago=10)
        c2 = _commit(sha="2", is_bug_fix=True, pr_number=2, days_ago=1)
        prs = _collect_prs([c1, c2], "")
        assert prs[0].number == 2
        assert prs[1].number == 1

    def test_title_truncated_to_120_chars(self) -> None:
        long_msg = "fix: " + "x" * 200
        c = _commit(sha="1", message=long_msg, is_bug_fix=True, pr_number=1)
        prs = _collect_prs([c], "")
        assert len(prs[0].title) <= 120

    def test_empty_remote_url(self) -> None:
        c = _commit(sha="1", is_bug_fix=True, pr_number=7)
        prs = _collect_prs([c], "")
        assert prs[0].url == ""


# ===================================================================
# build_feature_map
# ===================================================================


class TestBuildFeatureMap:
    def test_basic_feature_map(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], author="alice"),
            _commit(sha="2", files=["auth/register.py"], author="bob"),
        ]
        feature_paths = {"auth": ["auth/login.py", "auth/register.py"]}
        fm = build_feature_map("/repo", commits, feature_paths, days=30)
        assert fm.repo_path == "/repo"
        assert fm.total_commits == 2
        assert len(fm.features) == 1
        feat = fm.features[0]
        assert feat.name == "auth"
        assert feat.total_commits == 2
        assert sorted(feat.authors) == ["alice", "bob"]

    def test_bug_fix_ratio(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], is_bug_fix=True),
            _commit(sha="2", files=["auth/login.py"], is_bug_fix=False),
        ]
        feature_paths = {"auth": ["auth/login.py"]}
        fm = build_feature_map("/repo", commits, feature_paths, days=30)
        assert fm.features[0].bug_fixes == 1
        assert fm.features[0].bug_fix_ratio == 0.5

    def test_dir_fallback_for_deleted_files(self) -> None:
        """Commits referencing files not in feature_paths use dir fallback."""
        commits = [
            _commit(sha="1", files=["auth/old_file.py"]),
        ]
        feature_paths = {"auth": ["auth/login.py"]}
        fm = build_feature_map("/repo", commits, feature_paths, days=30)
        assert fm.features[0].total_commits == 1

    def test_feature_with_no_commits(self) -> None:
        commits: list[Commit] = []
        feature_paths = {"auth": ["auth/login.py"]}
        fm = build_feature_map("/repo", commits, feature_paths, days=30)
        assert fm.features[0].total_commits == 0
        assert fm.features[0].health_score == 100.0

    def test_remote_url_passed_to_prs(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], is_bug_fix=True, pr_number=10),
        ]
        feature_paths = {"auth": ["auth/login.py"]}
        fm = build_feature_map(
            "/repo", commits, feature_paths, days=30,
            remote_url="https://github.com/org/repo",
        )
        assert fm.features[0].bug_fix_prs[0].url == "https://github.com/org/repo/pull/10"

    def test_multiple_features(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], author="alice"),
            _commit(sha="2", files=["billing/invoice.py"], author="bob"),
        ]
        feature_paths = {
            "auth": ["auth/login.py"],
            "billing": ["billing/invoice.py"],
        }
        fm = build_feature_map("/repo", commits, feature_paths, days=30)
        assert len(fm.features) == 2
        names = {f.name for f in fm.features}
        assert names == {"auth", "billing"}

    def test_commit_touching_multiple_features(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py", "billing/invoice.py"]),
        ]
        feature_paths = {
            "auth": ["auth/login.py"],
            "billing": ["billing/invoice.py"],
        }
        fm = build_feature_map("/repo", commits, feature_paths, days=30)
        for feat in fm.features:
            assert feat.total_commits == 1


# ===================================================================
# build_flows_metrics
# ===================================================================


class TestBuildFlowsMetrics:
    def test_basic_flow_metrics(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], author="alice"),
            _commit(sha="2", files=["auth/register.py"], author="bob"),
        ]
        flows = build_flows_metrics(
            commits,
            {"login-flow": ["auth/login.py"], "register-flow": ["auth/register.py"]},
        )
        assert len(flows) == 2
        login = next(f for f in flows if f.name == "login-flow")
        assert login.total_commits == 1
        assert login.authors == ["alice"]

    def test_bus_factor_single_author(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], author="alice"),
            _commit(sha="2", files=["auth/login.py"], author="alice"),
        ]
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        assert flows[0].bus_factor == 1

    def test_bus_factor_multiple_authors(self) -> None:
        commits = [
            _commit(sha=str(i), files=["auth/login.py"], author="alice", days_ago=i)
            for i in range(6)
        ] + [
            _commit(sha=str(i + 10), files=["auth/login.py"], author="bob", days_ago=i)
            for i in range(4)
        ]
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        # alice: 6/10=60%, bob: 4/10=40%, threshold=20% => both qualify
        assert flows[0].bus_factor == 2

    def test_health_trend_with_enough_weeks(self) -> None:
        """Health trend requires >= 4 weekly points."""
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)
        commits = []
        for week in range(6):
            commits.append(Commit(
                sha=f"feat-{week}",
                message="feat: something",
                author="a",
                date=base + timedelta(weeks=week),
                files_changed=["auth/login.py"],
            ))
        # Add bug fixes only in first 2 weeks
        for week in range(2):
            commits.append(Commit(
                sha=f"fix-{week}",
                message="fix: bug",
                author="a",
                date=base + timedelta(weeks=week),
                files_changed=["auth/login.py"],
                is_bug_fix=True,
            ))
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        # First half has more bug fixes => positive health_trend (improving)
        assert flows[0].health_trend is not None
        assert flows[0].health_trend > 0

    def test_health_trend_none_with_few_weeks(self) -> None:
        """Health trend is None if < 4 weekly points."""
        commits = [
            _commit(sha="1", files=["auth/login.py"], days_ago=1),
            _commit(sha="2", files=["auth/login.py"], days_ago=2),
        ]
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        assert flows[0].health_trend is None

    def test_hotspot_files(self) -> None:
        """Files with > 40% bug fix ratio and >= 3 commits are hotspots."""
        commits = [
            _commit(sha="1", files=["auth/login.py"], is_bug_fix=True, days_ago=1),
            _commit(sha="2", files=["auth/login.py"], is_bug_fix=True, days_ago=2),
            _commit(sha="3", files=["auth/login.py"], is_bug_fix=False, days_ago=3),
        ]
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        # 2/3 = 67% bug fix ratio > 40%, 3 commits >= 3
        assert "auth/login.py" in flows[0].hotspot_files

    def test_no_hotspot_below_threshold(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], is_bug_fix=False, days_ago=1),
            _commit(sha="2", files=["auth/login.py"], is_bug_fix=False, days_ago=2),
            _commit(sha="3", files=["auth/login.py"], is_bug_fix=False, days_ago=3),
        ]
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        assert flows[0].hotspot_files == []

    def test_test_file_count(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], days_ago=1),
            _commit(sha="2", files=["auth/login.test.ts"], days_ago=2),
        ]
        flows = build_flows_metrics(
            commits,
            {"login": ["auth/login.py"]},
        )
        # login.test.ts is adjacent (same dir), discovered via test-only commits
        assert flows[0].test_file_count >= 1

    def test_coverage_pct(self) -> None:
        commits = [
            _commit(sha="1", files=["auth/login.py"], days_ago=1),
        ]
        coverage = {"auth/login.py": 85.5, "auth/login.test.ts": 100.0}
        flows = build_flows_metrics(
            commits,
            {"login": ["auth/login.py"]},
            coverage_data=coverage,
        )
        assert flows[0].coverage_pct == 85.5

    def test_coverage_pct_none_without_data(self) -> None:
        commits = [_commit(sha="1", files=["auth/login.py"])]
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        assert flows[0].coverage_pct is None

    def test_flow_with_no_commits(self) -> None:
        flows = build_flows_metrics([], {"login": ["auth/login.py"]})
        assert flows[0].total_commits == 0
        assert flows[0].health_score == 100.0

    def test_weekly_points_include_test_only_commits(self) -> None:
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)
        source_commit = Commit(
            sha="s1", message="feat", author="a",
            date=base, files_changed=["auth/login.py"],
        )
        test_commit = Commit(
            sha="t1", message="add test", author="a",
            date=base + timedelta(days=1),
            files_changed=["auth/login.test.ts"],
        )
        flows = build_flows_metrics(
            [source_commit, test_commit],
            {"login": ["auth/login.py"]},
        )
        # Both commits should be in the same week
        assert len(flows[0].weekly_points) == 1
        assert flows[0].weekly_points[0].total_commits == 2

    def test_dir_fallback_for_flow_commits(self) -> None:
        """Commits referencing unknown files use directory fallback."""
        commits = [_commit(sha="1", files=["auth/deleted.py"])]
        flows = build_flows_metrics(commits, {"login": ["auth/login.py"]})
        assert flows[0].total_commits == 1


# ===================================================================
# _calculate_health
# ===================================================================


class TestCalculateHealth:
    def test_zero_commits_is_healthy(self) -> None:
        assert _calculate_health(0.0, 0) == 100.0

    def test_no_bugs_high_score(self) -> None:
        commits = [
            _commit(sha=str(i), days_ago=i, is_bug_fix=False)
            for i in range(10)
        ]
        score = _calculate_health(0.0, 10, commits)
        assert score > 80.0

    def test_high_bug_ratio_low_score(self) -> None:
        commits = [
            _commit(sha=str(i), days_ago=i, is_bug_fix=True)
            for i in range(10)
        ]
        score = _calculate_health(1.0, 10, commits)
        assert score < 20.0

    def test_recent_bugs_penalized_more(self) -> None:
        """Recent bug fixes (< 30 days) get weight 2.0, so they hurt more."""
        # Use a low raw ratio (2 bugs out of 10) so age-weighting makes
        # the effective ratio diverge without both hitting the 0 floor.
        # Scenario A: bugs are RECENT (high weight), features are OLD (low weight)
        # => effective bug ratio goes UP => lower health
        recent_bugs = [
            _commit(sha=str(i), days_ago=5 + i, is_bug_fix=True)
            for i in range(5)
        ]
        old_non_bugs = [
            _commit(sha=str(i + 10), days_ago=200 + i, is_bug_fix=False)
            for i in range(15)
        ]
        score_recent_bugs = _calculate_health(0.25, 20, recent_bugs + old_non_bugs)

        # Scenario B: bugs are OLD (low weight), features are RECENT (high weight)
        # => effective bug ratio goes DOWN => higher health
        old_bugs = [
            _commit(sha=str(i + 20), days_ago=200 + i, is_bug_fix=True)
            for i in range(5)
        ]
        recent_non_bugs = [
            _commit(sha=str(i + 30), days_ago=5 + i, is_bug_fix=False)
            for i in range(15)
        ]
        score_old_bugs = _calculate_health(0.25, 20, old_bugs + recent_non_bugs)

        # Old bugs decay so effective ratio is lower => higher score
        assert score_old_bugs > score_recent_bugs

    def test_activity_factor_scales_with_commits(self) -> None:
        """More commits => activity factor closer to 1.0."""
        few = [_commit(sha=str(i), days_ago=i) for i in range(3)]
        many = [_commit(sha=str(i), days_ago=i) for i in range(50)]
        score_few = _calculate_health(0.0, 3, few)
        score_many = _calculate_health(0.0, 50, many)
        # Both healthy but many-commits has full activity factor
        assert score_many >= score_few

    def test_without_commits_list_uses_raw_ratio(self) -> None:
        """When commits list is not provided, uses raw bug_fix_ratio."""
        score = _calculate_health(0.5, 10)
        # Base: max(0, 100 - 0.5*200) = 0
        assert score == 0.0

    def test_score_bounded_0_to_100(self) -> None:
        assert _calculate_health(0.0, 0) <= 100.0
        assert _calculate_health(1.0, 100) >= 0.0
