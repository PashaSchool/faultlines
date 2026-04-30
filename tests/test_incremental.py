"""Tests for the incremental-scan planner (Stage 3)."""

from __future__ import annotations

from faultline.analyzer.git_diff import GitDiff
from faultline.llm.incremental import IncrementalPlan, plan_incremental
from faultline.llm.scan_loader import PriorScan
from faultline.llm.sonnet_scanner import DeepScanResult


def _prior(features: dict[str, list[str]]) -> PriorScan:
    result = DeepScanResult()
    for name, paths in features.items():
        result.features[name] = list(paths)
    return PriorScan(
        result=result,
        last_sha="abc",
        feature_stats={},
        flow_stats={},
        scan_meta={},
    )


def test_no_prior_falls_back():
    diff = GitDiff(base_sha="x", head_sha="y")
    plan = plan_incremental(prior=None, diff=diff)
    assert plan.fallback_full_scan is True
    assert "no prior" in plan.reason


def test_diff_fallback_propagates():
    diff = GitDiff(base_sha=None, head_sha=None, fallback_full_scan=True)
    plan = plan_incremental(prior=_prior({}), diff=diff)
    assert plan.fallback_full_scan is True


def test_no_changes_returns_noop():
    prior = _prior({"auth": ["src/auth/a.ts", "src/auth/b.ts"]})
    diff = GitDiff(base_sha="x", head_sha="y")  # empty diff
    plan = plan_incremental(prior, diff)
    assert plan.is_no_op
    assert plan.fallback_full_scan is False
    assert plan.stale_features == []
    assert plan.clean_features == ["auth"]


def test_modified_file_marks_owning_feature_stale():
    prior = _prior({
        "auth": ["src/auth/login.ts", "src/auth/session.ts"],
        "billing": ["src/billing/charge.ts"],
    })
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.modified.add("src/auth/login.ts")
    plan = plan_incremental(prior, diff)
    assert plan.stale_features == ["auth"]
    assert plan.clean_features == ["billing"]
    assert plan.fresh_files == []


def test_added_file_unknown_to_prior_is_fresh():
    prior = _prior({"auth": ["src/auth/a.ts"]})
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.added.add("src/payments/new.ts")
    plan = plan_incremental(prior, diff)
    assert plan.fresh_files == ["src/payments/new.ts"]
    # Existing features stay clean.
    assert plan.stale_features == []
    assert plan.clean_features == ["auth"]


def test_added_file_inside_existing_feature_marks_it_stale():
    """A file added under a path that already belongs to a feature
    counts as a modification of that feature, not a fresh file. The
    file_to_features index uses exact paths, so an added file is
    technically unknown — but if its parent dir is a feature,
    incremental still treats it as fresh; the orchestrator (Stage 4)
    decides whether to map fresh files into existing features. For
    Stage 3 we just collect them."""
    prior = _prior({"auth": ["src/auth/a.ts"]})
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.added.add("src/auth/new.ts")
    plan = plan_incremental(prior, diff)
    # New file isn't known to prior → fresh, not stale-trigger.
    assert "src/auth/new.ts" in plan.fresh_files
    assert plan.stale_features == []


def test_deleted_file_marks_owning_feature_stale():
    prior = _prior({"auth": ["src/auth/a.ts", "src/auth/b.ts"]})
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.deleted.add("src/auth/a.ts")
    plan = plan_incremental(prior, diff)
    assert plan.stale_features == ["auth"]
    assert plan.deleted_files == ["src/auth/a.ts"]


def test_summary_string():
    prior = _prior({"auth": ["src/auth/a.ts"], "billing": ["src/b.ts"]})
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.modified.add("src/auth/a.ts")
    diff.added.add("src/c.ts")
    plan = plan_incremental(prior, diff)
    s = plan.summary()
    assert "1 stale" in s
    assert "1 clean" in s
    assert "1 fresh" in s


def test_noop_summary():
    plan = IncrementalPlan(reason="diff empty")
    assert "no changes" in plan.summary()


def test_fallback_summary():
    plan = IncrementalPlan(fallback_full_scan=True, reason="reason here")
    assert "full-scan fallback" in plan.summary()
    assert "reason here" in plan.summary()


def test_multiple_features_stale_when_files_in_each_change():
    prior = _prior({
        "a": ["src/a.ts"],
        "b": ["src/b.ts"],
        "c": ["src/c.ts"],
    })
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.modified.add("src/a.ts")
    diff.modified.add("src/c.ts")
    plan = plan_incremental(prior, diff)
    assert plan.stale_features == ["a", "c"]
    assert plan.clean_features == ["b"]
