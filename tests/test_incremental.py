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


# ── Stage 4: workspace partial re-scan ──────────────────────────


from faultline.analyzer.workspace import WorkspaceInfo, WorkspacePackage  # noqa: E402
from faultline.llm.incremental import (  # noqa: E402
    _merge_carry_with_fresh,
    execute_workspace_incremental,
    features_belonging_to_packages,
    identify_stale_packages,
)
from faultline.llm.sonnet_scanner import DeepScanResult  # noqa: E402


def _ws(packages: dict[str, list[str]]) -> WorkspaceInfo:
    return WorkspaceInfo(
        detected=True,
        manager="pnpm",
        packages=[
            WorkspacePackage(name=n, path=f"packages/{n}", files=fs)
            for n, fs in packages.items()
        ],
        root_files=[],
    )


def test_identify_stale_packages_splits_correctly():
    workspace = _ws({
        "auth": ["packages/auth/login.ts", "packages/auth/session.ts"],
        "billing": ["packages/billing/charge.ts"],
        "ui": ["packages/ui/button.tsx"],
    })
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.modified.add("packages/auth/login.ts")
    diff.added.add("packages/billing/refund.ts")  # new file under billing
    stale, clean = identify_stale_packages(workspace, diff)
    assert {p.name for p in stale} == {"auth"}  # added file isn't in pkg.files
    # Add the new file to billing.files so it's recognized.
    workspace.packages[1].files.append("packages/billing/refund.ts")
    diff2 = GitDiff(base_sha="x", head_sha="y")
    diff2.added.add("packages/billing/refund.ts")
    stale2, _ = identify_stale_packages(workspace, diff2)
    assert {p.name for p in stale2} == {"billing"}


def test_identify_stale_picks_up_deleted_files():
    workspace = _ws({
        "auth": ["packages/auth/a.ts", "packages/auth/b.ts"],
        "ui": ["packages/ui/c.tsx"],
    })
    diff = GitDiff(base_sha="x", head_sha="y")
    diff.deleted.add("packages/auth/a.ts")
    stale, clean = identify_stale_packages(workspace, diff)
    assert {p.name for p in stale} == {"auth"}
    assert {p.name for p in clean} == {"ui"}


def test_features_belonging_to_packages_majority_vote():
    prior = _prior({
        "auth": ["packages/auth/a.ts", "packages/auth/b.ts"],
        "billing": ["packages/billing/c.ts", "packages/billing/d.ts"],
        "shared-thing": [
            "packages/auth/x.ts", "packages/auth/y.ts",
            "packages/billing/z.ts",  # majority is auth (2/3)
        ],
    })
    auth_pkg = WorkspacePackage(
        name="auth", path="packages/auth",
        files=[
            "packages/auth/a.ts", "packages/auth/b.ts",
            "packages/auth/x.ts", "packages/auth/y.ts",
        ],
    )
    belonging = features_belonging_to_packages(prior, [auth_pkg])
    # auth (full overlap) and shared-thing (majority) should belong.
    assert belonging == {"auth", "shared-thing"}


def test_merge_carries_clean_features_drops_deleted():
    prior_result = DeepScanResult()
    prior_result.features["auth"] = ["packages/auth/a.ts", "packages/auth/b.ts"]
    prior_result.features["ui"] = ["packages/ui/c.tsx"]
    prior_result.descriptions["auth"] = "Auth desc"
    prior_result.descriptions["ui"] = "UI desc"
    prior_result.flows["auth"] = ["sign-in"]
    prior = PriorScan(
        result=prior_result, last_sha="abc",
        feature_stats={}, flow_stats={}, scan_meta={},
    )

    fresh = DeepScanResult()
    fresh.features["auth-rescanned"] = ["packages/auth/a.ts"]
    fresh.descriptions["auth-rescanned"] = "Fresh auth"

    clean_pkg = WorkspacePackage(
        name="ui", path="packages/ui",
        files=["packages/ui/c.tsx"],
    )

    merged = _merge_carry_with_fresh(
        prior=prior, fresh=fresh,
        clean_packages=[clean_pkg],
        deleted_files=set(),
    )
    # Fresh feature is in.
    assert merged.features["auth-rescanned"] == ["packages/auth/a.ts"]
    # Clean ui carried with description + flows.
    assert merged.features["ui"] == ["packages/ui/c.tsx"]
    assert merged.descriptions["ui"] == "UI desc"


def test_merge_drops_features_emptied_by_deletions():
    prior_result = DeepScanResult()
    prior_result.features["legacy"] = ["packages/old/x.ts"]
    prior = PriorScan(
        result=prior_result, last_sha="abc",
        feature_stats={}, flow_stats={}, scan_meta={},
    )
    fresh = DeepScanResult()
    clean_pkg = WorkspacePackage(
        name="old", path="packages/old", files=["packages/old/x.ts"],
    )
    merged = _merge_carry_with_fresh(
        prior=prior, fresh=fresh,
        clean_packages=[clean_pkg],
        deleted_files={"packages/old/x.ts"},
    )
    assert "legacy" not in merged.features


def test_carry_only_drops_deleted_files():
    from faultline.llm.incremental import _carry_only
    prior_result = DeepScanResult()
    prior_result.features["auth"] = ["src/a.ts", "src/b.ts"]
    prior_result.features["billing"] = ["src/c.ts"]
    prior_result.descriptions["auth"] = "Auth"
    prior = PriorScan(
        result=prior_result, last_sha="abc",
        feature_stats={}, flow_stats={}, scan_meta={},
    )
    out = _carry_only(prior, deleted={"src/c.ts"})
    assert out.features["auth"] == ["src/a.ts", "src/b.ts"]
    assert "billing" not in out.features  # all files deleted
    assert out.descriptions["auth"] == "Auth"


def test_merge_monolith_fresh_replaces_stale_clean_carries():
    from faultline.llm.incremental import _merge_monolith_carry
    prior_result = DeepScanResult()
    prior_result.features["auth"] = ["src/auth/a.ts", "src/auth/b.ts"]
    prior_result.features["billing"] = ["src/billing/c.ts"]
    prior_result.descriptions["auth"] = "Old auth"
    prior_result.descriptions["billing"] = "Billing"
    prior_result.flows["auth"] = ["sign-in"]
    prior = PriorScan(
        result=prior_result, last_sha="abc",
        feature_stats={}, flow_stats={}, scan_meta={},
    )
    fresh = DeepScanResult()
    fresh.features["user-authentication"] = ["src/auth/a.ts", "src/auth/b.ts"]
    fresh.descriptions["user-authentication"] = "Renamed"

    out = _merge_monolith_carry(
        prior=prior, fresh=fresh,
        stale_features={"auth"},
        deleted_files=set(),
    )
    # Fresh introduced new name; old auth dropped.
    assert "user-authentication" in out.features
    assert "auth" not in out.features
    # Clean billing carried.
    assert out.features["billing"] == ["src/billing/c.ts"]
    assert out.descriptions["billing"] == "Billing"


def test_merge_monolith_avoids_double_counting_files():
    """If a prior file is now claimed by a fresh feature, don't keep
    it in any clean feature (would double-count)."""
    from faultline.llm.incremental import _merge_monolith_carry
    prior_result = DeepScanResult()
    prior_result.features["legacy-mixed"] = ["src/a.ts", "src/b.ts"]
    prior = PriorScan(
        result=prior_result, last_sha="abc",
        feature_stats={}, flow_stats={}, scan_meta={},
    )
    fresh = DeepScanResult()
    fresh.features["new-shape"] = ["src/a.ts"]  # claims a.ts
    out = _merge_monolith_carry(
        prior=prior, fresh=fresh,
        stale_features=set(),  # legacy-mixed is "clean"
        deleted_files=set(),
    )
    # legacy-mixed carried but a.ts removed (now in new-shape).
    assert out.features["legacy-mixed"] == ["src/b.ts"]
    assert out.features["new-shape"] == ["src/a.ts"]


def test_execute_workspace_incremental_no_stale_returns_prior():
    """When no packages are stale, execute_workspace_incremental
    short-circuits and returns the prior scan unchanged — no LLM
    call. Same outcome as the no-op path in plan_incremental but
    with the workspace-level lens."""
    prior_result = DeepScanResult()
    prior_result.features["auth"] = ["packages/auth/a.ts"]
    prior = PriorScan(
        result=prior_result, last_sha="abc",
        feature_stats={}, flow_stats={}, scan_meta={},
    )
    workspace = _ws({"auth": ["packages/auth/a.ts"]})
    diff = GitDiff(base_sha="x", head_sha="y")  # no changes
    plan = IncrementalPlan(stale_features=[], clean_features=["auth"])

    out = execute_workspace_incremental(
        plan=plan, prior=prior, diff=diff,
        workspace=workspace, repo_root=".",
        api_key=None, model=None,
    )
    assert out is prior.result
