"""Sprint 14 Day 3 — path-aware same-name collapse tests."""

from __future__ import annotations

from faultline.llm.pipeline import (
    _collapse_same_name_features,
    _disjoint_rename,
    _longest_common_path_prefix,
    _path_sets_disjoint,
)
from faultline.llm.sonnet_scanner import DeepScanResult


def test_path_sets_disjoint_basic():
    assert _path_sets_disjoint([{"a"}, {"b"}, {"c"}]) is True
    assert _path_sets_disjoint([{"a"}, {"a", "b"}]) is False
    assert _path_sets_disjoint([{"a"}]) is False  # single set


def test_longest_common_prefix():
    sets = [
        {"packages/cli/auth/x.ts", "packages/cli/auth/y.ts"},
        {"packages/cli/auth/z.ts"},
    ]
    assert _longest_common_path_prefix(sets) == ["packages", "cli", "auth"]

    sets = [
        {"packages/cli/auth/x.ts"},
        {"packages/frontend/auth/y.ts"},
    ]
    assert _longest_common_path_prefix(sets) == ["packages"]


def test_disjoint_rename_uses_first_differing_segment():
    paths = {
        "Credentials": {"packages/cli/credentials/x.ts"},
        "Credentials_dup": {"packages/frontend/credentials/y.ts"},
    }
    out = _disjoint_rename(["Credentials", "Credentials_dup"], paths)
    # Both renamed to <segment>/Credentials*
    values = list(out.values())
    assert any(v.startswith("cli/") for v in values)
    assert any(v.startswith("frontend/") for v in values)


def test_overlapping_features_still_merge():
    """Behaviour preserved for overlapping path sets."""
    result = DeepScanResult(
        features={
            "Auth": ["a.ts", "b.ts", "shared.ts"],
            "auth": ["c.ts", "shared.ts"],  # shared.ts overlap
        },
        flows={},
        descriptions={"Auth": "", "auth": ""},
    )
    out = _collapse_same_name_features(result)
    assert len([n for n in out.features if n.lower().endswith("auth")]) == 1


def test_disjoint_features_get_renamed_not_merged():
    """The n8n Credentials case: backend + frontend, disjoint paths."""
    result = DeepScanResult(
        features={
            "Credentials": ["packages/cli/credentials/index.ts"],
            "credentials": ["packages/frontend/credentials/index.ts"],
        },
        flows={"Credentials": ["a-flow"], "credentials": ["b-flow"]},
        descriptions={"Credentials": "backend", "credentials": "frontend"},
    )
    out = _collapse_same_name_features(result)
    # Both should still exist under different prefix-renamed keys
    assert len(out.features) == 2
    names = sorted(out.features)
    assert all("/" in n.lower() or n.lower() != "credentials" for n in names)


def test_disjoint_rename_preserves_flows_and_descriptions():
    result = DeepScanResult(
        features={
            "Issues": ["server/issues/api.ts"],
            "issues": ["public-pages/issues/page.tsx"],
        },
        flows={"Issues": ["create-issue"], "issues": ["view-public-issue"]},
        descriptions={"Issues": "", "issues": ""},
    )
    result.flow_descriptions["Issues"] = {"create-issue": "creates"}
    result.flow_descriptions["issues"] = {"view-public-issue": "views"}

    out = _collapse_same_name_features(result)

    # Find the renamed feature whose flow includes 'create-issue'
    found_create = any(
        "create-issue" in flows for flows in out.flows.values()
    )
    found_view = any(
        "view-public-issue" in flows for flows in out.flows.values()
    )
    assert found_create
    assert found_view


def test_three_disjoint_duplicates_all_renamed():
    result = DeepScanResult(
        features={
            "Service": ["packages/api/service.ts"],
            "service": ["packages/web/service.ts"],
            "SERVICE": ["packages/cli/service.ts"],
        },
        flows={},
        descriptions={"Service": "", "service": "", "SERVICE": ""},
    )
    out = _collapse_same_name_features(result)
    assert len(out.features) == 3
