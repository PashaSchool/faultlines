"""Sprint 16 Day 1 — offline smoke regression for the S15 bug.

The dify Sprint 15 live regression exposed a bug in
``commit_prefix_enrichment_pass`` where post-process mined commit
prefixes (``i18n:``, ``ui:``) and CREATED parallel features that
stole 19 of 24 paths from the auth feature backfilled by Layer A.

This smoke test is the regression gate. It loads the frozen
``assignments-dify.json`` snapshot captured at Sprint 15 (auth has
24 paths after Layer A, before post-process), runs the full
post-process stack against the real dify git repo, and asserts that
auth's 24 paths SURVIVE.

If a future change to:
  - ``post_process.commit_prefix_enrichment_pass``
  - ``DOMAIN_PROTECTED_PREFIXES``
  - or any path-redistributing stage
re-introduces the cannibalisation, this test fails before merge.

The smoke test runs in CI without an API key — it's pure offline
replay using local git log.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytest

# Skip the whole module if the dify benchmark repo isn't checked out
# locally. Without it _mine_commit_prefixes returns an empty dict and
# the regression case can't fire.
DIFY_PATH = Path("/Users/pkuzina/workspace/_faultlines-testrepos/dify")
FIXTURE = (
    Path(__file__).parent / "fixtures" / "dify-assignments-pre-postprocess.json"
)


def _build_features():
    """Reconstruct feature list from the frozen assignments fixture."""
    from faultline.models.types import Feature

    assignments: dict[str, str] = json.loads(FIXTURE.read_text())
    inverted: dict[str, list[str]] = defaultdict(list)
    for path, feat_name in assignments.items():
        inverted[feat_name].append(path)

    return [
        Feature(
            name=name,
            paths=sorted(paths),
            authors=[],
            total_commits=0,
            bug_fixes=0,
            bug_fix_ratio=0.0,
            last_modified=datetime(1970, 1, 1),
            health_score=100.0,
        )
        for name, paths in inverted.items()
    ]


@pytest.mark.skipif(
    not DIFY_PATH.is_dir(),
    reason="dify benchmark repo not checked out locally",
)
def test_commit_prefix_enrichment_does_not_cannibalise_auth():
    """The S15 regression: post-process must not steal auth's paths.

    Before fix: 24 → 5 (commit-mined ``i18n`` and ``ui`` features
    yanked 19 paths).
    After fix:  24 → 24 (DOMAIN_PROTECTED_PREFIXES + feature-token
    blocklist guard).
    """
    from faultline.analyzer.post_process import (
        _mine_commit_prefixes,
        commit_prefix_enrichment_pass,
        drop_noise_features,
        extract_overlooked_top_dirs,
        merge_sub_features,
        reattribute_noise_files,
        refine_by_path_signal,
    )

    feats = _build_features()
    auth_before = next(f for f in feats if f.name == "auth")
    starting_count = len(auth_before.paths)
    assert starting_count == 24, (
        f"fixture sanity: auth should start with 24 paths, got {starting_count}"
    )

    # Replay the full post_process pipeline.
    feats = merge_sub_features(feats)
    feats, *_ = reattribute_noise_files(feats)
    feats, _ = refine_by_path_signal(feats)

    commit_prefixes = _mine_commit_prefixes(str(DIFY_PATH))
    assert "i18n" in commit_prefixes, (
        "fixture sanity: dify must have commits with 'i18n:' prefix; "
        "the fixture was captured against a specific repo state — re-run "
        "the live scan if dify history shifted"
    )

    feats, _ = extract_overlooked_top_dirs(feats, commit_prefixes)
    feats, log = commit_prefix_enrichment_pass(
        feats, str(DIFY_PATH), commit_prefixes,
    )
    feats, _ = drop_noise_features(feats)

    # Assertion: auth still has all 24 paths. If any commit-mined
    # feature stole them, this fails.
    auth_after = next((f for f in feats if f.name == "auth"), None)
    assert auth_after is not None, "auth feature was DROPPED by post_process"
    assert len(auth_after.paths) == starting_count, (
        f"REGRESSION: post_process dropped auth from {starting_count} "
        f"→ {len(auth_after.paths)} paths. "
        f"Likely culprit: commit_prefix_enrichment_pass guard is broken. "
        f"Enrichment log: {log}"
    )

    # Assertion: no parallel ``i18n`` / ``ui`` / ``auth`` feature was
    # created that would be a duplicate of the workspace-named ones
    # (dify-web/i18n / dify-ui / auth).
    feature_names = {f.name for f in feats}
    forbidden = {"i18n", "ui"}  # auth is allowed — it's the real feature
    duplicates = forbidden & feature_names
    assert not duplicates, (
        f"REGRESSION: commit_prefix_enrichment created parallel "
        f"features {sorted(duplicates)} that should have been blocked "
        f"by feature-token guard. Enrichment log: {log}"
    )
