"""Reconstruct LIVE state from assignments-dify.json and run post_process
on REAL repo path. The previous debug used the final JSON state (where
Layer A would create auth=117); this version uses the mid-pipeline
state where auth=24 (matches live).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from faultline.models.types import Feature, FeatureMap


def _count(label: str, mapping: dict[str, list[str]]) -> None:
    auth = mapping.get("auth", [])
    print(f"  [{label:50s}]  auth={len(auth):4d}   total={len(mapping)}")


def main() -> int:
    # Load assignments cache (the LIVE mid-pipeline state)
    assignments = json.loads(
        Path.home().joinpath(".faultline/assignments-dify.json").read_text()
    )
    # Invert: feature_name → list[path]
    inverted: dict[str, list[str]] = defaultdict(list)
    for path, feat_name in assignments.items():
        inverted[feat_name].append(path)
    feature_paths = {k: sorted(v) for k, v in inverted.items()}

    print("=== LIVE state (from assignments-dify.json) ===")
    _count("assignments cache state", feature_paths)

    # Build minimal Feature list (no commit data — that's irrelevant for
    # path mutations).
    feats = [
        Feature(
            name=name,
            paths=paths,
            authors=[],
            total_commits=0,
            bug_fixes=0,
            bug_fix_ratio=0.0,
            last_modified=datetime(1970, 1, 1),
            health_score=100.0,
        )
        for name, paths in feature_paths.items()
    ]

    fmap = FeatureMap(
        repo_path="dify",
        remote_url="https://github.com/langgenius/dify",
        analyzed_at=datetime.now(),
        total_commits=4775,
        date_range_days=365,
        features=feats,
    )

    # Now run post_process stage-by-stage with REAL repo path.
    real_repo = "/Users/pkuzina/workspace/_faultlines-testrepos/dify"
    from faultline.analyzer.post_process import (
        merge_sub_features, reattribute_noise_files, refine_by_path_signal,
        extract_overlooked_top_dirs, commit_prefix_enrichment_pass,
        drop_noise_features, _mine_commit_prefixes,
    )

    feats_iter = list(fmap.features)
    feats_iter = merge_sub_features(feats_iter)
    _count("after merge_sub_features", {f.name: f.paths for f in feats_iter})

    feats_iter, *_ = reattribute_noise_files(feats_iter)
    _count("after reattribute_noise_files", {f.name: f.paths for f in feats_iter})

    feats_iter, _ = refine_by_path_signal(feats_iter)
    _count("after refine_by_path_signal", {f.name: f.paths for f in feats_iter})

    commit_prefixes = _mine_commit_prefixes(real_repo)
    print(f"\n  (mined {len(commit_prefixes)} commit prefixes)")
    print(f"  prefixes: {sorted(commit_prefixes.keys())[:20]}")

    feats_iter, log = extract_overlooked_top_dirs(feats_iter, commit_prefixes)
    if log:
        for line in log[:5]:
            print(f"    extract: {line}")
    _count("after extract_overlooked_top_dirs", {f.name: f.paths for f in feats_iter})

    feats_iter, log = commit_prefix_enrichment_pass(feats_iter, real_repo, commit_prefixes)
    if log:
        for line in log[:10]:
            print(f"    enrich: {line}")
    _count("after commit_prefix_enrichment_pass", {f.name: f.paths for f in feats_iter})

    feats_iter, dropped = drop_noise_features(feats_iter)
    if dropped:
        for n, r, ct in dropped[:5]:
            print(f"    drop_noise: {n} ({r}, {ct} files)")
    _count("after drop_noise_features", {f.name: f.paths for f in feats_iter})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
