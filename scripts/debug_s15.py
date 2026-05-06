"""Sprint 16 Day 0 — find which post-pipeline step drops Layer A's work.

Simulates the cli.py post-pipeline path on the saved S15 dify scan
(after applying Layer A backfill manually). Prints auth-feature path
count at each transformation point so we can pinpoint the regression.

Usage:
    source .venv/bin/activate
    python scripts/debug_s15.py
"""

from __future__ import annotations

import json
from pathlib import Path

from faultline.llm.flow_cluster import promote_virtual_clusters
from faultline.llm.sonnet_scanner import DeepScanResult


def _count(label: str, mapping: dict[str, list[str]]) -> None:
    auth = mapping.get("auth", [])
    print(f"  [{label:35s}]  auth={len(auth):4d} paths   total_features={len(mapping)}")


def main() -> int:
    fm = json.loads(Path("benchmarks/dify/sprint15-live.json").read_text())

    # Reconstruct DeepScanResult from saved JSON (post-pipeline.run state).
    result = DeepScanResult(
        features={f["name"]: list(f.get("paths", [])) for f in fm["features"]},
        flows={f["name"]: [fl["name"] for fl in f.get("flows", [])] for f in fm["features"]},
        descriptions={f["name"]: f.get("description", "") for f in fm["features"]},
    )

    print("=== Stage 0 — saved JSON ===")
    _count("after pipeline.run (live)", result.features)

    # Manually apply Layer A backfill (reproduce what S15 logic does).
    promote_virtual_clusters(result)
    print("\n=== Stage 1 — after Layer A backfill (synthetic) ===")
    _count("after promote_virtual_clusters", result.features)

    raw_mapping = dict(result.features)
    print("\n=== Stage 2 — cli line 660: raw_mapping = dict(features) ===")
    _count("raw_mapping snapshot", raw_mapping)

    feature_paths = raw_mapping  # no path_prefix for dify
    print("\n=== Stage 3 — cli line 936: feature_paths = raw_mapping ===")
    _count("feature_paths", feature_paths)

    # build_feature_map calls _merge_small_features inside.
    from faultline.analyzer.features import _merge_small_features
    merged = _merge_small_features(feature_paths)
    print("\n=== Stage 4 — _merge_small_features (inside build_feature_map) ===")
    _count("after _merge_small_features", merged)

    # Build minimal FeatureMap-compatible structure for _drop_noise_features.
    # _drop_noise_features expects a list of Pydantic Feature objects.
    from faultline.models.types import Feature
    from datetime import datetime
    feats = [
        Feature(
            name=name,
            paths=paths,
            authors=[],
            total_commits=0,  # cold by default — could trigger drop
            bug_fixes=0,
            bug_fix_ratio=0.0,
            last_modified=datetime(1970, 1, 1),
            health_score=100.0,
        )
        for name, paths in merged.items()
    ]
    from faultline.analyzer.features import _drop_noise_features
    after_drop = _drop_noise_features(feats)
    drop_map = {f.name: f.paths for f in after_drop}
    print("\n=== Stage 5 — _drop_noise_features (cli line 1106) ===")
    _count("after _drop_noise_features", drop_map)

    # Stage 6 — run_post_process (cli line 1132). Suspect.
    # We need a real FeatureMap with the right shape. Let's build a
    # minimal one with the post-drop features.
    from faultline.models.types import FeatureMap
    fmap = FeatureMap(
        repo_path="dify",
        remote_url="https://github.com/langgenius/dify",
        analyzed_at=datetime(2026, 5, 6),
        total_commits=4775,
        date_range_days=365,
        features=after_drop,
    )
    # Run each post_process stage individually with REAL repo path to
    # reproduce what live scan does.
    real_repo = "/Users/pkuzina/workspace/_faultlines-testrepos/dify"
    from faultline.analyzer.post_process import (
        merge_sub_features, reattribute_noise_files, refine_by_path_signal,
        extract_overlooked_top_dirs, commit_prefix_enrichment_pass,
        drop_noise_features, _mine_commit_prefixes,
    )
    feats_iter = list(fmap.features)
    feats_iter = merge_sub_features(feats_iter)
    fmap.features = feats_iter
    _count("Stage 6a after merge_sub_features", {f.name: f.paths for f in feats_iter})
    feats_iter, *_ = reattribute_noise_files(feats_iter)
    _count("Stage 6b after reattribute_noise_files", {f.name: f.paths for f in feats_iter})
    feats_iter, _ = refine_by_path_signal(feats_iter)
    _count("Stage 6c after refine_by_path_signal", {f.name: f.paths for f in feats_iter})
    commit_prefixes = _mine_commit_prefixes(real_repo)
    print(f"\n  (commit prefixes mined: {len(commit_prefixes)} found)")
    feats_iter, _ = extract_overlooked_top_dirs(feats_iter, commit_prefixes)
    _count("Stage 6d after extract_overlooked_top_dirs", {f.name: f.paths for f in feats_iter})
    feats_iter, _ = commit_prefix_enrichment_pass(feats_iter, real_repo, commit_prefixes)
    _count("Stage 6e after commit_prefix_enrichment_pass", {f.name: f.paths for f in feats_iter})
    feats_iter, _ = drop_noise_features(feats_iter)
    _count("Stage 6f after drop_noise_features", {f.name: f.paths for f in feats_iter})

    # Compare against final live JSON to see where the divergence kicks in.
    print("\n=== Live final JSON (saved 13:57) ===")
    final = {f["name"]: list(f.get("paths", [])) for f in fm["features"]}
    _count("final (live)", final)

    # Diagnostic: are auth's 117 (post-backfill) paths still in feature_paths?
    auth_paths_after_backfill = set(merged.get("auth", []))
    auth_paths_final_live = set(final.get("auth", []))
    print(f"\nauth paths post-backfill but missing in final live: "
          f"{len(auth_paths_after_backfill - auth_paths_final_live)}")
    sample_missing = sorted(auth_paths_after_backfill - auth_paths_final_live)[:5]
    for p in sample_missing:
        print(f"  - {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
