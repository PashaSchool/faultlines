"""
Incremental update for feature maps.

Takes an existing FeatureMap (from deep-scan or analyze) and new commits,
then updates health scores, commit counts, and file assignments — without
calling any LLM.

Usage:
    old_map = FeatureMap.model_validate_json(Path("feature-map.json").read_text())
    new_commits = get_commits(repo, since=old_map.analyzed_at)
    updated = incremental_update(old_map, new_commits, tracked_files)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from faultline.analyzer.features import _calculate_health, _collect_prs, _best_stem_match, _normalize_stem
from faultline.models.types import Commit, Feature, FeatureMap

logger = logging.getLogger(__name__)


def incremental_update(
    feature_map: FeatureMap,
    new_commits: list[Commit],
    tracked_files: list[str] | None = None,
) -> FeatureMap:
    """
    Updates an existing FeatureMap with new commits.

    For each new commit:
    1. Match changed files to existing features (O(1) lookup)
    2. New files → match by stem/directory to closest feature
    3. Recalculate health scores for affected features

    No LLM calls — pure heuristic matching + metric recalculation.

    Args:
        feature_map: Existing feature map to update.
        new_commits: Commits since last analysis.
        tracked_files: Current tracked files in repo (for detecting new files).

    Returns:
        Updated FeatureMap with new metrics.
    """
    if not new_commits:
        logger.info("No new commits — nothing to update")
        return feature_map

    # Build file → feature index from existing map
    file_to_feature: dict[str, str] = {}
    dir_to_feature: dict[str, str] = {}
    feature_stems: dict[str, str] = {}  # stem → feature_name

    for feat in feature_map.features:
        for fp in feat.paths:
            file_to_feature[fp] = feat.name
            parent = str(Path(fp).parent)
            if parent != ".":
                dir_to_feature.setdefault(parent, feat.name)
        # Register feature name stems for new file matching
        for stem in _feature_name_stems(feat.name):
            feature_stems[stem] = feat.name

    # Process each commit — assign to features
    feature_new_commits: dict[str, list[Commit]] = {}
    new_files_assigned: dict[str, list[str]] = {}  # feature → new file paths
    unmatched_files: list[str] = []

    for commit in new_commits:
        touched_features: set[str] = set()

        for fp in commit.files_changed:
            # Strategy 1: exact file match
            feat = file_to_feature.get(fp)

            # Strategy 2: directory match
            if not feat:
                parent = str(Path(fp).parent)
                while parent and parent != ".":
                    feat = dir_to_feature.get(parent)
                    if feat:
                        break
                    parent = str(Path(parent).parent)

            # Strategy 3: stem match for truly new files
            if not feat:
                file_stem = _normalize_stem(Path(fp).stem)
                feat = _best_stem_match(file_stem, feature_stems) if file_stem else None

                # Strategy 4: parent directory name match
                if not feat:
                    for part in reversed(Path(fp).parts[:-1]):
                        part_stem = _normalize_stem(part)
                        feat = _best_stem_match(part_stem, feature_stems) if part_stem else None
                        if feat:
                            break

            if feat:
                touched_features.add(feat)
                # Track new file assignment
                if fp not in file_to_feature:
                    file_to_feature[fp] = feat
                    new_files_assigned.setdefault(feat, []).append(fp)
            else:
                unmatched_files.append(fp)

        for feat_name in touched_features:
            feature_new_commits.setdefault(feat_name, []).append(commit)

    # Log summary
    affected_count = len(feature_new_commits)
    new_file_count = sum(len(fps) for fps in new_files_assigned.values())
    logger.info(
        "Incremental: %d new commits → %d features affected, %d new files assigned, %d unmatched",
        len(new_commits), affected_count, new_file_count, len(unmatched_files),
    )

    # Rebuild features with updated metrics
    updated_features: list[Feature] = []

    for feat in feature_map.features:
        new_commits_for_feat = feature_new_commits.get(feat.name, [])
        new_paths = new_files_assigned.get(feat.name, [])

        if not new_commits_for_feat and not new_paths:
            # Feature unchanged — keep as-is
            updated_features.append(feat)
            continue

        # Merge old + new
        all_paths = list(dict.fromkeys(feat.paths + new_paths))
        total_commits = feat.total_commits + len(new_commits_for_feat)
        new_bug_fixes = sum(1 for c in new_commits_for_feat if c.is_bug_fix)
        total_bug_fixes = feat.bug_fixes + new_bug_fixes
        bug_fix_ratio = total_bug_fixes / total_commits if total_commits > 0 else 0.0

        # Update authors
        new_authors = {c.author for c in new_commits_for_feat}
        all_authors = sorted(set(feat.authors) | new_authors)

        # Update last_modified
        if new_commits_for_feat:
            newest = max(c.date for c in new_commits_for_feat)
            last_modified = max(feat.last_modified, newest)
        else:
            last_modified = feat.last_modified

        # Recalculate health score
        health = _calculate_health(bug_fix_ratio, total_commits, new_commits_for_feat)

        # Collect new PRs
        new_prs = _collect_prs(new_commits_for_feat, feature_map.remote_url)
        all_prs = feat.bug_fix_prs + new_prs
        # Deduplicate PRs
        seen_pr_nums: set[int] = set()
        deduped_prs = []
        for pr in sorted(all_prs, key=lambda p: p.date, reverse=True):
            if pr.number not in seen_pr_nums:
                seen_pr_nums.add(pr.number)
                deduped_prs.append(pr)

        updated_features.append(Feature(
            name=feat.name,
            description=feat.description,
            paths=all_paths,
            authors=all_authors,
            total_commits=total_commits,
            bug_fixes=total_bug_fixes,
            bug_fix_ratio=round(bug_fix_ratio, 3),
            last_modified=last_modified,
            health_score=health,
            flows=feat.flows,  # flows stay unchanged in incremental
            bug_fix_prs=deduped_prs,
            coverage_pct=feat.coverage_pct,
            shared_attributions=feat.shared_attributions,
            symbol_health_score=feat.symbol_health_score,
        ))

    return FeatureMap(
        repo_path=feature_map.repo_path,
        remote_url=feature_map.remote_url,
        analyzed_at=datetime.now(tz=timezone.utc),
        total_commits=feature_map.total_commits + len(new_commits),
        date_range_days=feature_map.date_range_days,
        features=updated_features,
    )


def _feature_name_stems(name: str) -> list[str]:
    """Returns stem variants for a feature name for matching."""
    stems = [name]
    parts = name.split("-")
    stems.extend(parts)
    # Singular/plural
    if name.endswith("s"):
        stems.append(name[:-1])
    else:
        stems.append(name + "s")
    return [s for s in stems if len(s) >= 3]
