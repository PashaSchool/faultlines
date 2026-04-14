"""Incremental refresh orchestrator.

Given an existing feature-map JSON, computes the minimum work
needed to bring it up to HEAD:
  1. Fetch new commits since last scan
  2. Identify changed files via git diff + content hashes
  3. Route each changed file to its feature
  4. Re-compute metrics for affected features (no LLM)
  5. Optionally: detect files that belong to NO feature → flag as
     candidates for new feature detection (separate step, opt-in)

New-feature detection is intentionally decoupled: it's an LLM call
and should be explicit (`--detect-new`), not automatic on every refresh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from faultline.analyzer.incremental import incremental_update
from faultline.cache.freshness import FreshnessReport, check_freshness
from faultline.cache.hashing import changed_files, hash_files
from faultline.models.types import FeatureMap

logger = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    updated_map: FeatureMap
    freshness_before: FreshnessReport
    files_truly_modified: int       # after content-hash verification
    files_added: int
    files_removed: int
    orphan_files: list[str] = field(default_factory=list)  # files not matched to any feature
    llm_calls_saved: int = 0        # vs full re-scan

    def summary(self) -> str:
        return (
            f"Refresh complete: {self.freshness_before.commits_behind} commits behind, "
            f"{self.files_truly_modified} modified, {self.files_added} added, "
            f"{self.files_removed} removed. "
            f"{len(self.orphan_files)} orphan files detected."
        )


def refresh_feature_map(
    feature_map: FeatureMap,
    repo_path: str = ".",
    *,
    fetch_commits_fn=None,
) -> RefreshResult:
    """Bring a feature map up to the current git HEAD without LLM calls.

    Args:
        feature_map: The previously scanned feature map.
        repo_path: Path to the git repository.
        fetch_commits_fn: Callable(since: datetime) -> list[Commit].
            Defaults to faultline.analyzer.git.get_commits.
            Passed in to keep this module free of heavy imports.

    Returns:
        RefreshResult with the updated map and diagnostics.
    """
    freshness = check_freshness(feature_map, repo_path)

    if not freshness.is_stale:
        logger.info("Feature map already at HEAD — no refresh needed")
        # Still return the map stamped with current analyzed_at? No —
        # keep it unchanged so callers can detect "no-op" via the report.
        return RefreshResult(
            updated_map=feature_map,
            freshness_before=freshness,
            files_truly_modified=0,
            files_added=0,
            files_removed=0,
        )

    # 1. Fetch new commits
    if fetch_commits_fn is None:
        from faultline.analyzer.git import get_commits
        from git import Repo
        repo = Repo(repo_path)
        days_back = max(1, freshness.days_since_scan + 1)
        new_commits = get_commits(repo, days=days_back)
        # Filter to commits after last scan
        if feature_map.analyzed_at:
            new_commits = [c for c in new_commits if c.date > feature_map.analyzed_at]
    else:
        new_commits = fetch_commits_fn(feature_map.analyzed_at)

    # 2. Content-hash verification
    tracked_files_now = _list_tracked_files(repo_path)
    new_hashes = hash_files(tracked_files_now, repo_path)
    modified, added, removed = changed_files(feature_map.file_hashes, new_hashes)

    # 3. Run incremental_update (existing logic) for metric recomputation
    updated = incremental_update(
        feature_map=feature_map,
        new_commits=new_commits,
        tracked_files=tracked_files_now,
    )

    # 4. Update cache metadata
    updated.last_scanned_sha = freshness.current_sha
    updated.file_hashes = new_hashes
    updated.analyzed_at = datetime.now(tz=timezone.utc)

    # 5. Detect orphan files — files present in repo but not mapped
    mapped = set()
    for f in updated.features:
        mapped.update(f.paths)
    orphans = [f for f in tracked_files_now if f not in mapped]

    # 6. LLM savings: for each feature we didn't retouch, we saved
    # the sonnet call + the per-flow haiku call + the per-feature
    # symbol attribution call.
    retouched_features = sum(
        1 for old, new in zip(feature_map.features, updated.features)
        if old.total_commits != new.total_commits
    )
    unchanged = len(feature_map.features) - retouched_features
    saved = unchanged * 3  # rough: sonnet + flow haiku + symbol call

    logger.info(
        "Refresh: %d commits, %d modified, %d added, %d removed, %d orphans",
        len(new_commits), len(modified), len(added), len(removed), len(orphans),
    )

    return RefreshResult(
        updated_map=updated,
        freshness_before=freshness,
        files_truly_modified=len(modified),
        files_added=len(added),
        files_removed=len(removed),
        orphan_files=orphans,
        llm_calls_saved=saved,
    )


def _list_tracked_files(repo_path: str) -> list[str]:
    """Use `git ls-files` to get all tracked files (honors .gitignore)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=repo_path, text=True, timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]
