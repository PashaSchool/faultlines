"""Incremental-scan orchestrator (Stage 3).

Decides — given a prior scan and a git diff — what work the pipeline
needs to do this run. Pure logic: it does NOT call ``pipeline.run()``,
``deep_scan``, or any LLM. The CLI / caller takes the
:class:`IncrementalPlan` and either:

  * runs a full scan (``plan.fallback_full_scan == True``)
  * carries forward the prior scan unchanged (``plan.is_no_op``)
  * runs a partial re-scan on the features marked stale, then merges
    the result with the carry-forward (Stages 4 + 5 implement this)

This module is the only one that has to know about both
``PriorScan`` and ``GitDiff`` — keeping that knowledge here lets
the existing pipeline modules stay completely unchanged. If a user
runs ``faultline analyze`` without ``--incremental``, none of this
code is reached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from faultline.analyzer.git_diff import GitDiff
from faultline.llm.scan_loader import PriorScan


logger = logging.getLogger(__name__)


@dataclass
class IncrementalPlan:
    """Decision artefact.

    Attributes:
      stale_features: feature names whose paths overlap with the
        diff. These need re-analysis; their paths + flows + traced
        participants are NOT trusted carry-forward material.
      clean_features: feature names with no diff overlap. Their
        prior result is carried forward verbatim.
      fresh_files: files added in the diff that don't belong to any
        prior feature. These need classification by Sonnet (or
        token-match heuristic). Live outside the prior feature graph.
      deleted_files: files removed in the diff. Removed from any
        feature paths during merge.
      fallback_full_scan: when True, caller MUST run the full pipeline.
        Reasons: missing base_sha, unknown ref, non-git repo, no prior
        scan available.
      reason: short human-readable explanation. Logged + surfaced in
        CLI output so users can see why incremental did/didn't apply.
    """

    stale_features: list[str] = field(default_factory=list)
    clean_features: list[str] = field(default_factory=list)
    fresh_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    fallback_full_scan: bool = False
    reason: str = ""

    @property
    def is_no_op(self) -> bool:
        """True when nothing changed — caller can return prior verbatim."""
        return (
            not self.fallback_full_scan
            and not self.stale_features
            and not self.fresh_files
            and not self.deleted_files
        )

    def summary(self) -> str:
        if self.fallback_full_scan:
            return f"incremental: full-scan fallback ({self.reason})"
        if self.is_no_op:
            return "incremental: no changes — carrying prior scan forward"
        return (
            f"incremental: {len(self.stale_features)} stale feature(s), "
            f"{len(self.clean_features)} clean, "
            f"{len(self.fresh_files)} fresh file(s), "
            f"{len(self.deleted_files)} deleted"
        )


def plan_incremental(
    prior: PriorScan | None,
    diff: GitDiff,
) -> IncrementalPlan:
    """Compute the incremental plan from a loaded prior + computed diff.

    Defensive — never raises. Always returns a plan with either an
    actionable instruction or ``fallback_full_scan=True``.
    """
    if prior is None:
        return IncrementalPlan(
            fallback_full_scan=True,
            reason="no prior scan available",
        )
    if diff.fallback_full_scan:
        return IncrementalPlan(
            fallback_full_scan=True,
            reason="git diff couldn't establish a baseline",
        )

    changed = diff.changed_paths
    deleted = diff.deleted

    # Build reverse index file→features for cheap stale lookup.
    file_to_features: dict[str, list[str]] = {}
    for name, paths in prior.features.items():
        for p in paths:
            file_to_features.setdefault(p, []).append(name)

    stale: set[str] = set()
    for f in changed | deleted:
        for feat in file_to_features.get(f, ()):
            stale.add(feat)

    all_known_files = set(file_to_features)
    fresh = sorted(f for f in diff.added if f not in all_known_files)

    clean = [
        name for name in prior.features
        if name not in stale
    ]

    plan = IncrementalPlan(
        stale_features=sorted(stale),
        clean_features=sorted(clean),
        fresh_files=fresh,
        deleted_files=sorted(deleted),
        reason=("ok" if (stale or fresh or deleted) else "diff empty"),
    )
    logger.info("plan_incremental: %s", plan.summary())
    return plan
