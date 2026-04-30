"""Incremental-scan orchestrator (Stages 3-4).

Stage 3 — :func:`plan_incremental` decides what work to do. Pure
logic, no LLM calls, no pipeline imports.

Stage 4 — :func:`execute_workspace_incremental` runs the partial
re-scan for monorepo paths: subsets the workspace to only stale
packages, calls the existing ``deep_scan_workspace`` on that
subset, and merges the fresh result with the prior scan's
carry-forward features.

Both phases are kept here so the rest of the pipeline stays
unchanged. If ``--incremental`` is not passed, none of this code
runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from faultline.analyzer.git_diff import GitDiff
from faultline.llm.scan_loader import PriorScan

if TYPE_CHECKING:  # pragma: no cover
    from faultline.analyzer.workspace import WorkspaceInfo, WorkspacePackage
    from faultline.llm.cost import CostTracker
    from faultline.llm.sonnet_scanner import DeepScanResult


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


# ── Stage 4: workspace partial re-scan ──────────────────────────


def identify_stale_packages(
    workspace: "WorkspaceInfo",
    diff: GitDiff,
) -> tuple[list["WorkspacePackage"], list["WorkspacePackage"]]:
    """Split workspace packages into ``(stale, clean)`` based on diff.

    A package is stale when at least one file in its ``files`` list
    appears in ``diff.changed_paths`` or ``diff.deleted``. Otherwise
    it's clean — its prior result is reusable verbatim.
    """
    changed_or_deleted = diff.changed_paths | diff.deleted
    stale: list[WorkspacePackage] = []
    clean: list[WorkspacePackage] = []
    for pkg in workspace.packages:
        pkg_set = set(pkg.files)
        if pkg_set & changed_or_deleted:
            stale.append(pkg)
        else:
            clean.append(pkg)
    return stale, clean


def features_belonging_to_packages(
    prior: PriorScan,
    packages: list["WorkspacePackage"],
) -> set[str]:
    """Return the names of prior features whose files belong (mostly)
    to one of the given packages.

    Heuristic: a feature belongs to the package that owns the majority
    of its files. Required because feature names like ``auth/login``
    are LLM-derived from package ``auth`` — we can't rely on the path
    prefix alone (some prior features were renamed by critique).
    """
    pkg_file_sets: dict[str, set[str]] = {
        pkg.name: set(pkg.files) for pkg in packages
    }
    out: set[str] = set()
    for feat_name, feat_files in prior.features.items():
        if not feat_files:
            continue
        # Vote: which package owns the most of feat_files?
        best_pkg = ""
        best_count = 0
        for pkg_name, pkg_set in pkg_file_sets.items():
            count = sum(1 for f in feat_files if f in pkg_set)
            if count > best_count:
                best_pkg = pkg_name
                best_count = count
        # Treat as belonging to packages set when majority lives there.
        if best_pkg and best_count >= len(feat_files) // 2 + 1:
            out.add(feat_name)
    return out


def execute_workspace_incremental(
    *,
    plan: IncrementalPlan,
    prior: PriorScan,
    diff: GitDiff,
    workspace: "WorkspaceInfo",
    repo_root: Any,
    api_key: str | None,
    model: str | None,
    tracker: "CostTracker | None" = None,
    is_library: bool = False,
    use_tools: bool = False,
    commit_context: str | None = None,
    preferred_names: list[str] | None = None,
) -> "DeepScanResult":
    """Run the partial re-scan for a workspace repo.

    Strategy:
      1. Split workspace.packages into stale (any diff hit) and clean.
      2. If no stale packages → return prior scan as DeepScanResult
         (no LLM cost; caller already logged the no-op summary).
      3. Otherwise, build a sub-WorkspaceInfo containing only the
         stale packages and call ``deep_scan_workspace`` on it.
      4. Carry forward every prior feature that "belongs" to a clean
         package (majority-file-overlap heuristic).
      5. Drop carried features that overlap deleted files.
      6. Merge fresh result + carried = final DeepScanResult.

    The existing ``deep_scan_workspace`` is called unmodified — no
    risk of regressing the full-scan path.
    """
    from faultline.analyzer.workspace import WorkspaceInfo
    from faultline.llm.sonnet_scanner import (
        DeepScanResult, deep_scan_workspace,
    )

    stale_pkgs, clean_pkgs = identify_stale_packages(workspace, diff)
    if not stale_pkgs:
        logger.info(
            "execute_workspace_incremental: no stale packages — "
            "carrying prior scan forward"
        )
        return prior.result

    # Step 3: build subset workspace and re-scan.
    subset = WorkspaceInfo(
        detected=workspace.detected,
        manager=workspace.manager,
        packages=stale_pkgs,
        root_files=[],  # root files re-scan only on stale-root flag (skip for now)
    )
    logger.info(
        "execute_workspace_incremental: re-scanning %d stale package(s) "
        "(%d clean carried forward)",
        len(stale_pkgs), len(clean_pkgs),
    )
    fresh = deep_scan_workspace(
        subset,
        api_key=api_key,
        model=model,
        is_library=is_library,
        tracker=tracker,
        commit_context=commit_context,
        use_tools=use_tools,
        repo_root=repo_root,
        preferred_names=preferred_names,
    )
    if fresh is None:
        logger.warning(
            "execute_workspace_incremental: subset scan returned None "
            "— falling back to prior scan unchanged"
        )
        return prior.result

    # Step 4-6: merge.
    return _merge_carry_with_fresh(
        prior=prior,
        fresh=fresh,
        clean_packages=clean_pkgs,
        deleted_files=diff.deleted,
    )


def _merge_carry_with_fresh(
    *,
    prior: PriorScan,
    fresh: "DeepScanResult",
    clean_packages: list["WorkspacePackage"],
    deleted_files: set[str],
) -> "DeepScanResult":
    """Build the final ``DeepScanResult`` from prior carry + fresh.

    - Features in ``fresh`` win (they are the re-scan output for
      stale packages).
    - Features in ``prior`` that belong to a clean package are
      copied over with all their stats.
    - Deleted files are removed from any carried feature's path list;
      a feature whose path list becomes empty is dropped entirely.
    """
    from faultline.llm.sonnet_scanner import DeepScanResult

    out = DeepScanResult()

    # Start with fresh — these win over any prior carry.
    out.features.update({k: list(v) for k, v in fresh.features.items()})
    out.descriptions.update(dict(fresh.descriptions))
    out.flows.update({k: list(v) for k, v in fresh.flows.items()})
    out.flow_descriptions.update(
        {k: dict(v) for k, v in fresh.flow_descriptions.items()},
    )
    out.flow_participants.update(
        {k: dict(v) for k, v in fresh.flow_participants.items()},
    )
    out.cost_summary = fresh.cost_summary

    # Determine which prior features belong to clean packages.
    if clean_packages:
        carry_names = features_belonging_to_packages(prior, clean_packages)
    else:
        carry_names = set()

    fresh_names = set(out.features)
    carried = 0
    dropped_empty = 0
    for name in carry_names:
        if name in fresh_names:
            # Already produced by the re-scan; fresh wins.
            continue
        prior_files = prior.features.get(name) or []
        kept = [f for f in prior_files if f not in deleted_files]
        if not kept:
            dropped_empty += 1
            continue
        out.features[name] = kept
        if name in prior.result.descriptions:
            out.descriptions[name] = prior.result.descriptions[name]
        if name in prior.result.flows:
            out.flows[name] = list(prior.result.flows[name])
        if name in prior.result.flow_descriptions:
            out.flow_descriptions[name] = dict(prior.result.flow_descriptions[name])
        if name in prior.result.flow_participants:
            out.flow_participants[name] = dict(
                prior.result.flow_participants[name],
            )
        carried += 1

    logger.info(
        "incremental merge: fresh=%d, carried=%d, dropped_empty=%d, "
        "deleted_files=%d → total %d features",
        len(fresh_names), carried, dropped_empty, len(deleted_files),
        len(out.features),
    )
    return out
