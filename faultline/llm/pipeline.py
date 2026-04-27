"""Single entry point for the new Faultlines feature detection pipeline.

The legacy pipeline lives at ``cli.py:264-483`` and dispatches to one of
five strategies (workspace per-package, iterative, mixed-repo, import-graph,
candidate-based) based on a tangle of conditionals. Each strategy has its
own filtering, post-processing, and cost behaviour, which is what made
the rewrite necessary.

This module replaces all of that with a single ``run()`` function that:

  1. Picks ``deep_scan_workspace`` for workspace monorepos with 2+ real
     packages, or ``deep_scan`` (with ``detect_candidates``) for
     monoliths and single-package repos.
  2. Threads ``is_library`` from ``repo_classifier.detect_library`` into
     both paths so libraries get their flows stripped.
  3. Builds a single commit-context block once via
     ``build_commit_context`` and reuses it across every per-package
     LLM call.
  4. Threads a shared ``CostTracker`` through every call so the total
     cost printed at the end of an analyze run is the sum across all
     LLM invocations regardless of how many packages were touched.
  5. Returns a single ``DeepScanResult`` so the caller does not need to
     know which dispatch path ran.

This module does NOT touch ``cli.py``. The Day 9 cutover will introduce
a ``--legacy`` flag and switch the default in ``analyze --llm`` over.
Until then ``run()`` lives in parallel and is exercised purely by
``tests/test_pipeline.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faultline.analyzer.bucketizer import Bucket, bucket_summary, partition_files
from faultline.analyzer.workspace import WorkspaceInfo, WorkspacePackage
from faultline.llm.cost import CostTracker
from faultline.llm.sonnet_scanner import (
    DeepScanResult,
    build_commit_context,
    deep_scan,
    deep_scan_workspace,
)

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from faultline.analyzer.repo_classifier import RepoStructure
    from faultline.models.types import Commit

logger = logging.getLogger(__name__)


# Below this many packages a "workspace" isn't really a monorepo — fall
# back to the single-call path so the LLM still sees the whole repo.
_MIN_WORKSPACE_PACKAGES = 2


def run(
    *,
    analysis_files: list[str],
    workspace: "WorkspaceInfo | None",
    repo_structure: "RepoStructure | None",
    signatures: dict | None = None,
    commits: "list[Commit] | None" = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: CostTracker | None = None,
    commit_context_top_n: int = 30,
    commit_context_days: int = 90,
    use_tools: bool = False,
    repo_root=None,  # pathlib.Path; required when use_tools=True
    dedup: bool = False,
) -> DeepScanResult | None:
    """Run the new feature detection pipeline against a single repo.

    Args:
        analysis_files: All file paths to consider, relative to the
            analysis root (already stripped of any ``--src`` prefix by
            the caller).
        workspace: Optional ``WorkspaceInfo`` from
            ``analyzer.workspace.detect_workspace``. When ``detected``
            is True and at least :data:`_MIN_WORKSPACE_PACKAGES` real
            packages exist, the per-package path is taken.
        repo_structure: Optional ``RepoStructure`` from
            ``analyzer.repo_classifier.classify_repo``. Used purely for
            its ``is_library`` field, which suppresses flow generation
            on libraries (acceptance criterion C). When ``None`` the
            pipeline assumes the repo is an application.
        signatures: Optional AST-extracted signatures keyed by file path.
            Threaded through to ``deep_scan`` / ``deep_scan_workspace``
            so the LLM sees exports and routes for individual files.
        commits: Recent commits used to build the ``## Recent activity``
            context block injected into every per-package LLM call.
            Bounded by ``commit_context_top_n`` so it never blows the
            token budget.
        api_key: Anthropic API key. ``None`` falls back to the
            ``ANTHROPIC_API_KEY`` environment variable inside ``deep_scan``.
        model: Override Sonnet model id. ``None`` uses the module
            default in ``sonnet_scanner._MODEL``.
        tracker: Shared ``CostTracker`` for budget enforcement and
            end-of-run cost reporting. When omitted, calls run
            uncounted.
        commit_context_top_n: Maximum number of file/dir entries in the
            recent-activity block. Default 30.
        commit_context_days: Lookback window for commit context, in
            days. Default 90.

    Returns:
        A ``DeepScanResult`` containing features, flows, descriptions,
        flow descriptions, and (when a tracker is provided) the cost
        summary. Returns ``None`` only when both dispatch paths fail
        outright — partial failures within ``deep_scan_workspace``
        fall back to a single feature per package, never None.
    """
    is_library = bool(getattr(repo_structure, "is_library", False))
    if is_library:
        logger.info(
            "pipeline: repo classified as library — flows will be suppressed",
        )

    commit_context = build_commit_context(
        commits,
        top_n=commit_context_top_n,
        days=commit_context_days,
    )
    if commit_context:
        logger.info(
            "pipeline: built commit context (%d lines)",
            len(commit_context.splitlines()),
        )

    # Stage 1: Partition every file into one of five buckets. Only SOURCE
    # files proceed to LLM detection. Docs / infra materialize as synthetic
    # features below; tests and generated never reach the LLM.
    partition = partition_files(analysis_files)
    logger.info("pipeline: bucketizer partition: %s", bucket_summary(partition))

    source_files = partition[Bucket.SOURCE]
    doc_files = partition[Bucket.DOCUMENTATION]
    infra_files = partition[Bucket.INFRASTRUCTURE]

    if _should_use_workspace_path(workspace):
        filtered_workspace = _filter_workspace_sources(workspace, source_files)
        logger.info(
            "pipeline: workspace path (%d packages, %d source files)",
            len(filtered_workspace.packages),
            sum(len(p.files) for p in filtered_workspace.packages),
        )
        result = deep_scan_workspace(
            filtered_workspace,
            api_key=api_key,
            model=model,
            signatures=signatures,
            is_library=is_library,
            tracker=tracker,
            commit_context=commit_context,
            use_tools=use_tools,
            repo_root=repo_root,
        )
    else:
        logger.info(
            "pipeline: single-call path (%d source files, %d docs filtered out)",
            len(source_files),
            len(doc_files),
        )
        result = _run_single_call(
            analysis_files=source_files,
            signatures=signatures,
            api_key=api_key,
            model=model,
            tracker=tracker,
            is_library=is_library,
            commit_context=commit_context,
        )

    if result is None:
        return None

    # Stage 1.5 (Sprint 2): Cross-cluster dedup — collapse semantically
    # identical features that ended up split across packages (e.g. on
    # documenso "lib/document-signing" + "remix/document-signing" +
    # "trpc/document-signing" all describe the same product domain).
    # Runs BEFORE the docs/infra synthetic buckets are materialized
    # so those protected names never appear in the dedup input.
    if dedup:
        from faultline.llm.dedup import dedup_features
        before = len(result.features)
        result = dedup_features(
            result,
            api_key=api_key,
            model=model,
            tracker=tracker,
        )
        after = len(result.features)
        if after < before:
            logger.info(
                "pipeline: dedup collapsed %d → %d features (-%d)",
                before, after, before - after,
            )

    # Stage 2: Materialize synthetic features for non-source buckets.
    # deep_scan has its own internal docs partition as defensive fallback;
    # by pre-filtering here we make the bucketizer the single source of
    # truth. Adding to setdefault merges cleanly either way.
    if doc_files:
        result.features.setdefault("documentation", []).extend(doc_files)
        result.features["documentation"] = sorted(
            set(result.features["documentation"])
        )
        result.descriptions.setdefault(
            "documentation",
            "Documentation, tutorials, examples, and marketing pages.",
        )
        logger.info(
            "pipeline: materialized documentation feature (%d files)",
            len(result.features["documentation"]),
        )
    if infra_files:
        result.features.setdefault("shared-infra", []).extend(infra_files)
        result.features["shared-infra"] = sorted(
            set(result.features["shared-infra"])
        )
        logger.info(
            "pipeline: merged infrastructure into shared-infra (%d new files)",
            len(infra_files),
        )

    # Stage 3: Orphan validation. Every SOURCE file must land in exactly
    # one feature. Anything missing is a bug we want to surface, not a
    # silent fallback into shared-infra.
    _validate_source_coverage(source_files, result.features)

    return result


def _filter_workspace_sources(
    workspace: "WorkspaceInfo",
    source_files: list[str],
) -> "WorkspaceInfo":
    """Return a copy of ``workspace`` with non-source files stripped.

    Packages that end up empty are dropped — there is no point feeding
    the per-package LLM a bucket of zero source files.
    """
    source_set = set(source_files)
    filtered_packages: list[WorkspacePackage] = []
    for pkg in workspace.packages:
        pkg_sources = [f for f in pkg.files if f in source_set]
        if pkg_sources:
            filtered_packages.append(
                WorkspacePackage(name=pkg.name, path=pkg.path, files=pkg_sources)
            )
    return WorkspaceInfo(
        detected=workspace.detected,
        manager=workspace.manager,
        packages=filtered_packages,
        root_files=[f for f in workspace.root_files if f in source_set],
    )


def _validate_source_coverage(
    source_files: list[str],
    features: dict[str, list[str]],
) -> None:
    """Log a warning for any SOURCE file not attributed to any feature.

    The legacy behaviour was a silent fold into ``shared-infra`` via the
    ``_fold_stragglers_into_infra`` helper — that made quality regressions
    invisible. Here we surface orphans as a log warning AND still
    attribute them to ``shared-infra`` so downstream consumers don't
    choke on missing files. The two goals are not in tension: the user
    gets a working feature map AND we get a diagnostic signal.
    """
    attributed: set[str] = set()
    for paths in features.values():
        attributed.update(paths)
    orphans = sorted(set(source_files) - attributed)
    if not orphans:
        return
    logger.warning(
        "pipeline: %d source files not attributed to any feature — "
        "falling back into shared-infra. Sample (first 10):",
        len(orphans),
    )
    for path in orphans[:10]:
        logger.warning("  orphan: %s", path)
    features.setdefault("shared-infra", []).extend(orphans)
    features["shared-infra"] = sorted(set(features["shared-infra"]))


def _should_use_workspace_path(workspace: "WorkspaceInfo | None") -> bool:
    """Workspace dispatch needs an actual monorepo, not a stray marker.

    A repo with ``packages/`` containing one package is structurally
    a monolith and benefits from the single-call path: the LLM sees
    everything at once and can name features across the whole tree.
    Two or more packages is the threshold where per-package isolation
    starts paying off.
    """
    if workspace is None or not workspace.detected:
        return False
    return len(workspace.packages) >= _MIN_WORKSPACE_PACKAGES


def _run_single_call(
    *,
    analysis_files: list[str],
    signatures: dict | None,
    api_key: str | None,
    model: str | None,
    tracker: CostTracker | None,
    is_library: bool,
    commit_context: str | None,
) -> DeepScanResult | None:
    """Single-call dispatch: build candidates, hand them to ``deep_scan``.

    This is the path for monoliths, libraries, and small workspace repos
    that don't justify per-package isolation. Mirrors the legacy
    candidate-based strategy from ``cli.py:469-480`` but without the
    five-way conditional and the post-processing scattered across
    detector.py.
    """
    # Imported lazily so the pipeline module stays cheap to import in
    # test contexts that don't need the heuristic detector.
    from faultline.analyzer.features import detect_candidates

    candidates = detect_candidates(analysis_files)
    return deep_scan(
        analysis_files,
        candidates,
        api_key=api_key,
        signatures=signatures,
        is_library=is_library,
        model=model,
        tracker=tracker,
        commit_context=commit_context,
    )
