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
    dedup: bool = True,  # Fix #3: auto-on (was opt-in via --dedup)
    sub_decompose: bool = False,
    tool_flows: bool = False,
    critique: bool = False,
    trace_flows: bool = False,
    rename_generic: bool = False,  # Fix #4: REVERTED — Haiku too conservative
    smart_aggregators: bool = False,  # Sprint 8: classify features into 4 buckets
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

    # Stability hint: read canonical names from .faultline.yaml so
    # Sonnet can prefer them verbatim when naming detected features.
    preferred_names: list[str] = []
    locked_canonicals: frozenset[str] = frozenset()
    if repo_root is not None:
        try:
            from faultline.analyzer.repo_config import load_repo_config
            _cfg_pre = load_repo_config(repo_root)
            if _cfg_pre is not None:
                locked_canonicals = _cfg_pre.all_canonical_names()
                # Top-level canonicals only — sub-feature locks
                # (with ``/``) are handled later by parent-collapse.
                preferred_names = sorted(
                    n for n in locked_canonicals if "/" not in n
                )
        except Exception:  # noqa: BLE001 — opportunistic
            pass

    # Step 2 stability: load file → previous-canonical map. Used after
    # deep_scan returns to renormalize fresh feature names whose file
    # set overwhelmingly belongs to a previous canonical.
    prev_assignments: dict[str, str] = {}
    if repo_root is not None:
        try:
            from faultline.analyzer.assignments import load_assignments
            prev_assignments = load_assignments(repo_root)
            if prev_assignments:
                logger.info(
                    "pipeline: loaded %d prior file→feature assignments",
                    len(prev_assignments),
                )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.debug("pipeline: load_assignments skipped (%s)", exc)

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
            preferred_names=preferred_names or None,
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
            preferred_names=preferred_names or None,
        )

    if result is None:
        return None

    # Stage 1.35 (Step 2): Renormalize fresh feature names against the
    # previous scan's file → canonical map. If a detected feature has
    # ≥60% of its files previously assigned to a single locked canonical,
    # rename it to that canonical. Catches the cases where Sonnet
    # ignored the prompt-level preferred_names hint (Step 1).
    if prev_assignments and locked_canonicals:
        try:
            from faultline.analyzer.assignments import renormalize_features
            n = renormalize_features(
                result, prev_assignments,
                locked_canonicals=locked_canonicals,
            )
            if n:
                logger.info(
                    "pipeline: renormalized %d features via assignment cache",
                    n,
                )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning("pipeline: renormalize_features failed (%s)", exc)

    # Stage 1.4: Auto-fold universally-tooling packages into
    # shared-infra. Things like ``tsconfig``, ``eslint-config``,
    # ``prettier-config`` are tooling in ~99% of repos — they should
    # never show up as a product feature on the dashboard. Always
    # runs (no flag, no user config required); users can override
    # by listing the name in ``.faultline.yaml`` features (which
    # then aliases it to a real canonical, taking precedence).
    _auto_fold_tooling(result)

    # Stage 1.45: Same-name auto-merge. When per-package Sonnet calls
    # in ``deep_scan_workspace`` independently produce the same display
    # name across packages (n8n's ``Credentials`` × 3 across cli/core/
    # nodes-base, strapi's ``Admin`` × 2, ``Content Manager`` × 2),
    # the merge phase concatenates them as separate entries. This
    # deterministic pass folds same-named features by union of paths,
    # picking the variant with the most paths as canonical. Always
    # runs — no LLM cost, no flag, fixes the obvious case before the
    # opt-in semantic ``dedup`` ever runs.
    result = _collapse_same_name_features(result)

    # Stage 1.5 (Sprint 2): Cross-cluster dedup — collapse semantically
    # identical features that ended up split across packages (e.g. on
    # documenso "lib/document-signing" + "remix/document-signing" +
    # "trpc/document-signing" all describe the same product domain).
    # Runs BEFORE the docs/infra synthetic buckets are materialized
    # so those protected names never appear in the dedup input.
    #
    # Was opt-in (--dedup flag); promoted to always-on as Fix #3 from
    # the Fixable-accuracy work. Stage 1.45 (_collapse_same_name) now
    # handles exact duplicates deterministically with no LLM cost; the
    # remaining job for dedup is the harder semantic case (n8n's
    # Editor / Workflow Sdk / Workflow Index, dify's Workflow App /
    # Workflow / Web / Ui). Cap raised to 50 (was 12) to handle the
    # n8n-scale fragmentation. Caller can opt out by passing
    # ``dedup=False`` if they need to skip the LLM call.
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

    # Stage 1.55 was a Haiku batch rename for generic feature names
    # (Fix #4 from Fixable-accuracy work). Reverted May 2026 after
    # validation across 4 repos showed Haiku returned ``KEEP`` on
    # every flagged feature — the candidate-selection logic was
    # right, but the model was too conservative to actually rename
    # anything. Naming auto-rule moved -1pp net. The module
    # ``faultline.llm.rename_generic`` and its tests stay in tree
    # for future iteration (stronger prompt, Sonnet, or LLM-as-
    # judge approach), but it's no longer wired into ``run()``.

    # Stage 1.6: Apply user-supplied .faultline.yaml from repo root.
    # Always runs (no flag) so users can opt in by simply dropping a
    # config file into their repo. Force-merges and canonical
    # aliasing come from the user; engine stays neutral.
    if repo_root is not None:
        try:
            from faultline.analyzer.repo_config import (
                apply_repo_config, load_repo_config,
            )
            user_cfg = load_repo_config(repo_root)
            if user_cfg is not None:
                result = apply_repo_config(result, user_cfg)
                logger.info(
                    "pipeline: applied repo config from %s",
                    user_cfg.source_path,
                )
        except ValueError as exc:
            logger.error("pipeline: repo config error — %s", exc)

    # Stage 1.7 (Sprint 3): Sub-decomposition of oversized features.
    # Runs after dedup (so we see the post-merge size) and before
    # synthetic-bucket materialization (so docs / shared-infra never
    # become candidates). Splits any feature above 200 files into
    # 2-6 sub-features when the LLM proposes a clean split, otherwise
    # leaves the parent intact.
    if sub_decompose:
        from faultline.llm.sub_decompose import sub_decompose_oversized
        # Stability lock: skip sub-decomposing features whose names
        # are already canonical in .faultline.yaml (user features
        # OR auto_aliases). Prevents Sprint 3 from generating fresh
        # sub-feature names every run on the same parent — the
        # other half of the drift Sprint 5 fix uncovered.
        sub_locked: frozenset[str] = frozenset()
        if repo_root is not None:
            try:
                from faultline.analyzer.repo_config import load_repo_config
                _cfg = load_repo_config(repo_root)
                if _cfg is not None:
                    base = _cfg.all_canonical_names()
                    # Also lock derived parents: if "ui/foo" is canonical,
                    # then "ui" itself must not be re-decomposed into
                    # different children on the next run.
                    derived = {
                        n.split("/", 1)[0]
                        for n in base
                        if "/" in n
                    }
                    sub_locked = frozenset(base | derived)
            except Exception:  # noqa: BLE001 — opportunistic
                pass
        before = len(result.features)
        result = sub_decompose_oversized(
            result,
            api_key=api_key,
            model=model,
            tracker=tracker,
            repo_root=repo_root,
            locked_names=sub_locked,
        )
        after = len(result.features)
        if after != before:
            logger.info(
                "pipeline: sub-decompose %d → %d features (+%d)",
                before, after, after - before,
            )

    # Stage 1.8 (Sprint 4): Tool-augmented flow detection. Replaces
    # the legacy Haiku per-feature call when ``tool_flows`` is set.
    # Library mode skips this entirely. Synthetic buckets
    # (documentation / shared-infra / examples) are protected.
    if tool_flows:
        from faultline.llm.flow_detector_v2 import detect_flows_with_tools
        result = detect_flows_with_tools(
            result,
            repo_root=repo_root,
            is_library=is_library,
            api_key=api_key,
            model=model,
            tracker=tracker,
        )

    # Stage 1.9 (Sprint 5): Self-critique loop. Single Sonnet pass
    # flags weak names; up to 5 are re-investigated with tools and
    # renamed when the proposal is materially better than the
    # original. Opportunistic: any error returns the previous result
    # unchanged.
    # Stage 1.95 (Sprint 7): Call-graph flow trace. For every flow
    # detected by Sprint 4, BFS through the import graph from the
    # entry-point file:line and record every UI / state / API /
    # schema file that participates. No LLM calls — pure local
    # static analysis. Result is stashed on
    # ``result.flow_participants`` for the CLI injector to attach
    # to ``Flow.participants`` Pydantic objects downstream.
    if trace_flows and repo_root is not None:
        try:
            from faultline.analyzer.flow_tracer import trace_flow_callgraph
            result.flow_participants = trace_flow_callgraph(
                result, repo_root,
            )
            logger.info(
                "pipeline: trace_flows produced %d feature trace(s)",
                len(result.flow_participants),
            )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning("pipeline: trace_flows failed (%s) — skipping", exc)

    if critique:
        from faultline.llm.critique import critique_and_refine
        # Lock canonical names from the user's .faultline.yaml so
        # critique can't rename them — keeps scan output stable
        # across runs. Includes both user-managed ``features:`` and
        # engine-written ``auto_aliases:`` (Improvement #4).
        locked: frozenset[str] = frozenset()
        if repo_root is not None:
            try:
                from faultline.analyzer.repo_config import load_repo_config
                _cfg = load_repo_config(repo_root)
                if _cfg is not None:
                    locked = _cfg.all_canonical_names()
            except Exception:  # noqa: BLE001 — opportunistic
                pass
        result = critique_and_refine(
            result,
            repo_root=repo_root,
            api_key=api_key,
            model=model,
            tracker=tracker,
            locked_names=locked,
        )

    # Stage 1.97 (Sprint 8): Smart aggregator detection. One Sonnet
    # call classifies every feature into product / shared-aggregator /
    # developer-internal / tooling-infra. Aggregator features are
    # deleted; their files redistribute as shared_participants on
    # consuming product features (so a Button.tsx from shared-ui shows
    # up on every feature that uses it). Developer-internal features
    # rename to plain English (i18n → Translations) or fold into a
    # 'developer-infrastructure' bucket. Tooling-infra folds to
    # shared-infra. Runs AFTER trace_flows so flow re-attribution can
    # vote on Sprint 7 callgraph participants. Off by default until
    # Day 6 eval validates the lift.
    if smart_aggregators and is_library:
        # Smart aggregator detection is NOT for library repos. In a
        # library, every workspace package is a library MODULE — part
        # of the product surface, not aggregator infrastructure. The
        # excalidraw v8c run on May 5 illustrated the failure: the
        # classifier folded Math (19f), Common (24f), Utils (11f),
        # Renderer (16f) as developer-internal because they're small
        # by app-repo standards. They are not — they're the library.
        # Skip the stage in library mode to avoid demolishing the
        # public API surface of the library.
        logger.info(
            "smart_aggregators: skipped — library mode active. Each "
            "workspace package is a library module, not infrastructure."
        )
    elif smart_aggregators:
        try:
            from faultline.llm.aggregator_detector import classify_features
            from faultline.llm.aggregator_apply import apply_classifications
            from faultline.analyzer.aggregator_consumers import find_consumers

            classifications = classify_features(
                result, api_key=api_key, model=model, tracker=tracker,
            )

            # Build per-aggregator consumer maps. Symbol graph is
            # opportunistic — when --trace-flows wasn't used we don't
            # have one cached, so consumer resolution falls back to
            # filename heuristics inside find_consumers.
            sym_graph = None
            if repo_root is not None:
                try:
                    from faultline.analyzer.symbol_graph import build_symbol_graph
                    sym_graph = build_symbol_graph(
                        repo_root,
                        list({p for paths in result.features.values() for p in paths}),
                    )
                except Exception as exc:  # noqa: BLE001 — opportunistic
                    logger.warning(
                        "smart_aggregators: symbol graph build failed (%s) — "
                        "falling back to filename heuristic",
                        exc,
                    )

            consumer_maps: dict[str, dict[str, list[str]]] = {}
            for feat_name, verdict in classifications.items():
                if verdict.classification != "shared-aggregator":
                    continue
                files = list(result.features.get(feat_name, []))
                consumer_maps[feat_name] = find_consumers(
                    files,
                    aggregator_feature=feat_name,
                    result=result,
                    symbol_graph=sym_graph,
                )

            # Build a commit-count proxy from the available signal.
            # We don't have build_feature_map output here yet (that
            # runs in cli.py post-pipeline), but classifier-stage
            # safeguards only need a rough commit count to gate
            # large-feature folds. Use the file count as the proxy
            # ceiling — a feature with 200 files is large enough
            # that we treat it as "not foldable" regardless of
            # commit count. The real commit count check in
            # _is_too_large_to_fold needs an actual map; fall back
            # to None and rely on the file-count guard alone here.
            commit_counts: dict[str, int] | None = None

            before = len(result.features)
            result = apply_classifications(
                result, classifications, consumer_maps,
                commit_counts=commit_counts,
            )
            after = len(result.features)
            logger.info(
                "smart_aggregators: %d → %d features after classification",
                before, after,
            )
        except Exception as exc:  # noqa: BLE001 — opportunistic, never block
            logger.warning(
                "smart_aggregators: stage failed (%s) — keeping pre-Sprint-8 result",
                exc,
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
    _validate_source_coverage(
        source_files,
        result.features,
        shared_participants_map=getattr(result, "shared_participants_map", None),
    )

    # Stage 4 (Improvement #4): Auto-save discovered canonical names
    # back to ``.faultline.yaml`` so subsequent runs lock them
    # against Sprint 5 critique renaming. Only runs on repos that
    # already have a config file (we don't litter every scan target
    # with a new yaml file). Best-effort — never blocks the scan.
    if repo_root is not None:
        try:
            from faultline.analyzer.repo_config import auto_save_canonicals
            # write_if_missing=True so the first scan creates the yaml.
            # Without this, ``--incremental`` second-scan has no
            # canonical lock to anchor naming against — names drift
            # and the file-assignment cache underperforms.
            auto_save_canonicals(
                repo_root, result.features, result.descriptions,
                write_if_missing=True,
            )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning("pipeline: auto-save canonicals failed (%s)", exc)

        # Step 2 stability: persist this scan's file → feature
        # mapping so the NEXT scan can renormalize against it.
        try:
            from faultline.analyzer.assignments import save_assignments
            save_assignments(result, repo_root)
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning("pipeline: save_assignments failed (%s)", exc)

    return result


_PROTECTED_BUCKETS = frozenset({"shared-infra", "documentation", "examples"})


def _collapse_same_name_features(result: DeepScanResult) -> DeepScanResult:
    """Merge features that share a normalized display name.

    Workspace per-package scans frequently produce the same business
    name across multiple packages — n8n's ``Credentials`` appears in
    cli/core/nodes-base; strapi's ``Admin`` and ``Content Manager``
    each appear twice. The opt-in semantic ``dedup`` pass would catch
    these, but only if explicitly enabled. This deterministic pre-pass
    fixes the obvious case (exact name match, case-insensitive) for
    every scan with no LLM cost.

    Canonical pick rule: the variant with the most paths wins; on tie,
    alphabetical name order. Paths are union'd, deduplicated, and
    sorted so two consecutive runs produce byte-identical output.

    Synthetic buckets (``shared-infra``, ``documentation``, ``examples``)
    are never merge candidates even if they happened to coincide.
    """
    by_norm: dict[str, list[str]] = {}
    for name in result.features:
        if name in _PROTECTED_BUCKETS:
            continue
        norm = name.strip().lower()
        by_norm.setdefault(norm, []).append(name)

    if all(len(v) == 1 for v in by_norm.values()):
        return result  # no duplicates to collapse

    collapsed = 0
    for norm, names in sorted(by_norm.items()):
        if len(names) <= 1:
            continue
        # Canonical: most paths wins, alphabetical tiebreak (deterministic)
        canonical = max(names, key=lambda n: (len(result.features[n]), -ord(n[0]) if n else 0))
        # Alphabetical tiebreak: re-sort if multiple names tie on path count
        max_paths = len(result.features[canonical])
        tied = [n for n in names if len(result.features[n]) == max_paths]
        if len(tied) > 1:
            canonical = sorted(tied)[0]

        # Union all paths, dedupe, sort
        all_paths: set[str] = set()
        for n in names:
            all_paths.update(result.features[n])
        result.features[canonical] = sorted(all_paths)

        # Drop the non-canonical entries from every side channel
        for n in names:
            if n == canonical:
                continue
            result.features.pop(n, None)
            result.descriptions.pop(n, None)
            result.flows.pop(n, None)
            result.flow_descriptions.pop(n, None)
            collapsed += 1

    if collapsed:
        logger.info(
            "pipeline: collapsed %d same-name duplicate feature(s)",
            collapsed,
        )
    return result


def _auto_fold_tooling(result: DeepScanResult) -> None:
    """Move universally-tooling packages into ``shared-infra``.

    Looks at every detected feature whose name (last path segment)
    matches :data:`faultline.llm.dedup.TOOLING_PACKAGE_NAMES` and
    folds its files into the ``shared-infra`` bucket. The original
    feature key disappears.

    No-op when no feature matches. Always runs — users do not need
    to enumerate these in ``.faultline.yaml`` skip_features.
    Universal tooling-by-package-name knowledge belongs in the
    engine, not in every user's repo config.
    """
    from faultline.llm.dedup import TOOLING_PACKAGE_NAMES

    folded: list[str] = []
    for name in list(result.features.keys()):
        last = name.rsplit("/", 1)[-1].lower()
        if last in TOOLING_PACKAGE_NAMES:
            files = result.features.pop(name)
            result.features.setdefault("shared-infra", []).extend(files)
            result.descriptions.pop(name, None)
            result.flows.pop(name, None)
            result.flow_descriptions.pop(name, None)
            folded.append(name)

    if folded:
        # Sort + dedup the shared-infra path list
        si = result.features.get("shared-infra", [])
        result.features["shared-infra"] = sorted(set(si))
        logger.info(
            "pipeline: auto-folded %d tooling package(s) into shared-infra: %s",
            len(folded), folded,
        )


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
    shared_participants_map: dict[str, list] | None = None,
) -> None:
    """Log a warning for any SOURCE file not attributed to any feature.

    A file is "covered" if it appears as an owned path in some feature
    OR as a shared_participant of some feature (Sprint 8 redistribution).
    Sprint 8 deletes aggregator features and re-attributes their files
    as shared_participants on consumers; without honoring the side-
    channel here, every redistributed file would be flagged as an
    orphan and silently re-folded into shared-infra, undoing the work.

    The legacy behaviour was a silent fold into ``shared-infra`` via the
    ``_fold_stragglers_into_infra`` helper — that made quality regressions
    invisible. Here we surface true orphans as a log warning AND still
    attribute them to ``shared-infra`` so downstream consumers don't
    choke on missing files. The two goals are not in tension: the user
    gets a working feature map AND we get a diagnostic signal.
    """
    attributed: set[str] = set()
    for paths in features.values():
        attributed.update(paths)
    # Sprint 8: a file consumed as a shared_participant counts as
    # covered. Each entry in shared_participants_map[feat] is either a
    # SharedParticipant Pydantic object (post-Day 4 wiring) or a dict /
    # tuple legacy form; pull the file_path defensively.
    if shared_participants_map:
        for participants in shared_participants_map.values():
            for p in participants:
                fp = (
                    getattr(p, "file_path", None)
                    or (p.get("file_path") if isinstance(p, dict) else None)
                )
                if fp:
                    attributed.add(fp)
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
    preferred_names: list[str] | None = None,
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
        preferred_names=preferred_names,
    )
