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
    smart_aggregators: bool = False,  # Sprint 9: agentic 4-bucket classifier
    flow_judge: bool = True,  # Sprint 11: Haiku-judged flow re-attribution
    flow_cluster: bool = True,  # Sprint 12: virtual cluster promotion (Layer A)
    flow_symbols: bool = True,  # Sprint 12: per-flow symbol resolution (Layer B)
    flow_sweep: bool = True,  # Sprint 12: entry-point sweep + cross-val (Layer C)
    flow_resignal: bool = True,  # Sprint 13: re-judge with deterministic signals
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

    # Stage 1.97 (Sprint 9): Agentic aggregator classifier. The LLM
    # gets read-only tools (read_file_head, list_directory,
    # consumers_of, feature_summary, ...) and investigates each
    # suspicious feature before classifying it into product /
    # shared-aggregator / developer-internal / tooling-infra.
    # Replaces Sprint 8's single-shot classifier (which never picked
    # shared-aggregator on canonical cases like dify Contracts —
    # the model couldn't see who imported what).
    #
    # Library mode SKIPS this stage entirely: every workspace package
    # in a library is a public API module, not infrastructure (Day 5
    # of Sprint 8 lesson).
    if smart_aggregators and is_library:
        logger.info(
            "smart_aggregators: skipped — library mode active. Each "
            "workspace package is a library module, not infrastructure."
        )
    elif smart_aggregators:
        try:
            from faultline.llm.aggregator_agent import agentic_classify_features
            from faultline.llm.aggregator_apply import apply_classifications
            from faultline.analyzer.aggregator_consumers import find_consumers

            # Build SymbolGraph for consumer resolution and tool access.
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
                        "agent will fall back to filename heuristics",
                        exc,
                    )

            import os
            import anthropic
            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not resolved_key:
                raise RuntimeError("no API key for agentic classifier")
            client = anthropic.Anthropic(api_key=resolved_key)

            classifications = agentic_classify_features(
                result,
                repo_root=repo_root,
                symbol_graph=sym_graph,
                client=client,
                model=model or "claude-sonnet-4-6",
                tracker=tracker,
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

            before = len(result.features)
            result = apply_classifications(
                result, classifications, consumer_maps,
            )
            after = len(result.features)
            logger.info(
                "smart_aggregators: %d → %d features after agentic classification",
                before, after,
            )
        except Exception as exc:  # noqa: BLE001 — opportunistic, never block
            logger.warning(
                "smart_aggregators: stage failed (%s) — keeping pre-Sprint-9 result",
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

    # Stage 2.5: Final same-name collapse. Stage 1.45 already ran an
    # exact-name dedup early in the pipeline, but later stages
    # (sub_decompose, smart_aggregators, repo_config aliasing) can
    # re-introduce duplicate display names. Sub_decompose splitting
    # two distinct parents into similarly-named children is the
    # common offender — n8n's Credentials × 2 (backend defs +
    # frontend api) and plane's Issues × 2 (workspace + public
    # space) showed up in the Sprint 9 May 5 scans.
    #
    # Path union is the right call when both share a coherent
    # business domain. When they don't (plane's two Issues live in
    # different deployable apps), the merge does scatter — that's
    # the known trade-off documented in the bug investigation.
    # Smart detection (path-disjoint → rename instead of merge) is
    # tracked as a future follow-up.
    result = _collapse_same_name_features(result)

    # Stage 2.55 (Sprint 12 Day 3): Virtual cluster promotion. When the
    # detector produces a coherent set of flows for a domain (auth,
    # billing, notifications) but the feature menu has no matching
    # feature, those flows attach to whatever catch-all bucket happens
    # to own their entry-point file (i18n, ui shells, contracts).
    # ``flow_judge`` (Stage 2.6) cannot fix this — it only moves flows
    # between *existing* features. ``promote_virtual_clusters`` carves
    # the auth-named files out of those buckets and creates a synthetic
    # ``auth`` feature for the orphaned flows to land on. Trigger
    # conditions are conservative (≥3 flows, no domain-named feature
    # already, ≥1 matching path); see ``faultline.llm.flow_cluster``.
    if flow_cluster and not is_library:
        try:
            from faultline.llm.flow_cluster import promote_virtual_clusters
            created = promote_virtual_clusters(result)
            if created:
                logger.info(
                    "flow_cluster: promoted %d synthetic feature(s)",
                    created,
                )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning("flow_cluster: stage failed (%s) — skipping", exc)

    # Stage 2.6: Flow re-attribution. Some flows land on the wrong
    # feature because the flow detector attaches flows by entry-point
    # file ownership — auth UI components might be in feature
    # ``Auth`` while the route that triggers them lives in the parent
    # ``Studio`` app shell. Result: ``Auth`` has 38 files but 0
    # flows; flows like ``Authenticate with Password`` are stranded
    # on ``Studio`` or ``Vue Blocks``.
    #
    # Two-tier approach (Sprint 11):
    #   1. Haiku judge — sees every flow + the full feature menu
    #      and picks the best-fit owner. Catches semantic synonyms
    #      the heuristic misses (``Sign in via OAuth`` → Auth even
    #      without shared substring) and rejects false positives.
    #      Costs ~$0.20 per scan; only acts on confidence ≥4.
    #   2. Heuristic fallback — substring/token match. Runs when
    #      no API key, anthropic package missing, or judge call
    #      fails. Cheap ~30-50% recovery on the obvious cases.
    judge_ran = False
    if flow_judge:
        try:
            from faultline.llm.flow_judge import judge_flow_attribution
            _slug = None
            if repo_root is not None:
                try:
                    _slug = repo_root.name
                except Exception:  # noqa: BLE001
                    _slug = None
            moves = judge_flow_attribution(
                result,
                api_key=api_key,
                tracker=tracker,
                repo_slug=_slug,
            )
            judge_ran = True
            logger.info("flow_judge: applied %d high-confidence moves", moves)
        except Exception as exc:  # noqa: BLE001 — opportunistic; never block scan
            logger.warning(
                "flow_judge: stage failed (%s) — falling back to heuristic",
                exc,
            )
    if not judge_ran:
        _reattribute_flows_by_name_match(result)

    # Stage 2.7 (Sprint 12 Day 5): Per-flow symbol resolution. For
    # every flow, ask Haiku which exported symbols actually
    # participate in the user journey, then resolve to {file,
    # start_line, end_line} via ast_extractor. The output lives in
    # ``result.flow_participants`` and the CLI builds it into
    # ``Flow.participants[].symbols`` at FeatureMap-build time.
    #
    # This is the foundation for per-flow test-coverage attribution:
    # once we know which symbols a flow exercises and which line
    # ranges they occupy, we can intersect with coverage data to
    # surface real per-flow coverage % instead of feature averages.
    #
    # Skipped on libraries (no flows there). Best-effort — wraps in
    # try/except so a Haiku outage never blocks the scan.
    if flow_symbols and not is_library and repo_root is not None:
        try:
            from faultline.llm.flow_symbols import resolve_flow_symbols
            populated = resolve_flow_symbols(
                result,
                source_loader=_make_source_loader(repo_root),
                api_key=api_key,
                tracker=tracker,
                repo_slug=repo_root.name if repo_root else None,
            )
            if populated:
                logger.info(
                    "flow_symbols: populated %d flow(s) with symbol ranges",
                    populated,
                )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning(
                "flow_symbols: stage failed (%s) — flows will keep file-level "
                "participants from prior stages",
                exc,
            )

    # Stage 2.75 (Sprint 13 Day 1): Re-judge with deterministic
    # signals. After Layer B populated flow_participants with concrete
    # symbol-bearing files, we now know each flow's actual file
    # ownership distribution. Pick flows whose paths overwhelmingly
    # belong to a feature OTHER than their current owner (the
    # ``manage-billing-subscription``-in-``contracts`` class) and ask
    # Haiku once more — this time with the per-feature ownership
    # percentages in the prompt as evidence. Cheap (one batched call,
    # only flows with strong disagreement, typically <20 of 200).
    if flow_resignal and not is_library:
        try:
            from faultline.llm.flow_judge import re_judge_with_signals
            moves = re_judge_with_signals(
                result,
                api_key=api_key,
                tracker=tracker,
            )
            if moves:
                logger.info(
                    "flow_resignal: applied %d signal-based move(s)", moves,
                )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning("flow_resignal: stage failed (%s) — skipping", exc)

    # Stage 2.8 (Sprint 12 Day 6): Layer C — entry-point sweep + cross-
    # validation. Every route handler / exported handler-pattern symbol
    # in the repo is harvested. Anything not already covered by a flow
    # is sent to Haiku for clustering + naming, becoming a new flow
    # under the most plausible feature. A second cross-val Haiku pass
    # asks each feature "which neighbour flows ALSO touch you?" and
    # records secondary claims into ``flow_secondaries`` (the same
    # channel the primary judge populates via ``also_belongs_to``).
    #
    # Skipped on libraries (no flows). Wraps in try/except so any
    # failure logs and falls through.
    if flow_sweep and not is_library and signatures:
        try:
            from faultline.llm.flow_sweep import run_layer_c
            counts = run_layer_c(
                result,
                signatures=signatures,
                api_key=api_key,
                tracker=tracker,
            )
            logger.info(
                "flow_sweep: harvested=%d unattached=%d promoted=%d cross_val=%d",
                counts.get("harvested", 0),
                counts.get("unattached", 0),
                counts.get("promoted", 0),
                counts.get("cross_val_claims", 0),
            )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning("flow_sweep: stage failed (%s) — skipping", exc)

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


def _make_source_loader(repo_root):
    """Closure: rel_path → source text (or None on failure).

    Sprint 12 Day 5 — feeds flow_symbols. Reads files lazily from the
    repo working tree. Skips files >256 KB (binary / minified) so the
    Haiku prompt doesn't drown in transitive bundle output.
    """
    from pathlib import Path

    root = Path(repo_root) if not isinstance(repo_root, Path) else repo_root
    _MAX_BYTES = 256 * 1024

    def _load(rel: str) -> str | None:
        try:
            path = root / rel
            if not path.is_file():
                return None
            if path.stat().st_size > _MAX_BYTES:
                return None
            return path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            return None

    return _load


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
    # Normalize on the LAST path-segment so sub-decompose's slash-
    # prefixed children (e.g. ``parent/credentials``) collapse with
    # plain ``credentials``. n8n's May 5 v2 scan slipped 3 duplicate
    # pairs through (AI Workflow Builder × 2, Instance AI × 2,
    # Credentials × 2) precisely because Stage 2.5 was lowercase-only
    # and treated ``parent/credentials`` as different from
    # ``credentials``. Last-segment normalization fixes that.
    by_norm: dict[str, list[str]] = {}
    for name in result.features:
        if name in _PROTECTED_BUCKETS:
            continue
        norm = name.strip().lower().rsplit("/", 1)[-1]
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


_NAME_REATTRIBUTION_TOKEN_MIN = 4  # min chars for a feature-name token to count
_NAME_REATTRIBUTION_PROTECTED: frozenset[str] = frozenset({
    "shared-infra", "documentation", "examples",
    "developer-infrastructure",
})


def _normalize_token(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _flow_tokens(flow_name: str) -> set[str]:
    """Tokenize a flow name into normalized lowercase tokens.

    ``Authenticate with Password`` → {"authenticate", "with", "password"}
    ``manage-roles-flow`` → {"manage", "roles", "flow"}
    """
    import re
    parts = re.split(r"[\s\-_/]+", flow_name)
    return {_normalize_token(p) for p in parts if p}


def _name_matches_flow(feature_name: str, flow_tokens: set[str]) -> int:
    """Score how strongly a feature name matches a flow's tokens.

    Returns:
        0 — no match
        1 — feature-name appears as a SUBSTRING of any token (loose match,
            e.g. ``auth`` ⊂ ``authenticate``)
        2 — feature-name appears as a FULL token (strong match)
    """
    norm = _normalize_token(feature_name)
    if len(norm) < _NAME_REATTRIBUTION_TOKEN_MIN:
        return 0  # too short to be meaningful (avoids ``ai`` matching every flow)
    if norm in flow_tokens:
        return 2
    for tok in flow_tokens:
        if len(tok) >= _NAME_REATTRIBUTION_TOKEN_MIN and (
            norm in tok or tok in norm
        ):
            return 1
    return 0


def _reattribute_flows_by_name_match(result: DeepScanResult) -> None:
    """Move flows to features whose name better matches the flow's tokens.

    Solves the ``supabase Auth feature has 38 files but 0 flows while
    ``Authenticate with Password`` is stuck on Vue Blocks`` problem —
    flow detector attaches flows by entry-point file ownership which
    misses cases where UI components for a domain are split across
    multiple features.

    Conservative scoring: a destination feature must score AT LEAST 2
    (full-token match) AND beat the current owner's score by ≥1 to
    take a flow. Loose substring matches alone (score 1) don't move
    flows — they're informative but too noisy on their own.

    Mutates ``result.flows`` and ``result.flow_descriptions`` /
    ``result.flow_participants`` in place. ``result.features`` (the
    feature → owned-paths map) is untouched.
    """
    feature_names = [
        n for n in result.features
        if n not in _NAME_REATTRIBUTION_PROTECTED
    ]
    if not feature_names:
        return

    moved = 0
    for current_owner in list(result.flows.keys()):
        flow_list = result.flows.get(current_owner, [])
        if not flow_list:
            continue
        new_owner_flows: dict[str, list[str]] = {}  # destination → moved flows
        keep: list[str] = []

        for flow_name in flow_list:
            tokens = _flow_tokens(flow_name)
            current_score = _name_matches_flow(current_owner, tokens)

            # Find best alternate feature
            best_alt = None
            best_alt_score = 0
            for fname in feature_names:
                if fname == current_owner:
                    continue
                score = _name_matches_flow(fname, tokens)
                if score > best_alt_score:
                    best_alt = fname
                    best_alt_score = score

            # Conservative move rule. Two cases qualify:
            #   (a) Strong match: alt scores ≥2 (full token) AND beats
            #       current by ≥1
            #   (b) Loose match: alt scores ≥1 AND current scores 0,
            #       i.e. flow has zero connection to its current owner
            #       but at least a substring match on alt. Catches the
            #       supabase Auth case (``auth`` ⊂ ``authenticate``).
            should_move = best_alt is not None and (
                (best_alt_score >= 2 and best_alt_score > current_score)
                or (best_alt_score >= 1 and current_score == 0)
            )
            if should_move:
                new_owner_flows.setdefault(best_alt, []).append(flow_name)
                moved += 1
            else:
                keep.append(flow_name)

        # Apply moves: trim current owner's flows, append to destinations
        result.flows[current_owner] = keep
        for dest, fnames in new_owner_flows.items():
            result.flows.setdefault(dest, []).extend(fnames)

        # Migrate flow descriptions + participants too
        for dest, fnames in new_owner_flows.items():
            for fn in fnames:
                src_descs = result.flow_descriptions.get(current_owner, {})
                if fn in src_descs:
                    result.flow_descriptions.setdefault(dest, {})[fn] = (
                        src_descs.pop(fn)
                    )
                src_parts = result.flow_participants.get(current_owner, {})
                if fn in src_parts:
                    result.flow_participants.setdefault(dest, {})[fn] = (
                        src_parts.pop(fn)
                    )

    if moved:
        logger.info("pipeline: re-attributed %d flow(s) by name match", moved)


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
    OR as a shared_participant of some feature (Sprint 8/9 redistribution).
    Without honoring the side-channel, every redistributed file would
    be flagged as orphan and silently re-folded into shared-infra,
    undoing the work.

    The legacy behaviour was a silent fold via ``_fold_stragglers_into_infra``
    that made quality regressions invisible. Here we surface true orphans
    as a log warning AND still attribute them to ``shared-infra`` so
    downstream consumers don't choke.
    """
    attributed: set[str] = set()
    for paths in features.values():
        attributed.update(paths)
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
