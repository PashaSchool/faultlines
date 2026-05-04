import logging
import re
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich import print as rprint

from faultline.analyzer.git import load_repo, get_commits, get_tracked_files, estimate_commits, estimate_duration, get_remote_url, DEFAULT_MAX_COMMITS
from faultline.analyzer.features import detect_features_from_structure, build_feature_map, build_flows_metrics, split_large_features
from faultline.analyzer.repo_classifier import classify_repo, build_layer_context
from faultline.output.reporter import print_report
from faultline.output.writer import write_feature_map
from faultline.llm.detector import _DEFAULT_OLLAMA_HOST, _DEFAULT_OLLAMA_MODEL

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="faultline",
    help="Analyze git history to map features and track technical debt",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    repo_path: str = typer.Argument(
        ".",
        help="Path to the git repository",
    ),
    days: int = typer.Option(
        365,
        "--days", "-d",
        help="Number of days of history to analyze",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save feature-map.json",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save feature-map.json to disk",
    ),
    top: int = typer.Option(
        3,
        "--top",
        help="Number of top risk features to highlight",
    ),
    llm: bool = typer.Option(
        False,
        "--llm",
        help="Use an LLM to assign semantic names to detected features (results are cached)",
        is_flag=True,
    ),
    provider: str = typer.Option(
        "anthropic",
        "--provider",
        help="LLM provider: anthropic, ollama, or deepseek",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help=(
            "Model name override. "
            "Anthropic default: claude-haiku-4-5. "
            "Ollama default: llama3.1:8b (recommended). "
            "DeepSeek default: deepseek-chat (V3). "
            "Other Ollama options: mistral-nemo:12b (best quality), qwen2.5:7b."
        ),
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    ),
    deepseek_key: Optional[str] = typer.Option(
        None,
        "--deepseek-key",
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    ),
    ollama_url: str = typer.Option(
        _DEFAULT_OLLAMA_HOST,
        "--ollama-url",
        help="Ollama server URL",
    ),
    src: Optional[str] = typer.Option(
        None,
        "--src",
        help="Subdirectory to focus analysis on, e.g. src/ or app/. Ignores everything outside.",
    ),
    max_commits: int = typer.Option(
        DEFAULT_MAX_COMMITS,
        "--max-commits",
        help="Maximum number of commits to analyze",
    ),
    tool_use: bool = typer.Option(
        False,
        "--tool-use",
        help=(
            "Sprint 1 (experimental): give the per-package LLM read-only "
            "tools (read_file_head, list_directory, grep_pattern, "
            "get_file_commits) so it names features from actual file "
            "contents instead of guessing from paths. Anthropic + workspace "
            "monorepos only. Disables flow detection (Sprint 4 territory)."
        ),
        is_flag=True,
    ),
    dedup: bool = typer.Option(
        True,
        "--dedup/--no-dedup",
        help=(
            "Run a single Sonnet pass that sees every feature at once and "
            "merges semantic duplicates split across packages (e.g. five "
            "document-signing-* features → one). Promoted from opt-in "
            "(Sprint 2) to default-on as Fix #3 from the Fixable-accuracy "
            "work — closes the workflow-concept-fragmentation gap that "
            "Fix #1 (deterministic same-name merge) cannot reach. Adds "
            "~$0.30 per scan; pass --no-dedup to skip."
        ),
    ),
    sub_decompose: bool = typer.Option(
        False,
        "--sub-decompose",
        help=(
            "Sprint 3 (experimental): after dedup, split any feature with "
            ">200 files into 2-6 sub-features via tool-augmented LLM. "
            "Adds ~$0.30-0.60 per scan when 3-4 features cross the "
            "threshold; free when none do."
        ),
        is_flag=True,
    ),
    tool_flows: bool = typer.Option(
        False,
        "--tool-flows",
        help=(
            "Sprint 4 (experimental): replace the legacy Haiku per-feature "
            "flow detection with a tool-augmented Sonnet pass. The model "
            "uses find_route_handlers / find_event_handlers to ground each "
            "flow in a real entry point (file:line). Library repos skip "
            "this pass. Adds ~$2-4 per scan."
        ),
        is_flag=True,
    ),
    critique: bool = typer.Option(
        False,
        "--critique",
        help=(
            "Sprint 5 (experimental): final pass that flags weak feature "
            "or flow names and re-investigates each with tools. Up to 5 "
            "renames per scan. Adds ~$0.50-1.00 per scan."
        ),
        is_flag=True,
    ),
    rename_generic: bool = typer.Option(
        False,
        "--rename-generic",
        help=(
            "Experimental Haiku batch rename for generic feature names. "
            "Reverted from default-on (May 2026) — Haiku returned KEEP "
            "for nearly every candidate, so the pass paid the LLM cost "
            "without delivering renames. Kept behind opt-in flag for "
            "future iteration with a stronger prompt or Sonnet."
        ),
        is_flag=True,
    ),
    trace_flows: bool = typer.Option(
        False,
        "--trace-flows",
        help=(
            "Sprint 7 (experimental): walk the import graph from each "
            "Sprint 4 flow's entry point and enumerate every UI / state / "
            "API / schema file that participates, with per-symbol line "
            "ranges. Pure local analysis (no LLM calls). Requires "
            "--tool-flows."
        ),
        is_flag=True,
    ),
    legacy: bool = typer.Option(
        False,
        "--legacy",
        help=(
            "Fall back to the pre-rewrite 5-strategy detection pipeline. "
            "The default is the new single-call Sonnet pipeline "
            "(faultline.llm.pipeline.run). Use this only if the new pipeline "
            "produces worse results on your repo."
        ),
        is_flag=True,
    ),
    post_process: bool = typer.Option(
        True,
        "--post-process/--no-post-process",
        help=(
            "Apply post-process cleanup pipeline after feature_map is built: "
            "merge sub-features, re-attribute noise files, refine path signal, "
            "extract overlooked top-dirs, mine commit-prefix vocabulary, drop "
            "noise/vendored/phantom/mega-bucket. Default ON. Pass "
            "--no-post-process for raw scan output."
        ),
    ),
    line_attribution: bool = typer.Option(
        True,
        "--line-attribution/--no-line-attribution",
        help=(
            "Compute symbol-level health and coverage by indexing line-level "
            "git blame for files with SymbolAttributions. Adds 1-10 minutes "
            "to first scan (cached for subsequent runs). Default ON. Pass "
            "--no-line-attribution to skip and use file-level scoring only."
        ),
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help=(
            "Reuse the most recent saved scan for this repo as a baseline; "
            "only re-analyze features whose files changed since that scan's "
            "commit. No-op or full-fallback when no prior exists. Default OFF "
            "— without this flag the scan behaves exactly as before."
        ),
        is_flag=True,
    ),
    flows: bool = typer.Option(
        False,
        "--flows",
        help="Detect user-facing flows within features (requires --llm)",
        is_flag=True,
    ),
    symbols: bool = typer.Option(
        False,
        "--symbols",
        help=(
            "Attribute individual functions/classes to specific flows via LLM. "
            "Makes MCP responses return precise symbols instead of whole files "
            "(requires --llm --flows)."
        ),
        is_flag=True,
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help=(
            "[alpha] Upload the feature map to the Faultlines SaaS dashboard. "
            "Cloud sync is not yet in public beta — set FAULTLINES_EXPERIMENTAL=1 "
            "to enable."
        ),
        is_flag=True,
        hidden=True,
    ),
    coverage: Optional[str] = typer.Option(
        None,
        "--coverage",
        help="Path to coverage report (lcov.info or coverage-summary.json). Auto-detected if omitted.",
    ),
    # ── Analytics integration ──
    posthog_key: Optional[str] = typer.Option(
        None,
        "--posthog-key",
        help="PostHog API key (or set POSTHOG_API_KEY env var)",
    ),
    posthog_project: Optional[str] = typer.Option(
        None,
        "--posthog-project",
        help="PostHog project ID",
    ),
    posthog_host: str = typer.Option(
        "https://app.posthog.com",
        "--posthog-host",
        help="PostHog host URL (for self-hosted or local mock)",
    ),
    sentry_token: Optional[str] = typer.Option(
        None,
        "--sentry-token",
        help="Sentry auth token (or set SENTRY_AUTH_TOKEN env var)",
    ),
    sentry_org: Optional[str] = typer.Option(
        None,
        "--sentry-org",
        help="Sentry organization slug",
    ),
    sentry_project: Optional[str] = typer.Option(
        None,
        "--sentry-project",
        help="Sentry project slug",
    ),
    sentry_host: str = typer.Option(
        "https://sentry.io",
        "--sentry-host",
        help="Sentry host URL (for self-hosted or local mock)",
    ),
):
    """
    Analyzes a git repository and builds a feature map.

    Examples:
        faultline analyze
        faultline analyze ./my-project --days 90
        faultline analyze . --src src/
        faultline analyze . --llm --provider anthropic --api-key sk-ant-...
        faultline analyze . --llm --provider ollama --src src/
        faultline analyze . --llm --provider ollama --model llama3.2
        faultline analyze . --llm --flows
        faultline analyze . --llm --provider ollama --flows
        faultline analyze . --llm --flows --posthog-key phx_... --posthog-project 12345
        faultline analyze . --llm --flows --sentry-token sntrys_... --sentry-org my-org --sentry-project my-proj
    """
    repo_path = str(Path(repo_path).resolve())

    # --flows requires --llm
    if flows and not llm:
        llm = True

    if llm and provider not in ("anthropic", "ollama", "deepseek"):
        console.print(f"[red]Unknown provider '{provider}'. Use: anthropic, ollama, or deepseek[/red]")
        raise typer.Exit(1)

    if llm and provider == "ollama":
        try:
            import ollama as _ollama  # noqa: F401
        except ImportError:
            console.print(
                "[red]Ollama package not installed.[/red]\n"
                "Install with: [bold]pip install 'faultline[ollama]'[/bold]\n"
                "Or: [bold]pip install ollama[/bold]"
            )
            raise typer.Exit(1)

    try:
        # 1. Load the repository
        console.print(f"[blue]Analyzing:[/blue] {repo_path}")
        repo = load_repo(repo_path)
        remote_url = get_remote_url(repo)

        # 2. Validate LLM access early — before the long git analysis
        # Resolve effective API key based on provider
        if provider == "deepseek":
            import os as _os
            api_key = deepseek_key or _os.environ.get("DEEPSEEK_API_KEY") or api_key
        if llm:
            _validate_llm_access(provider, api_key, model, ollama_url)

        # 3. Pre-run estimate
        approx_count = estimate_commits(repo, days=days, max_commits=max_commits)
        if approx_count > 0:
            duration = estimate_duration(approx_count, use_llm=llm, use_flows=flows)
            console.print(f"[dim]~ {approx_count:,} commits in range → {duration}[/dim]")

        # 4. Fetch commits
        commits = get_commits(repo, days=days, max_commits=max_commits)
        if not commits:
            console.print("[yellow]No commits found for the specified period[/yellow]")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Found {len(commits)} commits over {days} days")

        # 5. Detect files and map to features
        files = get_tracked_files(repo, src=src)
        if src:
            console.print(f"[green]✓[/green] Found {len(files)} files under [dim]{src}[/dim]")
        else:
            console.print(f"[green]✓[/green] Found {len(files)} files")

        # Strip --src prefix so LLM/heuristic sees clean relative paths (e.g. EDR/... not src/views/EDR/...)
        analysis_files, path_prefix = _strip_src_prefix(files, src)

        # Classify repo structure to adapt LLM strategy. ``repo_root`` is
        # required for ``detect_library`` (Day 11 fix) — without it the
        # new pipeline never sees ``is_library=True`` and ``_detect_flows``
        # below happily generates user-flow output for libraries like trpc.
        repo_structure = classify_repo(analysis_files, repo_root=str(repo.working_tree_dir))
        layer_context = build_layer_context(repo_structure)
        if repo_structure.layout != "feature":
            console.print(f"[dim]Repo layout: {repo_structure.layout} (layer ratio: {repo_structure.layer_ratio:.0%})[/dim]")
        if repo_structure.is_library:
            signals = ", ".join(repo_structure.library_signals[:3])
            console.print(f"[dim]Library detected ({signals}) — flows will be suppressed[/dim]")

        # Detect workspace packages for per-package analysis
        from faultline.analyzer.workspace import detect_workspace
        workspace = detect_workspace(str(repo.working_tree_dir), analysis_files)
        if workspace.detected:
            console.print(
                f"[blue]Workspace detected:[/blue] {workspace.manager} — "
                f"{len(workspace.packages)} packages"
            )
            for pkg in workspace.packages:
                console.print(f"  [dim]{pkg.path}/[/dim] ({len(pkg.files)} files) → {pkg.name}")

        # Always extract AST signatures — needed for import graph clustering
        # and reused for flow detection when --flows is set.
        from faultline.analyzer.ast_extractor import extract_signatures
        extract_root = str(Path(str(repo.working_tree_dir)) / path_prefix) if path_prefix else str(repo.working_tree_dir)
        signatures = extract_signatures(analysis_files, extract_root)
        if signatures:
            console.print(f"[dim]Extracted signatures from {len(signatures)} files[/dim]")

        # ── Day 9: new-pipeline cutover ──
        # When the gate passes, run faultline.llm.pipeline.run() — a single
        # Sonnet-based dispatch that replaces the legacy 5-strategy block
        # below. We fall through to the legacy path when any of these are
        # true:
        #   • --legacy was passed explicitly
        #   • --llm is off (heuristic-only analysis)
        #   • provider is ollama or deepseek (the new pipeline is
        #     Anthropic-only for now)
        # The result is assigned to ``raw_mapping`` just like the legacy
        # branches do, so everything downstream (path_prefix restore,
        # build_feature_map, flow detection, analytics) stays unchanged.
        raw_mapping: dict[str, list[str]] | None = None
        _new_pipeline_result = None
        _use_new_pipeline = llm and not legacy and provider == "anthropic"
        if _use_new_pipeline:
            from faultline.llm.cost import CostTracker
            from faultline.llm.pipeline import run as _run_new_pipeline

            # Incremental pre-flight: when ``--incremental`` is set,
            # try to short-circuit the full pipeline by detecting a
            # no-op diff against the prior scan. Stages 4-5 will add
            # subset re-scan; for now, only the no-op shortcut is
            # active. Failure here is silent — full scan continues.
            _incremental_short_circuit = None
            _incremental_subset_inputs = None  # Stage 4 subset rescan
            if incremental:
                try:
                    from faultline.analyzer.git_diff import compute_git_diff
                    from faultline.llm.incremental import plan_incremental
                    from faultline.llm.scan_loader import (
                        find_prior_scan_for, load_scan_as_seed,
                    )
                    _prior_path = find_prior_scan_for(repo_path)
                    _prior = (
                        load_scan_as_seed(_prior_path)
                        if _prior_path is not None else None
                    )
                    _diff = compute_git_diff(
                        repo_path,
                        _prior.last_sha if _prior is not None else None,
                    )
                    _plan = plan_incremental(_prior, _diff)
                    console.print(
                        f"[blue]Incremental:[/blue] {_plan.summary()}"
                    )
                    if (
                        _prior is not None
                        and _plan.is_no_op
                        and not _plan.fallback_full_scan
                    ):
                        _incremental_short_circuit = _prior.result
                    elif (
                        _prior is not None
                        and not _plan.fallback_full_scan
                        and workspace.detected
                        and len(workspace.packages) >= 2
                    ):
                        # Stage 4 path: workspace with stale packages.
                        # Verify there's a clean+stale split worth doing.
                        from faultline.llm.incremental import (
                            identify_stale_packages,
                        )
                        _stale_pkgs, _clean_pkgs = identify_stale_packages(
                            workspace, _diff,
                        )
                        if _stale_pkgs and _clean_pkgs:
                            _incremental_subset_inputs = {
                                "mode": "workspace",
                                "plan": _plan,
                                "prior": _prior,
                                "diff": _diff,
                                "stale_count": len(_stale_pkgs),
                                "clean_count": len(_clean_pkgs),
                            }
                            console.print(
                                f"[blue]Incremental subset:[/blue] "
                                f"re-scanning {len(_stale_pkgs)} stale "
                                f"package(s); {len(_clean_pkgs)} clean "
                                f"package(s) carried forward."
                            )
                    elif (
                        _prior is not None
                        and not _plan.fallback_full_scan
                        and (_plan.stale_features or _plan.fresh_files)
                    ):
                        # Stage 5 path: monolith / non-workspace repo
                        # with stale features. Re-scan the file subset
                        # and carry the clean features forward.
                        _incremental_subset_inputs = {
                            "mode": "monolith",
                            "plan": _plan,
                            "prior": _prior,
                            "diff": _diff,
                            "stale_count": len(_plan.stale_features),
                            "fresh_count": len(_plan.fresh_files),
                        }
                        console.print(
                            f"[blue]Incremental subset:[/blue] "
                            f"re-scanning {len(_plan.stale_features)} stale "
                            f"feature(s) + {len(_plan.fresh_files)} fresh "
                            f"file(s); {len(_plan.clean_features)} clean "
                            f"feature(s) carried forward."
                        )
                except Exception as exc:  # noqa: BLE001 — opportunistic
                    console.print(
                        f"[yellow]⚠ Incremental pre-flight failed "
                        f"({type(exc).__name__}: {exc}) — running full scan."
                        f"[/yellow]"
                    )

            _cost_tracker = CostTracker()
            _need_full_scan = True
            if _incremental_short_circuit is not None:
                _new_pipeline_result = _incremental_short_circuit
                _need_full_scan = False
                console.print(
                    "[green]✓[/green] Incremental no-op: "
                    f"{len(_new_pipeline_result.features)} features carried "
                    "forward, no LLM calls made."
                )
            elif _incremental_subset_inputs is not None:
                # Stage 4 (workspace) or Stage 5 (monolith) partial re-scan.
                from faultline.llm.incremental import (
                    execute_monolith_incremental,
                    execute_workspace_incremental,
                )
                from faultline.llm.sonnet_scanner import build_commit_context
                _commit_ctx = build_commit_context(commits)
                _mode = _incremental_subset_inputs["mode"]
                try:
                    if _mode == "workspace":
                        _new_pipeline_result = execute_workspace_incremental(
                            plan=_incremental_subset_inputs["plan"],
                            prior=_incremental_subset_inputs["prior"],
                            diff=_incremental_subset_inputs["diff"],
                            workspace=workspace,
                            repo_root=Path(repo_path),
                            api_key=api_key,
                            model=model,
                            tracker=_cost_tracker,
                            use_tools=tool_use,
                            commit_context=_commit_ctx,
                        )
                    else:  # monolith
                        _new_pipeline_result = execute_monolith_incremental(
                            plan=_incremental_subset_inputs["plan"],
                            prior=_incremental_subset_inputs["prior"],
                            diff=_incremental_subset_inputs["diff"],
                            repo_root=Path(repo_path),
                            signatures=signatures,
                            api_key=api_key,
                            model=model,
                            tracker=_cost_tracker,
                            commit_context=_commit_ctx,
                        )
                    _need_full_scan = False
                    console.print(
                        f"[green]✓[/green] Incremental subset complete "
                        f"({_mode}): {len(_new_pipeline_result.features)} "
                        f"features total"
                    )
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"[yellow]⚠ Subset re-scan failed "
                        f"({type(exc).__name__}: {exc}) — falling back to "
                        f"full scan.[/yellow]"
                    )
                    # _need_full_scan stays True → full pipeline runs below
            if _need_full_scan:
                console.print(
                    "[blue]Running new pipeline[/blue] "
                    "(pass [dim]--legacy[/dim] to use the 5-strategy fallback)"
                )
                # FAULTLINE_FORCE_FLOWS=1 overrides library detection
                # so flows are computed even when the repo looks like
                # a library. Useful for apps that ship as packages
                # (Superset, Airflow, etc.) where pyproject.toml +
                # missing main.py mislabels them.
                import os as _os_force
                if (
                    _os_force.environ.get("FAULTLINE_FORCE_FLOWS") == "1"
                    and getattr(repo_structure, "is_library", False)
                ):
                    try:
                        repo_structure = repo_structure.__class__(
                            **{**repo_structure.__dict__, "is_library": False}
                        )
                    except Exception:
                        repo_structure.is_library = False
                    console.print(
                        "[yellow]FAULTLINE_FORCE_FLOWS=1 — overriding "
                        "library classification to keep flows enabled.[/yellow]"
                    )
                try:
                    _new_pipeline_result = _run_new_pipeline(
                        analysis_files=analysis_files,
                        workspace=workspace,
                        repo_structure=repo_structure,
                        signatures=signatures,
                        commits=commits,
                        api_key=api_key,
                        model=model,
                        tracker=_cost_tracker,
                        use_tools=tool_use,
                        repo_root=Path(repo_path),
                        dedup=dedup,
                        sub_decompose=sub_decompose,
                        tool_flows=tool_flows,
                        critique=critique,
                        trace_flows=trace_flows,
                        rename_generic=rename_generic,
                    )
                except Exception as exc:  # pragma: no cover - surfacing guidance
                    console.print(
                        f"[red]⚠ New pipeline raised {type(exc).__name__}: {exc}[/red]"
                    )
                    console.print(
                        "[red]   Falling through to LEGACY 5-strategy path — "
                        "post-rewrite improvements (catchall split, flow dedup, "
                        "CRUD enrichment, noise filter) will NOT apply to this scan.[/red]"
                    )
                    _new_pipeline_result = None

            if _new_pipeline_result is None:
                console.print(
                    "[red]⚠ New pipeline returned no features — falling through to legacy[/red]"
                )
            else:
                raw_mapping = dict(_new_pipeline_result.features)
                console.print(
                    f"[green]✓[/green] New pipeline: {len(raw_mapping)} features"
                )
                _cost_summary = _cost_tracker.summary()
                if _cost_summary["total_calls"] > 0:
                    console.print(
                        f"[dim]LLM cost: ${_cost_summary['total_cost_usd']:.3f} "
                        f"across {_cost_summary['total_calls']} calls "
                        f"({_cost_summary['total_input_tokens']:,} in / "
                        f"{_cost_summary['total_output_tokens']:,} out)[/dim]"
                    )

        # ── Workspace-aware analysis: per-package detection + merge ──
        _TS_JS_EXTS = {".ts", ".tsx", ".js", ".jsx"}
        _MIN_SIGNATURES_FOR_IMPORT_GRAPH = 10
        ts_js_sig_count = sum(1 for f in signatures if Path(f).suffix.lower() in _TS_JS_EXTS) if signatures else 0

        _MIN_PACKAGE_FILES = 15  # skip packages with fewer files
        _MAX_LLM_PACKAGES = 12  # limit LLM calls to largest packages
        _SKIP_PREFIXES = ("example", "sample", "demo", "template", "starter", "tutorial")

        if raw_mapping is not None:
            # New pipeline already produced the mapping above — skip the
            # legacy 5-strategy dispatch entirely. This is the Day 9
            # cutover; the legacy branches below stay reachable via
            # --legacy or whenever the new pipeline is gated off.
            pass
        elif workspace.detected and len(workspace.packages) >= 2 and llm:
            # Filter: skip example packages, sort by size, limit LLM calls
            llm_packages = [
                p for p in workspace.packages
                if len(p.files) >= _MIN_PACKAGE_FILES
                and not any(p.name.lower().startswith(s) for s in _SKIP_PREFIXES)
            ]
            skip_packages = [p for p in workspace.packages if p not in llm_packages]

            # Take only the largest packages for LLM analysis
            llm_packages.sort(key=lambda p: len(p.files), reverse=True)
            if len(llm_packages) > _MAX_LLM_PACKAGES:
                skip_packages.extend(llm_packages[_MAX_LLM_PACKAGES:])
                llm_packages = llm_packages[:_MAX_LLM_PACKAGES]

            console.print(
                f"[blue]Per-package analysis: {len(llm_packages)} packages via LLM, "
                f"{len(skip_packages)} skipped (examples/small)[/blue]"
            )
            raw_mapping: dict[str, list[str]] = {}
            feature_names_seen: set[str] = set()

            # Skipped packages → group into one "examples" feature or by category
            examples_files: list[str] = []
            other_skip_files: list[str] = []
            for pkg in skip_packages:
                if not pkg.files:
                    continue
                if any(pkg.name.lower().startswith(s) for s in _SKIP_PREFIXES):
                    examples_files.extend(pkg.files)
                else:
                    other_skip_files.extend(pkg.files)

            if examples_files:
                raw_mapping["examples"] = examples_files
            if other_skip_files:
                raw_mapping["other-packages"] = other_skip_files

            # Day 10: _SPLIT_THRESHOLD=200 removed (acceptance criterion D).
            # The new pipeline at faultline.llm.pipeline.run enforces an
            # 8-sub-feature hard cap at the prompt level, so the threshold
            # hack is gone on the default path. This legacy branch is only
            # reached via --legacy; every non-test, non-tiny package now
            # goes through detect_features_llm unconditionally. Small
            # packages incur a slightly higher LLM cost here — acceptable
            # for a deprecated emergency fallback.
            _TEST_PACKAGE_NAMES = {"tests", "test", "testing", "__tests__", "e2e", "cypress", "playwright", "jest"}

            for pkg in llm_packages:
                # Skip test-only packages — tests are not features
                if pkg.name.lower() in _TEST_PACKAGE_NAMES:
                    console.print(f"  [dim]{pkg.name} ({len(pkg.files)} files) → skipped (tests)[/dim]")
                    continue

                # Every package gets an LLM sub-split pass. If there's
                # only one cohesive feature, the LLM will return a single
                # entry and the downstream re-prefix produces
                # {pkg.name}/{sub}; the result is canonicalized later.
                pkg_prefix = pkg.path + "/"
                pkg_files = [f[len(pkg_prefix):] for f in pkg.files if f.startswith(pkg_prefix)]
                if not pkg_files:
                    raw_mapping[pkg.name] = list(pkg.files)
                    continue

                console.print(f"  [dim]Splitting {pkg.name} ({len(pkg_files)} files)...[/dim]")

                try:
                    from faultline.llm.detector import detect_features_llm, detect_features_ollama
                    pkg_sigs = {
                        f[len(pkg_prefix):]: sig
                        for f, sig in (signatures or {}).items()
                        if f.startswith(pkg_prefix)
                    }
                    pkg_commits = [
                        c for c in commits
                        if any(f.startswith(pkg_prefix) for f in c.files_changed)
                    ] if commits else None

                    if provider == "ollama":
                        pkg_mapping = detect_features_ollama(
                            pkg_files, model=model or "llama3.1:8b",
                            host=ollama_url, path_prefix="",
                            signatures=pkg_sigs or None,
                        )
                    else:
                        pkg_mapping = detect_features_llm(
                            pkg_files, api_key=api_key,
                            commits=pkg_commits, path_prefix="",
                            signatures=pkg_sigs or None,
                            layer_context=layer_context,
                            model=model,
                        )

                    # Prefix sub-features with package name
                    for feat_name, feat_files in pkg_mapping.items():
                        full_files = [pkg_prefix + f for f in feat_files]
                        raw_mapping[f"{pkg.name}/{feat_name}"] = full_files

                    console.print(f"  [dim]{pkg.name} → {len(pkg_mapping)} sub-features[/dim]")
                except Exception as e:
                    console.print(f"  [yellow]⚠ {pkg.name}: {e} — single feature[/yellow]")
                    raw_mapping[pkg.name] = list(pkg.files)

            # Root files → shared-infra (config, CI, tooling)
            if workspace.root_files:
                raw_mapping["shared-infra"] = workspace.root_files

            # Remove empty features and test-only features
            raw_mapping = {
                k: v for k, v in raw_mapping.items()
                if v and k.lower().rstrip("/") not in _TEST_PACKAGE_NAMES
                and not k.lower().endswith("/testing-utilities")
                and not k.lower().endswith("/test-utils")
            }

            console.print(
                f"[green]✓[/green] Workspace analysis: {len(raw_mapping)} features "
                f"across {len(workspace.packages)} packages"
            )

        # ── Mixed repo detection ──
        # If significant non-JS files exist alongside TS/JS, import graph only covers
        # part of the codebase. Use candidate-based detection for the whole repo instead.
        elif (llm and (len(analysis_files) - ts_js_sig_count) / max(len(analysis_files), 1) >= 0.30
                and ts_js_sig_count >= _MIN_SIGNATURES_FOR_IMPORT_GRAPH
                and (len(analysis_files) - ts_js_sig_count) >= 20):
            from faultline.analyzer.features import detect_candidates
            non_js_count = len(analysis_files) - ts_js_sig_count
            candidates = detect_candidates(analysis_files)
            n_candidates = len([n for n, p in candidates.items() if len(p) >= 2 and n not in {"backend", "frontend", "root", "init"}])
            console.print(f"[blue]Mixed repo ({ts_js_sig_count} JS + {non_js_count} other files) — candidate-based detection: {n_candidates} candidates[/blue]")
            raw_mapping = _detect_with_candidates(
                analysis_files, candidates, provider, api_key, model, ollama_url,
                commits=commits, path_prefix=path_prefix,
            )

        # ── Auto strategy: existing pipeline ──
        # Step 1 — Import graph clustering (primary, always deterministic)
        # Files connected through import chains form the same cluster.
        # Need meaningful number of TS/JS files — Python sigs are useful for flow detection
        # but not for import graph clustering which relies on JS/TS import statements.
        elif signatures and ts_js_sig_count >= _MIN_SIGNATURES_FOR_IMPORT_GRAPH:
            from faultline.analyzer.import_graph import build_import_clusters, scan_domains, load_tsconfig_paths, detect_monorepo_packages
            domains = scan_domains(analysis_files)
            domain_counts = {}
            for d in domains.values():
                if d != "__open__":
                    domain_counts[d] = domain_counts.get(d, 0) + 1
            if domain_counts:
                console.print(f"[dim]Domain boundaries: {len(domain_counts)} domains detected[/dim]")

            # Load tsconfig path aliases for better import resolution
            tsconfig_paths = load_tsconfig_paths(str(repo.working_tree_dir))
            if tsconfig_paths:
                console.print(f"[dim]tsconfig paths: {', '.join(tsconfig_paths.keys())}[/dim]")

            # Detect monorepo packages for bare import resolution
            monorepo_pkgs = detect_monorepo_packages(str(repo.working_tree_dir))
            if monorepo_pkgs:
                console.print(f"[dim]Monorepo packages: {len(monorepo_pkgs)} detected[/dim]")

            raw_mapping = build_import_clusters(
                analysis_files, signatures,
                tsconfig_paths=tsconfig_paths,
                monorepo_packages=monorepo_pkgs or None,
            )
            console.print(
                f"[dim]Import graph: {ts_js_sig_count} TS/JS files → {len(raw_mapping)} clusters[/dim]"
            )

            # Collapse plugin directories (e.g. app-store with 152 integrations → 1 cluster)
            from faultline.llm.detector import _collapse_plugin_features
            pre_collapse = len(raw_mapping)
            raw_mapping = _collapse_plugin_features(raw_mapping)
            if len(raw_mapping) < pre_collapse:
                console.print(
                    f"[dim]Plugin collapse: {pre_collapse} → {len(raw_mapping)} clusters[/dim]"
                )

            # Compute inter-cluster import connections for LLM context
            from faultline.analyzer.import_graph import compute_cluster_edges
            edges = compute_cluster_edges(
                raw_mapping, signatures,
                file_set=set(analysis_files),
                alias_map=tsconfig_paths,
                monorepo_packages=monorepo_pkgs or None,
            )
            if edges:
                total_cross = sum(sum(v.values()) for v in edges.values())
                console.print(f"[dim]Inter-cluster edges: {total_cross} cross-imports[/dim]")

            # Step 2a — LLM: merge related clusters into business features + name them
            if llm:
                raw_mapping = _merge_and_name_with_llm(
                    raw_mapping, provider, api_key, model, ollama_url,
                    commits=commits, layer_context=layer_context,
                    cluster_edges=edges,
                )
        elif llm:
            # No import graph (Python, Ruby, Go, etc.) — use candidate-based detection:
            # 1. Heuristics extract feature candidates (routers, pages, features dirs)
            # 2. LLM verifies, merges, and assigns unmatched files
            from faultline.analyzer.features import detect_candidates
            candidates = detect_candidates(analysis_files)
            n_candidates = len([n for n, p in candidates.items() if len(p) >= 2 and n not in {"backend", "frontend", "root", "init"}])
            console.print(f"[blue]Candidate-based detection: {n_candidates} candidates from heuristics[/blue]")
            raw_mapping = _detect_with_candidates(
                analysis_files, candidates, provider, api_key, model, ollama_url,
                commits=commits, path_prefix=path_prefix,
            )
        else:
            console.print("[dim]No TS/JS files — using directory heuristic[/dim]")
            raw_mapping = detect_features_from_structure(analysis_files)

        # Split oversized features — only for non-TS/JS repos (Django/Rails monoliths)
        # TS/JS repos already have fine-grained features from import graph + LLM merge.
        # Day 11: skip for the new pipeline. Sonnet's prompt already targets
        # 12-25 features and the docs collapse in _clean_inputs produces a
        # single ``documentation`` bucket; ``split_large_features`` would
        # explode it back into one sub-feature per tutorial directory
        # (regression observed on fastapi: 1 → 130+ documentation-tutorialNNN).
        if ts_js_sig_count < _MIN_SIGNATURES_FOR_IMPORT_GRAPH and _new_pipeline_result is None:
            raw_mapping = split_large_features(raw_mapping)

        # Restore full paths so commit matching works against git history
        if path_prefix:
            feature_paths = {
                name: [path_prefix + f for f in paths]
                for name, paths in raw_mapping.items()
            }
        else:
            feature_paths = raw_mapping

        console.print(f"[green]✓[/green] Detected {len(feature_paths)} features")

        # 5b. Symbol-level attribution for shared files (TS/JS only)
        shared_attributions = None
        if signatures and ts_js_sig_count >= _MIN_SIGNATURES_FOR_IMPORT_GRAPH:
            from faultline.analyzer.import_graph import resolve_symbol_imports, load_tsconfig_paths as _ltp
            from faultline.analyzer.shared_files import build_shared_attributions

            # Build signatures index with full paths (matching feature_paths)
            full_path_sigs = {}
            for rel, sig in signatures.items():
                full_key = (path_prefix + rel) if path_prefix else rel
                full_path_sigs[full_key] = sig

            from faultline.analyzer.import_graph import load_tsconfig_paths as _load_tsconfig
            tsconfig = locals().get("tsconfig_paths") or _load_tsconfig(str(repo.working_tree_dir))
            symbol_imports = resolve_symbol_imports(
                full_path_sigs,
                alias_map=tsconfig,
                monorepo_packages=locals().get("monorepo_pkgs"),
            )
            shared_attributions = build_shared_attributions(
                feature_paths, symbol_imports, full_path_sigs,
            )
            if shared_attributions:
                shared_count = sum(len(v) for v in shared_attributions.values())
                console.print(f"[dim]Symbol attribution: {shared_count} shared file mappings across {len(shared_attributions)} features[/dim]")

        # 6. Build the feature map.
        # Day 14 library-mode fix: libraries should not have their
        # per-module features collapsed into parent directories — that
        # undoes the per-stem candidate promotion in deep_scan. Skip
        # the small-feature merge pass only when the repo is a
        # library AND the new pipeline ran the single-call path (not
        # the workspace per-package loop). Workspace libraries like
        # trpc already get granularity via per-package isolation;
        # letting _merge_small_features fire there keeps the 1-file
        # helpers from drowning out real modules.
        _new_pipeline_used_workspace = (
            _new_pipeline_result is not None
            and workspace.detected
            and len(workspace.packages) >= 2
        )
        _skip_merge = (
            bool(repo_structure.is_library)
            and not _new_pipeline_used_workspace
        )
        # Refactor Day 3: build per-feature participants via SymbolGraph
        # BFS. This replaces the cross-feature ``shared_attributions``
        # surface as the primary attachment for symbol-scoped scoring
        # — works on workspace monorepos and Django apps where
        # shared_attributions stays empty. Cross-language: any repo with
        # a populated SymbolGraph (TS/JS/Py/Go/Rust per Sprint 1 Day 4-5).
        feature_participants: dict[str, list] = {}
        if line_attribution and signatures:
            try:
                from faultline.analyzer.feature_participants import (
                    build_feature_participants,
                )
                from faultline.analyzer.symbol_graph import build_symbol_graph
                participants_graph = build_symbol_graph(
                    str(repo.working_tree_dir),
                    list(analysis_files),
                    include_http_edges=False,
                )
                feature_participants = build_feature_participants(
                    feature_paths,
                    participants_graph,
                    repo_root=str(repo.working_tree_dir),
                )
                if feature_participants:
                    total_p = sum(len(v) for v in feature_participants.values())
                    console.print(
                        f"[dim]Feature participants: {total_p} across "
                        f"{len(feature_participants)} features "
                        f"(avg {total_p // max(len(feature_participants), 1)}/feature)[/dim]"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("feature_participants skipped (%s)", exc)
                feature_participants = {}

        # Sprint 3 Day 11 + Refactor Day 3: build BlameIndex over the
        # union of files referenced by EITHER shared_attributions
        # (legacy, cross-feature only) OR feature.participants (new
        # per-feature surface). The participant union covers many more
        # repos and is what Day 4 scoring reads from.
        blame_index = None
        if line_attribution and (shared_attributions or feature_participants):
            from faultline.analyzer.blame_index import BlameIndex
            files_to_index: set[str] = set()
            for attrs in (shared_attributions or {}).values():
                for attr in attrs:
                    files_to_index.add(attr.file_path)
            for parts in feature_participants.values():
                for p in parts:
                    files_to_index.add(p.path)
            if files_to_index:
                blame_index = BlameIndex(repo.working_tree_dir)
                stats = blame_index.index_files(sorted(files_to_index))
                console.print(
                    f"[dim]Blame index: {stats.indexed} indexed + "
                    f"{stats.cached} cached"
                    + (f" + {stats.failed} failed" if stats.failed else "")
                    + f" ({len(files_to_index)} files)[/dim]"
                )

        feature_map = build_feature_map(
            repo_path=repo_path,
            commits=commits,
            feature_paths=feature_paths,
            days=days,
            remote_url=remote_url,
            shared_attributions=shared_attributions,
            skip_small_feature_merge=_skip_merge,
            blame_index=blame_index,
            feature_participants=feature_participants or None,
        )

        # Refactor Day 3: attach participants to each Feature so that
        # downstream consumers (codegen, dashboard, scoring rewrite in
        # Day 4) can read them as a primary surface. Names that don't
        # match (rare — usually post-process renamed) silently no-op.
        if feature_participants:
            for feat in feature_map.features:
                feat.participants = feature_participants.get(feat.name, [])

        # 6a. Day 9: inject descriptions from the new pipeline, if we
        # used it. ``DeepScanResult.descriptions`` is keyed by the feature
        # name Sonnet returned, which may differ from the canonicalized
        # / path-prefixed name on ``feature_map.features[*]`` (e.g.
        # singular/plural, or ``api/auth`` vs ``auth``). Match with the
        # same fuzzy rule the deep-scan subcommand uses.
        if _new_pipeline_result is not None and _new_pipeline_result.descriptions:
            _inject_new_pipeline_descriptions(
                feature_map, _new_pipeline_result.descriptions,
            )

        # 6a.5 (Sprint 4): inject tool-augmented flow names. Runs only
        # when --tool-flows produced flows — i.e. result.flows is
        # populated. When this path is active we skip the legacy
        # _detect_flows entirely (the new pipeline IS the flow source).
        if (
            _new_pipeline_result is not None
            and _new_pipeline_result.flows
            and tool_flows
        ):
            _inject_new_pipeline_flows(
                feature_map,
                _new_pipeline_result.flows,
                _new_pipeline_result.flow_descriptions,
                commits,
                flow_participants=getattr(
                    _new_pipeline_result, "flow_participants", {},
                ),
            )

        # 6a.55: Commit-aware noise drop (Fix #2 from Fixable-accuracy
        # eval). Folds tiny-and-cold features (<4 files AND <30 commits
        # AND no flows) into shared-infra. The 30-commit escape hatch
        # protects hot small features like n8n's Workflows (3f/300c).
        # Runs always — independent of --post-process — because it's a
        # basic safety filter, not a transformation. Closes ~6 noise
        # features on n8n / strapi without touching legitimate slices.
        from faultline.analyzer.features import _drop_noise_features
        _n_before_noise = len(feature_map.features)
        feature_map.features = _drop_noise_features(feature_map.features)
        _n_after_noise = len(feature_map.features)
        if _n_after_noise < _n_before_noise:
            console.print(
                f"[dim]Noise filter: dropped {_n_before_noise - _n_after_noise} "
                f"tiny+cold feature(s) into shared-infra "
                f"({_n_before_noise} → {_n_after_noise})[/dim]"
            )

        # 6a.6: Populate Title Case display_name on every feature +
        # flow. Internal slug-form name stays untouched — it's the
        # stable ID used for dedup / config / API lookups. Dashboards
        # and reports prefer display_name when present so a buyer
        # sees "Authentication" rather than "user-authentication".
        _populate_display_names(feature_map)

        # 6a.7: Post-process cleanup. Runs by default (gated by
        # ``--no-post-process`` for raw output). Applies the same
        # transformations as scripts/cleanup_feature_map.py:
        # merge sub-features, re-attribute noise files, refine path
        # signal, extract overlooked top-dirs, mine commit-prefix
        # vocabulary, drop noise/vendored/phantom/mega-bucket.
        if post_process:
            from faultline.analyzer.post_process import run as run_post_process
            n_before = len(feature_map.features)
            fl_before = sum(len(f.flows) for f in feature_map.features)
            feature_map = run_post_process(
                feature_map, repo_path=str(repo.working_tree_dir),
            )
            n_after = len(feature_map.features)
            fl_after = sum(len(f.flows) for f in feature_map.features)
            if n_after != n_before or fl_after != fl_before:
                console.print(
                    f"[dim]Post-process: features {n_before} → {n_after}, "
                    f"flows {fl_before} → {fl_after}[/dim]"
                )

        # 6b. Read coverage data (if available).
        # Sprint 2 Day 9: prefer line-level data so we can scope coverage
        # to SymbolAttribution.line_ranges. Falls back to file-level
        # when no detailed data available (legacy contract).
        from faultline.analyzer.coverage import read_coverage, read_coverage_detailed
        coverage_data = read_coverage(str(repo.working_tree_dir), coverage_path=coverage)
        coverage_detailed = read_coverage_detailed(
            str(repo.working_tree_dir), coverage_path=coverage,
        )
        if coverage_data:
            console.print(
                f"[dim]Coverage data: {len(coverage_data)} files"
                + (f" (line-level for {len(coverage_detailed)})" if coverage_detailed else "")
                + "[/dim]"
            )
            _apply_feature_coverage(
                feature_map, coverage_data, path_prefix,
                detailed=coverage_detailed or None,
            )

        # 6c. Detect flows within each feature (optional)
        # Day 11: skip flow detection for libraries by default. Libraries
        # don't have user-facing flows — their "users" are developers
        # invoking APIs. Override with FAULTLINE_FORCE_FLOWS=1 env var
        # for repos that are both library AND app (e.g. excalidraw).
        import os as _os
        _force_flows = _os.environ.get("FAULTLINE_FORCE_FLOWS") == "1"
        if flows and repo_structure.is_library and not _force_flows:
            console.print(
                "[dim]Skipping flow detection — repo classified as library "
                "(set FAULTLINE_FORCE_FLOWS=1 to override)[/dim]"
            )
        # When --tool-flows produced flows already, skip the legacy
        # Haiku detector — the new pipeline is the source of truth.
        _tool_flows_active = bool(
            tool_flows and _new_pipeline_result is not None
            and _new_pipeline_result.flows
        )
        if flows and (not repo_structure.is_library or _force_flows) and not _tool_flows_active:
            from faultline.llm.flow_detector import detect_e2e_anchors
            e2e_anchors = detect_e2e_anchors(analysis_files)
            if e2e_anchors:
                console.print(
                    f"[dim]E2E anchors: {len(e2e_anchors)} flows detected from test files[/dim]"
                )
            feature_map = _detect_flows(
                feature_map=feature_map,
                repo_path=str(repo.working_tree_dir),
                analysis_files=analysis_files,
                path_prefix=path_prefix,
                commits=commits,
                provider=provider,
                api_key=api_key,
                model=model,
                ollama_url=ollama_url,
                signatures=signatures,
                remote_url=remote_url,
                coverage_data=coverage_data,
                e2e_anchors=e2e_anchors,
            )

        # 6c.5 Symbol-level attribution (optional, --symbols)
        if symbols and llm and flows:
            try:
                from faultline.symbols.pipeline import enrich_with_symbols
                console.print(
                    f"[blue]Attributing symbols to flows (provider={provider})...[/blue]"
                )
                enrich_with_symbols(
                    feature_map=feature_map,
                    signatures=signatures or {},
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    ollama_host=ollama_url,
                )
                enriched_flows = sum(
                    len([fl for fl in f.flows if fl.symbol_attributions])
                    for f in feature_map.features
                )
                total_flows = sum(len(f.flows) for f in feature_map.features)
                console.print(
                    f"[dim]Symbol attribution: {enriched_flows}/{total_flows} flows enriched[/dim]"
                )
            except Exception as _exc:
                console.print(
                    f"[yellow]Symbol attribution failed — falling back to file-level[/yellow] "
                    f"[dim]({_exc})[/dim]"
                )
        elif symbols and not (llm and flows):
            console.print(
                "[yellow]--symbols requires --llm --flows. Skipping.[/yellow]"
            )

        # 6d. Analytics integration (optional)
        import os
        _posthog_key = posthog_key or os.environ.get("POSTHOG_API_KEY")
        _sentry_token = sentry_token or os.environ.get("SENTRY_AUTH_TOKEN")
        impact_scores = None

        if _posthog_key or _sentry_token:
            impact_scores = _run_analytics(
                feature_map=feature_map,
                posthog_key=_posthog_key,
                posthog_project=posthog_project,
                posthog_host=posthog_host,
                sentry_token=_sentry_token,
                sentry_org=sentry_org,
                sentry_project=sentry_project,
                sentry_host=sentry_host,
            )

        # 6e. Sprint 3 Day 12: populate Flow.test_files +
        # SymbolAttribution.shared_with_flows from a single TestMap
        # built off the existing symbol_graph. Runs only when we have
        # both signatures (symbol graph backbone) and at least one
        # feature with attributions — otherwise the pass is a no-op.
        if line_attribution and signatures and shared_attributions:
            try:
                from faultline.analyzer.symbol_graph import build_symbol_graph
                from faultline.analyzer.test_mapper import (
                    apply_test_attribution, build_test_map,
                )
                tracked = list(get_tracked_files(str(repo.working_tree_dir)))
                test_graph = build_symbol_graph(
                    str(repo.working_tree_dir), tracked,
                    include_http_edges=False,
                )
                test_map = build_test_map(tracked, test_graph)
                if test_map.by_symbol or test_map.by_file:
                    apply_test_attribution(feature_map, test_map)
                    n_flows_with_tests = sum(
                        1 for f in feature_map.features
                        for fl in f.flows
                        if fl.test_files
                    )
                    if n_flows_with_tests:
                        console.print(
                            f"[dim]Test attribution: "
                            f"{len(test_map.by_symbol)} symbol→test mappings, "
                            f"{n_flows_with_tests} flows with tests[/dim]"
                        )
            except Exception as exc:  # noqa: BLE001 — opportunistic
                logger.warning(
                    "test attribution skipped (%s)", exc,
                )

        # 7. Print the report
        print_report(feature_map, impact_scores=impact_scores)

        # 7b. Stamp cache metadata (git HEAD + content hashes) for incremental refresh
        try:
            from faultline.cache.hashing import hash_files
            import subprocess as _sp
            try:
                head_sha = _sp.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo.working_tree_dir), text=True, timeout=5,
                ).strip()
                feature_map.last_scanned_sha = head_sha
            except Exception:
                pass
            all_tracked = sorted({
                p for f in feature_map.features for p in f.paths
            })
            feature_map.file_hashes = hash_files(all_tracked, str(repo.working_tree_dir))
        except Exception as _hash_exc:
            console.print(f"[dim]Cache metadata skipped: {_hash_exc}[/dim]")

        # 8. Save to disk
        if save:
            saved_path = write_feature_map(feature_map, output)
            console.print(f"[dim]Saved: {saved_path}[/dim]")

        # 8b. Optional: push to SaaS dashboard (alpha — experimental only)
        if push:
            import os as _os
            if not _os.environ.get("FAULTLINES_EXPERIMENTAL"):
                console.print(
                    "[yellow]--push is alpha and not yet available in public beta.[/yellow]\n"
                    "[dim]Set FAULTLINES_EXPERIMENTAL=1 to opt in once the cloud dashboard launches.[/dim]"
                )
            elif not _os.environ.get("FAULTLINE_API_KEY"):
                console.print(
                    "[yellow]--push set but FAULTLINE_API_KEY is empty — skipping cloud upload.[/yellow]"
                )
            else:
                try:
                    from faultline.cloud.sync import push_feature_map
                    console.print("[blue]Uploading to faultlines.dev...[/blue]")
                    result = push_feature_map(feature_map)
                    if result and result.get("ok"):
                        console.print(
                            f"[green]✓ Uploaded[/green] [dim]"
                            f"({result.get('feature_count')} features, "
                            f"{result.get('flow_count')} flows, scan_id={result.get('scan_id')})[/dim]"
                        )
                    else:
                        console.print(
                            "[yellow]Cloud upload failed — feature map saved locally only.[/yellow]"
                        )
                except Exception as _push_exc:
                    console.print(f"[yellow]Cloud upload error: {_push_exc}[/yellow]")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)


def _inject_new_pipeline_descriptions(
    feature_map,
    descriptions: dict[str, str],
) -> None:
    """Copy Sonnet-authored feature descriptions onto the built feature map.

    Sonnet returns descriptions keyed by the feature name it chose, which
    may differ from the canonicalized name that survives through
    ``build_feature_map`` (e.g. singular/plural collisions, or workspace
    prefix collapse). Match with a small ladder of rules:

      1. Exact name match
      2. Either name contains the other (handles ``api/auth`` ↔ ``auth``)
      3. Singular / plural strip (handles ``issue`` ↔ ``issues``)

    The first rule that fires wins. A feature that already has a
    description (from a previous scan or legacy path) is left alone.
    Mirrors the deep-scan subcommand helper at cli.py:1197-1202 so the
    two paths produce consistent output.
    """
    if not descriptions:
        return
    # Build a quick lookup. Exact match is the only safe rule: the new
    # pipeline writes descriptions keyed by the same feature name that
    # survives into ``feature_map.features``. Sprint 2 surfaced two
    # ways the previous fuzzy matcher leaked descriptions across
    # unrelated features:
    #   - Substring-in containment ("ai" found in "auth-emails", "types"
    #     found in "typescript").
    #   - Path-segment match: a feat "web/surveys" (web package's survey
    #     code) inherited the description of the "surveys" package
    #     because both ended in "surveys" after a slash split.
    # Singular/plural is preserved as a last resort because it's
    # bounded — it can only match within one identifier, not across
    # path segments.
    for feat in feature_map.features:
        if feat.description:
            continue
        if feat.name in descriptions:
            feat.description = descriptions[feat.name]
            continue
        # Bounded singular/plural fallback — only fires when the two
        # names differ ONLY by a trailing 's', no slashes involved on
        # either side. Anything broader (path-segment, substring) has
        # been shown to leak descriptions across unrelated features.
        if "/" not in feat.name:
            feat_stem = feat.name.rstrip("s")
            for desc_name, desc in descriptions.items():
                if "/" in desc_name:
                    continue
                if (
                    desc_name != feat.name
                    and desc_name.rstrip("s") == feat_stem
                ):
                    feat.description = desc
                    break


def _populate_display_names(feature_map) -> None:
    """Set ``display_name`` on every Feature + Flow from the slug name.

    Skips features that already have a ``display_name`` set (the LLM
    or a future override might supply one explicitly). Idempotent —
    safe to re-run on the same map.
    """
    from faultline.analyzer.humanize import (
        humanize_feature_name, humanize_flow_name,
    )
    for feat in feature_map.features:
        if not feat.display_name:
            feat.display_name = humanize_feature_name(feat.name)
        for fl in feat.flows or []:
            if not fl.display_name:
                fl.display_name = humanize_flow_name(fl.name)


_ENTRY_TRAIL_RE = re.compile(r"\s*\(entry:\s*([^:)]+):(\d+)\)\s*$")


def _split_entry_trail(desc: str | None) -> tuple[str | None, str | None, int | None]:
    """Pull a Sprint 4 ``(entry: file:line)`` suffix off a flow description.

    Returns ``(clean_description, entry_file, entry_line)``. When the
    suffix is missing or malformed, returns the original description
    plus ``(None, None)`` for the entry-point fields. The returned
    description has the trailing whitespace + suffix stripped so it
    reads cleanly in the dashboard.
    """
    if not desc:
        return desc, None, None
    m = _ENTRY_TRAIL_RE.search(desc)
    if not m:
        return desc, None, None
    file_part = m.group(1).strip() or None
    try:
        line_part: int | None = int(m.group(2))
    except ValueError:
        line_part = None
    cleaned = desc[: m.start()].rstrip() or None
    return cleaned, file_part, line_part


def _flow_specific_stats(
    flow_paths: set[str],
    feat_commits,
    feat_health_fn,
):
    """Compute per-flow git stats from a participant-file subset.

    Returns ``(total_commits, bug_fixes, bug_fix_ratio, last_modified,
    authors, health_score)`` based ONLY on commits that touch at
    least one file in ``flow_paths``. When the participant set is
    empty falls back to the parent feature's commits (caller should
    not call this with empty paths but defended for safety).
    """
    if not flow_paths:
        return None
    touching = [
        c for c in feat_commits
        if any(f in flow_paths for f in c.files_changed)
    ]
    total = len(touching)
    bugs = sum(1 for c in touching if c.is_bug_fix)
    ratio = (bugs / total) if total else 0.0
    last_mod = None
    authors_set: set[str] = set()
    for c in touching:
        authors_set.add(c.author)
        if last_mod is None or c.date > last_mod:
            last_mod = c.date
    health = feat_health_fn(ratio, total, touching)
    return total, bugs, ratio, last_mod, sorted(authors_set), health


def _inject_new_pipeline_flows(
    feature_map,
    flows: dict[str, list[str]],
    flow_descriptions: dict[str, dict[str, str]],
    commits,
    flow_participants: dict | None = None,
) -> None:
    """Attach Sprint 4 (tool-augmented) flow names + descriptions to the
    feature_map.

    Sprint 4's ``detect_flows_with_tools`` writes flow names into
    ``DeepScanResult.flows`` and per-flow descriptions into
    ``flow_descriptions`` with an ``(entry: file:line)`` suffix. We
    parse that suffix back into first-class
    ``Flow.entry_point_file`` / ``Flow.entry_point_line`` fields, so
    downstream consumers (dashboard, MCP) read structured data
    instead of regexing it back out.

    When ``flow_participants`` is provided (Sprint 7 trace_flows
    output), each Flow also gets a populated ``participants`` list
    with file / layer / symbol-range info for the entire call-graph
    reach.

    Per feature we build one ``Flow`` Pydantic per Sprint 4 name.
    The flow's ``paths`` inherit the parent feature's path list
    (Sprint 4 does flow-level naming, not file-level attribution),
    and metric fields inherit the parent feature's totals.
    """
    if not flows:
        return
    from faultline.analyzer.features import _calculate_health
    from faultline.models.types import Flow, FlowParticipant

    fp_map = flow_participants or {}
    dropped_hallucinated = 0

    for feat in feature_map.features:
        new_flow_names = flows.get(feat.name) or []
        if not new_flow_names:
            continue
        per_flow_descs = flow_descriptions.get(feat.name) or {}
        per_flow_traces = fp_map.get(feat.name) or {}
        # Pre-filter the parent feature's commits once — every flow
        # under this feature will only need to consider this subset
        # when computing its own stats.
        feat_path_set = set(feat.paths)
        feat_commits = [
            c for c in commits
            if any(f in feat_path_set for f in c.files_changed)
        ]
        new_flows: list[Flow] = []
        for name in new_flow_names:
            clean_desc, entry_file, entry_line = _split_entry_trail(
                per_flow_descs.get(name),
            )
            participants: list[FlowParticipant] = []
            for tp in per_flow_traces.get(name, []) or []:
                participants.append(FlowParticipant(
                    path=tp.file,
                    layer=tp.layer or "support",
                    depth=tp.depth,
                    side_effect_only=tp.side_effect_only,
                    symbols=list(tp.symbols),
                ))

            # P3: drop hallucinated flows. A flow with 0 participants
            # AND no entry_point recorded by Sonnet got NO file
            # attribution from any path: BFS yielded nothing,
            # directory-neighbor fallback yielded nothing, keyword
            # backfill yielded nothing. The flow name was generated
            # without grounding in real code. Keep flows whose
            # entry_point WAS recorded (they're real, just untraced)
            # — those still surface a meaningful "this is where it
            # starts" line in the dashboard.
            if not participants and entry_file is None:
                dropped_hallucinated += 1
                continue

            # P2: compute per-flow git stats from participant files
            # when the trace produced ≥2 unique participants. Fewer
            # than 2 = trace was too shallow to differentiate this
            # flow from its siblings, so inherit feature stats
            # (matches prior behavior — no regression).
            unique_paths = {p.path for p in participants if p.path}
            stats = None
            if len(unique_paths) >= 2:
                stats = _flow_specific_stats(
                    unique_paths, feat_commits, _calculate_health,
                )
            if stats is not None:
                total, bugs, ratio, last_mod, authors, health = stats
                flow_paths = sorted(unique_paths)
                # No commits touched the flow's participant subset
                # (rare — usually means newly-added files outside the
                # diff window). Fall back to parent for required
                # Pydantic fields so Flow.last_modified isn't None.
                if last_mod is None:
                    last_mod = feat.last_modified
                if not authors:
                    authors = list(feat.authors)
            else:
                total = feat.total_commits
                bugs = feat.bug_fixes
                ratio = feat.bug_fix_ratio
                last_mod = feat.last_modified
                authors = list(feat.authors)
                health = feat.health_score
                flow_paths = list(feat.paths)

            new_flows.append(Flow(
                name=name,
                description=clean_desc,
                entry_point_file=entry_file,
                entry_point_line=entry_line,
                paths=flow_paths,
                authors=authors,
                total_commits=total,
                bug_fixes=bugs,
                bug_fix_ratio=ratio,
                last_modified=last_mod,
                health_score=health,
                participants=participants,
            ))
        feat.flows = new_flows

    if dropped_hallucinated:
        import logging as _logging
        _logging.getLogger(__name__).info(
            "_inject_new_pipeline_flows: dropped %d hallucinated flow(s) "
            "(no entry_point + no participants)",
            dropped_hallucinated,
        )


def _strip_src_prefix(
    files: list[str],
    src: str | None,
) -> tuple[list[str], str]:
    """
    Strips the --src prefix from file paths so LLM/heuristic sees clean relative paths.
    Returns (normalized_files, prefix_to_restore).

    Example:
        src/views/EDR/Page.tsx  →  EDR/Page.tsx  (prefix = "src/views/")
    """
    if not src:
        return files, ""
    prefix = src.rstrip("/") + "/"
    stripped = [f[len(prefix):] for f in files if f.startswith(prefix)]
    return stripped, prefix


def _validate_llm_access(
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str,
) -> None:
    """Validates LLM connectivity before the long git analysis. Exits on failure."""
    if provider == "anthropic":
        from faultline.llm.detector import validate_api_key
        console.print("[dim]Validating Anthropic API key...[/dim]")
        is_valid, error_msg = validate_api_key(api_key=api_key)
        if not is_valid:
            console.print(f"[red]✗ {error_msg}[/red]")
            raise typer.Exit(1)
        console.print("[green]✓[/green] API key valid")

    elif provider == "ollama":
        from faultline.llm.detector import validate_ollama, _DEFAULT_OLLAMA_MODEL
        resolved_model = model or _DEFAULT_OLLAMA_MODEL
        console.print(f"[dim]Checking Ollama ({resolved_model})...[/dim]")
        is_valid, error_msg = validate_ollama(model=resolved_model, host=ollama_url)
        if not is_valid:
            console.print(f"[red]✗ {error_msg}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] Ollama ready ({resolved_model})")

    elif provider == "deepseek":
        from faultline.llm.deepseek_client import validate_deepseek, _DEFAULT_MODEL as _DS_MODEL
        resolved_model = model or _DS_MODEL
        console.print(f"[dim]Validating DeepSeek ({resolved_model})...[/dim]")
        # deepseek_key is passed via api_key parameter in CLI flow
        is_valid, error_msg = validate_deepseek(api_key=api_key, model=resolved_model)
        if not is_valid:
            console.print(f"[red]✗ {error_msg}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] DeepSeek ready ({resolved_model})")


def _merge_and_name_with_llm(
    cluster_mapping: dict[str, list[str]],
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str,
    commits=None,
    layer_context: str = "",
    cluster_edges: dict[str, dict[str, int]] | None = None,
) -> dict[str, list[str]]:
    """Merges import-graph clusters into business features and names them.

    Unlike simple naming, the LLM can merge N clusters → M features (M ≤ N),
    grouping clusters that serve the same business purpose even when they
    don't share direct import connections (e.g. a Redux slice + its page component).

    When commits are provided, top commit message keywords per cluster are injected
    into the prompt as semantic naming hints.

    Results are cached by cluster structure hash — same codebase → same output.
    Falls back to the original cluster_mapping on any LLM error.
    """
    if provider == "anthropic":
        from faultline.llm.detector import merge_and_name_clusters_llm
        console.print("[blue]Merging & naming features with Claude...[/blue]")
        named = merge_and_name_clusters_llm(
            cluster_mapping, api_key=api_key, commits=commits,
            layer_context=layer_context, cluster_edges=cluster_edges,
        )

    elif provider == "ollama":
        from faultline.llm.detector import merge_and_name_clusters_ollama, _DEFAULT_OLLAMA_MODEL
        resolved_model = model or _DEFAULT_OLLAMA_MODEL
        console.print(f"[blue]Merging & naming features with Ollama ({resolved_model})...[/blue]")
        named = merge_and_name_clusters_ollama(
            cluster_mapping, model=resolved_model, host=ollama_url, commits=commits, layer_context=layer_context
        )

    elif provider == "deepseek":
        from faultline.llm.detector import merge_and_name_clusters_deepseek, _DEFAULT_DEEPSEEK_MODEL
        resolved_model = model or _DEFAULT_DEEPSEEK_MODEL
        console.print(f"[blue]Merging & naming features with DeepSeek ({resolved_model})...[/blue]")
        named = merge_and_name_clusters_deepseek(
            cluster_mapping, api_key=api_key, model=resolved_model, commits=commits,
            layer_context=layer_context, cluster_edges=cluster_edges,
        )

    else:
        named = cluster_mapping

    labels = {"anthropic": "Claude", "ollama": "Ollama", "deepseek": "DeepSeek"}
    console.print(f"[green]✓[/green] {labels.get(provider, provider)} merged → {len(named)} features")
    return named


def _detect_with_llm(
    files: list[str],
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str,
    commits=None,
    path_prefix: str = "",
    signatures=None,
    layer_context: str = "",
) -> dict[str, list[str]]:
    """Sends files directly to LLM for feature detection (no import graph).

    Used for Python, Ruby, Go repos where import graph is unavailable.
    Falls back to directory heuristic on any LLM error.
    """
    if provider == "anthropic":
        from faultline.llm.detector import detect_features_llm
        result = detect_features_llm(
            files, api_key=api_key, commits=commits,
            path_prefix=path_prefix, signatures=signatures,
            layer_context=layer_context,
        )
    elif provider == "ollama":
        from faultline.llm.detector import detect_features_ollama, _DEFAULT_OLLAMA_MODEL
        resolved_model = model or _DEFAULT_OLLAMA_MODEL
        result = detect_features_ollama(
            files, model=resolved_model, host=ollama_url, commits=commits,
            path_prefix=path_prefix, signatures=signatures,
            layer_context=layer_context,
        )
    elif provider == "deepseek":
        from faultline.llm.detector import detect_features_deepseek, _DEFAULT_DEEPSEEK_MODEL
        resolved_model = model or _DEFAULT_DEEPSEEK_MODEL
        result = detect_features_deepseek(
            files, api_key=api_key, model=resolved_model, commits=commits,
            path_prefix=path_prefix, signatures=signatures,
            layer_context=layer_context,
        )
    else:
        result = {}

    if result:
        labels = {"anthropic": "Claude", "ollama": "Ollama", "deepseek": "DeepSeek"}
        console.print(f"[green]✓[/green] {labels.get(provider, provider)} detected {len(result)} features")
        return result

    # Fallback to heuristic
    console.print("[yellow]LLM detection failed — falling back to directory heuristic[/yellow]")
    return detect_features_from_structure(files)


def _detect_with_candidates(
    files: list[str],
    candidates: dict[str, list[str]],
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str,
    commits=None,
    path_prefix: str = "",
) -> dict[str, list[str]]:
    """Uses candidate-based detection: heuristics + LLM verification.

    Falls back to raw candidates if LLM fails (still better than directory heuristic).
    """
    if provider == "anthropic":
        from faultline.llm.detector import detect_features_with_candidates
        result = detect_features_with_candidates(
            files, candidates, api_key=api_key, commits=commits,
            path_prefix=path_prefix,
        )
    else:
        # For Ollama, fall back to raw candidates for now
        # TODO: add Ollama support for candidate verification
        result = candidates

    if result:
        label = "Claude" if provider == "anthropic" else "Ollama"
        console.print(f"[green]✓[/green] {label} verified → {len(result)} features")
        return result

    # Fallback to raw candidates
    console.print("[yellow]LLM verification failed — using raw candidates[/yellow]")
    return candidates


def _apply_feature_coverage(
    feature_map: "FeatureMap",
    coverage_data: dict[str, float],
    path_prefix: str,
    detailed: dict[str, "FileCoverage"] | None = None,
) -> None:
    """Compute coverage per feature/flow from coverage report data.

    Two-tier:
      Tier 1 (preferred): when ``detailed`` is provided AND the feature
        has ``shared_attributions``, compute line-scoped coverage by
        averaging ``FileCoverage.coverage_for_range`` over each
        ``SymbolAttribution.line_ranges``. Same logic for flows using
        ``flow.symbol_attributions``.
      Tier 2 (fallback): file-level — average pct over all
        feature/flow files (legacy behaviour).

    Mutates ``feature_map.features`` in place.
    """
    from faultline.analyzer.features import _is_test_file

    for feature in feature_map.features:
        # Tier 1: line-scoped — prefer per-feature participants
        # (Refactor Day 3) over legacy shared_attributions.
        scoped_pct = _coverage_for_attributions(
            feature.participants or feature.shared_attributions,
            detailed, path_prefix,
        )
        if scoped_pct is not None:
            feature.coverage_pct = scoped_pct
        else:
            # Tier 2: file-level average
            coverages = []
            for file_path in feature.paths:
                if _is_test_file(file_path):
                    continue
                full_path = f"{path_prefix}{file_path}" if path_prefix else file_path
                pct = _match_coverage(coverage_data, file_path, full_path)
                if pct is not None:
                    coverages.append(pct)
            if coverages:
                feature.coverage_pct = round(sum(coverages) / len(coverages), 1)

        # Apply coverage to flows within this feature.
        # Refactor Day 4: prefer ``flow.participants`` over legacy
        # ``flow.symbol_attributions`` — the former is the Sprint 7
        # call-graph trace populated whenever --trace-flows ran.
        for flow in feature.flows:
            flow_scoped = _coverage_for_attributions(
                flow.participants or flow.symbol_attributions,
                detailed, path_prefix,
            )
            if flow_scoped is not None:
                flow.coverage_pct = flow_scoped
                continue
            flow_coverages = []
            for file_path in flow.paths:
                if _is_test_file(file_path):
                    continue
                full_path = f"{path_prefix}{file_path}" if path_prefix else file_path
                pct = _match_coverage(coverage_data, file_path, full_path)
                if pct is not None:
                    flow_coverages.append(pct)
            if flow_coverages:
                flow.coverage_pct = round(sum(flow_coverages) / len(flow_coverages), 1)


def _coverage_for_attributions(
    attributions,
    detailed: "dict[str, FileCoverage] | None",
    path_prefix: str,
) -> float | None:
    """Average coverage over each item's line ranges.

    Refactor Day 4: accepts either ``SymbolAttribution`` or
    ``FlowParticipant`` items. SymbolAttribution exposes
    ``file_path`` + ``line_ranges`` directly; FlowParticipant
    exposes ``path`` + ``symbols`` and we derive ranges from each
    SymbolRange's ``start_line``/``end_line``.

    Returns ``None`` when no detailed data is available, no
    attributions were given, or none of the attributed files have
    line-level coverage. The caller should then fall back to file-
    level scoring.
    """
    if not attributions or not detailed:
        return None

    pcts: list[float] = []
    for item in attributions:
        # Duck-type both shapes.
        file_path = getattr(item, "file_path", None) or getattr(item, "path", "")
        ranges = getattr(item, "line_ranges", None)
        if ranges is None:
            symbols = getattr(item, "symbols", []) or []
            ranges = [(s.start_line, s.end_line) for s in symbols]
        if not ranges or not file_path:
            continue
        # Try with and without path_prefix
        candidates = [file_path]
        if path_prefix:
            candidates.append(f"{path_prefix}{file_path}")
        fc = None
        for cand in candidates:
            fc = detailed.get(cand)
            if fc is not None:
                break
        if fc is None:
            continue
        for (start, end) in ranges:
            range_pct = fc.coverage_for_range(start, end)
            if range_pct is not None:
                pcts.append(range_pct)
    if not pcts:
        return None
    return round(sum(pcts) / len(pcts), 1)


def _match_coverage(
    coverage_data: dict[str, float],
    file_path: str,
    full_path: str,
) -> float | None:
    """Match a file against coverage data with flexible path matching."""
    pct = coverage_data.get(full_path) or coverage_data.get(file_path)
    if pct is not None:
        return pct
    # Fuzzy suffix match for mismatched prefixes
    for cov_path, cov_pct in coverage_data.items():
        if cov_path.endswith(file_path) or file_path.endswith(cov_path.lstrip("/")):
            return cov_pct
    return None


def _detect_flows(
    feature_map,
    repo_path: str,
    analysis_files: list[str],
    path_prefix: str,
    commits,
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str,
    signatures: dict | None = None,
    remote_url: str = "",
    coverage_data: dict | None = None,
    e2e_anchors: dict | None = None,
):
    """
    Runs flow detection for each feature and attaches Flow objects to the FeatureMap.
    Returns the updated FeatureMap (features with .flows populated).
    """
    from faultline.llm.flow_detector import detect_flows_llm, detect_flows_ollama, _DEFAULT_OLLAMA_MODEL as _OLLAMA_MODEL
    from faultline.llm.flow_detector import _FlowFileMapping

    labels = {"anthropic": "Claude", "ollama": "Ollama", "deepseek": "DeepSeek"}
    label = labels.get(provider, provider)
    console.print(f"[blue]Detecting flows with {label}...[/blue]")

    # Reuse signatures from feature detection if provided; otherwise extract now.
    # analysis_files are stripped of path_prefix, so reconstruct the correct root:
    # git_root/src/ when --src src/ is used, or just git_root otherwise.
    if not signatures:
        from faultline.analyzer.ast_extractor import extract_signatures
        from pathlib import Path as _Path
        extract_root = str(_Path(repo_path) / path_prefix) if path_prefix else repo_path
        signatures = extract_signatures(analysis_files, extract_root)
        console.print(f"[dim]Extracted signatures from {len(signatures)} TS/JS files[/dim]")

    updated_features = []
    total_flows = 0

    for feature in feature_map.features:
        # Restore analysis-relative paths (strip prefix was applied earlier)
        if path_prefix:
            analysis_feature_files = [
                f[len(path_prefix):] for f in feature.paths
                if f.startswith(path_prefix)
            ]
        else:
            analysis_feature_files = list(feature.paths)

        if not analysis_feature_files:
            updated_features.append(feature)
            continue

        # Flow detection works on file signatures, not commits.
        # Metrics (health, bug ratio) may be empty for low-commit features,
        # but the flow structure is still valuable for EMs.

        # Skip flow detection for non-user-facing features (infra, configs, shared)
        _SKIP_FLOW_PREFIXES = (
            "shared-", "app-shell", "constants", "config", "custom-utils",
            "custom-hooks", "shared", "zustand", "slices", "@types",
        )
        if any(feature.name.startswith(p) or feature.name == p for p in _SKIP_FLOW_PREFIXES):
            updated_features.append(feature)
            continue

        # Filter e2e anchors to only those relevant to this feature's files
        feature_file_set = set(analysis_feature_files)
        feature_e2e = {
            flow_name: [f for f in files if f in feature_file_set]
            for flow_name, files in (e2e_anchors or {}).items()
        }
        feature_e2e = {k: v for k, v in feature_e2e.items() if v}

        # Collect commits touching this feature (for co-change enrichment)
        feature_commit_files = set(feature.paths)
        feature_commits = [
            c for c in commits
            if any(f in feature_commit_files for f in c.files_changed)
        ]

        # Detect flows for this feature
        if provider == "anthropic":
            flow_mappings = detect_flows_llm(
                feature_name=feature.name,
                feature_files=analysis_feature_files,
                signatures=signatures,
                api_key=api_key,
                e2e_anchors=feature_e2e or None,
                commits=feature_commits,
            )
        elif provider == "deepseek":
            from faultline.llm.flow_detector import detect_flows_deepseek, _DEFAULT_DEEPSEEK_MODEL as _DS_MODEL
            resolved_model = model or _DS_MODEL
            flow_mappings = detect_flows_deepseek(
                feature_name=feature.name,
                feature_files=analysis_feature_files,
                signatures=signatures,
                api_key=api_key,
                model=resolved_model,
                e2e_anchors=feature_e2e or None,
                commits=feature_commits,
            )
        else:
            resolved_model = model or _OLLAMA_MODEL
            flow_mappings = detect_flows_ollama(
                feature_name=feature.name,
                feature_files=analysis_feature_files,
                signatures=signatures,
                model=resolved_model,
                host=ollama_url,
                e2e_anchors=feature_e2e or None,
                commits=feature_commits,
            )

        if not flow_mappings:
            updated_features.append(feature)
            continue

        # Restore full paths in flow mappings for commit matching
        if path_prefix:
            flow_file_mappings = {
                m.flow_name: [path_prefix + f for f in m.files]
                for m in flow_mappings
            }
        else:
            flow_file_mappings = {m.flow_name: m.files for m in flow_mappings}

        # Build metrics for each flow using the feature's commits
        flows = build_flows_metrics(feature_commits, flow_file_mappings, remote_url=remote_url, coverage_data=coverage_data)

        # Filter out ghost flows (0 commits in the analyzed period)
        flows = [f for f in flows if f.total_commits > 0]

        total_flows += len(flows)

        updated_features.append(feature.model_copy(update={"flows": flows}))

    console.print(f"[green]✓[/green] Detected {total_flows} flows across {len(updated_features)} features")
    return feature_map.model_copy(update={"features": updated_features})


def _run_analytics(
    feature_map,
    posthog_key: str | None,
    posthog_project: str | None,
    posthog_host: str,
    sentry_token: str | None,
    sentry_org: str | None,
    sentry_project: str | None,
    sentry_host: str,
) -> list | None:
    """Fetches analytics data and computes impact scores."""
    import asyncio
    from faultline.integrations.base import PageMetrics, ErrorMetrics, compute_impact_scores

    traffic: list[PageMetrics] = []
    errors: list[ErrorMetrics] = []

    async def _fetch():
        nonlocal traffic, errors

        # PostHog
        if posthog_key and posthog_project:
            from faultline.integrations.posthog_provider import PostHogProvider
            ph = PostHogProvider(
                api_key=posthog_key,
                project_id=posthog_project,
                host=posthog_host,
            )
            console.print("[blue]Connecting to PostHog...[/blue]")
            if await ph.validate_connection():
                console.print("[green]✓[/green] PostHog connected")
                traffic = await ph.get_page_traffic(days=30)
                console.print(f"[dim]  {len(traffic)} routes with traffic data[/dim]")
                ph_errors = await ph.get_error_counts(days=30)
                if ph_errors:
                    errors.extend(ph_errors)
                    console.print(f"[dim]  {len(ph_errors)} routes with error data[/dim]")
            else:
                console.print("[yellow]✗ PostHog connection failed[/yellow]")
            await ph.close()

        # Sentry
        if sentry_token and sentry_org and sentry_project:
            from faultline.integrations.sentry_provider import SentryProvider
            sn = SentryProvider(
                auth_token=sentry_token,
                organization=sentry_org,
                project=sentry_project,
                host=sentry_host,
            )
            console.print("[blue]Connecting to Sentry...[/blue]")
            if await sn.validate_connection():
                console.print("[green]✓[/green] Sentry connected")
                sn_errors = await sn.get_error_counts(days=30)
                if sn_errors:
                    errors.extend(sn_errors)
                    console.print(f"[dim]  {len(sn_errors)} routes with error data[/dim]")
            else:
                console.print("[yellow]✗ Sentry connection failed[/yellow]")
            await sn.close()

    asyncio.run(_fetch())

    if not traffic and not errors:
        console.print("[yellow]No analytics data retrieved[/yellow]")
        return None

    # Build flow dicts for impact computation
    flows_data = []
    for feature in feature_map.features:
        if feature.flows:
            for flow in feature.flows:
                flows_data.append({
                    "name": flow.name,
                    "health_score": flow.health_score,
                    "paths": flow.paths,
                })
        else:
            flows_data.append({
                "name": feature.name,
                "health_score": feature.health_score,
                "paths": feature.paths,
            })

    scores = compute_impact_scores(flows_data, traffic, errors)
    console.print(f"[green]✓[/green] Computed {len(scores)} impact scores")

    return scores


@app.command(name="deep-scan")
def deep_scan(
    repo_path: str = typer.Argument(".", help="Path to the git repository"),
    days: int = typer.Option(365, "--days", "-d", help="Days of history to analyze"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Anthropic API key"),
    max_commits: int = typer.Option(DEFAULT_MAX_COMMITS, "--max-commits", help="Max commits to analyze"),
    src: Optional[list[str]] = typer.Option(None, "--src", help="Source directory filter"),
):
    """Deep scan using Sonnet — high-quality features + flows in one pass.

    Designed for SaaS initial scan. Uses Claude Sonnet for accurate feature
    detection with user flows. More expensive but much more accurate than
    the regular `analyze --llm` command.
    """
    from faultline.analyzer.features import detect_candidates, build_feature_map, build_flows_metrics
    from faultline.llm.sonnet_scanner import deep_scan as run_deep_scan, get_deep_scan_flows, get_deep_scan_descriptions, get_deep_scan_flow_descriptions

    console.print(f"[bold]Deep scanning:[/bold] {repo_path}")

    # 1. Load repo
    repo = load_repo(repo_path)
    if not repo:
        console.print("[red]✗[/red] Not a valid git repository")
        raise typer.Exit(1)

    remote_url = get_remote_url(repo)

    # 2. Get files
    all_files = get_tracked_files(repo)
    path_prefix = ""
    if src:
        path_prefix = src[0]
        all_files = [f for f in all_files if any(f.startswith(s) for s in src)]
    analysis_files = [f[len(path_prefix):] if path_prefix else f for f in all_files]
    console.print(f"[dim]{len(analysis_files)} files to analyze[/dim]")

    # 3. Get commits
    commits = get_commits(repo, days=days, max_commits=max_commits)
    console.print(f"[dim]{len(commits)} commits in {days} days[/dim]")

    # 4. Generate candidates (heuristics)
    candidates = detect_candidates(analysis_files)
    real_count = len([n for n, p in candidates.items()
                      if len(p) >= 2 and n not in {"backend", "frontend", "root", "init", "packages", "web", "api", "lib"}])
    console.print(f"[blue]Heuristics: {real_count} candidates[/blue]")

    # 4b. Extract signatures for flow detection
    from faultline.analyzer.ast_extractor import extract_signatures
    extract_root = str(Path(str(repo.working_tree_dir)) / path_prefix) if path_prefix else str(repo.working_tree_dir)
    sigs = extract_signatures(analysis_files, extract_root)
    if sigs:
        console.print(f"[dim]Extracted signatures from {len(sigs)} files[/dim]")

    # 5. Deep scan with Sonnet
    console.print("[bold blue]Calling Sonnet for deep analysis...[/bold blue]")
    feature_paths = run_deep_scan(
        analysis_files, candidates, api_key=api_key, signatures=sigs,
    )

    if not feature_paths:
        console.print("[red]✗[/red] Deep scan failed")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Sonnet detected {len(feature_paths)} features")

    # 6. Restore full paths
    if path_prefix:
        feature_paths = {
            name: [path_prefix + f for f in paths]
            for name, paths in feature_paths.items()
        }

    # 7. Build feature map with commit metrics
    feature_map = build_feature_map(
        repo_path=repo_path,
        commits=commits,
        feature_paths=feature_paths,
        days=days,
        remote_url=remote_url,
    )

    # 8. Inject descriptions from Sonnet (fuzzy match names)
    from faultline.llm.sonnet_scanner import match_flows_to_features
    descriptions = get_deep_scan_descriptions()
    feat_names = [f.name for f in feature_map.features]
    # Fuzzy match descriptions
    for feat in feature_map.features:
        for desc_name, desc in descriptions.items():
            if (feat.name == desc_name or feat.name in desc_name or desc_name in feat.name
                    or feat.name.rstrip("s") == desc_name.rstrip("s")):
                feat.description = desc
                break

    # 9. Inject flows from Sonnet with commit metrics
    raw_flow_data = get_deep_scan_flows()  # dict[sonnet_name → list[flow_name]]
    flow_data = match_flows_to_features(raw_flow_data, feat_names)
    flow_descriptions = get_deep_scan_flow_descriptions()

    for feat in feature_map.features:
        if feat.name not in flow_data:
            continue

        # Sonnet returns flow names without file mappings.
        # Each flow gets all feature files — commit metrics differentiate them.
        flow_names = flow_data[feat.name]
        flow_file_mappings = {fn: list(feat.paths) for fn in flow_names}

        feat_commits = [
            c for c in commits
            if any(f in set(feat.paths) for f in c.files_changed)
        ]
        feat.flows = build_flows_metrics(feat_commits, flow_file_mappings, remote_url)

        # Inject flow descriptions
        feat_flow_descs = flow_descriptions.get(feat.name, {})
        for flow in feat.flows:
            if flow.name in feat_flow_descs:
                flow.description = feat_flow_descs[flow.name]

    # 10. Report
    total_flows = sum(len(f.flows) for f in feature_map.features)
    console.print(f"[green]✓[/green] {total_flows} flows detected")
    print_report(feature_map)

    # 11. Save
    output_path = output or None
    saved = write_feature_map(feature_map, output_path)
    console.print(f"\n[green]Saved:[/green] {saved}")


@app.command()
def update(
    repo_path: str = typer.Argument(".", help="Path to the git repository"),
    scan: Optional[str] = typer.Option(None, "--scan", "-s", help="Path to existing feature-map JSON"),
    days: int = typer.Option(365, "--days", "-d", help="Days of history"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path"),
    max_commits: int = typer.Option(DEFAULT_MAX_COMMITS, "--max-commits"),
):
    """Incremental update — refreshes an existing scan with new commits.

    Finds the latest feature-map for this repo and updates it with commits
    since the last analysis. No LLM calls — pure heuristic matching.

    Much faster and cheaper than a full re-scan.
    """
    from faultline.analyzer.incremental import incremental_update
    from faultline.models.types import FeatureMap
    import glob

    console.print(f"[bold]Incremental update:[/bold] {repo_path}")

    # 1. Load repo
    repo = load_repo(repo_path)
    if not repo:
        console.print("[red]✗[/red] Not a valid git repository")
        raise typer.Exit(1)

    # 2. Find existing scan
    if scan:
        scan_path = Path(scan)
    else:
        # Find latest scan for this repo in ~/.faultline/
        home = Path.home() / ".faultline"
        repo_name = Path(repo_path).resolve().name.lower()
        pattern = str(home / f"feature-map-{repo_name}-*.json")
        matches = sorted(glob.glob(pattern))
        if not matches:
            console.print(f"[red]✗[/red] No existing scan found for '{repo_name}' in {home}")
            console.print("[dim]Run 'faultline deep-scan' first to create initial scan[/dim]")
            raise typer.Exit(1)
        scan_path = Path(matches[-1])

    console.print(f"[dim]Base scan: {scan_path.name}[/dim]")

    # 3. Load existing feature map
    feature_map = FeatureMap.model_validate_json(scan_path.read_text())
    last_analyzed = feature_map.analyzed_at
    console.print(f"[dim]Last analyzed: {last_analyzed.strftime('%Y-%m-%d %H:%M')}[/dim]")
    console.print(f"[dim]Features: {len(feature_map.features)}, commits: {feature_map.total_commits}[/dim]")

    # 4. Get new commits since last analysis
    from datetime import timezone
    all_commits = get_commits(repo, days=days, max_commits=max_commits)
    new_commits = [c for c in all_commits if c.date > last_analyzed]

    if not new_commits:
        console.print("[green]✓[/green] Already up to date — no new commits")
        raise typer.Exit(0)

    bug_fixes = sum(1 for c in new_commits if c.is_bug_fix)
    console.print(f"[blue]{len(new_commits)} new commits[/blue] ({bug_fixes} bug fixes)")

    # 5. Run incremental update
    updated = incremental_update(feature_map, new_commits)

    # 6. Report changes
    changed = []
    for old_feat, new_feat in zip(
        sorted(feature_map.features, key=lambda f: f.name),
        sorted(updated.features, key=lambda f: f.name),
    ):
        if old_feat.name == new_feat.name:
            delta_commits = new_feat.total_commits - old_feat.total_commits
            delta_bugs = new_feat.bug_fixes - old_feat.bug_fixes
            delta_health = new_feat.health_score - old_feat.health_score
            if delta_commits > 0:
                changed.append((new_feat.name, delta_commits, delta_bugs, delta_health))

    if changed:
        console.print(f"\n[bold]Changed features ({len(changed)}):[/bold]")
        for name, dc, db, dh in sorted(changed, key=lambda x: x[3]):
            health_color = "red" if dh < -5 else "green" if dh > 5 else "dim"
            sign = "+" if dh >= 0 else ""
            console.print(f"  {name}: +{dc} commits, +{db} bugs, [{health_color}]{sign}{dh:.0f} health[/{health_color}]")

    # 7. Report & save
    print_report(updated)
    output_path = output or None
    saved = write_feature_map(updated, output_path)
    console.print(f"\n[green]Saved:[/green] {saved}")


@app.command()
def evolve(
    repo_path: str = typer.Argument(".", help="Path to the git repository"),
    scan: Optional[str] = typer.Option(None, "--scan", "-s", help="Path to existing feature-map JSON"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Anthropic API key"),
):
    """Evolve — smart update that detects new features and flows.

    Compares current repo files with the last scan. New files matched to
    existing features by heuristics. New directories sent to Sonnet to
    determine: new feature, new flow, or addition to existing.

    Preserves existing feature map as source of truth.
    """
    from faultline.analyzer.evolve import detect_changes, evolve_with_llm, apply_simple_delta
    from faultline.models.types import FeatureMap
    import glob

    console.print(f"[bold]Evolving:[/bold] {repo_path}")

    # 1. Load repo
    repo = load_repo(repo_path)
    if not repo:
        console.print("[red]✗[/red] Not a valid git repository")
        raise typer.Exit(1)

    # 2. Find existing scan
    if scan:
        scan_path = Path(scan)
    else:
        home = Path.home() / ".faultline"
        repo_name = Path(repo_path).resolve().name.lower()
        # Find scans matching exact repo name (not prefix matches)
        pattern = str(home / f"feature-map-{repo_name}-*.json")
        matches = sorted(glob.glob(pattern))
        if not matches:
            console.print(f"[red]✗[/red] No existing scan found for '{repo_name}'")
            console.print("[dim]Run 'faultline deep-scan' first[/dim]")
            raise typer.Exit(1)
        scan_path = Path(matches[-1])

    console.print(f"[dim]Base scan: {scan_path.name}[/dim]")

    # 3. Load existing feature map
    feature_map = FeatureMap.model_validate_json(scan_path.read_text())
    console.print(f"[dim]Features: {len(feature_map.features)}, "
                  f"flows: {sum(len(f.flows) for f in feature_map.features)}[/dim]")

    # 4. Get current tracked files
    current_files = get_tracked_files(repo)
    console.print(f"[dim]Current files: {len(current_files)}[/dim]")

    # 5. Detect changes
    delta = detect_changes(feature_map, current_files)

    new_matched = sum(len(v) for v in delta.matched_files.values())
    console.print(f"[blue]Changes:[/blue] {len(delta.new_files) + new_matched + sum(1 for d in delta.new_directories)} new files, "
                  f"{len(delta.deleted_files)} deleted, "
                  f"{len(delta.new_directories)} new directories")

    if not delta.new_files and not delta.deleted_files and not delta.new_directories and not delta.matched_files:
        console.print("[green]✓[/green] No structural changes — feature map is up to date")
        raise typer.Exit(0)

    # 6. Apply changes
    if delta.needs_llm:
        console.print(f"[bold blue]New directories detected — calling Sonnet...[/bold blue]")
        updated = evolve_with_llm(feature_map, delta, current_files, api_key=api_key)
    else:
        console.print("[dim]No new directories — applying heuristic changes only[/dim]")
        updated = apply_simple_delta(feature_map, delta)

    # 7. Report
    old_feat_count = len(feature_map.features)
    new_feat_count = len(updated.features)
    old_flow_count = sum(len(f.flows) for f in feature_map.features)
    new_flow_count = sum(len(f.flows) for f in updated.features)

    if new_feat_count > old_feat_count:
        console.print(f"[green]✓ {new_feat_count - old_feat_count} new feature(s) added[/green]")
    if new_flow_count > old_flow_count:
        console.print(f"[green]✓ {new_flow_count - old_flow_count} new flow(s) added[/green]")
    if delta.matched_files:
        console.print(f"[dim]{new_matched} files added to existing features[/dim]")
    if delta.deleted_files:
        console.print(f"[dim]{len(delta.deleted_files)} deleted files cleaned up[/dim]")

    print_report(updated)

    output_path = output or None
    saved = write_feature_map(updated, output_path)
    console.print(f"\n[green]Saved:[/green] {saved}")


@app.command()
def refresh(
    repo_path: str = typer.Argument(".", help="Repo to refresh (must have a prior scan in ~/.faultline/)"),
    map_path: Optional[str] = typer.Option(
        None, "--map",
        help="Path to existing feature-map JSON. Defaults to the most recent ~/.faultline/feature-map-*.json",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Where to write the refreshed feature map. Defaults to ~/.faultline/ with a new timestamp.",
    ),
    check_only: bool = typer.Option(
        False, "--check",
        help="Report freshness without writing a refreshed map.",
    ),
    detect_new: bool = typer.Option(
        False, "--detect-new",
        help=(
            "After refresh, classify orphan files (not in any feature) via LLM. "
            "Proposes extensions of existing features or entirely new features. "
            "Requires ANTHROPIC_API_KEY."
        ),
    ),
    refresh_symbols: bool = typer.Option(
        False, "--refresh-symbols",
        help=(
            "Update symbol-level attributions for flows: clean up removed "
            "symbols and re-attribute newly added ones. Body-only changes "
            "are preserved. Requires ANTHROPIC_API_KEY only when new symbols "
            "appear."
        ),
    ),
    auto_apply: bool = typer.Option(
        False, "--auto-apply",
        help="With --detect-new, automatically apply high-confidence proposals to the map.",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help="Anthropic API key for --detect-new (defaults to ANTHROPIC_API_KEY env var)",
    ),
):
    """
    Incrementally update a feature map to match the current git HEAD.

    Runs the existing analyzer/incremental.py pipeline (no LLM calls)
    and updates content/symbol hashes. Orders of magnitude cheaper
    than a full --llm scan and preserves flow + symbol attributions
    on untouched features.

    Examples:
        faultlines refresh
        faultlines refresh /path/to/repo --check
        faultlines refresh . --output latest.json
    """
    import json as _json
    from faultline.cache.refresh import refresh_feature_map
    from faultline.cache.freshness import check_freshness
    from faultline.models.types import FeatureMap
    from faultline.output.writer import write_feature_map

    # Locate the map to refresh
    if map_path:
        map_file = Path(map_path).expanduser()
    else:
        home = Path.home() / ".faultline"
        candidates = sorted(home.glob("feature-map-*.json"))
        if not candidates:
            console.print(
                "[red]No feature map found.[/red] "
                "Run `faultlines analyze . --llm --flows` first."
            )
            raise typer.Exit(1)
        map_file = candidates[-1]

    if not map_file.exists():
        console.print(f"[red]Map not found:[/red] {map_file}")
        raise typer.Exit(1)

    console.print(f"[dim]Loading:[/dim] {map_file}")
    fm = FeatureMap.model_validate_json(map_file.read_text())

    if check_only:
        report = check_freshness(fm, repo_path)
        if not report.is_stale:
            console.print("[green]✓ Feature map is up to date with HEAD[/green]")
        else:
            console.print(
                f"[yellow]Stale:[/yellow] {report.commits_behind} commit(s) behind. "
                f"{report.changed_files_count} file(s) changed. "
                f"{'New files detected.' if report.has_new_files else ''}"
            )
        return

    result = refresh_feature_map(fm, repo_path)

    if not result.freshness_before.is_stale and not detect_new and not refresh_symbols:
        console.print("[green]✓ Already up to date — no refresh needed[/green]")
        return

    updated_map = result.updated_map

    # Symbol-level incremental (opt-in)
    if refresh_symbols:
        import os as _os
        _api_key = api_key or _os.environ.get("ANTHROPIC_API_KEY")
        from faultline.cache.symbols import refresh_symbol_attributions
        console.print("[blue]Refreshing symbol attributions...[/blue]")
        sym_report = refresh_symbol_attributions(
            feature_map=updated_map,
            repo_path=repo_path,
            api_key=_api_key,
        )
        console.print(f"[dim]{sym_report.summary()}[/dim]")
        if sym_report.symbols_added and not _api_key:
            console.print(
                "[yellow]New symbols detected but no ANTHROPIC_API_KEY — "
                "re-attribution skipped. Existing attributions preserved.[/yellow]"
            )

    # Orphan classification (opt-in, LLM-based)
    if detect_new and result.orphan_files:
        console.print(
            f"[blue]Classifying {len(result.orphan_files)} orphan file(s)...[/blue]"
        )
        from faultline.cache.discovery import discover_from_orphans, apply_report
        report = discover_from_orphans(
            orphan_files=result.orphan_files,
            feature_map=updated_map,
            api_key=api_key,
        )
        console.print(f"[dim]{report.summary()}[/dim]")

        # Pretty-print proposals
        if report.extensions:
            console.print("\n[bold]Extensions of existing features:[/bold]")
            for p in report.extensions:
                files_str = _trim_file_list(p.files)
                console.print(
                    f"  [green]→[/green] [bold]{p.extends_feature}[/bold] "
                    f"gains {len(p.files)} file(s) "
                    f"[dim]({p.confidence}, {p.reason})[/dim]"
                )
                console.print(f"    {files_str}")

        if report.new_features:
            console.print("\n[bold]Candidate new features:[/bold]")
            for p in report.new_features:
                files_str = _trim_file_list(p.files)
                console.print(
                    f"  [cyan]+[/cyan] [bold]{p.new_feature_name}[/bold] "
                    f"({len(p.files)} files, {p.confidence})"
                )
                if p.new_feature_description:
                    console.print(f"    [dim]{p.new_feature_description}[/dim]")
                console.print(f"    {files_str}")

        if auto_apply:
            applied = apply_report(updated_map, report, only_high_confidence=True)
            console.print(
                f"\n[green]✓ Auto-applied {applied} high-confidence proposal(s)[/green]"
            )
        elif report.extensions or report.new_features:
            console.print(
                "\n[dim]Review above. Re-run with --detect-new --auto-apply to apply "
                "high-confidence proposals, or run a full `faultlines analyze` for a "
                "fresh scan.[/dim]"
            )

    # Save updated map
    saved_path = write_feature_map(updated_map, output)

    console.print(f"\n[green]✓ Refresh complete[/green]")
    console.print(f"  Commits behind before: {result.freshness_before.commits_behind}")
    console.print(f"  Files modified: {result.files_truly_modified}")
    console.print(f"  Files added: {result.files_added}")
    console.print(f"  Files removed: {result.files_removed}")
    console.print(f"  LLM calls saved: ~{result.llm_calls_saved}")
    if result.orphan_files and not detect_new:
        console.print(
            f"  [yellow]⚠ {len(result.orphan_files)} orphan file(s) not mapped to any feature.[/yellow]"
        )
        console.print(
            f"    Run `faultlines refresh --detect-new` to classify them via LLM."
        )
    console.print(f"\n[dim]Saved:[/dim] {saved_path}")


def _trim_file_list(files: list[str], max_shown: int = 4) -> str:
    """Format a file list for terminal display."""
    if len(files) <= max_shown:
        return ", ".join(files)
    return ", ".join(files[:max_shown]) + f", +{len(files) - max_shown} more"


@app.command()
def watch(
    repo_path: str = typer.Argument(".", help="Repo to watch"),
    debounce: float = typer.Option(
        30.0, "--debounce",
        help="Seconds of silence after last file change before refreshing (default 30)",
    ),
    daemon: bool = typer.Option(
        False, "--daemon",
        help="Run in background (fork + detach). Use `faultlines watch-stop` to kill.",
    ),
    map_path: Optional[str] = typer.Option(
        None, "--map",
        help="Explicit feature-map JSON. Defaults to latest in ~/.faultline/.",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Log refresh events"),
):
    """
    Watch a repo and auto-refresh the feature map on file changes.

    Foreground by default (Ctrl-C to stop). Use --daemon to detach.
    Only triggers metric refresh — no LLM calls, no cost.

    Examples:
        faultlines watch                         # foreground, current dir
        faultlines watch /path/to/repo --daemon  # background
        faultlines watch . --debounce 10         # react faster
    """
    from faultline.watch import run_watcher, start_daemon

    if daemon:
        try:
            pid = start_daemon(repo_path, debounce_seconds=debounce, map_path=map_path)
            console.print(f"[green]✓ Watcher started[/green] (pid {pid})")
            console.print(f"[dim]Stop with: faultlines watch-stop {repo_path}[/dim]")
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
    else:
        try:
            run_watcher(
                repo_path=repo_path,
                debounce_seconds=debounce,
                map_path=map_path,
                verbose=verbose,
            )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)


@app.command(name="watch-status")
def watch_status(
    repo_path: str = typer.Argument(".", help="Repo to check"),
):
    """Check whether a watcher daemon is running for this repo."""
    from faultline.watch import watcher_status
    status = watcher_status(repo_path)
    if status.running:
        import datetime as _dt
        started = _dt.datetime.fromtimestamp(status.started_at or 0).strftime("%Y-%m-%d %H:%M")
        console.print(f"[green]✓ Running[/green] (pid {status.pid}, started {started})")
    else:
        console.print("[yellow]Not running[/yellow]")


@app.command(name="watch-stop")
def watch_stop(
    repo_path: str = typer.Argument(".", help="Repo whose watcher to stop"),
):
    """Stop a background watcher daemon."""
    from faultline.watch import stop_daemon
    if stop_daemon(repo_path):
        console.print("[green]✓ Stopped[/green]")
    else:
        console.print("[yellow]No watcher running[/yellow]")


@app.command(hidden=True)
def pull(
    repo: Optional[str] = typer.Argument(
        None,
        help="Repo slug (defaults to current directory's folder name). Example: 'soc0'",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Where to save the merged feature map. Default: ~/.faultline/feature-map-<slug>-cloud.json",
    ),
):
    """Pull the latest scan for a repo with user overrides applied.

    Overrides include custom feature names, aliases, and labels set in the
    dashboard. Requires FAULTLINE_API_KEY. MCP then matches queries
    against those overrides — if the team calls it 'labels', AI finds it
    under 'labels' even though the LLM originally named it 'tags'.
    """
    import json
    import os
    import re
    from faultline.cloud.sync import pull_feature_map

    if repo is None:
        repo = Path.cwd().name
    slug = re.sub(r"[^a-z0-9]+", "-", repo.lower())[:60]

    if not os.environ.get("FAULTLINES_EXPERIMENTAL"):
        rprint(
            "[yellow]`pull` is alpha and not yet available in public beta.[/yellow]\n"
            "[dim]Set FAULTLINES_EXPERIMENTAL=1 to opt in once the cloud dashboard launches.[/dim]"
        )
        raise typer.Exit(code=1)

    if not os.environ.get("FAULTLINE_API_KEY"):
        rprint("[red]FAULTLINE_API_KEY not set.[/red] Create a key at your dashboard → Settings → API keys.")
        raise typer.Exit(code=1)

    rprint(f"Pulling latest scan for [bold]{slug}[/bold]…")
    data = pull_feature_map(slug)
    if data is None:
        rprint(f"[yellow]No scan found for '{slug}'.[/yellow]")
        raise typer.Exit(code=1)

    target = output or (Path.home() / ".faultline" / f"feature-map-{slug}-cloud.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2))

    features = data.get("features", [])
    applied = data.get("_meta", {}).get("overrides_applied", 0)
    renamed = sum(1 for f in features if f.get("display_name") and f["display_name"] != f.get("original_name", f.get("name")))
    rprint(f"[green]✓[/green] Saved {len(features)} features to {target}")
    rprint(f"  {applied} override(s) available · {renamed} renamed")


@app.command(name="suggest-config")
def suggest_config(
    repo_path: str = typer.Argument(
        ".",
        help="Path to the git repository",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        help=(
            "Write suggestions to .faultline.yaml instead of "
            "printing to stdout. Existing file is preserved — "
            "suggestions land under a fresh ``# Suggested by "
            "faultline suggest-config`` header."
        ),
        is_flag=True,
    ),
):
    """Suggest a starter ``.faultline.yaml`` from repo signals.

    Discovers canonical-feature names from:
      - Workspace package names (package.json, pyproject.toml,
        Cargo.toml).
      - CODEOWNERS team assignments (root, .github/, docs/).

    Prints the suggestions in YAML form so you can review and
    drop into ``.faultline.yaml`` (or pass ``--write`` to do it
    automatically).
    """
    from faultline.analyzer.auto_alias_discoverer import discover_aliases
    from faultline.analyzer.git import get_tracked_files, load_repo
    import yaml as _yaml

    repo_path_resolved = str(Path(repo_path).resolve())
    try:
        repo = load_repo(repo_path_resolved)
    except Exception as exc:
        console.print(f"[red]Error loading repo:[/red] {exc}")
        raise typer.Exit(1) from exc

    files = get_tracked_files(repo)
    rules = discover_aliases(repo_path_resolved, files)

    if not rules:
        console.print(
            "[yellow]No signals found.[/yellow] No workspace manifest "
            "and no CODEOWNERS file detected — there is nothing to "
            "suggest. You can hand-author a `.faultline.yaml` "
            "instead."
        )
        return

    yaml_block: dict = {"features": {}}
    for r in rules:
        entry: dict[str, object] = {}
        if r.description:
            entry["description"] = r.description
        if r.variants:
            entry["variants"] = list(r.variants)
        yaml_block["features"][r.canonical] = entry

    rendered = _yaml.safe_dump(
        yaml_block, sort_keys=False, allow_unicode=True, indent=2,
    )
    header = (
        "# Suggested by `faultline suggest-config` — review and edit "
        "before scanning.\n"
        "# Each feature below was derived from a workspace package "
        "name or a CODEOWNERS team.\n"
        "# Empty `variants` is fine; the engine fills in matches "
        "automatically when you run `faultline analyze`.\n\n"
    )

    if write:
        target = Path(repo_path_resolved) / ".faultline.yaml"
        if target.exists():
            console.print(
                f"[yellow]{target.name} already exists.[/yellow] "
                "Suggestions printed to stdout instead — merge "
                "manually."
            )
            console.print()
            console.print(header + rendered, end="")
        else:
            target.write_text(header + rendered, encoding="utf-8")
            console.print(
                f"[green]✓[/green] Wrote {len(rules)} suggested "
                f"canonical(s) to {target}"
            )
    else:
        console.print(header + rendered, end="")


@app.command()
def version():
    """Shows the faultline version."""
    from faultline import __version__
    rprint(f"faultline [bold blue]v{__version__}[/bold blue]")


if __name__ == "__main__":
    app()
