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
    flows: bool = typer.Option(
        False,
        "--flows",
        help="Detect user-facing flows within features (requires --llm)",
        is_flag=True,
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
            from faultline.llm.pipeline import run as _run_new_pipeline
            console.print(
                "[blue]Running new pipeline[/blue] "
                "(pass [dim]--legacy[/dim] to use the 5-strategy fallback)"
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
                )
            except Exception as exc:  # pragma: no cover - surfacing guidance
                console.print(
                    f"[yellow]New pipeline raised {type(exc).__name__}: {exc} — "
                    f"falling through to legacy[/yellow]"
                )
                _new_pipeline_result = None

            if _new_pipeline_result is None:
                console.print(
                    "[yellow]New pipeline returned no features — "
                    "falling through to legacy[/yellow]"
                )
            else:
                raw_mapping = dict(_new_pipeline_result.features)
                console.print(
                    f"[green]✓[/green] New pipeline: {len(raw_mapping)} features"
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
                            ollama_url=ollama_url, path_prefix="",
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
        feature_map = build_feature_map(
            repo_path=repo_path,
            commits=commits,
            feature_paths=feature_paths,
            days=days,
            remote_url=remote_url,
            shared_attributions=shared_attributions,
            skip_small_feature_merge=_skip_merge,
        )

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

        # 6b. Read coverage data (if available)
        from faultline.analyzer.coverage import read_coverage
        coverage_data = read_coverage(str(repo.working_tree_dir), coverage_path=coverage)
        if coverage_data:
            console.print(f"[dim]Coverage data: {len(coverage_data)} files[/dim]")
            _apply_feature_coverage(feature_map, coverage_data, path_prefix)

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
        if flows and (not repo_structure.is_library or _force_flows):
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

        # 7. Print the report
        print_report(feature_map, impact_scores=impact_scores)

        # 8. Save to disk
        if save:
            saved_path = write_feature_map(feature_map, output)
            console.print(f"[dim]Saved: {saved_path}[/dim]")

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
    for feat in feature_map.features:
        if feat.description:
            continue
        for desc_name, desc in descriptions.items():
            if (
                feat.name == desc_name
                or feat.name in desc_name
                or desc_name in feat.name
                or feat.name.rstrip("s") == desc_name.rstrip("s")
            ):
                feat.description = desc
                break


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
) -> None:
    """Computes average line coverage per feature from coverage report data.

    Mutates feature_map.features in place, setting coverage_pct on each feature
    that has matching files in the coverage report.
    """
    from faultline.analyzer.features import _is_test_file

    for feature in feature_map.features:
        coverages = []
        for file_path in feature.paths:
            if _is_test_file(file_path):
                continue
            # Try matching with and without path_prefix
            full_path = f"{path_prefix}{file_path}" if path_prefix else file_path
            pct = _match_coverage(coverage_data, file_path, full_path)
            if pct is not None:
                coverages.append(pct)
        if coverages:
            feature.coverage_pct = round(sum(coverages) / len(coverages), 1)

        # Apply coverage to flows within this feature
        for flow in feature.flows:
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
def version():
    """Shows the faultline version."""
    from faultline import __version__
    rprint(f"faultline [bold blue]v{__version__}[/bold blue]")


if __name__ == "__main__":
    app()
