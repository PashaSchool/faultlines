import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich import print as rprint

from faultline.analyzer.git import load_repo, get_commits, get_tracked_files, estimate_commits, estimate_duration, get_remote_url, DEFAULT_MAX_COMMITS
from faultline.analyzer.features import detect_features_from_structure, build_feature_map, build_flows_metrics
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
        help="Use an LLM to semantically detect features from the file tree",
        is_flag=True,
    ),
    provider: str = typer.Option(
        "anthropic",
        "--provider",
        help="LLM provider: anthropic or ollama",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model name override (default: claude-haiku-4-5 for anthropic, qwen2.5-coder:7b for ollama)",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
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
    flows: bool = typer.Option(
        False,
        "--flows",
        help="Detect user-facing flows within features (requires --llm)",
        is_flag=True,
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
    """
    repo_path = str(Path(repo_path).resolve())

    # --flows requires --llm
    if flows and not llm:
        llm = True

    if llm and provider not in ("anthropic", "ollama"):
        console.print(f"[red]Unknown provider '{provider}'. Use: anthropic or ollama[/red]")
        raise typer.Exit(1)

    try:
        # 1. Load the repository
        console.print(f"[blue]Analyzing:[/blue] {repo_path}")
        repo = load_repo(repo_path)
        remote_url = get_remote_url(repo)

        # 2. Validate LLM access early — before the long git analysis
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

        # Extract AST signatures when --llm is set (reused for feature anchors + flows)
        signatures: dict = {}
        if llm:
            from faultline.analyzer.ast_extractor import extract_signatures
            extract_root = str(Path(str(repo.working_tree_dir)) / path_prefix) if path_prefix else str(repo.working_tree_dir)
            signatures = extract_signatures(analysis_files, extract_root)
            if signatures:
                console.print(f"[dim]Extracted signatures from {len(signatures)} TS/JS files[/dim]")

        if llm:
            raw_mapping = _detect_with_llm(analysis_files, provider, api_key, model, ollama_url, commits=commits, path_prefix=path_prefix, signatures=signatures)
        else:
            raw_mapping = detect_features_from_structure(analysis_files)

        # Restore full paths so commit matching works against git history
        if path_prefix:
            feature_paths = {
                name: [path_prefix + f for f in paths]
                for name, paths in raw_mapping.items()
            }
        else:
            feature_paths = raw_mapping

        console.print(f"[green]✓[/green] Detected {len(feature_paths)} features")

        # 6. Build the feature map
        feature_map = build_feature_map(
            repo_path=repo_path,
            commits=commits,
            feature_paths=feature_paths,
            days=days,
            remote_url=remote_url,
        )

        # 6b. Detect flows within each feature (optional)
        if flows:
            from faultline.analyzer.coverage import read_coverage
            coverage_data = read_coverage(str(repo.working_tree_dir))
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
            )

        # 7. Print the report
        print_report(feature_map)

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


def _detect_with_llm(
    files: list[str],
    provider: str,
    api_key: str | None,
    model: str | None,
    ollama_url: str,
    commits: list | None = None,
    path_prefix: str = "",
    signatures: dict | None = None,
) -> dict[str, list[str]]:
    """Runs LLM feature detection with the chosen provider. Falls back to heuristic on failure."""
    if provider == "anthropic":
        from faultline.llm.detector import detect_features_llm
        console.print("[blue]Mapping features with Claude...[/blue]")
        feature_paths = detect_features_llm(files, api_key=api_key, commits=commits, path_prefix=path_prefix, signatures=signatures)

    elif provider == "ollama":
        from faultline.llm.detector import detect_features_ollama, _DEFAULT_OLLAMA_MODEL
        resolved_model = model or _DEFAULT_OLLAMA_MODEL
        console.print(f"[blue]Mapping features with Ollama ({resolved_model})...[/blue]")
        feature_paths = detect_features_ollama(files, model=resolved_model, host=ollama_url, commits=commits, path_prefix=path_prefix, signatures=signatures)

    else:
        feature_paths = {}

    if feature_paths:
        label = "Claude" if provider == "anthropic" else "Ollama"
        console.print(f"[green]✓[/green] {label} mapped {len(feature_paths)} features")
    else:
        console.print("[yellow]⚠ LLM detection failed — using heuristic fallback[/yellow]")
        feature_paths = detect_features_from_structure(files)

    return feature_paths


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
):
    """
    Runs flow detection for each feature and attaches Flow objects to the FeatureMap.
    Returns the updated FeatureMap (features with .flows populated).
    """
    from faultline.llm.flow_detector import detect_flows_llm, detect_flows_ollama, _DEFAULT_OLLAMA_MODEL as _OLLAMA_MODEL
    from faultline.llm.flow_detector import _FlowFileMapping

    label = "Claude" if provider == "anthropic" else "Ollama"
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

        # Detect flows for this feature
        if provider == "anthropic":
            flow_mappings = detect_flows_llm(
                feature_name=feature.name,
                feature_files=analysis_feature_files,
                signatures=signatures,
                api_key=api_key,
            )
        else:
            resolved_model = model or _OLLAMA_MODEL
            flow_mappings = detect_flows_ollama(
                feature_name=feature.name,
                feature_files=analysis_feature_files,
                signatures=signatures,
                model=resolved_model,
                host=ollama_url,
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
        feature_commit_files = set(feature.paths)
        feature_commits = [
            c for c in commits
            if any(f in feature_commit_files for f in c.files_changed)
        ]
        flows = build_flows_metrics(feature_commits, flow_file_mappings, remote_url=remote_url, coverage_data=coverage_data)
        total_flows += len(flows)

        updated_features.append(feature.model_copy(update={"flows": flows}))

    console.print(f"[green]✓[/green] Detected {total_flows} flows across {len(updated_features)} features")
    return feature_map.model_copy(update={"features": updated_features})


@app.command()
def version():
    """Shows the faultline version."""
    from faultline import __version__
    rprint(f"faultline [bold blue]v{__version__}[/bold blue]")


if __name__ == "__main__":
    app()
