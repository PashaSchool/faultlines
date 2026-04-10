from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from faultline.models.types import FeatureMap, Feature

console = Console()


def _health_color(score: float) -> str:
    if score >= 75:
        return "green"
    elif score >= 50:
        return "yellow"
    else:
        return "red"


def _health_icon(score: float) -> str:
    if score >= 75:
        return "✓"
    elif score >= 50:
        return "!"
    else:
        return "✗"


def print_summary(feature_map: FeatureMap) -> None:
    """Prints the analysis summary panel."""
    total_bugs = sum(f.bug_fixes for f in feature_map.features)
    avg_health = sum(f.health_score for f in feature_map.features) / len(feature_map.features) \
        if feature_map.features else 0

    console.print()
    console.print(Panel(
        f"[bold]Repository:[/bold] {feature_map.repo_path}\n"
        f"[bold]Analyzed:[/bold] last {feature_map.date_range_days} days\n"
        f"[bold]Total commits:[/bold] {feature_map.total_commits}\n"
        f"[bold]Features found:[/bold] {len(feature_map.features)}\n"
        f"[bold]Bug fix commits:[/bold] {total_bugs}\n"
        f"[bold]Average health score:[/bold] [{_health_color(avg_health)}]{avg_health:.1f}/100[/]",
        title="[bold blue]FeatureMap Analysis[/bold blue]",
        border_style="blue",
    ))


def print_features_table(feature_map: FeatureMap) -> None:
    """Prints a risk-sorted table of all features, then flows if present."""
    _print_features_flat(feature_map)

    if any(f.flows for f in feature_map.features):
        _print_features_with_flows(feature_map)


def _coverage_color(pct: float) -> str:
    if pct >= 70:
        return "green"
    elif pct >= 40:
        return "yellow"
    else:
        return "red"


def _print_features_flat(feature_map: FeatureMap) -> None:
    """Standard flat table without flows."""
    has_coverage = any(f.coverage_pct is not None for f in feature_map.features)

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="Features by Risk",
        title_style="bold",
    )

    table.add_column("", width=3)
    table.add_column("Feature", style="bold", min_width=15)
    table.add_column("Health", justify="center", width=10)
    table.add_column("Commits", justify="right", width=10)
    table.add_column("Bug Fixes", justify="right", width=10)
    table.add_column("Bug %", justify="right", width=8)
    table.add_column("Authors", justify="right", width=10)
    table.add_column("Files", justify="right", width=8)
    if has_coverage:
        table.add_column("Cov %", justify="right", width=8)

    for feature in feature_map.sorted_by_risk():
        display_health = feature.symbol_health_score if feature.symbol_health_score is not None else feature.health_score
        color = _health_color(display_health)
        icon = _health_icon(display_health)
        bug_pct = f"{feature.bug_fix_ratio * 100:.1f}%"

        row = [
            f"[{color}]{icon}[/]",
            feature.name,
            f"[{color}]{display_health:.0f}[/]",
            str(feature.total_commits),
            f"[{color}]{feature.bug_fixes}[/]" if feature.bug_fixes > 0 else "0",
            f"[{color}]{bug_pct}[/]",
            str(len(feature.authors)),
            str(len(feature.paths)),
        ]
        if has_coverage:
            if feature.coverage_pct is not None:
                cov_color = _coverage_color(feature.coverage_pct)
                row.append(f"[{cov_color}]{feature.coverage_pct:.0f}%[/]")
            else:
                row.append("[dim]—[/dim]")
        table.add_row(*row)

    console.print()
    console.print(table)


def _print_features_with_flows(feature_map: FeatureMap) -> None:
    """Two-level table: features as section headers, flows as nested rows."""
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="Feature & Flow Map by Risk",
        title_style="bold",
    )

    table.add_column("", width=3)
    table.add_column("Feature / Flow", min_width=22)
    table.add_column("Health", justify="center", width=10)
    table.add_column("Commits", justify="right", width=10)
    table.add_column("Bug Fixes", justify="right", width=10)
    table.add_column("Bug %", justify="right", width=8)

    for feature in feature_map.sorted_by_risk():
        f_display_health = feature.symbol_health_score if feature.symbol_health_score is not None else feature.health_score
        f_color = _health_color(f_display_health)
        f_icon = _health_icon(f_display_health)
        f_pct = f"{feature.bug_fix_ratio * 100:.1f}%"

        # Feature header row
        table.add_row(
            f"[{f_color}]{f_icon}[/]",
            f"[bold]{feature.name}[/bold]",
            f"[{f_color} bold]{f_display_health:.0f}[/]",
            f"[bold]{feature.total_commits}[/bold]",
            f"[{f_color} bold]{feature.bug_fixes}[/]" if feature.bug_fixes > 0 else "[bold]0[/bold]",
            f"[{f_color} bold]{f_pct}[/]",
        )

        # Nested flow rows — sorted by bug_fix_ratio descending
        sorted_flows = sorted(feature.flows, key=lambda fl: fl.bug_fix_ratio, reverse=True)
        for i, flow in enumerate(sorted_flows):
            fl_color = _health_color(flow.health_score)
            fl_pct = f"{flow.bug_fix_ratio * 100:.1f}%"
            connector = "└─" if i == len(sorted_flows) - 1 else "├─"

            table.add_row(
                "",
                f"  [dim]{connector}[/dim] [{fl_color}]{flow.name}[/{fl_color}]",
                f"[{fl_color}]{flow.health_score:.0f}[/]",
                str(flow.total_commits),
                f"[{fl_color}]{flow.bug_fixes}[/]" if flow.bug_fixes > 0 else "0",
                f"[{fl_color}]{fl_pct}[/]",
            )

    console.print()
    console.print(table)


def print_top_risks(feature_map: FeatureMap, top: int = 3) -> None:
    """Prints the top highest-risk features with details."""
    risky = [f for f in feature_map.sorted_by_risk() if f.bug_fixes > 0][:top]

    if not risky:
        console.print("\n[green]No critical risk zones detected![/green]")
        return

    console.print(f"\n[bold red]Top {top} risk zones:[/bold red]")

    for i, feature in enumerate(risky, 1):
        color = _health_color(feature.health_score)
        console.print(
            f"  {i}. [bold {color}]{feature.name}[/bold {color}] — "
            f"{feature.bug_fixes} bug fixes out of {feature.total_commits} commits "
            f"({feature.bug_fix_ratio * 100:.1f}%)"
        )
        if feature.description:
            console.print(f"     [dim]{feature.description}[/dim]")


def _impact_color(level: str) -> str:
    if level == "critical":
        return "red bold"
    elif level == "high":
        return "red"
    elif level == "medium":
        return "yellow"
    elif level == "low":
        return "blue"
    return "green"


def print_impact_scores(scores: list) -> None:
    """Prints the impact scores table from analytics data."""
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title="Impact Scores (Analytics)",
        title_style="bold",
    )

    table.add_column("Flow", min_width=20)
    table.add_column("Health", justify="center", width=10)
    table.add_column("Pageviews", justify="right", width=12)
    table.add_column("Errors", justify="right", width=10)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Impact", justify="center", width=12)

    for s in scores:
        h_color = _health_color(s.health_score)
        i_color = _impact_color(s.impact_level)
        err_color = "red" if s.error_count > 100 else "dim"

        table.add_row(
            s.flow_name,
            f"[{h_color}]{s.health_score:.0f}[/]",
            f"{s.pageviews:,}" if s.pageviews > 0 else "[dim]—[/]",
            f"[{err_color}]{s.error_count:,}[/]" if s.error_count > 0 else "[dim]—[/]",
            f"{s.score:.0f}",
            f"[{i_color}]{s.impact_level.upper()}[/]",
        )

    console.print()
    console.print(table)


def print_report(feature_map: FeatureMap, impact_scores: list | None = None) -> None:
    """Prints the full terminal report."""
    print_summary(feature_map)
    print_features_table(feature_map)
    if impact_scores:
        print_impact_scores(impact_scores)
    print_top_risks(feature_map)
    console.print()
