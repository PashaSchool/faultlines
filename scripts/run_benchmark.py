"""CLI entry point for the benchmark harness.

Usage::

    python scripts/run_benchmark.py \\
        --repo formbricks \\
        --benchmarks ./benchmarks \\
        --scan /tmp/formbricks-tooluse-fmap.json \\
        --report report.md

Or score multiple repos at once and emit one combined report::

    python scripts/run_benchmark.py \\
        --repo formbricks --scan /tmp/formbricks-fmap.json \\
        --repo documenso --scan /tmp/documenso-fmap.json \\
        --report report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from faultline.benchmark.loader import load_benchmark
from faultline.benchmark.report import render_markdown, score


app = typer.Typer(add_completion=False)


def _load_detected(scan_path: Path) -> tuple[dict, dict]:
    """Read a faultline feature-map JSON and return (features, flows)."""
    raw = json.loads(scan_path.read_text(encoding="utf-8"))
    feats_in = raw.get("features", raw)
    features: dict[str, list[str]] = {}
    flows: dict[str, list[str]] = {}

    if isinstance(feats_in, list):
        for entry in feats_in:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name:
                continue
            paths = entry.get("paths") or entry.get("files") or []
            features[name] = list(paths)
            flow_list = entry.get("flows") or []
            flow_names: list[str] = []
            for fl in flow_list:
                if isinstance(fl, dict):
                    n = fl.get("name")
                    if n:
                        flow_names.append(n)
                elif isinstance(fl, str):
                    flow_names.append(fl)
            if flow_names:
                flows[name] = flow_names
    elif isinstance(feats_in, dict):
        for name, body in feats_in.items():
            if isinstance(body, dict):
                features[name] = list(body.get("paths") or body.get("files") or [])
            else:
                features[name] = list(body or [])

    return features, flows


@app.command()
def main(
    repo: list[str] = typer.Option(
        ...,
        "--repo",
        help="Repo name (matches benchmarks/<repo>/ directory). Repeatable.",
    ),
    scan: list[Path] = typer.Option(
        ...,
        "--scan",
        help="Path to scan output JSON. Pair index-aligned with --repo.",
    ),
    benchmarks: Path = typer.Option(
        Path("benchmarks"),
        "--benchmarks",
        help="Directory holding ground-truth YAML per repo.",
    ),
    report: Path | None = typer.Option(
        None,
        "--report",
        help="Write Markdown report here (default: stdout).",
    ),
    library: list[str] = typer.Option(
        [],
        "--library",
        help="Repo names to treat as libraries (skips flow recall).",
    ),
):
    if len(repo) != len(scan):
        typer.echo("--repo and --scan must have the same number of values.", err=True)
        raise typer.Exit(code=2)

    library_set = set(library)
    scores = []
    for r, s in zip(repo, scan):
        spec = load_benchmark(r, benchmarks)
        feats, flows = _load_detected(s)
        scores.append(score(
            spec, feats, flows,
            is_library=(r in library_set),
        ))

    text = render_markdown(scores)
    if report:
        report.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {report}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    app()
