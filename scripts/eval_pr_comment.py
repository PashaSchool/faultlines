"""Sprint 16 Day 3 — render PR comment markdown from eval-results.json.

The eval-llm.yml workflow (also Day 3) calls this and posts the
output as a sticky PR comment so reviewers see metrics + delta vs
baseline + first few misses without leaving GitHub.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _row(repo: str, data: dict) -> str:
    if data["status"] == "no-scan-available":
        return f"| {repo} | — | — | — | — | _no scan available_ |"
    feat = data["feature"]
    cov = feat["coverage"]
    prec = feat["precision"]
    f1 = feat["f1"]
    delta = feat.get("delta_coverage", 0.0)
    delta_emoji = (
        "✅" if delta > 0.005
        else "⚠️" if delta < -0.005
        else "—"
    )
    delta_str = f"{delta:+.1%}" if delta else "—"
    n_exp = feat.get("n_expected", "?")
    n_elig = feat.get("n_detected_eligible", "?")
    n_total = feat.get("n_detected_total", "?")
    sample = f"{n_exp}exp / {n_elig}elig / {n_total}total"
    return (
        f"| {repo} | {cov:.1%} | {prec:.1%} | {f1:.1%} | "
        f"`{sample}` | {delta_emoji} {delta_str} |"
    )


def _tier_row(repo: str, data: dict) -> str:
    if data["status"] != "scored":
        return ""
    t = data.get("tier_breakdown", {})
    return (
        f"| {repo} | {t.get('product', 0)} | "
        f"{t.get('supporting', 0)} | {t.get('hidden', 0)} |"
    )


def _flow_row(repo: str, data: dict) -> str:
    if data["status"] != "scored" or not data.get("flow"):
        return ""
    fl = data["flow"]
    if not fl.get("n_expected"):
        return ""
    return (
        f"| {repo} | {fl['coverage']:.1%} | {fl['precision']:.1%} | "
        f"{fl['f1']:.1%} | `{fl['n_expected']}exp / {fl['n_detected_eligible']}elig` |"
    )


def _miss_list(repo: str, data: dict, max_n: int = 5) -> str:
    if data["status"] != "scored":
        return ""
    misses = data.get("misses", []) or []
    if not misses:
        return ""
    lines = [f"\n**{repo}** missing:"]
    for m in misses[:max_n]:
        exp = m.get("expected") or "?"
        det = m.get("detected")
        mode = m.get("mode") or ""
        suffix = f"  _[{mode}]_" if mode and mode != "UNCATEGORISED" else ""
        if det:
            lines.append(f"  - `{exp}` → mapped to `{det}`{suffix}")
        else:
            lines.append(f"  - `{exp}` → not detected{suffix}")
    if len(misses) > max_n:
        lines.append(f"  - … {len(misses) - max_n} more")
    return "\n".join(lines)


def _failure_breakdown(repos: dict) -> str:
    """Aggregate failure mode counts across repos."""
    from collections import Counter
    total = Counter()
    for data in repos.values():
        for mode, count in (data.get("failure_modes") or {}).items():
            total[mode] += count
    if not total:
        return ""
    lines = ["\n**Failure modes** (across all repos):"]
    for mode, count in total.most_common():
        nice = mode.replace("_", " ").title()
        lines.append(f"  - `{count}×` {nice}")
    return "\n".join(lines)


def render(results: dict) -> str:
    summary = results.get("summary", {})
    repos = results.get("repos", {})

    parts: list[str] = []
    parts.append("### 📊 Faultlines Eval Results")
    parts.append("")

    if summary:
        parts.append(
            f"**Aggregate** (across {summary['n_repos']} repos): "
            f"coverage **{summary['avg_coverage']:.1%}** · "
            f"precision **{summary['avg_precision']:.1%}** · "
            f"F1 **{summary['avg_f1']:.1%}**"
        )
        parts.append("")

    parts.append("#### Features")
    parts.append("")
    parts.append("| repo | coverage | precision | F1 | sample size | Δ vs main |")
    parts.append("|---|---:|---:|---:|---|---:|")
    for repo in sorted(repos):
        parts.append(_row(repo, repos[repo]))

    flow_rows = [_flow_row(r, repos[r]) for r in sorted(repos)]
    flow_rows = [r for r in flow_rows if r]
    if flow_rows:
        parts.append("")
        parts.append("#### Flows")
        parts.append("")
        parts.append("| repo | coverage | precision | F1 | sample size |")
        parts.append("|---|---:|---:|---:|---|")
        parts.extend(flow_rows)

    tier_rows = [_tier_row(r, repos[r]) for r in sorted(repos)]
    tier_rows = [r for r in tier_rows if r]
    if tier_rows:
        parts.append("")
        parts.append("#### Detected feature tiers")
        parts.append("_Hidden tier (synthetic / tooling / docs / deployment) excluded from precision._")
        parts.append("")
        parts.append("| repo | product | supporting | hidden |")
        parts.append("|---|---:|---:|---:|")
        parts.extend(tier_rows)

    breakdown = _failure_breakdown(repos)
    if breakdown:
        parts.append(breakdown)

    miss_blocks = [
        _miss_list(repo, repos[repo]) for repo in sorted(repos)
    ]
    miss_blocks = [b for b in miss_blocks if b]
    if miss_blocks:
        parts.append("\n<details><summary>Missing features (first 5 per repo, with mode tags)</summary>\n")
        parts.extend(miss_blocks)
        parts.append("\n</details>")

    parts.append("")
    parts.append(
        "_Run via `eval-llm.yml`. Ground truth: "
        "[`tests/eval/ground_truth.json`](../tree/main/tests/eval/ground_truth.json)._"
    )
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="path to eval-results.json")
    parser.add_argument("--out", default="-", help="output file or '-' for stdout")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    md = render(data)

    if args.out == "-":
        print(md)
    else:
        Path(args.out).write_text(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
