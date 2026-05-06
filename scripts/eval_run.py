"""Sprint 16 Day 3 — eval CLI for the dev → main CI gate.

Loads ground_truth.json, picks up the latest feature-map JSON for
each repo, runs the LLM-as-judge, writes one results file the PR-
comment formatter consumes.

Usage:
    python scripts/eval_run.py
    python scripts/eval_run.py --repos axios immich
    python scripts/eval_run.py --json-out .faultline/eval-results.json

Default behaviour: scan ``benchmarks/<repo>/feature-map-final.json``
or ``feature-map-*.json`` (latest by mtime). When neither exists,
the repo is reported as ``no-scan-available`` and counted as 0/0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tests.eval.failure_modes import classify_all, summarise
from tests.eval.judge import judge_run

ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH = ROOT / "tests" / "eval" / "ground_truth.json"
DEFAULT_OUT = ROOT / ".faultline" / "eval-results.json"


def _latest_feature_map(repo: str) -> Path | None:
    """Pick the most relevant feature-map JSON for ``repo``.

    Priority:
        1. benchmarks/<repo>/feature-map-final.json
        2. benchmarks/<repo>/sprint*-live.json (latest)
        3. ~/.faultline/feature-map-<repo>-*.json (latest)
    """
    bench = ROOT / "benchmarks" / repo
    final = bench / "feature-map-final.json"
    if final.exists():
        return final
    lives = sorted(
        bench.glob("sprint*-live.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if lives:
        return lives[0]
    home = Path.home() / ".faultline"
    user_scans = sorted(
        home.glob(f"feature-map-{repo}-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if user_scans:
        return user_scans[0]
    return None


def _detected_features(fm_path: Path) -> tuple[list[str], list[str]]:
    """Extract feature names + flow names from a feature-map JSON."""
    fm = json.loads(fm_path.read_text())
    feature_names: list[str] = []
    flow_names: list[str] = []
    for f in fm.get("features", []):
        feature_names.append(f["name"])
        for fl in f.get("flows", []) or []:
            flow_names.append(fl["name"])
    return feature_names, flow_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="*", default=None,
                        help="Subset of repos to evaluate; default = all in ground_truth.json")
    parser.add_argument("--json-out", default=str(DEFAULT_OUT))
    parser.add_argument("--baseline", default=None,
                        help="Path to a previous eval-results.json for delta comparison")
    args = parser.parse_args()

    truth = json.loads(GROUND_TRUTH.read_text())
    truth_repos = [k for k in truth if not k.startswith("_")]
    repos = args.repos or truth_repos
    unknown = [r for r in repos if r not in truth_repos]
    if unknown:
        print(f"unknown repos in ground_truth: {unknown}", file=sys.stderr)
        return 2

    baseline: dict = {}
    if args.baseline:
        try:
            baseline = json.loads(Path(args.baseline).read_text())
        except (OSError, json.JSONDecodeError):
            baseline = {}

    out: dict = {"repos": {}, "summary": {}}
    for repo in repos:
        truth_entry = truth[repo]
        expected_features = truth_entry.get("expected_features", [])
        expected_flows = truth_entry.get("expected_flows", [])

        fm_path = _latest_feature_map(repo)
        if fm_path is None:
            out["repos"][repo] = {
                "status": "no-scan-available",
                "feature": {"coverage": 0.0, "precision": 0.0, "f1": 0.0},
                "flow": {"coverage": 0.0, "precision": 0.0, "f1": 0.0},
            }
            continue

        detected_features, detected_flows = _detected_features(fm_path)

        feat_result = judge_run(
            expected=expected_features, detected=detected_features, repo=repo,
        )
        flow_result = judge_run(
            expected=expected_flows, detected=detected_flows, repo=f"{repo}-flows",
        ) if expected_flows else None

        baseline_repo = baseline.get("repos", {}).get(repo, {})
        # Sprint 16 Day 4 — classify each miss into a failure mode so
        # the PR comment shows a breakdown, not just a percentage.
        miss_pairs = [
            (m.expected, m.detected)
            for m in feat_result.matches if not m.is_hit
        ]
        sample_paths: list[str] = []
        try:
            fm_data = json.loads(fm_path.read_text())
            for f in fm_data.get("features", [])[:3]:
                sample_paths.extend(f.get("paths", [])[:5])
        except (OSError, json.JSONDecodeError):
            pass
        classified = classify_all(
            miss_pairs,
            detected_features=detected_features,
            sample_paths=sample_paths,
        ) if miss_pairs else []

        out["repos"][repo] = {
            "status": "scored",
            "scan_path": str(fm_path.relative_to(ROOT) if str(fm_path).startswith(str(ROOT)) else fm_path),
            "feature": {
                "coverage": round(feat_result.coverage, 3),
                "precision": round(feat_result.precision, 3),
                "f1": round(feat_result.f1, 3),
                "delta_coverage": round(
                    feat_result.coverage - baseline_repo.get("feature", {}).get("coverage", feat_result.coverage),
                    3,
                ),
            },
            "flow": {
                "coverage": round(flow_result.coverage, 3) if flow_result else None,
                "precision": round(flow_result.precision, 3) if flow_result else None,
                "f1": round(flow_result.f1, 3) if flow_result else None,
            } if flow_result else None,
            "misses": [
                {
                    "expected": cm.expected,
                    "detected": cm.detected,
                    "mode": cm.mode.name,
                    "reasoning": cm.reasoning,
                }
                for cm in classified
            ],
            "failure_modes": {
                mode.name: count
                for mode, count in summarise(classified).items()
            } if classified else {},
            "extras": feat_result.extras,
        }

    # Aggregate.
    scored = [
        r for r in out["repos"].values()
        if r.get("status") == "scored"
    ]
    if scored:
        out["summary"] = {
            "n_repos": len(scored),
            "avg_coverage": round(
                sum(r["feature"]["coverage"] for r in scored) / len(scored),
                3,
            ),
            "avg_precision": round(
                sum(r["feature"]["precision"] for r in scored) / len(scored),
                3,
            ),
            "avg_f1": round(
                sum(r["feature"]["f1"] for r in scored) / len(scored),
                3,
            ),
        }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))

    # Console summary.
    print(f"\nEval results → {out_path}")
    if "avg_coverage" in out["summary"]:
        s = out["summary"]
        print(f"  avg coverage:  {s['avg_coverage']:.1%}")
        print(f"  avg precision: {s['avg_precision']:.1%}")
        print(f"  avg F1:        {s['avg_f1']:.1%}")
    for repo, data in sorted(out["repos"].items()):
        if data["status"] == "no-scan-available":
            print(f"  {repo:12s}  no scan available — skipped")
            continue
        feat = data["feature"]
        print(
            f"  {repo:12s}  cov={feat['coverage']:.1%}  "
            f"prec={feat['precision']:.1%}  f1={feat['f1']:.1%}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
