"""PR and change risk analysis.

Given a list of changed files and a feature map, produces a risk report:
which features are affected, how healthy they are, which files are
historically co-changed but missing from the changeset, and what the
regression probability is.

No LLM calls. Pure git history + feature map lookup.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

# Co-change analysis constants
_MAX_FILES_PER_COMMIT = 30
_MIN_FILE_COMMITS = 2
_COCHANGE_THRESHOLD = 0.15
_MAX_COCHANGE_RESULTS = 10


def predict_impact(
    changed_files: list[str],
    feature_map: dict[str, Any],
    repo_path: str = ".",
    days: int = 365,
) -> dict[str, Any]:
    """Predict the impact of a set of file changes.

    Args:
        changed_files: Files being changed (relative to repo root).
        feature_map: Loaded feature-map JSON dict.
        repo_path: Path to git repo for co-change history.
        days: Lookback window for co-change analysis.

    Returns:
        Impact report with affected features, missing co-changes,
        risk level, and regression probability.
    """
    features = feature_map.get("features", [])

    # 1. Find affected features
    affected = _find_affected_features(changed_files, features)

    # 2. Find co-change partners that are missing from the changeset
    cochange_pairs = _compute_cochange_pairs(repo_path, days)
    missing = _find_missing_cochanges(changed_files, cochange_pairs)

    # 3. Calculate risk signals
    risk_signals = _build_risk_signals(affected, missing, changed_files)

    # 4. Overall risk level
    risk_level = _calculate_risk_level(risk_signals)

    # 5. Regression probability from historical patterns
    regression_prob = _estimate_regression_probability(
        changed_files, features, repo_path, days,
    )

    return {
        "changed_files": changed_files,
        "affected_features": [
            {
                "name": f["name"],
                "description": f.get("description"),
                "health": round(f.get("health_score", 0)),
                "bug_fix_ratio": round(f.get("bug_fix_ratio", 0) * 100, 1),
                "coverage_pct": f.get("coverage_pct"),
                "bus_factor": _get_bus_factor(f),
                "owners": f.get("authors", [])[:3],
                "affected_flows": [
                    fl["name"] for fl in f.get("flows", [])
                    if any(cf in fl.get("paths", []) for cf in changed_files)
                ],
            }
            for f in affected
        ],
        "missing_cochanges": missing,
        "risk_level": risk_level,
        "risk_signals": risk_signals,
        "regression_probability": regression_prob,
    }


def _find_affected_features(
    changed_files: list[str],
    features: list[dict],
) -> list[dict]:
    """Find features that contain any of the changed files."""
    changed_set = set(changed_files)
    affected = []
    for f in features:
        paths = set(f.get("paths", []))
        if paths & changed_set:
            affected.append(f)
    return affected


def _compute_cochange_pairs(
    repo_path: str,
    days: int,
) -> list[tuple[str, str, float]]:
    """Compute co-change pairs from git log via subprocess.

    Fully self-contained — no imports from analyzer. Reads git log
    directly to avoid coupling with the main analysis pipeline.
    """
    try:
        result = subprocess.run(
            [
                "git", "log",
                f"--since={days} days ago",
                "--name-only",
                "--format=COMMIT_SEP",
            ],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=30,
        )
        if result.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    # Parse commits
    file_commits: dict[str, set[int]] = defaultdict(set)
    commit_idx = 0
    current_files: list[str] = []

    for line in result.stdout.splitlines():
        if line == "COMMIT_SEP":
            if current_files and len(current_files) <= _MAX_FILES_PER_COMMIT:
                for f in current_files:
                    file_commits[f].add(commit_idx)
            current_files = []
            commit_idx += 1
        elif line.strip():
            current_files.append(line.strip())

    # Last commit
    if current_files and len(current_files) <= _MAX_FILES_PER_COMMIT:
        for f in current_files:
            file_commits[f].add(commit_idx)

    # Filter low-activity files
    active_files = {
        f: shas for f, shas in file_commits.items()
        if len(shas) >= _MIN_FILE_COMMITS
    }

    # Count co-occurrences
    commit_to_files: dict[int, list[str]] = defaultdict(list)
    for f, commits in active_files.items():
        for c in commits:
            commit_to_files[c].append(f)

    pair_both: dict[tuple[str, str], int] = defaultdict(int)
    for files in commit_to_files.values():
        if len(files) < 2:
            continue
        for f1, f2 in combinations(sorted(files), 2):
            pair_both[(f1, f2)] += 1

    # Jaccard scores
    pairs: list[tuple[str, str, float]] = []
    for (f1, f2), both in pair_both.items():
        a = len(active_files.get(f1, set()))
        b = len(active_files.get(f2, set()))
        denom = a + b - both
        if denom > 0:
            score = both / denom
            if score >= _COCHANGE_THRESHOLD:
                pairs.append((f1, f2, round(score, 2)))

    return sorted(pairs, key=lambda x: x[2], reverse=True)


def _find_missing_cochanges(
    changed_files: list[str],
    cochange_pairs: list[tuple[str, str, float]],
) -> list[dict[str, Any]]:
    """Find files that historically co-change with the changed files
    but are NOT in the changeset.
    """
    changed_set = set(changed_files)
    missing: dict[str, float] = {}

    for f1, f2, score in cochange_pairs:
        if f1 in changed_set and f2 not in changed_set:
            if f2 not in missing or score > missing[f2]:
                missing[f2] = score
        elif f2 in changed_set and f1 not in changed_set:
            if f1 not in missing or score > missing[f1]:
                missing[f1] = score

    return [
        {"file": f, "cochange_score": s, "confidence": _confidence_label(s)}
        for f, s in sorted(missing.items(), key=lambda x: x[1], reverse=True)
    ][:_MAX_COCHANGE_RESULTS]


def _confidence_label(score: float) -> str:
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _get_bus_factor(feature: dict) -> int:
    """Get minimum bus factor across feature flows, or author count."""
    flow_factors = [
        fl.get("bus_factor", 99) for fl in feature.get("flows", [])
    ]
    if flow_factors:
        return min(flow_factors)
    authors = feature.get("authors", [])
    return min(len(authors), 5) if authors else 1


def _build_risk_signals(
    affected: list[dict],
    missing: list[dict],
    changed_files: list[str],
) -> list[str]:
    """Build human-readable risk signals."""
    signals: list[str] = []

    for f in affected:
        health = f.get("health_score", 100)
        if health < 30:
            signals.append(
                f"Feature '{f['name']}' has critical health ({round(health)}). "
                f"Most recent commits are bug fixes."
            )
        elif health < 50:
            signals.append(
                f"Feature '{f['name']}' is at risk (health {round(health)})."
            )

        cov = f.get("coverage_pct")
        if cov is not None and cov < 30:
            signals.append(
                f"Feature '{f['name']}' has only {round(cov)}% test coverage."
            )

        bus = _get_bus_factor(f)
        if bus == 1:
            owners = f.get("authors", [])[:2]
            signals.append(
                f"Feature '{f['name']}' has bus factor 1 "
                f"(only maintained by {', '.join(owners)})."
            )

    high_confidence_missing = [m for m in missing if m["confidence"] == "high"]
    if high_confidence_missing:
        files = [m["file"] for m in high_confidence_missing[:3]]
        signals.append(
            f"Files that historically co-change are missing from this changeset: "
            f"{', '.join(files)}."
        )

    return signals


def _calculate_risk_level(signals: list[str]) -> str:
    """Overall risk level based on signal count and severity."""
    critical_keywords = ["critical health", "bus factor 1", "co-change are missing"]
    critical_count = sum(
        1 for s in signals
        if any(k in s.lower() for k in critical_keywords)
    )
    if critical_count >= 2:
        return "critical"
    if critical_count >= 1 or len(signals) >= 3:
        return "high"
    if len(signals) >= 1:
        return "medium"
    return "low"


def _estimate_regression_probability(
    changed_files: list[str],
    features: list[dict],
    repo_path: str,
    days: int,
) -> float:
    """Estimate probability that this change will need a follow-up bug fix.

    Based on the bug-fix ratio of affected features weighted by recency.
    Not a prediction — a historical baseline: "X% of changes to this code
    resulted in bug fixes."
    """
    affected = _find_affected_features(changed_files, features)
    if not affected:
        return 0.0

    # Weighted average of bug-fix ratios, weighted by number of changed
    # files in each feature (more touched files = more influence)
    changed_set = set(changed_files)
    total_weight = 0.0
    weighted_ratio = 0.0

    for f in affected:
        overlap = len(changed_set & set(f.get("paths", [])))
        ratio = f.get("bug_fix_ratio", 0.0)
        weighted_ratio += ratio * overlap
        total_weight += overlap

    if total_weight == 0:
        return 0.0

    return round(weighted_ratio / total_weight, 2)
