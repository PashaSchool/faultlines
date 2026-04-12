"""MCP tools for impact analysis.

Registers additional tools on the shared MCP server instance.
Import this module from mcp_server.py to activate impact tools.
"""

from __future__ import annotations

from typing import Any

from faultline.mcp_server import mcp, _load_map, _inject_warning, _savings_metadata
from faultline.impact.risk import predict_impact


@mcp.tool()
def analyze_change_impact(changed_files: list[str], repo_path: str = ".") -> dict[str, Any]:
    """Predict the impact of changing specific files.

    Use this BEFORE submitting a PR or making a refactor. Returns:
    - Which features and flows are affected
    - Files that historically co-change but are missing from your changes
    - Risk level (critical/high/medium/low)
    - Regression probability based on historical bug-fix patterns
    - Coverage gaps and bus factor warnings

    This uses git history patterns (behavioral), not just imports
    (structural) — catches runtime, API, and cross-boundary dependencies
    that static analysis misses.

    Args:
        changed_files: List of files being changed (relative to repo root).
            Example: ["src/payments/stripe.ts", "src/payments/checkout.ts"]
        repo_path: Path to the git repository root. Defaults to current dir.
    """
    fm = _load_map()
    result = predict_impact(
        changed_files=changed_files,
        feature_map=fm,
        repo_path=repo_path,
    )
    result["_savings_metadata"] = _savings_metadata(len(changed_files))
    return _inject_warning(result, fm)


@mcp.tool()
def get_regression_risk(changed_files: list[str]) -> dict[str, Any]:
    """Quick check: how likely is this change to cause a regression?

    Returns a probability (0.0-1.0) based on how buggy the affected
    features have been historically. Use this for a fast go/no-go
    signal before merging.

    Args:
        changed_files: Files being changed (relative to repo root).
    """
    fm = _load_map()
    features = fm.get("features", [])
    changed_set = set(changed_files)

    affected = []
    for f in features:
        paths = set(f.get("paths", []))
        if paths & changed_set:
            affected.append(f)

    if not affected:
        return _inject_warning({
            "regression_probability": 0.0,
            "risk_level": "low",
            "reason": "Changed files don't belong to any tracked feature.",
            "_savings_metadata": _savings_metadata(0),
        }, fm)

    total_weight = 0.0
    weighted_ratio = 0.0
    for f in affected:
        overlap = len(changed_set & set(f.get("paths", [])))
        weighted_ratio += f.get("bug_fix_ratio", 0.0) * overlap
        total_weight += overlap

    prob = round(weighted_ratio / total_weight, 2) if total_weight else 0.0

    if prob >= 0.6:
        risk = "critical"
    elif prob >= 0.4:
        risk = "high"
    elif prob >= 0.2:
        risk = "medium"
    else:
        risk = "low"

    return _inject_warning({
        "regression_probability": prob,
        "risk_level": risk,
        "affected_features": [
            {"name": f["name"], "health": round(f.get("health_score", 0))}
            for f in affected
        ],
        "reason": (
            f"{round(prob * 100)}% of historical changes to these features "
            f"resulted in bug fixes."
        ),
        "_savings_metadata": _savings_metadata(len(affected)),
    }, fm)
