"""Faultlines MCP server.

Exposes the latest feature-map JSON as tools that AI coding agents
(Cursor, Claude Code, Cline, Aider) can call to get precise codebase
context instead of grepping and reading random files.

Tools:
    list_features       -- overview of all features with health scores
    find_feature        -- semantic search by name or description
    get_feature_files   -- exact file list for a feature
    get_hotspots        -- riskiest features (lowest health)
    get_feature_owners  -- top contributors for a feature
    get_flow_files      -- files belonging to a user-facing flow
    get_repo_summary    -- high-level repo stats

Run:
    faultlines-mcp                  # uses default map location
    FAULTLINE_MAP_PATH=... faultlines-mcp

Install for Cursor (``~/.cursor/mcp.json``)::

    {
      "mcpServers": {
        "faultlines": {
          "command": "faultlines-mcp"
        }
      }
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("faultlines")


# Per-request token-savings metadata. AI agents usually ignore this,
# but the SaaS dashboard aggregates it across calls to show savings.
_AVG_GREP_FILES_PER_QUERY = 15          # files an agent would read without MCP
_AVG_TOKENS_PER_FILE = 2500             # ~10KB at 4 chars/token
_AVG_MCP_RESPONSE_TOKENS = 500          # typical MCP response size
_STALE_DAYS = 30                        # warn after this many days


def _stale_warning(fm: dict[str, Any]) -> str | None:
    """Return a warning string if the feature map is older than _STALE_DAYS."""
    analyzed_at = fm.get("analyzed_at")
    if not analyzed_at:
        return None
    from datetime import datetime, timezone
    try:
        ts = datetime.fromisoformat(analyzed_at.replace("Z", "+00:00"))
        age = (datetime.now(tz=timezone.utc) - ts).days
        if age > _STALE_DAYS:
            return (
                f"Feature map is {age} days old. Results may be outdated. "
                f"Run `faultlines analyze .` to refresh."
            )
    except (ValueError, TypeError):
        pass
    return None


def _inject_warning(result: dict[str, Any], fm: dict[str, Any]) -> dict[str, Any]:
    """Add stale_warning and freshness fields to the result.

    freshness compares last_scanned_sha to the current git HEAD and
    tells the AI agent how many commits behind the feature map is.
    """
    warning = _stale_warning(fm)
    if warning:
        result["stale_warning"] = warning

    freshness = _git_freshness(fm)
    if freshness is not None:
        result["freshness"] = freshness
        if freshness.get("is_stale"):
            behind = freshness.get("commits_behind", 0)
            auto_on = os.environ.get("FAULTLINE_AUTO_REFRESH") in ("1", "true", "yes")
            if auto_on:
                result["stale_warning"] = (
                    f"Feature map is {behind} commit(s) behind HEAD. "
                    f"A background refresh has been triggered — next query "
                    f"will see fresh data."
                )
            else:
                result["stale_warning"] = (
                    f"Feature map is {behind} commit(s) behind HEAD. "
                    f"Run `faultlines refresh` for an LLM-free incremental update, "
                    f"or set FAULTLINE_AUTO_REFRESH=1 to enable automatic updates."
                )
    return result


def _git_freshness(fm: dict[str, Any]) -> dict[str, Any] | None:
    """Compare stored SHA to current HEAD. Returns None if git unavailable."""
    scanned_sha = fm.get("last_scanned_sha", "")
    repo_path = fm.get("repo_path", "")
    if not scanned_sha or not repo_path:
        return None

    import subprocess
    try:
        current = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, text=True, timeout=3,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if current == scanned_sha:
        return {"is_stale": False, "current_sha": current[:8], "scanned_sha": scanned_sha[:8]}

    try:
        behind = int(subprocess.check_output(
            ["git", "rev-list", "--count", f"{scanned_sha}..{current}"],
            cwd=repo_path, text=True, timeout=5,
        ).strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        behind = 0

    return {
        "is_stale": True,
        "current_sha": current[:8],
        "scanned_sha": scanned_sha[:8],
        "commits_behind": behind,
    }


def _load_map() -> dict[str, Any]:
    """Loads the most recent feature-map JSON.

    Precedence:
        1. ``FAULTLINE_MAP_PATH`` environment variable (explicit path)
        2. Most recent ``~/.faultline/feature-map-*.json``
        3. Raises RuntimeError with install instructions
    """
    explicit = os.environ.get("FAULTLINE_MAP_PATH")
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise RuntimeError(
                f"FAULTLINE_MAP_PATH={explicit} does not exist. "
                f"Run `faultlines analyze .` first."
            )
        data = json.loads(p.read_text())
        _maybe_auto_refresh(p, data)
        return data

    home_dir = Path.home() / ".faultline"
    if not home_dir.exists():
        raise RuntimeError(
            "No feature map found at ~/.faultline/. "
            "Run `faultlines analyze /path/to/your/repo --llm --flows` first."
        )

    scans = sorted(home_dir.glob("feature-map-*.json"))
    if not scans:
        raise RuntimeError(
            "No feature-map-*.json found. "
            "Run `faultlines analyze /path/to/your/repo --llm --flows` first."
        )
    latest = scans[-1]
    data = json.loads(latest.read_text())
    _maybe_auto_refresh(latest, data)
    return data


def _maybe_auto_refresh(path: Path, data: dict[str, Any]) -> None:
    """If FAULTLINE_AUTO_REFRESH is enabled, kick off a background refresh.

    Non-blocking: returns immediately. Current query sees the data as it
    was when loaded; next query will see the refreshed version.
    """
    try:
        from faultline.cache.auto_refresh import maybe_trigger_refresh
        maybe_trigger_refresh(path, data)
    except Exception:
        # Auto-refresh is best-effort — never break a tool call because
        # of a refresh issue.
        pass


def _savings_metadata(
    files_returned: int,
    *,
    tool_name: str | None = None,
    query_arg: str | None = None,
) -> dict[str, Any]:
    """Estimate tokens saved vs a naive grep-and-read-files workflow.

    Side effect: when ``tool_name`` is provided AND ``FAULTLINE_API_KEY``
    is set in the environment, the call is enqueued for cloud telemetry.
    No-op otherwise.
    """
    without_mcp = _AVG_GREP_FILES_PER_QUERY * _AVG_TOKENS_PER_FILE
    with_mcp = _AVG_MCP_RESPONSE_TOKENS + (files_returned * _AVG_TOKENS_PER_FILE)
    saved = max(0, without_mcp - with_mcp)

    if tool_name and os.environ.get("FAULTLINE_API_KEY"):
        try:
            from faultline.cloud.event_buffer import record_mcp_event
            record_mcp_event(
                tool_name=tool_name,
                query_arg=query_arg,
                files_returned=files_returned,
                tokens_saved=saved,
            )
        except Exception:
            # Telemetry must never break a tool call.
            pass

    return {
        "estimated_tokens_saved": saved,
        "files_returned": files_returned,
        "baseline_tokens": without_mcp,
    }


@mcp.tool()
def list_features() -> dict[str, Any]:
    """List all features detected in the codebase with health scores.

    Use this when the user asks "what's in this codebase" or "show me
    all the features". Returns a compact overview sorted by risk so
    the riskiest code is visible first. For details on a specific
    feature, follow up with ``find_feature``.
    """
    fm = _load_map()
    features = sorted(
        fm.get("features", []),
        key=lambda f: f.get("health_score", 100),
    )
    return _inject_warning({
        "repo_path": fm.get("repo_path", ""),
        "total_features": len(features),
        "total_commits": fm.get("total_commits", 0),
        "features": [
            {
                "name": f["name"],
                "description": f.get("description"),
                "health": round(f.get("health_score", 0)),
                "bug_fix_ratio": round(f.get("bug_fix_ratio", 0) * 100, 1),
                "commits": f.get("total_commits", 0),
                "file_count": len(f.get("paths", [])),
                "flow_count": len(f.get("flows", [])),
                "coverage_pct": f.get("coverage_pct"),
            }
            for f in features
        ],
        "_savings_metadata": _savings_metadata(0, tool_name="list_features"),
    }, fm)


@mcp.tool()
def find_feature(query: str) -> dict[str, Any] | None:
    """Find a feature by semantic name or description.

    Use this BEFORE reading random files. Much faster than grep and
    returns the full context: file list, health, ownership, flows.
    Match is case-insensitive substring against the feature name and
    description.

    Args:
        query: Feature name or keyword (e.g. "payments", "auth",
            "checkout", "rich text editor")
    """
    fm = _load_map()
    q = query.lower()
    for f in fm.get("features", []):
        name = (f.get("name") or "").lower()
        desc = (f.get("description") or "").lower()
        if q in name or q in desc:
            return _inject_warning({
                "name": f["name"],
                "description": f.get("description"),
                "health": round(f.get("health_score", 0)),
                "bug_fix_ratio": round(f.get("bug_fix_ratio", 0) * 100, 1),
                "coverage_pct": f.get("coverage_pct"),
                "files": f.get("paths", []),
                "file_count": len(f.get("paths", [])),
                "owners": f.get("authors", [])[:5],
                "flows": [
                    {"name": fl["name"], "health": round(fl.get("health_score", 0))}
                    for fl in f.get("flows", [])
                ],
                "_savings_metadata": _savings_metadata(
                    len(f.get("paths", [])),
                    tool_name="find_feature",
                    query_arg=query,
                ),
            }, fm)
    return None


@mcp.tool()
def get_feature_files(feature_name: str) -> dict[str, Any]:
    """Get the exact list of files that belong to a feature.

    Use this to scope a refactor or code review to the files that
    actually matter, instead of grepping the whole repo. Returns
    both source files and test files where available.

    Args:
        feature_name: Exact feature name from ``list_features``
    """
    fm = _load_map()
    for f in fm.get("features", []):
        if f.get("name") == feature_name:
            return _inject_warning({
                "feature": feature_name,
                "files": f.get("paths", []),
                "file_count": len(f.get("paths", [])),
                "hotspot_files": [
                    h for fl in f.get("flows", [])
                    for h in fl.get("hotspot_files", [])
                ][:5],
                "_savings_metadata": _savings_metadata(
                    len(f.get("paths", [])),
                    tool_name="get_feature_files",
                    query_arg=feature_name,
                ),
            }, fm)
    return {"error": f"Feature '{feature_name}' not found", "available": [
        f["name"] for f in fm.get("features", [])
    ]}


@mcp.tool()
def get_hotspots(limit: int = 5) -> dict[str, Any]:
    """Get the riskiest features in the codebase.

    Use this when the user asks "where are the bugs", "what should I
    refactor next", or "what parts of the code are broken". Returns
    features sorted by health score (worst first) with their hotspot
    files — the specific files accumulating the most bug fixes.

    Args:
        limit: Max features to return (default 5)
    """
    fm = _load_map()
    risky = sorted(
        fm.get("features", []),
        key=lambda f: f.get("health_score", 100),
    )[:limit]

    result = []
    for f in risky:
        hotspot_files: list[str] = []
        for fl in f.get("flows", []):
            hotspot_files.extend(fl.get("hotspot_files", []))
        result.append({
            "name": f["name"],
            "description": f.get("description"),
            "health": round(f.get("health_score", 0)),
            "bug_fix_ratio": round(f.get("bug_fix_ratio", 0) * 100, 1),
            "bug_fixes": f.get("bug_fixes", 0),
            "commits": f.get("total_commits", 0),
            "coverage_pct": f.get("coverage_pct"),
            "hotspot_files": hotspot_files[:3],
            "owners": f.get("authors", [])[:3],
        })

    return _inject_warning({
        "hotspots": result,
        "_savings_metadata": _savings_metadata(limit, tool_name="get_hotspots"),
    }, fm)


@mcp.tool()
def get_feature_owners(feature_name: str) -> dict[str, Any]:
    """Get the people who maintain a feature.

    Use this when the user asks "who owns X", "who should review this
    PR", or "who knows about Y". Returns top contributors sorted by
    commit count. Also reports bus factor risk if there's only one
    active owner.

    Args:
        feature_name: Exact feature name from ``list_features``
    """
    fm = _load_map()
    for f in fm.get("features", []):
        if f.get("name") == feature_name:
            authors = f.get("authors", [])
            flow_bus_factors = [
                fl.get("bus_factor", 1) for fl in f.get("flows", [])
            ]
            min_bus_factor = min(flow_bus_factors) if flow_bus_factors else len(authors) or 1
            return _inject_warning({
                "feature": feature_name,
                "owners": authors,
                "total_contributors": len(authors),
                "bus_factor": min_bus_factor,
                "at_risk": min_bus_factor == 1,
                "_savings_metadata": _savings_metadata(
                    1, tool_name="get_feature_owners", query_arg=feature_name,
                ),
            }, fm)
    return {"error": f"Feature '{feature_name}' not found"}


@mcp.tool()
def get_flow_files(feature_name: str, flow_name: str) -> dict[str, Any]:
    """Get files belonging to a specific user-facing flow.

    Use this for PR reviews or targeted refactoring. A flow is a
    named user journey (e.g. "checkout-flow", "manage-team-flow")
    that spans multiple files within a feature.

    Args:
        feature_name: Parent feature name
        flow_name: Flow name from ``find_feature`` results
    """
    fm = _load_map()
    for f in fm.get("features", []):
        if f.get("name") != feature_name:
            continue
        for fl in f.get("flows", []):
            if fl.get("name") == flow_name:
                return _inject_warning({
                    "feature": feature_name,
                    "flow": flow_name,
                    "description": fl.get("description"),
                    "files": fl.get("paths", []),
                    "file_count": len(fl.get("paths", [])),
                    "health": round(fl.get("health_score", 0)),
                    "bug_fix_ratio": round(fl.get("bug_fix_ratio", 0) * 100, 1),
                    "hotspot_files": fl.get("hotspot_files", []),
                    "_savings_metadata": _savings_metadata(
                        len(fl.get("paths", [])),
                        tool_name="get_flow_files",
                        query_arg=f"{feature_name}/{flow_name}",
                    ),
                }, fm)
    return {"error": f"Flow '{flow_name}' in feature '{feature_name}' not found"}


@mcp.tool()
def get_repo_summary() -> dict[str, Any]:
    """High-level stats about the repo: features, commits, health, risk.

    Use this for "give me an overview of this codebase" or when
    starting work on an unfamiliar repo. Returns aggregated metrics
    without file-level detail.
    """
    fm = _load_map()
    features = fm.get("features", [])
    total_bug_fixes = sum(f.get("bug_fixes", 0) for f in features)
    avg_health = (
        sum(f.get("health_score", 0) for f in features) / len(features)
        if features else 0
    )
    at_risk = sum(1 for f in features if f.get("health_score", 100) < 50)
    with_coverage = [
        f.get("coverage_pct") for f in features if f.get("coverage_pct") is not None
    ]
    avg_coverage = sum(with_coverage) / len(with_coverage) if with_coverage else None

    return _inject_warning({
        "repo_path": fm.get("repo_path", ""),
        "remote_url": fm.get("remote_url", ""),
        "analyzed_at": fm.get("analyzed_at", ""),
        "date_range_days": fm.get("date_range_days", 0),
        "total_commits": fm.get("total_commits", 0),
        "total_features": len(features),
        "total_flows": sum(len(f.get("flows", [])) for f in features),
        "total_bug_fixes": total_bug_fixes,
        "avg_health_score": round(avg_health, 1),
        "avg_coverage_pct": round(avg_coverage, 1) if avg_coverage is not None else None,
        "features_at_risk": at_risk,
        "_savings_metadata": _savings_metadata(0, tool_name="get_repo_summary"),
    }, fm)


def main() -> None:
    """Entry point for the ``faultlines-mcp`` console script."""
    # Register impact analysis tools (adds 2 more MCP tools)
    import faultline.impact.mcp_tools  # noqa: F401
    # Register symbol-level attribution tools (adds 2 more MCP tools)
    import faultline.symbols.mcp_tools  # noqa: F401

    mcp.run()


if __name__ == "__main__":
    main()
