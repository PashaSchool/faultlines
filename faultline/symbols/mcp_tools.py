"""MCP tools that expose symbol-level attribution.

Registers additional tools on the shared MCP server. AI agents get
precise function-level context when available, falling back to full
file paths when symbols haven't been attributed.
"""

from __future__ import annotations

from typing import Any

from faultline.mcp_server import mcp, _load_map, _inject_warning, _savings_metadata


def _build_deeplinks(
    remote_url: str,
    file_path: str,
    line_ranges: list[Any],
) -> list[str]:
    """Build GitHub blob URLs for each line range.

    Returns ``["{remote}/blob/HEAD/{file}#L{start}-L{end}", ...]``.
    Falls back to a bare file path when remote_url is empty (e.g.
    local-only scans). Range items are accepted as ``[start, end]``
    or ``(start, end)`` — JSON sends lists, Python sends tuples.
    """
    if not file_path:
        return []
    base = remote_url.rstrip("/") if remote_url else ""
    out: list[str] = []
    for rng in line_ranges or []:
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        start, end = rng[0], rng[1]
        if base:
            out.append(f"{base}/blob/HEAD/{file_path}#L{start}-L{end}")
        else:
            out.append(f"{file_path}#L{start}-L{end}")
    return out


@mcp.tool()
def find_symbols_in_flow(feature_name: str, flow_name: str) -> dict[str, Any]:
    """Get precise symbols (functions, classes) that belong to a flow.

    Returns a list of symbols grouped by file, so the AI agent can read
    only the relevant functions instead of the full file. Falls back
    to full file paths when symbol-level attribution is unavailable.

    Use this for PR reviews or refactoring where you need to touch
    only the code that's part of a specific user journey.

    Args:
        feature_name: Parent feature name (from list_features).
        flow_name: Flow name (from find_feature or get_flow_files).
    """
    fm = _load_map()
    for f in fm.get("features", []):
        if f.get("name") != feature_name:
            continue
        for fl in f.get("flows", []):
            if fl.get("name") != flow_name:
                continue

            attributions = fl.get("symbol_attributions", [])
            if attributions:
                remote = fm.get("remote_url", "")
                return _inject_warning({
                    "feature": feature_name,
                    "flow": flow_name,
                    "precision": "symbol-level",
                    "attributions": [
                        {
                            "file": a.get("file_path"),
                            "symbols": a.get("symbols", []),
                            "line_ranges": a.get("line_ranges", []),
                            "attributed_lines": a.get("attributed_lines", 0),
                            "total_file_lines": a.get("total_file_lines", 0),
                            "deeplinks": _build_deeplinks(
                                remote, a.get("file_path", ""), a.get("line_ranges", []),
                            ),
                        }
                        for a in attributions if a.get("symbols")
                    ],
                    "fallback_files": fl.get("paths", []),
                    "hint": "Read only the symbols listed. Use deeplinks for direct GitHub navigation. Fall back to fallback_files when full context is needed.",
                    "_savings_metadata": _savings_metadata(
                        sum(len(a.get("symbols", [])) for a in attributions)
                    ),
                }, fm)

            # Fallback when --symbols wasn't enabled at scan time
            return _inject_warning({
                "feature": feature_name,
                "flow": flow_name,
                "precision": "file-level",
                "attributions": [],
                "fallback_files": fl.get("paths", []),
                "hint": (
                    "Symbol-level attribution not available for this scan. "
                    "Re-run with `faultlines analyze . --llm --flows --symbols` "
                    "for precise function-level context."
                ),
                "_savings_metadata": _savings_metadata(len(fl.get("paths", []))),
            }, fm)

    return _inject_warning({
        "error": f"Flow '{flow_name}' in feature '{feature_name}' not found",
    }, fm)


@mcp.tool()
def find_symbols_for_feature(feature_name: str) -> dict[str, Any]:
    """Get the feature's shared symbols (types, interfaces, enums).

    Returns types and interfaces that are shared across all flows in
    the feature. These are the contracts/models your AI agent needs
    to understand the feature's data shape.

    Use this when starting work on a feature — types first, then
    dive into specific flows with find_symbols_in_flow.

    Args:
        feature_name: Feature name from list_features.
    """
    fm = _load_map()
    for f in fm.get("features", []):
        if f.get("name") != feature_name:
            continue

        shared = f.get("shared_attributions", [])
        remote = fm.get("remote_url", "")
        return _inject_warning({
            "feature": feature_name,
            "description": f.get("description"),
            "shared_symbols": [
                {
                    "file": a.get("file_path"),
                    "symbols": a.get("symbols", []),
                    "line_ranges": a.get("line_ranges", []),
                    "deeplinks": _build_deeplinks(
                        remote, a.get("file_path", ""), a.get("line_ranges", []),
                    ),
                }
                for a in shared if a.get("symbols")
            ],
            "all_files": f.get("paths", []),
            "flow_count": len(f.get("flows", [])),
            "hint": (
                "Shared symbols are types, interfaces, and enums used across "
                "all flows. For function-level attribution per flow, use "
                "find_symbols_in_flow."
            ),
            "_savings_metadata": _savings_metadata(
                sum(len(a.get("symbols", [])) for a in shared)
            ),
        }, fm)

    return _inject_warning({"error": f"Feature '{feature_name}' not found"}, fm)
