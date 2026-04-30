"""Rehydrate a previously-saved scan JSON into a :class:`DeepScanResult`.

Used by the incremental pipeline to seed a new run with the prior
state. Without this, ``--incremental`` would have no baseline to
diff against.

The JSON shape produced by ``faultline analyze`` (via
``output/writer.py``) carries everything we need: features with
their paths, descriptions, flows, flow descriptions, and traced
flow participants. We unpack them back into the dict-of-dicts
shape ``DeepScanResult`` exposes.

No LLM calls. Pure I/O + reshape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from faultline.llm.sonnet_scanner import DeepScanResult


logger = logging.getLogger(__name__)


class PriorScan:
    """Lightweight wrapper around a loaded prior scan.

    Carries the rehydrated :class:`DeepScanResult` plus extras needed
    by the incremental pipeline that don't live on the result itself:
    the SHA the prior scan was taken at, and per-feature stats we want
    to carry forward unchanged when a feature wasn't touched by the
    diff.
    """

    def __init__(
        self,
        result: DeepScanResult,
        last_sha: str | None,
        feature_stats: dict[str, dict[str, Any]],
        flow_stats: dict[str, dict[str, dict[str, Any]]],
        scan_meta: dict[str, Any],
    ) -> None:
        self.result = result
        self.last_sha = last_sha
        self.feature_stats = feature_stats
        self.flow_stats = flow_stats
        self.scan_meta = scan_meta

    @property
    def features(self) -> dict[str, list[str]]:
        return self.result.features

    def stats_for(self, feature_name: str) -> dict[str, Any] | None:
        return self.feature_stats.get(feature_name)

    def flow_stats_for(
        self, feature_name: str, flow_name: str,
    ) -> dict[str, Any] | None:
        return (self.flow_stats.get(feature_name) or {}).get(flow_name)


# Feature-level fields we carry forward verbatim when the feature
# wasn't touched by the diff. Stored separately from
# ``DeepScanResult`` because they're CLI-side enrichment, not core
# pipeline state.
_FEATURE_CARRY_FIELDS = (
    "health_score",
    "bug_fix_ratio",
    "bug_fixes",
    "total_commits",
    "authors",
    "last_modified",
    "coverage_pct",
    "symbol_health_score",
    "shared_attributions",
    "bug_fix_prs",
    "display_name",
)

_FLOW_CARRY_FIELDS = (
    "health_score",
    "bug_fix_ratio",
    "bug_fixes",
    "total_commits",
    "authors",
    "last_modified",
    "coverage_pct",
    "test_file_count",
    "bug_fix_prs",
    "display_name",
    "entry_point_file",
    "entry_point_line",
    "participants",
    "symbol_attributions",
    "hotspot_files",
    "bus_factor",
    "weekly_points",
    "health_trend",
)


def load_scan_as_seed(json_path: str | Path) -> PriorScan:
    """Read a saved scan JSON and rehydrate it as a :class:`PriorScan`.

    Raises ``FileNotFoundError`` if the path doesn't exist and
    ``ValueError`` if the JSON is malformed or doesn't look like a
    scan output.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"scan JSON not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc

    if not isinstance(data, dict) or not isinstance(
        data.get("features"), list,
    ):
        raise ValueError(
            f"{path}: top-level shape doesn't look like a scan "
            f"(missing 'features' list)",
        )

    result = DeepScanResult()
    feature_stats: dict[str, dict[str, Any]] = {}
    flow_stats: dict[str, dict[str, dict[str, Any]]] = {}

    for feat in data["features"]:
        if not isinstance(feat, dict):
            continue
        name = feat.get("name")
        if not name or not isinstance(name, str):
            continue

        paths = feat.get("paths") or []
        if isinstance(paths, list):
            result.features[name] = [str(p) for p in paths]

        desc = feat.get("description")
        if isinstance(desc, str) and desc:
            result.descriptions[name] = desc

        # Carry-forward stats for unchanged features.
        feature_stats[name] = {
            k: feat[k] for k in _FEATURE_CARRY_FIELDS if k in feat
        }

        # Flows.
        flows = feat.get("flows") or []
        if not isinstance(flows, list):
            continue

        flow_names: list[str] = []
        per_flow_descs: dict[str, str] = {}
        per_flow_stats: dict[str, dict[str, Any]] = {}
        per_flow_participants: dict[str, list[Any]] = {}

        for fl in flows:
            if not isinstance(fl, dict):
                continue
            fname = fl.get("name")
            if not isinstance(fname, str) or not fname:
                continue
            flow_names.append(fname)
            fdesc = fl.get("description")
            if isinstance(fdesc, str) and fdesc:
                per_flow_descs[fname] = fdesc
            per_flow_stats[fname] = {
                k: fl[k] for k in _FLOW_CARRY_FIELDS if k in fl
            }
            ps = fl.get("participants")
            if isinstance(ps, list):
                # Keep raw dicts — the trace-flow stage will rebuild
                # FlowParticipant objects from them when needed.
                per_flow_participants[fname] = ps

        if flow_names:
            result.flows[name] = flow_names
        if per_flow_descs:
            result.flow_descriptions[name] = per_flow_descs
        if per_flow_stats:
            flow_stats[name] = per_flow_stats
        if per_flow_participants:
            result.flow_participants[name] = per_flow_participants

    scan_meta = {
        "analyzed_at": data.get("analyzed_at"),
        "repo_path": data.get("repo_path"),
        "remote_url": data.get("remote_url"),
        "total_commits": data.get("total_commits"),
        "date_range_days": data.get("date_range_days"),
        "file_hashes": data.get("file_hashes") or {},
        "symbol_hashes": data.get("symbol_hashes") or {},
    }

    last_sha = data.get("last_scanned_sha")
    if last_sha is not None and not isinstance(last_sha, str):
        last_sha = None

    logger.info(
        "scan_loader: loaded %s — %d features, %d flows, last_sha=%s",
        path.name,
        len(result.features),
        sum(len(v) for v in result.flows.values()),
        (last_sha or "none")[:12],
    )
    return PriorScan(
        result=result,
        last_sha=last_sha,
        feature_stats=feature_stats,
        flow_stats=flow_stats,
        scan_meta=scan_meta,
    )


def find_prior_scan_for(
    repo_root: str | Path,
    cache_dir: str | Path | None = None,
) -> Path | None:
    """Locate the most recent saved scan for ``repo_root``.

    Looks under ``~/.faultline/`` (or ``cache_dir`` override) for files
    named ``feature-map-{repo_slug}-{timestamp}.json`` and returns the
    most recently-modified one. Returns ``None`` when nothing is found.
    """
    base = Path(cache_dir) if cache_dir else Path.home() / ".faultline"
    if not base.is_dir():
        return None
    repo_slug = Path(repo_root).resolve().name.lower().replace(" ", "-")
    pattern = f"feature-map-{repo_slug}-*.json"
    candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
