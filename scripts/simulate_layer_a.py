"""Sprint 12 Day 2 — apply Layer A on a frozen feature-map JSON.

Lets us evaluate cluster promotion on baseline scans without re-running
LLM. Loads JSON → builds DeepScanResult → runs promote_virtual_clusters
→ writes new JSON → run eval_flow_attribution.py against it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from faultline.llm.flow_cluster import promote_virtual_clusters
from faultline.llm.sonnet_scanner import DeepScanResult


def load_into_result(fm: dict) -> DeepScanResult:
    features = {f["name"]: list(f.get("paths", [])) for f in fm.get("features", [])}
    flows: dict[str, list[str]] = {}
    flow_descriptions: dict[str, dict[str, str]] = {}
    flow_participants: dict[str, dict[str, list]] = {}
    descriptions: dict[str, str] = {}
    for f in fm.get("features", []):
        descriptions[f["name"]] = f.get("description") or ""
        ff = []
        fd: dict[str, str] = {}
        fp: dict[str, list] = {}
        for fl in f.get("flows", []):
            ff.append(fl["name"])
            if fl.get("description"):
                fd[fl["name"]] = fl["description"]
            if fl.get("participants"):
                fp[fl["name"]] = list(fl["participants"])
        flows[f["name"]] = ff
        if fd:
            flow_descriptions[f["name"]] = fd
        if fp:
            flow_participants[f["name"]] = fp
    return DeepScanResult(
        features=features,
        flows=flows,
        descriptions=descriptions,
        flow_descriptions=flow_descriptions,
        flow_participants=flow_participants,
    )


def write_back(fm: dict, result: DeepScanResult) -> dict:
    """Reproduce feature-map JSON shape from the mutated result."""
    by_name = {f["name"]: f for f in fm.get("features", [])}
    new_feats: list[dict] = []
    for name, paths in result.features.items():
        existing = by_name.get(name)
        if existing is None:
            existing = {
                "name": name,
                "description": result.descriptions.get(name, ""),
                "paths": list(paths),
                "authors": [],
                "total_commits": 0,
                "bug_fixes": 0,
                "bug_fix_ratio": 0.0,
                "last_modified": "1970-01-01T00:00:00Z",
                "health_score": 100.0,
                "flows": [],
            }
        else:
            existing = dict(existing)
            existing["paths"] = list(paths)
            existing["description"] = result.descriptions.get(name, existing.get("description", ""))

        # Move flows in by-name. To preserve metadata we look them up
        # in the original JSON by flow name across all features.
        original_flows: dict[str, dict] = {}
        for f in fm.get("features", []):
            for fl in f.get("flows", []):
                original_flows.setdefault(fl["name"], fl)

        new_flows = []
        for flow_name in result.flows.get(name, []):
            src = original_flows.get(flow_name)
            if src is not None:
                new_flows.append(src)
            else:
                # Synthesised flow — minimal stub.
                new_flows.append({
                    "name": flow_name,
                    "paths": [],
                    "authors": [],
                    "total_commits": 0,
                    "bug_fixes": 0,
                    "bug_fix_ratio": 0.0,
                    "last_modified": "1970-01-01T00:00:00Z",
                    "health_score": 100.0,
                })
        existing["flows"] = new_flows
        new_feats.append(existing)

    out = dict(fm)
    out["features"] = new_feats
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: simulate_layer_a.py <input.json> <output.json>")
        return 1
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    fm = json.loads(in_path.read_text())
    result = load_into_result(fm)
    n = promote_virtual_clusters(result)
    print(f"promoted {n} synthetic feature(s)")
    if n:
        new_features = [name for name in result.features if name in {"auth", "billing", "notifications"}]
        for name in new_features:
            paths = result.features.get(name, [])
            flows = result.flows.get(name, [])
            print(f"  {name:18s} paths={len(paths)} flows={len(flows)}")
    new_fm = write_back(fm, result)
    out_path.write_text(json.dumps(new_fm, indent=2, default=str))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
