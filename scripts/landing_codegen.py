"""
Generate TypeScript benchmark-repos.ts entries for the 5 verified repos.

Reads benchmarks/<repo>/feature-map-verified.json and emits a TS code block
ready to splice into faultlines-app/src/lib/benchmark-repos.ts.
"""

import json
import math
import sys
from pathlib import Path


def health_score(ratio: float) -> float:
    """Sigmoid: matches landing page formula 100 / (1 + exp(8 * (ratio - 0.55)))."""
    return 100.0 / (1.0 + math.exp(8.0 * (ratio - 0.55)))


def title_case(name: str) -> str:
    """checkout → Checkout · order-management → Order Management."""
    return " ".join(w.capitalize() for w in name.replace("/", " ").replace("-", " ").split())


def js_str(s: str) -> str:
    """Safely encode a string as JS literal."""
    return json.dumps(s, ensure_ascii=False)


def emit_participant(p: dict) -> str:
    """Emit one Participant entry as inline TS object."""
    parts = [f"path: {js_str(p['path'])}"]
    if p.get("symbol"):
        parts.append(f"symbol: {js_str(p['symbol'])}")
    if p.get("lineStart") is not None:
        parts.append(f"lineStart: {int(p['lineStart'])}")
    if p.get("lineEnd") is not None:
        parts.append(f"lineEnd: {int(p['lineEnd'])}")
    if p.get("role"):
        parts.append(f"role: {js_str(p['role'])}")
    shared = p.get("sharedWith") or []
    if shared:
        parts.append("sharedWith: [" + ", ".join(js_str(s) for s in shared) + "]")
    return "{ " + ", ".join(parts) + " }"


def participants_from_attributions(
    attributions: list[dict],
) -> list[dict]:
    """Flatten SymbolAttribution[] into [Participant] for landing codegen.

    Each attribution may carry multiple symbols and multiple line ranges;
    we emit one Participant per (symbol, range) pair so the UI can deep-
    link.  ``sharedWith`` carries the multi-attribution badge data.
    """
    out: list[dict] = []
    for attr in attributions or []:
        ranges = attr.get("line_ranges") or []
        symbols = attr.get("symbols") or [None]
        roles = attr.get("roles") or {}
        shared = attr.get("shared_with_flows") or []
        for sym in symbols:
            for (start, end) in ranges:
                out.append({
                    "path": attr["file_path"],
                    "symbol": sym,
                    "lineStart": start,
                    "lineEnd": end,
                    "role": roles.get(sym) if sym else None,
                    "sharedWith": shared,
                })
    return out


def emit_flow(flow: dict) -> str:
    name = flow.get("display_name") or title_case(flow.get("name", ""))
    ratio = flow.get("bug_fix_ratio", 0)
    commits = flow.get("total_commits", 0)
    files = len(flow.get("paths", []))
    health = round(health_score(ratio))
    fields = [
        f"name: {js_str(name)}",
        f"health: {health}",
        "cov: null",
        f"ratio: {round(ratio * 100)}",
        f"commits: {commits}",
        f"files: {files}",
    ]
    # Sprint 4 optional fields
    participants = participants_from_attributions(flow.get("symbol_attributions"))
    if participants:
        block = "[" + ", ".join(emit_participant(p) for p in participants[:30]) + "]"
        fields.append(f"participants: {block}")
    test_files = flow.get("test_files") or []
    if test_files:
        block = "[" + ", ".join(js_str(t) for t in test_files[:20]) + "]"
        fields.append(f"testFiles: {block}")
    return "        { " + ", ".join(fields) + " }"


def emit_feature(f: dict) -> str:
    name = f.get("display_name") or title_case(f.get("name", ""))
    ratio = f.get("bug_fix_ratio", 0)
    commits = f.get("total_commits", 0)
    files = len(f.get("paths", []))
    health = round(health_score(ratio))
    flows = (f.get("flows") or [])[:8]
    total_flows = len(f.get("flows") or [])

    # Build extras object for new optional fields
    extras: dict[str, str] = {}
    sym_health = f.get("symbol_health_score")
    if sym_health is not None:
        extras["lineScopedHealth"] = str(round(float(sym_health)))
    cov = f.get("coverage_pct")
    if cov is not None:
        extras["lineScopedCoverage"] = str(round(float(cov)))
    # Feature-level participants from shared_attributions
    feat_participants = participants_from_attributions(f.get("shared_attributions"))
    if feat_participants:
        block = "[" + ", ".join(emit_participant(p) for p in feat_participants[:50]) + "]"
        extras["participants"] = block

    if flows:
        flow_block = "[\n" + ",\n".join(emit_flow(fl) for fl in flows) + "\n      ]"
        flows_arg = flow_block
    else:
        flows_arg = "undefined"

    # Compose extras inline as TS object literal
    extras_str = ""
    if extras:
        kv = ", ".join(f"{k}: {v}" for k, v in extras.items())
        extras_str = f", {{ {kv} }}"
    else:
        extras_str = ""

    return (
        f"      feat({js_str(name)}, \"\", {health}, {round(ratio * 100)}, "
        f"{commits}, {files}, {flows_arg}, null, {total_flows}{extras_str})"
    )


REPO_META = {
    "gitea": {
        "title": "~/gitea",
        "url": "https://github.com/go-gitea/gitea",
        "lang": "Go + TS · GitHub clone · 45k★",
        "detected": "Go forge",
    },
    "ollama": {
        "title": "~/ollama",
        "url": "https://github.com/ollama/ollama",
        "lang": "Go · LLM runtime · 96k★",
        "detected": "Go LLM runtime",
    },
    "meilisearch": {
        "title": "~/meilisearch",
        "url": "https://github.com/meilisearch/meilisearch",
        "lang": "Rust · search engine · Cargo workspace · 47k★",
        "detected": "Cargo workspace",
    },
    "saleor": {
        "title": "~/saleor",
        "url": "https://github.com/saleor/saleor",
        "lang": "Python + TS · headless e-commerce · Django + GraphQL · 21k★",
        "detected": "Django apps",
    },
    "supabase": {
        "title": "~/supabase",
        "url": "https://github.com/supabase/supabase",
        "lang": "TS monorepo · BaaS · pnpm workspace · 75k★",
        "detected": "pnpm workspace",
    },
}


def emit_repo(slug: str) -> str:
    meta = REPO_META[slug]
    d = json.loads(Path(f"benchmarks/{slug}/feature-map-verified.json").read_text())
    features = d.get("features", [])
    n_features = len(features)
    n_flows = sum(len(f.get("flows") or []) for f in features)
    n_files = sum(len(f.get("paths") or []) for f in features)
    total_commits = d.get("total_commits") or sum(f.get("total_commits", 0) for f in features)
    scan_date = (d.get("analyzed_at") or "")[:10]
    window_days = d.get("date_range_days", 365)

    feats_block = ",\n".join(emit_feature(f) for f in sorted(
        features, key=lambda x: -x.get("total_commits", 0)
    ))

    return f"""  {{
    id: "{slug}",
    title: {js_str(meta["title"])},
    repoUrl: {js_str(meta["url"])},
    shortName: "{slug}",
    langLabel: {js_str(meta["lang"])},
    detected: {js_str(meta["detected"])},
    fileCount: "{n_files:,} files",
    commitCount: "{total_commits:,} commits",
    featureCount: {n_features},
    flowCount: {n_flows},
    elapsed: "—",
    cost: "—",
    outputFile: "~/.faultline/feature-map-{slug}.json",
    topNote: "{n_features} features",
    scanDate: {js_str(scan_date)},
    windowDays: {window_days},
    features: [
{feats_block},
    ],
  }}"""


if __name__ == "__main__":
    out = []
    for slug in ["gitea", "ollama", "meilisearch", "saleor", "supabase"]:
        out.append(emit_repo(slug))
    print(",\n".join(out) + ",")
