# Faultline Repo Self-Audit

> My manual ground truth for the featuremap repo, written before
> looking at the live scan output. Used to score the engine
> end-to-end on a codebase I've worked in deeply.
>
> Convention: **product features** are user-facing capabilities of
> the Faultline CLI; **flows** are end-to-end actions a user (or
> the engine) performs.

---

## Top-level architecture

The CLI ingests a git repo, classifies files via a bucketizer, runs
a multi-stage detection pipeline (Sprints 1-7 added incrementally),
and writes a feature-map JSON. There is also a SaaS dashboard side
(separate repo) and an MCP server inside this repo.

```
faultline/
  cli.py                    — typer entrypoint (analyze + sub-cmds)
  analyzer/
    bucketizer.py           — single file classifier
    features.py             — heuristic feature detection
    workspace.py            — pnpm/turbo/nx workspace detection
    repo_classifier.py      — library vs application
    import_graph.py         — TS/JS import resolution
    ast_extractor.py        — regex TS/JS/Py extractor
    incremental.py          — incremental scan
    shared_files.py         — symbol attribution
    git.py                  — git wrapper
    coverage.py             — coverage parser
    repo_config.py          — .faultline.yaml loader
    symbol_graph.py         — Sprint 7
    flow_tracer.py          — Sprint 7
    layer_classifier.py     — Sprint 7
    humanize.py             — Title Case
  llm/
    pipeline.py             — single entry run()
    sonnet_scanner.py       — deep_scan / deep_scan_workspace
    tool_use_scan.py        — Sprint 1 tool-use loop
    tools.py                — read/list/grep/git + flow tools
    dedup.py                — Sprint 2
    sub_decompose.py        — Sprint 3
    flow_detector_v2.py     — Sprint 4
    critique.py             — Sprint 5
    flow_detector.py        — legacy Haiku
    detector.py             — legacy 5-strategy
    cost.py                 — CostTracker
  benchmark/
    loader.py / metrics.py / report.py    — Sprint 6
  output/
    reporter.py / writer.py
  mcp/                      — MCP server
  symbols/                  — symbol enrichment
```

Big buckets I expect the engine to surface as features:

---

## Expected features (my ground truth)

### 1. Detection Pipeline Core
End-to-end orchestration of feature/flow detection. Includes
`pipeline.run`, bucketizer, deep_scan, workspace detection.

- **Files:** `faultline/llm/pipeline.py`,
  `faultline/llm/sonnet_scanner.py`,
  `faultline/analyzer/bucketizer.py`,
  `faultline/analyzer/workspace.py`,
  `faultline/analyzer/features.py`,
  `faultline/analyzer/repo_classifier.py`.
- **Aliases I'd accept:** `pipeline`, `feature-detection`,
  `deep-scan`, `workspace-detection`.

### 2. Tool-Augmented Per-Package Detection (Sprint 1)
LLM gets read-only tools to inspect file contents instead of
guessing from paths.

- **Files:** `faultline/llm/tool_use_scan.py`,
  `faultline/llm/tools.py`.
- **Aliases:** `tool-use-detection`, `per-package-detection`,
  `naming-intelligence`.

### 3. Cross-Cluster Dedup (Sprint 2)
Single Sonnet pass that merges semantic duplicates across packages.

- **Files:** `faultline/llm/dedup.py`.
- **Aliases:** `dedup`, `cross-cluster-reconciliation`,
  `feature-merging`.

### 4. Sub-Decomposition (Sprint 3)
Splits oversized features (>200 files) into 2-6 named sub-features
via tool-augmented scan.

- **Files:** `faultline/llm/sub_decompose.py`.
- **Aliases:** `sub-decomposition`, `feature-splitting`.

### 5. Tool-Augmented Flow Detection (Sprint 4)
Per-feature flow detection grounded in real route/event handlers.

- **Files:** `faultline/llm/flow_detector_v2.py`,
  parts of `faultline/llm/tools.py`
  (`find_route_handlers`, `find_event_handlers`).
- **Aliases:** `flow-detection`, `tool-flows`.

### 6. Self-Critique Loop (Sprint 5)
Final pass that flags weak names and re-investigates with tools.

- **Files:** `faultline/llm/critique.py`.
- **Aliases:** `critique`, `name-refinement`, `self-critique`.

### 7. Call-Graph Flow Tracing (Sprint 7)
BFS from each Sprint 4 entry point through the import graph;
classifies each touched file as ui/state/api-client/api-server/
schema/support.

- **Files:** `faultline/analyzer/symbol_graph.py`,
  `faultline/analyzer/flow_tracer.py`,
  `faultline/analyzer/layer_classifier.py`.
- **Aliases:** `flow-tracing`, `call-graph`, `participants`,
  `trace-flows`.

### 8. Repo Config & Aliasing
`.faultline.yaml` loader; canonical aliases / skip rules /
force-merges; auto-fold built-in tooling packages.

- **Files:** `faultline/analyzer/repo_config.py`, parts of
  `faultline/llm/dedup.py` (TOOLING_PACKAGE_NAMES),
  parts of `faultline/llm/pipeline.py` (`_auto_fold_tooling`).
- **Aliases:** `repo-config`, `aliasing`, `canonical-names`,
  `skip-rules`.

### 9. Display Names / Humanization
Title Case display labels for features and flows.

- **Files:** `faultline/analyzer/humanize.py`,
  `cli._populate_display_names`.
- **Aliases:** `display-names`, `humanize`, `labels`.

### 10. Cost Tracking
CostTracker thread-safe budget enforcement and end-of-run summary.

- **Files:** `faultline/llm/cost.py`.
- **Aliases:** `cost`, `budget-enforcement`,
  `cost-tracking`.

### 11. Output Writer + Reporter
Terminal report rendering + feature-map.json writer.

- **Files:** `faultline/output/writer.py`,
  `faultline/output/reporter.py`.
- **Aliases:** `output`, `reporter`, `terminal-report`.

### 12. Coverage Integration
Reads lcov / coverage-summary.json and attaches per-file pct to
features/flows.

- **Files:** `faultline/analyzer/coverage.py`,
  `cli._apply_feature_coverage`.
- **Aliases:** `coverage`, `coverage-integration`.

### 13. AST + Import Graph (Symbol Resolution)
Regex TS/JS/Python extractor + path resolution + monorepo detection.

- **Files:** `faultline/analyzer/ast_extractor.py`,
  `faultline/analyzer/import_graph.py`.
- **Aliases:** `ast`, `signatures`, `import-graph`,
  `symbol-resolution`.

### 14. Symbol-Level Attribution
Per-symbol feature attribution for shared files; `--symbols` flag.

- **Files:** `faultline/symbols/`,
  `faultline/analyzer/shared_files.py`.
- **Aliases:** `symbol-attribution`, `symbols`.

### 15. Incremental Scan
Re-scan only changed files since last run.

- **Files:** `faultline/analyzer/incremental.py`,
  `cli` `pull` / `refresh` parts.
- **Aliases:** `incremental`, `incremental-scan`, `cache`.

### 16. Benchmark Harness (Sprint 6)
YAML ground-truth + scoring script + Markdown report.

- **Files:** `faultline/benchmark/`,
  `scripts/run_benchmark.py`.
- **Aliases:** `benchmark`, `harness`, `scoring`.

### 17. MCP Server
Model Context Protocol server exposing the feature-map.

- **Files:** `faultline/mcp/`.
- **Aliases:** `mcp`, `mcp-server`.

### 18. Analytics Integration (PostHog / Sentry)
Per-feature pageviews + errors → impact score.

- **Files:** `cli._run_analytics`, related impact code.
- **Aliases:** `analytics`, `posthog-integration`,
  `sentry-integration`, `impact-score`.

### 19. Cloud Sync (Push)
Optional upload to Faultlines SaaS.

- **Files:** `faultline/cloud/`, `cli._push_*`.
- **Aliases:** `cloud-sync`, `push`, `saas-upload`.

### 20. Legacy Fallback Path
Pre-rewrite 5-strategy detector kept under `--legacy`.

- **Files:** `faultline/llm/detector.py`,
  `faultline/llm/flow_detector.py`,
  `faultline/llm/deepseek_client.py`.
- **Aliases:** `legacy`, `5-strategy-fallback`.

### 21. Documentation / Tests / Sprint Plans
Synthetic buckets — should NOT count as a real product feature.
The bucketizer puts these in `documentation` automatically; my
audit excludes them from "real features".

---

## Expected flows (subset, not exhaustive)

### Detection Pipeline Core
- `Run Full Scan` — entry: `cli.analyze`. User runs `faultline
  analyze`, pipeline executes through buckets → packages →
  dedup → output.
- `Detect Workspace Packages` — entry:
  `analyzer.workspace.detect_workspace`.
- `Build Commit Context` — entry:
  `sonnet_scanner.build_commit_context`.

### Tool-Augmented Per-Package
- `Run Per-Package Tool Scan` — entry:
  `tool_use_scan.tool_use_scan_package`.
- `Dispatch Read Tool` — entry: `tools.dispatch_tool`.

### Dedup
- `Run Dedup Pass` — entry: `dedup.dedup_features`.
- `Apply Merge Operations` — entry: `dedup._apply_merges`.

### Sub-Decompose
- `Sub-Decompose Oversized Feature` — entry:
  `sub_decompose.sub_decompose_oversized`.
- `Validate Split Coverage` — entry:
  `sub_decompose._validate_split`.

### Tool-Flows
- `Detect Flows With Tools` — entry:
  `flow_detector_v2.detect_flows_with_tools`.
- `Find Route Handlers` — entry: `tools.find_route_handlers`.

### Critique
- `Run Critique Pass` — entry: `critique.critique_and_refine`.
- `Rewrite Feature Name` — entry: `critique._rewrite_feature_name`.

### Trace
- `Trace Flow Callgraph` — entry:
  `flow_tracer.trace_flow_callgraph`.
- `BFS Walk From Entry` — entry: `flow_tracer.trace_flow`.
- `Classify File Layer` — entry:
  `layer_classifier.classify_file`.

### Repo Config
- `Load .faultline.yaml` — entry:
  `repo_config.load_repo_config`.
- `Apply Repo Config Aliases` — entry:
  `repo_config.apply_repo_config`.

### Output
- `Write Feature Map JSON` — entry: `output.writer.write_feature_map`.
- `Print Terminal Report` — entry:
  `output.reporter.print_report`.

### Cloud Sync
- `Upload Feature Map` — entry: `cloud/sync.push_feature_map`.
- `Pull Cloud Scan` — entry: cli `pull` command.

### Analytics
- `Fetch PostHog Pageviews` — entry: posthog client.
- `Fetch Sentry Errors` — entry: sentry client.
- `Compute Impact Score` — entry: `_compute_impact_scores`.

### MCP
- `Serve MCP Tool Calls` — entry: `mcp.server`.

### CLI Subcommands
- `Run Watch Mode` — entry: `cli.watch`.
- `Run Refresh / Incremental` — entry: `cli.refresh`.
- `Run Deep Scan Subcommand` — entry: `cli.deep_scan`.

---

## Layer expectations for trace participants

For Faultline's own code, the layers I expect:

- **`api-server`-shaped** (entry-points): `cli.py` typer commands —
  but path patterns probably won't classify as api-server (no
  routes/* or app.get patterns). Likely "support" — that's a real
  miss.
- **`state`**: very little. Maybe `_last_scan_result` global in
  `sonnet_scanner.py`.
- **`schema`**: Pydantic models in `models/types.py` — but my
  classifier's regex looks for prisma/Django/Sequelize patterns,
  not Pydantic `BaseModel`. So `models/types.py` probably ends
  up "support" instead of "schema".
- **`ui`**: zero — this is a CLI, no React/Next.
- **`api-client`**: maybe `analyzer/git.py` (HTTP-ish git remote
  fetches), or `cloud/sync.py` (POST to /api/cloud/scans).
- **`support`**: most utility code (`humanize.py`, `validation.py`,
  cost calculations, AST extractors).

**Predicted layer mismatch:** Faultline's classifier was tuned for
JS/TS web apps. On a Python CLI like itself, ui/state/api-server
mostly won't fire — much will land "support". This is a real
limitation worth knowing.

---

## My predictions for the engine's output

After watching documenso scans, I expect for this repo:

- **15-25 features** detected. Some of my 20 expected may merge
  (e.g. "AST + Import Graph" + "Symbol-Level Attribution" might
  collapse into "code analysis" or split differently).
- **~70-90% recall** — I expect the engine to find most of my
  list. Misses likely on things I labelled separately that the
  engine merges, or vice versa.
- **Generic-name rate: 0%** — the auto-fold + critique should
  catch any.
- **Cost: $1-3** — small repo, ~50-80 source files in `faultline/`.
- **Flows: 30-60** total, ~5 grounded with entry points (most
  Faultline functions are imported, not "user-facing flows" in
  the Sprint 4 sense).
- **Layer breakdown:** dominated by `support` because Faultline
  is a Python CLI, not a JS/TS web app.

The interesting discoveries will be:
1. Did the engine find features I didn't think to enumerate?
2. Did it merge things I had as separate?
3. How well does the layer classifier handle Python files?

After the scan completes I'll diff this audit against engine output
in `SELF_AUDIT_COMPARISON.md`.
