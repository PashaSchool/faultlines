# Changelog

All notable changes to `faultlines` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] — 2026-04-14

### Added
- **File watcher daemon** (`faultline/watch/`). Observes a repo for changes
  and triggers incremental refresh after a debounce window (default 30 s
  of silence). New CLI commands: `faultlines watch`, `watch-status`,
  `watch-stop`. Runs foreground by default; `--daemon` forks to background.
- **Ollama provider for symbol attribution.** `--symbols --provider ollama`
  now works end-to-end using a local `qwen2.5-coder:14b` (or
  user-specified) model via the Ollama HTTP API. Zero cost for power
  users with a local GPU.
- Optional dependency group: `pip install 'faultlines[watch]'` pulls
  `watchdog>=4.0.0`.

### Changed
- `attribute_symbols_to_flows` and `enrich_with_symbols` now accept a
  `provider` kwarg (`"anthropic" | "ollama"`) and an `ollama_host`
  parameter.
- Watcher ignores `.git/`, `node_modules/`, `__pycache__/`, `.venv/`,
  `dist/`, `build/`, and similar directories; only reacts to source-file
  extensions.

### Tests
- 16 new tests covering path filtering, PID tracking, watcher lifecycle,
  and refresh behaviour.
- 3 new throttle-config tests for the auto-refresh env var.
- Total: 647 tests passing.

## [0.9.1] — 2026-04-14

### Added
- `FAULTLINE_AUTO_REFRESH_THROTTLE` environment variable to control the
  throttle window between background refresh attempts in the MCP server.
  Accepts seconds as integer (`"0"` disables throttling). Configurable
  per-project via the MCP client config.

### Tests
- 3 new tests for throttle override, `0` disables throttling, and
  invalid values falling back to the default.

## [0.9.0] — 2026-04-14

### Added
- **MCP-triggered background auto-refresh** (`faultline/cache/auto_refresh.py`).
  When `FAULTLINE_AUTO_REFRESH=1` is set, the MCP server checks the
  feature map's `last_scanned_sha` against git HEAD on every tool call.
  If stale and not throttled, a background thread runs the incremental
  refresh and writes the updated map back to disk. The current query
  returns immediately with a `stale_warning`.
- File-based lock (`~/.faultline/.refresh-{slug}.lock`) that prevents
  concurrent refreshes across MCP instances. Stale locks auto-released
  after 10 minutes or when the owning PID is dead.
- Per-repo throttle (default: 5 minutes between attempts).
- Opt-in via env var; default behaviour is unchanged.

### Tests
- 13 new tests: lock acquire/release, stale-lock auto-release, throttle
  behaviour, env-gated activation, cheap freshness check.

## [0.8.0] — 2026-04-14

### Added
- **Symbol-level incremental refresh** (`faultline/cache/symbols.py`).
  Compares stored symbol body hashes to the current AST extraction and:
  - Cleans up attributions for removed symbols.
  - Re-attributes newly added symbols via one targeted LLM call per
    affected feature.
  - Preserves attribution when only the function body changed (name and
    purpose unchanged — the attribution is still valid).
- CLI flag: `faultlines refresh --refresh-symbols`.

### Tests
- 13 new tests for delta computation (added / removed / body-modified),
  attribution cleanup, and body-only preservation.

## [0.7.0] — 2026-04-14

### Added
- **Orphan file discovery** (`faultline/cache/discovery.py`). After a
  refresh, classifies files that don't belong to any feature:
  - **Heuristic stage** — directory-prefix match against existing
    feature paths (no LLM).
  - **LLM stage** — for unmatched orphans, Claude decides whether they
    extend an existing feature or form a new one.
  - Returns structured `FeatureProposal` objects with confidence levels.
- CLI flags:
  - `faultlines refresh --detect-new` to classify orphans.
  - `faultlines refresh --detect-new --auto-apply` to apply
    high-confidence proposals.
- `apply_report()` mutates the feature map with approved proposals
  (extend existing or create placeholder new features).

### Reliability
- Unknown `extends_feature` targets downgraded to `skip`.
- Files not in the original prompt group are stripped from LLM output
  (prevents hallucinated file paths).
- Files the LLM didn't account for are auto-marked `skip`.

### Tests
- 14 new tests covering heuristic matching, LLM response parsing,
  hallucination rejection, apply behaviour, and confidence filtering.

## [0.6.0] — 2026-04-14

### Added
- **Incremental refresh + git-SHA freshness tracking** (`faultline/cache/`).
  New CLI command `faultlines refresh` runs the existing `analyzer/
  incremental.py` pipeline and updates content hashes without any LLM
  calls. Preserves flow and symbol attributions on untouched features.
- `FeatureMap` model extended with:
  - `last_scanned_sha` — git HEAD at scan time.
  - `file_hashes` — `{path: sha256}` for change detection.
  - `symbol_hashes` — `{path: {symbol: sha256}}` for per-symbol caching.
- MCP server now reports `freshness` on every response:
  `{is_stale, commits_behind, current_sha, scanned_sha}`.
- `stale_warning` upgraded with actionable advice
  (`"Run faultlines refresh"`).
- Analyze pipeline stamps `last_scanned_sha` and `file_hashes` on save
  so refreshes work from day one.

### Tests
- 11 new tests for content/symbol hashing, freshness detection,
  no-op when fresh, and incremental hash updates.

## [0.5.0] — 2026-04-14

### Added
- **Symbol-level attribution for flows** (`faultline/symbols/`). AI
  agents get precise function-level context instead of whole files.
- New CLI flag: `faultlines analyze . --llm --flows --symbols`.
- Two new MCP tools:
  - `find_symbols_in_flow(feature, flow)` — returns
    `[{file, symbols}]` plus `fallback_files`.
  - `find_symbols_for_feature(feature)` — returns shared types and
    interfaces.
- Types, interfaces, and enums are always attributed at the feature
  level (shared context). Functions, classes, and constants can belong
  to multiple flows if used across journeys.
- Hallucinated symbols are rejected against the AST extract.

### Model changes
- `Flow.symbol_attributions: list[SymbolAttribution] = []`.

### Tests
- 15 new tests for symbol extraction, attribution parsing, mapping
  validation, and pipeline orchestration. 577 total passing.

## [0.4.0] — 2026-04-12

### Added
- **Stale feature map warning in MCP responses.** Maps older than 30
  days produce a `stale_warning` field asking the user to re-scan.
- **Impact analysis module** (`faultline/impact/`). Fully isolated,
  reads from the feature map plus git history.
  - `predict_impact(changed_files, feature_map, repo_path)` — returns
    affected features, missing co-changes, risk signals, regression
    probability, bus-factor and coverage-gap warnings.
  - Two new MCP tools: `analyze_change_impact` and `get_regression_risk`.
  - Uses git co-change patterns (behavioral), not AST imports. Catches
    cross-boundary and runtime dependencies that static analysis misses.
  - Zero LLM calls — pure git history + feature map lookup, < 1 s.

### Notes
- `cochange_detector.py` moved from `analyzer/` to `impact/cochange.py`
  (it was unused in `analyzer/`; impact now owns it).

### Tests
- 19 new impact tests. 562 total passing.

## [0.3.0] — 2026-04-11

### Added
- **MCP server for AI coding agents** (`faultline/mcp_server.py`,
  `faultlines-mcp` console script).
- 7 tools for Cursor, Claude Code, Cline, and Aider:
  `list_features`, `find_feature`, `get_feature_files`, `get_hotspots`,
  `get_feature_owners`, `get_flow_files`, `get_repo_summary`.
- Every response includes `_savings_metadata` estimating tokens saved
  vs a naive grep-and-read-files workflow — enables a real-time ROI
  dashboard.
- Install: `pip install 'faultlines[mcp]'`, add one config line to
  `~/.cursor/mcp.json`.
- Optional dependency group: `mcp = ["mcp>=1.0.0"]`.

### Tests
- 22 new MCP server tests. 543 total passing.

## [0.2.1] — 2026-04-10

### Fixed
- **Ollama workspace sub-splitting**. `detect_features_ollama` was
  receiving `ollama_url=` from the workspace dispatch, but the function
  signature expects `host=`. Every per-package call fell back to a
  single feature, bypassing sub-split. Now each package > 30 files
  gets proper sub-feature detection under Ollama.

## [0.2.0] — 2026-04-10

### Added
- **Per-feature and per-flow test coverage tracking**.
- Support for multiple coverage report formats with auto-detection:
  - Python `coverage.py` (`.coverage` SQLite, `coverage.json`).
  - Cobertura XML (`coverage.xml`).
  - Jest / NYC (`coverage-summary.json`).
  - LCOV (`lcov.info`, `coverage.lcov`).
- CLI flag: `--coverage [path]` with auto-detect when omitted.
- Coverage is applied to both features and flows in the CLI output and
  persisted in the JSON.
- Terminal report colors: green ≥ 70 %, yellow 40–70 %, red < 40 %.
- Flexible path matching (handles different repo prefixes).
- pytest coverage config wired with a 65 % threshold and HTML report
  generation.

### Tests
- 5 new tests for the Python, Cobertura, and Jest coverage parsers.

## [0.1.0] — 2026-04-10

### Added
- **Initial public release.** CLI tool that analyses a Git repository
  to detect features automatically from history and score them by
  bug-fix ratio, churn, and bus factor. No Jira required.
- Key capabilities:
  - Heuristic feature detection (free) and LLM-powered semantic
    detection (Sonnet / Ollama).
  - User-facing flow detection (`--flows`).
  - Monorepo workspace support (pnpm, npm, yarn, Turborepo, Nx, Lerna,
    Cargo, Go).
  - Library vs application classification (flows suppressed for
    libraries by default).
  - Deterministic ordering and reproducible output.
- Ships with 516 tests.
