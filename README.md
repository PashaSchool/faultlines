# faultline

> **Map features in any codebase from git history. No Jira required.**

**faultline** analyses your git commit history to automatically detect features and modules in your codebase, then shows which ones are accumulating the most bug fixes — your technical debt hotspots.

No integrations. No configuration. Just point it at any git repo.

[![PyPI](https://img.shields.io/pypi/v/faultline)](https://pypi.org/project/faultline/)
[![Python](https://img.shields.io/pypi/pyversions/faultline)](https://pypi.org/project/faultline/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Why faultline?

Engineering managers need to know *where* the technical debt is before they can act on it. Most tools require you to tag tickets in Jira or Linear. faultline reads the truth directly from your git history.

```
✗ payments    — health: 23   38 bug fixes / 112 commits (33.9%)
!  auth        — health: 54   12 bug fixes / 48 commits  (25.0%)
✓  dashboard   — health: 91    2 bug fixes / 67 commits   (3.0%)
```

---

## Installation

```bash
pip install faultline
```

Requires Python 3.11+.

### With Ollama support (local LLM)

```bash
pip install 'faultline[ollama]'
```

---

## Quick Start

```bash
# Analyse the current directory
faultline analyze .

# Analyse a specific repo
faultline analyze ./path/to/repo

# Focus on a source subdirectory (recommended for frontend/monorepo projects)
faultline analyze . --src src/

# Last 90 days, show top 5 risk zones
faultline analyze . --days 90 --top 5

# Just print, don't save
faultline analyze . --no-save

# AI-powered semantic feature detection
faultline analyze . --llm --src src/

# AI feature detection with flow breakdown
faultline analyze . --llm --flows --src src/
```

---

## AI-Powered Feature Detection

By default faultline groups files by directory structure (fast, no API needed). With `--llm` enabled, Claude or a local Ollama model reads the full file tree and returns a **semantic feature map** — grouping files by business domain, not folder names.

```
Without --llm:  "components", "views", "hooks"        ← technical layers
With --llm:     "user-auth", "payments", "dashboard"  ← business features
```

### Anthropic Claude (cloud)

```bash
# Pass your API key directly
faultline analyze . --llm --api-key sk-ant-...

# Or use an environment variable
export ANTHROPIC_API_KEY=sk-ant-...
faultline analyze . --llm

# Recommended for large projects — focus on source files only
faultline analyze . --llm --src src/
```

Get your API key at [console.anthropic.com](https://console.anthropic.com) → API Keys.
Uses **Claude Haiku** for cost-efficient analysis (~$0.001 per repo).

### Ollama (local, free, private)

Run analysis entirely on your machine — no API key, no data leaves your computer.
Recommended for private repositories.

```bash
# 1. Install Ollama
brew install ollama        # macOS
# or: curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the recommended model (one time, ~4.7 GB)
ollama pull llama3.1:8b

# 3. Start the server
ollama serve

# 4. Install the ollama package
pip install 'faultline[ollama]'

# 5. Run
faultline analyze . --llm --provider ollama --src src/
```

### Recommended models

| Model | Size | Semantic quality | Notes |
|-------|------|-----------------|-------|
| `llama3.1:8b` | 4.7 GB | ★★★★★ | **Default — best balance of quality and size** |
| `mistral-nemo:12b` | 7.1 GB | ★★★★★ | Best overall quality, higher RAM requirement |
| `qwen2.5:7b` | 4.7 GB | ★★★★ | Good alternative |
| `llama3.2:3b` | 2.0 GB | ★★★ | Use when RAM is limited |

```bash
# Best quality (requires ~8 GB RAM)
ollama pull mistral-nemo:12b
faultline analyze . --llm --provider ollama --model mistral-nemo:12b

# Lightweight option
ollama pull llama3.2:3b
faultline analyze . --llm --provider ollama --model llama3.2:3b
```

The API key is validated **before** the analysis starts — no waiting to discover a bad key.
Falls back silently to heuristic mode if the LLM is unavailable.

---

## Flow Detection

With `--flows`, faultline goes one level deeper — breaking each feature into **user-facing flows**: named end-to-end sequences of actions (e.g. `checkout-flow`, `refund-flow`).

Requires `--llm`. Works with both Anthropic and Ollama.

```bash
faultline analyze . --llm --flows --src src/
faultline analyze . --llm --provider ollama --flows --src src/
```

Each flow includes:
- **Health score** with trend detection (improving / degrading)
- **Bus factor** — flags single-contributor flows
- **Hotspot files** — source files with >40% bug fix ratio
- **Test file count** — associated test coverage
- **Bug fix PRs** — linked pull requests that fixed bugs
- **Weekly timeline** — commit activity per ISO week

Output shows two tables: features overview, then features with nested flows:

```
                         Features by Risk
╭──────────────┬────────┬─────────┬───────────┬────────╮
│ Feature      │ Health │ Commits │ Bug Fixes │  Bug % │
├──────────────┼────────┼─────────┼───────────┼────────┤
│ payments     │   23   │     112 │        38 │ 33.9%  │
│ auth         │   54   │      48 │        12 │ 25.0%  │
╰──────────────┴────────┴─────────┴───────────┴────────╯

                    Feature & Flow Map by Risk
╭──────┬────────────────────────┬────────┬─────────┬───────────╮
│      │ Feature / Flow         │ Health │ Commits │ Bug Fixes │
├──────┼────────────────────────┼────────┼─────────┼───────────┤
│  ✗   │ payments               │   23   │     112 │        38 │
│      │   ├─ checkout-flow     │   18   │      67 │        28 │
│      │   └─ refund-flow       │   41   │      45 │        10 │
│  !   │ auth                   │   54   │      48 │        12 │
│      │   └─ login-flow        │   54   │      48 │        12 │
╰──────┴────────────────────────┴────────┴─────────┴───────────╯
```

---

## Focusing on a Subdirectory

Use `--src` to restrict analysis to a specific folder. Everything outside that path is ignored.

```bash
# Sources in src/
faultline analyze . --src src/

# Sources in app/
faultline analyze . --src app/

# Monorepo — analyse one package
faultline analyze . --src packages/api/src/
```

Automatically excluded regardless of location:
- Dependencies: `node_modules/`, `vendor/`, `.venv/`
- Build output: `dist/`, `build/`, `.next/`, `coverage/`
- Tooling: `.github/`, `.husky/`, `.storybook/`, `.circleci/`
- Binary and generated files: `.map`, `.lock`, `.woff`, `.ttf`

---

## Output

### Terminal

```
╭──────────────────────────────────────────────╮
│             FeatureMap Analysis               │
│                                               │
│ Repository:           /path/to/repo           │
│ Analyzed:             last 365 days           │
│ Total commits:        847                     │
│ Features found:       12                      │
│ Bug fix commits:      143                     │
│ Average health score: 61.3/100                │
╰──────────────────────────────────────────────╯
```

### JSON

Results are saved to `~/.faultline/` with a timestamped filename:

```json
{
  "repo_path": "/path/to/repo",
  "remote_url": "https://github.com/org/repo",
  "analyzed_at": "2026-03-06T12:00:00Z",
  "total_commits": 847,
  "date_range_days": 365,
  "features": [
    {
      "name": "payments",
      "description": "Handles Stripe payment processing and subscription billing.",
      "health_score": 23.0,
      "bug_fix_ratio": 0.339,
      "bug_fixes": 38,
      "total_commits": 112,
      "authors": ["alice", "bob"],
      "paths": ["src/payments/stripe.py", "src/payments/webhooks.py"],
      "bug_fix_prs": [
        {
          "number": 142,
          "url": "https://github.com/org/repo/pull/142",
          "title": "fix: handle expired card retry",
          "author": "alice",
          "date": "2026-03-01T09:00:00Z"
        }
      ],
      "flows": [
        {
          "name": "checkout-flow",
          "health_score": 18.0,
          "bus_factor": 1,
          "hotspot_files": ["src/payments/stripe/charge.ts"],
          "health_trend": -0.12,
          "test_file_count": 3,
          "bug_fixes": 28,
          "total_commits": 67
        }
      ]
    }
  ]
}
```

Custom output path:

```bash
faultline analyze . --output ./reports/health.json
```

---

## How It Works

1. **Reads git history** — up to `--max-commits` commits within the requested date range
2. **Collects tracked files** — respects `--src` filter, skips build output and tooling
3. **Maps files to features** — multiple strategies:
   - **Directory heuristics** (default): groups by first meaningful directory (`src/payments/` → `payments`)
   - **Import graph clustering**: files connected by imports are grouped together (TS/JS)
   - **Co-change detection**: files that frequently change together in commits are grouped
   - **LLM semantic grouping** (`--llm`): sends the file tree to Claude or Ollama, gets back a `{feature: [files]}` mapping by business domain
4. **Scans commit history** — for each feature, counts total commits and bug fix commits
5. **Calculates health scores** — age-weighted bug fix ratio (recent bugs penalised 2x more than old ones), scaled by commit activity
6. **Detects flows** (`--flows`): analyses file signatures, co-change pairs, and route anchors within each feature to map user-facing journeys
7. **Assigns unassigned files** — files not claimed by any flow get assigned by directory proximity

### Bug fix detection

Commit messages are classified as bug fixes using pattern matching:

**Detected as bug fix:** `fix`, `bug`, `hotfix`, `patch`, `revert`, `regression`, `crash`, `error`, `broken`, `resolve`, `timeout`, `null pointer`, `race condition`, `deadlock`, `memory leak`, `overflow`

**Excluded (false positives):** `fix typo`, `fix lint`, `fix formatting`, `fix test`, `fix docs`, `fix ci`, `fix merge`, `fix import`

---

## Health Score

| Score | Status | Meaning |
|-------|--------|---------|
| 75–100 | ✓ Green | Healthy — low bug fix ratio |
| 50–74 | ! Yellow | Watch — moderate technical debt |
| 0–49 | ✗ Red | Critical — high bug fix ratio |

The health score uses **age-weighted bug fix ratio**: recent bugs (< 30 days) carry 2x weight, while older bugs decay to 0.5x. This means a feature that's actively accumulating bugs scores worse than one with historical bugs that have since stabilised.

---

## CLI Reference

### `faultline analyze [REPO_PATH]`

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | — | Subdirectory to focus on, e.g. `src/` or `app/` |
| `--days` | `365` | Days of git history to analyse |
| `--max-commits` | `5000` | Maximum commits to read |
| `--top` | `3` | Number of top risk zones to highlight |
| `--output` | `~/.faultline/` | Output file path |
| `--no-save` | — | Skip saving JSON output |
| `--llm` | `false` | Use AI for semantic feature detection |
| `--flows` | `false` | Detect user-facing flows within features (requires `--llm`) |
| `--provider` | `anthropic` | LLM provider: `anthropic` or `ollama` |
| `--model` | — | Model override (default: `claude-haiku-4-5` / `llama3.1:8b`) |
| `--api-key` | env | Anthropic API key (`ANTHROPIC_API_KEY` env var) |
| `--ollama-url` | `http://localhost:11434` | Custom Ollama server URL |

### `faultline version`

Prints the current version.

---

## Use Cases

- **Engineering managers**: identify which features carry the most technical debt
- **Sprint planning**: prioritise refactoring based on health scores, not gut feeling
- **Code reviews**: spot high-risk areas before they become incidents
- **CI integration**: track health scores over time, alert on degradation
- **New team members**: understand which parts of the codebase need the most care

---

## License

[MIT](LICENSE)
