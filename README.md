# faultlines

> **Map features in any codebase from git history alone. No Jira required.**

Point `faultlines` at a git repo and get back a feature map — which parts of your product accumulate the most bug fixes, who owns what, and where the risk is hiding. Works on any language, any size, any repo.

[![PyPI](https://img.shields.io/pypi/v/faultlines)](https://pypi.org/project/faultlines/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/faultlines)](https://pypi.org/project/faultlines/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/PashaSchool/featuremap)](https://github.com/PashaSchool/featuremap)

## Try it in 30 seconds

```bash
pip install faultlines
export ANTHROPIC_API_KEY=sk-ant-...   # or use Ollama for free local analysis
faultlines analyze .
```

That's it. You'll see something like:

```
╭──────────────────────────── FeatureMap Analysis ─────────────────────────────╮
│ Repository: /home/you/your-project                                           │
│ Features found: 22     Bug fix commits: 125     Health: 60.2/100             │
╰──────────────────────────────────────────────────────────────────────────────╯

                                Features by Risk
╭─────┬────────────────────────┬────────┬─────────┬───────────┬────────────────╮
│     │ Feature                │ Health │ Commits │ Bug Fixes │ Bug %          │
├─────┼────────────────────────┼────────┼─────────┼───────────┼────────────────┤
│ ✗   │ payments               │   23   │     112 │        38 │ 33.9%          │
│ ✗   │ booking-engine         │   29   │     487 │       151 │ 31.0%          │
│ !   │ auth                   │   54   │      48 │        12 │ 25.0%          │
│ ✓   │ dashboard              │   83   │      67 │         2 │  3.0%          │
╰─────┴────────────────────────┴────────┴─────────┴───────────┴────────────────╯
```

## What it does

1. **Reads your git history** — commits, blame, file changes over the last year
2. **Clusters files into features** — using Claude Sonnet (or local Ollama) to group by business domain, not folder names
3. **Scores each feature** — by bug-fix density, churn, bus factor, and age-weighted trends
4. **Detects user flows** — with `--flows`, breaks features into end-to-end user journeys (`checkout-flow`, `signup-flow`, `manage-team-flow`)
5. **Outputs JSON** — `~/.faultlines/feature-map-*.json`, ready for dashboards or CI

### Without LLM vs with LLM

```
Without --llm:  "components", "views", "hooks"           ← technical folders
With --llm:     "payments", "booking-engine", "auth"     ← business features
```

The LLM reads your file tree and commit messages to produce names an engineering manager would actually use.

## Tested on real OSS repos

Every number below is from a real `faultlines analyze --llm --flows` run. Reproduce any of them yourself.

| Repo | Files | Features | Flows | Time | What it found |
|------|------:|:--------:|:-----:|:----:|---------------|
| [cal.com](https://github.com/calcom/cal.com) | 10,463 | 282 | 725 | 23m | trpc/viewer, web/bookings, ee/billing, web/settings |
| [plane](https://github.com/makeplane/plane) | 4,932 | 134 | 408 | 12m | web/issues, editor/extensions, web/workspace, web/pages |
| [Ghost](https://github.com/TryGhost/Ghost) | 6,898 | 101 | 281 | 14m | admin-x, ghost/members, ghost/email, stats |
| [outline](https://github.com/outline/outline) | 2,390 | 22 | 188 | 6m | rich-text-editor, api-backend, dashboard, plugins |
| [documenso](https://github.com/documenso/documenso) | 2,530 | 49 | 191 | 8m | trpc/envelope, remix/document-signing, ee/billing |
| [formbricks](https://github.com/formbricks/formbricks) | 3,316 | 33 | 136 | 8m | web/survey, web/organization, web/auth |
| [excalidraw](https://github.com/excalidraw/excalidraw) | 1,225 | 15 | 28 | 4m | excalidraw/shared-ui, excalidraw/data, renderer |
| [trpc](https://github.com/trpc/trpc) | 1,573 | 14 | 37 | 1m | server/core, client/links, openapi, next-adapter |
| [gin](https://github.com/gin-gonic/gin) | 130 | 22 | — | 15s | binding, render, context, recovery, logger |
| [fastapi](https://github.com/fastapi/fastapi) | 2,981 | 14 | — | 80s | routing, dependencies, security, openapi, middleware |

Libraries (gin, fastapi) are auto-detected and show modules instead of business features. Flows are suppressed for libraries since they don't have end-user journeys.

## Installation

```bash
pip install faultlines
```

Requires Python 3.11+.

### With Ollama (local, free, private)

```bash
pip install 'faultlines[ollama]'
```

## Usage

### Basic (heuristic, no API key needed)

```bash
faultlines analyze .
faultlines analyze ./path/to/repo
faultlines analyze . --days 90 --top 5
```

### AI-powered (recommended)

```bash
# Claude (cloud) — best quality
export ANTHROPIC_API_KEY=sk-ant-...
faultlines analyze . --llm

# With flow detection
faultlines analyze . --llm --flows

# Focus on source directory
faultlines analyze . --llm --flows --src src/

# Ollama (local, free)
ollama pull llama3.1:8b
faultlines analyze . --llm --provider ollama
```

Get your API key at [console.anthropic.com](https://console.anthropic.com).

### Monorepo support

Automatically detects and analyses workspace packages:

- pnpm (`pnpm-workspace.yaml`)
- npm/yarn (`package.json` workspaces)
- Turborepo, Nx, Lerna
- Cargo workspaces, Go workspaces

Large monorepos are scanned per-package in parallel (4 workers).

## Flow detection

With `--flows`, each feature is broken into user-facing flows — named action sequences like `checkout-flow` or `manage-team-flow`.

```
╭──────┬───────────────────────────┬────────┬─────────┬───────────╮
│      │ Feature / Flow            │ Health │ Commits │ Bug Fixes │
├──────┼───────────────────────────┼────────┼─────────┼───────────┤
│  ✗   │ web/settings              │   23   │     206 │       144 │
│      │   ├─ manage-org-roles     │   11   │      37 │        30 │
│      │   ├─ manage-oauth-clients │   15   │      13 │        10 │
│      │   └─ manage-billing       │   50   │      24 │        12 │
│  ✓   │ auth/server               │   87   │      18 │         6 │
│      │   └─ login-flow           │   87   │      18 │         6 │
╰──────┴───────────────────────────┴────────┴─────────┴───────────╯
```

Flow names and health scores are real — from actual Haiku detection against git commit history.

## Output format

Results save to `~/.faultlines/feature-map-{repo}-{timestamp}.json`:

```json
{
  "repo_path": "/path/to/repo",
  "remote_url": "https://github.com/org/repo",
  "features": [
    {
      "name": "payments",
      "description": "Stripe payment processing and subscription billing",
      "health_score": 23.0,
      "bug_fix_ratio": 0.339,
      "total_commits": 112,
      "bug_fixes": 38,
      "authors": ["alice", "bob"],
      "paths": ["src/payments/stripe.py", "src/payments/webhooks.py"],
      "flows": [
        {
          "name": "checkout-flow",
          "health_score": 18.0,
          "total_commits": 67,
          "bug_fixes": 28,
          "bus_factor": 1,
          "hotspot_files": ["src/payments/charge.ts"]
        }
      ]
    }
  ]
}
```

## Health score

The health score (0–100) uses an **age-weighted sigmoid** based on bug-fix ratio:

| Bug fix % | Health | Status |
|:---------:|:------:|--------|
| 0–20% | 85–99 | ✓ Healthy — mostly feature work |
| 20–40% | 55–85 | ✓ Normal active development |
| 40–55% | 30–55 | ! Elevated — worth watching |
| 55–75% | 10–30 | ✗ High debt — maintenance-dominant |
| 75%+ | 0–10 | ✗ Critical — almost all bug fixes |

Recent bugs (< 30 days) weigh 2x more than older ones, so features that are actively getting worse score lower.

## CLI reference

```
faultlines analyze [REPO_PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--llm` | off | AI-powered semantic feature detection |
| `--flows` | off | Detect user-facing flows (requires `--llm`) |
| `--provider` | `anthropic` | LLM provider: `anthropic` or `ollama` |
| `--model` | auto | Model override |
| `--api-key` | env | Anthropic API key |
| `--src` | — | Focus on subdirectory (e.g. `src/`) |
| `--days` | `365` | Days of git history |
| `--top` | `3` | Top risk zones to highlight |
| `--output` | `~/.faultlines/` | Output file path |
| `--no-save` | — | Don't save JSON |
| `--legacy` | — | Use pre-rewrite 5-strategy pipeline |
| `--ollama-url` | `localhost:11434` | Custom Ollama URL |

## How it works

1. `git log` + `git blame` → commit history with file attribution
2. Heuristic candidates from directory structure and import graph
3. Claude Sonnet (or Ollama) merges/renames/splits candidates into business features via an operations-based prompt
4. Per-feature health scoring with sigmoid-weighted bug-fix ratio
5. Optional: Haiku detects user-facing flows per feature from signatures and commit patterns
6. JSON output + terminal report

For monorepos, step 3 runs per-package in parallel (ThreadPoolExecutor, 4 workers).

## Cost

| Repo size | Estimated LLM cost | Time |
|-----------|:------------------:|:----:|
| < 500 files | ~$0.01–0.05 | 15–30s |
| 500–2,000 | ~$0.10–0.30 | 1–5m |
| 2,000–5,000 | ~$0.30–0.70 | 5–12m |
| 5,000–10,000 | ~$0.70–2.00 | 10–25m |

Ollama is free (runs locally). Heuristic mode (no `--llm`) is free and instant.

## Who is this for

- **Engineering managers** — see where technical debt actually lives, not where you think it lives
- **Tech leads** — prioritise refactoring with data, not gut feeling
- **New team members** — understand which parts of the codebase need the most care
- **Sprint planning** — make "should we refactor X?" a data conversation

## Contributing

Issues, PRs, and feedback welcome at [github.com/PashaSchool/featuremap](https://github.com/PashaSchool/featuremap).

## License

[MIT](LICENSE)
