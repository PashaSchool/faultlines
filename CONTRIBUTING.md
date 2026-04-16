# Contributing to Faultlines

## Dev setup

```bash
git clone https://github.com/PashaSchool/faultlines.git
cd faultlines

python -m venv .venv
source .venv/bin/activate

pip install -e '.[mcp,watch]'
pip install pytest pytest-cov pre-commit

pre-commit install
```

## Running tests

```bash
pytest tests/                                    # all tests
pytest tests/test_cache.py -v                    # one file
pytest tests/ -k "not test_reads_api_key_from"   # skip the known flaky test
```

Current target: **65% coverage** minimum. CI fails below that.

## Pre-commit hooks

The `.pre-commit-config.yaml` runs:

- `ruff` lint + format
- `gitleaks` secrets scan
- `detect-secrets` entropy-based scan
- Basic hygiene (trailing whitespace, EOF, yaml/toml validity, large files)

Hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

If a commit fails because of `detect-secrets`, regenerate the baseline:

```bash
detect-secrets scan --baseline .secrets.baseline \
  --exclude-files '^\.venv/|^\.git/|\.png$|\.webp$'
```

Review the diff carefully — don't add real secrets to the baseline.

## CI

Two GitHub Actions workflows run on push and PR:

- **CI** — tests on Python 3.11 and 3.12, ruff lint, wheel build.
- **Secrets scan** — gitleaks against the full git history.

## Releasing a new version

1. Bump `version` in `pyproject.toml` following SemVer.
2. Update `CHANGELOG.md` under a new version heading.
3. Commit: `git commit -m "chore: bump version to X.Y.Z"`.
4. Tag and push: `git tag vX.Y.Z && git push --tags`.
5. Build and publish:

   ```bash
   rm -rf dist/
   python -m build --no-isolation
   twine upload dist/*
   ```

## Architecture

Modules are intentionally isolated:

- `faultline/analyzer/` — git parsing, coverage, features detection
- `faultline/llm/` — Claude / Ollama clients and pipeline
- `faultline/symbols/` — symbol-level attribution (opt-in, `--symbols`)
- `faultline/impact/` — change impact prediction (used by MCP + CLI)
- `faultline/cache/` — incremental refresh, freshness tracking, orphan discovery
- `faultline/watch/` — file watcher daemon (opt-in, `watch` command)
- `faultline/mcp_server.py` — MCP server for AI agents
- `faultline/cli.py` — command entry points
- `faultline/output/` — terminal report + JSON writer
- `faultline/models/` — Pydantic types (the contract between modules)

New features should land in their own module and import from `analyzer/` +
`models/` only. Never import from `cli.py` or other feature modules.
