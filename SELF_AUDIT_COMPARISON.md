# Self-Audit vs Engine Output

> Ground truth: `SELF_AUDIT.md` (my manual feature list, written
> before reading the scan).
> Engine output: `/tmp/featuremap-self-scan.json` (full Sprint 1-7
> stack).

---

## Headline numbers

| metric | value |
|---|---|
| my expected features | ~20 |
| engine detected features | **3** |
| LLM calls | **2** |
| cost | **$0.04** |
| total source files | 143 |
| files in `shared-infra` (catch-all) | 95 |
| flows detected | 0 |

**The engine massively underdelivered on this repo.** Recall ~5%
(only `benchmark` + `documentation` matched anything in my list;
`shared-infra` doesn't count as a real feature).

---

## What the engine produced

```
95  shared-infra      (Shared Infra)
 5  benchmark         (Benchmark)
 1  documentation     (Documentation)
```

That's it. 92 source files I'd expect to land in 15-20 distinct
features were folded into the catch-all bucket.

---

## Root cause: library mode misfires on a Python CLI

```
Library detected (pyproject.toml/setup.py present with no
main.py/app.py/manage.py at repo root) — flows will be suppressed
```

Faultline IS a CLI (entry-point: `faultline.cli:app` registered as
a `pyproject.toml` console script), but the heuristic in
`repo_classifier.detect_library` only checks for `main.py` /
`app.py` / `manage.py` at the **repo root**. It never looks at
`pyproject.toml` `[project.scripts]` for actual console entry
points.

Result:
1. `is_library=True` propagates everywhere.
2. Sprint 4 flow detection is suppressed.
3. The single-call `deep_scan` library prompt returns one
   conservative feature ("benchmark") covering 5 files; the LLM
   declines to split the rest.
4. Pipeline orphan-fallback dumps the remaining 92 source files
   into `shared-infra`.
5. Sub-decompose / dedup / critique never see anything to work on
   because there are no >200-file features.
6. Trace-flows has nothing to trace because there are no flows.

This is a **detect_library false positive** — the heuristic is too
narrow. The fix: also check `pyproject.toml`'s
`[project.scripts]` / `[tool.poetry.scripts]` keys; if any
console script is registered, treat as application.

---

## Recall by my expected list

| my expected feature | matched in engine output? |
|---|---|
| 1. Detection Pipeline Core | ❌ folded into shared-infra |
| 2. Tool-Augmented Detection (Sprint 1) | ❌ shared-infra |
| 3. Cross-Cluster Dedup (Sprint 2) | ❌ shared-infra |
| 4. Sub-Decomposition (Sprint 3) | ❌ shared-infra |
| 5. Tool-Augmented Flow Detection (Sprint 4) | ❌ shared-infra |
| 6. Self-Critique Loop (Sprint 5) | ❌ shared-infra |
| 7. Call-Graph Flow Tracing (Sprint 7) | ❌ shared-infra |
| 8. Repo Config & Aliasing | ❌ shared-infra |
| 9. Display Names / Humanization | ❌ shared-infra |
| 10. Cost Tracking | ❌ shared-infra |
| 11. Output Writer + Reporter | ❌ shared-infra |
| 12. Coverage Integration | ❌ shared-infra |
| 13. AST + Import Graph | ❌ shared-infra |
| 14. Symbol-Level Attribution | ❌ shared-infra |
| 15. Incremental Scan | ❌ shared-infra |
| 16. **Benchmark Harness** | ✅ `benchmark` (5 files) |
| 17. MCP Server | ❌ shared-infra |
| 18. Analytics Integration | ❌ shared-infra |
| 19. Cloud Sync (Push) | ❌ shared-infra |
| 20. Legacy Fallback Path | ❌ shared-infra |

**Recall: 1/20 = 5%.**

---

## What actually went well (despite the disaster)

- **No crash** — the pipeline ran clean to completion with no
  fallbacks or errors.
- **`benchmark` feature got detected** with a clean Title Case
  display name and a sensible description. The 5 files
  attributed are exactly the ones in `faultline/benchmark/`.
- **Cost is honest** — $0.04 reflects that almost no real work
  happened; we didn't pay for a fake good output.

---

## Lessons from this validation

This is the highest-value test we've run because it surfaces a
real engine bug on a codebase I have ground truth for. Three
take-aways:

### 1. `detect_library` is too narrow
**Bug:** treats every Python project with `pyproject.toml` as a
library. Misses CLI tools, FastAPI apps that don't use `app.py`
naming, Django apps in subdirs, and so on.

**Fix:** add console-script detection. ~10 lines in
`analyzer/repo_classifier.py`. Should land before any real
production claim.

### 2. Single-call library mode collapses too aggressively
Even if `is_library=True` were correct, the library prompt's
"prefer fewer, bigger features" guidance is pulling the LLM
toward 1-3 features for a 100+ source-file repo. The prompt
needs a minimum-feature-count guidance for libraries too —
maybe "1-15 features depending on package boundaries".

### 3. Workspace path always wins on monorepos
The four other repos we tested (formbricks, documenso,
trpc, fastapi) all triggered `deep_scan_workspace` because they
have `packages/` directories. Faultline doesn't — it's a single
Python package — so it took the single-call path. That path
hasn't been stress-tested as much. Worth running against more
single-package Python/Go repos before claiming generality.

---

## Action items

| # | item | effort | API |
|---|---|---|---|
| 1 | Fix `detect_library` to honour `pyproject.toml` console scripts | 30 min | none |
| 2 | Re-run featuremap self-scan after the fix | 15 min | ~$1-2 |
| 3 | Tune library-mode prompt for minimum granularity | 1 hour | ~$2-3 to validate |
| 4 | Add a single-package-Python repo (like Flask, click, rich) to benchmark suite for ongoing stress-testing | 1 hour | ~$3-5 |

After (1) + (2) the recall on featuremap should land in the 60-80%
range that the documenso and formbricks runs hit.

---

## v2 results — after the detect_library fix

Re-scanned with the new console-script-aware classifier
(commit pending). Major recovery.

| metric | v1 | **v2** | docunenso |
|---|---|---|---|
| is_library | True (false positive) | **False** ✓ | False |
| features | 3 | **7** | 35 |
| flows total | 0 | **26** | 114 |
| flows grounded | 0 | **12 (46%)** | 114 (100%) |
| flows with participants | 0 | **12** | 114 |
| cost | $0.04 | **$0.14** | $3.43 |
| my expected recall | 5% | **~50%** | n/a |

### What v2 produced

```
47 files  Analysis Core
21        Shared Infra
13        LLM Pipeline       ← absorbs 6 of my sprints
13        Feature Detection
 3        Digest
 3        Output Reporting
 1        Documentation
```

### Remaining limitations

Two subtle issues showed up clearly in v2:

**Issue A — single-call path is too consolidating.**
Faultline-the-repo doesn't have a `packages/` directory, so the
pipeline takes the single-call `deep_scan` route. That code path
asks the LLM to emit a small handful of features over the whole
repo. On documenso the workspace path produces ~30-40 features;
on faultline the single-call path produces 7. Six of my discrete
sprints (Sprint 1 tool-use, Sprint 2 dedup, Sprint 3 sub-decompose,
Sprint 4 flow detection, Sprint 5 critique, Sprint 7 trace)
collapsed into a single `LLM Pipeline` feature. Sub-decompose
didn't fire because the largest feature (47 files) is below the
200-file threshold.

**Issue B — small-repo prompt guidance is missing.**
The single-call prompt encourages "1-12 features depending on
size" but doesn't have a "below-50-files repos still benefit
from 8-15 features" hint. Result is conservative collapse.

### Recommended fixes

| # | item | effort | API |
|---|---|---|---|
| 1 | Lower sub-decompose threshold for small repos (default 200, but `min(200, total_files / 4)` for repos under 200 files) | 30 min | $0.50 retest |
| 2 | Tighten single-call prompt: "even on small repos, prefer 8-15 features over 3-5 when source files exceed 30" | 15 min | $0.50 retest |
| 3 | Document these as known limitations in CLAUDE.md until calibrated | 5 min | none |

After (1) + (2) faultline-self-scan should hit 70%+ recall.

### What this means for the product

The detect_library fix unlocked everything. On a real repo
(documenso) we already see 30-40 features and 100% flow grounding.
On a small Python single-package CLI (faultline itself) we see
the single-call consolidation issue clearly because there's no
workspace structure to drive granularity.

For external Python single-package projects (Flask, click, rich,
typer apps) we should expect the same shape until issue A and B
are addressed. Worth flagging openly to early users.

The `Benchmark Harness` regression (visible as a feature in v1,
hidden in v2) is the same Issue A: in v2 it lives inside one of
the bigger buckets and doesn't earn its own row because no one
package boundary or directory split made it stand out.
