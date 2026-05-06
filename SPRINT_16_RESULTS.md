# Sprint 16 — Results (Eval Harness + dev → main CI Gate)

**Branch:** `feat/sprint16-eval-harness`
**Status:** All 4 days shipped. Smoke gate green. LLM eval workflow
ready (needs `ANTHROPIC_API_KEY` configured on the GitHub repo to
activate; falls back to graceful zero-score otherwise).
**Date:** 2026-05-06

## What shipped

| Day | Deliverable | Tests |
|-----|-------------|-------|
| 1 | `tests/eval/ground_truth.json` (5 repos × ~70 features + 30 flows) | 1 (smoke regression) |
| 1 | `tests/eval/test_eval_smoke.py` — guards against the S15 commit_prefix_enrichment bug forever | 1 |
| 1 | `.github/workflows/eval-smoke.yml` — always-on, free, blocks merge | n/a |
| 2 | `tests/eval/judge.py` — LLM-as-judge with caching, deterministic params | 19 |
| 3 | `scripts/eval_run.py` — CLI driving the judge across all 5 repos | n/a |
| 3 | `scripts/eval_pr_comment.py` — markdown formatter | n/a |
| 3 | `.github/workflows/eval-llm.yml` — opt-in via `eval` PR label or workflow_dispatch | n/a |
| 4 | `tests/eval/failure_modes.py` — 6-category taxonomy + LLM classifier | 13 |
| 4 | `eval_run.py` + `eval_pr_comment.py` updated to surface failure breakdown | n/a |

**Total new tests: 33, all green.**

## Architecture

```
PR opens → main branch
  │
  ├─ eval-smoke.yml  ─── always runs, FREE, ≤90s
  │   ├─ pytest tests/        (excludes tests/eval/)
  │   └─ pytest tests/eval/test_eval_smoke.py
  │       └─ replays S15 commit_prefix_enrichment regression
  │           on frozen fixture; auth must survive 24 paths
  │
  └─ eval-llm.yml  ─── opt-in via `eval` label / workflow_dispatch
      ├─ python scripts/eval_run.py
      │   ├─ load tests/eval/ground_truth.json
      │   ├─ pick latest feature-map JSON per repo
      │   ├─ judge_run() — LLM-as-judge for each repo
      │   ├─ classify_all() — failure mode for every miss
      │   └─ write .faultline/eval-results.json
      ├─ python scripts/eval_pr_comment.py → markdown
      └─ marocchino/sticky-pull-request-comment → posts to PR
```

## Sample PR comment (rendered offline, no LLM)

```markdown
### 📊 Faultlines Eval Results

**Aggregate** (across 5 repos): coverage **88%** · precision **91%** · F1 **89%**

| repo     | coverage | precision | F1    | Δ vs main |
|----------|---------:|----------:|------:|----------:|
| axios    |    87.5% |     92.1% | 89.7% | ✅ +1.2%   |
| immich   |    91.0% |     88.4% | 89.7% | ✅ +0.5%   |
| dify     |    96.5% |     94.2% | 95.3% | ⚠️ -2.3%   |
| supabase |    97.3% |     95.8% | 96.5% | — —       |
| fastapi  |    89.0% |     90.0% | 89.5% | — —       |

**Failure modes** (across all repos):
  - 5× Over Clustering
  - 3× Generic Naming
  - 2× Missing Key
  - 1× Hallucinated

<details><summary>Missing features (first 5 per repo, with mode tags)</summary>

**dify** missing:
  - `auth` → mapped to `i18n`  _[OVER_CLUSTERING]_
  - `marketplace` → not detected  _[MISSING_KEY]_
…
</details>
```

## Defences against fragile evals

1. **LLM-judge variance**: `temperature=0`, sorted-input cache key,
   stable hash output. Two consecutive runs produce byte-identical
   metric numbers from the same scan input.

2. **Hallucinated expected**: `judge_run` filters returned matches
   against the supplied `expected_set`. Anything the model invents
   gets dropped.

3. **Silent-drop fill-in**: if the judge skips an expected feature in
   its output, `judge_run` adds a `quality=none` placeholder so
   coverage denominator stays correct.

4. **Cache invalidation**: cache key is `hash(sorted(expected) +
   sorted(detected))`. Adding / renaming a feature busts the cache;
   pure scan re-runs hit it.

5. **CI cost predictability**: opt-in label means full LLM eval runs
   only when a human asks. Default smoke tier costs $0 forever.

6. **Graceful no-key fallback**: every LLM call returns a structured
   zero-score result on missing API key, ImportError, or network
   failure. CI never crashes; it just reports `coverage=0` so a
   human notices.

## Connection to S15

`tests/eval/test_eval_smoke.py::test_commit_prefix_enrichment_does_not_cannibalise_auth`
replays the exact bug that ate Sprint 15. If a future change to
`post_process.commit_prefix_enrichment_pass` or the
`_DOMAIN_PROTECTED_PREFIXES` / feature-token guard re-introduces
the cannibalisation, this test fails on the PR and merge is blocked.

The S15 fix is now permanently regression-protected.

## How to activate

The smoke gate is live the moment this branch merges:

```bash
git checkout main
git merge --no-ff feat/sprint16-eval-harness
git push
```

The LLM tier needs `ANTHROPIC_API_KEY` added to repo Secrets:

```
GitHub repo → Settings → Secrets and variables → Actions →
New repository secret: ANTHROPIC_API_KEY
```

Then opt-in by labelling a PR `eval` (or trigger manually:
Actions → eval-llm → Run workflow).

## Cost expectations

| Tier | Trigger | Cost / run | Latency |
|---|---|---|---|
| eval-smoke | every PR + push | $0 | ≤ 90s |
| eval-llm | `eval` label / dispatch | ~$0.20–1.50 | 2–8 min |

Per-repo LLM cost breakdown (when activated):
- judge_run: 1 Haiku call per repo (≤$0.05)
- classify_all: 1 Haiku call per miss (typically 5–15 calls per repo)
- All cached after first run; subsequent re-runs of same scan = $0.

## What's NOT in Sprint 16 (deferred)

- README rewrite with live numbers — needs a real LLM eval run first.
  Schedule when API key is added on the repo.
- `/eval` page on faultlines.dev — frontend work, separate sprint.
- Failure mode tuning — taxonomy locked at 6 modes; refinement comes
  after observing real misclassifications across 5–10 PR runs.
- Ground truth review — every repo entry has `needs_review: true`;
  schedule maintainer / co-reviewer pass when convenient.

## Verdict

**Merge S16.** All four days of work shipped tested. The smoke gate
is the most valuable single piece — it makes the S15 bug
permanently unfixable to re-introduce. The LLM tier is opt-in
specifically so cost stays bounded for solo / small-team setups.

Sprint 17 (next): live regression on supabase to validate Sprint 15
fixes are general (the original deferred scope before Sprint 16
took priority).

## How to verify locally

```bash
# Smoke gate (always runs)
source .venv/bin/activate
pytest tests/eval/test_eval_smoke.py -v

# All eval module tests
pytest tests/eval/ -q --no-cov

# Render a sample PR comment from a frozen scan
python scripts/eval_run.py --repos dify --json-out /tmp/eval.json
python scripts/eval_pr_comment.py /tmp/eval.json
```
