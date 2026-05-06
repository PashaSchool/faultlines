# Sprint 16 — Eval Harness + dev→main CI Gate

**Branch:** `feat/sprint16-eval-harness` (нова, з main після merge S15)
**Goal:** **production-grade eval system** — кожен PR з dev/feat/* → main блокується якщо accuracy/coverage впали відносно main baseline. Failure mode taxonomy дає кожен fail-case ім'я та категорію — реальна продуктова якість, не vibes.

## Чому це окремий sprint

Поточний `eval_flow_attribution.py` має circularity bug: truth.yaml автогенерований з того самого classifier якого тестує. Реальної ground truth немає. Без неї всі цифри "98.8%" частково self-confirm. Sprint 16 розриває це.

CI gate додатково:
- Не дає тихо проштовхнути регресію
- Дає automation story для interview ("кожен PR проходить eval, блокується якщо coverage падає >2pp")
- Робить landing's `/eval` endpoint реальним (живі цифри, не mock)

## Скоуп

✅ **In scope**
- Manual ground truth для 5 OSS repos (axios + immich + dify + supabase + fastapi).
- LLM-as-judge module (Haiku, temperature=0, cache).
- Failure mode taxonomy (6 classes).
- Two-tier GitHub Action:
  - **Smoke** (always): unit tests + offline replay → free, fast, blocks merge.
  - **Eval** (opt-in via PR label `eval` or workflow_dispatch): full LLM eval, posts metrics delta + failure breakdown to PR.
- README + landing `/eval` page оновлено з живими числами.

❌ **Out of scope**
- Re-eval Sprint 15 fixes (вже закомічено, S17 окремо validation на dify/supabase).
- Per-PR full LLM scan (занадто дорого, ~$2 кожен PR; eval — opt-in).
- New benchmark repos — 5 досить для статистики.

## Архітектура

```
tests/eval/
  __init__.py
  ground_truth.json          # 5 repos × ~20 features each, manually curated
  judge.py                   # LLM-as-judge for coverage / precision
  failure_modes.py           # 6-class taxonomy + classifier
  test_eval_smoke.py         # offline tests (S15 backfill replay, etc.)
  test_eval_llm.py           # LLM-driven, gated by ANTHROPIC_API_KEY env

.github/workflows/
  ci.yml                     # existing — keep
  eval-smoke.yml             # NEW — runs on every PR to main, free
  eval-llm.yml               # NEW — runs on `eval` label / dispatch, costs

scripts/
  eval_run.py                # CLI: load truth + scan, compute metrics
  eval_pr_comment.py         # builds PR comment markdown from results
```

## Day-by-day

### Day 1 — Ground truth + smoke harness

`tests/eval/ground_truth.json`:

```json
{
  "axios": {
    "expected_features": [
      "http-client", "interceptors", "cancel-tokens",
      "request-transforms", "adapters", "defaults", "error-handling"
    ],
    "expected_flows": [
      "intercept-request", "intercept-response", "cancel-pending-request"
    ],
    "source": "https://github.com/axios/axios#features",
    "annotated_by": "pkuzina",
    "annotated_at": "2026-05-07"
  },
  "immich": { ... },
  "dify": { ... },
  "supabase": { ... },
  "fastapi": { ... }
}
```

**Annotation work:** ~3 hours. Read each repo's README + top docs, list 15-25 features that the maintainer would name. NOT файлові директорії. Reference URLs так перевіряючий може verify.

`tests/eval/test_eval_smoke.py`:

- Replay S15's debug_s15_v2.py logic as a unit test.
- Loads frozen `assignments-dify.json` snapshot, runs full post_process, asserts auth=24 paths survive.
- Catches future regressions of the commit_prefix_enrichment bug.
- No LLM, no API key needed.

### Day 2 — LLM-as-judge

`tests/eval/judge.py`:

```python
JUDGE_SYSTEM = """You evaluate a feature detection tool against
ground-truth feature lists curated by the repo's maintainers.

For each EXPECTED feature, find the BEST matching DETECTED feature
(or "NONE" if missing). Use semantic matching:
  - "http-client" matches "request-handling" or "axios-core"
  - "interceptors" matches "request-interceptors" but NOT "interceptor"

Return JSON with match quality:
  exact   — names + scope identical
  partial — overlapping but ambiguous
  none    — no equivalent detected

OUTPUT: {"matches": [{"expected": "...", "detected": "...|NONE",
                     "quality": "exact|partial|none"}]}"""

def judge_run(repo: str, expected: list[str], detected: list[str],
              client) -> JudgeResult:
    response = client.messages.create(
        model="claude-haiku-4-5",
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": json.dumps({
            "expected": expected, "detected": detected,
        })}],
        max_tokens=2048,
    )
    matches = parse_response(response)
    coverage = sum(1 for m in matches if m.quality != "none") / len(expected)
    precision = sum(1 for m in matches if m.quality != "none") / len(detected)
    return JudgeResult(coverage, precision, matches)
```

Cache verdicts by `hash(expected + detected)` so re-runs of the same
state are free.

`tests/eval/test_eval_llm.py`:

- Pytest tests with `@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"))`.
- Per-repo: load saved feature-map JSON + ground truth → run judge → assert thresholds:
  - `coverage ≥ 80%` per repo
  - `coverage ≥ 85%` aggregate
- Failures upload `eval-results.json` artifact.

### Day 3 — GitHub Actions

`.github/workflows/eval-smoke.yml`:

```yaml
name: eval-smoke

on:
  pull_request:
    branches: [main]

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e '.[dev]'
      - run: pytest tests/eval/test_eval_smoke.py -v
      - run: pytest tests/ --ignore=tests/eval -v
```

`.github/workflows/eval-llm.yml`:

```yaml
name: eval-llm

on:
  pull_request:
    branches: [main]
    types: [labeled]
  workflow_dispatch:

jobs:
  llm-eval:
    if: ${{ github.event.label.name == 'eval' || github.event_name == 'workflow_dispatch' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e '.[dev]'
      - run: python scripts/eval_run.py --repos axios immich dify supabase fastapi
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: .faultline/eval-results.json
      - uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const r = JSON.parse(fs.readFileSync('.faultline/eval-results.json'));
            const md = require('./scripts/eval_pr_comment.js')(r);
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: md
            });
```

PR comment shape:

```
### 📊 Eval Results

| repo     | coverage | precision | Δ vs main |
|----------|---------:|----------:|----------:|
| axios    |    87.5% |     92.1% |    +1.2pp |
| immich   |    91.0% |     88.4% |    +0.5pp |
| dify     |    96.5% |     94.2% |    -2.3pp ⚠ |
| supabase |    97.3% |     95.8% |    +0.0pp |
| fastapi  |    89.0% |     90.0% |    +0.0pp |

### Failure Modes (5 misses on dify)
- 3× OVER_CLUSTERING (auth + signin merged into i18n)
- 1× UNDER_CLUSTERING (workflow split into 5 pieces)
- 1× HALLUCINATED (no files match)

[full report ↓](artifact)
```

### Day 4 — Failure mode taxonomy

`tests/eval/failure_modes.py`:

```python
class FailureMode(Enum):
    OVER_CLUSTERING = "Two distinct features merged into one"
    UNDER_CLUSTERING = "One feature split into multiple"
    GENERIC_NAMING = "Returned src/utils-style folder name"
    HALLUCINATED = "Feature doesn't correspond to any files"
    MISSING_KEY = "Major README feature not detected"
    LANGUAGE_BIAS = "Degraded behavior on Go/Rust vs TS/Python"

def classify_miss(expected: str, detected: str | None,
                  detected_features: list[str], paths: list[str],
                  client) -> FailureMode:
    """Single Haiku call: given the miss context, pick the failure
    mode. Confidence ≥4 required, else mark UNCATEGORIZED."""
    ...
```

Each FAIL-case in eval gets classified. Breakdown surfaces in PR
comment + `/eval` page.

## Метрики цілі (Sprint 16 acceptance)

| Item | Target |
|---|---|
| Manual ground truth coverage | ≥ 5 repos, ≥ 80 features total |
| LLM-as-judge match accuracy | tested by humans on 30-sample manual review, ≥ 85% agreement with judge verdict |
| Smoke tier runtime | ≤ 90s (offline) |
| Eval tier runtime | ≤ 8 min (5 repos × small benchmark) |
| Eval tier cost | ≤ $1.50 per run |
| Failure taxonomy coverage | ≥ 6 modes, ≥ 90% of misses classified |
| README updated with live numbers | yes |

## Risks

1. **Manual annotation subjective.** Mitigation: cross-check against repo's official "Features" section in README. Document sources in `ground_truth.json`. Allow disputes via GitHub issue.

2. **LLM-judge variance.** Mitigation: temperature=0, deterministic params, cache. Run judge 3× on first 30 samples → measure variance, accept if <5%.

3. **CI cost creep.** Mitigation: opt-in label gating means full eval runs only when human asks. Default smoke tier costs $0.

4. **Failure taxonomy drift.** Mitigation: lock taxonomy in `failure_modes.py` constants. Add new modes via PR review only.

## Connection to S15 verdict

Day 1's smoke test directly reproduces the S15 commit_prefix_enrichment
bug fix using `scripts/debug_s15_v2.py` logic. If a future change
re-introduces the bug, smoke tier blocks the PR. This makes the
S15 fix permanently regression-protected.

## Connection to landing

After Sprint 16 ships:
- `faultlines.dev/eval` shows live coverage / precision per repo
- README has the production-grade abzac:
  > "Coverage: 89% across 5 OSS repos (auto-evaluated nightly via
  >  GitHub Actions). Precision: 91%. Reproducible: pytest tests/eval/."

This unlocks the AI Product Engineer interview narrative —
production-grade eval, not vibes.
