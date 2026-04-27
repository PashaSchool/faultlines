# Sprint 6 — Benchmark formalization

> **Roadmap context:** `SPRINTS_ROADMAP.md` (final sprint). Previous
> sprints landed all the detection improvements:
> Sprint 1 (`b178a7b`+) tool-augmented per-package detection,
> Sprint 2 (`f5cea51`+) cross-cluster dedup,
> Sprint 3 (`cbd6222`+) sub-decomposition,
> Sprint 4 tool-augmented flow detection,
> Sprint 5 self-critique loop.
>
> **Branch:** `feat/tool-use-detection`
> **Status:** Plan + Day 1 build (no API)
> **Estimated effort:** 3-5 focused days
> **Estimated API spend:** $20-40 one-time to seed scans

---

## 1. Why this sprint exists

Until now we've been comparing scans informally — counting
shared-infra files, eyeballing generic names, describing changes in
prose. **For a 95% accuracy claim to be a real claim, we need
automated metrics on a fixed test set.**

Without a benchmark we can't:
- Tell if a prompt tweak silently regresses precision.
- Compare Sonnet 4.6 vs a future Sonnet 5 quantitatively.
- Show a buyer "documenso scored X% feature recall" instead of
  hand-waving.
- Add CI gates that block PRs with -2pp regressions.

---

## 2. Sprint 6 goal in one sentence

**Build a fixed set of ground-truth feature maps for 5+ open-source
repos, plus a script that scans each, computes feature recall /
precision / flow recall / file-attribution accuracy / generic-name
rate, and emits a Markdown report.**

---

## 3. Strategic decisions already locked

- **No LLM-generated ground truth.** Human-verified to start. We can
  add LLM ground-truth as a second tier later.
- **YAML format** for ground truth — easier to hand-edit than JSON.
- **5 repos minimum:** formbricks, documenso, cal.com, gin, fastapi.
  Mix of app + library, JS + Go + Python.
- **Metrics: 5 of them.**
  1. Feature recall: `|detected ∩ expected| / |expected|`
  2. Feature precision: `|detected ∩ expected| / |detected|`
  3. Flow recall (apps only): `|detected_flows ∩ expected_flows| /
     |expected_flows|`
  4. File attribution accuracy on a 50-file random sample.
  5. Generic-name rate: features with names in a generic blocklist /
     total features.
- **Markdown report** generated per repo + a summary table.
- **CI integration deferred** until the harness is stable.
- **Branch continues** on `feat/tool-use-detection`.

---

## 4. Architecture overview

```
benchmarks/
  formbricks/
    expected_features.yaml
    expected_flows.yaml
    expected_attribution_sample.yaml   ← 50 random files manually labeled
  documenso/
    ...
  cal-com/
  gin/
  fastapi/

scripts/
  run_benchmark.py            ← CLI entry point
                                  - --repos formbricks documenso ...
                                  - --scan-results /path/to/json
                                  - --emit-report report.md

faultline/benchmark/
  __init__.py
  loader.py                   ← parse YAML ground truth
  metrics.py                  ← compute the 5 metrics
  report.py                   ← Markdown rendering
```

---

## 5. Ground-truth format

```yaml
# benchmarks/formbricks/expected_features.yaml
# Last updated: 2026-04-27 by pkuzina
# Sources: README + SDK docs + manual code review

features:
  - name: surveys
    description: Survey creation, editing, sharing, and response collection.
    aliases:                  # other names that should map here
      - surveys/general
      - web/surveys
      - web/survey-editor
    must_include:             # files we expect to be attributed here
      - apps/web/modules/survey/editor/page.tsx
      - apps/web/modules/survey/list/page.tsx
    package_hint: surveys / web

  - name: contacts
    description: Contact ingestion, attributes, and segmentation.
    aliases:
      - contacts
    must_include:
      - packages/database/types/contact.ts
    package_hint: web / database
```

Aliases let us **match by domain even when the engine picks a
different surface name**. Recall counts an expected feature as found
when ANY of its aliases appears in detected output.

```yaml
# benchmarks/formbricks/expected_attribution_sample.yaml
# 50 random files chosen via `git ls-files | shuf -n 50` on 2026-04-27.
# Each labelled with the expected feature name (or alias).

samples:
  - path: apps/web/modules/auth/login.tsx
    expected: auth
  - path: packages/email/templates/invite.tsx
    expected: email
  ...
```

---

## 6. Metrics

### Feature recall
For each expected feature, check whether any of its `name` /
`aliases` appears as a key in the detected feature map. Count
fraction.

### Feature precision
Inverse: every detected feature, does it map to exactly one expected
feature (via alias / exact match)? Unmapped detected features are
"phantom features" → drop precision.

### Flow recall (apps only)
For each expected flow, does the detected feature map have a flow
with the same name (after lowercase + de-hyphen)? Library repos
skip this metric entirely.

### File attribution accuracy
For each of the 50 sample files, look up which feature it ended up
attributed to in the scan output. Match against expected. Score is
the fraction correct (where "correct" means same feature OR an alias
of the expected).

### Generic-name rate
Detected features whose name is in a curated blocklist (`lib`,
`utils`, `shared`, `core`, `general`, `misc`, `helpers`, `types`,
`ui`, `api`, `main`, `base`, `common`) divided by total detected.

---

## 7. Implementation plan, day by day

### Day 1 — Loader + metrics + tests (no API spend)

**Files:**
- `faultline/benchmark/__init__.py`
- `faultline/benchmark/loader.py` — `load_expected_features(path)`,
  `load_expected_flows(path)`, `load_expected_attribution(path)`.
  Raises clear errors on malformed YAML.
- `faultline/benchmark/metrics.py` —
  - `feature_recall(expected, detected) -> float`
  - `feature_precision(expected, detected) -> float`
  - `flow_recall(expected_flows, detected_flows) -> float`
  - `attribution_accuracy(samples, detected) -> float`
  - `generic_name_rate(detected) -> float`
- `tests/test_benchmark_loader.py`
- `tests/test_benchmark_metrics.py` — 25+ tests covering each
  metric: perfect score, half score, zero score, edge cases (empty
  expected, empty detected, alias matching, file outside any
  feature).

**Acceptance:** all tests pass, no network.

### Day 2 — Report renderer + run script (no API spend)

- `faultline/benchmark/report.py` — `render_markdown(repo, metrics,
  scan_path) -> str`. Produces a single page with the 5 metrics +
  per-feature recall breakdown.
- `scripts/run_benchmark.py` — typer CLI. Loads expected YAML,
  loads scan JSON, computes metrics, writes report.
- `benchmarks/README.md` — how to add a new repo.

### Day 3-5 — Seed ground truth (no API spend, manual)

- For each of 5 repos, hand-author the YAML files. **This is the
  bulk of the work.** ~1-2 hours per repo to read README, scan
  product docs, label 50-file sample.
- After landing the YAMLs, run scans (live API) once with the full
  Sprint 1-5 stack and emit reports.
- Write `SPRINT_6_RESULTS.md` summarizing the headline numbers per
  repo + a single overall scorecard.

---

## 8. What's explicitly out of scope

1. **Auto-generated ground truth via LLM.** Want human-verified
   first.
2. **Open-sourcing the benchmark suite.** Stays internal until v1.
3. **CI integration.** Defer until the harness is stable.
4. **Statistical significance bars.** Just point estimates for now.
5. **Multi-temperature averaging.** One run per scan.

---

## 9. Cost discipline

| Stage | Per scan | Total |
|---|---|---|
| Sprint 1-5 full stack (formbricks-class) | ~$5-7 | × 5 repos = $25-35 |
| Benchmark harness itself (no LLM) | $0 | $0 |
| **Total Sprint 6 API spend** | | **~$25-35** (one-time) |

Day 1-2 (harness build) are zero-API. Day 3-5 (ground-truth
authoring + initial scoring scans) are where the budget lands.

---

## 10. Definition of done

- [ ] Two commits: harness build (Day 1+2), ground truth + results
      (Day 3-5).
- [ ] 5 ground-truth YAML directories committed under `benchmarks/`.
- [ ] `scripts/run_benchmark.py` runs end-to-end on at least 2 repos.
- [ ] 5 metrics auto-computed per repo, written to a Markdown report.
- [ ] 25+ unit tests in `tests/test_benchmark_*.py`.
- [ ] All Sprint 1-5 tests still pass.
- [ ] `SPRINT_6_RESULTS.md` documents per-repo headline numbers.
- [ ] Overall scorecard line: weighted average across 5 repos.

When green, **the full 6-sprint roadmap is closed**. The detection
engine is benchmark-validated, and the team can quantitatively
defend any "X% accuracy" claim. Decision after that: merge
`feat/tool-use-detection` to `main`, ship the new defaults.
