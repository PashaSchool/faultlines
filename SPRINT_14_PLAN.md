# Sprint 14 — Feature Pipeline Overhaul + Stage Re-order

**Branch:** `feat/sprint14-feature-pipeline` (нова, з main після merge S13)
**Goal:** перевести feature-detection pipeline у стан, де (1) фічі з 200+ файлів коректно розбиваються, (2) docs/infra файли не "з'їдають" корисний source, (3) sub_decompose і dedup не зливають різні бізнес-фічі під одне ім'я. Плюс мінорний flow fix із S13 verdict.

## Що драйвить sprint

External review (Problem 2):
> a) Hard cap "1-8 features per package" відрізає хвіст
> b) Bucketizer ріже потрібні файли в DOCUMENTATION/INFRASTRUCTURE
> c) Same-name collapse зливає різні features
> d) Dedup занадто агресивний на 1.5
> e) Sub_decompose без зворотного контролю

S13 verdict miss:
- `sign-in-to-shared-webapp` залишився в `i18n` бо resignal pass (Stage 2.75) у поточному порядку не бачить participants для flows які прийшли через tool_flows БЕЗ Layer B-resolved symbols.

## Скоуп

✅ **In scope**
- Dynamic feature cap per package size (replace hard cap 8).
- Bucketizer sanity-check: import-graph fan-in promotes DOCS/INFRA files back to SOURCE.
- Same-name collapse path-aware (don't merge `auth/UserService` + `admin/UserService`).
- Dedup conservative actor/event rule.
- Sub_decompose parent retention as aggregator.
- Re-order Stage 2.75 → 2.85 (after Layer C).

❌ **Out of scope**
- Sprint 12 flow pipeline rework (that's done).
- Dashboard UI changes (data shape stays).
- New benchmark repos.

## Day-by-day

### Day 1 — bucketizer sanity-check

**Файл:** `faultline/analyzer/bucketizer.py` (extend) + new `faultline/analyzer/bucket_promote.py`

- Build import-graph from extracted signatures.
- For every file in DOCS/INFRA bucket: count incoming imports from SOURCE files.
- If `source_inbound ≥ N` (default 3) → promote back to SOURCE bucket.
- Runs once at end of `partition_files` before pipeline consumes.
- Tests: synthetic graph where `schema/types.ts` is referenced by 5 source files → promoted; lone `LICENSE` stays in INFRA.

### Day 2 — dynamic feature cap per package

**Файл:** `faultline/llm/sonnet_scanner.py` (prompt + post-process)

- Compute cap = `max(3, min(15, floor(files_count / 8)))`.
- Inject into per-package prompt instead of hard "1-8".
- Two-pass: if Sonnet returns exactly the cap, second call asks "did you merge unrelated features? expand to up to 18 if so".
- Cost guard: second pass only when first call returned ≥ cap.
- Tests: package with 200 files asks for cap=15; with 30 files asks for cap=4.

### Day 3 — same-name collapse path-aware

**Файл:** `faultline/llm/pipeline.py::_collapse_same_name_features` (extend)

- Currently: any two features with same `name` → union.
- New: if path-disjoint AND no shared parent dir → rename both with parent-dir prefix instead of merge.
  - `n8n` Credentials: `backend/Credentials` (server defs) + `frontend/Credentials` (api) → keep both, prefix.
- Tests: identical-name path-disjoint → 2 features remain; identical-name overlapping paths → merged (existing behaviour).

### Day 4 — dedup actor/event rule + sub_decompose parent retention

**Files:** `faultline/llm/dedup.py`, `faultline/llm/sub_decompose.py`

- Dedup: extend prompt with rule "if features describe DIFFERENT actors (Employee vs Contractor) or DIFFERENT events (Onboarding vs Renewal), DO NOT merge".
- Sub_decompose: when splitting feature with 200+ files into 2-6 subs, keep parent as aggregator (`name=parent, paths=union, flows=[]`); subs become first-class.
- Tests: dedup fixture with `Onboard Employee` + `Onboard Contractor` → 2 features survive. Sub-decompose 250-file feature → parent + 5 subs.

### Day 5 — Stage re-order (S13 follow-up)

**File:** `faultline/llm/pipeline.py`

- Move Stage 2.75 (resignal) to Stage 2.85 (after Layer C / sweep).
- Reason: Layer C populates participants for promoted flows; resignal needs those.
- Regression: ensure dify `sign-in-to-shared-webapp` now moves to `auth`.

### Day 6 — full regression on dify

- Run live scan with full S13+S14 pipeline.
- Compare metrics:
  - Attribution accuracy: target ≥99 % (S13 was 98.8 %).
  - Symbol coverage: maintain ≥92 %.
  - Feature count: ≥17 (currently 19, may drop 1-2 from collapse fix; should not regress to 15).
  - The known sign-in-to-shared-webapp miss must be resolved.
- Update SPRINT_14_RESULTS.md.

### Day 7 — buffer + supabase regression

- Run on supabase to ensure feature pipeline changes don't regress repos that already passed.
- Polish + final commit + branch merge plan.

## Метрики цілі

| Metric | S13 (dify) | S14 target |
|---|---:|---:|
| Attribution accuracy | 98.8 % | ≥ 99 % |
| Symbol coverage | 92.1 % | ≥ 92 % (no regression) |
| Feature count | 19 | 17–22 (no collapse on real distinct features) |
| Cost / scan | $2.27 | ≤ $2.80 |
| Pre-existing test failures | 4 | 4 (don't fix unrelated) |

## Ризики

1. **Bucketizer promotion може створити noise** — types-only files з багатьма imports promoted back, перевантажують feature_paths. Mitigation: filter `.d.ts` from promotion.
2. **Dynamic cap може ламати compact packages** — пакети з 30 файлами дістануть cap=4, могли б нормально 6. Mitigation: floor at 3, not lower.
3. **Same-name path-aware rename змінює feature_name** → incremental cache invalidation. Mitigation: incremental.py перевірити, можливо потрібен migration helper.
4. **Sub_decompose parent retention може дублювати paths** — file appears у parent AND sub. Mitigation: parent stores ONLY union, sub stores own subset; render code already supports.
