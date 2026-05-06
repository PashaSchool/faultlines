# Sprint 15 — Attribution Fixes (S14 follow-up)

**Branch:** `feat/sprint15-attribution-fixes`
**Goal:** виправити 6 misses, що S14 залишив на dify, без deg на supabase / immich. Три точкові виправлення; жоден не зав'язаний на конкретний репо — це загальний клас багу, який dify просто проявив найвиразніше.

## Three fixes

### 1. Layer A backfill mode (Day 1)

**Bug:** `_menu_has_domain("auth")` тригериться як тільки в меню є будь-яка `auth`-named feature, навіть якщо вона має 5 paths і не покриває реальний auth surface. Layer A пропускає.

**Fix:** додати поріг розміру + режим backfill — коли domain feature існує АЛЕ меньше за `MIN_DOMAIN_PATHS` AND ≥3 catch-all flows матчать domain:
- Не створювати нову фічу.
- Промотити paths/flows у ІСНУЮЧУ domain feature (backfill).
- Логіка `apply_promotions` accept існуючий feat як target.

**Не зачіпає:** репо де domain feature вже добре покрита.

### 2. flow_judge cache flow-set hash (Day 1)

**Bug:** `_load_cache` invalidates тільки на feature-set hash зміну. Якщо feature menu стабільне між run'ами але flow names різні (новий tool_flows запуск, нові sweep promotions), cached `keep` verdicts replay не там.

**Fix:** додати `flow_set_hash` в cache header. Invalidate при зміні any of (feature_hash, flow_hash).

**Не зачіпає:** перший скан репо — там кешу немає взагалі.

### 3. Resignal asymmetric threshold (Day 2)

**Bug:** `re_judge_with_signals` requires `best_other - current ≥ 0.30`. Для auth flow в i18n: i18n[1301 files] ownership ≈ 20%, auth[5 files] ownership ≈ 0% — різниця -20%, рух не відбувається.

**Fix:** asymmetric threshold:
- catch-all source (`i18n`, `ui`, `web`, generic shell names) → domain target (`auth`, `billing`, `notifications`) → threshold = 0.15 (was 0.30)
- domain → domain → залишити 0.30 (conservative)
- Сигнал = source ownership *< 50%* AND target має домен hint.

**Не зачіпає:** flows які вже у domain-specific фічах (resignal на them триг тільки коли строге +30% дисагріемент).

## Day-by-day

### Day 1

- `flow_cluster.py::plan_promotions` extend з backfill підтримкою.
- `flow_cluster.py::apply_promotions` уміє додавати в існуючу фічу.
- `flow_judge._feature_set_hash` → `_set_hashes` (commit feature + flow).
- Tests: synthetic dify case (5-path auth + 6 stranded flows) → backfill fires.

### Day 2

- `flow_judge.re_judge_with_signals` — asymmetric threshold:
  - new helper `_is_catchall_owner(name)` — i18n / ui / web / shell / app shell heuristic.
  - threshold = 0.15 if catch-all → domain.
- Tests: synthetic catch-all-source flow with 0% target ownership but domain target → still moves on the ASYMMETRIC path.

### Day 3 — live regression

- Run on **dify** + **supabase** + **immich**.
- Targets:
  - dify ≥ 99 % accuracy (currently 96.5 %).
  - supabase ≥ 97 % (currently 97.3 % baseline; should not regress).
  - immich ≥ 90 % (currently 90 %; same).
  - All other metrics (symbol coverage, flow count, cost) within ±5%.

## Метрики цілі

| Repo | S14 | **S15 target** |
|---|---:|---:|
| dify accuracy | 96.5 % | ≥ 99 % |
| dify symbol cov | 91.9 % | ≥ 91 % |
| supabase accuracy | 97.3 % (baseline) | ≥ 97 % (no regression) |
| immich accuracy | 90.0 % (baseline) | ≥ 90 % (no regression) |

## Risks

1. **Backfill mode може overfill** — якщо існуюча domain feature легітимно мала (e.g. supabase's `vue-blocks/auth` — 1 sub-feature), backfill з catch-all bucket розіб'є її структуру. Mitigation: тільки backfill з `_DOMAIN_FEATURE_NAME_HINTS`-matching feature names, не з прямого літерала якщо path count < 5.

2. **Asymmetric threshold може over-move borderline cases** — catch-all source з 25% ownership може мати legitimate stake в flow. Mitigation: cumulative guard — also require `target_ownership ≥ 0.05` (not zero), щоб target був хоча б minimally relevant.

3. **flow_set_hash invalidates cache occасionally** — більше Haiku calls при non-trivial changes. Mitigation: acceptable — точність важливіша за cache hit rate.
