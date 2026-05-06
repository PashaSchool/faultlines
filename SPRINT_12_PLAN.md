# Sprint 12 — Flow Pipeline Overhaul

**Branch:** `feat/sprint12-flow-overhaul` (нова, з `feat/sprint11-semantic-flows`)
**Goal:** перевести flow-пайплайн з "наближеної атрибуції + entry-point only" у стан, де **(1) кожен flow прибитий до правильної фічі**, **(2) кожен flow має точні `{file, start_line, end_line}` діапазони на рівні символів**, **(3) кількість виявлених flows максимізована**.

## Контекст / діагноз (dify)

`flow_judge` (Sprint 11) перевіряє Auth-flows проти існуючого меню фіч. У dify меню Auth-фічі **не існує** (auth-сторінки лежать у пакеті `web`, який поглинули translations → пакет потрапив у `i18n`). Тому 9 з 11 "i18n flows" насправді auth: `reset-password-with-email-flow`, `webapp-email-signin-flow`, `signup-email-verification-flow`, тощо. `entry_point` у більшості flows = `None`, `participants = []` → жодних line-ranges. Інфраструктура (`Flow.participants`, `SymbolRange.start/end`, `flow_tracer.trace_flow_callgraph`) є, але вмикається тільки під `--trace-flows` і не обмежується до конкретних символів.

## Скоуп

✅ **In scope**
- Re-attribution flows у нові *синтетичні* фічі (Layer A — virtual clusters з кишень мега-бакетів).
- Per-flow точні line ranges на рівні символів — `Flow.participants[].symbols[].{start_line, end_line}`.
- Sweep entry-point'ів (Layer C) — підняти кількість flows за рахунок неприв'язаних route handlers / API endpoints.
- Тести на dify, documenso, supabase (regression на трьох найболючіших репо).

❌ **Out of scope**
- Renaming / deleting існуючих доменних фіч (Workflow / Knowledge Base / Billing і т.д. **не чіпаємо**).
- Зміна `Feature.paths` mainstream-фіч поза синтетичними кластерами.
- Touching `shared_participants` / aggregator pipeline.
- Rewrite `tool_flows` (Sprint 4) — він залишається опт-ін.

## Архітектура (нова)

```
detect_flows_llm   ─┐
tool_flows         ─┤  Stage 1.x — як зараз
                    ▼
                 raw flows (name → list per feature)
                    │
Stage 2.5      ─────┤  dedup / cap / merge — як зараз
                    ▼
NEW Stage 2.55:  Virtual Cluster Promotion (Layer A)
   ─ збирає flows зі strong domain tokens (auth/billing/etc),
     які зараз сидять у catch-all buckets (i18n, ui, web shells),
     знаходить domain-coherent path prefix (web/app/signin/*,
     web/app/forgot-password/* …) → промує **новий feature**
     "auth" зі справжніми paths, переносить flows туди.
                    │
Stage 2.6      ─────┤  flow_judge (Sprint 11) — без змін, але
                    │  меню тепер містить синтетичні кластери.
                    ▼
NEW Stage 2.7:   Per-Flow Symbol Resolution (Layer B)
   ─ для кожного flow:
       1. trace_flow_callgraph → BFS до глибини 3, повертає
          кандидатські файли + усі exported symbols у них.
       2. Haiku judge ("symbol picker"): дивиться на flow.name +
          flow.description + список (file, symbol, kind, signature)
          → повертає підмножину {file, symbol} які реально беруть
          участь у user journey.
       3. Resolver (deterministic): для кожного {file, symbol}
          знаходить start_line/end_line через AST-extractor
          (`analyzer/ast_extractor.py`) → пише в
          flow.participants[].symbols.
                    ▼
NEW Stage 2.8:   Entry-Point Sweep (Layer C)
   ─ збирає всі route handlers / API endpoints / event subscribers
     (через `tool_use_scan` reuse), фільтрує ті, які НЕ потрапили
     у жоден flow, групує за path-prefix + token similarity,
     генерує flow-кандидати, прогоняє через 2.6 + 2.7.
```

## Артефакти / файли

```
faultline/llm/
  flow_judge.py             — без змін, але приймає extended menu
  flow_cluster.py           — NEW (Layer A): virtual cluster promotion
  flow_symbols.py           — NEW (Layer B): per-flow symbol picker
  flow_sweep.py             — NEW (Layer C): unattached-entry-points sweep
  pipeline.py               — wires нові stages 2.55 / 2.7 / 2.8

faultline/analyzer/
  flow_tracer.py            — extend: вертає symbol-level не file-level
  ast_extractor.py          — extend: get_symbol_range(file, name) helper

tests/
  test_flow_cluster.py      — NEW
  test_flow_symbols.py      — NEW
  test_flow_sweep.py        — NEW
  test_flow_pipeline_e2e.py — NEW (regression на mocked dify slice)
```

## Day-by-day

### Day 1 — діагностика baseline + dataset

- Snapshot поточного `feature-map-final.json` для dify, documenso, supabase у `benchmarks/<repo>/sprint12-baseline.json`.
- Скрипт `scripts/eval_flow_attribution.py`: для repo + ground-truth (вручну анотовано 30-50 flows на dify) обчислює `attribution_accuracy`, `flow_count`, `symbol_coverage` (% flows які мають хоч один symbol-range).
- Зафіксувати baseline-метрики в `EVAL_REPORT.md`.

**Ground truth для dify (30 flows manually):** годинна робота — позначити correct feature для топ-30 flows.

### Day 2 — Layer A: Virtual Cluster Promotion

- `faultline/llm/flow_cluster.py::promote_virtual_clusters(result, repo_root)`:
  - DOMAIN_TOKENS = `{auth: [...], billing: [...], notifications: [...], onboarding: [...], search: [...]}` — мінімум 5 кластерів спершу.
  - Для кожного кластеру: знайти flows у `result.flows` чий name матчить токен; якщо їх ≥3 і domain feature **не існує** в меню → знайти найдовший common path prefix серед flow paths → створити новий `Feature` з тим prefix як `paths`, винести paths з catch-all бакетів.
  - Тести: synthetic fixture з 5 auth flows у "i18n" фічі → після promotion 1 нова "auth" фіча зі своїми paths.

### Day 3 — Layer A wiring + flow_judge re-run

- Стейдж 2.55 у `pipeline.py` перед існуючим flow_judge.
- Run на dify → перевірити, що auth flows тепер у "auth" фічі а не в "i18n".
- Ground truth metric: attribution_accuracy має піднятись з ~50% до ≥85%.

### Day 4 — Layer B core: symbol picker prompt + resolver

- `faultline/analyzer/ast_extractor.py::get_symbol_range(file_path, symbol_name) -> SymbolRange | None` — детермінована функція через regex/AST (вже є парсер для TS/JS, треба додати Python signature regex).
- `faultline/llm/flow_symbols.py::pick_flow_symbols(flow, candidate_files, client) -> list[(file, symbol)]`:
  - Будує prompt: flow name + description + список `(file, exported_symbols_with_signatures)`.
  - Cap: ≤15 файлів / ≤80 символів на flow (token budget).
  - System prompt: "повертай тільки символи які РЕАЛЬНО викликаються в user journey, не helper utilities де flow тільки імпортує тип".
  - Output JSON schema: `[{"file": ..., "symbol": ..., "confidence": 1-5}]`.
- Cache verdicts по hash(flow_name + sorted(candidate_files)) у `~/.faultline/flow_symbols_cache.json` (як у `flow_judge.py`).

### Day 5 — Layer B integration

- Стейдж 2.7 у `pipeline.py`.
- Для кожного flow:
  1. Викликати `trace_flow_callgraph` (BFS depth=3) → candidate files.
  2. Збирати всі exported symbols через `ast_extractor`.
  3. `pick_flow_symbols` → відсіяний список.
  4. Resolver → `SymbolRange` з line_start/line_end → `flow.participants[].symbols`.
- Метрика: `symbol_coverage` — % flows у яких ≥1 symbol_range.
- Параметр `--flow-symbols` (default true) щоб можна було вимкнути для бюджету.

### Day 6 — Layer C: entry-point sweep

- `faultline/llm/flow_sweep.py::sweep_unattached_entry_points(result, repo_root)`:
  - Reuse `tool_use_scan` для збору route handlers / API endpoints / event subscribers.
  - Фільтр: символи які НЕ присутні у жодному `flow.participants[*].symbols`.
  - Кластеризація: групуємо за path-prefix + handler-name token similarity.
  - Кожен кластер → flow candidate `{name, description, entry_point_file, entry_point_line, candidate_files}`.
  - Прогоняємо через flow_judge (Stage 2.6) + symbol picker (Stage 2.7).

### Day 7 — Regression suite + tuning

- E2E на dify, documenso, supabase. Зрівняти метрики зі snapshot з Day 1.
- Targets:
  - `attribution_accuracy`: ≥85% (з ~50%)
  - `symbol_coverage`: ≥75% (з ~5%)
  - `flow_count` дельта: +20-40% (Layer C додає flows)
  - Cost per scan: ≤2.5× baseline (Haiku, не Sonnet)
- Tune: domain tokens (Layer A), prompt rules для symbol picker (Layer B), кластеризація-thresholds (Layer C).

## Метрики успіху (мінімум для merge)

| Метрика | Baseline (dify) | Target |
|---|---|---|
| Attribution accuracy | ~50% | ≥85% |
| Flows з ≥1 symbol_range | ~5% | ≥75% |
| Flow count | 137 | ≥165 |
| Avg symbols per flow | 0 | 3-8 |
| Cost per scan | $0.20 | ≤$0.50 |

## Ризики

1. **Layer A може промувати false-positive кластери** — наприклад, у бібліотечному repo "auth" може означати щось інше. Mitigation: вмикається тільки якщо ≥3 flows з domain tokens AND немає матчингової фічі AND знайдено path prefix який покриває ≥70% цих flows.
2. **Layer B token budget** — на flow з 50 candidate files prompt може не влізти. Mitigation: cap 15 files, ranking по depth у callgraph + entry-point distance.
3. **AST extractor для Python signature** — зараз primary target TS/JS, треба додати Python regex/AST. Mitigation: якщо extractor failед на конкретному symbol → залишаємо файл без symbol_range, не падаємо.
4. **Determinism** — symbol picker під temperature=0 + cache → byte-identical results на повторному скані. Тести перевірятимуть.

## Out-of-scope але треба занотувати

- Підтримка цих symbol_ranges у dashboard / coverage.py — окремий sprint після того, як дані будуть у JSON.
- Cross-repo flow detection (Sprint 13+).
- Більше ніж 5 domain clusters у Layer A — додаємо за потребою у наступних sprints.
