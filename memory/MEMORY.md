# Faultline — Session Memory

## Поточний стан (2026-03-02)

### Що зроблено в цій сесії (2026-03-02)

1. **Ollama default model** змінено з `qwen2.5-coder:7b` → `llama3.1:8b` в `detector.py` і `flow_detector.py`
2. **Commit keywords у merge prompt** — `_extract_cluster_keywords()` в `detector.py`, top-4 слова per cluster → inject в LLM prompt
3. **E2E anchors для flow detection** — `detect_e2e_anchors()` в `flow_detector.py`, виявляє `.spec.ts`, `.cy.ts`, `.e2e.ts` файли → flow names як авторитетні підказки для LLM
4. **`cli.py`** оновлено: `commits` передається в `_merge_and_name_with_llm`, `e2e_anchors` в `_detect_flows`
5. **README.md** — нова таблиця Ollama моделей (llama3.1:8b default, mistral-nemo:12b best quality, qwen2.5:7b alternative, llama3.2:3b lightweight)
6. **GitHub Pages лендинг** створено: `featuremap/docs/index.html` (32 KB, self-contained)
   - Структура: Hero → Pain Points → How it works → Features 2×2 → Stats → CTA form → Footer
   - Темна тема (#08090d), акцент #00e5a0, JetBrains Mono + Inter
   - `pip install faultline` snippet з copy button
   - Terminal mockup із реальним output + HTML `<table>` для "Features by Risk"
   - CTA форма: Name, Work email, Telegram/Slack, Role dropdown, Submit (Formspree endpoint — замінити `YOUR_FORM_ID`)
   - GitHub Pages: Settings → Pages → Branch: main → Folder: /docs

### Поточний flow аналізу
```
extract_signatures (завжди, не лише при --llm)
  ↓
build_import_clusters (import_graph.py) — primary, deterministic
  ↓  fallback: detect_features_from_structure (якщо нема TS/JS файлів)
_extract_cluster_keywords (з commits) → inject у LLM merge prompt
merge_and_name_clusters_llm/ollama (якщо --llm) — cached by cluster SHA256
  ↓
build_feature_map
  ↓  (якщо --flows)
detect_e2e_anchors → detect_flows_llm/ollama per feature
  ↓
print_report → write_feature_map
```

### Незавершена робота (попередня сесія)

**Проблема**: import graph створює один гігантський кластер (~1800/2156 файлів) через довгі ланцюжки імпортів у великих проєктах.

**Рішення реалізовано але НЕ протестовано**:
- `_MAX_CLUSTER_FRACTION = 0.25` в `import_graph.py`
- Phase 3 в `build_import_clusters`: якщо кластер > 25% всіх файлів → split по директоріях
- `_merge_feature_count_hint(n_clusters)` в `detector.py` — підказка для LLM "aim for N-M features, max X per feature"

**Команда для тесту (НЕ запускалась)**:
```bash
rm -rf ~/.faultline/llm-cache/
source .venv/bin/activate
faultline analyze /Users/pkuzina/workspace/platform --src src/ --llm --provider ollama --model llama3.1:8b --no-save
```

### Pre-existing test failures (не пов'язані з нашими змінами)
- `test_non_ts_file_skipped_by_extract_signatures`
- `test_collapses_to_dirs_for_large_repos`

## Архітектура проєкту

### Файли
```
faultline/
├── cli.py                        — точка входу, typer команди
├── analyzer/
│   ├── git.py                    — load_repo, get_commits, get_tracked_files
│   ├── features.py               — detect_features_from_structure, build_feature_map, build_flows_metrics
│   ├── cochange_detector.py      — Union-Find co-change clustering (fallback)
│   ├── import_graph.py           — PRIMARY: import dependency clustering
│   └── ast_extractor.py          — extract_signatures (TS/JS/Python imports/exports)
├── llm/
│   ├── detector.py               — merge_and_name_clusters_llm/ollama, detect_features_llm/ollama
│   └── flow_detector.py          — detect_flows_llm/ollama, detect_e2e_anchors
└── output/
    ├── reporter.py               — print_report (terminal tables)
    └── writer.py                 — write_feature_map (~/.faultline/*.json)
docs/
└── index.html                    — GitHub Pages лендинг (Product Hunt ready)
```

### Кеш
- `~/.faultline/llm-cache/` — merge cache, key = `merge_` + SHA256 cluster structure, TTL 90d

### Dashboard
- `dashboard/` — Vite+React+TypeScript, читає з `~/.faultline/*.json`
- Запуск: `cd dashboard && npm run dev` → http://localhost:5173

## Ключові константи

| Константа | Значення | Де |
|---|---|---|
| `_MAX_IMPORT_FANIN` | 8 | `import_graph.py` |
| `_MAX_CLUSTER_FRACTION` | 0.25 | `import_graph.py` |
| `_MODEL` | `claude-haiku-4-5-20251001` | `detector.py`, `flow_detector.py` |
| `_DEFAULT_OLLAMA_MODEL` | `llama3.1:8b` | `detector.py`, `flow_detector.py` |
| `_NAME_CACHE_TTL_DAYS` | 90 | `detector.py` |

## Провайдери LLM
- `anthropic` — Claude Haiku 4.5, найкращий результат
- `ollama` — локально, llama3.1:8b default; mistral-nemo:12b best quality
- `gemini` — не реалізовано

## Тестовий репо
- `/Users/pkuzina/workspace/platform` — основний репо для тестування
- 5000 commits, 2156 files under src/, 1853 TS/JS files
- `--src src/` завжди потрібен для цього репо

## Бізнес контекст
- TAM: $4.8B–$6.2B | SAM: $520M–$780M
- Реалістичний Year 1 ARR: $36K–$84K (не $393K)
- Gross margin: 87–93%
- Конкурентне вікно: 12–18 місяців (CodeScene найближчий)
- Pricing: Team $22/dev/mo | Business $38/dev/mo | Enterprise custom
- PLG стратегія: PR bot як viral acquisition channel
