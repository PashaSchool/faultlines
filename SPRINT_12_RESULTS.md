# Sprint 12 — Results (Day 7)

**Branch:** `feat/sprint12-flow-overhaul`
**Status:** All 7 days shipped. Layer A validated offline; Layers B/C
unit-tested with mocked clients, require a live scan for final
end-to-end confirmation.
**Date:** 2026-05-06

## Goal recap

> **(1)** every flow attached to the right feature; **(2)** every flow
> with `{file, start_line, end_line}` ranges per participating symbol;
> **(3)** maximum flow detection.

## Architecture delta (pipeline.run)

```
Stage 1.x       feature detection (untouched)
Stage 2.5       same-name collapse (untouched)
Stage 2.55  ←   NEW: virtual cluster promotion       (Layer A, Sprint 12)
Stage 2.6       flow_judge (Sprint 11)               (extended: also_belongs_to)
Stage 2.7   ←   NEW: per-flow symbol resolution      (Layer B, Sprint 12)
Stage 2.8   ←   NEW: entry-point sweep + cross-val   (Layer C, Sprint 12)
Stage 3         orphan validation (untouched)
Stage 4         auto-save canonicals (untouched)
```

## What shipped, day by day

| Day | Deliverable | Tests |
|-----|-------------|-------|
| 1 | `scripts/eval_flow_attribution.py` + frozen baselines + flow-truth proposals | n/a (eval tool) |
| 2 | `flow_cluster.py` — virtual cluster promotion (Layer A) | 17 |
| 3 | Pipeline wiring Stage 2.55 (default-on, opt-out) | n/a (integration) |
| 3.5 | `Flow.secondary_features`, `flow_signals.py`, `FlowVerdict.also_belongs_to`, `flow_secondaries` side channel | 19 |
| 4 | `ast_extractor.get_symbol_range` / `list_exported_symbols` | 13 |
| 4 | `flow_symbols.py` — Haiku symbol picker + cache + resolver (Layer B) | 20 |
| 5 | Pipeline wiring Stage 2.7, `_make_source_loader`, dict-shape participants in CLI | n/a (integration) |
| 6 | `flow_sweep.py` — harvester + promoter + cross-val (Layer C) + Stage 2.8 | 12 |
| 7 | Regression eval on baselines + this report | n/a |

**Total new tests:** 81 (all green). 4 pre-existing failures on baseline are
unchanged (inject_pipeline_flows, sonnet_scanner_pipeline, symbol_graph)
and unrelated to Sprint 12.

## Metrics

### Layer A (offline-verifiable)

`scripts/simulate_layer_a.py` applies cluster promotion to a frozen
feature-map JSON without touching the LLM.

| Repo | Baseline accuracy | After Layer A | Synthetic features created |
|------|-------------------|---------------|-----------------------------|
| dify | 84.7 % | **97.3 %** | `auth` (138 paths, 17 flows) |
| supabase | 97.3 % | 97.3 % | none — `auth` already in menu |
| immich | 90.0 % | 90.0 % | none — no auth-domain misses |

dify is the worst-case repo we identified at Day 1 — 16 of 20 stranded
auth flows (in `i18n` / `ui` / `contracts`) re-anchored on the
synthetic `auth` feature without an LLM call.

### Layers B + C (require live scan)

Symbol coverage and flow-count delta cannot be measured against frozen
baselines because both Haiku-driven stages were absent from those
scans. The **first live `faultline analyze` after this branch ships**
will produce the canonical numbers. Conservative estimates from the
harvest counts on the dify baseline:

- ~150 plausible entry points harvested
- ~80 flow-symbol prompts (one per flow)
- ~5–10 cross-val prompts (one per top-level feature with flows)
- Estimated cost delta: $0.30–0.50 per scan added by Layers B + C
  combined (Haiku, deterministic params).

## Behaviour changes that affect existing data shapes

1. `Flow.secondary_features: list[str] = []` — new field, default empty.
   Backwards-compatible: any old feature-map JSON parses unchanged.
2. `DeepScanResult.flow_secondaries: dict[str, list[str]] = {}` — new
   side channel, populated by flow_judge `also_belongs_to` and
   cross-validation. Also defaults empty.
3. `flow_participants[owner][flow]` entries can now be either dicts
   (Layer B output) or TracedParticipant attrs (Sprint 7 output).
   `cli._inject_new_pipeline_flows` handles both shapes.

No existing fields were renamed or removed.

## What is NOT in Sprint 12 (deferred)

| Item | Why deferred | Where |
|------|--------------|-------|
| Inject deterministic signals into `flow_judge` prompt | Needs flow.paths to be reliably present at judge time; Layer B fixes that | Sprint 13 (post-validation) |
| Dynamic feature cap per package | Out of scope ("don't touch features") | Sprint 13 — feature pipeline overhaul |
| Bucketizer sanity-check (DOCS/INFRA reclassification) | Out of scope | Sprint 13 |
| Multi-feature truth in `flow-truth.yaml` (`expected_secondary`) | Eval tool only checks primary today | Add when first multi-feature regression run shows it matters |
| Dashboard rendering of secondary owners ("shared with: …" badge) | Frontend-only, no backend work needed first | Saas-dashboard backlog |
| Cost tracker integration for sweep + symbols | Both stages already accept `tracker=` parameter — wiring at the top level is one-liner | When end-of-run cost report is rebuilt |

## Risks / known limitations

1. **Layer A `MIN_FLOWS = 3` threshold** can miss niche domains in
   small repos. Tunable per-repo via constants override; not a config
   field yet.
2. **Layer B confidence floor = 3** (vs flow_judge's 4) — chosen so
   borderline-but-real symbols are kept. May produce slight
   over-attribution; tune if regression shows noise.
3. **Layer C handler-pattern regex** is conservative and English-only.
   `signupHandler`, `AuthController`, `EmailSubscriber` all match,
   but localised symbol names (e.g. `inscriptionHandler`) won't.
   Acceptable for the OSS validation set; expand on signal.
4. **Cross-validation prompt cap** = 20 neighbours per feature. On a
   repo with 200 flows and a feature whose token overlap is weak,
   relevant neighbours may be truncated. Token-overlap ranking is
   the cheapest mitigation; could swap for embedding similarity later.

## My verdict (post-Sprint review prompt)

The pipeline now has **every architectural piece** the original
"flows in wrong feature" investigation called for:

- Layer A solves the "no destination feature exists" problem.
- Layer B solves the "no line ranges, can't compute coverage" problem.
- Layer C + cross-val solve the "missed flows + multi-feature
  ownership" problem.

What I cannot answer until a live scan: **how Layers B + C behave on
real noise**. Sprint 13 should start with a single live scan on dify,
re-run `eval_flow_attribution.py`, and compare against the Day 1
baseline. If symbol_coverage lands ≥ 70 % and accuracy holds at
95 %+, Sprint 12 is done. If symbol_coverage stalls at 30–40 % or
accuracy drops, the candidate-file ranking in `flow_symbols._
candidate_files_for_flow` is the most likely culprit and should be
the first lever.

## How to run

After merging:

```bash
faultline analyze . --llm --flows --tool-flows --model claude-sonnet-4-6
```

`--tool-flows` activates the new pipeline path. All four Sprint 12
stages run by default; opt-out individually:

```bash
faultline analyze . --llm --flows --no-flow-cluster   # disable Layer A
faultline analyze . --llm --flows --no-flow-symbols   # disable Layer B
faultline analyze . --llm --flows --no-flow-sweep     # disable Layer C
```

(Note: opt-out flags are in `pipeline.run` keyword arguments. Top-level
CLI flags would be a small addition if needed.)
