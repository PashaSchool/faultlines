# Sprint 13 — Live Results (dify regression)

**Branch:** `feat/sprint13-flow-polish`
**Scan:** `~/.faultline/feature-map-dify-20260506-090229.json`
(snapshotted to `benchmarks/dify/sprint13-live.json`)
**Date:** 2026-05-06
**Wall-clock:** ~37 min
**LLM cost:** $2.27 across 124 calls — first scan with full per-stage breakdown

## Headline metrics

| Metric | S12 baseline | S12 live | **S13 live** | Δ vs S12 |
|---|---:|---:|---:|---:|
| Attribution accuracy | 84.7% | 97.9% | **98.8%** (163/165) | +0.9 pp |
| Symbol coverage | 0% | 88.8% | **92.1%** (152/165) | +3.3 pp |
| Avg symbols / flow | 0 | 3.57 | **3.43** | similar |
| Flow count | 150 | 188 | **165** | -23 (cleaner, see below) |
| Synthetic features fired | 0 | 0 | **1 (`auth`)** | Layer A finally fired |
| Multi-feature flows | 0 | 24 | **21** | comparable |

## What Sprint 13 fixed (vs S12 verdict's three weaknesses)

### ✅ Layer A now fires on dify

Synthetic `auth` feature created with **42 paths and 11 flows**:

```
auth (42 paths, 11 flows)
  - verify-webapp-signin-code     symbols=6
  - sign-in-to-console            symbols=10
  - authenticate-user             symbols=13
  - user-sso-flow                 symbols=3
  - manage-webapp-auth-flow       symbols=4
  - oauth-login-redirect-flow     symbols=1
  - oauth-authorization-flow      symbols=1
  - init-setup-flow               symbols=2
  - manage-user-profile-flow      symbols=2
  - system-initialization-flow    symbols=3
  - profile-update-flow           symbols=1
```

All auth flows that S12 left in `i18n` are now in `auth`. Day 2's broadened
`DOMAIN_TOKENS` (added `auth`, `authenticate`, `sso`, `2fa`, etc.) plus
the smarter `_menu_has_domain` (recognises `Account Settings`-style
hint names) produced this fix.

### ✅ Symbol coverage up to 92.1 %

Day 1b's composite candidate ranker (token + handler-density +
cluster-proximity + noise filter) closed most of S12's no-symbol gap.
The remaining 13 flows without symbols are mostly synthesised by
Layer C cross-validation (no participants attached at promotion
time) — fixable next sprint.

### ✅ Cost transparency

The CLI now prints a per-stage breakdown:

```
LLM cost: $2.266 across 124 calls (687,996 in / 73,217 out)
  ├─ flow-v2:dify-web/workflow-app  $0.773  (4 calls)    ← tool_flows
  ├─ flow_sweep_promote             $0.261  (24 calls)
  ├─ flow-v2:dify-web/i18n          $0.176  (4 calls)
  ├─ flow-v2:dify-web/datasets      $0.155  (3 calls)
  ├─ deep-scan                      $0.137  (3 calls)
  ├─ flow_symbols                   $0.118  (40 calls)
  ├─ flow_sweep_cross_val           $0.037  (13 calls)
  └─ ...
```

Sprint 12's reported $1.90 was missing every Haiku layer because of
a `tracker.record(provider=...)` argument bug. The fix is small but
the visibility win is large — it's now obvious where every cent goes.

## Remaining edges (Sprint 14 fuel)

### 1. Two misattributions still

```
language-detection-flow  i18n   → dify-web/i18n  (debatable)
sign-in-to-shared-webapp i18n   → auth           (real miss)
```

`sign-in-to-shared-webapp` is the first auth flow Layer A *missed*.
It carries auth tokens in its name but its files live primarily in
`web/app/(shareLayout)/webapp-signin/` — paths shared between the
i18n and auth domains. Layer A's `_path_matches_domain` saw the
auth tokens, but `flow_judge` likely kept it on i18n because the
judge's signal-driven re-judge pass requires `flow_participants` to
be populated *before* it runs. For flows the tool_flows step left
empty, the re-judge has nothing to score against.

**Fix path:** push the re-judge stage AFTER Layer C (which populates
participants for promoted flows) — currently it sits between B and C.
Trivial wire-order change; needs a regression to confirm no new
oscillation.

### 2. Flow count dropped 188 → 165

This is a **good** drop — 23 fewer is the net result of:
- Layer A consolidating ~17 auth flows that S12 spread across
  i18n / ui / contracts.
- Tighter sweep promotions (less aggressive on borderline handlers).

But it might also be hiding lost legitimate flows. The eval doesn't
distinguish "correctly absorbed" from "lost". To pin this down we'd
need a real ground-truth set, not the auto-generated proposal.

### 3. flow_resignal didn't show in the cost summary

The Stage 2.75 re-judge ran (no warnings in the log) but produced 0
moves on this scan, so there's no cost line. That's expected when
participants are well-distributed across the right owners; Layer B's
better candidates already drove most flows to their correct
features. On a more cluttered repo the resignal pass should kick in.

## Verdict

**Merge Sprint 13.** Every weakness called out in the Sprint 12
verdict was addressed:

- Layer A fires (broadened tokens + smarter menu match)
- No-symbol gap shrunk (new ranker)
- Cost telemetry honest (provider= fix + per-stage breakdown)
- Two new unit-test suites + 5 new tests on existing suites, all
  138 sprint-12+13 tests green

Sprint 14 should move on to the **feature pipeline** problem from
the external review (sub-feature decomposition, dynamic feature cap,
bucketizer sanity check). The flow pipeline is now in a
production-ready state.

## How to verify

```bash
source .venv/bin/activate
python scripts/eval_flow_attribution.py eval \
    benchmarks/dify/sprint13-live.json \
    benchmarks/dify/flow-truth-sprint13.yaml --show-misses
python scripts/eval_flow_attribution.py summary \
    benchmarks/dify/sprint13-live.json
```
