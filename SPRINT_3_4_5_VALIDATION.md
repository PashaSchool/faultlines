# Sprints 3-5 — Live validation note

> Brief validation done after Sprints 3, 4, 5 modules + wiring landed
> on `feat/tool-use-detection`. Full per-sprint results docs
> (SPRINT_3_RESULTS.md / SPRINT_4_RESULTS.md / SPRINT_5_RESULTS.md)
> are deferred to a later session — this note records the headline
> outcome from one combined run.
>
> **Date:** 2026-04-27
> **Repo:** documenso (16 packages, ~1800 source files)
> **Command:**
> ```
> faultline analyze /tmp/documenso \
>   --llm --tool-use --dedup --sub-decompose --tool-flows --critique \
>   --max-commits 200 -o /tmp/documenso-fullstack-v2.json
> ```

---

## Headline numbers

| metric | value |
|---|---|
| total features | **40** (Sprint 1+2 alone produced 30) |
| flows total | **166** |
| features with flows | **26 / 40** |
| flows with `(entry: file:line)` grounding | **166 / 166 (100%)** |
| LLM calls | 202 |
| total cost | **$5.32** |
| runtime | ~14 min |

Within the per-sprint budget projections:
- Sprint 3 sub-decompose: +$0.30-0.60 expected, fired on 2-3 oversized
  features.
- Sprint 4 tool-flows: +$2-4 expected; ~$2.50 actual based on
  per-feature call count.
- Sprint 5 critique: +$0.50-1.00 expected; minor renames applied.

---

## What each sprint contributed

### Sprint 3 — sub-decomposition

The biggest visible win is **`team-and-organisation-management 206f`**
(a Sprint 2 dedup result) splitting into 6 sub-features:

```
team-and-organisation-management/member-and-group-management   73 files
team-and-organisation-management/organisation-lifecycle        51
team-and-organisation-management/team-lifecycle                45
team-and-organisation-management/email-and-domain-identity     27
team-and-organisation-management/webhooks-tokens-folders       21
team-and-organisation-management/sso-and-auth-portal           11
```

This is the structure a PM would expect — distinct sub-areas inside
"organisations & teams", named by what they do. Sprint 1+2 alone
produced one 206-file blob; Sprint 3 unlocked the granularity.

### Sprint 4 — tool-augmented flow detection

**Bug found and fixed during validation:**
`detect_flows_with_tools` wrote flows into
`DeepScanResult.flows` correctly, but the CLI's output writer
never picked them up — `feature_map.features[*].flows` stayed
empty. First run produced **0 flows**.

Fix: new helper `_inject_new_pipeline_flows` in `cli.py` that
attaches Sprint 4's flow names (with their `(entry: file:line)`
description trail) onto the Pydantic `Flow` objects on each
`Feature`. Also gates the legacy `_detect_flows` Haiku path off
when `--tool-flows` is the source of truth.

After fix: **166 flows across 26/40 features, 100% grounded in
real entry points**. Sample:

```
organisation-and-team-management → create-organisation
  (entry: apps/remix/app/components/dialogs/organisation-create-dialog.tsx)

organisation-and-team-management → accept-or-decline-organisation-invitation
  (entry: apps/remix/app/routes/_unauthenticated+/organisations.invite.$token.tsx)

organisation-and-team-management → verify-team-email
  (entry: apps/remix/app/routes/_unauthenticated+/teams.verify.$token.tsx)

organisation-and-team-management → manage-organisation-billing
  (entry: apps/remix/app/routes/_authenticated+/o.billing.tsx)
```

Each flow name is an imperative business action; each entry point
resolves to a real file in the repo.

### Sprint 5 — self-critique

The critique pass ran without errors (no `Traceback` / `ValueError`
from the run log). Specific renames the model proposed are mixed
into the final feature list — distinguishing "Sprint 5 rename" from
"Sprint 1+2 result" requires diffing against an `--no-critique`
control run, which we did not capture. **The validation here is
weaker** than for Sprints 3+4: no obvious damage, no obvious
distinct win.

---

## Bug discovered and fixed: cli.py flow injection

Before fix:
```
Sprint 4 ran → result.flows populated → never reached output JSON
→ 0 flows in feature map
```

After fix:
```
Sprint 4 ran → result.flows populated → _inject_new_pipeline_flows
builds Pydantic Flow objects on each Feature → flows surface in JSON
→ 166 flows, all grounded
```

This is exactly the class of bug live validation surfaces — the
modules + tests for Sprint 4 all passed, but the integration with
the existing CLI flow path was incomplete. Now closed with a small
helper + a gating check on the legacy `_detect_flows` call.

---

## What remains before merging to main

1. **Run full stack on formbricks** to confirm parity (~$5-6).
2. **Distinct Sprint 5 validation:** A/B run with `--critique` vs
   without, count renames applied. ~$5 incremental.
3. **Hand-author benchmark ground truth** for 5 repos, score with
   `scripts/run_benchmark.py`, write `SPRINT_6_RESULTS.md`. ~$25-35
   one-time.
4. **Polish: Flow.entry_point_file/entry_point_line as first-class
   Pydantic fields** — currently lives in description string. Sprint
   6 polish.

Total remaining API spend to close all 6 sprints: **~$35-45**.
Per-scan steady state: **$5-7 with full stack on
formbricks/documenso-class repos**.
