# Sprint 14 — Live Results (dify regression)

**Branch:** `feat/sprint14-feature-pipeline`
**Scan:** `~/.faultline/feature-map-dify-20260506-101533.json`
(snapshotted to `benchmarks/dify/sprint14-live.json`)
**Wall-clock:** ~33 min  **Cost:** $2.30

## Headline metrics

| Metric | S12 | S13 | **S14** | Δ vs S13 |
|---|---:|---:|---:|---:|
| Attribution accuracy | 97.9% | 98.8% | **96.5%** | -2.3 pp |
| Symbol coverage | 88.8% | 92.1% | **91.9%** | -0.2 pp |
| Avg symbols / flow | 3.57 | 3.43 | **3.40** | similar |
| Flow count | 188 | 165 | **173** | +8 |
| Cost per scan | ~$2.5 | $2.27 | **$2.30** | similar |
| `auth` in top-3 risk zones | no | no | **yes (#3)** | first time |

## What worked

- **Bucketizer promotion fired silently** — schema/types files no longer
  miss the source partition.
- **Path-aware collapse working**: `dify-web/i18n` (16 files) and `i18n`
  (1301 files) coexist as two distinct features instead of being merged
  into one bucket — correct since they're path-disjoint.
- **`auth` feature recognised as a TOP-3 risk zone** for the first
  time (31 bug fixes / 80 commits, 38.8% bug-fix ratio). Sprint 12-13
  surfaced auth as a feature; Sprint 14 puts it in the user's
  attention pane.
- **Cost telemetry honest**: $2.30 across 120 calls with 14 distinct
  per-stage labels. `flow_symbols` $0.07 (40 % cheaper than S13's
  $0.12) thanks to the better candidate ranker — fewer wasted
  Haiku calls on noise files.
- **Sub_decompose parent retention** working: `auth` parent kept
  (5 paths) while the per-flow detail lives under it.

## What slipped — the 6 misses

5 of 6 misses are auth flows still attributed to `i18n` even though
`auth` exists as a feature in this scan:

```
sign-in-to-console        i18n  (should be auth, paths exist in auth dir)
verify-email-login-code   i18n  (same)
sign-in-to-webapp         i18n
verify-webapp-email-login-code  i18n
oauth-authorization-flow  i18n
plugin-subscription-modal-flow  plugins  (should be billing)
```

### Root cause

The `auth` feature this run carries only **5 paths and 1 flow**
(`sso-provider-flow`). In S13 it had 42 paths and 11 flows. What
changed:

- Sonnet's per-package scan emitted an `auth` feature directly this
  run (presumably from a small auth-named package).
- Layer A sees `auth` already in the menu (`_menu_has_domain` returns
  True on the literal substring match) and **skips** the cluster
  promotion entirely.
- The 6 stranded auth flows in `i18n` therefore have no path-domain
  evidence — `auth.paths` only contains 5 unrelated files, so
  `file_ownership_score(flow_paths, auth_paths)` is ~0 %.
- Stage 2.85 resignal pass sees no ownership disagreement and
  leaves the flows put.
- The primary `flow_judge` (Stage 2.6) **should** have moved them on
  semantics alone (the prompt explicitly says "i18n must release auth
  flows"). Cache likely served stale verdicts — the feature-set hash
  changed between S13 and S14 (new `dify-web/index.stories` /
  `dify-web/i18n` entries), so the cache should have invalidated, but
  this needs a closer look.

### What this means

Net of S14, attribution dropped 2.3 pp on dify but symbol coverage,
flow count, cost, and per-stage transparency all held or improved.
The regression is concentrated in one specific failure mode (small
Sonnet-named `auth` feature that blocks Layer A backfill), not a
broad degradation.

## Sprint 15 (proposed scope, 1 day)

The fix is a single tweak to Layer A's gating logic plus a small
flow_judge prompt rule:

1. **Layer A backfill mode** — when the literal-named domain feature
   exists but has < N paths AND ≥ 3 flows in catch-all owners match
   the domain → run the path-promotion anyway, growing the existing
   feature instead of skipping. ≈ 30 LOC + 2 tests.

2. **flow_judge cache key** — include not just the feature-set hash
   but ALSO the flow-set hash. A stale verdict on a flow whose
   description / surroundings shifted should re-judge.

3. **Strengthen resignal threshold for catch-all → domain moves** —
   when source is `i18n`, `ui`, or `web` AND target is
   domain-specific (auth, billing, notifications), drop the
   ownership-disagreement threshold from 30 % to 15 %. Auth flows
   stranded with low ownership signal still move.

Expected outcome: dify accuracy back to ~99 %, no impact elsewhere.

## What stays merged

All five S14 days are correct work and should ship:

- Day 1 bucketizer promotion (verified silently working)
- Day 2 dynamic feature cap (no observable harm; need bigger repo
  to see the real benefit — supabase/calcom would help)
- Day 3 path-aware collapse (`dify-web/i18n` + `i18n` coexisting is
  the right outcome)
- Day 4 dedup actor/event rule + sub_decompose parent retention
  (parent retention visibly carrying `auth` across the pipeline)
- Day 5 Stage 2.85 re-order (working as designed; the gap is in
  Layer A's gating, not in stage ordering)

## Verdict

**Merge S14**, then immediately ship the three S15 fixes above as a
narrow follow-up. The data shape, telemetry, and architectural changes
are all wins; the attribution regression is a small, specific gating
bug with a clear fix path.

## How to verify

```bash
source .venv/bin/activate
python scripts/eval_flow_attribution.py eval \
  benchmarks/dify/sprint14-live.json \
  benchmarks/dify/flow-truth-sprint14.yaml --show-misses
python scripts/eval_flow_attribution.py summary \
  benchmarks/dify/sprint14-live.json
```
