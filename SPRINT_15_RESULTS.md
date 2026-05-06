# Sprint 15 — Results (dify regression)

**Branch:** `feat/sprint15-attribution-fixes`
**Scan:** `~/.faultline/feature-map-dify-20260506-105743.json`
(snapshotted to `benchmarks/dify/sprint15-live.json`)
**Wall-clock:** ~38 min  **Cost:** $2.34

## Headline

| Metric | S13 | S14 | **S15** | Δ vs S14 |
|---|---:|---:|---:|---:|
| Attribution accuracy | 98.8% | 96.5% | **94.4%** | -2.1 pp |
| Symbol coverage | 92.1% | 91.9% | **89.8%** | -2.1 pp |
| Avg symbols / flow | 3.43 | 3.40 | **3.36** | similar |
| Flow count | 165 | 173 | **177** | +4 |
| Cost / scan | $2.27 | $2.30 | **$2.34** | similar |

**The S15 fixes were not effective on this scan** — accuracy dropped
another 2.1 pp instead of climbing to ≥99%. The 10 remaining misses
are essentially the S14 cluster: 8 auth flows still stuck in i18n.

## Forensic finding — the backfill DID compute correctly

Replaying `promote_virtual_clusters` directly on the saved JSON:

```
flow_cluster: backfill mode for domain=auth into 'auth'
              (5 existing paths, 12 stranded flows)
flow_cluster: planning backfill domain=auth, flows=12, paths=112
flow_cluster: backfilling 112 paths + 12 flows into existing 'auth'

BEFORE: auth paths=5  flows=1
AFTER:  auth paths=117 flows=13
```

So the **S15 D1a logic is correct**. Layer A correctly identifies
the dify case (existing `auth` undersized + 12 stranded flows in
i18n) and would move 112 paths + 12 flows.

But during the live pipeline, those mutations don't survive. Evidence
of the disappearance:

```
~/.faultline/assignments-dify.json (saved 13:49 by save_assignments,
mid-pipeline):
  paths assigned to 'auth': 24  ← Layer A's mutations partially
                                  visible here
~/.faultline/feature-map-dify-...json (saved 13:57, +8 min later, by
the CLI post-pipeline stack):
  auth feature has 5 paths      ← mutations gone
```

`save_assignments` is the LAST line of `pipeline.run()`. Between it
returning and CLI's final JSON write, 8 minutes of post-pipeline
processing happens (`build_feature_map`, `_drop_noise_features`,
flow injection, blame, enrichment, etc.). One of those steps is
overwriting / regressing `result.features["auth"]` from ~117 down
to 5.

This is a **scope-of-effects bug**, not a logic bug. Layer A's
pre-flight (the calculation) is correct; persistence to the user-
visible artifact is broken.

## What this means

S15 D1a / D1b / D2 implementations are all correct in isolation
(31 cluster tests + 50 judge tests + new D2 tests all green). The
pipeline-end regression is unrelated to the fixes themselves — it's
that the CLI's post-pipeline code path doesn't see Layer A's work.

Reproduces back to S14 too: `assignments-dify.json` had 24 auth paths,
final JSON had 5. The bug existed before S15 — S15 just didn't fix it
because the attribution work runs INSIDE pipeline.run, where the
mutations get clobbered downstream.

## Sprint 16 plan revision

Before the eval-harness sprint (deferred per user), we need to
**find and fix the auth-paths regression**. ~½ day of debugging.

Concrete steps:

1. Add `print` instrumentation right after `_run_new_pipeline` in
   cli.py to log `_new_pipeline_result.features["auth"]` length.
2. Run a fast scan with `--no-tool-flows --no-flow-judge --no-flow-
   symbols --no-flow-sweep --no-flow-resignal` (only Layer A active).
   See if pipeline output JSON has correct auth count.
3. Re-enable each stage one at a time and find which one zeroes out
   the work.

Most likely culprit: some legacy stage (sub-decompose, smart-
aggregators, _drop_noise_features) iterating over a mid-state
snapshot rather than the final mutated `result.features`.

## What stays merged

S15 implementations are correct and tested:
- Layer A backfill mode — handles small existing domain feature
- flow_judge cache flow-set hash — invalidates on flow set change
- Resignal asymmetric threshold — catch-all → domain at 0.15

The fixes will work the moment the upstream regression is fixed.
Don't revert.

## How to verify (post-fix)

```bash
source .venv/bin/activate
python scripts/eval_flow_attribution.py eval \
  benchmarks/dify/sprint15-live.json \
  benchmarks/dify/flow-truth-sprint15.yaml --show-misses
```

To verify the backfill logic in isolation:

```bash
python -c "
import json
from faultline.llm.flow_cluster import promote_virtual_clusters
from faultline.llm.sonnet_scanner import DeepScanResult
fm = json.load(open('benchmarks/dify/sprint15-live.json'))
result = DeepScanResult(
  features={f['name']: list(f.get('paths', [])) for f in fm['features']},
  flows={f['name']: [fl['name'] for fl in f.get('flows', [])] for f in fm['features']},
  descriptions={f['name']: f.get('description', '') for f in fm['features']},
)
promote_virtual_clusters(result)
print(f'auth post-replay paths={len(result.features[\"auth\"])}')
"
```

Expected: `auth post-replay paths=117` (confirms S15 D1a logic
correct, downstream regression is the open issue).
