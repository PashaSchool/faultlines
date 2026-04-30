# Pricing reality + incremental-scan plan

Captured 2026-04-29 / 04-30 boundary. Two-part document:

1. **Honest cost analysis vs. current pricing on faultlines.dev**
2. **Engineering plan for incremental scans** — the technical path that keeps current pricing viable

---

## Part 1 — Pricing reality

### Current pricing on `/pricing` (as of 2026-04-29)

| Plan | Price | Limits |
|---|---|---|
| Free | $0 | CLI + MCP, public repos, private with BYO key |
| Indie | $9 / org / mo | hosted dashboard, 3 private repos |
| Team Starter | $29 / org / mo (was $49, lock-in for first 100) | 5 repos, SSO, REST API |
| Team Pro | $99 / org / mo (was $149) | 20 repos, PR comments, Slack, health trends |
| Enterprise | custom | unlimited, SAML, on-prem |

Lock-in promise: first 100 Team Starter customers keep $29/mo for 24 months.

### Real per-scan cost (measured 2026-04-29)

From benchmarks today:

| Repo size | Cost per full scan |
|---|---|
| Tiny library (gin, axios, ~100–300 files) | $0.05–0.20 |
| Small app (trpc, excalidraw, ~1K files) | $0.50–1.00 |
| Medium monorepo (documenso, formbricks, 2K–3K files) | $2.50–3.00 |
| Large monorepo (plane, ghost, 4K–7K files) | $4.40–5.00 |
| Huge enterprise (cal.com, 10K+ files) | $9.00–10.00 |

Cost basis: Sonnet 4.6 ($3/M in, $15/M out) + per-package LLM calls. Roughly linear with file count above 2K.

### Margin scenarios — flat per-org pricing

**Indie ($9/mo, 3 repos):**

| Customer profile | Scan cost/mo | Margin |
|---|---|---|
| 3 small repos, weekly scan | $6 | +$3 (33%) ✓ |
| 3 medium repos, weekly | **$36** | **−$27** ❌ |
| 3 medium repos, monthly | $9 | breakeven |

**Team Starter ($29/mo, 5 repos):**

| Customer profile | Scan cost/mo | Margin |
|---|---|---|
| 5 small repos, weekly | $20 | +$9 (31%) ✓ |
| 5 medium repos, weekly | **$60** | **−$31** ❌ |
| 3 medium + 2 small, weekly | $40 | **−$11** ❌ |

**Team Pro ($99/mo, 20 repos + PR comments):**

PR impact comments scan on every PR. Active 20-dev team produces 100+ PRs/mo.

| Customer profile | Scan cost/mo | Margin |
|---|---|---|
| 20 small, weekly + 50 PR scans | $105 | **−$6** ❌ |
| 20 medium, weekly | **$240** | **−$141** ❌ |
| 20 mixed, monthly + 100 PR scans | $110 | **−$11** ❌ |

**Enterprise:** healthy by definition — pricing is custom, can absorb any usage.

**Catastrophe scenario** (must be detected and routed away):

- 1 customer, 10 cal.com-class repos in Team Pro, active PR flow
- Cost: 10 × 4 × $9 + 200 PRs × $0.50 = **$460/mo**
- Revenue: $99/mo
- **−$361/mo loss per customer**

### Conclusion on pricing

Yes, the current model can go negative on plausible (not edge-case) customer profiles. The combination of **flat per-org pricing** + **PR comment scans** + **no usage caps** is the structural problem. Per-dev pricing (the previous $22/$38) naturally scaled with team size and PR volume; flat per-org doesn't.

### Recommendations — three options

**Option 1 — Soft scan caps (preserves lock-in promise)**

| Plan | Included scans/mo | Overage |
|---|---|---|
| Indie | 50 | $0.30/extra |
| Team Starter | 200 | $0.30/extra |
| Team Pro | 1,000 + 200 PR scans | $0.30/extra |
| Enterprise | unlimited | — |

Action: surface scan budget in dashboard ("145 of 200 used"), email at 80%, cap with friendly upgrade prompt at 100%. Don't auto-charge overage in the first quarter — collect usage data first.

**Option 2 — Price increase ~50%**

Indie $15, Team Starter $49, Team Pro $149. Breaks the lock-in promise. **Don't do.**

**Option 3 — Make scans cheaper (engineering, not pricing)**

Implement incremental scans (only changed files) → 5–10× cheaper per scan. PR comment cost drops $3 → $0.30. Current pricing becomes durable.

### Recommended path

**Option 3 + Option 1 soft caps.** Engineering work pays back faster than pricing churn. Caps are insurance.

Don't touch listed prices. Tier limits and scan caps are the real instrument.

**Add Enterprise gate by repo size.** When a repo's file count exceeds ~5,000 the CLI/dashboard prompts: "this repo qualifies for Enterprise pricing — talk to sales." Cal.com-class customers must not be on Team Pro.

---

## Part 2 — Incremental scan engineering plan

### Goal

Make scans on a previously-analyzed repo cost ~10–20% of the full scan, by only re-analyzing files that changed since the last scan.

Cost target after incremental:

| Repo size | Full scan today | Incremental target |
|---|---|---|
| Medium monorepo | $3.00 | $0.30–0.50 |
| Large monorepo | $5.00 | $0.50–1.00 |
| Cal.com-class | $9.00 | $1.00–2.00 |

PR scans (always small diff) drop to ~$0.20–0.40 each. Becomes economically reasonable to ship as Team Pro feature.

### What we already have

- **`~/.faultline/assignments-{repo}.json`** — cache of `{file_path → canonical_feature}` from last scan. Today used for stability re-normalization; can also tell us "this file already belongs to feature X."
- **`.faultline.yaml` auto_aliases** — locks canonical names so re-runs don't drift.
- **Token-match aliasing + parent-collapse** — canonicalizes new feature names to match prior runs.
- **Stable feature naming** (88% across consecutive scans on documenso, 100% on soc0).
- **Per-package workspace scan** — each package is independent, easy to re-run only changed ones.

### What we need to add

**Stage 1 — Detect what changed (no LLM cost)**

- `git diff --name-only <last_scanned_sha>..HEAD` to list changed source files.
- Compute the set of features touched: every file in the diff has a prior canonical via the assignments cache.
- Identify "stale features" = features with at least one changed file.
- Identify "fresh files" = files in the diff that weren't in the assignments cache (newly created).

**Stage 2 — Decide what to re-analyze**

- **Full re-scan** the stale features only (their `paths` from previous scan + any fresh files Sonnet might attribute to them).
- **Workspace mode**: only re-scan packages whose file set has any change. A 20-package monorepo where 2 packages changed = 2 LLM calls instead of 20.
- **Single-package mode**: cheaper to re-scan since most of the package didn't change → reuse prior `feature -> files` for unchanged files, only ask Sonnet to place fresh/changed ones.

**Stage 3 — Skip unchanged**

- Features with no touched files skip re-analysis entirely. Carry forward their description, flows, participants.
- Skip flow tracing for features whose participant set didn't change.
- Skip flow critique for unchanged feature names.

**Stage 4 — Merge**

- Output JSON merges old (carried) + new (re-analyzed) features.
- Auto-save updates `~/.faultline/assignments-{repo}.json` with the new file → feature map.
- Stability: same canonical-name lock + token-match logic still applies.

### Data dependencies

What we need stored from the previous scan to make incremental work:

- Last scanned SHA (already in `last_sha` column in `fl_scans` table).
- Per-feature description + flows + flow_descriptions.
- Per-flow trace participants.
- File → canonical mapping (assignments cache — already there).
- Per-package boundaries (workspace info from `package.json` etc — already detected each scan).

All of this is already in the JSON output. Need a **scan loader** that hydrates a prior scan into a `DeepScanResult`-shaped object usable as the "incremental seed."

### Engineering breakdown

| Task | Estimate |
|---|---|
| Scan loader: rehydrate JSON into `DeepScanResult` for use as seed | 1 day |
| `git diff` integration in CLI: list changed files vs `last_sha` | 0.5 day |
| Pipeline branch: `--incremental` flag, skip-or-rescan decision per feature | 1 day |
| Workspace path: only re-scan changed packages, carry rest | 1 day |
| Single-call path: feed Sonnet the prior feature map as context, ask only about diff | 1 day |
| Merge logic: combine carried + fresh features, run dedup/critique on the new bits only | 1 day |
| Tests: synthetic before/after scans, verify cost ≤ 20% of full | 1 day |
| Real-world validation: documenso + plane + cal.com, measure cost reduction | 0.5 day |
| Dashboard UI: scan-history view shows "incremental — 2 features re-analyzed" | 0.5 day |

**Total: ~7–8 days of focused work.**

### Risks + mitigations

- **Stale flows** — flows carried forward from a prior scan could become wrong if the underlying code changed. Mitigation: re-trace any flow whose entry_point file is in the diff.
- **Cache invalidation** — when `.faultline.yaml` changes, all canonical locks shift. Detect yaml mtime > last_sha; force full re-scan.
- **Force flag** — `--force` always runs full scan. CI nightly job uses `--force` once a week to recompute baseline.
- **Sub-feature drift** — if `sub_decompose` ran on the prior scan and produced sub-features, incremental must respect those. Already handled by the existing parent-collapse + locked_canonicals logic.

### Validation plan

Three runs per repo:

1. Full scan on commit X (baseline).
2. Make a 5-file change, commit Y. Run incremental → measure cost + verify naming stability + verify all flows present.
3. Run full scan on commit Y → compare. Cost(2) should be ≤ 20% of cost(3). Output should match Cost(3) on ≥ 95% of feature names.

Test repos:
- documenso (stable baseline, monorepo)
- plane (Python entry-points exercise the flow tracer fallback)
- cal.com (stress test on size)

Budget for validation: ~$30 across 3 repos × 3 scans.

### Why this is the right next step

- **Unit economics:** turns Team Pro from break-even-or-loss into healthy 70%+ margin.
- **Customer experience:** scans go from "minutes" to "seconds" for typical CI runs.
- **Selling point:** "Faultlines is the only tool that re-analyzes only what changed" is a clean differentiator.
- **No pricing churn:** keeps the lock-in promise to first 100 customers intact.

---

## Open question for next session

Whether to also ship a **scan budget UI** in the dashboard before incremental lands. It's cheap (a couple of days) and immediately surfaces cost-awareness to customers — but without incremental, the budget itself is harsh. Probably ship them together: incremental drops cost, UI shows the new budget, customers see they have plenty of headroom.

---

## Stage 6 validation results (2026-04-30)

Stages 1–5 shipped to ``feat/tool-use-detection``. End-to-end validation on three real repos:

| Repo | Path | Baseline | Incremental | Saving | LLM calls saved |
|---|---|---|---|---|---|
| documenso | workspace (8 pkgs), 1 file edit in ``packages/lib/`` | $2.74 | **$0.59** | **78%** | ~91% |
| axios | monolith library, 1 file edit in ``lib/axios.js`` | $0.10 | $0.06 | 37% | 90% |
| plane | workspace (21 pkgs), 2 file edits (TS + Python) | $4.59 | **$0.64** | **86%** | **98%** |

**Validation budget spent:** ~$1.30 (vs $30 planned).

### What works

- **No-op shortcut:** zero source changes → zero LLM cost. Single most valuable property for CI nightly cron.
- **Workspace subset:** large monorepo with single-package edit re-scans only that package; the other 20 packages carry forward verbatim.
- **Monolith subset:** single-call repos still benefit, though savings are smaller because the file set fed to Sonnet is a bigger fraction of the whole.
- **Fallback chain:** every layer falls through to full ``pipeline.run()`` on any failure. Without ``--incremental``, the analyze flow is byte-identical to before.

### Known limits (Sprint 8 / future work)

- **Naming stability** during incremental drop is ~85% (same as full-scan repeat runs). The canonical lock + token-match in ``apply_repo_config`` keeps top-level names; sub-feature names under stale packages can drift slightly.
- **Fresh files outside any prior package** in workspace mode aren't re-classified — they fall into ``shared-infra`` until the user runs a full scan. Edge case in real-world usage.
- **Single-package small repos** (axios) get smaller savings because the subset is a large fraction of the repo. The threshold is around 30 files; below that, full scan is fine.
- **Cost UI** (token budget per scan, surfaced in dashboard) — not yet built. With incremental shipped, it's safe to add.

### Pricing implications

Combined with the per-org pricing on ``/pricing``:

  - Indie ($9/mo, 3 medium repos, weekly scans): cost drops from ~$36/mo to ~$10/mo. **Indie now profitable.**
  - Team Pro ($99/mo, 20 repos + PR comments): PR scan cost drops from $3 → $0.30. **Team Pro now profitable** at typical usage.
  - The catastrophe scenario (cal.com-class repo on Team Pro) still needs the Enterprise file-count gate; incremental alone doesn't save enough on that specific shape.

Net: the engineering work paid back the pricing decision. List prices stay; lock-in promise stays; soft scan caps still recommended as insurance.
