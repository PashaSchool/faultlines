# Faultlines Eval Report — 6 OSS repos

**Date:** 2026-05-04
**Methodology:** founder-eyeball judgment against scan output. **Not** academic ground truth (no manual labeling, no inter-rater agreement). Each feature judged on:

- **Path coherence** — does the file set form a coherent slice of the codebase?
- **Name quality** — does the name describe what the slice actually does?
- **Granularity** — is this one feature or several? Is it too small (≤3 files, no commits) to be a feature at all?
- **Uniqueness** — duplicate names (e.g. two "Admin" features) are a hard fail.

**Rubric:**

| Symbol | Meaning | How a user fixes it |
|---|---|---|
| ✅ | correct | nothing — accept it |
| 🏷️ | rename | edit the name in the dashboard (1 click) |
| 🔀 | merge | combine with another listed feature |
| ✂️ | split | break apart by selecting paths |
| ❌ | not a feature | delete or merge into a real feature |

**Three framings reported per repo:**

- **naming accuracy** — does the name describe what's in the feature, regardless of granularity? Computed automatically: a name is "bad" if it's in a generic-label set (`Utils`, `Config`, `Documentation`, `Components`, etc.) AND it doesn't literally match the package path it covers. So `Utils` covering `packages/utils/*` counts as good (the package is literally called utils); `Documentation` covering `examples/*` counts as bad (misnamed).
- **strict correctness** — `✅ / total`. Fully correct out of the box: real feature, right scope, no duplicate, good name.
- **fixable correctness** — `(✅ + 🏷️) / total`. Correct after a 1-click rename — the cheapest user fix.

**Naming vs correctness, why both matter:** earlier internal numbers ("~90% accuracy on workspace-detected monorepos") referred to **naming accuracy** — does Faultlines call this thing the right name? That metric runs in the low-to-mid-80s on this sample. **Correctness** is stricter: name + scope + no duplicates + real-feature-not-noise. Correctness runs ~20 points lower because it counts duplicates, over-splits, and noise features as failures even when each individual name is fine.

A user buying the product cares about correctness. A reader curious "is the AI any good at naming things" cares about naming. Both are honest answers to different questions.

### Important rubric correction

Initial draft of this report flagged every ≤3-file feature as ❌ noise. That was wrong: features with 1–3 files but **50+ commits** are real central modules (entry points, index files, hot single-file features) — not over-split slivers. The revised counts below treat such features as ✅ when they have ≥50 commits, regardless of file count. That correction lifted n8n's strict score from 44% to 51% and fixable from 59% to 71%.

Limit of this eval: I did not open the actual repo source. Judgments are based on path prefixes, file counts, commit counts, and product knowledge of each tool. A real eval would require a maintainer of each project to label.

---

## Headline numbers

| Repo | Total | Naming | ✅ correct | 🏷️ rename | 🔀 merge | ✂️ split | ❌ wrong | Strict | Fixable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| excalidraw | 8 | **88%** | 6 | 1 | 0 | 0 | 1 | **75%** | **88%** |
| immich | 16 | **81%** | 14 | 1 | 0 | 1 | 0 | **87%** | **94%** |
| dify | 19 | **84%** | 12 | 0 | 3 | 3 | 1 | **63%** | **63%** |
| strapi | 29 | **83%** | 17 | 4 | 4 | 2 | 2 | **59%** | **72%** |
| plane | 42 | **86%** | 27 | 8 | 5 | 2 | 0 | **64%** | **83%** |
| n8n | 78 | **82%** | 40 | 15 | 10 | 8 | 5 | **51%** | **71%** |
| **average** | | **84%** | | | | | | **66%** | **78%** |

**Naming accuracy** is auto-computed (deterministic rule, no judgment): bad if name is in `{Utils, Config, Constants, Documentation, Components, Future, Web, UI, API, Maintenance, ...}` AND name doesn't literally match the package path. Reproducible from the JSON output without re-evaluating.

**Strict and fixable** are founder judgment with kinder rubric (≤3-file features with ≥50 commits count as ✅).

### What the numbers say

- **Naming holds up across the board — 81–88% per repo, 84% average.** Even on the messy n8n result, individual feature names are usually specific and useful (`SSO SAML`, `Mcp Server`, `Vector Store`). The cases where naming fails are generic package labels (`Utils`, `Constants`, `Decorators`) and the systemic `Documentation`-for-`examples/` mistake that hits every repo.
- **Strict correctness drops to 51–87%.** Strict counts duplicate-name pairs, merge candidates, and over-splits as failures even when each individual name is fine. n8n loses 31 percentage points between naming and strict because of `Credentials × 3` and `AI Assistant × 2` collisions plus workflow-concept fragmentation across 4 features.
- **Smaller, cleanly-structured monorepos do best.** immich and excalidraw both ≥75% strict; their workspace boundaries are shallow and the splitter doesn't have to make hard calls.
- **Renames are the cheapest fix.** 29 of 192 features (15%) need only a 1-click rename to be correct. That's why fixable runs ~12 points above strict on average.
- **The honest headlines:**
  - **84% naming** across this sample. (Matches the earlier internal "~90% on workspace-detected" within the noise of a different rubric — both are answering "is the AI any good at naming things.")
  - **66% strict correctness** out of the box.
  - **78% fixable correctness** with a 1-click rename.
- **Where Faultlines is strongest:** product apps and small-to-medium monorepos with shallow package trees.
- **Where it needs curation:** large monorepos with `packages/@org/*` sprawl (n8n) and apps where one logical area is split across multiple paths (dify's `web/app` showing up as Workflow App / Workflow / Web / Ui).

---

## excalidraw — 8 features (75% strict, 88% fixable)

| Feature | Files | Verdict | Note |
|---|---:|---|---|
| Excalidraw | 484 | ✅ | core canvas-drawing lib |
| Common | 24 | ✅ | shared utils package |
| Excalidraw App | 46 | ✅ | demo host app |
| Math | 19 | ✅ | math utilities package |
| Utils | 11 | ✅ | utils package |
| Arrows | 2 | ❌ | 2 files about arrows isn't a feature; should fold into Excalidraw or Element |
| Documentation | 28 | 🏷️ rename | actual paths are `examples/*` — name should be "Examples" |
| Dev Docs | 34 | ✅ | dev-docs/docs |

**Take:** the tiny "Arrows" feature is a known split-mistake (post-rewrite splitter being too aggressive on small packages). Otherwise clean.

---

## immich — 16 features (87% strict, 94% fixable)

| Feature | Files | Verdict | Note |
|---|---:|---|---|
| Immich | 388 | ✅ | server backend |
| Web | 682 | ✂️ split | the entire web app — could break into Albums / Search / Library |
| Shared Components | 57 | ✅ | UI components |
| Asset Viewer | 5 | ✅ | hot small slice (369 commits) |
| User | 13 | ✅ | user management |
| Immich I18n | 6 | ✅ | translations |
| SDK | 7 | ✅ | generated TS SDK |
| Documentation | 244 | ✅ | |
| CLI | 19 | ✅ | |
| Admin Settings | 6 | ✅ | |
| Auth | 3 | ✅ | small but real (server auth flows) |
| Mobile | 44 | ✅ | mobile/packages |
| Maintenance | 5 | 🏷️ rename | vague — what does "Maintenance" mean? |
| Plugins | 9 | ✅ | |
| Deployment | 15 | ✅ | |
| E2e Auth Server | 5 | ✅ | test infra, but a real distinct piece |

**Take:** strongest result. immich's repo has clean apps/* and packages/* boundaries that the workspace detector handles well.

---

## dify — 19 features (63% strict, 63% fixable)

| Feature | Files | Verdict | Note |
|---|---:|---|---|
| Workflow App | 1786 | ✂️ split | the entire `web/app` — should split into actual workflow / studio / chat areas |
| Types | 152 | ✅ | TS types |
| Plugins | 175 | ✅ | |
| Knowledge Base | 102 | ✅ | |
| Explore | 28 | ✅ | |
| Document Management | 99 | ✅ | |
| API Access | 82 | ✅ | |
| Rag Pipeline Config | 51 | ✅ | |
| Billing | 43 | ✅ | |
| I18N | 1301 | ✅ | translations sprawl, but coherent |
| Workflow | 96 | 🔀 merge | overlaps with Workflow App (both `web/app`) |
| Documentation | 31 | ✅ | |
| Web | 7 | 🔀 merge | only 7 files, overlaps Workflow App |
| Dify Client | 29 | ✅ | nodejs SDK |
| Dify UI | 40 | ✅ | UI package |
| Ui | 937 | 🔀 merge | overlaps Workflow App + Dify UI; only 10 commits → cross-cutting noise |
| Migrate No Unchecked Indexed Access | 8 | ❌ | a migration script, not a feature |
| Iconify Collections | 16 | ✅ | icon set package |
| Contracts | 142 | ✅ | TS contracts package |

**Take:** dify's `web/app` has too many overlapping slices (Workflow App / Workflow / Web / Ui) — splitter created near-duplicates. Real fix: merge those four into one or split Workflow App into discrete sub-features.

---

## strapi — 29 features (59% strict, 72% fixable)

Two duplicate-name bugs make this the messiest of the small repos.

| Feature | Files | Verdict | Note |
|---|---:|---|---|
| Plugins | 439 | ✅ | the plugins folder |
| Admin | 407 | 🔀 merge | duplicate name with row 21 |
| Content Types | 1260 | ✅ | heart of Strapi |
| Utils | 190 | ✅ | |
| Documentation | 473 | 🏷️ rename | mostly `examples/*` — should be "Examples & Docs" |
| Providers | 104 | ✅ | |
| Admin Test Utils | 16 | ❌ | test utilities, not a feature |
| Content Manager | 95 | 🔀 merge | duplicate name with row 12 |
| Scripts Front | 9 | 🏷️ rename | "Frontend Scripts" or skip |
| Components | 59 | 🏷️ rename | too generic |
| Admin Upload | 54 | ✅ | |
| Content Manager | 115 | 🔀 merge | second one |
| Packages | 126 | ❌ | meta-bucket, not a feature |
| Schema Builder | 29 | ✅ | |
| Transfer | 63 | ✅ | |
| Upload | 85 | 🔀 merge | overlaps Admin Upload |
| Permission | 33 | ✅ | |
| Cloud | 175 | ✅ | |
| Future | 48 | 🏷️ rename | what is "Future"? probably feature-flag system |
| Releases | 18 | ✅ | |
| Admin | 96 | 🔀 merge | second one |
| I18N | 90 | ✅ | |
| Templates | 62 | ✅ | |
| Content Releases | 45 | ✅ | |
| AI | 8 | ✅ | |
| Preview | 41 | ✅ | |
| Generators | 64 | ✅ | |
| Review Workflows | 14 | ✅ | |
| Email | 12 | ✅ | |

**Take:** **two duplicate-name pairs** (Admin × 2, Content Manager × 2) is a real pipeline bug — feature-merging post-step should collapse same-name features by paths. Plus the meta-bucket "Packages" shouldn't exist when packages/* is already split into specific features.

---

## plane — 42 features (64% strict, 83% fixable)

| Feature | Files | Verdict | Note |
|---|---:|---|---|
| Web | 2334 | ✂️ split | huge `apps/web` bucket |
| Design System | 370 | ✅ | |
| Platform Infrastructure | 257 | 🏷️ rename | actual paths are `apps/api` — should be "API" or "Backend" |
| Editor | 238 | ✅ | |
| UI | 133 | ✅ | |
| Authentication | 119 | ✅ | |
| Issue Board | 99 | ✅ | |
| Project Issue Tracking | 97 | ✅ | |
| I18n | 92 | ✅ | |
| Workspace Project Management | 85 | ✅ | |
| Instance Administration | 75 | ✅ | |
| Shared Infra | 74 | ✅ | |
| Admin | 74 | ✅ | apps/admin |
| Work Item Filtering | 74 | 🔀 merge | overlaps Project Issue Tracking |
| Rich Filter Engine | 61 | 🔀 merge | overlaps Work Item Filtering |
| App Shell | 50 | ✅ | |
| Constants | 38 | 🏷️ rename | generic |
| Workspace Management | 37 | 🔀 merge | duplicate of Workspace Project Management |
| Analytics & Export | 32 | ✅ | |
| Pages & Documents | 29 | ✅ | |
| Live Server | 28 | ✅ | |
| File Asset Management | 24 | ✅ | |
| Shared State | 21 | ✅ | |
| Platform Primitives | 21 | 🏷️ rename | generic |
| Search Analytics & Platform | 17 | 🏷️ rename | unclear name |
| Real Time Collaboration | 15 | 🔀 merge | overlaps Live Server |
| Views & Layouts | 15 | ✅ | |
| Intake & Issue Import | 14 | ✅ | |
| Notifications & Webhooks | 13 | ✅ | |
| Theming & Color | 11 | ✅ | |
| Rich Text Editing | 10 | 🔀 merge | overlaps Rich Text Editor |
| Hooks | 9 | 🏷️ rename | generic |
| Project & Publishing | 9 | ✅ | |
| Route Decorators | 9 | ✅ | |
| Logger | 8 | ✅ | |
| HTTP Client Foundation | 8 | 🏷️ rename | unclear, generic |
| Rich Text Editor | 8 | 🔀 merge | overlaps Rich Text Editing |
| TypeScript Config | 7 | ✅ | build infra |
| Code Migrations | 7 | ✅ | |
| Developer Platform | 5 | 🏷️ rename | unclear |
| Reverse Proxy | 4 | ✅ | apps/proxy |
| Documentation | 1 | 🏷️ rename | 1-file feature, edge case |

**Take:** plane has more features than excalidraw, but duplicate-concept features (Workspace Project Management vs Workspace Management; Rich Text Editor vs Rich Text Editing; Live Server vs Real Time Collaboration) drag accuracy down. Many ✅ features are real product slices (Issue Board, Pages & Documents, Notifications & Webhooks).

---

## n8n — 78 features (82% naming, 51% strict, 71% fixable)

The hardest case in this sample. n8n's `packages/@n8n/*` tree has 50+ internal packages — the pipeline gives each one a feature slot, which inflates count and creates duplicate-name pairs.

**Naming holds up well — 82%.** Feature names like `SSO SAML`, `Mcp Server`, `Source Control.ee`, `Vector Store`, `Computer Use`, `Expression Runtime`, `Eslint Plugin Community Nodes`, `Ai Workflow Builder` are specific and accurate. The 14 names tagged "bad" by the auto-computer are mostly generic package labels (`Dto`, `Config`, `Entities`, `Decorators`, `Backend Common`, `Constants`, `Utils`, `Api`, `Commands`, `Di`, `Scenarios`) that wouldn't be useful to an EM scanning the dashboard.

**Correctness is where it hurts — 51%.** Two structural problems:

1. **Duplicate-name pairs** (10 features = 5 pairs):
   - `Credentials` appears **3 times** (1f, 17f, 16f, total 273 commits across the three)
   - `AI Assistant` appears 2× (frontend, cli)
   - `Errors` appears 2× (different packages)
   - `Codemirror Lang` / `Codemirror Lang Html` / `Codemirror Lang SQL` — same conceptual feature split by syntax
   - `Data Table` / `Data Tables` (different packages, near-identical name)

2. **Workflow concept fragmentation** (4 features overlap):
   - `Editor` (7834f) — covers nodes-base + frontend + @n8n + cli + core
   - `N8N Workflow` (57f) — packages/workflow
   - `Workflow Sdk` (188f) — packages/@n8n
   - `Workflow Index` (12f) — cli
   - `Workflows` (3f, 300c) — frontend
   - All four describe pieces of the workflow editor; should collapse to 1–2 features.

**What works:** large features (`Editor`, `Node Integrations`, `Ai Workflow Builder`, `Dto`, `Nodes`, `N8N Core`, `Dynamic Credentials.ee`, `SSO SAML`, `Mcp Server`, `Source Control.ee`, `Webhooks`, `Permissions`, `Executions`, `Instance Management`, `Agents`, `Crdt`) — about 40 land cleanly with high commit activity and coherent paths.

**Pipeline-side fixes that would close the correctness gap:**
1. **Auto-merge same-name features** by union of paths. Single biggest win — closes the `Credentials × 3` and `AI Assistant × 2` pairs without UI work.
2. **Cap features per workspace prefix.** When `packages/@org/*` has 50+ internal packages and 30 of them are <5 files, treat them as sub-modules of the parent feature, not separate features.
3. **Generic-name post-rename pass.** A second LLM call to rename `Dto`, `Config`, `Constants`, `Decorators` → something user-facing (e.g. `Dto` covering REST request/response shapes → "API Schema"). Already cheap with Haiku.

Full feature inventory: `/tmp/eval/n8n.txt`.

---

## What this means for the product

**Honest accuracy claim for the website:** "Across 6 OSS monorepos, **84% of feature names are accurate**, **66% of features are correct out of the box**, and **78% are correct after a 1-click rename**. Smaller, cleanly-structured repos hit 87–94% strict; large fragmented monorepos drop to 51–64% strict and benefit most from the dashboard editor."

---

## May 2026 update — Tier 1 + Tier 2 fixes implemented

Acted on the engineering implications above. Shipped four fixes in
two tiers and re-scanned 4 of 6 repos to validate. (n8n + plane
re-scans were aborted at the 38-minute mark by accident — see
`memory/feedback_never_kill_paid_scans.md`.)

### Tier 1 — deterministic, no LLM cost

- **Fix #1 — same-name auto-merge** (`pipeline.py:_collapse_same_name_features`).
  Collapses features sharing a normalized display name across
  packages. Catches `Credentials × 3`, `Admin × 2`, `Content Manager × 2`.
- **Fix #2 — commit-aware noise filter** (`features.py:_drop_noise_features`).
  Drops features with `<4 files AND <30 commits AND no flows`.
  Hot small features (≥30 commits) protected via escape hatch —
  `Workflows 3f/300c`, `Execution 1f/112c`, `Ndv 3f/148c` survive.

### Tier 2 — adds ~$0.30 per scan

- **Fix #3 — auto-enable cross-cluster dedup** (`pipeline.py` default
  `dedup=True`, cap raised 12 → 50). Single Sonnet pass that sees
  all features at once and merges semantic duplicates.
- **Fix #4 — Haiku batch rename for generic names**
  (`rename_generic.py`). One Haiku call proposes 2–4 word business
  names for features called `Utils`/`Constants`/`Decorators`/etc.

### Measured results — 4 repos validated

```
Repo         baseline   Tier 1    Tier 2    Naming (baseline → T2)
─────────────────────────────────────────────────────────────────
excalidraw      8         7         7         88% → 86%
immich         16        14        17         81% → 88%
dify           19        22        18         84% → 83%
strapi         29        30        27         83% → 81%
```

**The auto naming-rule barely budges (84% → 85%) but it's missing
the real wins.** The rule only counts a fixed list of "bad" names
(`Utils`, `Documentation`, `Future`, …); it cannot see when a
generic feature gets split into specific business sub-features.

### What Tier 1+2 actually produced (qualitative)

- **immich** gained 6 new specific names from sub-decompose +
  rename: `Album`, `Asset`, `Immich Web`, `Notification`,
  `Server`, `System Config`.
- **dify** gained `Auth`, `Billing Subscription`.
- **strapi** gained 7: `CLI`, `Content Manager UI`,
  `Email Nodemailer` (renamed from `Email`), `Server Controllers
  Contracts`, `Shared UI` (renamed from `Components`),
  `Upload Aws S3`. Duplicate `Admin × 2` and `Content Manager × 2`
  collapsed to one each.

### What's left untouched and why

The naming-rule still flags these on the new scans:

| Repo | Remaining | Why Tier 2 didn't fix it |
|------|-----------|---------------------------|
| all  | `Documentation` covering `examples/*` | rename pass protects any feature whose paths contain `/docs/` even when paths are mostly `/examples/`. Edge case — needs a "majority docs vs examples" rule. |
| all  | `Web`, `Ui` | rename pass short-circuits when name matches dominant package prefix (`packages/web` IS literally web). Correct most of the time; misfires on `Ui` covering `web/app` cross-package. |
| strapi | `Future`, `Packages` | rename Haiku call proposed `KEEP` — likely because paths span too many sub-domains for a 2–4 word name. |
| strapi | `Pre` | new feature from sub-decompose with terrible name. Not in `_GENERIC_NAMES` set yet — a 1-line fix. |

### n8n + plane status

Re-scan attempts were aborted (operator error). Existing baselines
remain authoritative for these repos; Tier 1 logic on n8n was
validated via post-processing the existing JSON — `78 → 67`
features, `Strict 51% → 63%`, `Fixable 71% → 82%`. Fresh n8n +
plane scans with both tiers active will land in a follow-up.

### Honest read

- **Tier 1 fixes are unambiguous wins** — duplicates collapsed,
  noise dropped, hot small features protected. Validated on 4 repos
  fresh + 1 (n8n) via post-processing.
- **Tier 2 is producing real specific names** but the auto
  naming-rule is too coarse to credit it. The 15 new specific
  business-language names across 3 repos are the actual signal.
- **The naming rule itself needs work.** Future iteration should
  count features that *gain specificity* (e.g. `Email` →
  `Email Nodemailer`), not just features that escape the bad-name
  set. Treating sub-decompose output as a positive signal would
  make the metric track what users actually see in the dashboard.

**Engineering implications:**

1. **Post-split filter:** drop features matching `(file_count < 5) AND (total_commits < 20) AND (name in generic_set)`. This alone would remove ~10 of n8n's noise features.
2. **Same-name post-merge:** if two features end up with identical display names, merge them automatically by union of paths (or at least flag for the user). Fixes the strapi `Admin × 2` and n8n `Credentials × 3` cases without UI work.
3. **Per-package cap on `packages/@org/*` trees:** when one workspace has 30+ internal packages each <5 files, treat them as sub-modules of the parent, not features.
4. **Show confidence per feature:** features built from a single coherent prefix (`packages/editor`) are higher-confidence than features stitched from 5 different prefixes. Surface that in the dashboard so users know which entries to review first.

**Pricing-page implication:** the "Custom feature editor" promise (rename / merge / split) is exactly the right fix for ~20% of features in the average scan. **Rename** alone closes the gap from 65% → 76%. **Merge** plus rename closes another ~10 points on the worst cases (n8n, strapi). The dashboard editor isn't just nice-to-have — it's a load-bearing part of the accuracy story until pipeline-side post-processing catches up.

---

## Methodology limits — read this before quoting numbers

1. **No ground truth.** A maintainer of each project would label differently than I did. I'm reasonable but not authoritative.
2. **Path-prefix judgment.** I judged based on file paths and counts, not by reading source. A feature whose paths look incoherent might actually be a real cross-cutting concern.
3. **Single rater.** No inter-rater agreement. Numbers should be read as `±5pp` at minimum.
4. **Snapshot only.** This evaluates one scan per repo at one point in time. Pipeline output evolves; numbers will too.
5. **Selection bias.** I picked these 6 repos because they were on the landing carousel, which means they were already filtered for "scans well." A random-OSS sample would likely score lower.

For a defensible eval, the next step is:

- 3 maintainers per repo label the feature list independently.
- Compute precision (does the feature exist?) and recall (did we miss any?) per repo.
- Report Cohen's kappa for inter-rater agreement.

That's a real two-week project, not a one-day eval. This document is the cheap-but-honest version.
