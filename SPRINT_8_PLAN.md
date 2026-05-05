# Sprint 8 — Smart Aggregator Detection

**Branch:** `feat/sprint8-aggregator-detection`
**Started:** 2026-05-05
**Goal:** Faultlines stops emitting fake "features" for shared
infrastructure (DTO packages, shared UI, utils, locales) and
re-attributes their files + flows to the real product features
that consume them. Every surviving feature must be readable by a
CTO without translation.

## Why this sprint

Eval on 11 OSS repos surfaced a recurring failure mode that's
not just a naming bug — it's a category mistake. The pipeline
groups files by package directory and emits each as a feature.
That works for `apps/web` (real product surface) and breaks for
`packages/api-types/dto/` (multi-domain schemas) or
`packages/shared-ui/` (UI primitives used by every feature).

Concrete examples we measured:

- **n8n's `Dto`** — 78 files, 397 commits, 24 flows. Flows include
  `embed-login-flow`, `create-workflow-flow`, `configure-ai-gateway-flow`.
  Each flow's name is correct. None of them belong to "Dto" — they
  belong to Auth, Workflows, AI Gateway respectively. The flow
  detector found the journey from a DTO file (`embed-login-body.dto.ts`)
  but the journey lives in the consuming feature.
- **strapi / cal.com `Shared UI`** — Button, Modal, Dropdown components
  used across 10+ features. Currently shows up as one feature called
  "Shared UI". A CTO opening the dashboard sees "Shared UI" and learns
  nothing about the product. The button file should appear as a
  participant in every feature that uses it.
- **n8n `i18n`, `docs`, `assets`, `dev-tools`** — developer-facing
  internals. Devs maintain them; product owners don't think about
  them as features. Either fold into one labeled bucket or rename
  to plain English ("Translations", "Internal Documentation").

Net: file-level `1:1 file→feature` mapping is the wrong abstraction
for shared infrastructure. We need `N:M` for shared participants
plus business-language naming on what survives.

## What changes

### 1. Four-bucket classification (LLM-driven)

A new Sonnet pass runs after Tier 1 + Tier 2 fixes. It sees every
detected feature as `(name, sample_paths, sub-folder distribution,
flow names)` and classifies each into:

| Class | Definition | Pipeline action |
|-------|-----------|-----------------|
| `product-feature` | Files all serve one user-facing concept | Keep, possibly rename for CTO clarity |
| `shared-aggregator` | Files are shared across multiple product features (DTOs, UI primitives, schemas) | Delete the feature; files become `shared_participants` of consuming features |
| `developer-internal` | Real maintenance area but not a product feature (i18n, docs/internal, assets, dev-tools, e2e infra) | Rename to plain English OR fold into one `developer-infrastructure` bucket |
| `tooling-infra` | Build/lint/test configs | Folded into `shared-infra` (existing `_auto_fold_tooling`) |

The LLM does NOT get a pre-defined list of "aggregator names". It
classifies based on what the paths and flows reveal, drawing on its
training data of typical codebase structures. Naming is one signal
among many; it's not authoritative.

### 2. Multi-owner file model

`Feature.paths` continues to mean "files this feature owns" (1:1).

New: `Feature.shared_participants: list[SharedParticipant]` lists
files that the feature USES but doesn't own. A `Button.tsx` from
shared-ui gets listed as `shared_participant` in every feature that
imports it.

Owned file → full weight in metrics (commits, bug ratio).
Shared participant → partial weight via existing line-attribution
(if a feature uses 12 of 100 lines, contribute 0.12 weight).

### 3. Consumer resolution via Sprint 7 callgraph

For each file in a classified `shared-aggregator`, walk the import
graph (Sprint 7 already builds this for flows) to find which
features' files import it. The file becomes a participant in each
consumer.

Fallbacks when callgraph misses (dynamic imports, unresolved):
- Filename heuristic match (`embed-login-body.dto.ts` → look for
  `embed-login` symbols in other features)
- Last resort: file folds into `shared-infra`

### 4. Flow re-attribution

For each flow currently attached to a `shared-aggregator`:
- Examine `flow.participants` (Sprint 7 trace data)
- Group participants by their current feature owner
- If ≥60% of participants live in one consumer feature → move flow
  there
- If no clear majority → drop flow (it was noise from misattribution)

### 5. CTO-readable rename pass (Day 6)

After classification + re-attribution, every surviving feature gets
a final check: "would a CTO understand this name in 3 seconds?"

A second LLM pass (Haiku, cheap) reviews each feature and proposes
business-language replacements for developer-jargon names that
slipped through:

- `Decorators` → "Permission Decorators" (if scope is access control)
- `Backend Common` → "Workflow Helpers" (if scope is workflow utilities)
- `Pre` → drop or rename
- `Future` → "Feature Flags" (if that's what it actually is)

Original name preserved as alias so existing references resolve.

## Pipeline order after Sprint 8

```
Stage 1.4   _auto_fold_tooling                 (existing, deterministic)
Stage 1.45  _collapse_same_name_features       (Tier 1, deterministic)
Stage 1.5   dedup_features                     (Tier 2, default-on)
Stage 1.55  rename_generic_features            (Tier 2, opt-in after revert)
Stage 1.6   ⭐ smart_aggregator_detection      (Sprint 8, default-on)
Stage 1.7   sub_decompose_oversized            (existing, opt-in)
Stage 1.9   critique_and_refine                (existing, opt-in)
Stage 1.95  ⭐ cto_readable_rename             (Sprint 8, default-on)
Stage 2     materialize synthetic buckets
```

## Cost

- **Stage 1.6** — one Sonnet call per scan, batched with all
  candidates. Estimated $0.10–0.20 per scan.
- **Stage 1.95** — one Haiku call per scan. Estimated $0.001–0.005.
- **Total Sprint 8 add:** ~$0.15–0.25 per scan. On top of typical
  $1–3 scan cost it's <10%.
- Cached in incremental mode → cost amortizes to $0 on subsequent
  scans of the same repo.

## What can break

1. **`_validate_source_coverage`** currently asserts every source file
   appears in exactly one feature. New: file is "covered" if it's an
   owned path OR a shared_participant anywhere. Need careful test.
2. **Sprint 7 callgraph not 100%** — some imports unresolved (dynamic,
   re-exports, namespace imports). Mitigation: filename fallback +
   final fold to shared-infra.
3. **LLM mis-classifies** — could call a real product feature an
   aggregator. Mitigation:
   - Confidence threshold (only act when LLM expresses high
     confidence)
   - User override via `.faultline.yaml` (force-product /
     force-aggregator lists)
4. **Multi-owner metrics** — split full credit across consumers vs
   give full credit to each? Decision: use existing line-scoped
   attribution for shared participants; owned paths get full weight
   as today.
5. **Flag rollout** — ship behind `--smart-aggregators` (default off)
   first, eval, then promote to default-on if numbers improve.

## Day-by-day

### Day 1 — LLM classifier + unit tests (no pipeline wiring)

Build `faultline/llm/aggregator_detector.py`:

- `classify_features(result: DeepScanResult, *, api_key, model,
  tracker) -> dict[str, FeatureClassification]`
- One Sonnet batch call with all features
- Returns per-feature: `class`, `reasoning`, `consumer_features`
  (when aggregator), `proposed_name` (when rename suggested)
- System prompt embeds the 4-bucket rubric + examples + "CTO test"

Tests on synthetic fixtures:
- Real product feature gets `product-feature`
- DTO package gets `shared-aggregator` with multi-domain reasoning
- i18n locale folder gets `developer-internal` with proposed name
- tsconfig package gets `tooling-infra`
- Edge: legit i18n admin UI feature gets `product-feature` (not
  fooled by name alone)

### Day 2 — Callgraph extension for consumer resolution

Extend `faultline/analyzer/flow_tracer.py` (or new sibling module):

- `find_consumers(file_paths: list[str], result: DeepScanResult,
  repo_root: Path) -> dict[str, list[str]]`
- For each file in input, returns list of feature names whose owned
  files import it
- Reuses existing import-graph infrastructure
- Fallbacks: filename match → shared-infra

Tests with synthetic import graphs.

### Day 3 — Multi-owner model + validation

`faultline/models/types.py`:

```python
class SharedParticipant(BaseModel):
    file_path: str
    role: Literal["consumer", "co-owner"] = "consumer"
    line_weight: float = 1.0  # from line-scoped attribution

class Feature(BaseModel):
    paths: list[str]              # owned (existing)
    shared_participants: list[SharedParticipant] = []  # NEW
    ...
```

Relax `pipeline._validate_source_coverage`:
- A file passes validation if found in any feature's `paths` OR
  any feature's `shared_participants[*].file_path`
- Aggregator-owned files that lost their owning feature must
  appear as shared_participant somewhere or fold to shared-infra

Tests for the new invariants.

### Day 4 — Flow re-attribution + dev-internal handling

New module `faultline/llm/aggregator_apply.py`:

- `apply_classifications(result, classifications, consumer_map) ->
  result`
- Consumes Day 1 classifications + Day 2 consumer map
- For each `shared-aggregator`:
  - Distribute owned paths as `shared_participants` in consumers
  - Delete the aggregator feature
  - For each flow on the aggregator: re-attribute via Sprint 7
    participants (≥60% rule) or drop
- For each `developer-internal`:
  - Rename to LLM-proposed business term
  - OR (if no good rename) fold into `developer-infrastructure`
    bucket
- For each `tooling-infra`:
  - Already handled by `_auto_fold_tooling`; idempotent here

Tests: synthetic input with one of each class → expected output.

### Day 5 — E2E validation on real scans

Wire Stage 1.6 + 1.95 into `pipeline.run()` behind
`--smart-aggregators` flag (default OFF on Day 5).

Re-scan 4 small/medium repos: dify + immich + supabase + ghost.
For each:
- Compare baseline / Tier 1 / Tier 2 / Sprint 8 outputs
- Verify Dto / Shared UI / i18n features are gone or renamed
- Verify previously-misattributed flows now live in correct features
- Verify no flows or owned files are silently lost
- Document deltas

### Day 6 — CTO-readable rename + final promotion

Implement Stage 1.95 (`cto_readable_rename`):
- One Haiku call reviewing every surviving feature name
- Propose rename only when the name fails the "CTO test"
- Apply with original kept as alias

Final scan pass on dify/immich/supabase/ghost + n8n.
Update EVAL_REPORT.md with measured before/after.
If avg Strict lifts ≥5pp without regressions → promote
`--smart-aggregators` to default-on.
Update landing block with new numbers.

## Eval expectation

Per the design discussion, conservative estimate after Sprint 8
on 11 repos:

| Metric | Before | After Sprint 8 |
|--------|-------:|---------------:|
| Strict | 79% | ~86% |
| Fixable | 90% | ~95% |
| Flow real-journey | 83% | ~90% |

Most movement on monorepos with shared schema packages (n8n, dify,
supabase, cal.com when re-scanned). Repos that were already clean
(saleor, meilisearch, gitea) move ~0–2pp.

## Rollback story

If Sprint 8 produces worse numbers in eval:
- `--smart-aggregators` stays opt-in; no production users affected
- New code lives in three new modules + one model addition; revert
  by removing the stage wiring + the `shared_participants` reads
- Existing scans (pre-Sprint-8) keep working — `shared_participants`
  defaults to empty list

## Out of scope (future sprints)

- **Cross-repo aggregator detection** — same DTO package shared
  across `frontend/` and `backend/` repos. Sprint 9.
- **User-facing override UI** — let buyer rename or re-classify in
  the dashboard, persist to `.faultline.yaml`. Already partially
  exists via feature_overrides table.
- **Co-ownership conflicts** — if two features both claim a shared
  file with strong weight, surface as a "shared component" view in
  the dashboard. Future polish.

## Working agreements

- Each day's PR ships independently with tests passing
- No skipping `_validate_source_coverage` — change it correctly
- `--smart-aggregators` stays opt-in until Day 6 eval validates
- Day 3 checkpoint with founder before continuing — verify model
  change is acceptable before locking it in
- Cost guardrail: never run a paid scan in CI; manual scans only
  with explicit budget

## Status (pause point — May 5 2026)

Days 1–5 done; Day 6 (CTO-readable rename pass + promotion to
default-on) deferred to a fresh session. The branch stays open;
`--smart-aggregators` ships opt-in default-off so no production
scan gets the unvalidated behavior.

What's working:

- Days 1–4 modules + 66 unit tests, all green
- Day 5 wiring into pipeline.run() with try/except guard
- Library mode: stage skips entirely (validated on excalidraw
  v8e — clean 7-feature output, main 484-file Excalidraw library
  preserved)
- Structural safeguards pinned in tests:
  * `_MAX_FOLD_FILES = 50` — refuses fold on big features
  * `_MAX_FOLD_COMMITS = 200` — refuses fold on active features
  * Largest-feature lock when ≥30 files
  * Caller-supplied `locked_features` frozenset
- Test-infra always-folds rule pinned in prompt + 2 unit tests
- `CostTracker.record` API correctly threaded

What's NOT validated yet (Day 6 work):

- App-repo aggregator redistribution. Day 5 v1 (with bug) silently
  fell back to Tier-2; v2 (after API fix) over-folded due to
  prompt sharpening; v3 (after structural safeguards) only
  validated on excalidraw which is library mode. **No real
  end-to-end test of `shared_participants` distribution on a
  multi-domain DTO/contracts package.**
- The CTO-readable rename pass (Stage 1.95) is not implemented —
  the prompt-level `proposed_name` from the classifier is the only
  rename signal today.

Day 6 entry conditions:

1. Run `--smart-aggregators` on dify (Contracts package) and ghost
   (large monorepo) with full structural safeguards. Expect
   `shared_participants > 0` on at least one feature.
2. If redistribution does not fire, debug by reading the actual
   classifier verdicts (currently logged at INFO level — may need
   to bump to WARNING or capture explicitly to a file).
3. Once redistribution validated, implement Stage 1.95 CTO-rename
   pass and re-run.
4. Update EVAL_REPORT.md and landing block with the new numbers.
5. Promote `--smart-aggregators` to default-on.

Approximate cost to reach Day 6 sign-off: $1–3 in Anthropic
spend depending on how many app-repo iterations are needed.

This session burn (Sprint 8): ~$6 across Days 1–5 + retries.
~$5 of that was retries / debugging the API mismatch + over-fold
+ stale yaml. Day 6 should benefit from the lessons and finish
in one round.
