# Sprint 9 — Agentic Aggregator Detection

**Branch:** `feat/sprint9-agentic-aggregators` (to be opened)
**Status:** Plan only. Sprint 8 paused on `feat/sprint8-aggregator-detection` as reference.
**Goal:** Replace the single-shot LLM classifier from Sprint 8 with a tool-augmented agentic loop that investigates each suspicious feature the same way a human reviewer would — read sample files, check imports, see who consumes whom, then decide.

## Why Sprint 8 didn't deliver

Single-shot Sonnet classification was fundamentally weaker than what's possible:

- Saw only feature names + sample paths (text)
- Couldn't read file content
- Couldn't follow imports
- Couldn't iterate / refine
- Made one classification call for all features at once

Result: across 4 dify scans the model NEVER picked `shared-aggregator` even on the canonical case (`Contracts` 142 cross-domain TS files). It read package names + path patterns and concluded "developer-internal" because that's the safest default for "files I don't have evidence about."

The user's intuition: "you can analyze code by reading and understanding because you have tools to investigate. Why can't our algorithm do the same?" Right answer: it can. We just have to give it the same tools.

## Architecture

Reuse existing `faultline/llm/tool_use_scan.py` infrastructure. Sprint 1 already built the agentic loop pattern for naming intelligence; Sprint 9 extends the same pattern for aggregator detection.

### Tools the agent will use

Faultlines already exposes these in `faultline/llm/tools.py` (used by Sprint 1):

```
read_file_head(path, lines=50)      # actual file content
list_directory(dirpath)             # folder layout
grep_pattern(pattern, path_glob)    # find usage / patterns
get_file_commits(path, limit=5)     # git history
find_route_handlers(path_glob)      # domain-specific
find_event_handlers(path_glob)      # domain-specific
```

Sprint 9 adds two import-graph tools:

```
imports_of(file)                    # files THIS one imports (Sprint 7 forward)
consumers_of(file)                  # files that import THIS one (Sprint 7 reverse)
feature_summary(name)               # paths + flows + commits per feature
```

The Sprint 7 `SymbolGraph` already has the forward/reverse edges; we just expose them via tool dispatch.

### Agent workflow per repo

Single Sonnet `messages.create` loop with tool dispatch (same pattern as `tool_use_scan`). The agent gets:

1. List of all detected features (post Tier 1+2)
2. The four-bucket rubric (product / shared-aggregator / developer-internal / tooling-infra)
3. The CTO-readable test
4. The tools above

Sample agent reasoning trace:

```
Agent: list features → sees ["Workflow App", "Contracts" (142f), "Plugins", ...]

Agent: "Contracts" is suspicious — generic name. Investigate.
  read_file_head packages/contracts/dto/auth/login.dto.ts
  → "type LoginBody = { email, password }"
  consumers_of packages/contracts/dto/auth/login.dto.ts
  → ["apps/cli/auth/login.controller.ts", "apps/web/auth/login.tsx"]
  Both imports live in feature "Authentication".

  read_file_head packages/contracts/dto/billing/plan.dto.ts
  → "type Plan = { id, name, monthly_price }"
  consumers_of → ["apps/web/billing/upgrade.tsx"] in "Billing"

  list_directory packages/contracts/dto
  → ["auth/", "billing/", "ai/", "workflow/", "data-table/"]
  Five distinct domains, none of which is named "Contracts".

  Verdict: shared-aggregator. consumer_features = [Authentication,
  Billing, AI Gateway, Workflow, Data Table]. Confidence 5.
  Reasoning: 142-file package whose sub-folders match other features'
  names; every file is a type imported by code living in those features;
  no business logic of its own.

Agent: "Workflow App" → 1714 files in web/app. Probably the main product.
  Largest feature in the scan. No need to investigate — locked.
  Verdict: product-feature, no rename, confidence 5.

... etc per feature
```

Final output is the same `dict[str, FeatureClassification]` the Sprint 8 apply step expects, but every verdict is grounded in actual code investigation.

### Pipeline integration

Stage 1.97 (the same hook Sprint 8 used) calls a new `agentic_classify_features` instead of the single-shot one. Apply step (`apply_classifications`), structural safeguards, and library-mode skip stay unchanged from Sprint 8 — those parts worked. Day 5's lessons all carry over.

```
Stage 1.97 (agentic):
  classifications = agentic_classify_features(result, repo_root, ...)
  if is_library: skip (same rule as Sprint 8)
  consumer_maps = find_consumers(...)  # Sprint 8 module, unchanged
  apply_classifications(...)            # Sprint 8 module, unchanged
```

## Cost

- Tool-use scans run multiple LLM calls per investigation. Estimate: 5-15 tool round-trips for a suspicious feature, 2-3 for an obvious one.
- Per-scan estimate: $0.50 – $2.00 for app repos. Library repos still skip the stage entirely.
- Cached in incremental mode → cost amortizes to $0 on subsequent scans.

Sprint 8 single-shot cost was ~$0.10. Sprint 9 is ~10× more expensive but produces decisions that are actually grounded.

## Day-by-day plan

### Day 1 — extend tools, add import-graph entries
- Add `imports_of` and `consumers_of` to `tools.py` dispatch table
- Add `feature_summary` tool that reads from the in-flight `DeepScanResult`
- Unit tests: each new tool with synthetic inputs
- No LLM calls; purely local

### Day 2 — agent prompt + classify loop
- New module `faultline/llm/aggregator_agent.py`
- Reuses `tool_use_scan` loop body
- Prompt embeds 4-bucket rubric, CTO test, examples of when to read/grep/check imports
- Returns `dict[str, FeatureClassification]` matching the Sprint 8 schema
- Unit tests with mocked tool responses; no real API call

### Day 3 — pipeline wiring + smoke test on dify
- Wire Stage 1.97 to call agentic classifier when `--smart-aggregators` is set
- Library mode skip stays
- Apply step + structural safeguards stay
- Run on dify (single repo, ~$1-2 cost)
- Inspect: did `Contracts` get classified as `shared-aggregator`? Did `shared_participants` fire? Compare output to Sprint 8 v8e baseline.

### Day 4 — multi-repo validation
- If Day 3 succeeds: re-scan immich + ghost + supabase (~$3-5 total)
- If `shared_participants` fires consistently: validation passes
- If still 0: debug agent prompt + tool feedback loop

### Day 5 — Sprint 9.5: CTO-readable rename pass
- Day 6 from Sprint 8, deferred. Now uses the agent — same tool access — to propose business-language renames for surviving generic names
- The agent reads sample files, decides whether to rename, returns proposals
- Apply via `_rewrite_feature_name` from critique.py (existing helper)

### Day 6 — final eval + landing update
- Update EVAL_REPORT.md with measured numbers
- If avg Strict lifts ≥5pp without regressions → promote to default-on
- Update landing eval block with new figures

## What stays the same

- `SharedParticipant` model + `Feature.shared_participants` field (Day 3 of Sprint 8)
- `apply_classifications` and structural safeguards (Day 4 of Sprint 8 + Day 5 fixes)
- `find_consumers` callgraph wrapper (Day 2 of Sprint 8)
- Library mode skip
- `--smart-aggregators` opt-in flag (default off until eval validates)

What changes is only the classifier itself — single-shot → agentic.

## What can break

1. **Tool-loop runaway cost.** A misbehaving prompt could send the agent on dozens of useless tool calls per feature. Mitigation: hard cap on tool calls per feature (start at 15), per-scan cap on total tool calls (start at 200), use the existing `CostTracker` to abort if scan exceeds `--max-cost`.

2. **Tool latency.** Each tool call is local and fast (read file, query graph), but the LLM round-trips between them are the bottleneck. Cache file reads within a single scan.

3. **Ambiguous decisions.** The agent might still hedge to `developer-internal` on borderline cases. Mitigation: prompt explicitly enumerates the import-evidence → shared-aggregator rule, and the agent must cite at least 3 distinct consuming features when it picks aggregator.

4. **Sprint 8 reference branch divergence.** As Sprint 9 lands, the Sprint 8 branch becomes outdated. We can either delete it after Sprint 9 ships or keep it as a "what we tried first" reference.

## Out of scope (future sprints)

- Cross-repo aggregator detection (one shared-types package across `frontend/` and `backend/` repos)
- User-facing override UI in the dashboard (already partially exists via `feature_overrides` table)
- Symbol-level multi-feature attribution (today's `participants` are file-level; per-symbol granularity is a future polish)

## Working agreements

- Each day's PR ships independently with tests passing
- `--smart-aggregators` stays opt-in until Day 6 eval validates
- Cost guardrail: never run a paid scan in CI; manual scans only with explicit budget
- Tool-call hard caps in code, not just prompt instruction
- Day 3 checkpoint: smoke test on dify before scaling to multi-repo round

## Status (pause point — May 5 2026)

Days 1–3 done; Days 4–6 deferred. Sprint 9 ships as **opt-in
default-off** — `--smart-aggregators` stays a power-user flag, no
production scan touches it. The branch (`feat/sprint9-agentic-aggregators`)
has all the work; promote to default-on only after a future
session validates redistribution actually fires on real cases.

What's working:

- Days 1–2 modules + 102 unit tests, all green
- Day 3 wiring into pipeline.run() + library-mode skip
- Smoke-tested on dify: agent fires, makes ~70 tool round-trips,
  produces 20 specific business-named features (Workflow Editor,
  Datasets, Plugin Detail & Configuration, Plugin Installation,
  Plugin Management Page, Marketplace Browsing — all way more
  CTO-readable than baseline)
- Library mode (excalidraw equivalent) properly skips the stage

What's NOT validated:

- **Aggregator redistribution still doesn't fire.** The agent
  investigated dify's `Contracts` (142 cross-domain TS files)
  using tools — read sample files, walked the import graph —
  and independently classified it as `product-feature`, NOT
  `shared-aggregator`. So `shared_participants` stayed 0 across
  every feature. Either:
    (a) The agent's call is correct: Contracts has substantive
        content and is best treated as a real product surface,
        not an infrastructure aggregator. The dashboard shows
        it as one feature instead of scattering type files
        across 12 consumers.
    (b) The prompt still under-pushes redistribution. A "if a
        file is imported by ≥3 distinct features, MUST redistribute"
        hard rule could change the verdict.

  We don't know which is true without more iteration. Cost so far
  ~$11 across Sprint 8 + Sprint 9. Pausing here.

What still works as Sprint 9 contribution:

- Significantly better feature names via the agent's
  proposed_name renames (Workflow Editor, Plugin Detail &
  Configuration, Marketplace Browsing — all gained from agent
  investigation, not single-shot guess)
- Library mode skip (no main-product-folding regressions)
- Structural safeguards (largest-feature lock, file/commit caps
  on fold paths only)
- Opt-in flag means production users untouched

Day 4–6 entry conditions for resuming in a fresh session:

1. Decide on the redistribution philosophy: is shared-aggregator
   a categorical bucket (DTOs always redistribute) or a contextual
   one (only redistribute when consumers clearly outweigh the
   feature's own content)? Today the agent picks (b); the user
   wanted (a). Resolve before running paid scans.

2. If (a): strengthen prompt with a HARD rule — "if 3+ distinct
   features import any file in this feature, classify as
   shared-aggregator regardless of file content." Re-run on dify
   to validate. Then immich + ghost.

3. If (b): accept current behavior, document it on the landing,
   ship as-is.

4. Day 5 (CTO-readable rename pass) — partly done already via the
   agent's proposed_name. Day 6 final eval + landing update happens
   only if redistribution validates in Day 4.

Approximate remaining cost: $2–5 depending on iteration count.

This session burn (Sprint 9): ~$3.50. Total Sprint 8 + Sprint 9
burn ≈ $11. The branch stays open for future revisit.
