# Sprint 5 — Self-critique loop

> **Roadmap context:** `SPRINTS_ROADMAP.md`. Previous sprints:
> Sprint 1 (`b178a7b`+), Sprint 2 (`f5cea51`+), Sprint 3
> (`cbd6222`+), Sprint 4 (latest commit on `feat/tool-use-detection`).
>
> **Branch:** `feat/tool-use-detection`
> **Status:** Plan + Day 1 build (no API)
> **Estimated effort:** 2-3 focused days
> **Estimated API spend:** $1-3 across all dev iterations

---

## 1. Why this sprint exists

### What still slips through

After Sprints 1-4 the engine produces well-named features, dedup'd
across packages, sized appropriately, with grounded flows. But on
real validations a small handful of **weak names** still appear:

- Generic placeholders the model picks when a cluster is genuinely
  miscellaneous (e.g. `lib/platform-infrastructure` on documenso —
  166 files, not actually one infrastructure thing).
- Names that re-emerge after dedup creates a new feature
  (e.g. `email` 54 files becoming a catch-all for unrelated
  email-adjacent things on documenso post-Sprint-2).
- Sub-features Sprint 3 produced when the parent had no clean
  internal structure (e.g. some `web/surveys/*` slices).
- Flows whose entry point is real but whose name is too abstract
  ("manage-resources").

The first four sprints have no mechanism for the engine to
**look at its own output** and ask "is this actually meaningful?".
Sprint 5 adds that mechanism.

### What Sprint 5 fixes

Add a final **critique stage** that reads the entire feature map
+ flows JSON and flags weakness:

- Generic / vague feature or flow names.
- Features whose description is suspiciously short or boilerplate.
- Sub-feature splits where the names do not differ enough to
  justify the split.

For each flagged item, the loop **re-investigates with the same
tools as Sprint 1+3** (`read_file_head`, `grep_pattern`, etc.) and
proposes a replacement. If the replacement is materially better
(not just a synonym), apply it; otherwise leave the original.

**Cap: 1 critique pass, max 5 items re-investigated.** No
recursion, no infinite refinement.

---

## 2. Sprint 5 goal in one sentence

**After all detection stages finish, run a single Sonnet critique
pass that flags weak names; for each flagged item, run a focused
tool-augmented re-investigation and apply the new name if it is
materially better than the original.**

Example expected change on documenso:

```
Before (Sprint 1-4 final)        After (Sprint 5 critique)
─────                            ─────
lib/platform-infrastructure 166  pdf-rendering-infrastructure 92
                                 stripe-webhook-handlers      48
                                 internal-cli-tooling         26
   "Mixed bucket of platform        "70% of files matched PDF
    code."                            generation patterns; renamed
                                       on second pass."
```

---

## 3. Strategic decisions already locked

- **Reuse Sprint 1 tool dispatcher** for re-investigation calls.
  Same `tool_use_scan` loop with a "rename or split this feature"
  prompt.
- **Two-pass design:**
  1. **Critique pass** — single Sonnet call, no tools, sees the
     whole feature map + flow names. Outputs a list of flagged
     items with reasons.
  2. **Re-investigation pass** — for each flagged item (capped),
     tool-augmented rename or 2-way split.
- **Cap: 1 critique pass, max 5 re-investigations.** A weak name
  surviving 5 attempts likely reflects genuinely ambiguous code.
- **Apply only when materially better.** Heuristic: edit distance
  ≥ 4 chars, or the new name contains a domain word the old one
  did not. If the diff is "manage-things" → "manage-stuff", skip.
- **Branch continues** on `feat/tool-use-detection`.
- **Behind a flag:** `--critique`. Off by default.

---

## 4. Architecture overview

```
INPUT: DeepScanResult (post Sprint 1+2+3+4)
        ↓
        ↓ pipeline.run, after detect_flows_with_tools
        ↓
[Stage 5 — NEW Sprint 5 path]
        ↓
   ┌────────────────────────────────────────────┐
   │  critique_features(result, …)              │
   │                                            │
   │  1. Build summaries: name, description,    │
   │     file_count, flow_names                 │
   │  2. One Sonnet call:                       │
   │     "List items that look generic / vague" │
   │     → [{kind: "feature"|"flow",            │
   │         name: "...", reason: "..."},       │
   │        ...]                                │
   │  3. For each (capped at 5):                │
   │     - if kind=feature: tool-augmented      │
   │       rename via tool_use_scan with        │
   │       single-feature prompt                │
   │     - if kind=flow: rename via simple      │
   │       Haiku call (no tools needed)         │
   │     - validate "materially better"         │
   │     - apply if yes, log if no              │
   │  4. Return updated DeepScanResult          │
   └────────────────────────────────────────────┘
        ↓
OUTPUT: refined DeepScanResult + critique log
```

---

## 5. Prompt design

### Critique prompt (no tools)

```
You are reviewing a feature map for quality. The map already passed
through naming, dedup, and sub-decomposition. Your job is to flag
the ≤ 5 weakest names — entries whose name is too generic, vague,
or fails to convey what the code does.

INPUT
A JSON array:
  [{"kind": "feature", "name": "...", "description": "...",
    "file_count": N},
   {"kind": "flow", "feature": "...", "name": "...",
    "description": "..."},
   ...]

OUTPUT (JSON only, no prose)
{
  "weak": [
    {
      "kind": "feature",
      "name": "lib/platform-infrastructure",
      "reason": "Generic name; 166 files suggests this should split."
    },
    {
      "kind": "flow",
      "feature": "billing",
      "name": "manage-things",
      "reason": "Vague — what is being managed?"
    }
  ]
}

RULES
- 0-5 entries. Quality over quantity.
- Skip protected names: documentation, shared-infra, examples.
- Skip names that are already specific even if short
  (auth, billing, signing — these are real domain words).
- For flows, the "feature" field tells the rename pass which bucket
  to update.
```

### Rename prompt (tool-augmented, per feature)

Reuses the Sprint 3 sub-decomposition tools and a slightly
different prompt: "given this feature, propose either a better
single name OR a 2-way split — explain why."

---

## 6. Implementation plan, day by day

### Day 1 — Critique module + tests (no API spend)

**Files:**
- `faultline/llm/critique.py`
  - `_CRITIQUE_SYSTEM_PROMPT`
  - `_RENAME_SYSTEM_PROMPT`
  - `_build_critique_summaries(result) -> list[dict]`
  - `_parse_critique(text) -> list[dict] | None`
  - `_is_materially_better(old: str, new: str) -> bool`
  - `_rewrite_feature_name(result, old_name, new_name) -> None`
  - `critique_and_refine(result, *, max_items=5, …) -> DeepScanResult`
- `tests/test_critique.py`
  - `_is_materially_better`: typo / synonym → False;
    "lib" → "pdf-rendering" → True; case-only diff → False;
    completely different → True; etc.
  - `_rewrite_feature_name`: renames in features + descriptions +
    flows + flow_descriptions atomically.
  - `_build_critique_summaries`: protected names skipped, flows
    flattened, file_count populated.
  - `critique_and_refine` end-to-end with fake client:
    - empty critique → no-op.
    - one weak feature → tool_use_scan called → rename applied.
    - one flow flagged → simple rename applied.
    - replacement not materially better → original kept.
    - tracker records both critique + rename token usage.
  - 18+ tests target.

**Acceptance:** all tests pass, no network.

### Day 2 — Pipeline + CLI wire (no API spend)

- `pipeline.run(critique: bool = False)` parameter.
- After `detect_flows_with_tools`, if `critique`, call
  `critique_and_refine(result, …)`.
- `cli.py` adds `--critique` flag.

### Day 3 — Live validation (deferred until user opts in)

- Documenso run with full stack (`--llm --tool-use --dedup
  --sub-decompose --tool-flows --critique`). Compare to without-
  critique output. Expect 2-4 names refined, 0-1 features split.
- Formbricks similar.
- Write `SPRINT_5_RESULTS.md`.

---

## 7. What's explicitly out of scope

1. **Recursive critique.** One pass max.
2. **Auto-applying critique by default.** Stays opt-in.
3. **Cross-repo critique memory.** No "last time you flagged X"
   memory between scans.
4. **Critique of the critique.** No meta loop.

---

## 8. Cost discipline

| Stage cumulative | Per scan |
|---|---|
| Sprint 1+2+3+4 (formbricks-class) | ~$4-6 |
| + Sprint 5 (`--critique`, 1+5 LLM calls) | +$0.50-1.00 |
| **Total** | **~$5-7** |

For dev iterations across Day 1-3, total budget should stay
**under $5**.

---

## 9. Definition of done

- [ ] 1 commit on `feat/tool-use-detection` for Day 1+2 (module +
      wiring), 1 commit for Day 3 (validation + results).
- [ ] `--critique` flag works on documenso AND formbricks.
- [ ] At least 2 features renamed across the two test runs.
- [ ] No incorrect rewrite (rename rejected = original survives).
- [ ] 18+ new unit tests in `tests/test_critique.py`.
- [ ] All Sprint 1-4 tests still pass.
- [ ] `SPRINT_5_RESULTS.md` documents the renames + the rejected
      proposals (transparency).
