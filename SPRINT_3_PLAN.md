# Sprint 3 — Sub-decomposition for oversized features

> **Roadmap context:** `SPRINTS_ROADMAP.md`. Previous sprints:
> `SPRINT_1_PLAN.md` / `SPRINT_1_RESULTS.md`,
> `SPRINT_2_PLAN.md` / `SPRINT_2_RESULTS.md`.
>
> **Branch:** `feat/tool-use-detection`
> **Status:** Plan, not yet implemented
> **Estimated effort:** 1-2 focused days
> **Estimated API spend:** $1-3 across all dev iterations

---

## 1. Why this sprint exists

### The problem Sprints 1 + 2 did not solve

After Sprint 1 (per-package tool-use) + Sprint 2 (cross-cluster dedup),
the engine produces business-readable feature names but some features
end up too **coarse**:

**documenso, post-Sprint-2 (commit `f5cea51`):**

```
328  document-signing                     ← 4 packages merged
206  organisation-and-team-management     ← 3 packages merged
192  user-authentication                  ← 3 packages + 7 auth/* merged
185  prisma-database                      ← 4 prisma sub-features
166  lib/platform-infrastructure
157  envelope-management                  ← 3 packages merged
```

**formbricks, post-Sprint-2:**

```
982  web/surveys                          ← apps/web's survey-* sub-features
598  documentation                        (synthetic — never split)
204  database
195  web/ee
117  surveys                              ← surveys package merged
```

A feature with 982 files is not actionable. An EM looking at a
dashboard row labelled "web/surveys" with 982 files cannot click in
and learn anything more specific than "the survey app is big". The
file:flow:health metric also blurs at this scale — a regression
inside the survey editor gets diluted by stable response-collection
code, masking the signal.

The product target from Sprint 1's plan was: **"features over 300
files: < 3 per scan"**. Sprint 1+2 left documenso with 1 such feature
and formbricks with 4. Sprint 3 closes that gap.

### What Sprint 3 fixes

After dedup completes, walk the feature map. For every feature whose
file count exceeds a threshold (default: 200), re-invoke the Sprint 1
tool-augmented detector against just that feature's files. Replace
the single feature with the returned sub-features when the LLM
proposes a clean 2-6-way split; otherwise leave it alone.

Same tools (`read_file_head`, `list_directory`, `grep_pattern`,
`get_file_commits`), same Sonnet model, similar ~5 tool-call budget
per feature (smaller than Sprint 1's per-package 15-call budget
because we're inside one cohesive area).

This is **Stage 3** in the roadmap diagram.

---

## 2. Sprint 3 goal in one sentence

**For each feature larger than 200 files, run a focused tool-augmented
sub-decomposition pass that splits the feature by its real internal
sub-domains, while preserving the parent feature's name as the prefix
for the resulting sub-features.**

Example expected change on documenso:

```
Before (Sprint 1+2)             After (Sprint 3)
─────                           ─────
document-signing 328  files     document-signing/field-validation       62
                                document-signing/pdf-manipulation       58
                                document-signing/recipient-flows        72
                                document-signing/audit-certificates     38
                                document-signing/api-handlers           48
                                document-signing/recipient-ui           50
                                ───
                                6 sub-features, total 328 files

web/surveys 982 files            web/surveys/editor       210
                                 web/surveys/preview      120
                                 web/surveys/responses     95
                                 web/surveys/templates    140
                                 web/surveys/list-and-folders 180
                                 web/surveys/sharing       72
                                 web/surveys/settings     165
                                 ───
                                 7 sub-features, total 982 files
```

Cost: roughly **+$0.50 per scan** when 3-4 features cross the
threshold (most scans don't trigger any).

---

## 3. Strategic decisions already locked

- **Reuse Sprint 1 tool dispatcher.** No new tools. The `tools.py`
  module from `b178a7b` already covers everything we need
  (`read_file_head`, `list_directory`, `grep_pattern`,
  `get_file_commits`).
- **Reuse `tool_use_scan` loop.** Same Anthropic message loop, same
  budget cap, same JSON parse. Sprint 3's contribution is the
  dispatcher around it that selects which features to re-analyze
  and the prompt that frames the split task.
- **Run AFTER dedup, BEFORE flow detection.** Dedup may merge two
  package-scoped features into one large feature; sub-decomposition
  re-splits it semantically. Flow detection then attaches flows to
  the leaf sub-features.
- **Naming:** sub-features get the parent name as prefix
  (`document-signing/field-validation`). When dedup creates an
  unprefixed feature like `signing` and we sub-split it, the result
  is `signing/X`, `signing/Y`, etc.
- **Branch continues** on `feat/tool-use-detection`.
- **Behind a flag initially.** New CLI flag `--sub-decompose` (or
  shorter — TBD Day 1). Default off until validated on 4+ repos.
- **Hard cap: 6 sub-features** per parent. If the model returns
  more, keep the top-6 by file count and absorb the rest into the
  largest sibling.
- **Skip protected names:** `documentation`, `shared-infra`,
  `examples` are NEVER sub-decomposed regardless of size.
- **Skip when LLM declines.** If the model returns
  `{features: []}` or one sub-feature, keep the parent intact.

---

## 4. Architecture overview

```
INPUT: DeepScanResult (post-dedup or post-Sprint-1)
        ↓
        ↓ pipeline.run, after dedup_features, before stage-2 buckets
        ↓
[Stage 3 — NEW Sprint 3 path]
        ↓
   ┌────────────────────────────────────────────────┐
   │  sub_decompose_oversized(result, threshold,    │
   │                          max_sub, repo_root)   │
   │                                                 │
   │  For each (name, files) in result.features:    │
   │    if name in PROTECTED → skip                 │
   │    if len(files) ≤ threshold → skip            │
   │                                                 │
   │    sub = tool_use_scan(                        │
   │        package_name=name,                       │
   │        files=files,                             │
   │        repo_root=repo_root,                     │
   │        client=…, model=…,                       │
   │        tool_budget=8,        # half Sprint 1   │
   │    )                                           │
   │                                                 │
   │    if sub has 2-6 features with all files     │
   │       attributed → replace name with           │
   │       {name/sub1, name/sub2, …}                │
   │    else → keep parent unchanged                │
   │                                                 │
   │  Update flows, descriptions, flow_descriptions │
   │  to follow the rename.                         │
   └────────────────────────────────────────────────┘
        ↓
OUTPUT: DeepScanResult with oversized features split
```

The downstream code (flow detection, output writer) does not change.
Sub-decomposition is a transformation between dedup and the rest.

---

## 5. Prompt design

The system prompt is **a focused variant of Sprint 1's prompt**
because the task differs:

```
You are a senior engineer subdividing a single feature into its
real internal sub-domains. The feature already has a business name
and a coherent overall purpose. Your job is to discover the 2-6
distinct sub-areas inside it.

INPUT
Feature name: <NAME>
File count: N
File paths: (the full list, repo-relative)

You have the same read-only tools as Sprint 1 (read_file_head,
list_directory, grep_pattern, get_file_commits). Use them to look
at representative files for each suspected sub-area.

OUTPUT (JSON only, no prose)
{
  "subfeatures": [
    {"name": "field-validation", "paths": [...], "description": "..."},
    {"name": "pdf-manipulation", "paths": [...], "description": "..."},
    ...
  ]
}

RULES
- Return between 2 and 6 sub-features. Fewer than 2 means no clean
  split exists; in that case return {"subfeatures": []}.
- Cover EVERY input file in exactly one sub-feature. No leftovers.
- Sub-feature names must NOT include the parent name (we add the
  prefix on our side, so don't write "document-signing/validation"
  — write "validation").
- Avoid generic names: "core", "lib", "utils", "shared", "general",
  "misc", "main", "common".
- Each sub-feature description: one sentence, business-readable.
```

User prompt is the parent feature's name, file count, and full file
list as a numbered text block.

---

## 6. Implementation plan, day by day

### Day 1 — Module + unit tests (no API spend)

**Files created:**

- `faultline/llm/sub_decompose.py`
  - `_SUB_DECOMPOSE_SYSTEM_PROMPT` constant
  - `_build_user_prompt(name, files) -> str`
  - `_parse_subfeatures(text) -> dict | None` (reuse the parse
    pattern from `dedup.py` / `tool_use_scan.py`)
  - `_validate_split(subfeatures, parent_files, max_sub=6)
    -> list[dict] | None` — returns the list if it covers every
    parent file with ≥2 sub-features and ≤max_sub; else None.
  - `sub_decompose_feature(name, files, *, client=…, model=…,
    tracker=…, repo_root, tool_budget=8) -> list[dict] | None`
    — the per-feature entry point. Internally calls
    `tool_use_scan` from Sprint 1 with this sprint's prompt and
    parser plumbed in.
  - `sub_decompose_oversized(result, *, threshold=200, max_sub=6,
    client=…, model=…, tracker=…, api_key=…, repo_root)
    -> DeepScanResult` — top-level entry point that walks
    `result.features`, applies sub-decomposition where appropriate,
    rewrites flows / descriptions / flow_descriptions to the new
    keys.
  - Protected-names guard reused from `dedup._PROTECTED_NAMES`.

- `tests/test_sub_decompose.py`
  - `_validate_split` unit tests:
    - 3 sub-features that cover all files → list returned
    - 5 sub-features that miss 2 files → None (no split)
    - 7 sub-features → None (over cap)
    - 1 sub-feature → None (no split)
  - `sub_decompose_oversized` end-to-end with a fake client:
    - feature ≤ threshold → never invokes the LLM
    - protected name → never invokes
    - LLM returns clean split → feature replaced with name/sub keys
    - LLM returns leftover files → original kept, log warning
    - LLM returns 1 sub-feature → original kept
    - flows + descriptions get rewritten to the new keys
    - tracker records the call's tokens
  - 18+ tests target.

**Acceptance:** all tests pass, no network.

### Day 2 — Wire + validation runs ($1-3)

**Files modified:**

- `faultline/llm/pipeline.py`
  - Add `sub_decompose: bool = False` parameter to `run()`.
  - After dedup, if `sub_decompose`, call
    `sub_decompose_oversized(result, …)`.

- `faultline/cli.py`
  - Add `--sub-decompose` flag.
  - Pass through to `pipeline.run(sub_decompose=…)`.

**Validation runs:**
1. **documenso** with `--llm --tool-use --dedup --sub-decompose`.
   Expected: `document-signing 328` splits into 4-6 sub-features;
   `organisation-and-team-management 206` similarly. Cost ≤ $1.
2. **formbricks** same flags.
   Expected: `web/surveys 982` splits into 5-8 sub-features (this
   is the headline win). Cost ≤ $1.50.

**Write `SPRINT_3_RESULTS.md`** with:
- Per-feature before/after sub-feature counts
- Sample sub-feature names + descriptions
- Cost
- Failure modes observed (if any)

**Acceptance for Sprint 3 closure:**
- documenso: at least 2 features ≥200 files split into 3+
  sub-features each (rule: top 2 oversized features must be
  decomposed).
- formbricks: `web/surveys 982f` split into 5-8 sub-features.
- Features over 300 files in either repo: < 3.
- Total dev API spend < $5.
- Existing tests still pass + 18+ new sub-decompose tests.
- `SPRINT_3_RESULTS.md` documents results.

---

## 7. What's explicitly out of scope for Sprint 3

1. **Recursive sub-decomposition.** A sub-feature with >200 files
   stays as-is. (Cap at 1 level deep — keeps cost predictable.)
2. **Tool-augmented flow detection.** Sprint 4.
3. **Self-critique on sub-feature names.** Sprint 5.
4. **Splitting `documentation` / `shared-infra` / `examples`.** They
   are synthetic; their structure isn't a feature decomposition
   problem.
5. **Auto-applying sub-decomposition by default.** Stays opt-in
   (`--sub-decompose`) until cross-validated on at least 4 repos.

---

## 8. Cost discipline

| Stage | Per scan |
|---|---|
| Sprint 1 (formbricks, with size guard) | ~$1.50 |
| + Sprint 2 (dedup) | +$0.30 (already wired) |
| + Sprint 3 (sub-decompose, ~3 features oversized) | +$0.30-0.60 |
| **Total** | **~$2-2.50** |

For dev iterations across Day 1-2, total budget should stay
**under $5**.

---

## 9. Open questions to resolve in Day 1

1. **Threshold default:** 200 files? 250? 300? Going with 200 — that
   matches `SPRINTS_ROADMAP.md` Sprint 3 spec.
2. **Tool budget per sub-feature pass:** 8 calls (vs Sprint 1's 15).
   Smaller because the input is one cohesive area, not a whole
   package.
3. **Order of operations between dedup and sub-decompose:** dedup
   first, sub-decompose second. Reasoning: dedup may CREATE
   oversized features by merging; we want sub-decompose to see the
   final post-dedup map.
4. **Name collisions:** if dedup-merged feature is named `signing`
   and sub-decompose proposes a `signing` sub-feature, prefix it →
   `signing/signing`. Hard cap on this collision is fine; `dedup`
   already had a similar safeguard.
5. **Failure mode if sub-decompose fails:** log + leave parent
   unchanged. Same opportunistic pattern as Sprint 2.

---

## 10. Files to read for context before starting

1. `SPRINT_3_PLAN.md` — this file
2. `SPRINT_1_RESULTS.md` + `SPRINT_2_RESULTS.md` — what the dedup
   output looks like in practice
3. `faultline/llm/tool_use_scan.py` — the Sprint 1 loop we reuse
4. `faultline/llm/dedup.py` — pattern for opportunistic LLM passes
   (failure mode, tracker, fake-client testing)
5. `faultline/llm/pipeline.py` — where the new stage hooks in
6. `tests/test_dedup.py` and `tests/test_tool_use_scan.py` —
   fake-client testing patterns to reuse

---

## 11. Definition of done

Sprint 3 ships when ALL of these are true:

- [ ] 2 new commits on `feat/tool-use-detection` (Day 1, Day 2)
- [ ] `--sub-decompose` flag works on documenso AND formbricks
- [ ] documenso: top 2 oversized features (≥200f) each split into
      3+ sub-features
- [ ] formbricks: `web/surveys 982f` splits into 5-8 sub-features
- [ ] Features over 300 files in either repo: < 3
- [ ] Dev API spend < $5
- [ ] 18+ new unit tests in `tests/test_sub_decompose.py`
- [ ] All Sprint 1 + 2 tests still pass
- [ ] `SPRINT_3_RESULTS.md` documents the side-by-side comparison

When green, Sprint 3 is closed. Decision: extend with Sprint 4
(flow detection with tool use) on the same branch, or merge to main.

---

## 12. Decision log (already locked)

- ✅ Reuse Sprint 1 tools + `tool_use_scan` loop.
- ✅ Run after dedup, before flow detection.
- ✅ Threshold default 200 files.
- ✅ Tool budget 8 calls per sub-feature pass.
- ✅ Hard cap 6 sub-features per parent.
- ✅ Opt-in `--sub-decompose` flag, off by default.
- ✅ Skip `documentation` / `shared-infra` / `examples`.
- ✅ One level deep — no recursive sub-decomposition.
- ✅ Opportunistic failure mode (log + keep parent).
