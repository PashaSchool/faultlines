# Sprint 4 — Flow detection with tool use

> **Roadmap context:** `SPRINTS_ROADMAP.md`. Previous sprints:
> Sprint 1 (`b178a7b`+), Sprint 2 (`f5cea51`+),
> Sprint 3 (`cbd6222`).
>
> **Branch:** `feat/tool-use-detection`
> **Status:** Plan + Day 1 build (no API)
> **Estimated effort:** 4-5 days
> **Estimated API spend:** $3-8 across all dev iterations

---

## 1. Why this sprint exists

### The problem Sprints 1-3 did not solve

Today, after Sprints 1-3, every feature has a business-readable name
and the right granularity. But **flows** still come from the legacy
Haiku per-feature call (`flow_detector.detect_flows_llm`) which only
gets a file list + feature name. It guesses flows from filename
patterns and produces names like `view-detections-flow`,
`manage-issues-flow`, `process-data-flow`.

These names are reasonable but ungrounded. There is no link from a
flow to a real Next.js route, an Express handler, a FastAPI endpoint,
or a webhook subscription. An EM clicking into "manage-issues-flow"
has no idea where to start reading the code.

### What Sprint 4 fixes

Replace the Haiku per-feature call with a **tool-augmented** version.
The LLM gets two new tools that surface actual entry points:

- `find_route_handlers(path_glob)` — regex/AST scan for Next.js,
  Express, FastAPI, tRPC, Hono.
- `find_event_handlers(path_glob)` — scan for `addEventListener`,
  `on('event',...)`, queue / webhook patterns.

Plus the existing Sprint 1 tools (`read_file_head`, `grep_pattern`).

Each flow in the output carries `entry_point_file` + `entry_point_line`
fields, so a click in the dashboard takes the user to the actual
function that starts the flow.

For libraries, flows stay suppressed (Sprint 1 behaviour preserved).

---

## 2. Sprint 4 goal in one sentence

**Replace per-feature flow guessing with a per-feature tool-augmented
LLM pass that grounds every flow in a real route handler / API
endpoint / event subscription, recording the entry-point file:line in
the output JSON.**

Example expected change on documenso `document-signing`:

```
Before (Haiku, no tools)         After (tool-augmented)
─────                            ─────
sign-document-flow               sign-document
                                   entry: apps/remix/app/routes/
                                          sign.$token.tsx:24
                                   description: User opens the signing
                                                page from an email
                                                link, fills fields,
                                                clicks Sign.
                                 cancel-signing
                                   entry: apps/remix/app/routes/
                                          documents.cancel.ts:12
                                 resend-signing-email
                                   entry: trpc/document/
                                          resend-document.ts:8
                                 ...
```

---

## 3. Strategic decisions already locked

- **Reuse Sprint 1 tool dispatcher.** Add 2 new tools to `tools.py`:
  `find_route_handlers`, `find_event_handlers`. They follow the same
  schema/safety conventions as the existing 4.
- **Reuse `tool_use_scan` loop** with `system_prompt` override
  (already plumbed in Sprint 3).
- **Library mode passthrough.** When `is_library=True`, skip Sprint
  4 entirely — same as the legacy Sprint-1 flow detection
  suppression.
- **Output shape:** add `entry_point_file` and `entry_point_line` to
  each flow. Existing `DeepScanResult.flows` is `dict[str,
  list[str]]`; new info goes into `flow_descriptions[feature][flow]`
  as `"<description> (entry: <file>:<line>)"` so we don't need a
  schema change. Cleaner full schema is Sprint 6 polish.
- **Behind a flag:** `--tool-flows`. Off by default. Pair with
  `--llm --flows --tool-use --dedup --sub-decompose` for the full
  Sprint 1-4 stack.
- **Tool budget:** 6 calls per feature (smaller than Sprint 1's 15
  and Sprint 3's 8 — flow detection should be cheap once the
  feature is well-named).
- **Branch continues** on `feat/tool-use-detection`.

---

## 4. Architecture overview

```
INPUT: DeepScanResult (post Sprint 1+2+3)
        ↓
        ↓ pipeline.run, after sub_decompose, before output
        ↓
[Stage 4 — NEW Sprint 4 path]
        ↓
   ┌──────────────────────────────────────────────┐
   │  detect_flows_with_tools(result, …)          │
   │                                              │
   │  Skip if is_library=True.                    │
   │                                              │
   │  For each feature in result.features:        │
   │    if feature in PROTECTED → skip            │
   │                                              │
   │    flows = tool_use_scan(                    │
   │        system_prompt=_FLOW_PROMPT,           │
   │        package_name=feature_name,            │
   │        files=feature_files,                  │
   │        repo_root=…,                          │
   │        tool_budget=6,                        │
   │    )                                         │
   │                                              │
   │    Validate: each flow has name + entry      │
   │    Update result.flows[feature]              │
   │    Update result.flow_descriptions[feature]  │
   └──────────────────────────────────────────────┘
        ↓
OUTPUT: DeepScanResult with grounded flows
```

---

## 5. New tools

### `find_route_handlers(path_glob: str = "") -> list[str]`

Regex-scan the source tree for route handler declarations. Returns
matches as `path:line: pattern` strings. Caps at 50 results.

Frameworks detected (regex patterns, no AST):
- Next.js: `app/**/page.tsx`, `app/**/route.ts`, `pages/api/**`
- Express: `app.get(`, `app.post(`, `router.get(`, `router.post(`,
  etc.
- FastAPI: `@app.get(`, `@router.post(`, `@app.api_route(`
- tRPC: `procedure.query(`, `procedure.mutation(`,
  `t.procedure.input(`, `protectedProcedure.input(`
- Hono: `app.get(`, `app.post(` (overlaps with Express but matches
  on Hono import context — kept generic)
- Remix: `loader =`, `action =` exported in `routes/**`

### `find_event_handlers(path_glob: str = "") -> list[str]`

Regex-scan for event-driven entry points:
- DOM/JS: `addEventListener(`, `\.on\(['"][a-z]+['"]`
- Webhooks: `app.post('/webhooks/', ...)`, Stripe `Webhook.constructEvent`
- Queues: `queue.process(`, `worker.on(`, `consumer.subscribe(`
- Slack/Discord: `bot.on(`, `client.on(['"][a-z]+['"]`

### Tool schema

Both tools are read-only, follow the existing schema pattern, and
register in `TOOL_SCHEMAS` so `tool_use_scan` automatically exposes
them. Path safety uses the same `_safe_resolve` helper.

---

## 6. Prompt design

```
You are a senior engineer mapping a feature's user-facing flows.
You know the feature's name and file list. Use the tools to find
real entry points (route handlers, API endpoints, event handlers).

WORKFLOW
1. Run find_route_handlers and find_event_handlers scoped to this
   feature's path prefix.
2. For ambiguous matches, read_file_head to confirm the entry
   point's purpose.
3. Group entry points into 1-8 named flows. A flow is a user
   journey — multiple endpoints can share one flow.
4. Return final JSON. Do not narrate.

NAMING RULES
- Flow names: imperative business action. "create-document",
  "cancel-subscription", not "endpoint-handler".
- 1-8 flows per feature. Cap at 8.
- Library code: if you can't find a real entry point, return
  {"flows": []}.

OUTPUT
{
  "flows": [
    {
      "name": "create-document",
      "description": "User uploads a PDF and configures recipients.",
      "entry_point_file": "apps/web/app/routes/documents.new.tsx",
      "entry_point_line": 24
    }
  ]
}
```

---

## 7. Implementation plan, day by day

### Day 1 — Two new tools + `detect_flows_with_tools` module + tests

**Files:**
- `faultline/llm/tools.py` — add `find_route_handlers`,
  `find_event_handlers`. Schemas in `TOOL_SCHEMAS`. Dispatcher
  routes them. Path safety + caps reused.
- `faultline/llm/flow_detector_v2.py` (new):
  - `_FLOW_SYSTEM_PROMPT`
  - `detect_flows_for_feature(name, files, ...) -> list[dict] | None`
  - `detect_flows_with_tools(result, *, repo_root, is_library, ...)
    -> DeepScanResult` — top-level walker.
- `tests/test_tools_flow.py` — fixture-based tests for the two new
  tools (Next.js, Express, FastAPI, tRPC, webhook, queue patterns).
- `tests/test_flow_detector_v2.py` — fake-client tests for the
  module: skip-when-library, per-feature loop, valid flows
  validated, unparseable response handled, tracker recording.

**Acceptance:** all tests pass, no network, ~30+ new tests.

### Day 2 — Pipeline + CLI wire-in (no API spend)

- `pipeline.run(tool_flows: bool = False)` parameter.
- After `sub_decompose_oversized`, if `tool_flows`, call
  `detect_flows_with_tools(result, ...)`.
- `cli.py` — `--tool-flows` flag.

### Day 3-4 — Live validation (deferred until user opts in)

- Documenso run with `--llm --tool-use --dedup --sub-decompose
  --tool-flows`. Verify ≥80% of flows have real `entry_point_file`
  values that resolve to actual files.
- Formbricks same. Library mode (gin or fastapi) confirms flow
  suppression.
- Write `SPRINT_4_RESULTS.md`.

### Day 5 — Polish

- Tighten any prompt issues surfaced in validation.
- Optionally add `flow.entry_point_url` (constructed from git remote
  + file:line) so the dashboard can deep-link.

---

## 8. What's explicitly out of scope

1. **First-class `entry_point_file`/`entry_point_line` fields on
   `Flow` Pydantic model.** For Sprint 4 they live inside the
   description string `(entry: file:line)`. Schema change is Sprint
   6 polish.
2. **AST-based handler detection.** Stay with regex + tool calls.
3. **Multi-file flow tracing.** Each flow has one entry; the LLM
   doesn't need to walk the full call graph.
4. **Cross-feature flow consolidation.** Each feature's flows stay
   in that feature's bucket. Cross-feature dedup is Sprint 5.

---

## 9. Cost discipline

| Stage cumulative | Per scan |
|---|---|
| Sprint 1+2+3 (formbricks-class) | ~$2.30 |
| + Sprint 4 (`--tool-flows`) | +$2-4 |
| **Total** | **~$4-6** |

Per-feature flow detection makes one Sonnet call per feature.
Documenso has ~30 features post-dedup → ~30 calls. With caching, the
per-call cost is ~$0.07 input + ~$0.04 output ≈ $0.11. ~$3-4 total.

For dev iterations, total budget should stay **under $15**.

---

## 10. Open questions for Day 1

1. **Do we replace or augment the legacy `flow_detector`?** Augment.
   `detect_flows_with_tools` is opt-in via `--tool-flows`; the
   legacy path stays for Ollama/DeepSeek users.
2. **Where does `entry_point_line` live in the JSON?** As text in
   the description: `"User signs a document. (entry: apps/web/app/
   routes/sign.tsx:24)"`. Sprint 6 promotes it to a real field.
3. **Tool budget per feature:** 6. Validated empirically in
   Sprint 1 + 3 (smaller is fine for narrower scope).

---

## 11. Definition of done

Sprint 4 ships when ALL of these are true:

- [ ] 2 commits on `feat/tool-use-detection` (Day 1+2 wire, Day 3-5
      validation+polish)
- [ ] `--tool-flows` flag works on documenso AND formbricks
- [ ] ≥80% of detected flows have `entry: file:line` that resolves
      to a real file
- [ ] Library mode (gin or fastapi) produces zero flows
- [ ] Existing Sprint 1+2+3 tests still pass + 30+ new tests
- [ ] Dev API spend < $15
- [ ] `SPRINT_4_RESULTS.md` documents validation
