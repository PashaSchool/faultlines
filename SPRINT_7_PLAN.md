# Sprint 7 — Call-graph flow tracing

> **Roadmap context:** New sprint requested after Sprints 1-6 +
> validation. Addresses user feedback that Sprint 4 flows only show
> API entry-points without tracing through to UI / state / data
> layers — what an EM actually wants to see in the dashboard.
>
> **Branch:** `feat/tool-use-detection`
> **Status:** Plan, not yet implemented
> **Estimated effort:** 5-7 focused days
> **Estimated API spend:** $0-2 per scan (static analysis dominates)

---

## 1. Why this sprint exists

### What Sprint 4 produces today

```
Flow: Send Document for Signing
  entry_point: apps/remix/app/routes/.../documents.$id.edit.tsx:22
```

That's it. One file:line. The dashboard click goes to a single
route handler, but the user has no map of:
- Which UI components render the document-edit view?
- Which state store holds form state during editing?
- Which tRPC procedure / API endpoint sends the document?
- Which Prisma schema tables are mutated?
- For each file, which **lines** belong to this specific flow vs
  to other flows in the same file?

### What an EM actually needs

```
Flow: Send Document for Signing
  entry: apps/remix/app/routes/.../documents.$id.edit.tsx:22

  participants:
    UI:
      apps/remix/app/components/document/document-edit-form.tsx
        DocumentEditForm (lines 24-160)
        useDocumentForm hook call (lines 38-42)
      packages/ui/components/forms/recipient-list.tsx
        RecipientList (lines 18-92)

    state:
      apps/remix/app/stores/document-edit-store.ts
        useDocumentEditStore (lines 10-85)
        sendDocumentMutation usage (lines 64-70)

    api-client:
      packages/trpc/client/document.ts
        document.send mutation hook (lines 14-22)

    api-server:
      packages/trpc/server/document.ts
        sendDocumentProcedure (lines 8-58)
      packages/lib/server-only/document/send-document.ts
        sendDocument function (lines 24-180)

    schema:
      packages/prisma/schema.prisma
        Document, Recipient models (referenced)
      packages/prisma/migrations/...
        Send-document-related migrations
```

This is what makes the feature map **operationally useful**:
- Bug ticket "users complain signing email doesn't arrive" →
  click flow → see every file involved → know where to look.
- Refactor planning → know full surface area of a flow before
  touching it.
- Onboarding → new dev clicks a flow and reads the full vertical
  slice end-to-end.

---

## 2. Sprint 7 goal in one sentence

**For every flow detected in Sprint 4, statically trace from the
entry-point through the import graph to enumerate every
participating UI component, state store, API client, API server,
and schema file — with per-symbol line ranges — using mostly free
local AST analysis and a small LLM disambiguation step only when
heuristics tie.**

---

## 3. Strategic decisions already locked

- **Static analysis dominates.** Most of the work is reading the
  TS/Python AST and walking imports — no LLM needed.
- **LLM only for tiebreakers.** When file-path heuristics can't
  classify a file as ui/state/api/schema (e.g. a shared util),
  one Haiku call decides. Cap: 5 LLM calls per scan, $0-0.30.
- **Reuse existing infrastructure.** `ast_extractor.py` already
  produces SymbolRange data; `import_graph.py` already builds an
  import map. Sprint 7 wires them together with a flow walker.
- **Layer taxonomy fixed at 5:**
  `ui`, `state`, `api-client`, `api-server`, `schema`. Anything
  not classifiable lands as `support` (visible but separated).
- **Depth cap 3.** Walk imports up to 3 hops from entry point.
  Beyond that, results are too noisy.
- **Frontend frameworks supported:** React (functional + hooks),
  Remix routes, Next.js pages/app router, Vue, Svelte. Backend:
  tRPC, FastAPI, Express, NestJS.
- **State store patterns detected:** Zustand (`create(...)`),
  Redux Toolkit (`createSlice`), Jotai (`atom`), React Context
  (`createContext`), Pinia (Vue), Svelte stores
  (`writable`/`readable`).
- **Branch continues** on `feat/tool-use-detection`.
- **Behind a flag:** `--trace-flows`. Off by default, opt-in.

---

## 4. Architecture overview

```
INPUT: DeepScanResult with Sprint 4 flows already detected
        ↓
        ↓ pipeline.run, after detect_flows_with_tools, before output
        ↓
[Stage 4.5 — NEW Sprint 7 path]
        ↓
   ┌──────────────────────────────────────────────────┐
   │  trace_flow_callgraph(result, repo_root)         │
   │                                                   │
   │  1. Build symbol graph once per scan:            │
   │     - For each source file, AST-extract exports  │
   │       + imports + scope ranges (line spans).     │
   │     - Build forward map: (file, symbol) → callers│
   │     - Build reverse map: (file, symbol) → callees│
   │                                                   │
   │  2. For each flow with entry_point_file:         │
   │     a. Find the entry symbol (function/component │
   │        starting at entry_point_line).            │
   │     b. BFS through callees up to depth=3.        │
   │     c. For each touched (file, symbol):          │
   │        - Classify the file's layer (heuristic).  │
   │        - Record the symbol's line range.         │
   │                                                   │
   │  3. Layer classifier:                            │
   │     - Path regex first (apps/web/components/**   │
   │       → ui; prisma/** → schema; routes/**.tsx    │
   │       → ui; routes/**.ts/route.ts → api-server). │
   │     - Content regex second (createSlice / atom / │
   │       create() → state).                         │
   │     - LLM tiebreaker for unclassified files.     │
   │                                                   │
   │  4. Attach to result.flows as FlowParticipant    │
   │     entries via the result_extras carrier.       │
   └──────────────────────────────────────────────────┘
        ↓
OUTPUT: enriched DeepScanResult; downstream Pydantic Flow gains
         a participants: list[FlowParticipant] field.
```

---

## 5. New Pydantic types

```python
class FlowParticipant(BaseModel):
    path: str             # repo-relative file path
    layer: Literal["ui", "state", "api-client",
                   "api-server", "schema", "support"]
    symbols: list[SymbolRange]  # ranges in this file used by the flow
    # optional human-readable role ("Form component", "Mutation hook")
    role: str | None = None


class Flow(BaseModel):
    # ... existing fields ...
    participants: list[FlowParticipant] = []
```

`SymbolRange` already exists in `models/types.py`.

---

## 6. Implementation plan, day by day

### Day 1 — Symbol-import graph builder (no API spend)

**Files:**
- `faultline/analyzer/symbol_graph.py`
  - `build_symbol_graph(repo_root, source_files) -> SymbolGraph`
  - `SymbolGraph` dataclass with:
    - `exports: dict[file, list[SymbolRange]]` — symbols each file
      exports
    - `imports: dict[file, list[(import_path, symbol)]]` — what each
      file imports
    - `forward: dict[(file, symbol), list[(file, symbol)]]` —
      callees (X uses Y)
    - `reverse: dict[(file, symbol), list[(file, symbol)]]` —
      callers (Y is used by X)
  - `resolve_import_path(from_file, import_str, tsconfig_paths)
    -> str | None` — resolves bare imports + path aliases.
- `tests/test_symbol_graph.py` — TS, JS, Python fixtures + 25+
  unit tests.

**Reuses:** `analyzer.ast_extractor.extract_signatures`,
`analyzer.import_graph.load_tsconfig_paths`.

### Day 2 — Per-flow walker (no API spend)

**Files:**
- `faultline/analyzer/flow_tracer.py`
  - `trace_flow(graph, entry_file, entry_line, depth=3)
    -> list[(file, symbol_range)]` — BFS through `forward` from
    the entry symbol; return reachable set.
  - `find_entry_symbol(graph, file, line)` — picks the symbol
    whose range contains the entry line.
  - 15+ unit tests with synthetic graphs.

### Day 3 — Layer classifier (no API spend by default)

**Files:**
- `faultline/analyzer/layer_classifier.py`
  - `classify_file(path, content) -> Layer | None`
  - `_PATH_PATTERNS` mapping common conventions:
    - `routes/**/page.{tsx,jsx}` → ui
    - `routes/**/route.{ts,js}` → api-server
    - `pages/api/**` → api-server
    - `components/**` → ui
    - `stores/**`, `*Store.ts`, `slices/**`, `atoms/**` → state
    - `lib/**/server-only/**`, `**/server.ts` → api-server
    - `lib/**/api*`, `client/**` → api-client
    - `prisma/**`, `**/schema.prisma`, `migrations/**` → schema
  - Content regex fallback for `createSlice`, `create(<store>)`,
    `atom(...)`, `useContext`, etc.
  - `classify_with_llm(unclassified, repo_summary)` — Haiku
    tiebreaker for the rest. Cap 5 calls.
- `tests/test_layer_classifier.py` — 30+ test cases per
  framework.

### Day 4 — Wire into pipeline + Pydantic + display (no API spend)

- `faultline/llm/pipeline.py` — gain `trace_flows: bool` parameter.
  After `detect_flows_with_tools`, if `trace_flows`, call
  `trace_flow_callgraph(result, repo_root)`.
- `faultline/models/types.py` — add `FlowParticipant` + extend
  `Flow.participants`.
- `faultline/cli.py` — `--trace-flows` flag, plumb through.
- `_inject_new_pipeline_flows` extended to attach participants.

### Day 5 — Validation + dashboard (small API)

- Run on documenso with `--trace-flows`.
- Verify each flow has 5-30 participants spanning at least 3
  layers.
- Update saas-dashboard `FlowDetailPage` (if exists) to render the
  layered participant list.
- Write `SPRINT_7_RESULTS.md`.

### Day 6-7 — Polish + edge cases

- Cyclic imports (visited set in BFS).
- Re-exports (`export { X } from './...'`).
- Dynamic imports (`import('./x')`).
- Server-side hooks vs client-side hooks (Next.js boundaries).
- Per-flow line range collapse: when 5 symbols of one file
  participate in one flow, group into 1 FlowParticipant with
  multiple SymbolRanges.

---

## 7. What's explicitly out of scope

1. **Full type-checked dataflow analysis.** We use AST + symbol
   resolution, not TypeChecker / mypy.
2. **Cross-language tracing.** A flow that crosses TS frontend →
   Go backend stays separate (each per its own build).
3. **Runtime traces.** No instrumentation, no recordings — pure
   static analysis.
4. **Generated code.** Skipped via the existing Bucketizer.
5. **Auto-detected frameworks.** We support fixed regex patterns;
   exotic frameworks fall back to LLM.

---

## 8. Cost discipline

| stage cumulative | per scan |
|---|---|
| Sprint 1+2+3+4+5 (Haiku optimized) | ~$2-3 |
| + Sprint 7 trace (mostly local) | +$0-0.30 |
| **Total** | **~$2.30-3.30** |

---

## 9. Open questions for Day 1

1. **Where to walk by default — callees only, or callees + callers?**
   Callees only (downward from entry). Callers (upward) tend to
   include unrelated parents that just happen to import the
   entry module.
2. **Frontend route → component tracing across boundaries**
   (server component → client component). Treat both as `ui`
   layer, distinguish in `role` field.
3. **State stores that aren't files but configs (Zustand + Jotai
   often live next to the store consumer).** Detect via content
   regex within consumer file, not separate file.

---

## 10. Definition of done

- [ ] 5 commits on `feat/tool-use-detection` (Day 1-5)
- [ ] `--trace-flows` works on documenso AND formbricks
- [ ] Each Sprint 4 flow has 3+ participant files, 2+ layers
- [ ] No Pydantic schema break (add-only, all new fields optional)
- [ ] Existing tests pass + 70+ new unit tests
- [ ] Dashboard view shows participants in a layered list
- [ ] `SPRINT_7_RESULTS.md` documents validation
