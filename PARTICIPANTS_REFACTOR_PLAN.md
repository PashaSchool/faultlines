# Per-feature participants refactor plan

## What Sprint 4 Day 15 surfaced

Re-running the integrated pipeline on saleor (Python) and supabase
(TS workspace monorepo) showed every Sprint 1-3 surface field empty:

```
features: 13–26
  with symbol_health_score: 0
  total feature.shared_attributions: 0
  total flow.symbol_attributions: 0
  total flow.test_files: 0
```

The wiring is correct. The data isn't there. Two reasons stack:

1. **`build_shared_attributions` is cross-feature only.**
   It only emits a `SymbolAttribution` for files that live in **2+
   features**. In a workspace monorepo every file belongs to exactly
   one package → exactly one feature → no shared files → empty
   output. Saleor's Django apps have the same property.

2. **`resolve_symbol_imports` reads `sig.source`** which is only
   stored for TS/JS files. Even after Sprint 1 Day 4-5 added Go and
   Rust to `extract_signatures`, this caller can't see them.

The first one matters more. The user's spec (verbatim):

> якщо є фіча задетекчина, я хочу бачити всі файли які будь яким
> чином імпортуються в цю фічу, і з лінійками коду. Бо може бути
> шарений компонент і він може використовуватись в різних інших
> фічах, і це окей, значить я очікую що він зявиться файлом у
> всіх місцях де він використовується.

The user wants per-feature **participants** (every file imported,
with line ranges, possibly shared with N flows). The current model
hard-gates on "appears in 2+ features", which silently disables the
whole symbol-scoped chain on every workspace and Django repo.

## Refactor plan (post Sprint 1-4 work)

### Day 1 — model: `Feature.participants` / `Flow.participants`

Add two Pydantic fields:

```python
class Participant(BaseModel):
    file_path: str
    symbol: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    role: str | None = None
    shared_with_flows: list[str] = []

class Feature(BaseModel):
    ...
    participants: list[Participant] = []

class Flow(BaseModel):
    ...
    participants: list[Participant] = []  # already exists via FlowParticipant
```

`shared_attributions` stays for backward compatibility but stops
being the primary attachment surface.

### Day 2 — builder: per-feature symbol-graph BFS

New `analyzer/feature_participants.py`:

```python
def build_feature_participants(
    features: list[Feature],
    graph: SymbolGraph,
) -> dict[str, list[Participant]]:
    """For each feature, BFS forward through symbol_graph from its
    files. Collect every reachable (file, symbol, lines) and emit
    one Participant per (file, symbol) pair."""
```

Same kind of BFS `flow_tracer.py` already does, seeded from a
feature's full file set instead of a flow's entry-point.

### Day 3 — wire into pipeline

Replace the `if signatures and ts_js_sig_count >= ...` gate with a
call to the new builder. Build `BlameIndex` over the union of every
participant's file. Build `TestMap` once. Run
`apply_test_attribution` against the new participant model.

### Day 4 — scoring math reads participants

`_compute_line_scoped_health` switches its input from
`feature.shared_attributions` to `feature.participants`. Same logic,
new source. `_apply_feature_coverage` does the same. Drop the
"only shared files" assumption.

### Day 5 — codegen + UI

`scripts/landing_codegen.py`: emit `feature.participants` from the
new field; `flow.participants` already wired (Sprint 4 Day 13).
`ScanCarousel` already renders the Sprint 4 elements — they'll just
start showing real numbers once the JSON has them.

### Day 6-7 — cross-language

Lift `resolve_symbol_imports` to dispatch by suffix and use the
existing Go/Rust resolvers in `analyzer/symbol_graph.py`. Then
`build_feature_participants` works on Python / Go / Rust without a
TS-specific gate.

### Day 8 — re-scan validation

Re-scan saleor + supabase + ollama with the refactored pipeline.
Expect: every feature has populated `participants`, every flow has
`test_files` when tests exist, every shared attribution has
`shared_with_flows` populated.

## Estimated effort

8 working days. Smaller than Sprint 1+2+3 because the data
layer (`blame_index`, `coverage_detailed`, `test_mapper`,
`SymbolGraph`) is already in place — this refactor mostly re-routes
existing components onto a per-feature primary surface.

## Acceptance check

A run on saleor should print:
```
Feature participants: 412 across 13 features (avg 32/feature)
Blame index: 412 indexed + ... cached
Test attribution: 1,744 symbol→test mappings, 89 flows with tests
Post-process: features 13 → 19, flows 207 → 207
```

And the landing page should show:
- Features with populated `lineScopedHealth` ≠ `health`
- Flows with `tests×N` and `shared×M` badges
- Participant counts in the "Entry → Touches" column
