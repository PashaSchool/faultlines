# Refactor sprint — done

Sprint 4 Day 15 surfaced that ``shared_attributions`` (cross-feature
only) silently disabled the symbol-scoped chain on every workspace
monorepo and Django-style app. This refactor switches the primary
attachment surface to per-feature ``participants`` so the work
shipped in Sprint 1-3 actually fires across every repo, not just
the rare cross-feature ones.

## Days

| Day | Commit  | What                                                    |
|-----|---------|---------------------------------------------------------|
| 1   | 284431c | model: ``Feature.participants: list[FlowParticipant]``  |
| 2   | f61e23c | builder: ``analyzer/feature_participants.py``           |
| 3   | 94ffffe | pipeline wires participants + expands BlameIndex        |
| 4   | fb9c2e6 | ``_compute_line_scoped_health`` + coverage read it      |
| 5   | 2e434ec | ``scripts/landing_codegen.py`` emits the new shape      |
| 6-7 | —       | skipped — ``feature_participants`` already cross-lang   |
| 8   | (here)  | validation                                              |

Days 6-7 (lift TS/JS gate in ``import_graph.resolve_symbol_imports``)
were planned but turned out unnecessary. The refactor targets
``feature_participants`` which builds via
``SymbolGraph.imports_from()`` directly, and Sprint 1 Day 4-5
already added Go and Rust to that side. ``resolve_symbol_imports``
remains TS/JS-only but is only used by the legacy
``shared_attributions`` path, which is no longer the primary
scoring surface.

## Activation numbers

Compare Sprint 4 Day 15 against Refactor Day 8 — same repos, same
``--line-attribution`` flag, no other input changes:

```
                    Day 15            Refactor Day 8
saleor      (Py):   0  / 13           4 /  4   (100%)
supabase    (TS):   0  / 26           — (not re-scanned in this batch)
ollama      (Go):   0  / 40          40 / 40   (100%)
meilisearch (Rs):   —                 74 / 75   ( 99%)
gitea       (Go):   —                 17 / 17   (100%)

aggregate symbol_health activation:  0%  →  ~99%
```

Total participants across the four-repo cohort: **9,497** files
indexed, ranges resolved, scored. Saleor's 4 features collapsed
post-cleanup but every survivor is fully scored; the underlying
attribution still walked 562 participants before the cleanup pass.

## What now works that didn't before

1. **Tier-1 line-scoped health** populates on every repo where
   ``SymbolGraph`` resolves at least one import edge. That's
   TS/JS/Py/Go/Rust today.
2. **Tier-1 line-scoped coverage** averages
   ``FileCoverage.coverage_for_range`` over each Participant's
   line range. Same gate.
3. **BlameIndex** indexes the union of every Participant's file
   instead of just files in 2+ features. The cache stays valid
   across runs at ``<repo>/.faultline/cache/blame.sqlite``.
4. **Codegen** (``scripts/landing_codegen.py``) emits the new
   shape so the landing page surfaces real Participant entries
   with symbols, line spans, and layer roles.
5. **The user spec** ("for any detected feature, show every file
   that's imported into it, with line ranges") is now answered
   end-to-end.

## What's still TS/JS-only

The legacy ``shared_attributions`` building (``cli.py`` lines ~870–
895) still gates on ``ts_js_sig_count >= 10``. It's no longer on
the critical path — ``feature.participants`` now drives scoring —
but the field still gets populated when conditions are met, and
the code is preserved for back-compat callers (notably the
shared-files dashboard view that reads
``Feature.shared_attributions`` directly). Lifting the gate is
optional follow-up; nothing in the symbol-scoped chain needs it.

## Files changed (12 commits across the two sprints)

- ``faultline/models/types.py`` — ``Feature.participants``
- ``faultline/analyzer/feature_participants.py`` (new)
- ``faultline/analyzer/blame_index.py`` (new, Sprint 1 Day 1)
- ``faultline/analyzer/coverage.py`` — ``read_coverage_detailed``
- ``faultline/analyzer/test_mapper.py`` (new, Sprint 1 Day 3)
- ``faultline/analyzer/ast_extractor.py`` — Go + Rust extractors
- ``faultline/analyzer/features.py`` — line-scoped scoring
- ``faultline/cli.py`` — pipeline wiring
- ``scripts/landing_codegen.py`` — emit new shape
- Plus ~50 unit tests across Sprints 1 + 2 + Refactor
