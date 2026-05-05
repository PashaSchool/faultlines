# Sprint 9 Scan Log — repos run with the agentic classifier

Tracking which OSS repos have been scanned with `--smart-aggregators`
(Sprint 9 agentic classifier + Stage 2.5 final-pass dedup) so we
don't re-scan the same repo without reason and the eval picture
keeps growing across sessions.

## Legend

- ✅ scanned with current Sprint 9 pipeline (agent + Stage 2.5)
- 🟡 scanned with EARLIER Sprint 9 (no Stage 2.5 final-pass) — re-scan when we want clean dedup
- ⬜ not scanned with Sprint 9 yet — still on Tier 2 / baseline output
- 🔒 library mode — Sprint 9 stage skips by design; no need to re-scan

## Status (last updated 2026-05-05)

| Repo | Sprint 9 status | Strict | Fixable | Notes |
|------|-----------------|-------:|--------:|-------|
| **dify** | 🟡 sprint9b (no Stage 2.5) | 75% | 90% | clean baseline scan, no duplicates so Stage 2.5 wouldn't change result |
| **plane** | 🟡 sprint9 → re-scanning now | ~65% | ~70% | Issues × 2, Workspace × 2 dup bug — re-scan in flight |
| **n8n** | 🟡 sprint9 (no Stage 2.5) | ~57% | ~67% | Credentials × 2 dup, Editor 7504f mega-bucket. Re-scan deferred — biggest ($8) |
| **immich** | ⬜ → scanning now | — | — | first Sprint 9 run in flight |
| **ghost** | ⬜ → scanning now | — | — | first Sprint 9 run in flight |
| **excalidraw** | 🔒 library mode | n/a | n/a | Sprint 9 skips libraries by design |
| **saleor** | ⬜ | — | — | Sprint 8/Tier-2 era; clean baseline 92%/92% |
| **meilisearch** | ⬜ | — | — | library; mostly skip |
| **gitea** | ⬜ | — | — | already strong baseline 75%/94% |
| **ollama** | ⬜ | — | — | library |
| **strapi** | ⬜ | — | — | Tier-2 result; would benefit from Sprint 9 rename pass |
| **supabase** | ⬜ | — | — | Tier-2 result, manual judgment 76%/92% |
| **calcom** | ❌ excluded | — | — | dropped from scan list — too expensive (10k+ files, ~$8-10/scan) and excluded from landing aggregate anyway |

## Cost burned on Sprint 9 scans so far

  dify       $2.47
  plane v1   $3.76
  n8n v1     $7.97
  ───────────────
  total     ~$14.20

  This round (in-flight): plane re-scan + immich + ghost
  Estimated: $3.76 + $1.16 + $1.89 ≈ $6.80
  Budget: $10. Remaining headroom: ~$3.20.

## What changed in the pipeline since the early Sprint 9 runs

- **Stage 2.5 final-pass collapse** (commit `13f6fa5`) catches
  duplicate display names introduced AFTER Stage 1.45 by
  sub_decompose / smart_aggregators / repo_config aliasing.
  Without it, plane's Issues × 2 / Workspace × 2 and n8n's
  Credentials × 2 leaked through to the dashboard.
- All structural safeguards from Sprint 8 carry over: largest-
  feature lock (≥30 files), `_MAX_FOLD_FILES=50`,
  `_MAX_FOLD_COMMITS=200`, library-mode skip.

## Open follow-ups

- **n8n re-scan with Stage 2.5** — biggest single cost ($8). Deferred
  until next budget window.
- **Sprint 10 candidate**: smart duplicate handling — when two same-name
  features have disjoint path trees (different `apps/`), rename one
  with parent-path prefix instead of collapsing via path union.
- **n8n Editor mega-bucket** (7504f) — agent didn't ✂ split. Future:
  always run sub_decompose on features above some file threshold even
  when classifier says product-feature.

## How to update this doc

When a Sprint 9 scan finishes:
1. Bump the repo's row to ✅ or 🟡 depending on whether it's the
   current pipeline (Stage 2.5 active) or earlier
2. Fill in Strict / Fixable from manual eval (or note "pending eval")
3. Add cost line in the burn tracker
4. Note any duplicates / issues in the Notes column
