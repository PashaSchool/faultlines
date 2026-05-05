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
| **plane** | ✅ sprint9b (Stage 2.5) | **77%** | **81%** | re-scan fixed Issues × 2 / Workspace × 2; new specifics: Issue Management, Public Issue Board API |
| **n8n** | 🟡 sprint9 (no Stage 2.5) | ~57% | ~67% | Credentials × 2 dup, Editor 7504f mega-bucket. Re-scan deferred — biggest ($8) |
| **immich** | ✅ sprint9 (Stage 2.5) | **85%** | **95%** | best result yet — many specifics, no dupes |
| **ghost** | ✅ sprint9 (Stage 2.5) | **85%** | **97%** | clean win — Members, Translations, Admin X Settings, Theme Engine, Sodo Search, Custom Redirects |
| **excalidraw** | 🔒 library mode | n/a | n/a | Sprint 9 skips libraries by design |
| **saleor** | ✅ sprint9 | **~95%** | **95%** | already strong, Sprint 9 confirmed clean: Product, Order Management, Checkout, App Integrations |
| **meilisearch** | ✅ sprint9 | **~95%** | **100%** | clean Rust crates: Index Scheduler, Update, Search, Dump, Federated Network |
| **gitea** | ✅ sprint9 | **~95%** | **100%** | biggest improvement (+20pp): Repository, Auth, Repo Issue, Notifications & Email, Repository Import & Migration |
| **ollama** | ✅ sprint9 | **~85%** | **93%** | Model Inference, MLX Inference Runner, CLI & Desktop Launcher, Desktop Application Shell |
| **strapi** | ⚠ sprint9 (false library skip) | ~62% | ~85% | repo_classifier.detect_library mis-flagged on examples/docs/www; agent skipped → Tier-2 result |
| **supabase** | ✅ sprint9 | **~77%** | **~85%** | 25 → 13 features; Studio, UI Block Library, Vue Blocks, Design System Documentation |
| **calcom** | ❌ excluded | — | — | dropped from scan list — too expensive (10k+ files, ~$8-10/scan) and excluded from landing aggregate anyway |

## Cost burned on Sprint 9 scans

  dify       $2.47
  plane v1   $3.76
  n8n v1     $7.97
  ───────── (round 1 = $14.20)
  plane v2   $3.59  ← Stage 2.5 fix
  immich     $1.53
  ghost      $2.59
  ───────── (round 2 = $7.71, under $10 budget)
  ─────────────────
  TOTAL    ~$21.91 across all Sprint 9 attempts

  Round 2 highlights: 3 clean scans, 0 duplicates anywhere,
  immich + ghost both 85%/95%+ — the agent shines on
  product-driven monorepos.

  ─── Round 3 (May 5 evening) ──────
  saleor       $0.79
  gitea        $0.77
  strapi       $0.30  (false library skip)
  supabase     $2.09
  ollama       $0.46
  meilisearch  $1.51
  ──────── ($5.92, well under \$10 budget)
  ─────────────────
  GRAND TOTAL  ≈ \$27.83 across all Sprint 9 attempts

  Round 3 highlights:
  • gitea +20pp strict over baseline (biggest single-repo lift)
  • saleor, meilisearch confirmed clean on already-strong baselines
  • supabase compressed 25 → 13 features
  • strapi blocked by false library detection — separate bug

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
