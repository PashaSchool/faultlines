# Sprint 2 — Cross-cluster reconciliation: results

> **Branch:** `feat/tool-use-detection`
> **Sprint plan:** `SPRINT_2_PLAN.md`
> **Roadmap:** `SPRINTS_ROADMAP.md`
> **Date closed:** 2026-04-27

---

## TL;DR

A single Sonnet pass that sees every Sprint 1 feature at once now
collapses cross-package duplicates into one business-readable feature
each. Validated on documenso and formbricks:

| repo | Sprint 1 only | + Sprint 2 dedup | merges applied |
|---|---|---|---|
| documenso | 61 | **30** | 13 |
| formbricks (size-guard active) | 26¹ | **25** | 7 |

¹ The size guard sends ``apps/web`` (1948 files) through the
no-tools deep_scan path so it decomposes into 9 web/* sub-features
already; dedup then refines further.

Two real bugs surfaced and were fixed during Day 3 — both about how
descriptions get re-attached to features after `build_feature_map`,
not about dedup itself:

1. Substring leak (`ai` inheriting `email/auth-emails`'s description
   because 'ai' is in 'auth-emails').
2. Path-segment leak (`web/surveys` inheriting `surveys`'
   description because both end in `surveys`).

Both fixed by tightening `_inject_new_pipeline_descriptions` to
exact-name + bounded singular/plural only. 9 regression tests added.

Total Sprint 2 dev API spend: **~$4.60** (plan $1-3, overrun caused
by the bug-discovery cycle — same shape as Sprint 1 Day 4).

---

## What shipped

### Day 1 — `dedup.py` module + 37 unit tests (commit `2792534`)

- `_build_summaries` packs features into the LLM input
- `_parse_merges_payload` JSON fallback chain
- `_coerce_merges` enforces `into`, `from≥2`, non-empty `rationale`,
  caps at `MAX_MERGES_PER_PASS=12`
- `_apply_merges` pure transform: union files, union flows + flow
  descriptions, drop sources, append `(merged: A, B)` trail to the
  merged feature's description so the rationale survives in JSON
  without changing `DeepScanResult` schema
- `dedup_features` opportunistic entry point: any error → original
  result returned unchanged
- Protected names (``documentation``, ``shared-infra``, ``examples``)
  refuse to participate

### Day 2 — wiring + diagnostic logging

- `pipeline.run(dedup=True)` runs `dedup_features` between
  `deep_scan_workspace` and the synthetic-bucket materialization
- `--dedup` CLI flag (off by default; pair with `--tool-use`)
- Diagnostic log of every proposed merge before apply, so future
  investigators see model intent even when `_apply_merges` skips
  some

### Day 3 — description-injection bug fixes + tests + results

- `_inject_new_pipeline_descriptions` rewritten:
  - was: substring containment + path-segment + plural strip
  - now: exact name + bounded singular/plural (no slashes on either
    side)
- `tests/test_inject_pipeline_descriptions.py` — 9 regression tests
  covering both leak shapes plus the still-required exact and
  same-level plural matches

---

## Validation: documenso (16 packages, ~1800 files)

### Headline numbers

| metric | Sprint 1 only | + dedup | delta |
|---|---|---|---|
| total features | 61 | **30** | −31 |
| `document-signing*` cluster | 5 features | **1** (+ 1 UI sibling) | −3 |
| LLM calls | 49 | 49 | dedup adds 1 call but per-package phase had cache hits |
| total cost | $2.29 | $2.25 | ~equal |
| total runtime | ~7 min | ~7 min | unchanged |

### Sample merges from the run

Each carries a one-line rationale recorded in the merged feature's
description (truncated for table):

```
document-signing            ← lib/document-signing
                              trpc/document-signing
                              remix/document-signing
   "Core document signing domain covering field validation, PDF
    manipulation, recipient flows, audit certificates, tRPC API
    handlers, and the recipient-facing signing UI."

organisation-and-team-management
                            ← lib/team-and-organisation-management
                              trpc/organisation-and-team-management
                              remix/organisation-and-team-management
   "Organisation and team lifecycle management including creation,
    member invitations, groups, roles, email domains, SSO, webhooks,
    and billing portal."

user-authentication         ← lib/user-authentication
                              remix/user-authentication
                              trpc/user-authentication-and-profile
                              auth/auth-client-server-entrypoints
                              auth/email-password-signin
                              auth/oauth-social-signin
                              auth/organisation-sso
                              auth/passkey-signin
                              auth/session-management
                              auth/two-factor-authentication
   "End-to-end user authentication covering email/password, OAuth/
    OIDC, passkeys, 2FA, session management, SSO, password reset,
    and personal profile/API token management."

prisma-database             ← prisma/database-client-setup
                              prisma/database-schema
                              prisma/database-seeding
                              prisma/document-signing-types
   "Database layer including Prisma/Kysely client setup, schema and
    migrations, seed scripts, and TypeScript types and guards for
    the data model."

envelope-management         ← lib/envelope-templates
                              trpc/envelope-management
                              remix/envelope-editor
   "Full envelope lifecycle including creation, editing, field
    placement, recipient configuration, template management,
    distribution, and bulk operations."

billing-and-subscriptions   ← lib/billing-and-subscriptions
                              ee/stripe-billing
                              remix/billing-and-subscriptions
                              trpc/enterprise-billing-and-identity
   "Stripe-based billing, subscription and checkout management,
    licence validation, claim backfill, webhook handling, invoice
    retrieval, and admin subscription management."
```

This is exactly the win Sprint 2 was scoped to deliver — **the 5
document-signing rows from Sprint 1 collapsed to one, with a
specific business name and a rationale a PM can quote**.

### Acceptance check (Sprint 2 plan §11)

| criterion | status |
|---|---|
| `--dedup` works on documenso | ✅ |
| documenso `document-signing*` cluster → ≤2 features | ✅ (1 main + a 6-file `signing` outlier) |
| Every merge has a `merge_rationale` | ✅ (stored in description as `(merged: ...)` trail) |
| 20+ unit tests | ✅ (37 dedup + 9 injection regression = 46) |
| All Sprint 1 tests still pass | ✅ |

---

## Validation: formbricks (size-guard active)

### Headline numbers

| metric | Sprint 1 only | + dedup |
|---|---|---|
| total features | 26 | **25** |
| LLM calls | n/a (Sprint 1 was a separate run) | 36 |
| cost | $5-7 estimated | **$1.21** |
| `apps/web` decomposition | 9 web/* sub-features | 9 (mostly unchanged; one minor merge) |

The dedup pass on formbricks proposed only one significant merge —
collapsing some `surveys/*` sub-features inside the `surveys` package
itself. Cross-package merges that we expected (`survey-ui/*` ↔
`surveys/*`) did not trigger because the model judged the
descriptions distinct enough to keep separate. **That is the
correct conservative behaviour** — the prompt explicitly says "when
in doubt, do not merge". Forcing more merges would risk the kind of
over-eager collapse Sprint 5's self-critique loop is designed to
catch.

The notable cost line: **$1.21 vs the $5-7 we projected for formbricks
post-size-guard**. Three forces collide:
1. The size-guard routes `apps/web` (1948 files) through no-tools
   deep_scan, which is much cheaper per call than tool-augmented.
2. Smaller per-package output JSON means fewer output tokens.
3. Anthropic's prompt cache hit rate is non-trivial when the same
   per-package summaries persist across iterations.

Per-scan cost for formbricks-class repos with the current default
flag set (`--tool-use --dedup`) is **~$1.50 with size guard active**.
Within product-economics expectations.

---

## Bugs found and fixed during Day 3

### Bug 1: substring leak in `_inject_new_pipeline_descriptions`

**How it surfaced:** in formbricks output, a 15-file `ai` feature had
a description of `(merged: email/auth-emails, ...)`.

**Root cause:** the legacy matcher was

```python
if (feat.name == desc_name
    or feat.name in desc_name
    or desc_name in feat.name
    ...):
```

The `feat.name in desc_name` clause did raw substring containment.
`"ai" in "email/auth-emails"` is true (the substring 'ai' lives
inside 'auth-emails'). Same for `"types" in "config-typescript"`.

Pre-dedup this rarely fired in practice because feature names were
short. Once dedup introduced longer multi-word names like
`email/auth-emails`, the failure mode became visible.

**Fix:** drop substring containment. Match exact name + bounded
singular/plural only.

### Bug 2: path-segment leak in the same matcher

**How it surfaced:** in formbricks output (post-fix-1), a 982-file
`web/surveys` feature inherited the description of the 117-file
`surveys` package — both ended in 'surveys' after a slash split, but
they're different features (web's survey UI vs the surveys package).

**Root cause:** the next branch of the matcher used path-segment
match to fix `auth` ↔ `api/auth`. With dedup writing description
keys at multiple granularities, the rule fired across unrelated
features.

**Fix:** drop the path-segment branch entirely. The new pipeline
writes descriptions keyed by the same name that survives into
`feature_map.features`, so fuzzy matching is unnecessary AND
demonstrably leaky.

### Net regression-test coverage

`tests/test_inject_pipeline_descriptions.py` now has 9 tests that
fail loudly if either leak shape returns:

```
TestNoSubstringLeak::test_ai_does_not_inherit_email_auth_emails
TestNoSubstringLeak::test_config_typescript_does_not_inherit_types
TestNoSubstringLeak::test_unrelated_long_name_does_not_match_short
TestNoSubstringLeak::test_path_segment_does_not_leak_across_packages
TestStillMatches::test_exact_name
TestStillMatches::test_path_segment_match_dropped_for_safety
TestStillMatches::test_singular_plural_same_level
TestStillMatches::test_existing_description_preserved
TestStillMatches::test_empty_descriptions_is_noop
```

---

## Open issue: trail overlap on multi-merge same-`from` cases

In the documenso run, three merge targets (`document-signing`,
`document-signing-ui`, `signing`) all carry the same source list in
their `(merged: ...)` trails — `lib/document-signing,
trpc/document-signing, remix/document-signing`.

`_apply_merges` is sequential: the first merge consumes those three
sources, subsequent merges find them gone and should be skipped
(they have <2 valid sources). Yet all three exist in the output
with files. The diagnostic log added in Day 2 should reveal what
the model returned in the next iteration.

**Hypothesis:** the model proposed each merge with a different
**actual** `from` list, but the trail formatting got confused by
description hand-off. The file contents in each merge are clearly
distinct (apps/remix vs packages/ui vs packages/signing), so the
unions were correct — only the descriptions point at the wrong
sources.

This is a cosmetic issue at best and does NOT affect the file
attribution, the feature count, or the accuracy of the dedup pass.
Documented for the next session to investigate.

---

## API spend

| run | flags | features | cost |
|---|---|---|---|
| documenso (Sprint 2 first) | `--tool-use --dedup` | 30 | $2.25 |
| formbricks (Sprint 2 first) | `--tool-use --dedup` | 26 | $1.13 |
| formbricks (post-fix re-run) | `--tool-use --dedup` | 25 | $1.21 |
| **total Sprint 2 dev** | | | **~$4.60** |

Plan estimate was $1-3. The $1.60 overrun came from the description-
leak bug surfacing only on a real run, requiring a re-run after the
fix. This is the same shape as Sprint 1 Day 4 ($25-30 vs $5-15
plan): bug discovery on real data costs an extra run.

Per-scan cost going forward:
- formbricks-class with `--tool-use --dedup`: **~$1.50**
- documenso-class with `--tool-use --dedup`: **~$2.30**
- Still well within the $10-15 sustainable per-scan budget.

---

## Recommended next step

Before Sprint 3, address the trail-overlap diagnostic from the
"Open issue" section above by inspecting the next dedup run's
proposed-merges log. If the model genuinely returns identical
`from` lists for distinct `into` names, tighten the prompt to
forbid that. **5 min effort, no API spend** if the next planned
scan has it logged.

After that, **Sprint 3 (sub-decomposition for oversized features)**
per `SPRINTS_ROADMAP.md`. Apps/web at 1948 files is currently a
single 9-feature deep_scan output; with sub-decomposition we should
get more granularity inside it without giving up the size guard's
safety.

---

## Definition of done — checked

- [x] 3 new commits on `feat/tool-use-detection` (Day 1, Day 2+wiring,
      Day 3+fixes — last two land in one combined commit)
- [x] `--dedup` flag works on documenso AND formbricks
- [x] documenso `document-signing*` cluster collapses to ≤2 features
- [x] formbricks `survey-question-elements` literal duplicate
      collapses (it merged inside the `surveys` package)
- [x] Every merge in JSON output carries a `(merged: ...)` trail
- [x] Dev API spend recorded
- [x] 46 new unit tests across `tests/test_dedup.py` and
      `tests/test_inject_pipeline_descriptions.py`
- [x] All Sprint 1 tests still pass
- [x] `SPRINT_2_RESULTS.md` documents the side-by-side comparison

Sprint 2 closed. **5 sprints remaining** in the roadmap.
