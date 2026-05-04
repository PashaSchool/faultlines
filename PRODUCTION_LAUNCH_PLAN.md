# Faultlines — Production Launch Plan

Three workstreams to take Faultlines from "demo on the landing" to a
shippable SaaS product. They can run in parallel — none blocks the
others — but the order below is the recommended sequence.

| # | Workstream | Effort | Why this order |
|---|---|---|---|
| 1 | GitHub App + cloud scans | 7-10d | Distribution unlock — ICP (engineering managers) won't install a CLI |
| 2 | Stripe billing | 3-5d | Without it nobody can pay even if they want to |
| 3 | MCP publishing (Cursor + Anthropic) | 2-3d | Cheapest hype — gets Faultlines in front of every Cursor user |

Total realistic: **12-18 working days** for a solo dev to ship all three
production-ready.

---

## 1. GitHub App + cloud scans

### What it does end-to-end

```
1. User clicks "Install Faultlines" on the landing page.
2. GitHub redirect → user picks repos (org or personal).
3. Faultlines App is installed → installation event fires →
   queue an initial scan per selected repo.
4. Worker fetches installation token, clones the repo (depth 500),
   runs `faultline analyze --llm --flows --line-attribution`.
5. Result lands in Postgres via existing `import_scans.py` schema.
6. On every push to the default branch → re-scan automatic
   (incremental — only changed files re-attribute).
7. On every PR opened / synchronize →
   "Faultlines: this PR touches 3 features. auth (47% bug ratio,
   hotspot)" comment posted via installation token.
8. Weekly digest email to org admins with the top 3 hotspots.
```

### Detailed breakdown

#### 1.1 GitHub App registration (one-time, manual)

Create at <https://github.com/settings/apps/new>:

- **Name:** Faultlines
- **Homepage URL:** <https://faultlines.dev>
- **Callback URL:** <https://faultlines.dev/api/github/callback>
- **Webhook URL:** <https://faultlines.dev/api/webhooks/github>
- **Webhook secret:** random 32-byte string, store as `GH_WEBHOOK_SECRET`
- **Permissions:**
  - Contents: Read (for `git clone`)
  - Metadata: Read
  - Pull requests: Write (for posting comments)
  - Checks: Write (so Faultlines can show up as a status check on PRs)
- **Events:** `installation`, `installation_repositories`, `push`,
  `pull_request`
- **Generate private key** → downloads PEM. Store as `GH_APP_PRIVATE_KEY`
  in env (or AWS Secrets Manager / Vercel encrypted env).
- **Capture:** `GH_APP_ID`, `GH_APP_SLUG` for install URLs.

#### 1.2 Database schema (one migration)

```sql
CREATE TABLE fl_gh_installations (
  installation_id BIGINT PRIMARY KEY,
  account_login   TEXT NOT NULL,
  account_type    TEXT,                -- "User" | "Organization"
  selected_repos  TEXT[],              -- repo full names ("org/repo")
  installed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  user_id         UUID,                -- linked Faultlines user
  org_id          UUID                 -- linked Faultlines org
);

CREATE TABLE fl_scan_jobs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  installation_id BIGINT REFERENCES fl_gh_installations(installation_id),
  repo_full_name  TEXT NOT NULL,
  trigger         TEXT,                -- "install" | "push" | "pr" | "manual"
  status          TEXT NOT NULL,       -- "queued" | "running" | "succeeded" | "failed"
  pr_number       INT,                 -- only for PR scans
  sha             TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at      TIMESTAMPTZ,
  finished_at     TIMESTAMPTZ,
  error           TEXT,
  cost_usd        NUMERIC
);

CREATE INDEX idx_scan_jobs_status ON fl_scan_jobs (status, created_at);
```

#### 1.3 Auth helper — `lib/github-auth.ts`

- **App JWT** — RS256-signed with `GH_APP_PRIVATE_KEY`, 10-minute TTL,
  payload `{iss: APP_ID, exp: now+600}`.
- **Installation token** — POST `/app/installations/{id}/access_tokens`
  with the App JWT, returns a 1-hour token scoped to the installation.
- Cache per `installation_id` in memory (or Redis if multi-instance).

#### 1.4 Webhook handler `/api/webhooks/github`

```
POST /api/webhooks/github
  1. Verify HMAC-SHA256 signature header against payload + secret.
  2. Parse `X-GitHub-Event` header.
  3. Dispatch:
     installation.created            → upsert + enqueue initial scan per repo
     installation.deleted            → cleanup (cascade by FK)
     installation_repositories.added → enqueue initial scan
     installation_repositories.removed → mark scans archived
     push (default branch)           → enqueue scan with SHA
     pull_request.opened|synchronize → enqueue PR scan with PR #
  4. Respond 202 Accepted immediately — worker picks up async.
```

#### 1.5 Job queue (Postgres-based, no Redis needed)

```sql
UPDATE fl_scan_jobs
SET status = 'running', started_at = NOW()
WHERE id = (
  SELECT id FROM fl_scan_jobs
  WHERE status = 'queued'
  ORDER BY created_at
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
RETURNING *;
```

`FOR UPDATE SKIP LOCKED` is what makes this safe across multiple workers
without race conditions.

#### 1.6 Worker process

A separate long-running Node or Python process — **not Vercel Functions**
(60-second timeout would kill cal.com-size scans). Recommended hosts:

- **Fly.io** — $5-10/mo, easy Dockerfile, scales to zero when idle
- **Railway** — similar profile
- **Render** — slightly more expensive but simpler UI

Loop:

```python
while True:
  job = poll_queue()
  if not job:
    time.sleep(5); continue

  token = get_installation_token(job.installation_id)
  tmpdir = mkdtemp()
  try:
    subprocess.run([
      "git", "clone",
      f"https://x-access-token:{token}@github.com/{job.repo_full_name}.git",
      tmpdir, "--depth", "500",
    ], check=True)

    result = subprocess.run([
      "faultline", "analyze", tmpdir,
      "--llm", "--flows", "--line-attribution",
      "--model", "claude-sonnet-4-6",
      "--output", f"{tmpdir}/feature-map.json",
    ], env={"ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"], **os.environ})

    import_scan(f"{tmpdir}/feature-map.json", installation_id=job.installation_id)
    if job.trigger == "pr":
      post_pr_comment(job, token)
    send_digest_email(job)

    update_job(job.id, status="succeeded", cost=parse_cost(result.stderr))
  except Exception as e:
    update_job(job.id, status="failed", error=str(e))
  finally:
    shutil.rmtree(tmpdir)
```

#### 1.7 PR comment

After a PR-scoped scan finishes, compute the diff between base SHA and
head SHA, identify which features moved, and post:

```markdown
## Faultlines analysis

This PR touches **3 features**:

| Feature | Health | Hotspot? |
|---|---|---|
| 🔴 auth | 32 | yes — 58% bug-fix ratio |
| 🟡 billing | 67 | no |
| 🟢 ui-components | 89 | no |

[View full report →](https://faultlines.dev/scans/abc123)
```

Posted via PATCH `/repos/{owner}/{repo}/issues/{pr_number}/comments`
using the installation token.

#### 1.8 Dashboard integration

Wire the existing `/(dashboard)/github-app` page:

- "Install Faultlines" button → `https://github.com/apps/{APP_SLUG}/installations/new`
- After install: list installed repos + most recent scan status per repo
- "Re-scan now" button → enqueue manual job
- Per-repo scan history (10 most recent)

#### 1.9 Notifications

- **Email (Resend, ~$20/mo for 10k emails):** digest after every scan
  with the top 3 hotspots and a link to the full report.
- **Slack (optional):** later — same payload, OAuth to a Slack app that
  posts to a chosen channel.

#### 1.10 Security checklist

- ✅ HMAC-SHA256 signature verification on every webhook
- ✅ Installation tokens have 1-hour TTL — never persist
- ✅ Code is wiped from worker disk after every scan (`shutil.rmtree`)
- ✅ `ANTHROPIC_API_KEY` lives in env only, never committed
- ✅ Rate limit `/api/webhooks/github` (10 req/sec per IP)
- ✅ Cost cap per installation (e.g. $50/mo) — bail early if exceeded

#### 1.11 Running costs

- Vercel Pro: $20/mo (already paid, hosts Next.js)
- Fly.io worker: $5-10/mo (small VM, scales to zero)
- Postgres: existing (Supabase / Neon)
- Anthropic API: $5-15 per cal.com-sized scan, $0.50-2 per push (incremental)
- Resend: $20/mo for 10k emails

---

## 2. MCP server publishing — Cursor + Anthropic

The Faultlines MCP server already exists per `CLAUDE.md`. This work is
about getting it listed in the public directories so every Cursor / Claude
Desktop user can install it with one click.

### 2.1 Cursor MCP marketplace

**Where:** <https://cursor.com/mcp> + GitHub PR to
<https://github.com/cursor-ai/mcp-registry> (or the equivalent
registry repo Cursor uses).

**Required artifacts:**

- `mcp.json` manifest at the project root with:
  - `name: "faultlines"`
  - `description: "Feature-level technical-debt analysis from git
    history. Ask Cursor about hotspots, ownership, and refactor
    candidates without leaving the editor."`
  - `version`, `homepage`, `repository`
  - `runtime: "stdio"`
  - `command` to launch (`faultlines-mcp`)
  - `tools` schema (auto-emitted by the existing MCP server)
- README with install instructions (`pip install faultlines && faultlines-mcp`
  or via npx wrapper).
- 3-5 screenshots of Cursor using the tools (hotspots, feature lookup,
  flow trace).
- Demo video / GIF (~30s) showing one tool call result.

**Submission:**

1. Open PR against Cursor's MCP registry with the manifest entry.
2. Wait for review — usually 1-3 days.
3. After merge, listed at <https://cursor.com/mcp>.

### 2.2 Anthropic / Claude Desktop directory

**Where:** Anthropic publishes MCP servers via the Claude Desktop
extensions list and the public MCP server registry at
<https://github.com/modelcontextprotocol/servers>.

**Required artifacts:**

- Same `mcp.json` shape; Anthropic uses the standard MCP manifest spec
- Add entry under the `community` section of the
  `modelcontextprotocol/servers` README via PR
- Optional: submit to <https://anthropic.com/claude/mcp-servers> if
  there's an application form for the curated directory

**Bonus distribution:**

- One-line install snippet for Claude Desktop config:
  ```json
  {
    "mcpServers": {
      "faultlines": {
        "command": "faultlines-mcp"
      }
    }
  }
  ```
- Tweet thread / Hacker News post the day the listing goes live.

### 2.3 Pre-publication polish

Before either listing goes live:

- Make sure all 11 MCP tools have clear `description` and JSON-schema
  parameters — they show up verbatim in Cursor's tool picker.
- Add a `--version` flag and `--help` to `faultlines-mcp` binary.
- Write a 1-page user guide: "Five questions to ask Faultlines from
  your editor."

---

## 3. Stripe billing integration

Goal: actually charge customers for Team / Team Pro / Enterprise.

### 3.1 Stripe account setup

1. Create Stripe account (or use existing).
2. Verify business identity, set up bank for payouts.
3. Enable **Stripe Billing** + **Stripe Customer Portal** + **Stripe
   Tax** (auto VAT/sales-tax on EU/US/UK invoices).
4. Get `STRIPE_SECRET_KEY` (live + test) and `STRIPE_WEBHOOK_SECRET`.

### 3.2 Products + prices in Stripe Dashboard

| Tier | Stripe product | Stripe price | Lookup key |
|---|---|---|---|
| Team Starter (locked) | `prod_team_starter` | $29/mo recurring | `team_starter_locked_29` |
| Team Starter (list) | same | $49/mo recurring | `team_starter_list_49` |
| Team Pro (locked) | `prod_team_pro` | $99/mo recurring | `team_pro_locked_99` |
| Team Pro (list) | same | $149/mo recurring | `team_pro_list_149` |
| Enterprise | manual invoicing | — | — |

Lookup keys let the code reference prices by name instead of fragile IDs.

### 3.3 Database schema

```sql
ALTER TABLE fl_orgs
  ADD COLUMN stripe_customer_id TEXT,
  ADD COLUMN stripe_subscription_id TEXT,
  ADD COLUMN plan TEXT NOT NULL DEFAULT 'free',  -- free | team | team_pro | enterprise
  ADD COLUMN plan_locked_until TIMESTAMPTZ,      -- 24-month early-bird lock
  ADD COLUMN current_period_end TIMESTAMPTZ;

CREATE TABLE fl_billing_events (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  org_id        UUID REFERENCES fl_orgs(id),
  stripe_event_id TEXT UNIQUE,                   -- idempotency
  type          TEXT,
  payload       JSONB,
  received_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

`stripe_event_id` UNIQUE constraint makes webhook handlers
idempotent — replays are no-ops.

### 3.4 Checkout flow

```
1. Dashboard "Upgrade to Team" button →
   POST /api/billing/checkout {price_lookup_key: "team_starter_locked_29"}

2. Server creates a Stripe Checkout Session:
   - mode: "subscription"
   - line_items: [{price: <looked-up>, quantity: 1}]
   - customer: existing or new (linked to org)
   - success_url: /settings/billing?checkout=success
   - cancel_url:  /pricing
   - metadata: {org_id: "..."}

3. Redirect user to session.url.

4. After payment: Stripe sends checkout.session.completed webhook →
   server links org to subscription, sets plan = "team".
```

### 3.5 Webhook handler `/api/webhooks/stripe`

```
POST /api/webhooks/stripe
  1. Verify Stripe signature (constructEvent with STRIPE_WEBHOOK_SECRET).
  2. Idempotency check — UNIQUE INSERT into fl_billing_events.
  3. Dispatch:
     checkout.session.completed       → set org.plan, org.stripe_subscription_id
     customer.subscription.updated    → update plan + current_period_end
     customer.subscription.deleted    → downgrade to free at period end
     invoice.payment_succeeded        → no-op (logging only)
     invoice.payment_failed           → email org admin, soft-degrade after 14d
  4. Respond 200.
```

### 3.6 Customer portal

One-click button "Manage billing" in `/settings/billing` opens the
Stripe-hosted Customer Portal. Stripe handles:

- Card updates
- Plan upgrade / downgrade
- Cancellation
- Invoice history download

Server only needs to create a billing portal session and redirect.

### 3.7 Plan enforcement

Middleware on protected routes / API endpoints checks:

```ts
const org = await getOrg(req);
if (org.plan === "free" && org.scans_this_month >= 5) {
  return new Response("Upgrade required", { status: 402 });
}
```

Free-tier limits:
- 1 repo connected via GitHub App (more requires upgrade)
- 5 scans per month per repo
- 14-day scan history retention
- No PR comments
- No Slack integration

Team:
- Unlimited repos
- Unlimited scans
- 1-year scan history
- PR comments
- Slack integration

Team Pro:
- Everything in Team
- AI Chat unlimited
- Custom retention
- Priority support
- Multi-org analytics

### 3.8 Early-bird logic

For the first 100 teams on each plan, lock the discounted price for 24
months by setting `plan_locked_until = NOW() + INTERVAL '24 months'`.
A nightly cron job moves locked customers to the list price after their
lock expires (creates a Stripe price update via API).

### 3.9 Invoices & tax

Stripe Tax handles VAT for EU customers, sales tax for US states with
threshold passed. Invoices auto-send to the customer's email on
`invoice.finalized`. Configure invoice branding (logo, colors) in
Stripe Dashboard.

### 3.10 Test plan

Stripe test-mode keys for local dev. Test cards:
- `4242 4242 4242 4242` — succeeds
- `4000 0000 0000 0341` — fails with insufficient funds
- `4000 0000 0000 9995` — fails with card declined

Use `stripe listen --forward-to localhost:3000/api/webhooks/stripe`
during dev to relay live webhooks to the local server.

---

## Short plan — sentence per step

### GitHub App (12 steps)

1. Register Faultlines GitHub App at github.com/settings/apps with read-code + PR-write permissions and `installation`/`push`/`pull_request` events.
2. Add `fl_gh_installations` and `fl_scan_jobs` tables to Postgres via a new migration.
3. Write `lib/github-auth.ts` that generates an App JWT (RS256) and exchanges it for installation tokens with per-installation caching.
4. Build `/api/webhooks/github` Next.js route with HMAC-SHA256 signature verification and dispatch by event type.
5. On `installation.created` upsert installation, list selected repos, and enqueue initial scan jobs.
6. On `push` to default branch and `pull_request.opened/synchronize` enqueue scan jobs in `fl_scan_jobs` with trigger and SHA.
7. Deploy a separate Node worker on Fly.io that polls the queue with `FOR UPDATE SKIP LOCKED` SELECT.
8. Worker clones the repo with the installation token and runs `faultline analyze --llm --flows --line-attribution` as a Python subprocess.
9. After success the worker calls existing `import_scans.py` to push results to Postgres.
10. On PR scans the worker posts a comment with top features touched + their health using the installation token.
11. Wire `/(dashboard)/github-app` page to show the install button, installation status, and per-repo scan history.
12. Add Resend email digest with the top-3 hotspots and ship to production: Vercel for Next.js + Fly.io for worker + Postgres + secrets in env.

### MCP publishing (5 steps)

13. Polish the existing MCP server — clear tool descriptions, JSON-schema parameters, `--version`, `--help`, and a 1-page user guide.
14. Author an `mcp.json` manifest at the project root with name, description, runtime, command, and homepage.
15. Open a PR to <https://github.com/cursor-ai/mcp-registry> adding Faultlines under the analysis category with screenshots and a 30-second demo GIF.
16. Open a PR to <https://github.com/modelcontextprotocol/servers> adding Faultlines under the community section.
17. After both listings go live, post a Hacker News + Twitter thread demonstrating one tool call from inside Cursor.

### Stripe billing (10 steps)

18. Create the Stripe account, enable Billing + Customer Portal + Stripe Tax, and capture `STRIPE_SECRET_KEY` + `STRIPE_WEBHOOK_SECRET`.
19. Create products and recurring prices for Team Starter ($29/$49) and Team Pro ($99/$149) with lookup keys.
20. Add `stripe_customer_id`, `stripe_subscription_id`, `plan`, `plan_locked_until`, and `current_period_end` to `fl_orgs`, plus a `fl_billing_events` table for idempotent webhooks.
21. Build `/api/billing/checkout` Next.js route that creates a Stripe Checkout Session and redirects.
22. Build `/api/webhooks/stripe` route that verifies signature, deduplicates by `stripe_event_id`, and updates org plans on `checkout.session.completed`, `subscription.updated`, and `subscription.deleted`.
23. Build `/api/billing/portal` route that creates a Customer Portal session for plan/cancellation management.
24. Wire `/settings/billing` dashboard page with current plan, usage stats, "Upgrade" button, and "Manage billing" button.
25. Add middleware that enforces tier limits (free = 1 repo + 5 scans/mo, team = unlimited, etc.) on protected routes.
26. Add a nightly cron job that moves customers whose `plan_locked_until` has passed to the list price via Stripe API.
27. Test end-to-end with `stripe listen` locally, then ship to production with live keys after one successful test transaction.

---

## 2-week sprint plan (Mon-Fri × 2) — full launch with team features

Two-week plan splitting "core SaaS rails" (week 1) from "team
collaboration + observability surface area we promised on the
pricing page" (week 2). Testing is baked into every day rather
than concentrated on a single Friday — each day ends with a
verifiable end-of-day milestone, and weeks end with a buffer +
hardening block.

**Why two weeks instead of one:** the pricing-page feature
comparison advertises Slack digests, health trends, outbound
webhooks, and audit logs as Pro-tier features. Shipping without
them lets the first paying customer catch us in a lie. User
confirmed: "не хочу без них стартувати" — bundle them into the
launch sprint, don't ship after.

# Week 1 — Core SaaS rails

This is the compressed version of the workstreams above. Same ship
target — production paid SaaS with cloud scans + MCP distribution
+ Sentry/PostHog observability — squeezed into one calendar week
through ruthless scope cuts and parallel work.

### Scope cuts to make 5 days work

| Cut | Reason |
|---|---|
| Slack notifications | Email-only at launch, Slack post-launch |
| Email digest weekly schedule | Per-scan email at launch, scheduled digest post-launch |
| Customer Portal full UX | Stripe-hosted portal only; no in-app cancellation flow |
| Enterprise tier checkout | Manual invoicing through email |
| Plan downgrade / proration | Subscription deletes at period end; no mid-cycle proration |
| MCP screenshots + demo GIF | One screenshot + 30s loom; submit polished assets in week 2 |
| Per-PR diff comments | Plain "this PR touches X features" comment; diff-based health delta in week 2 |
| Multi-org analytics on Pro | Single-org only at launch; multi-org post-launch |

The cuts touch UX polish, not the core path. Anyone can sign up,
connect a repo, get a scan, see a PR comment, pay, and Faultlines
records what they did.

### Monday — GitHub App foundation

| Hour | Task |
|---|---|
| 09-10 | Register Faultlines GitHub App on github.com/settings/apps with read-Contents + write-PullRequests permissions and `installation`/`push`/`pull_request` events; download private key, store `GH_APP_ID`, `GH_PRIVATE_KEY`, `GH_WEBHOOK_SECRET`, `GH_APP_SLUG` in env |
| 10-12 | Migration creating `fl_gh_installations` and `fl_scan_jobs` tables in Postgres |
| 12-13 | Lunch + coffee |
| 13-15 | Write `lib/github-auth.ts` — App JWT (RS256, 10-min) → installation token (1-hour) with per-installation cache |
| 15-17 | Build `/api/webhooks/github` route with HMAC-SHA256 verification + dispatcher stub (logs each event type, no handlers yet) |
| 17-18 | Provision Fly.io worker app with empty Dockerfile, ANTHROPIC_API_KEY env, healthcheck endpoint |

**Monday end-of-day:** webhook endpoint receives events, signs them,
DB migrations live. No scans run yet.

### Tuesday — Worker + scan integration

| Hour | Task |
|---|---|
| 09-11 | Worker poll loop: SELECT FROM `fl_scan_jobs` ... `FOR UPDATE SKIP LOCKED`; idle sleep 5s |
| 11-13 | Worker action: `git clone` with installation token + `subprocess.run("faultline analyze ...")` + capture stdout/stderr + cleanup tmpdir |
| 13-14 | Lunch |
| 14-16 | Webhook handler: on `installation.created` upsert installation + enqueue jobs per repo; on `push` enqueue with SHA; on `pull_request.opened/synchronize` enqueue with PR number |
| 16-17 | After-scan hook in worker: call existing `import_scans.py` to push to DB |
| 17-18 | First end-to-end test on a private throwaway repo |

**Tuesday end-of-day:** real GitHub event triggers a worker scan
that lands in Postgres. No PR comments yet, no payments.

### Wednesday — Stripe billing

| Hour | Task |
|---|---|
| 09-10 | Set up Stripe account (or activate existing), enable Billing + Customer Portal + Stripe Tax, capture live + test keys |
| 10-11 | Create Stripe products + recurring prices for Team Starter ($29 locked / $49 list) and Team Pro ($99 locked / $149 list) with lookup keys |
| 11-12 | DB migration: `fl_orgs.stripe_customer_id`, `stripe_subscription_id`, `plan`, `plan_locked_until`, `current_period_end` + `fl_billing_events` table for idempotent webhooks |
| 12-13 | Lunch |
| 13-14 | `/api/billing/checkout` route — accept lookup key, create Stripe Checkout Session, redirect |
| 14-16 | `/api/webhooks/stripe` route — verify signature, dedupe by `stripe_event_id` UNIQUE, dispatch on `checkout.session.completed`, `subscription.updated`, `subscription.deleted` |
| 16-17 | `/api/billing/portal` route — create Customer Portal session, redirect |
| 17-18 | Wire `/settings/billing` dashboard page: current plan, usage, "Upgrade" / "Manage billing" buttons; test the full flow with a Stripe test card |

**Wednesday end-of-day:** test card buys a subscription, webhook
flips the org to "team" plan, Customer Portal works.

### Thursday — MCP publishing + PR comments

| Hour | Task |
|---|---|
| 09-10 | Polish existing `faultlines-mcp` server: tool descriptions, JSON-schema parameters, `--version`, `--help` |
| 10-11 | Write `mcp.json` manifest at project root — name, runtime, command, homepage, tools list |
| 11-13 | Open PRs to `cursor-ai/mcp-registry` and `modelcontextprotocol/servers` adding Faultlines under analysis / community section |
| 13-14 | Lunch |
| 14-16 | Worker PR comment integration: on PR-scoped scan finish, compute touched-features list, post comment via PR API with installation token |
| 16-17 | Add plan-enforcement middleware: `org.plan === "free" && scans_this_month >= 5` returns 402 |
| 17-18 | Resend integration for "scan complete" email per scan (replaces digest for v1) |

**Thursday end-of-day:** Cursor + Anthropic PRs open, real PR
comments fire, free tier hits the 5-scan limit, paid users skip
the wall.

### Friday — Sentry + PostHog + ship

| Hour | Task |
|---|---|
| 09-10 | Sentry setup: install `@sentry/nextjs` in landing-app + Sentry SDK in Fly.io worker; capture release version + environment |
| 10-11 | Sentry boundary tests: throw in `/api/webhooks/github`, `/api/webhooks/stripe`, and the worker — confirm errors land in Sentry dashboard within seconds |
| 11-12 | PostHog setup: install `posthog-node` server-side + `posthog-js` client; track key events: `github_app_installed`, `scan_started`, `scan_finished`, `pr_comment_posted`, `checkout_started`, `subscription_created`, `feature_clicked` |
| 12-13 | Lunch |
| 13-14 | PostHog session replay + funnel: install → first scan → upgrade signup. Verify drop-off events match the funnel definition |
| 14-15 | End-to-end smoke test: install app on real repo → push commit → scan completes → PR opens → comment posted → upgrade to Team → Customer Portal opens → cancel → downgrade at period end. Capture Sentry + PostHog evidence at every step |
| 15-16 | Production deploy: Vercel (Next.js), Fly.io (worker), Postgres, all secrets via env. Run smoke test against production |
| 16-17 | Hacker News + Twitter post when MCP listings go live; submit Faultlines to ProductHunt scheduled for next Tuesday |
| 17-18 | Buffer / coffee / fix the one thing that's definitely broken at this point |

**Friday end-of-day:** Faultlines ships with paid plans, GitHub App
flow, MCP listing pending, and full observability via Sentry +
PostHog. Real users can sign up and pay.

### Risk register for the 5-day sprint

| Risk | Mitigation |
|---|---|
| Cursor MCP review takes >5 days | Submit PR Thursday morning; listing goes live in week 2; we'll still have Anthropic listing for hype |
| Stripe webhook signing fails in production | Test with `stripe listen` Wed evening; have rollback plan to delay billing release |
| First real scan exceeds Fly.io worker memory | Cap `max-commits` to 2000 for v1, lift later; worst case fall back to local scan |
| Sentry / PostHog noise drowns real signals | Set sample rate to 0.5 + ignore_urls on health checks; tune in week 2 |
| Solo dev burnout | Friday afternoon is buffer; if behind, ship without PostHog session replay (Sentry alone covers errors, PH funnel can come Monday) |

### What's deferred from week 1 to week 2

These are the features promised on the pricing-page comparison
table that don't fit week 1 — they form the bulk of week 2.

- Slack OAuth + Block Kit digest sending
- Microsoft Teams via incoming webhook URL
- Email digest weekly schedule (vs per-scan email from week 1)
- Outbound webhooks (signed POST + retry, not GitHub App inbound)
- Audit logs (table, writer middleware, viewer page)
- Health trends over time (multi-scan diff + chart)
- Subscriptions to specific features / flows (the "team subscribes
  to feature X" idea from the dashboard)
- Diff-based PR comment with health delta (week 1 ships plain
  "this PR touches X features")
- Polished MCP screenshots + 30s loom asset
- Per-tier scan history retention enforcement (90d / 1y / unlimited)
- Mid-cycle Stripe proration on plan change
- ProductHunt launch (scheduled for week 2 Tuesday)

# Week 2 — Team features + hardening

Same Mon-Fri shape. Each day ships one major feature end-to-end
with its tests; Friday is a hardening + ProductHunt day rather
than another feature day. Goal: by Friday-EOD week 2, every row
on the pricing-page Feature comparison is real, not aspirational.

### Monday — Audit logs + outbound webhooks

| Hour | Task |
|---|---|
| 09-10 | Migration: `fl_audit_log (id, org_id, actor_user_id, action, resource_type, resource_id, ip, user_agent, ts, metadata jsonb)` + index on `(org_id, ts DESC)` |
| 10-12 | Audit-log writer: middleware in Next.js API routes that records mutating actions (subscription change, webhook create, member invite, plan change). Read-only routes skipped. |
| 12-13 | Lunch |
| 13-15 | Migration + UI: `fl_webhook_endpoints (id, org_id, url, secret, events text[], created_at, last_delivery_at, last_status)` + `/dashboard/settings/webhooks` page (create / revoke / view recent deliveries) |
| 15-17 | Outbound webhook delivery: HMAC-SHA256 signed POST, 5-second timeout, 3 retries with exponential backoff, dead-letter to `fl_webhook_failures`. Trigger from `scan.completed`, `feature.health_changed` events. |
| 17-18 | **Test:** trigger a scan → check audit log row appears, webhook fires to webhook.site, signature verifies. Throw mid-delivery → confirm retry + DLQ. |

**Monday EOD:** every mutating action writes audit row; webhook
endpoints can be created and receive signed events.

### Tuesday — Health trends over time

| Hour | Task |
|---|---|
| 09-11 | Backfill: ensure historical scans persist (week 1 already stores them). Add `feature_health_history` materialized view: `(feature_id, scan_id, scan_date, health_score, bug_fix_ratio, total_commits)`. Refresh on each scan import. |
| 11-13 | API: `GET /api/features/[id]/history?days=90` returning `[{scanDate, health, ratio, commits}]`. Cache 5 min via `Cache-Control: s-maxage=300`. |
| 13-14 | Lunch |
| 14-16 | UI: install Recharts, build `<HealthTrendChart>` component on `/dashboard/features/[id]` — line chart of health over time with hover tooltip showing scan date + delta vs previous scan. |
| 16-17 | Empty / sparse state: feature with <2 scans renders "Need 2+ scans to show trend" placeholder, not a broken chart. |
| 17-18 | **Test:** Playwright run — navigate to a feature with 5+ scans, screenshot the chart, assert tooltip on hover. Run against ghost + cal.com fixtures (both have history). |

**Tuesday EOD:** every feature page shows historical health
trend; "Health trends over time" pricing-row no longer aspirational.

### Wednesday — Slack OAuth + Microsoft Teams webhook

| Hour | Task |
|---|---|
| 09-10 | Slack app registration on api.slack.com/apps with `chat:write`, `incoming-webhook`, `channels:read` scopes. Capture `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET`. |
| 10-12 | OAuth flow: `/api/integrations/slack/install` redirects to Slack → `/api/integrations/slack/callback` exchanges code → store team_id + access_token + bot_user_id + selected channel in `fl_integrations` table. |
| 12-13 | Lunch |
| 13-14 | Block Kit digest renderer: `renderHotspotDigest(features) → SlackBlocks` with a header, top-3 hotspots as section blocks with health bars (Unicode ▁▂▃▅█), CTA button to dashboard. |
| 14-16 | Microsoft Teams: simpler — paste-in incoming webhook URL form on `/dashboard/settings/integrations`. Adaptive Card renderer mirrors Block Kit shape. |
| 16-17 | Send-test: "Send test digest" button on integration settings → posts a sample digest to verify wiring before real cron fires. |
| 17-18 | **Test:** install Slack app on dogfood workspace → channel selection → click send-test → verify message renders correctly. Same for Teams via webhook URL. |

**Wednesday EOD:** Slack + Teams both deliver formatted digests on
demand; cron-triggered sending arrives Thursday.

### Thursday — Subscriptions + cron worker

| Hour | Task |
|---|---|
| 09-10 | Migration: `fl_subscriptions (id, org_id, user_id, target_kind enum('feature','flow','org'), target_id, channel enum('email','slack','teams'), frequency enum('daily','weekly','on_change'), last_sent_at, created_at)` |
| 10-12 | Subscription UI: on every `/dashboard/features/[id]` and `/dashboard/flows/[id]` page add a "Subscribe" button → modal with channel toggle (email default-on, Slack/Teams enabled if integration connected) + frequency radio. List subscriptions on `/dashboard/settings/subscriptions`. |
| 12-13 | Lunch |
| 13-15 | Cron worker: Vercel Cron (`vercel.ts` schedule `"0 9 * * *"`) → `/api/cron/digests` route. Query `fl_subscriptions WHERE due(frequency, last_sent_at)`. For each row: fetch latest data → render → dispatch via channel router. |
| 15-16 | `on_change` frequency: separate trigger from `scan.completed` event handler — diff prev vs new health, send only if delta > threshold (5 points or status flip). |
| 16-17 | Email channel: Resend integration with React Email template for digest (reuse Slack Block Kit shape, render to HTML). |
| 17-18 | **Test:** subscribe self to a feature on email + Slack (both `daily`) → manually invoke cron route in dev → verify both channels receive identical content. Subscribe to `on_change` → trigger fake scan with bumped bug ratio → verify only `on_change` subscriber gets pinged. |

**Thursday EOD:** every promised channel × frequency combination
works end-to-end; pricing-page "Slack digests" row real.

### Friday — Hardening, E2E, ProductHunt launch

| Hour | Task |
|---|---|
| 09-11 | Full E2E Playwright suite covering both weeks: install GitHub App → push commit → scan → PR comment → upgrade to Team via Stripe → audit log entry written → subscribe to feature → fake scan completes → digest delivered to email + Slack → cancel subscription via Customer Portal → downgrade at period end. |
| 11-12 | Sentry + PostHog hardening: review week 1's noisy alerts, tune sample rates, add ignore-rules for health checks. Add release annotation for week-2 deploy. Confirm session replay captures the new subscription + integration flows. |
| 12-13 | Lunch |
| 13-14 | Documentation: README updated with screenshot of feature page health trend, Slack digest, audit log viewer. CHANGELOG entry covering the week's features. Pricing page comparison reviewed row-by-row against shipped reality (every "yes" must be defensible). |
| 14-15 | Load test: kick off concurrent scans on 5 throwaway repos via the Fly.io worker, watch Postgres `FOR UPDATE SKIP LOCKED` queue depth, verify Sentry catches any timeouts. Tune worker concurrency cap if needed. |
| 15-16 | Production deploy: full week-2 release. Smoke-test all promised pricing-page features against production with a real Stripe test card. |
| 16-17 | ProductHunt launch goes live (scheduled for Tuesday but published Friday afternoon for weekend traction). Hacker News + Twitter posts. Notify week-1 early-bird signups via email. |
| 17-18 | Buffer / fix the one thing that broke during launch / coffee. |

**Friday EOD week 2:** Faultlines ships at full feature parity with
the pricing page; every comparison-table row backed by code in
production; ProductHunt live.

### Testing strategy across both weeks

Tests aren't a separate day-block — every feature gets a test
within the same day's hour budget:

| Layer | When | What |
|---|---|---|
| Unit | Throughout | Hour-budget within each day; catch logic bugs at the function level |
| Integration | EOD each day | The "**Test:**" line in each day above; runs against real Postgres + real Stripe test mode + real Slack dogfood workspace |
| E2E (Playwright) | Friday week 2 | Full install → scan → upgrade → subscribe → digest flow |
| Load | Friday week 2 | 5 concurrent scans on Fly.io worker |
| Manual smoke | Both Fridays + after every prod deploy | Buy a sub with a Stripe test card, install GH App on a private repo, verify a digest arrives |

Sentry + PostHog are the always-on observers — week 1 wires
them, week 2 hardens them.

### Risk register for the 2-week sprint

| Risk | Mitigation |
|---|---|
| Slack OAuth review takes longer than expected | Slack apps in development mode work for any workspace — review only required for distribution. We don't need distribution week 1; we ship for known orgs only and submit for review week 3 |
| Subscription cron misfires (sends digest twice) | `last_sent_at` updated in same transaction as send; cron uses `FOR UPDATE SKIP LOCKED` on the subscription row |
| Outbound webhook DLQ silently fills | `/dashboard/settings/webhooks` shows last delivery + 7-day failure count; Sentry alert on DLQ depth > 100 |
| Health-trend chart breaks on features with 1 scan | Empty state handled Tuesday; verified via fixtures with 1, 2, and 50 scans |
| ProductHunt traffic crashes Fly.io worker | Pre-warm 2 worker instances Friday morning; rate-limit `/api/billing/checkout` to 30 rpm; static-render landing |
| Solo dev burnout across 10 working days | Friday week 2 has buffer hour 17-18; if week 1 slips, cut "Diff-based PR comment" from Thursday week 1 (it's already deferred to v2 in the cuts) |

### What's deferred from week 2 to v3 (post-launch)

- Slack app distribution review (week 1-2 ships in dev mode; submit
  for full distribution week 3 once we have real install metrics)
- MS Teams native app (week 2 ships incoming-webhook only)
- Audit log retention auto-purge after 90 days (column added,
  cleanup cron later)
- Webhook secret rotation UI with grace period (basic create/revoke
  ships week 2; rotation later)
- Health trend hourly granularity (daily ships week 2)
- Multi-org analytics on Team Pro
- Mid-cycle Stripe proration on plan change
- Polished MCP demo GIF (week 1 submits with screenshot only)

---

## Original 17-step breakdown (for reference)

The detailed 27-step plan above the 5-day sprint stays as the
source of truth for what each step entails — the sprint just
collapses several of them into the same hour block. If anything in
the sprint is unclear, the corresponding numbered step (1-27) has
the full context.
