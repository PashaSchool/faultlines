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

## Sequencing — what to do first

**Week 1:** GitHub App skeleton (steps 1-6) + Stripe products and DB
schema (steps 18-20). Both can run in parallel since they touch
different files.

**Week 2:** GitHub App worker (steps 7-11) + Stripe checkout flow
(steps 21-25). The worker is the long pole; Stripe checkout fills
in around it.

**Week 3:** MCP publishing (steps 13-17), email notifications, plan
enforcement (steps 26-27), and end-to-end testing of all three
workstreams together. Ship to production at the end of the week.

Total: **~3 working weeks** for a focused solo dev to ship a paying SaaS
with cloud scans, MCP distribution, and Stripe-backed subscriptions.
