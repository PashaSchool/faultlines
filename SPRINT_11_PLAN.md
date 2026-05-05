# Sprint 11 — LLM-Judged Flow Re-attribution

**Branch:** `feat/sprint11-semantic-flows`
**Approach:** Option B — Haiku judge per flow
**Goal:** Lift flow-attribution correctness from heuristic ~30-50% (Stage 2.6) to ~90-95% by asking Haiku, for each flow, which feature it semantically belongs to.

## Scope (strict)

✅ **In scope:**
- `result.flows` (feature_name → list[flow_name]) re-attribution
- `result.flow_descriptions` migration alongside
- `result.flow_participants` migration alongside

❌ **Out of scope:**
- Feature classification (Sprint 9 agent path stays)
- Feature names / renames
- Feature counts (no creating/deleting features)
- `Feature.paths` owned files
- `shared_participants` (Sprint 8/9)

Sprint 11 ONLY moves existing flows between existing features. No reshape, just clean attribution.

## Why Haiku judge over embeddings

User picked Option B for accuracy. Tradeoffs:

| | Embedding | Haiku | Hybrid |
|--|-----------|-------|--------|
| Cost/scan | ~$0.001 | ~$0.20 | ~$0.05 |
| Accuracy | 75-85% | 90-95% | 92% |
| Determinism | strong | temperature=0 | strong |
| Latency | fast | medium (per-flow call) | medium |

Haiku is the right choice when accuracy matters more than $0.20 per scan. Solo dev / B2B SaaS — accuracy wins.

## Architecture

### New module: `faultline/llm/flow_judge.py`

```python
def judge_flow_attribution(
    result: DeepScanResult,
    *,
    api_key: str | None = None,
    model: str = "claude-haiku-4-5",
    tracker: CostTracker | None = None,
) -> DeepScanResult:
    """Re-attribute flows using a Haiku batch call.

    For each flow currently attached to feature X, ask Haiku: given
    the flow name, the flow's description (when present), and the
    list of all detected features in this scan, which feature does
    this flow logically belong to?

    Batched: single call with up to 50 flows + the feature menu.
    Larger scans split into multiple calls. Each flow gets a verdict
    + confidence (1-5).
    """
```

### Pipeline placement: replace Stage 2.6

```
Stage 2.5  collapse_same_name (final dedup)
Stage 2.6  judge_flow_attribution  ← NEW (replaces heuristic)
           Falls back to _reattribute_flows_by_name_match when
           no API key (cheap heuristic stays as safety net)
Stage 3    orphan validation
```

The current Stage 2.6 (Sprint 9 heuristic) becomes the fallback when:
- No API key available
- Cost budget exhausted
- Haiku call fails

### Prompt design

```
SYSTEM: You re-attribute user-flow names to their best-fit feature
in a code-analysis tool. Each flow currently belongs to a feature
based on which file owns its entry-point. That heuristic misses
cases — auth-related flows might land on the parent app shell
instead of the Auth feature itself.

For each flow, pick the feature whose business domain best matches
the flow's name and description. Conservative bias: when no feature
clearly matches, KEEP the current owner (set decision="keep").

INPUT (JSON):
{
  "features": [
    {"name": "Auth", "description": "User login and signup."},
    {"name": "Billing", "description": "Stripe checkout."},
    ...
  ],
  "flows": [
    {"name": "Authenticate with Password",
     "description": "Login form submission.",
     "current_owner": "Vue Blocks"},
    ...
  ]
}

OUTPUT (JSON only):
{
  "moves": [
    {"flow": "Authenticate with Password",
     "from": "Vue Blocks",
     "to": "Auth",
     "confidence": 5,
     "reasoning": "Login form is the canonical Auth surface."}
  ]
}
```

## Cost guards

- **Per-scan budget cap:** skip Sprint 11 stage when scan total cost > `--max-cost` (default unset)
- **Per-flow batch:** 50 flows per Haiku call max → bigger scans split into 2-4 calls
- **Cache:** same flow-name on next scan reuses verdict from `~/.faultline/flow-verdicts-<repo>.json`
- **Confidence floor:** only act on verdicts ≥4 (or fall back to heuristic)
- **Fallback:** when API call fails or budget exhausted, run Stage 2.6 heuristic (current behavior)

## Risk register

1. **Haiku misjudges multi-domain flows.** "Manage Subscription Settings" — Billing or Settings? Mitigation: prompt asks for confidence 1-5; threshold 4+ to act, otherwise keep current.

2. **Determinism via temperature=0** — same input same output. Caching makes consecutive scans free for unchanged flows.

3. **Cost overrun on huge repos.** n8n's 288 flows × $0.001 = $0.30 alone. Plus on-top-of-existing scan cost. Mitigation: per-scan budget cap.

4. **Wrong moves degrade dashboard.** A misattributed flow now is worse than the original. Mitigation: confidence floor + always keep current owner as default. Lots of unit tests.

## 5-day plan

### Day 1 — module + mocked tests

- Open `faultline/llm/flow_judge.py`
- Pure functions: `_select_flows_for_judging`, `_build_prompt`, `_parse_response`, `_apply_verdicts`
- Unit tests with synthetic flows + mocked Anthropic client
- No real API call; no pipeline wiring

### Day 2 — pipeline wiring + fallback

- Wire Stage 2.6 to call `judge_flow_attribution` when API key + within budget
- Fall back to `_reattribute_flows_by_name_match` (current heuristic) otherwise
- Add `--no-flow-judge` opt-out flag (default on)
- Cost tracker wiring

### Day 3 — caching layer

- Save flow verdicts to `~/.faultline/flow-verdicts-<repo>.json` keyed by flow-name
- Reuse on next scan (skip Haiku call) when flow + features unchanged
- Invalidate when feature set changes (different feature names → re-judge all)

### Day 4 — validation runs

- Re-scan supabase (Auth flow case), n8n (workflow flows), dify
- Check: do `Auth` features now have flows? Do CRUD-junk flows still leak?
- Cost check: real Haiku spend per repo
- ~$5-10 in validation runs

### Day 5 — ship + landing update

- Update SPRINT_9_SCAN_LOG.md with Sprint 11-era flow numbers
- Recompute landing flow real-journey % across repos that benefited
- Promote landing eval block if numbers improve materially

## What I'll learn each day

Day 1: how the prompt structure produces stable verdicts on the
mocked fixtures.

Day 2: whether the fallback chain feels right when budget triggers.

Day 3: cache hit rate on a re-scan of a repo I already judged.

Day 4: real-world supabase Auth feature finally has flows attached.

Day 5: aggregate flow real-journey % lifts noticeably (target +5-8pp).

## Out of scope (future)

- **Embedding similarity layer** as cheap pre-filter for top-3 candidates per flow before Haiku judge. Hybrid Option C from the design call. Sprint 12 candidate.
- **Flow renaming** (re-write CRUD-junk flow names like `delete-ee-flow` to real journeys). Different problem. Sprint 13 candidate.
- **Flow merging** (deduplicate flows with near-identical names across features). Sprint 14 candidate.

## Working agreements

- Each day's PR ships independently, tests passing
- `--no-flow-judge` opt-out from day 1; default-on once Day 4 validates
- Cost guardrail: never run a paid scan in CI; manual scans only with explicit budget
- Day 4 checkpoint with founder before promoting to default landing
