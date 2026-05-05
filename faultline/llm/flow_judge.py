"""Sprint 11 — LLM-judged flow re-attribution.

Replaces Sprint 9's Stage 2.6 substring heuristic with a Haiku
batch call that sees each flow + the full feature menu and picks
the best-fit owner. Catches semantic synonyms the heuristic misses
(e.g. ``Sign in via OAuth`` belongs to ``Auth`` even though no
shared substring) and rejects false-positive substring matches the
heuristic would have moved.

Pure-functions split:
    _select_flows_for_judging — gather every flow in the result with
                                 its current owner + description
    _build_prompt              — feature menu + flows → JSON-friendly
                                 user message
    _parse_response            — robust JSON extraction (prose-around-
                                 JSON, malformed input fallbacks)
    _apply_verdicts            — mutate ``result.flows`` /
                                 ``result.flow_descriptions`` /
                                 ``result.flow_participants`` in place

Day 1 deliverable: module + mocked-client tests. Pipeline wiring
lands on Day 2; caching on Day 3.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from faultline.llm.cost import CostTracker, deterministic_params

if TYPE_CHECKING:
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 8_192
DEFAULT_BATCH_SIZE = 50  # flows per Haiku call
CONFIDENCE_FLOOR = 4     # only act on verdicts ≥4

# Synthetic buckets that never participate in flow attribution.
# Stage 2 materializes these AFTER feature detection; their "flows"
# are a quirk of upstream packaging, not real journeys.
_PROTECTED_BUCKETS: frozenset[str] = frozenset({
    "shared-infra",
    "documentation",
    "examples",
    "developer-infrastructure",
})


@dataclass
class FlowVerdict:
    """One judge decision for one flow.

    ``decision`` is either ``"move"`` (with non-None ``destination``)
    or ``"keep"`` (current owner stays). ``confidence`` rates the
    judge's certainty 1–5; downstream applies only when ≥
    :data:`CONFIDENCE_FLOOR`.
    """

    flow_name: str
    current_owner: str
    decision: str  # "move" | "keep"
    destination: str | None
    confidence: int  # 1..5
    reasoning: str = ""


@dataclass
class FlowEntry:
    """One flow as seen by the judge — name + current owner +
    optional description for context."""

    name: str
    current_owner: str
    description: str = ""


_SYSTEM_PROMPT = """\
You re-attribute user-flow names to their best-fit feature in a
code-analysis tool. Each flow currently belongs to a feature based
on which file owns its entry-point — a heuristic that misses cases
where UI components for one domain are split across multiple
features.

For each flow you receive, pick the feature whose business domain
best matches the flow's name and description. Conservative bias:
when no feature clearly matches, KEEP the current owner.

EXAMPLES OF CORRECT MOVES

  flow="Authenticate with Password", current="Vue Blocks",
  features include "Auth" → move to Auth (decision=move, conf=5)

  flow="Manage Auth Configuration", current="Studio",
  features include "Auth" → move to Auth (conf=5)

  flow="Sign in via OAuth", current="Studio",
  features include "Authentication" → move to Authentication
  (conf=5; OAuth sign-in IS authentication even without shared word)

EXAMPLES OF CORRECT KEEPS

  flow="Open Studio Dashboard", current="Studio" → keep (conf=5)

  flow="Manage Subscription Settings", current="Studio",
  features include "Billing" AND "Settings" → keep (conf=2;
  ambiguous — could fit either, conservative default).

  flow="Update Configuration", current="Studio" → keep (conf=2;
  no domain-specific signal in the name).

CONFIDENCE
  5 — clear domain match, current owner clearly wrong
  4 — strong match, current owner weakly fits at best
  3 — could fit, but ambiguous; default to keep
  1-2 — low signal; always keep

Only ``move`` verdicts with confidence ≥4 are applied downstream.

INPUT (JSON)
{
  "features": [
    {"name": "Auth", "description": "..."},
    ...
  ],
  "flows": [
    {"name": "<flow_name>",
     "description": "<one-liner if available>",
     "current_owner": "<feature_name>"},
    ...
  ]
}

OUTPUT (JSON only, no prose, no markdown fences)
{
  "verdicts": [
    {"flow": "<flow_name>",
     "from": "<current_owner>",
     "decision": "move" | "keep",
     "to": "<dest_feature>" | null,
     "confidence": 1..5,
     "reasoning": "<one short sentence>"}
  ]
}

Cover EVERY flow in the input. Do not silently drop any.
"""


def _select_flows_for_judging(result: "DeepScanResult") -> list[FlowEntry]:
    """Walk every (feature, flow) pair into a flat list. Skips
    protected synthetic buckets and features with empty flow lists.
    """
    out: list[FlowEntry] = []
    for feature_name, flow_names in result.flows.items():
        if feature_name in _PROTECTED_BUCKETS:
            continue
        if not flow_names:
            continue
        feat_descs = result.flow_descriptions.get(feature_name, {})
        for flow in flow_names:
            out.append(FlowEntry(
                name=flow,
                current_owner=feature_name,
                description=feat_descs.get(flow, "") if isinstance(feat_descs, dict) else "",
            ))
    return out


def _build_prompt(
    flows: list[FlowEntry],
    features: dict[str, str],  # feature_name → description
) -> str:
    """One JSON message: feature menu + flows."""
    payload = {
        "features": [
            {"name": name, "description": desc}
            for name, desc in features.items()
        ],
        "flows": [
            {
                "name": f.name,
                "description": f.description,
                "current_owner": f.current_owner,
            }
            for f in flows
        ],
    }
    return (
        "Re-attribute these flows to the best-fit feature.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def _parse_response(text: str) -> list[dict] | None:
    """Extract the verdicts array. None on parse failure so the
    caller can fall back gracefully (no moves applied)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    verdicts = data.get("verdicts")
    return verdicts if isinstance(verdicts, list) else None


def _coerce_verdict(entry: dict) -> FlowVerdict | None:
    """Validate one verdict entry. Returns None on missing required
    fields; loose fields (reasoning) are best-effort."""
    flow = (entry.get("flow") or "").strip()
    decision = (entry.get("decision") or "").strip().lower()
    current = (entry.get("from") or "").strip()
    if not flow or decision not in {"move", "keep"} or not current:
        return None

    dest_raw = entry.get("to")
    destination: str | None = None
    if isinstance(dest_raw, str) and dest_raw.strip():
        destination = dest_raw.strip()

    # ``move`` requires a destination
    if decision == "move" and destination is None:
        return None

    try:
        confidence = int(entry.get("confidence", 3))
    except (TypeError, ValueError):
        confidence = 3
    confidence = max(1, min(5, confidence))

    reasoning = (entry.get("reasoning") or "").strip()

    return FlowVerdict(
        flow_name=flow,
        current_owner=current,
        decision=decision,
        destination=destination,
        confidence=confidence,
        reasoning=reasoning,
    )


def _apply_verdicts(
    result: "DeepScanResult",
    verdicts: list[FlowVerdict],
    *,
    confidence_floor: int = CONFIDENCE_FLOOR,
) -> int:
    """Move flows according to high-confidence verdicts.

    Mutates ``result.flows``, ``result.flow_descriptions``, and
    ``result.flow_participants``. Returns the count of moves made.

    Skips:
      * verdicts below the confidence floor (left alone)
      * verdicts whose destination feature doesn't exist in the
        current scan (probably model hallucination)
      * verdicts whose ``from`` mismatches the actual current owner
        (race / out-of-date)
    """
    moves = 0
    for v in verdicts:
        if v.decision != "move":
            continue
        if v.confidence < confidence_floor:
            continue
        if v.destination is None:
            continue
        if v.destination not in result.features:
            logger.debug(
                "flow_judge: skipping move to unknown destination %r",
                v.destination,
            )
            continue

        # Verify current owner still has this flow
        current_flows = result.flows.get(v.current_owner, [])
        if v.flow_name not in current_flows:
            logger.debug(
                "flow_judge: %r no longer on %r — skipping",
                v.flow_name, v.current_owner,
            )
            continue

        # Move flow name
        current_flows.remove(v.flow_name)
        result.flows.setdefault(v.destination, []).append(v.flow_name)

        # Migrate description
        src_descs = result.flow_descriptions.get(v.current_owner, {})
        if isinstance(src_descs, dict) and v.flow_name in src_descs:
            desc = src_descs.pop(v.flow_name)
            result.flow_descriptions.setdefault(
                v.destination, {},
            )[v.flow_name] = desc

        # Migrate participants (Sprint 7 callgraph data)
        src_parts = result.flow_participants.get(v.current_owner, {})
        if isinstance(src_parts, dict) and v.flow_name in src_parts:
            parts = src_parts.pop(v.flow_name)
            result.flow_participants.setdefault(
                v.destination, {},
            )[v.flow_name] = parts

        moves += 1

    return moves


# ── Anthropic client protocol (so tests can inject a fake) ────────────


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...  # pragma: no cover


def _features_for_prompt(result: "DeepScanResult") -> dict[str, str]:
    """Feature menu sent to the judge: name → short description.

    Skips protected buckets so the model can't suggest moving a flow
    to ``shared-infra``.
    """
    out: dict[str, str] = {}
    for name in result.features:
        if name in _PROTECTED_BUCKETS:
            continue
        out[name] = result.descriptions.get(name, "")
    return out


def judge_flow_attribution(
    result: "DeepScanResult",
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracker: CostTracker | None = None,
    client: _AnthropicLike | None = None,
) -> int:
    """Run Haiku judge over every flow and apply high-confidence
    moves in place. Returns the count of moves made.

    Returns 0 (no-op) when:
      * no flows to judge
      * no API key and no client injected
      * Haiku call fails or returns unparseable output
      * every verdict was below confidence floor

    The caller can detect "judge ran but did nothing" vs "judge
    didn't run" via logger output.
    """
    flows = _select_flows_for_judging(result)
    if not flows:
        return 0

    features = _features_for_prompt(result)
    if not features:
        return 0

    # Build / get client
    if client is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("flow_judge: no API key — skipping")
            return 0
        try:
            from anthropic import Anthropic
        except ImportError:
            logger.warning("flow_judge: anthropic package missing — skipping")
            return 0
        client = Anthropic(api_key=api_key)

    # Batch flows
    batches = [flows[i : i + batch_size] for i in range(0, len(flows), batch_size)]
    all_verdicts: list[FlowVerdict] = []

    for batch in batches:
        params = deterministic_params(model)
        try:
            response = client.messages.create(
                model=model,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_prompt(batch, features)}],
                max_tokens=DEFAULT_MAX_TOKENS,
                **params,
            )
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning(
                "flow_judge: API call failed (%s) — skipping batch of %d",
                exc, len(batch),
            )
            continue

        if tracker is not None:
            usage = getattr(response, "usage", None)
            in_t = int(getattr(usage, "input_tokens", 0) or 0)
            out_t = int(getattr(usage, "output_tokens", 0) or 0)
            if in_t or out_t:
                tracker.record(
                    provider="anthropic",
                    model=model,
                    input_tokens=in_t,
                    output_tokens=out_t,
                    label="flow_judge",
                )

        text = ""
        for block in getattr(response, "content", []) or []:
            if hasattr(block, "text"):
                text += block.text

        raw = _parse_response(text)
        if not raw:
            logger.info("flow_judge: empty/unparsable response for batch of %d", len(batch))
            continue

        for entry in raw:
            v = _coerce_verdict(entry)
            if v is not None:
                all_verdicts.append(v)

    moves = _apply_verdicts(result, all_verdicts)
    if moves:
        logger.info(
            "flow_judge: re-attributed %d flow(s) (from %d verdicts across %d batches)",
            moves, len(all_verdicts), len(batches),
        )
    return moves
