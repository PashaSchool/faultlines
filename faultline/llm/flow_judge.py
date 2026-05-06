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

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from faultline.llm.cost import CostTracker, deterministic_params

if TYPE_CHECKING:
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 8_192
DEFAULT_BATCH_SIZE = 50  # flows per Haiku call
CONFIDENCE_FLOOR = 4     # only act on verdicts ≥4

# Cache file naming. Verdicts persist across scans of the same repo
# so re-runs only judge new/changed flows. Invalidated when the
# feature menu changes (different feature set → re-judge all).
DEFAULT_CACHE_DIR = Path.home() / ".faultline"
_CACHE_VERSION = 1  # bump when verdict schema changes

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
    # Sprint 12 Day 3.5 — multi-feature ownership. Names of OTHER
    # features the flow also legitimately belongs to (besides
    # ``destination`` / ``current_owner``). Empty list = single-owner
    # flow (most common). Applied as ``result.flow_secondaries`` so
    # downstream renders can show "shared with: X, Y" badges.
    also_belongs_to: list[str] = field(default_factory=list)


@dataclass
class FlowEntry:
    """One flow as seen by the judge — name + current owner +
    optional description for context.

    Sprint 13 — ``signals`` is an optional per-feature score map
    ``{feature_name: {ownership: 0..1, centrality: 0..1}}`` injected
    by the post-Layer-B re-judge pass. When present, the judge sees
    deterministic evidence on top of pure semantics.
    """

    name: str
    current_owner: str
    description: str = ""
    signals: dict[str, dict[str, float]] | None = None


_SYSTEM_PROMPT = """\
You re-attribute user-flow names to their best-fit feature in a
code-analysis tool. Each flow currently belongs to a feature based
on which file owns its entry-point — a heuristic that misses cases
where UI components for one domain are split across multiple
features.

For each flow you receive, pick the feature whose business domain
best matches the flow's name and description.

DECISION RULES (apply in order)

  1. CATCH-ALL OWNERS MUST RELEASE. If current_owner is a generic
     bucket — translations ("I18N", "Locales"), framework UI shells
     ("UI", "Web", "App Shell", "Studio Dashboard", "Vue Blocks"),
     mega-buckets like "Workflow App" or "Frontend" — and ANY
     domain-specific feature in the menu plausibly fits the flow,
     MOVE (conf=5). These buckets accumulate flows by accident of
     file ownership; they are never the right semantic owner.

  2. DOMAIN MATCH WINS. If the flow name carries a clear domain
     signal (auth/login/signup/password/oauth → Auth-named feature;
     billing/invoice/subscription/pricing → Billing; dataset/
     knowledge → Datasets; etc.) and the menu contains a feature
     for that domain, MOVE (conf=5).

  3. KEEP ONLY WHEN OWNER IS DOMAIN-SPECIFIC. If current_owner is
     itself the domain-specific feature for this flow, KEEP (conf=5).

  4. AMBIGUOUS → KEEP. If two domain features both fit and current
     owner sits between them, KEEP (conf=2).

EXAMPLES

  flow="Authenticate User", current="I18N",
  menu has "Authentication & Access"
    → MOVE to "Authentication & Access" (conf=5)
    (rule 1: I18N is translations bucket; rule 2: auth signal)

  flow="Create Dataset From Pipeline", current="I18N",
  menu has "Datasets"
    → MOVE to "Datasets" (conf=5)

  flow="Sign in via OAuth", current="Studio Dashboard",
  menu has "Auth"
    → MOVE to "Auth" (conf=5)

  flow="Browse Plugin Marketplace", current="Plugin Detail & Configuration"
    → KEEP (conf=5; rule 3, current owner IS the plugin feature)

  flow="Update Configuration", current="Settings"
    → KEEP (conf=4; no other plausible owner)

CONFIDENCE
  5 — clear domain match, current owner clearly wrong
  4 — strong match, current owner weakly fits at best
  3 — could fit, but ambiguous; default to keep
  1-2 — low signal; always keep

Only ``move`` verdicts with confidence ≥4 are applied downstream.

MULTI-FEATURE OWNERSHIP (also_belongs_to)
Some flows legitimately participate in MULTIPLE features. Examples:
  - "Create Organization" touches Auth (user invited), Billing (plan
    selection), and Notifications (welcome email).
  - "Subscribe to Plan" touches Billing (Stripe call) AND Auth
    (renew session token).
  - "Send Magic Link" touches Auth AND Notifications.

When a flow clearly spans 2-3 domains, return ``also_belongs_to``
with the additional feature names. This is a side channel — it does
not replace the primary owner. Cap at 3; only include features whose
involvement is OBVIOUS (named in the flow, or the flow's described
behaviour explicitly mentions them). When in doubt, leave empty.

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
     "reasoning": "<one short sentence>",
     "also_belongs_to": ["<other_feature>", ...]}
  ]
}

``also_belongs_to`` is an array of additional feature names; default
to ``[]`` when the flow has a single clear owner.

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
    """One JSON message: feature menu + flows.

    Sprint 13 — when any ``FlowEntry`` carries ``signals``, render
    them as a "deterministic_evidence" block in the per-flow object.
    The judge is instructed (in the system prompt) to weight high
    ownership / centrality scores when disagreeing with the current
    owner.
    """
    has_signals = any(f.signals for f in flows)
    payload = {
        "features": [
            {"name": name, "description": desc}
            for name, desc in features.items()
        ],
        "flows": [
            _flow_to_payload(f) for f in flows
        ],
    }
    intro = (
        "Re-attribute these flows to the best-fit feature.\n"
        + (
            "\nEach flow includes deterministic_evidence "
            "(file_ownership_pct + call_graph_centrality_pct per "
            "candidate feature). Use it as ground truth: when a "
            "feature has dramatically higher ownership than the "
            "current owner, MOVE there even if the flow's name "
            "looks neutral.\n"
            if has_signals else ""
        )
    )
    return intro + "\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def _flow_to_payload(f: FlowEntry) -> dict:
    out: dict = {
        "name": f.name,
        "description": f.description,
        "current_owner": f.current_owner,
    }
    if f.signals:
        # Trim to the top 5 features by ownership to keep prompt small.
        ranked = sorted(
            f.signals.items(),
            key=lambda kv: -kv[1].get("ownership", 0.0),
        )[:5]
        evidence = []
        for name, scores in ranked:
            ow = round(scores.get("ownership", 0.0) * 100)
            cn = round(scores.get("centrality", 0.0) * 100)
            if ow == 0 and cn == 0:
                continue
            evidence.append({
                "feature": name,
                "file_ownership_pct": ow,
                "call_graph_centrality_pct": cn,
            })
        if evidence:
            out["deterministic_evidence"] = evidence
    return out


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

    raw_also = entry.get("also_belongs_to") or []
    also_belongs_to: list[str] = []
    if isinstance(raw_also, list):
        for x in raw_also:
            if isinstance(x, str) and x.strip():
                v = x.strip()
                # Drop anything pointing back at primary destination /
                # current owner — multi-ownership only adds NEW
                # features; primary stays in ``destination``.
                if v != destination and v != current and v not in also_belongs_to:
                    also_belongs_to.append(v)
        also_belongs_to = also_belongs_to[:3]  # spec cap

    return FlowVerdict(
        flow_name=flow,
        current_owner=current,
        decision=decision,
        destination=destination,
        confidence=confidence,
        reasoning=reasoning,
        also_belongs_to=also_belongs_to,
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
        # Sprint 12 Day 3.5 — record multi-feature ownership for any
        # verdict (move OR keep) whose also_belongs_to is non-empty
        # AND every named feature actually exists in the menu. Skipping
        # unknown features prevents hallucinated targets from sneaking
        # into the side channel.
        if v.also_belongs_to:
            valid = [f for f in v.also_belongs_to if f in result.features]
            if valid:
                result.flow_secondaries.setdefault(v.flow_name, [])
                for f in valid:
                    if f not in result.flow_secondaries[v.flow_name]:
                        result.flow_secondaries[v.flow_name].append(f)

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


# ── Cache layer (Day 3) ──────────────────────────────────────────────


def _slugify_repo(repo_path: str) -> str:
    """Filesystem-safe slug from a repo path. Last path segment,
    lowercase, non-alphanum collapsed to hyphens."""
    if not repo_path:
        return "unknown"
    name = Path(repo_path).name or "unknown"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unknown"


def _feature_set_hash(features: dict[str, str]) -> str:
    """Stable short hash of the feature menu. Changes when feature
    names change → cache invalidates so the judge sees a fresh menu."""
    payload = json.dumps(sorted(features.keys()), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, repo_slug: str) -> Path:
    return cache_dir / f"flow-verdicts-{repo_slug}.json"


def _load_cache(
    cache_dir: Path | None,
    repo_slug: str,
    feature_hash: str,
) -> dict[str, dict[str, Any]]:
    """Read cached verdicts for this repo. Returns the verdict map
    only when the stored feature_set_hash matches the current one;
    on hash mismatch the cache is treated as empty (full re-judge).
    """
    if cache_dir is None:
        return {}
    path = _cache_path(cache_dir, repo_slug)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if data.get("version") != _CACHE_VERSION:
        return {}
    if data.get("feature_set_hash") != feature_hash:
        return {}
    raw = data.get("verdicts", {})
    return raw if isinstance(raw, dict) else {}


def _save_cache(
    cache_dir: Path | None,
    repo_slug: str,
    feature_hash: str,
    verdict_map: dict[str, dict[str, Any]],
) -> None:
    """Persist verdicts. Creates parent directory if needed; silently
    swallows IO errors so a broken cache never blocks a scan."""
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, repo_slug)
        payload = {
            "version": _CACHE_VERSION,
            "feature_set_hash": feature_hash,
            "verdicts": verdict_map,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    except OSError as exc:
        logger.warning("flow_judge: cache save failed (%s)", exc)


def _verdict_to_dict(v: FlowVerdict) -> dict[str, Any]:
    """Serialize a FlowVerdict for the cache file."""
    return asdict(v)


def _verdict_from_dict(d: dict[str, Any]) -> FlowVerdict | None:
    """Reverse of _verdict_to_dict. None on malformed input so cache
    poisoning can't crash a scan."""
    try:
        raw_also = d.get("also_belongs_to") or []
        also = [x for x in raw_also if isinstance(x, str)]
        return FlowVerdict(
            flow_name=d["flow_name"],
            current_owner=d["current_owner"],
            decision=d["decision"],
            destination=d.get("destination"),
            confidence=int(d["confidence"]),
            reasoning=d.get("reasoning", ""),
            also_belongs_to=also,
        )
    except (KeyError, TypeError, ValueError):
        return None


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


def re_judge_with_signals(
    result: "DeepScanResult",
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    tracker: CostTracker | None = None,
    client: _AnthropicLike | None = None,
    disagreement_threshold: float = 0.30,
) -> int:
    """Sprint 13 Day 1 — second judge pass with deterministic signals.

    Runs AFTER Layer B (flow_symbols) populates ``flow_participants``
    so we can compute file ownership per (flow, feature). Selects only
    flows where some non-owner feature has ownership_score ≥
    current_owner_score + ``disagreement_threshold`` and asks Haiku to
    decide. The judge sees the per-feature evidence in the prompt.

    Returns count of moves applied. Skips silently if no flows have
    enough participant data to compute signals.
    """
    from faultline.llm.flow_signals import (
        file_ownership_distribution,
    )

    features = _features_for_prompt(result)
    if not features:
        return 0

    feature_paths = {
        name: result.features.get(name, []) for name in features
    }

    # Collect (owner, flow_name, paths) for every flow with ≥2 paths
    # in flow_participants. Anything fewer can't disagree meaningfully.
    candidates: list[FlowEntry] = []
    for owner, flow_names in result.flows.items():
        if owner in _PROTECTED_BUCKETS:
            continue
        feat_parts = result.flow_participants.get(owner, {})
        for flow_name in flow_names:
            parts = feat_parts.get(flow_name, [])
            paths: list[str] = []
            for p in parts:
                fp = (
                    p.get("path") if isinstance(p, dict)
                    else getattr(p, "path", None)
                )
                if fp and fp not in paths:
                    paths.append(fp)
            if len(paths) < 2:
                continue
            ownership = file_ownership_distribution(paths, feature_paths)
            current_score = ownership.get(owner, 0.0)
            best_other = max(
                ((n, s) for n, s in ownership.items() if n != owner),
                key=lambda x: x[1],
                default=("", 0.0),
            )
            if best_other[1] - current_score < disagreement_threshold:
                continue
            descs = result.flow_descriptions.get(owner, {})
            description = descs.get(flow_name, "") if isinstance(descs, dict) else ""
            candidates.append(FlowEntry(
                name=flow_name,
                current_owner=owner,
                description=description,
                signals={
                    name: {
                        "ownership": ownership.get(name, 0.0),
                        "centrality": 0.0,  # Layer C centrality TBD
                    }
                    for name in features
                },
            ))

    if not candidates:
        return 0

    if client is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return 0
        try:
            from anthropic import Anthropic
        except ImportError:
            return 0
        client = Anthropic(api_key=api_key)

    params = deterministic_params(model)
    try:
        response = client.messages.create(
            model=model,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _build_prompt(candidates, features),
            }],
            max_tokens=DEFAULT_MAX_TOKENS,
            **params,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("re_judge_with_signals: API call failed (%s)", exc)
        return 0

    if tracker is not None:
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                tracker.record(
                    model=model,
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                )
        except Exception:  # noqa: BLE001
            pass

    text = ""
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")
    raw = _parse_response(text) or []
    verdicts = [
        v for entry in raw
        if (v := _coerce_verdict(entry)) is not None
    ]
    return _apply_verdicts(result, verdicts)


def judge_flow_attribution(
    result: "DeepScanResult",
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    tracker: CostTracker | None = None,
    client: _AnthropicLike | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    repo_slug: str | None = None,
) -> int:
    """Run Haiku judge over every flow and apply high-confidence
    moves in place. Returns the count of moves made.

    Returns 0 (no-op) when:
      * no flows to judge
      * no API key and no client injected
      * Haiku call fails or returns unparseable output
      * every verdict was below confidence floor

    Cache (Day 3): when ``cache_dir`` is set, verdicts persist to
    ``<cache_dir>/flow-verdicts-<repo_slug>.json``. On the next scan
    of the same repo, cached verdicts replay for unchanged flows
    (only fresh flows hit the API). Cache invalidates on feature-
    set change (different feature menu → re-judge all).

    Caller can pass ``cache_dir=None`` to disable caching entirely.
    """
    flows = _select_flows_for_judging(result)
    if not flows:
        return 0

    features = _features_for_prompt(result)
    if not features:
        return 0

    # Cache lookup
    if repo_slug is None:
        repo_slug = _slugify_repo(getattr(result, "repo_path", "") or "")
    feature_hash = _feature_set_hash(features)
    cache = _load_cache(cache_dir, repo_slug, feature_hash)

    cached_verdicts: list[FlowVerdict] = []
    fresh_flows: list[FlowEntry] = []
    for f in flows:
        # Cache key includes the current owner so the same flow on a
        # different owner gets re-judged
        key = f"{f.current_owner}::{f.name}"
        if key in cache:
            v = _verdict_from_dict(cache[key])
            if v is not None:
                cached_verdicts.append(v)
                continue
        fresh_flows.append(f)

    if cached_verdicts:
        logger.info(
            "flow_judge: cache hit on %d/%d flow(s) — %d to judge fresh",
            len(cached_verdicts), len(flows), len(fresh_flows),
        )

    # Build / get client (only when there's fresh work)
    if fresh_flows:
        if client is None:
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("flow_judge: no API key — skipping fresh batch")
                # Still apply cached verdicts even without API
                fresh_flows = []
            else:
                try:
                    from anthropic import Anthropic
                except ImportError:
                    logger.warning("flow_judge: anthropic package missing")
                    fresh_flows = []
                else:
                    client = Anthropic(api_key=api_key)

    # Batch fresh flows
    batches = (
        [fresh_flows[i : i + batch_size] for i in range(0, len(fresh_flows), batch_size)]
        if fresh_flows else []
    )
    all_verdicts: list[FlowVerdict] = list(cached_verdicts)
    fresh_verdicts: list[FlowVerdict] = []

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
                fresh_verdicts.append(v)
                all_verdicts.append(v)

    # Persist cache: keep cached entries + add fresh ones
    if cache_dir is not None:
        merged: dict[str, dict[str, Any]] = dict(cache)  # start with old hits
        for v in fresh_verdicts:
            key = f"{v.current_owner}::{v.flow_name}"
            merged[key] = _verdict_to_dict(v)
        _save_cache(cache_dir, repo_slug, feature_hash, merged)

    moves = _apply_verdicts(result, all_verdicts)
    if moves:
        logger.info(
            "flow_judge: re-attributed %d flow(s) "
            "(%d cached + %d fresh across %d batches)",
            moves, len(cached_verdicts), len(fresh_verdicts), len(batches),
        )
    return moves
