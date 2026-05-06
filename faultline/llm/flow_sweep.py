"""Sprint 12 Day 6 — Layer C: entry-point sweep + cross-validation.

The flow detector misses entry points that don't fit its tool-augmented
heuristics — Express middleware bodies, dynamic route mounts, message-
queue subscribers, internal admin endpoints. This module:

    1. Harvests every plausible entry point (route + exported handler)
       across the repo.
    2. Filters out those already covered by a flow's symbol set.
    3. Asks Haiku to cluster + name the unattached ones as new flows.
    4. Runs a cross-validation pass: for each feature, asks Haiku
       which UNATTACHED candidates also belong as secondary owners
       of existing flows.

Three public functions for three stages so each is mockable in tests:

    harvest_entry_points(signatures, repo_root) -> list[EntryPoint]
    find_unattached(entry_points, result)        -> list[EntryPoint]
    promote_unattached_as_flows(unattached, result, client) -> int
    cross_validate_neighbours(result, unattached, client)   -> int
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from faultline.llm.cost import CostTracker, deterministic_params

if TYPE_CHECKING:
    from faultline.analyzer.ast_extractor import FileSignature
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4_096
MAX_PROMOTE_BATCH = 30   # max entry points sent to Haiku per call
MAX_CROSS_VAL_BATCH = 20 # max neighbours per feature in cross-val
PROMOTE_CONFIDENCE_FLOOR = 4

# Recognised route-handler symbol-name patterns. Conservative —
# anything matching gets considered a candidate entry point even
# when the language doesn't expose route metadata.
_HANDLER_PATTERNS = (
    r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)$",
    r"^handle[A-Z]",          # handleSubmit, handleClick
    r".*Handler$",            # createHandler, signupHandler
    r".*Controller$",         # AuthController
    r".*Subscriber$",          # EmailSubscriber
    r"^(create|update|delete|fetch|list|get|set|sync)[A-Z]",
)
_HANDLER_RE = re.compile("|".join(_HANDLER_PATTERNS))


@dataclass(frozen=True)
class EntryPoint:
    """One harvested route / handler symbol.

    ``route_method`` and ``route_path`` are populated when the
    underlying file metadata exposed them (Next.js App Router,
    FastAPI, Express). Otherwise both stay ``None`` and the entry
    is identified by its symbol alone.
    """

    file: str
    symbol: str
    kind: str          # "function" | "class" | "const" | "route"
    start_line: int
    end_line: int
    route_method: str | None = None
    route_path: str | None = None

    @property
    def display_name(self) -> str:
        if self.route_method and self.route_path:
            return f"{self.route_method} {self.route_path}"
        return f"{self.file}::{self.symbol}"


# ── Anthropic protocol ─────────────────────────────────────────────────


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...  # pragma: no cover


# ── Stage 1: harvest ──────────────────────────────────────────────────


def harvest_entry_points(
    signatures: dict[str, "FileSignature"],
) -> list[EntryPoint]:
    """Collect every plausible entry point across the repo.

    A symbol qualifies as an entry point if any of:
        * its name matches a HTTP handler pattern (GET / POST / ...)
        * its name matches one of the handler regexes
        * the file's ``routes`` list is non-empty AND the symbol
          appears in ``exports`` (route handlers are exported)
    """
    out: list[EntryPoint] = []
    for path, sig in signatures.items():
        # Route metadata: keep one EntryPoint per declared route.
        # ``sig.routes`` items look like ``"GET /api/users"``.
        symbol_by_name = {sr.name: sr for sr in sig.symbol_ranges}
        for route in sig.routes:
            method, _, route_path = route.partition(" ")
            method = method.upper().strip()
            route_path = route_path.strip()
            anchor = symbol_by_name.get(method)
            if anchor is None and sig.exports:
                # Pick the first exported symbol as anchor — best-effort.
                anchor = symbol_by_name.get(sig.exports[0])
            if anchor is None:
                continue
            out.append(EntryPoint(
                file=path,
                symbol=anchor.name,
                kind="route",
                start_line=anchor.start_line,
                end_line=anchor.end_line,
                route_method=method or None,
                route_path=route_path or None,
            ))

        # Pattern-matched handlers.
        for sr in sig.symbol_ranges:
            if not _HANDLER_RE.match(sr.name):
                continue
            ep = EntryPoint(
                file=path,
                symbol=sr.name,
                kind=sr.kind,
                start_line=sr.start_line,
                end_line=sr.end_line,
            )
            # Skip if already covered by route metadata for same symbol.
            if any(
                e.file == ep.file and e.symbol == ep.symbol
                for e in out
            ):
                continue
            out.append(ep)

    return out


# ── Stage 2: find unattached ──────────────────────────────────────────


def _flow_symbol_index(result: "DeepScanResult") -> set[tuple[str, str]]:
    """Build a (file, symbol) set covering every flow's participants."""
    seen: set[tuple[str, str]] = set()
    for feat_parts in result.flow_participants.values():
        for parts in feat_parts.values():
            for p in parts:
                # Support dict and TracedParticipant attr shapes
                file = (
                    p.get("path") if isinstance(p, dict)
                    else getattr(p, "path", None)
                )
                if not file:
                    continue
                syms = (
                    p.get("symbols") if isinstance(p, dict)
                    else getattr(p, "symbols", None)
                ) or []
                for s in syms:
                    name = (
                        s.get("name") if isinstance(s, dict)
                        else getattr(s, "name", None)
                    )
                    if name:
                        seen.add((file, name))
    return seen


def find_unattached(
    entry_points: list[EntryPoint],
    result: "DeepScanResult",
) -> list[EntryPoint]:
    covered = _flow_symbol_index(result)
    out: list[EntryPoint] = []
    for ep in entry_points:
        if (ep.file, ep.symbol) in covered:
            continue
        out.append(ep)
    return out


# ── Stage 3: Haiku promoter ───────────────────────────────────────────


_PROMOTE_SYSTEM_PROMPT = """\
You name new user-flows from a list of unattached route handlers.

Each input is a (file, symbol, kind, route?) triple representing an
HTTP/event handler that no flow currently owns. Group those that
participate in the same user journey, give each cluster a kebab-case
flow name + a short description, and pick the most plausible owner
feature from the menu.

GROUPING RULES
  - Multiple handlers on the same resource (POST /sessions + DELETE
    /sessions) → one ``manage-sessions-flow`` cluster.
  - CRUD pairs (createX + updateX + deleteX + listX) on the same
    resource → ``manage-X-flow`` (singular).
  - Single isolated handler → its own one-handler flow.

NAMING
  - kebab-case, ends with ``-flow``, ≤4 tokens.
  - Domain-led: ``reset-password-flow`` not
    ``post-reset-password-flow``.

OWNER SELECTION
  - Pick from the supplied feature menu. If none fit obviously,
    leave ``feature_owner`` empty and the caller will skip.

OUTPUT (JSON only, no prose)
{
  "promotions": [
    {"flow_name": "<kebab-case-flow>",
     "description": "<one short sentence>",
     "feature_owner": "<feature_from_menu_or_empty>",
     "confidence": 1..5,
     "members": [
        {"file": "<path>", "symbol": "<name>"}
     ]}
  ]
}

Confidence:
  5 — clearly belongs together AND owner is obvious
  4 — strong cluster, owner plausible
  3 — weaker grouping; default to single-handler flow
  1-2 — do not include in output

Cover every UNATTACHED handler exactly once.
"""


def _promote_prompt(
    entries: list[EntryPoint],
    feature_menu: dict[str, str],
) -> str:
    payload = {
        "feature_menu": [
            {"name": n, "description": d} for n, d in feature_menu.items()
        ],
        "unattached": [
            {
                "file": e.file,
                "symbol": e.symbol,
                "kind": e.kind,
                "route_method": e.route_method,
                "route_path": e.route_path,
            }
            for e in entries
        ],
    }
    return (
        "Cluster + name these unattached handlers as new flows.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def _parse_promote_response(text: str) -> list[dict] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    promos = data.get("promotions")
    return promos if isinstance(promos, list) else None


def promote_unattached_as_flows(
    unattached: list[EntryPoint],
    result: "DeepScanResult",
    *,
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    tracker: CostTracker | None = None,
    confidence_floor: int = PROMOTE_CONFIDENCE_FLOOR,
) -> int:
    """Cluster unattached entry points + add as new flows. Returns
    count of flows created."""
    if not unattached:
        return 0
    feature_menu = {
        name: result.descriptions.get(name, "")
        for name in result.features
        if name not in {"shared-infra", "documentation", "examples"}
    }
    if not feature_menu:
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

    created = 0
    # Batch to keep the prompt under the limit.
    batches = [
        unattached[i : i + MAX_PROMOTE_BATCH]
        for i in range(0, len(unattached), MAX_PROMOTE_BATCH)
    ]
    for batch in batches:
        params = deterministic_params(model)
        try:
            response = client.messages.create(
                model=model,
                system=_PROMOTE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": _promote_prompt(batch, feature_menu),
                }],
                max_tokens=DEFAULT_MAX_TOKENS,
                **params,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("flow_sweep: promote API call failed (%s)", exc)
            continue

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
        promotions = _parse_promote_response(text) or []
        created += _apply_promotions(promotions, batch, result, confidence_floor)

    return created


def _apply_promotions(
    promotions: list[dict],
    batch: list[EntryPoint],
    result: "DeepScanResult",
    confidence_floor: int,
) -> int:
    by_key = {(e.file, e.symbol): e for e in batch}
    created = 0
    for promo in promotions:
        flow_name = (promo.get("flow_name") or "").strip()
        owner = (promo.get("feature_owner") or "").strip()
        try:
            confidence = int(promo.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        members = promo.get("members") or []

        if not flow_name or not owner or confidence < confidence_floor:
            continue
        if owner not in result.features:
            continue
        # Don't re-create a flow that already exists.
        if any(flow_name in flows for flows in result.flows.values()):
            continue

        # Build participant entries from members.
        participants: list[dict] = []
        for m in members:
            file = m.get("file") if isinstance(m, dict) else None
            sym = m.get("symbol") if isinstance(m, dict) else None
            if not file or not sym:
                continue
            ep = by_key.get((file, sym))
            if ep is None:
                continue
            from faultline.models.types import SymbolRange
            participants.append({
                "path": ep.file,
                "layer": "support",
                "depth": 0,
                "side_effect_only": False,
                "symbols": [SymbolRange(
                    name=ep.symbol,
                    start_line=ep.start_line,
                    end_line=ep.end_line,
                    kind=ep.kind,
                )],
            })

        if not participants:
            continue

        result.flows.setdefault(owner, []).append(flow_name)
        desc = (promo.get("description") or "").strip()
        if desc:
            result.flow_descriptions.setdefault(owner, {})[flow_name] = desc
        result.flow_participants.setdefault(owner, {})[flow_name] = participants
        created += 1

    return created


# ── Stage 4: cross-validation ─────────────────────────────────────────


_CROSS_VAL_SYSTEM_PROMPT = """\
You decide whether a feature should claim secondary ownership of
flows that ARE NOT primarily attached to it.

For each feature you receive: name, description, current flows,
and a list of CANDIDATE neighbour flows (each with their primary
owner + description). Return only neighbours whose user journey
clearly TOUCHES this feature too — e.g. "create-organization-flow"
primary in Auth obviously touches Billing (plan selection) and
Notifications (welcome email).

Be conservative. Adding a secondary owner everywhere defeats the
purpose; only pick flows whose connection to THIS feature is
explicit and load-bearing.

OUTPUT (JSON only)
{
  "secondary_claims": [
    {"flow_name": "<flow>",
     "confidence": 1..5,
     "reasoning": "<one short sentence>"}
  ]
}

Confidence:
  5 — flow's name or behaviour explicitly invokes this feature
  4 — strong functional dependency on this feature
  ≤3 — do not include
"""


def _cross_val_prompt(
    feature_name: str,
    feature_desc: str,
    own_flows: list[str],
    neighbours: list[tuple[str, str, str]],   # (flow_name, primary_owner, description)
) -> str:
    payload = {
        "feature": {"name": feature_name, "description": feature_desc},
        "current_flows": own_flows,
        "neighbours": [
            {"flow_name": fn, "primary_owner": po, "description": d}
            for fn, po, d in neighbours
        ],
    }
    return (
        "Which neighbour flows ALSO belong to this feature?\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def _parse_cross_val_response(text: str) -> list[dict] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    claims = data.get("secondary_claims")
    return claims if isinstance(claims, list) else None


def cross_validate_neighbours(
    result: "DeepScanResult",
    *,
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    tracker: CostTracker | None = None,
    confidence_floor: int = 4,
) -> int:
    """Per-feature pass: ask Haiku which neighbour flows also belong.

    Writes into ``result.flow_secondaries`` (the same channel the
    primary judge uses for ``also_belongs_to``). Returns count of
    secondary claims recorded.

    A "neighbour" of feature F is any flow whose primary owner is
    NOT F. To bound prompt cost, we cap candidates to
    ``MAX_CROSS_VAL_BATCH`` per feature, ranked by name-token
    overlap with the feature name.
    """
    feature_menu = {
        name: result.descriptions.get(name, "")
        for name in result.features
        if name not in {"shared-infra", "documentation", "examples"}
    }
    if not feature_menu:
        return 0

    # Index flow → (primary_owner, description) for quick lookup.
    flow_index: dict[str, tuple[str, str]] = {}
    for owner, flow_names in result.flows.items():
        descs = result.flow_descriptions.get(owner, {})
        for fn in flow_names:
            flow_index[fn] = (owner, descs.get(fn, "") if isinstance(descs, dict) else "")

    if client is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return 0
        try:
            from anthropic import Anthropic
        except ImportError:
            return 0
        client = Anthropic(api_key=api_key)

    claims_recorded = 0
    for fname, fdesc in feature_menu.items():
        own_flows = result.flows.get(fname, [])
        neighbours_all = [
            (fn, po, d) for fn, (po, d) in flow_index.items()
            if po != fname
        ]
        if not neighbours_all:
            continue
        # Token-overlap ranking
        feat_tokens = set(re.split(r"[\s\-_/]+", fname.lower()))
        feat_tokens.discard("")
        ranked = sorted(
            neighbours_all,
            key=lambda x: -len(feat_tokens & set(re.split(r"[\s\-_/]+", x[0].lower()))),
        )
        batch = ranked[:MAX_CROSS_VAL_BATCH]
        if not batch:
            continue

        params = deterministic_params(model)
        try:
            response = client.messages.create(
                model=model,
                system=_CROSS_VAL_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": _cross_val_prompt(
                        fname, fdesc, own_flows, batch,
                    ),
                }],
                max_tokens=DEFAULT_MAX_TOKENS,
                **params,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("flow_sweep: cross-val API call failed (%s)", exc)
            continue

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
        claims = _parse_cross_val_response(text) or []
        for claim in claims:
            flow_name = (claim.get("flow_name") or "").strip()
            try:
                confidence = int(claim.get("confidence", 0))
            except (TypeError, ValueError):
                confidence = 0
            if not flow_name or confidence < confidence_floor:
                continue
            if flow_name not in flow_index:
                continue
            existing = result.flow_secondaries.setdefault(flow_name, [])
            if fname not in existing:
                existing.append(fname)
                claims_recorded += 1

    return claims_recorded


# ── Top-level Stage 2.8 driver ────────────────────────────────────────


def run_layer_c(
    result: "DeepScanResult",
    signatures: dict[str, "FileSignature"],
    *,
    api_key: str | None = None,
    client: _AnthropicLike | None = None,
    tracker: CostTracker | None = None,
) -> dict[str, int]:
    """Run the full Layer C sequence. Returns counts for telemetry."""
    entry_points = harvest_entry_points(signatures)
    unattached = find_unattached(entry_points, result)
    promoted = promote_unattached_as_flows(
        unattached, result,
        client=client, api_key=api_key, tracker=tracker,
    )
    cross_val = cross_validate_neighbours(
        result,
        client=client, api_key=api_key, tracker=tracker,
    )
    return {
        "harvested": len(entry_points),
        "unattached": len(unattached),
        "promoted": promoted,
        "cross_val_claims": cross_val,
    }
