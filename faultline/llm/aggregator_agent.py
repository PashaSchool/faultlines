"""Sprint 9 — agentic aggregator classifier.

Replaces Sprint 8's single-shot Sonnet classifier with a tool-augmented
loop. The agent gets the four-bucket rubric, the list of detected
features, and a tool kit for investigation: read sample files, list
folders, check who imports whom, summarize a feature. It iterates —
forms a hypothesis, queries tools to test it, refines, and only then
emits a verdict.

This is the same workflow a human reviewer follows when triaging a
new repo: don't classify by name; read the code, check the import
graph, decide based on actual evidence.

Public entry point: :func:`agentic_classify_features`. Output shape
matches Sprint 8's :class:`FeatureClassification` dict so the existing
:mod:`faultline.llm.aggregator_apply` and structural safeguards
continue to work unchanged.

Day 2 scope: the agent loop + mocked-client unit tests. Pipeline
wiring is Day 3. No real API calls in tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from faultline.llm.aggregator_detector import FeatureClassification
from faultline.llm.tools import (
    AGGREGATOR_TOOL_SCHEMAS,
    TOOL_SCHEMAS,
    dispatch_tool,
)
from faultline.llm.tool_use_scan import (
    _block_text,
    _block_type,
    _content_to_serializable,
    _final_text,
    _tool_use_fields,
)

if TYPE_CHECKING:  # pragma: no cover
    from faultline.analyzer.symbol_graph import SymbolGraph
    from faultline.llm.cost import CostTracker
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 16_384

# Per-scan tool budget. The agent investigates several features;
# each can take 3-10 tool calls. 200 is a generous ceiling that
# also caps total cost in case of runaway behaviour.
DEFAULT_TOOL_BUDGET = 200

# Synthetic buckets the pipeline owns directly — never sent to the
# classifier. Same set Sprint 8 used.
_PROTECTED_NAMES: frozenset[str] = frozenset({
    "shared-infra",
    "documentation",
    "examples",
    "developer-infrastructure",
})


_SYSTEM_PROMPT = """\
You are a senior engineer triaging detected "features" in a code-
analysis tool. Your job is to classify each feature into one of four
buckets so the dashboard surfaces real product features and hides
shared infrastructure under one drawer.

You have READ-ONLY tools to investigate the codebase: open files,
list directories, search patterns, check who imports whom, summarize
a feature's stats. USE THE TOOLS. Do not guess from names alone —
single-shot guessing was the previous approach and it failed because
'Contracts' and 'Shared UI' read as developer-y but are actually
either real products or shared infrastructure depending on what's
inside them. Only the code tells you.

THE FOUR BUCKETS

product-feature
  Files all serve ONE user-facing concept. A CTO would understand
  what this feature does for end-users. The name reads as
  business-domain language ("Authentication", "Booking Creation",
  "AI Chat", "Workflow Editor", "Invoice Generation").

shared-aggregator
  Files are CONSUMED by multiple product features through imports.
  The feature has no user-facing surface of its own — it's pure
  infrastructure that other features depend on. Examples:
    - DTO / schema / contracts packages
    - Shared UI primitive libraries (Button, Modal, Card)
    - Cross-service utilities (http-client, logger, rate-limiter)

  TO DECIDE: for a candidate, pick 2-3 sample files and call
  consumers_of on each. If consumers live in 3+ DISTINCT product
  features, it's a shared-aggregator. If consumers all live in
  one feature or in untracked code, it's something else.

  When you classify here, list consumer_features with the actual
  features whose code imports the aggregator's files.

developer-internal
  Real maintenance area but not a product feature. Devs work here;
  CTOs don't think of it as a feature.

  (a) ALWAYS FOLD (proposed_name MUST be null):
        Test infrastructure (E2E, fixtures, mocks), CI/CD configs,
        build pipelines, release scripts, internal one-shot
        migration scripts, codegen output, benchmark harnesses.

  (b) RENAME (proposed_name = plain English business label):
        i18n locale JSON files → "Translations"
        Internal-only docs site → "Internal Documentation"
        Static assets / images / fonts → "Static Assets"
        Real schema-versioning UI → "Database Migrations"

  Use proposed_name=null for (a). Use a label for (b).

tooling-infra
  Build/lint/test/format configs as workspace packages. ESLint,
  Prettier, TypeScript, Tailwind configs. The pipeline already
  folds most of these earlier; if one reaches you, classify here.

WORKFLOW

For EACH feature in the input list:

1. Call feature_summary(name) to get its stats.

2. Decide if investigation is needed:
   - Large feature (100+ files) with specific name → likely
     product-feature, no investigation needed unless name is
     suspicious. Skip to step 4.
   - Generic name (Utils, Common, Types, Contracts, Shared, ...) →
     ALWAYS investigate before deciding.
   - Small feature (1-30 files) with cold history → likely
     developer-internal. Quick investigation only.
   - Small + active (3 files, 200+ commits) → likely a hot central
     module. Skip investigation, classify product-feature.

3. Investigate as needed:
   - read_file_head on 2-3 sample files: what's in there?
     Pure types? Logic? UI primitives?
   - list_directory on the parent: how is the package laid out?
     Subfolders matching other features = aggregator signal.
   - consumers_of on 2-3 sample files: who imports them? If
     consumers span 3+ distinct features → shared-aggregator.
   - grep_pattern when needed (e.g., search for "user-facing" or
     "API endpoint" patterns).
   - get_file_commits to see what work the file has been about.

4. Decide and rate confidence (1-5):
   - 5 = clear from evidence. You read files, checked consumers,
     it's unambiguous.
   - 4 = strong signal but some uncertainty. Acceptable to act on.
   - 3 = could go either way. Default to product-feature
     (conservative — losing a real feature is much worse than
     leaving a clutter feature the user can rename later).
   - 1-2 = genuinely unsure. Mark product-feature, confidence 2.

5. Emit one classification entry. Move to the next feature.

When all features are classified, return the final JSON object —
NO OTHER PROSE.

OUTPUT FORMAT

{
  "classifications": [
    {
      "name": "<original feature name>",
      "class": "product-feature" | "shared-aggregator" | "developer-internal" | "tooling-infra",
      "confidence": 1..5,
      "reasoning": "<one sentence — cite what you investigated and what you concluded>",
      "consumer_features": ["..."],   // only for shared-aggregator
      "proposed_name": "..." | null    // only for developer-internal (b) or product-feature with unclear name
    },
    ...
  ]
}

Cover EVERY feature in the input list. Do not silently drop any.
"""


def _build_user_prompt(feature_names: list[str]) -> str:
    """Initial message: list the features to classify."""
    feature_list = "\n".join(f"  - {name}" for name in feature_names)
    return (
        f"Features to classify ({len(feature_names)} total):\n"
        f"{feature_list}\n\n"
        f"Investigate using the tools — call feature_summary on each, "
        f"investigate the suspicious ones with read_file_head, "
        f"list_directory, consumers_of as needed. When all features "
        f"are classified, return the JSON object."
    )


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...  # pragma: no cover


def _select_features(result: "DeepScanResult") -> list[str]:
    """Every non-protected feature gets classified."""
    return [n for n in result.features if n not in _PROTECTED_NAMES]


def _parse_classifications(text: str) -> list[dict] | None:
    """Extract the classifications array from the model's final
    response. Returns None on parse failure so the caller can fall
    back gracefully (no classifications applied → result unchanged).
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    classifications = data.get("classifications")
    return classifications if isinstance(classifications, list) else None


def _coerce_classification(entry: dict) -> FeatureClassification | None:
    """Validate a single classifier entry. Same shape as Sprint 8."""
    name = (entry.get("name") or "").strip()
    cls = (entry.get("class") or "").strip()
    if not name or cls not in {
        "product-feature", "shared-aggregator",
        "developer-internal", "tooling-infra",
    }:
        return None

    try:
        confidence = int(entry.get("confidence", 3))
    except (TypeError, ValueError):
        confidence = 3
    confidence = max(1, min(5, confidence))

    reasoning = (entry.get("reasoning") or "").strip()

    consumer_features: list[str] | None = None
    if cls == "shared-aggregator":
        raw_consumers = entry.get("consumer_features") or []
        if isinstance(raw_consumers, list):
            consumer_features = [
                c.strip() for c in raw_consumers
                if isinstance(c, str) and c.strip()
            ]

    proposed_name: str | None = None
    raw_proposed = entry.get("proposed_name")
    if isinstance(raw_proposed, str) and raw_proposed.strip():
        proposed_name = raw_proposed.strip()

    return FeatureClassification(
        feature_name=name,
        classification=cls,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning=reasoning,
        consumer_features=consumer_features,
        proposed_name=proposed_name,
    )


def _dispatch_with_context(
    tool_name: str,
    tool_input: dict,
    repo_root: Path,
    *,
    symbol_graph: "SymbolGraph | None",
    scan_result: "DeepScanResult",
) -> str:
    """Wrap dispatch_tool so Sprint 9's import-graph and feature-
    summary tools get the symbol_graph and scan_result they need.

    The agent itself doesn't pass these (they're not in its tool
    input schema) — the wrapper injects them by name before
    calling the dispatcher.
    """
    if tool_name in {"imports_of", "consumers_of"}:
        tool_input = dict(tool_input)
        tool_input["_symbol_graph"] = symbol_graph
    elif tool_name == "feature_summary":
        tool_input = dict(tool_input)
        tool_input["_scan_result"] = scan_result
    return dispatch_tool(tool_name, tool_input, repo_root)


def agentic_classify_features(
    result: "DeepScanResult",
    *,
    repo_root: Path,
    symbol_graph: "SymbolGraph | None",
    client: _AnthropicLike,
    model: str = DEFAULT_MODEL,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tracker: "CostTracker | None" = None,
    on_tool_call: Any = None,
) -> dict[str, FeatureClassification]:
    """Run the agentic classifier and return verdicts keyed by
    feature name.

    No-op when no candidates qualify. Returns an empty dict on parse
    failure so the caller can leave features untouched.
    """
    feature_names = _select_features(result)
    if not feature_names:
        return {}

    tools_for_calls = TOOL_SCHEMAS + AGGREGATOR_TOOL_SCHEMAS
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _build_user_prompt(feature_names)},
    ]
    tool_calls_made = 0

    while True:
        # Budget exhausted → drop tools and prompt the model to wrap up.
        tools_for_call = tools_for_calls if tool_calls_made < tool_budget else []

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": _SYSTEM_PROMPT,
            "messages": messages,
        }
        if tools_for_call:
            kwargs["tools"] = tools_for_call
        else:
            messages.append({
                "role": "user",
                "content": (
                    "Tool budget exhausted. Return the final classifications "
                    "JSON now using only the evidence you have already "
                    "gathered. Cover every feature you were given."
                ),
            })
            kwargs["messages"] = messages

        response = client.messages.create(**kwargs)
        content = list(getattr(response, "content", []) or [])
        stop_reason = getattr(response, "stop_reason", None)

        if tracker is not None:
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            if input_tokens or output_tokens:
                tracker.record(
                    provider="anthropic",
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    label="aggregator_agent",
                )
                tracker.check_budget()

        messages.append({
            "role": "assistant",
            "content": _content_to_serializable(content),
        })

        if stop_reason != "tool_use":
            text = _final_text(content)
            entries = _parse_classifications(text)
            if not entries:
                logger.warning(
                    "aggregator_agent: could not parse final classifications "
                    "(stop_reason=%s, %d chars)",
                    stop_reason, len(text),
                )
                return {}
            out: dict[str, FeatureClassification] = {}
            for entry in entries:
                v = _coerce_classification(entry)
                if v is None:
                    continue
                # Last-write-wins on duplicate names; keep highest confidence
                existing = out.get(v.feature_name)
                if existing is None or v.confidence > existing.confidence:
                    out[v.feature_name] = v
            logger.info(
                "aggregator_agent: classified %d feature(s) using %d tool calls",
                len(out), tool_calls_made,
            )
            return out

        # Dispatch every tool_use block, build a single user turn
        # with all tool_result blocks, loop.
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if _block_type(block) != "tool_use":
                continue
            tool_id, tool_name, tool_input = _tool_use_fields(block)
            tool_calls_made += 1
            try:
                result_str = _dispatch_with_context(
                    tool_name, tool_input, repo_root,
                    symbol_graph=symbol_graph,
                    scan_result=result,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                result_str = f"ERROR: tool dispatch crashed: {exc}"
            is_error = result_str.startswith("ERROR:")

            if on_tool_call is not None:
                try:
                    on_tool_call(tool_name, tool_input, result_str)
                except Exception:
                    logger.debug("on_tool_call hook raised; ignoring", exc_info=True)

            logger.info(
                "aggregator_agent: %s(%s) → %d chars%s",
                tool_name,
                ", ".join(f"{k}={v!r}" for k, v in tool_input.items())[:120],
                len(result_str),
                " [ERROR]" if is_error else "",
            )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_str,
                "is_error": is_error,
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning(
                "aggregator_agent: tool_use stop with no blocks dispatched"
            )
            return {}
