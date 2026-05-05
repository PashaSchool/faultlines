"""Sprint 8 — Smart Aggregator Detection.

Classifies every detected feature into one of four buckets so the
pipeline can re-attribute shared infrastructure (DTO packages,
shared UI primitives, locales, dev-internal docs) to the product
features that actually own those user journeys.

The classifier is one batched Sonnet call per scan. It does NOT use
hardcoded folder names — every decision draws on the model's prior
knowledge of typical codebase structures, applied to the specific
paths and flow names of THIS repo. A real "i18n" admin feature
(translation management UI) gets ``product-feature``; an "i18n"
folder of locale JSON files gets ``developer-internal``. Same name,
different verdict, because the LLM reads what's actually in there.

Output is consumed by ``aggregator_apply.py`` (Day 4) to delete
aggregator features, redistribute their files as
``shared_participants`` on consumers, and rename developer-internal
buckets to plain English.

This module is the Day 1 deliverable: pure classification, no
pipeline wiring, no feature mutation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from faultline.llm.cost import CostTracker, deterministic_params

if TYPE_CHECKING:
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SAMPLE_PATHS = 6
DEFAULT_SAMPLE_FLOWS = 5

# Synthetic buckets the pipeline materializes itself; never send
# them to the classifier. Same protection as Stage 1.45.
_PROTECTED_NAMES: frozenset[str] = frozenset({
    "shared-infra",
    "documentation",
    "examples",
    "developer-infrastructure",
})

FeatureClass = Literal[
    "product-feature",
    "shared-aggregator",
    "developer-internal",
    "tooling-infra",
]


@dataclass
class FeatureClassification:
    """One classifier verdict for one feature.

    ``feature_name`` is the original Sonnet output name (the dict key
    in ``DeepScanResult.features``). ``classification`` is one of the
    four buckets. ``confidence`` is the model's self-rated confidence
    on a 1–5 scale; downstream stages can choose to act only on
    ≥4 verdicts to avoid wrong calls.

    For ``shared-aggregator`` outputs, ``consumer_features`` lists the
    other product features the LLM thinks should consume the
    aggregator's files. The Day 2 callgraph step uses this as a hint
    when import resolution is ambiguous.

    For ``developer-internal`` outputs, ``proposed_name`` is a
    plain-English replacement (e.g. "Translations" for an i18n
    locale folder). When None, the feature folds into the synthetic
    ``developer-infrastructure`` bucket instead.
    """

    feature_name: str
    classification: FeatureClass
    confidence: int  # 1..5
    reasoning: str
    consumer_features: list[str] | None = None  # only for shared-aggregator
    proposed_name: str | None = None  # only for developer-internal or
                                       # product-feature with unclear name


_SYSTEM_PROMPT = """\
You classify detected "features" in a code-analysis tool into one of
four buckets so the dashboard surfaces real product features and
hides shared infrastructure.

You DO NOT use folder names as the rule. ``i18n`` could be a real
product (translation-management UI) or a folder of locale JSON files;
``dto`` could be a shared schemas package or a per-feature DTO
folder. Read the paths and flows — apply your knowledge of typical
codebase structures — decide based on what the feature ACTUALLY
contains.

THE CTO TEST
A feature must be readable by a CTO, EM, or business analyst with
no codebase context. "Authentication", "Billing", "Issue Board",
"Webhook Delivery" pass. "Dto", "Backend Common", "Decorators",
"Shared UI", "Lib", "Utils", "Pre", "Future" do NOT pass — these
describe code organization, not product surface.

THE FOUR BUCKETS

product-feature
  Files all serve ONE user-facing concept. A CTO would understand
  what this feature does for end-users. The name reads as
  business-domain language.
  Examples: "Authentication", "Booking Creation", "AI Chat",
  "Workflow Editor", "Invoice Generation".

shared-aggregator
  Files serve MULTIPLE unrelated business domains as shared
  infrastructure. Common patterns:
    - DTO/schema packages with sub-folders for auth, workflows,
      billing, etc.
    - UI primitive libraries (Button, Modal, Dropdown used by
      every product feature)
    - Type contracts shared across services
    - Shared validation, parsing, formatting helpers
  The right place for these files is AS PARTICIPANTS in the
  features that actually use them — not as a feature on their own.
  When you classify a feature here, also list which product
  features you'd expect to consume its files.

developer-internal
  Real maintenance area but not a product feature. Devs work on
  this; product owners and CTOs don't think of it as a feature.
  Examples:
    - i18n locale JSON files (NOT translation-management UI)
    - Internal documentation folders
    - Static assets / images / fonts
    - E2E test scaffolding
    - Dev tooling scripts
    - Migration scripts
  When possible, propose a plain-English replacement name
  ("Translations", "Internal Documentation", "Static Assets").
  If the area is too miscellaneous to label, leave proposed_name
  null — the pipeline will fold it into a generic
  "developer-infrastructure" bucket.

tooling-infra
  Build, lint, test, format configs as workspace packages.
  ESLint/Prettier/TypeScript/Tailwind configs, Vitest setup,
  Babel presets. The pipeline already folds these into
  ``shared-infra`` automatically before you see them; if one
  reaches you, classify here for completeness.

CONFIDENCE
Rate your confidence 1 to 5:
  5 = unambiguous (clear DTO sub-tree, obvious build config)
  4 = strong signal but some uncertainty
  3 = could go either way; act conservatively
  1-2 = genuinely unsure; downstream skips the rewrite

When in doubt, prefer ``product-feature`` with confidence 3. False
positives on aggregator detection lose product features; false
negatives leave noise but don't break the dashboard.

OUTPUT FORMAT
Return a JSON object:
{
  "classifications": [
    {
      "name": "<original feature name>",
      "class": "product-feature" | "shared-aggregator" | "developer-internal" | "tooling-infra",
      "confidence": 1..5,
      "reasoning": "<one sentence explaining the call>",
      "consumer_features": ["<name>", ...],   // only for shared-aggregator
      "proposed_name": "<plain English>" | null  // only for developer-internal
    },
    ...
  ]
}

No prose outside the JSON. No markdown fences.
"""


def _format_user_message(
    candidates: list[tuple[str, list[str], list[str]]],
) -> str:
    """One feature per ## section: name, sample paths, flow names."""
    parts = ["Features to classify:\n"]
    for name, paths, flows in candidates:
        parts.append(f"\n## {name}")
        parts.append("  paths:")
        for p in paths[:DEFAULT_SAMPLE_PATHS]:
            parts.append(f"    - {p}")
        if flows:
            parts.append("  flows:")
            for fl in flows[:DEFAULT_SAMPLE_FLOWS]:
                parts.append(f"    - {fl}")
    parts.append(
        "\nReturn one JSON object with a 'classifications' array; "
        "one entry per feature above."
    )
    return "\n".join(parts)


def _select_candidates(
    result: "DeepScanResult",
) -> list[tuple[str, list[str], list[str]]]:
    """Build (name, sample_paths, flow_names) triples for every
    non-protected feature in the result."""
    out: list[tuple[str, list[str], list[str]]] = []
    for name, paths in result.features.items():
        if name in _PROTECTED_NAMES:
            continue
        flows = result.flows.get(name, [])
        out.append((name, list(paths), list(flows)))
    return out


def _parse_response(text: str) -> list[dict]:
    """Extract the classifications array from the model response.

    Returns an empty list on any parse failure — the caller falls
    back to leaving features unchanged. Never raises; this stage is
    opportunistic.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    classifications = data.get("classifications", [])
    if not isinstance(classifications, list):
        return []
    return classifications


def _coerce_classification(entry: dict) -> FeatureClassification | None:
    """Validate and shape one model entry into a FeatureClassification.

    Returns None for entries that don't have at minimum a name, class,
    and confidence. Looser fields (reasoning, consumer_features,
    proposed_name) are best-effort.
    """
    name = (entry.get("name") or "").strip()
    cls = (entry.get("class") or "").strip()
    if not name or cls not in {
        "product-feature",
        "shared-aggregator",
        "developer-internal",
        "tooling-infra",
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


def classify_features(
    result: "DeepScanResult",
    *,
    api_key: str | None = None,
    model: str | None = None,
    tracker: CostTracker | None = None,
) -> dict[str, FeatureClassification]:
    """Classify every non-protected feature in a single batched call.

    Returns a mapping ``feature_name -> FeatureClassification``.
    No-op (returns empty dict) when the API call fails or no
    candidates qualify — the pipeline must keep running.
    """
    candidates = _select_candidates(result)
    if not candidates:
        return {}

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("aggregator_detector: no API key — skipping")
        return {}

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("aggregator_detector: anthropic package missing — skipping")
        return {}

    chosen_model = model or DEFAULT_MODEL
    client = Anthropic(api_key=api_key)
    params = deterministic_params(chosen_model)

    try:
        response = client.messages.create(
            model=chosen_model,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": _format_user_message(candidates)}
            ],
            max_tokens=8192,
            **params,
        )
    except Exception as exc:  # noqa: BLE001 — opportunistic
        logger.warning("aggregator_detector: API call failed (%s) — skipping", exc)
        return {}

    if tracker is not None:
        tracker.record(
            provider="anthropic",
            model=chosen_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            label="aggregator_detect",
        )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    raw_entries = _parse_response(text)
    if not raw_entries:
        logger.info("aggregator_detector: empty or unparsable response")
        return {}

    out: dict[str, FeatureClassification] = {}
    for entry in raw_entries:
        verdict = _coerce_classification(entry)
        if verdict is None:
            continue
        # Last-write-wins on duplicate names, but keep highest confidence
        existing = out.get(verdict.feature_name)
        if existing is None or verdict.confidence > existing.confidence:
            out[verdict.feature_name] = verdict

    if out:
        by_class: dict[str, int] = {}
        for v in out.values():
            by_class[v.classification] = by_class.get(v.classification, 0) + 1
        logger.info(
            "aggregator_detector: classified %d feature(s) — %s",
            len(out),
            ", ".join(f"{c}={n}" for c, n in sorted(by_class.items())),
        )
    return out
