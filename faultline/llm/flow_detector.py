"""
LLM-based flow detection within a feature.

Takes a feature's files + their extracted signatures and asks Claude (or Ollama)
to identify distinct user-facing flows — named sequences of actions a user takes
end-to-end through the codebase.

A "flow" is richer than a feature:
  - Feature: "payments"  (business domain)
  - Flows:   "checkout-flow", "refund-flow", "subscription-flow"
"""
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0

from faultline.analyzer.ast_extractor import FileSignature


_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# If a feature has more files than this, send only exports+routes (skip imports)
_SIGNATURE_TRIM_THRESHOLD = 30

# Directories that contain end-to-end tests
_E2E_DIRS = {"e2e", "cypress", "playwright", "integration", "acceptance"}

# File suffixes that indicate an end-to-end test file
_E2E_SUFFIXES = (
    ".spec.ts", ".spec.tsx", ".spec.js", ".spec.jsx",
    ".e2e.ts", ".e2e.js",
    ".cy.ts", ".cy.tsx", ".cy.js",
    ".feature",  # Gherkin/Cucumber
)

_FLOW_SYSTEM_PROMPT = """\
You are a senior software architect analyzing a codebase to identify user-facing flows.

## Task

Given a feature name and the signatures (exports, routes, imports) of its files, \
identify the distinct user-facing flows within that feature.

## What is a flow?

A flow is a named, end-to-end sequence of actions a user takes. It spans multiple \
technical layers (UI component → API route → service → data layer).

Examples:
- "login-flow": LoginForm → POST /api/login → AuthService → session store
- "checkout-flow": CartSummary → POST /api/pay → PaymentService → Stripe
- "filter-flow": SearchBar + FilterPanel → GET /api/search?filters → QueryBuilder

## Rules

1. Flow names: lowercase, hyphen-separated, end in "-flow". Max 3 words. \
   Examples: "login-flow", "password-reset-flow", "checkout-flow".
2. Each file must appear in exactly one flow. No omissions.
3. Only create flows when you see genuinely distinct user journeys. If the \
   feature naturally has one primary user journey, return one flow — do not \
   force artificial splits.
4. Shared utilities, types, and constants that serve multiple flows should \
   be placed in the most closely related flow.
5. Do NOT invent files. Only use the exact paths provided.
6. If an <e2e-anchors> section is provided, treat those flow names as the \
   authoritative names for this feature's flows — they were written by humans \
   describing real user journeys. Prefer these names over invented ones and \
   assign files accordingly.
7. If a <co-changes> section is provided, files that change together frequently \
   are a strong signal they belong to the same flow.
8. If a file has API routes (GET/POST/PUT/DELETE), it is an entry point to a flow. \
   Group other files that serve the same route prefix into the same flow.
"""

_FLOW_USER_PROMPT = """\
Feature: {feature_name}
{e2e_context}
Files and their signatures:
{signatures_text}
{extra_context}
Identify the distinct user-facing flows within the "{feature_name}" feature.
Assign every file to exactly one flow.
"""


def detect_e2e_anchors(files: list[str]) -> dict[str, list[str]]:
    """Finds end-to-end test files and extracts flow names from their filenames.

    Detects files that are either:
    - Inside an e2e directory (e2e/, cypress/, playwright/, etc.)
    - Named with an e2e suffix (.spec.ts, .cy.ts, .e2e.ts, .feature)

    Returns a dict mapping flow_name → [file_paths]. The flow name is derived
    from the filename by stripping test suffixes and normalizing to kebab-case.

    Example:
        "checkout-flow.spec.ts"  → {"checkout-flow": ["tests/e2e/checkout-flow.spec.ts"]}
        "password_reset.cy.ts"   → {"password-reset-flow": ["cypress/password_reset.cy.ts"]}
    """
    result: dict[str, list[str]] = {}

    for f in files:
        path = Path(f)
        parts_lower = [p.lower() for p in path.parts[:-1]]

        is_e2e_dir = any(d in _E2E_DIRS for d in parts_lower)
        is_e2e_suffix = any(path.name.endswith(s) for s in _E2E_SUFFIXES)

        if not (is_e2e_dir or is_e2e_suffix):
            continue

        stem = path.name
        for suffix in _E2E_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        # Strip any remaining .test or .spec
        stem = re.sub(r"\.(test|spec)$", "", stem)

        # Normalize to kebab-case
        flow_name = re.sub(r"[_\s]+", "-", stem).lower().strip("-")
        if not flow_name:
            continue

        if not flow_name.endswith("-flow"):
            flow_name = flow_name + "-flow"

        result.setdefault(flow_name, []).append(f)

    return result


def _format_e2e_anchors(e2e_anchors: dict[str, list[str]]) -> str:
    """Formats e2e anchors as a prompt section for the LLM.

    Returns empty string if no anchors are provided.
    """
    if not e2e_anchors:
        return ""

    lines = [
        f"  {flow_name} → {', '.join(files)}"
        for flow_name, files in sorted(e2e_anchors.items())
    ]
    return (
        "\n<e2e-anchors>\n"
        "End-to-end test files — use these flow names as authoritative anchors:\n"
        + "\n".join(lines)
        + "\n</e2e-anchors>\n"
    )


_MAX_FLOW_COCHANGE_PAIRS = 10
_MAX_FLOW_ROUTE_ENTRIES = 10


def _build_flow_extra_context(
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    commits: list | None,
) -> str:
    """Builds co-change + route hint context for flow detection prompts."""
    parts: list[str] = []

    # Co-change pairs within this feature
    if commits:
        from faultline.analyzer.features import compute_cochange
        feature_set = set(feature_files)
        # Filter commits to only those touching this feature's files
        filtered = [
            c for c in commits
            if any(f in feature_set for f in c.files_changed)
        ]
        if filtered:
            pairs = compute_cochange(filtered)
            # Keep only pairs where both files are in this feature
            pairs = [(f1, f2, s) for f1, f2, s in pairs if f1 in feature_set and f2 in feature_set]
            if pairs:
                lines = [
                    f"  {f1} ↔ {f2} ({int(s * 100)}%)"
                    for f1, f2, s in pairs[:_MAX_FLOW_COCHANGE_PAIRS]
                ]
                parts.append(
                    "<co-changes>\n"
                    "Files changed together frequently — strong signal they belong to the same flow:\n"
                    + "\n".join(lines)
                    + "\n</co-changes>"
                )

    # Route anchors from signatures
    route_lines: list[str] = []
    for path in feature_files:
        sig = signatures.get(path)
        if sig and sig.routes:
            routes_str = ", ".join(sig.routes[:3])
            route_lines.append(f"  {path} → {routes_str}")
            if len(route_lines) >= _MAX_FLOW_ROUTE_ENTRIES:
                break
    if route_lines:
        parts.append(
            "<route-anchors>\n"
            "Files with API routes — each distinct route prefix likely maps to a different flow:\n"
            + "\n".join(route_lines)
            + "\n</route-anchors>"
        )

    return ("\n" + "\n\n".join(parts) + "\n") if parts else ""


class _FlowFileMapping(BaseModel):
    flow_name: str
    files: list[str]


class _FlowDetectionResponse(BaseModel):
    flows: list[_FlowFileMapping]


def detect_flows_llm(
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    api_key: str | None = None,
    e2e_anchors: dict[str, list[str]] | None = None,
    commits: list | None = None,
) -> list[_FlowFileMapping]:
    """
    Detects user-facing flows within a feature using Claude.

    Args:
        feature_name: The name of the feature (e.g. "payments").
        feature_files: All files belonging to this feature.
        signatures: Pre-extracted file signatures (from ast_extractor).
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        e2e_anchors: Optional dict of flow_name → [files] from e2e test detection.
            When provided, these flow names are used as authoritative anchors.
        commits: Optional list of Commit objects for co-change analysis.

    Returns:
        List of FlowFileMapping objects, empty on any failure.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not feature_files:
        return []

    client = anthropic.Anthropic(api_key=key)
    return _call_flow_detection(client, feature_name, feature_files, signatures, e2e_anchors, commits)


def detect_flows_ollama(
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    model: str = _DEFAULT_OLLAMA_MODEL,
    host: str = _DEFAULT_OLLAMA_HOST,
    e2e_anchors: dict[str, list[str]] | None = None,
    commits: list | None = None,
) -> list[_FlowFileMapping]:
    """
    Detects user-facing flows using a local Ollama model.

    Args:
        e2e_anchors: Optional dict of flow_name → [files] from e2e test detection.
            When provided, these flow names are used as authoritative anchors.
        commits: Optional list of Commit objects for co-change analysis.

    Returns:
        List of FlowFileMapping objects, empty on any failure.
    """
    if not feature_files:
        return []

    try:
        import ollama as _ollama
    except ImportError:
        return []

    signatures_text = _build_signatures_text(feature_files, signatures)
    e2e_context = _format_e2e_anchors(e2e_anchors or {})
    extra_context = _build_flow_extra_context(feature_files, signatures, commits)
    prompt = _FLOW_USER_PROMPT.format(
        feature_name=feature_name,
        e2e_context=e2e_context,
        signatures_text=signatures_text,
        extra_context=extra_context,
    )

    try:
        client = _ollama.Client(host=host)
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _FLOW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format=_FlowDetectionResponse.model_json_schema(),
        )
        parsed = _FlowDetectionResponse.model_validate_json(response.message.content)
        return _filter_valid_files(parsed.flows, set(feature_files))
    except (ValidationError, Exception):
        return []


def _call_flow_detection(
    client: anthropic.Anthropic,
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    e2e_anchors: dict[str, list[str]] | None = None,
    commits: list | None = None,
) -> list[_FlowFileMapping]:
    """Calls Claude for flow detection. Returns [] on any failure."""
    signatures_text = _build_signatures_text(feature_files, signatures)
    e2e_context = _format_e2e_anchors(e2e_anchors or {})
    extra_context = _build_flow_extra_context(feature_files, signatures, commits)
    prompt = _FLOW_USER_PROMPT.format(
        feature_name=feature_name,
        e2e_context=e2e_context,
        signatures_text=signatures_text,
        extra_context=extra_context,
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.parse(
                model=_MODEL,
                max_tokens=2048,
                temperature=0,
                system=_FLOW_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=_FlowDetectionResponse,
            )
            return _filter_valid_files(response.parsed_output.flows, set(feature_files))
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("Flow detection failed (attempt %d/%d): %s. Retrying in %.1fs...", attempt + 1, _MAX_RETRIES, e, delay)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            ValidationError,
        ):
            return []
        except anthropic.APIStatusError:
            return []
    return []


def _build_signatures_text(
    feature_files: list[str],
    signatures: dict[str, FileSignature],
) -> str:
    """
    Formats file signatures as a compact text block for the LLM prompt.
    For large features (>_SIGNATURE_TRIM_THRESHOLD), imports are omitted.
    """
    large_feature = len(feature_files) > _SIGNATURE_TRIM_THRESHOLD
    lines: list[str] = []

    for path in feature_files:
        sig = signatures.get(path)
        if sig is None:
            lines.append(f"  {path} → (no signatures extracted)")
            continue

        parts = []
        if sig.exports:
            parts.append(f"exports: {', '.join(sig.exports[:8])}")
        if sig.routes:
            parts.append(f"routes: {', '.join(sig.routes[:5])}")
        if sig.imports:
            if large_feature:
                # Keep only relative imports (flow-critical: who calls whom)
                internal = [i for i in sig.imports if i.startswith(".")]
                if internal:
                    parts.append(f"imports: {', '.join(internal[:5])}")
            else:
                parts.append(f"imports: {', '.join(sig.imports[:5])}")

        if parts:
            lines.append(f"  {path} → {' | '.join(parts)}")
        else:
            lines.append(f"  {path}")

    return "\n".join(lines)


def _filter_valid_files(
    flows: list[_FlowFileMapping],
    allowed_files: set[str],
) -> list[_FlowFileMapping]:
    """Removes hallucinated files, deduplicates, and assigns unassigned files."""
    assigned: set[str] = set()
    result: list[_FlowFileMapping] = []

    # Pass 1: validate and deduplicate
    for flow in flows:
        valid = [f for f in flow.files if f in allowed_files and f not in assigned]
        if valid:
            assigned.update(valid)
            result.append(_FlowFileMapping(flow_name=flow.flow_name, files=valid))

    # Pass 2: assign unassigned files to closest flow by directory overlap
    unassigned = allowed_files - assigned
    if unassigned and result:
        # Build dir→flow index from existing assignments
        flow_dirs: dict[str, int] = {}
        for i, flow in enumerate(result):
            for f in flow.files:
                parent = str(Path(f).parent)
                flow_dirs.setdefault(parent, i)

        extra: dict[int, list[str]] = defaultdict(list)
        for f in sorted(unassigned):
            parent = str(Path(f).parent)
            if parent in flow_dirs:
                extra[flow_dirs[parent]].append(f)
            else:
                # No directory match — assign to the largest flow
                largest_idx = max(range(len(result)), key=lambda i: len(result[i].files))
                extra[largest_idx].append(f)

        for i, files in extra.items():
            result[i] = _FlowFileMapping(
                flow_name=result[i].flow_name,
                files=result[i].files + files,
            )

    return result
