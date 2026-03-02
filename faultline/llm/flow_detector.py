"""
LLM-based flow detection within a feature.

Takes a feature's files + their extracted signatures and asks Claude (or Ollama)
to identify distinct user-facing flows — named sequences of actions a user takes
end-to-end through the codebase.

A "flow" is richer than a feature:
  - Feature: "payments"  (business domain)
  - Flows:   "checkout-flow", "refund-flow", "subscription-flow"
"""
import os
import re
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

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
"""

_FLOW_USER_PROMPT = """\
Feature: {feature_name}
{e2e_context}
Files and their signatures:
{signatures_text}

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

    Returns:
        List of FlowFileMapping objects, empty on any failure.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not feature_files:
        return []

    client = anthropic.Anthropic(api_key=key)
    return _call_flow_detection(client, feature_name, feature_files, signatures, e2e_anchors)


def detect_flows_ollama(
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    model: str = _DEFAULT_OLLAMA_MODEL,
    host: str = _DEFAULT_OLLAMA_HOST,
    e2e_anchors: dict[str, list[str]] | None = None,
) -> list[_FlowFileMapping]:
    """
    Detects user-facing flows using a local Ollama model.

    Args:
        e2e_anchors: Optional dict of flow_name → [files] from e2e test detection.
            When provided, these flow names are used as authoritative anchors.

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
    prompt = _FLOW_USER_PROMPT.format(
        feature_name=feature_name,
        e2e_context=e2e_context,
        signatures_text=signatures_text,
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
) -> list[_FlowFileMapping]:
    """Calls Claude for flow detection. Returns [] on any failure."""
    signatures_text = _build_signatures_text(feature_files, signatures)
    e2e_context = _format_e2e_anchors(e2e_anchors or {})
    prompt = _FLOW_USER_PROMPT.format(
        feature_name=feature_name,
        e2e_context=e2e_context,
        signatures_text=signatures_text,
    )

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
    except (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        ValidationError,
    ):
        return []


def _build_signatures_text(
    feature_files: list[str],
    signatures: dict[str, FileSignature],
) -> str:
    """
    Formats file signatures as a compact text block for the LLM prompt.
    For large features (>_SIGNATURE_TRIM_THRESHOLD), imports are omitted.
    """
    trim_imports = len(feature_files) > _SIGNATURE_TRIM_THRESHOLD
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
        if not trim_imports and sig.imports:
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
    """Removes hallucinated file paths and flows with no valid files."""
    result = []
    for flow in flows:
        valid = [f for f in flow.files if f in allowed_files]
        if valid:
            result.append(_FlowFileMapping(flow_name=flow.flow_name, files=valid))
    return result
