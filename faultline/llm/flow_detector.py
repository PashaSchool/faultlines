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

import anthropic
from pydantic import BaseModel, ValidationError

from faultline.analyzer.ast_extractor import FileSignature


_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# If a feature has more files than this, send only exports+routes (skip imports)
_SIGNATURE_TRIM_THRESHOLD = 30

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
"""

_FLOW_USER_PROMPT = """\
Feature: {feature_name}

Files and their signatures:
{signatures_text}

Identify the distinct user-facing flows within the "{feature_name}" feature.
Assign every file to exactly one flow.
"""


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
) -> list[_FlowFileMapping]:
    """
    Detects user-facing flows within a feature using Claude.

    Args:
        feature_name: The name of the feature (e.g. "payments").
        feature_files: All files belonging to this feature.
        signatures: Pre-extracted file signatures (from ast_extractor).
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.

    Returns:
        List of FlowFileMapping objects, empty on any failure.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not feature_files:
        return []

    client = anthropic.Anthropic(api_key=key)
    return _call_flow_detection(client, feature_name, feature_files, signatures)


def detect_flows_ollama(
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    model: str = _DEFAULT_OLLAMA_MODEL,
    host: str = _DEFAULT_OLLAMA_HOST,
) -> list[_FlowFileMapping]:
    """
    Detects user-facing flows using a local Ollama model.

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
    prompt = _FLOW_USER_PROMPT.format(
        feature_name=feature_name,
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
) -> list[_FlowFileMapping]:
    """Calls Claude for flow detection. Returns [] on any failure."""
    signatures_text = _build_signatures_text(feature_files, signatures)
    prompt = _FLOW_USER_PROMPT.format(
        feature_name=feature_name,
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
