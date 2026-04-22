"""LLM-based symbol → flow attribution.

For each feature with flows, asks Claude which functions belong to
which flows based on symbol names, file paths, and flow descriptions.
Types and interfaces are attributed to the feature level only.

One LLM call per feature (not per flow). Skips features that have
no flows or no flow-eligible symbols.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from faultline.llm.cost import CostTracker, deterministic_params
from faultline.models.types import Feature, SymbolAttribution
from faultline.symbols.extractor import FileSymbols

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:14b"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_MAX_SYMBOLS_IN_PROMPT = 120


def attribute_symbols_to_flows(
    feature: Feature,
    file_symbols: dict[str, FileSymbols],
    *,
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    ollama_host: str = _DEFAULT_OLLAMA_HOST,
    tracker: CostTracker | None = None,
) -> None:
    """Mutates the feature in place to add symbol_attributions on each flow.

    Rules:
      1. Types/interfaces/enums → feature.shared_attributions (existing field).
      2. Functions/classes/consts → one or more flow.symbol_attributions.
      3. A symbol can belong to multiple flows if used across journeys.
      4. Falls back silently if LLM call fails — file-level paths remain
         as the primary attribution.

    Args:
        feature: Feature to enrich. flows must already be populated.
        file_symbols: Output from extractor.extract_file_symbols().
        provider: "anthropic" (default) or "ollama" for local-free attribution.
        api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env).
            Ignored for provider="ollama".
        model: Model id override. Defaults to Sonnet for anthropic,
            qwen2.5-coder:14b for ollama.
        ollama_host: Ollama server URL. Only used when provider="ollama".
        tracker: Shared cost tracker. Ollama calls record zero cost.
    """
    if not feature.flows:
        return

    feature_file_symbols: dict[str, FileSymbols] = {}
    relevant_symbols: dict[str, list[str]] = {}

    for path in feature.paths:
        fs = file_symbols.get(path)
        if not fs:
            continue
        feature_file_symbols[path] = fs
        if fs.flow_symbols:
            relevant_symbols[path] = fs.flow_symbols

    # Attribute types/interfaces/enums to feature level directly (no LLM).
    _attach_feature_attributions(feature, feature_file_symbols)

    if not relevant_symbols:
        return

    # Nothing to do if flows are empty or all flows already span
    # all files (no meaningful split possible).
    total_flow_symbols = sum(len(s) for s in relevant_symbols.values())
    if total_flow_symbols == 0:
        return

    if provider == "ollama":
        mapping = _ask_ollama(
            feature=feature,
            relevant_symbols=relevant_symbols,
            model=model or _DEFAULT_OLLAMA_MODEL,
            host=ollama_host,
        )
    else:
        mapping = _ask_llm(
            feature=feature,
            relevant_symbols=relevant_symbols,
            api_key=api_key,
            model=model or _DEFAULT_MODEL,
            tracker=tracker,
        )
    if not mapping:
        return

    _apply_mapping_to_flows(feature, feature_file_symbols, mapping)


def _attach_feature_attributions(
    feature: Feature,
    file_symbols: dict[str, FileSymbols],
) -> None:
    """Adds feature-level shared_attributions for types/interfaces."""
    for path, fs in file_symbols.items():
        if not fs.feature_symbols:
            continue
        ranges = _resolve_ranges(fs.feature_symbols, fs.symbol_lines)
        roles = {s: fs.symbol_roles.get(s, "type") for s in fs.feature_symbols}
        feature.shared_attributions.append(
            SymbolAttribution(
                file_path=path,
                symbols=fs.feature_symbols,
                line_ranges=ranges,
                attributed_lines=_sum_lines(ranges),
                total_file_lines=fs.total_file_lines,
                roles=roles,
            ),
        )


def _ask_llm(
    *,
    feature: Feature,
    relevant_symbols: dict[str, list[str]],
    api_key: str | None,
    model: str,
    tracker: CostTracker | None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Returns {flow_name: [{file: str, symbols: [str]}]} or None on error."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic not installed — skipping symbol attribution")
        return None

    prompt = _build_prompt(feature, relevant_symbols)
    if not prompt:
        return None

    try:
        client = Anthropic(api_key=api_key) if api_key else Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
            **deterministic_params(model),
        )
    except Exception as exc:
        logger.warning("symbol attribution LLM call failed: %s", exc)
        return None

    if tracker is not None:
        try:
            tracker.record(
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except Exception:
            pass

    text = response.content[0].text if response.content else ""
    return _parse_response(text)


def _ask_ollama(
    *,
    feature: Feature,
    relevant_symbols: dict[str, list[str]],
    model: str,
    host: str,
) -> dict[str, list[dict[str, Any]]] | None:
    """Local-free alternative to _ask_llm using Ollama HTTP API."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — required for Ollama symbol attribution")
        return None

    prompt = _build_prompt(feature, relevant_symbols)
    if not prompt:
        return None

    try:
        response = httpx.post(
            f"{host.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0},
            },
            timeout=180.0,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Ollama symbol attribution call failed: %s", exc)
        return None

    try:
        text = response.json().get("response", "")
    except ValueError:
        return None

    return _parse_response(text)


def _build_prompt(
    feature: Feature,
    relevant_symbols: dict[str, list[str]],
) -> str:
    """Build the per-feature attribution prompt."""
    symbol_count = sum(len(s) for s in relevant_symbols.values())
    if symbol_count == 0:
        return ""

    # Truncate to avoid blowing the token budget on huge features.
    shown_symbols: dict[str, list[str]] = {}
    remaining = _MAX_SYMBOLS_IN_PROMPT
    for path, syms in relevant_symbols.items():
        if remaining <= 0:
            break
        take = min(len(syms), remaining)
        shown_symbols[path] = syms[:take]
        remaining -= take

    symbols_block = "\n".join(
        f"  {path}: {', '.join(syms)}"
        for path, syms in shown_symbols.items()
    )

    flows_block = "\n".join(
        f"  - {fl.name}" + (f" — {fl.description}" if fl.description else "")
        for fl in feature.flows
    )

    return f"""You are attributing functions/classes to user-facing flows within a feature.

Feature: {feature.name}
Description: {feature.description or "(none)"}

Flows:
{flows_block}

Available symbols (functions, classes, constants) per file:
{symbols_block}

TASK: For each flow, list which symbols from the files above belong to that flow.

Rules:
  - A symbol can belong to MULTIPLE flows if it's shared across journeys (e.g. a validator used by both checkout and refund).
  - Only attribute symbols that clearly relate to the flow by name or by what the function does (inferred from the name).
  - Skip symbols that are utility helpers unlikely to be touched when the flow changes.
  - Do NOT include types, interfaces, or enums (they are handled separately).

Output JSON only, no prose. Format:
{{
  "flow-name-1": [
    {{"file": "path/to/file.ts", "symbols": ["funcA", "funcB"]}},
    {{"file": "path/to/other.ts", "symbols": ["funcC"]}}
  ],
  "flow-name-2": [
    {{"file": "path/to/file.ts", "symbols": ["funcD"]}}
  ]
}}
"""


def _parse_response(text: str) -> dict[str, list[dict[str, Any]]] | None:
    """Parse the JSON block from the LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    if text.endswith("```"):
        text = text[: -3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("symbol attribution returned invalid JSON")
        return None

    if not isinstance(data, dict):
        return None
    return data


def _apply_mapping_to_flows(
    feature: Feature,
    file_symbols: dict[str, FileSymbols],
    mapping: dict[str, list[dict[str, Any]]],
) -> None:
    """Turn the LLM mapping into SymbolAttribution objects on each flow."""
    flows_by_name = {fl.name: fl for fl in feature.flows}

    for flow_name, attributions in mapping.items():
        flow = flows_by_name.get(flow_name)
        if not flow or not isinstance(attributions, list):
            continue

        for entry in attributions:
            if not isinstance(entry, dict):
                continue
            file_path = entry.get("file")
            symbols = entry.get("symbols")
            if not file_path or not isinstance(symbols, list):
                continue

            fs = file_symbols.get(file_path)
            if fs is None:
                continue

            # Only keep symbols that were actually in the input
            # (prevents hallucinations).
            valid = [
                s for s in symbols
                if isinstance(s, str) and s in fs.flow_symbols
            ]
            if not valid:
                continue

            ranges = _resolve_ranges(valid, fs.symbol_lines)
            roles = {s: fs.symbol_roles.get(s, "helper") for s in valid}
            flow.symbol_attributions.append(
                SymbolAttribution(
                    file_path=file_path,
                    symbols=valid,
                    line_ranges=ranges,
                    attributed_lines=_sum_lines(ranges),
                    total_file_lines=fs.total_file_lines,
                    roles=roles,
                ),
            )


def _resolve_ranges(
    symbols: list[str],
    symbol_lines: dict[str, tuple[int, int]],
) -> list[tuple[int, int]]:
    """Look up line ranges for the given symbols and merge overlaps.

    Ranges are sorted by start line and adjacent/overlapping spans are
    merged so a flow attributing five contiguous methods reports one
    range, not five — this keeps GitHub deeplinks tidy.
    """
    raw = [symbol_lines[s] for s in symbols if s in symbol_lines]
    if not raw:
        return []
    raw.sort(key=lambda r: r[0])
    merged: list[tuple[int, int]] = [raw[0]]
    for start, end in raw[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _sum_lines(ranges: list[tuple[int, int]]) -> int:
    """Total lines covered by a (already-merged) list of ranges."""
    return sum(end - start + 1 for start, end in ranges)
