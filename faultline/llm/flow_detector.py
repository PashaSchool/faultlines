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

# Features with more files than this get a deep second-pass flow detection
_DEEP_FLOW_THRESHOLD = 20

# Max source lines to include per entry-point file in deep pass
_DEEP_MAX_LINES_PER_FILE = 80

# Max entry-point files to read in deep pass
_DEEP_MAX_ENTRY_POINTS = 8

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

A flow is a concrete user journey — something a real user DOES in the product. \
It starts with a user intent and ends with a visible outcome.

Good flow examples:
- "login-flow": user enters credentials → validates → creates session → redirects
- "create-alert-flow": user configures alert rules → saves → gets confirmation
- "csv-export-flow": user selects data → triggers export → downloads file
- "filter-entities-flow": user sets filters → query updates → table re-renders
- "delete-integration-flow": user clicks delete → confirms → integration removed

A flow is NOT a technical layer, data pattern, or code structure.

## Rules

1. Flow names: lowercase, hyphen-separated, end in "-flow". Max 3 words. \
   The name must describe a USER ACTION — start with a verb or describe what happens. \
   Examples: "login-flow", "create-rule-flow", "search-entities-flow", \
   "edit-profile-flow", "export-report-flow", "delete-integration-flow".

2. CRITICAL — flow names must describe WHAT THE USER DOES, not how code is organized. \
   Think: "If I were a user of this product, what action am I performing?"

3. Only create flows when you see genuinely distinct user journeys. If the \
   feature naturally has one primary user journey, return one flow — do not \
   force artificial splits. Scale with feature size: \
   - Small features (≤10 files): 1–3 flows \
   - Medium features (11–30 files): 2–6 flows \
   - Large features (31+ files): 4–10 flows \
   More than 12 means you are splitting too finely — merge related sub-actions.

4. It is OK to skip files that have no user-facing behavior: README files, \
   type definitions (.d.ts), config files, mock stubs, pure constants. \
   Do NOT force every file into a flow — only assign files that participate \
   in a real user journey. It is BETTER to leave files unassigned than to \
   create a fake flow for them.

5. Shared utilities, types, and constants that serve multiple flows should \
   be placed in the most closely related flow OR omitted entirely.

6. Do NOT invent files. Only use the exact paths provided.

7. If an <e2e-anchors> section is provided, treat those flow names as the \
   authoritative names for this feature's flows — they were written by humans \
   describing real user journeys. Prefer these names over invented ones and \
   assign files accordingly.

8. If a <co-changes> section is provided, files that change together frequently \
   are a strong signal they belong to the same flow.

9. If a file has API routes (GET/POST/PUT/DELETE), it is an entry point to a flow. \
   Group other files that serve the same route prefix into the same flow.

## Anti-patterns — NEVER use these flow name patterns

BAD — technical layer names (describe code structure, not user actions):
  "data-flow", "data-display-flow", "data-fetch-flow", "data-loading-flow",
  "view-flow", "display-flow", "render-flow", "layout-flow", "page-layout-flow",
  "navigation-flow", "routing-flow", "state-management-flow",
  "error-handling-flow", "validation-flow", "utility-flow",
  "component-flow", "hook-flow", "service-flow", "store-flow",
  "configuration-flow", "initialization-flow", "setup-flow"

GOOD — rewrite technical names as user actions:
  "search-entities-flow" instead of "data-display-flow"
  "configure-alerts-flow" instead of "configuration-flow"
  "browse-inventory-flow" instead of "navigation-flow"
  "retry-failed-request-flow" instead of "error-handling-flow"

If you cannot identify a genuine user action for a group of files, \
DO NOT create a flow for them — leave those files unassigned.
"""

_FLOW_USER_PROMPT = """\
Feature: {feature_name}
{e2e_context}
Files and their signatures:
{signatures_text}
{extra_context}
Identify the distinct user-facing flows within the "{feature_name}" feature.
Assign files to flows based on which user journey they serve. Skip files with no user-facing behavior (READMEs, .d.ts, configs, mocks).
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

    # For very large features, chunk files and detect flows per chunk
    _MAX_FILES_PER_FLOW_CALL = 150
    if len(feature_files) > _MAX_FILES_PER_FLOW_CALL:
        phase1 = _chunked_flow_detection(
            client, feature_name, feature_files, signatures, e2e_anchors, commits,
            chunk_size=_MAX_FILES_PER_FLOW_CALL,
        )
    else:
        phase1 = _call_flow_detection(client, feature_name, feature_files, signatures, e2e_anchors, commits)

    # Phase 2: deep detection for large features — reads entry-point source code
    if len(feature_files) >= _DEEP_FLOW_THRESHOLD:
        deep_flows = _deep_flow_detection(client, feature_name, feature_files, signatures, phase1)
        if deep_flows:
            logger.info("Deep flow detection found %d additional flows for '%s'", len(deep_flows), feature_name)
            phase1.extend(deep_flows)

    return phase1


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


_DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def detect_flows_deepseek(
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    api_key: str | None = None,
    model: str = _DEFAULT_DEEPSEEK_MODEL,
    base_url: str | None = None,
    e2e_anchors: dict[str, list[str]] | None = None,
    commits: list | None = None,
) -> list[_FlowFileMapping]:
    """Detects user-facing flows using DeepSeek API. Returns [] on failure."""
    if not feature_files:
        return []

    from faultline.llm.deepseek_client import call_deepseek_parsed

    signatures_text = _build_signatures_text(feature_files, signatures)
    e2e_context = _format_e2e_anchors(e2e_anchors or {})
    extra_context = _build_flow_extra_context(feature_files, signatures, commits)
    prompt = _FLOW_USER_PROMPT.format(
        feature_name=feature_name,
        e2e_context=e2e_context,
        signatures_text=signatures_text,
        extra_context=extra_context,
    )

    parsed = call_deepseek_parsed(
        _FLOW_SYSTEM_PROMPT, prompt, _FlowDetectionResponse,
        api_key=api_key, model=model, base_url=base_url, max_tokens=4096,
    )
    if not parsed:
        return []
    return _filter_valid_files(parsed.flows, set(feature_files))


def _chunked_flow_detection(
    client: anthropic.Anthropic,
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    e2e_anchors: dict[str, list[str]] | None = None,
    commits: list | None = None,
    chunk_size: int = 150,
) -> list[_FlowFileMapping]:
    """Splits large features into directory-based chunks for flow detection.

    Groups files by their top-level subdirectory, then runs flow detection
    per chunk. Merges results, deduplicating flow names.
    """
    from collections import defaultdict

    # Group files by first meaningful directory
    dir_groups: dict[str, list[str]] = defaultdict(list)
    for f in feature_files:
        parts = Path(f).parts
        group = parts[0] if parts else "__root__"
        # Go deeper if first dir is too generic
        for i, part in enumerate(parts[:-1]):
            if part.lower() not in ("src", "app", "apps", "packages", "lib", "components"):
                group = part
                break
        dir_groups[group].append(f)

    # Merge small groups into chunks of ~chunk_size
    chunks: list[list[str]] = []
    current_chunk: list[str] = []
    for group_files in sorted(dir_groups.values(), key=len, reverse=True):
        if len(current_chunk) + len(group_files) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
        current_chunk.extend(group_files)
    if current_chunk:
        chunks.append(current_chunk)

    logger.info("Chunked flow detection for '%s': %d files → %d chunks", feature_name, len(feature_files), len(chunks))

    all_flows: list[_FlowFileMapping] = []
    seen_names: set[str] = set()

    for i, chunk_files in enumerate(chunks):
        chunk_result = _call_flow_detection(
            client, feature_name, chunk_files, signatures, e2e_anchors, commits,
        )
        for flow in chunk_result:
            if flow.flow_name not in seen_names:
                all_flows.append(flow)
                seen_names.add(flow.flow_name)
            else:
                # Merge files into existing flow with same name
                for existing in all_flows:
                    if existing.flow_name == flow.flow_name:
                        existing_set = set(existing.files)
                        new_files = [f for f in flow.files if f not in existing_set]
                        existing.files.extend(new_files)
                        break
        logger.info("Chunk %d/%d: %d files → %d flows", i + 1, len(chunks), len(chunk_files), len(chunk_result))

    return all_flows


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
                max_tokens=4096,
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
    Includes symbol names with their kinds (function, class, const, etc.)
    to help the LLM understand what each file contains.
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

        # Show symbol names with kinds for richer context
        if sig.symbol_ranges:
            symbols_by_kind: dict[str, list[str]] = {}
            for sr in sig.symbol_ranges[:12]:
                symbols_by_kind.setdefault(sr.kind, []).append(sr.name)
            symbol_parts = []
            for kind, names in symbols_by_kind.items():
                symbol_parts.append(f"{kind}: {', '.join(names[:6])}")
            parts.append(" | ".join(symbol_parts))
        elif sig.exports:
            parts.append(f"exports: {', '.join(sig.exports[:8])}")

        if sig.routes:
            parts.append(f"routes: {', '.join(sig.routes[:5])}")

        if sig.imports:
            if large_feature:
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


# File extensions that indicate a flow name was derived from a filename
_FILE_EXTENSIONS = frozenset({
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".py", ".rb", ".go", ".rs", ".java", ".kt",
    ".css", ".scss", ".less", ".html", ".vue", ".svelte",
    ".json", ".yaml", ".yml", ".toml", ".md", ".txt",
    ".d.ts", ".d.mts",
})


def _is_filename_flow(flow_name: str) -> bool:
    """Detects if a flow name was derived from a filename rather than a user journey.

    Examples of bad names: "default-config.ts-flow", "index.ts-flow", "readme.md-flow"
    Examples of good names: "login-flow", "checkout-flow", "password-reset-flow"
    """
    # Strip "-flow" suffix for analysis
    name = flow_name
    if name.endswith("-flow"):
        name = name[:-5]

    # Check if name contains a file extension
    for ext in _FILE_EXTENSIONS:
        if ext in name:
            return True

    # Check if name matches common non-semantic patterns
    # e.g. "index-flow", "main-flow" when derived from index.ts
    return False


# Technical/generic flow names that describe code structure, not user actions
_TECHNICAL_FLOW_NAMES = frozenset({
    "data-flow", "data-display-flow", "data-fetch-flow", "data-loading-flow",
    "data-management-flow", "data-processing-flow", "data-sync-flow",
    "view-flow", "display-flow", "render-flow", "layout-flow", "page-layout-flow",
    "navigation-flow", "routing-flow", "state-management-flow", "state-flow",
    "error-handling-flow", "validation-flow", "utility-flow", "helper-flow",
    "component-flow", "hook-flow", "service-flow", "store-flow",
    "configuration-flow", "initialization-flow", "setup-flow", "config-flow",
    "loading-flow", "fetching-flow", "caching-flow", "storage-flow",
    "styling-flow", "theming-flow", "animation-flow",
})

# Partial patterns — if a flow name contains ONLY these words (minus "flow"), it's technical
_TECHNICAL_FLOW_WORDS = frozenset({
    "data", "display", "view", "render", "layout", "page",
    "navigation", "routing", "state", "management",
    "error", "handling", "validation", "utility", "helper",
    "component", "hook", "service", "store", "config",
    "loading", "fetching", "caching", "storage",
    "styling", "theming", "animation", "setup", "init",
})


def _is_technical_flow(flow_name: str) -> bool:
    """Detects if a flow name describes a technical layer rather than a user action."""
    normalized = flow_name.lower().strip()

    # Exact match
    if normalized in _TECHNICAL_FLOW_NAMES:
        return True

    # Check if ALL words (excluding "flow") are technical
    words = set(normalized.replace("-flow", "").split("-"))
    words.discard("")
    if words and words.issubset(_TECHNICAL_FLOW_WORDS):
        return True

    return False


def _filter_valid_files(
    flows: list[_FlowFileMapping],
    allowed_files: set[str],
) -> list[_FlowFileMapping]:
    """Removes hallucinated files, filename-based flows, deduplicates, and assigns unassigned files."""
    assigned: set[str] = set()
    result: list[_FlowFileMapping] = []

    # Pass 1: validate files and reject bad flow names
    for flow in flows:
        if _is_filename_flow(flow.flow_name):
            logger.info("Rejected filename-based flow: '%s'", flow.flow_name)
            continue
        if _is_technical_flow(flow.flow_name):
            logger.info("Rejected technical flow name: '%s'", flow.flow_name)
            continue
        valid = [f for f in flow.files if f in allowed_files and f not in assigned]
        if valid:
            assigned.update(valid)
            result.append(_FlowFileMapping(flow_name=flow.flow_name, files=valid))

    # Pass 2: assign unassigned files to closest flow by directory overlap ONLY
    # Files without a directory match are left unassigned — better than inflating a random flow
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
            # else: leave unassigned — no fake flow, no inflating largest

        for i, files in extra.items():
            result[i] = _FlowFileMapping(
                flow_name=result[i].flow_name,
                files=result[i].files + files,
            )

    return _cap_flows(result)


# Maximum flows per feature — merge the smallest flows into their nearest neighbor
_MAX_FLOWS_PER_FEATURE = 12


def _cap_flows(flows: list[_FlowFileMapping]) -> list[_FlowFileMapping]:
    """Merges excess flows into the most related flow by directory overlap."""
    if len(flows) <= _MAX_FLOWS_PER_FEATURE:
        return flows

    # Sort by file count descending — keep the largest, merge the smallest
    flows_sorted = sorted(flows, key=lambda f: len(f.files), reverse=True)
    keep = list(flows_sorted[:_MAX_FLOWS_PER_FEATURE])
    merge = flows_sorted[_MAX_FLOWS_PER_FEATURE:]

    # Build dir→flow index from kept flows
    flow_dirs: dict[str, int] = {}
    for i, flow in enumerate(keep):
        for f in flow.files:
            parent = str(Path(f).parent)
            flow_dirs.setdefault(parent, i)

    for orphan in merge:
        target_idx = _find_merge_target(orphan, keep, flow_dirs)
        keep[target_idx] = _FlowFileMapping(
            flow_name=keep[target_idx].flow_name,
            files=keep[target_idx].files + orphan.files,
        )
        logger.info("Merged flow '%s' into '%s' (cap exceeded)", orphan.flow_name, keep[target_idx].flow_name)

    return keep


def _find_merge_target(
    orphan: _FlowFileMapping,
    targets: list[_FlowFileMapping],
    flow_dirs: dict[str, int],
) -> int:
    """Finds the best merge target for an orphan flow by directory overlap."""
    scores: dict[int, int] = defaultdict(int)
    for f in orphan.files:
        parent = str(Path(f).parent)
        if parent in flow_dirs:
            scores[flow_dirs[parent]] += 1

    if scores:
        return max(scores, key=scores.get)

    # No directory match — merge into the largest flow
    return max(range(len(targets)), key=lambda i: len(targets[i].files))


# ── Deep flow detection (phase 2) ──────────────────────────────────────
#
# For large features, phase 1 only sees file names and export signatures.
# Phase 2 reads the actual source of entry-point files (route handlers,
# main page components, index barrels) to discover flows hidden inside them.

_DEEP_FLOW_SYSTEM_PROMPT = """\
You are analyzing the SOURCE CODE of key entry-point files in a feature to \
discover user-facing flows that file-level analysis missed.

You already have a list of flows detected in phase 1. Your job is to find \
ADDITIONAL flows by reading the actual code — function bodies, route handlers, \
component render trees, event handlers, state transitions.

## Rules

1. Only return NEW flows not already covered by the existing ones.
2. Flow names: lowercase, hyphen-separated, end in "-flow". Must describe a USER ACTION.
3. For each new flow, list the files that participate (from the full file list provided).
4. Do NOT rename or modify existing flows — only ADD new ones.
5. If you find no new flows, return an empty list.
6. Do NOT create flows for technical patterns (data fetching, error handling, etc.) — \
   only for distinct user journeys visible in the code.
"""

_DEEP_FLOW_USER_PROMPT = """\
Feature: {feature_name}

Existing flows from phase 1:
{existing_flows}

Source code of key entry-point files:
{source_snippets}

All files in this feature:
{all_files}

Find additional user-facing flows that phase 1 missed. \
Return only NEW flows not already covered above. If none found, return empty list.\
"""


def _identify_entry_points(
    feature_files: list[str],
    signatures: dict[str, FileSignature],
) -> list[str]:
    """Identifies the most important entry-point files to read in deep pass.

    Priority order:
    1. Files with API routes (GET/POST/PUT/DELETE handlers)
    2. Page components (page.tsx, index.tsx in view directories)
    3. Files with many exports (barrel files, main components)
    4. Files with the most imports from other feature files
    """
    scored: list[tuple[str, int]] = []

    for path in feature_files:
        sig = signatures.get(path)
        if not sig or not sig.source:
            continue

        score = 0
        # Route handlers are highest priority
        if sig.routes:
            score += 50 + len(sig.routes) * 10

        # Page/index files in view directories
        name_lower = Path(path).name.lower()
        if name_lower in ("page.tsx", "page.ts", "page.jsx"):
            score += 40
        elif name_lower.startswith("index."):
            if any(d in path.lower() for d in ("views/", "pages/", "screens/")):
                score += 30

        # Files with many exports (rich components)
        if sig.exports:
            score += min(len(sig.exports) * 3, 20)

        # Files importing many feature siblings
        if sig.imports:
            internal = [i for i in sig.imports if i.startswith(".")]
            score += min(len(internal) * 2, 15)

        if score > 0:
            scored.append((path, score))

    scored.sort(key=lambda x: -x[1])
    return [path for path, _ in scored[:_DEEP_MAX_ENTRY_POINTS]]


def _build_source_snippets(
    entry_points: list[str],
    signatures: dict[str, FileSignature],
) -> str:
    """Reads source code of entry-point files, truncated to key sections."""
    snippets: list[str] = []

    for path in entry_points:
        sig = signatures.get(path)
        if not sig or not sig.source:
            continue

        lines = sig.source.split("\n")
        # Take first N lines — usually contains imports, component definition, route handlers
        truncated = lines[:_DEEP_MAX_LINES_PER_FILE]
        source = "\n".join(truncated)
        if len(lines) > _DEEP_MAX_LINES_PER_FILE:
            source += f"\n// ... ({len(lines) - _DEEP_MAX_LINES_PER_FILE} more lines)"

        snippets.append(f"--- {path} ---\n{source}")

    return "\n\n".join(snippets)


def _deep_flow_detection(
    client: anthropic.Anthropic,
    feature_name: str,
    feature_files: list[str],
    signatures: dict[str, FileSignature],
    existing_flows: list[_FlowFileMapping],
) -> list[_FlowFileMapping]:
    """Phase 2: reads entry-point source code to find additional flows.

    Only runs for features with >_DEEP_FLOW_THRESHOLD files.
    Returns additional flows to append (not replacing phase 1 results).
    """
    entry_points = _identify_entry_points(feature_files, signatures)
    if not entry_points:
        return []

    source_snippets = _build_source_snippets(entry_points, signatures)
    if not source_snippets:
        return []

    existing_flows_text = "\n".join(
        f"  {fl.flow_name}: {', '.join(fl.files[:5])}"
        + (f" ... +{len(fl.files)-5} more" if len(fl.files) > 5 else "")
        for fl in existing_flows
    ) if existing_flows else "  (none detected in phase 1)"

    all_files_text = "\n".join(f"  {f}" for f in feature_files)

    prompt = _DEEP_FLOW_USER_PROMPT.format(
        feature_name=feature_name,
        existing_flows=existing_flows_text,
        source_snippets=source_snippets,
        all_files=all_files_text,
    )

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.parse(
                model=_MODEL,
                max_tokens=2048,
                temperature=0,
                system=_DEEP_FLOW_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=_FlowDetectionResponse,
            )
            # Only keep flows that don't duplicate existing names
            existing_names = {fl.flow_name for fl in existing_flows}
            new_flows = [
                fl for fl in response.parsed_output.flows
                if fl.flow_name not in existing_names
            ]
            return _filter_valid_files(new_flows, set(feature_files))
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("Deep flow detection failed (attempt %d/%d): %s", attempt + 1, _MAX_RETRIES, e)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
        except Exception:
            return []
    return []
