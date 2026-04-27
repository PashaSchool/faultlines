"""Tool-augmented per-package feature detection (Sprint 1, Day 2).

Replacement path for ``deep_scan`` when the per-package LLM call needs
to read actual file contents instead of guessing names from paths.

Public entry point: :func:`tool_use_scan`. It runs an Anthropic
``messages.create`` loop, dispatching ``tool_use`` blocks to the
read-only tools defined in :mod:`faultline.llm.tools`, until the model
returns a final JSON answer or hits the per-package tool budget.

Output shape matches what ``deep_scan_workspace`` currently expects
from a per-package call:

    {
        "features": [
            {"name": "billing", "paths": [...], "description": "..."},
            ...
        ]
    }

Day 2 scope: the function plus mocked-client tests. Wiring into
``pipeline.run`` is Day 3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .tools import TOOL_SCHEMAS, dispatch_tool

if TYPE_CHECKING:  # pragma: no cover
    from .cost import CostTracker
    from .sonnet_scanner import DeepScanResult


logger = logging.getLogger(__name__)


DEFAULT_TOOL_BUDGET = 15
# Output token cap. The Anthropic Python SDK requires streaming for any
# request whose max_tokens implies > 10 min processing — empirically
# that threshold is ~21333 for Sonnet 4.6 without streaming. We sit
# below that with margin. Note: very large packages (>800 files) can
# still truncate the final JSON because every path lands in the output;
# Sprint 3 (sub-decomposition) is the real fix for that.
DEFAULT_MAX_TOKENS = 16_384
DEFAULT_MODEL = "claude-sonnet-4-6"


# ── Prompts ────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a senior engineer mapping a code package into business-readable
features. You have read-only tools to inspect the codebase: open files,
list directories, search for patterns, view commit messages.

GOAL
Group the supplied source files into 1-8 cohesive features. Each feature
must have a name a product manager would recognize ("billing",
"document-signing", "team-invitations") — not a tech-stack noun ("lib",
"utils", "shared", "general", "core", "api", "ui").

WORKFLOW
1. Look at the file list and the package name.
2. For files whose purpose is ambiguous from the path, USE TOOLS:
   - read_file_head to see imports and top-level declarations
   - grep_pattern to trace concepts across the package
   - get_file_commits to see what work the file is associated with
3. Group files into features. Every supplied file must belong to
   exactly one feature.
4. Return final JSON. Do not narrate.

NAMING RULES (HARD)
- No generic names: lib, utils, shared, general, core, misc, common,
  base, helpers, types, ui, api.
- If a cluster of files is genuinely cross-cutting infrastructure
  (logging, error formatting, config loading), name it for what it
  does ("logging", "error-formatting", "config-loading"), not "shared".
- Prefer 2-3 word business names over single tech words.
- Use the same name across runs for the same concept.

OUTPUT (final message, after tool use is done)
Respond with ONLY a JSON object, no prose:

{
  "features": [
    {
      "name": "billing",
      "paths": ["packages/web/lib/stripe.ts", "..."],
      "description": "Stripe checkout and subscription webhooks."
    }
  ]
}

CONSTRAINTS
- Cover every file in the input list exactly once.
- 1-8 features per package. If the package is small (<10 files), often 1.
- Keep descriptions to one sentence.
"""


def _build_user_prompt(package_name: str, files: list[str]) -> str:
    file_list = "\n".join(files)
    return (
        f"Package: {package_name}\n"
        f"File count: {len(files)}\n\n"
        f"Source files:\n{file_list}\n\n"
        f"Investigate using tools as needed, then return the feature JSON."
    )


# ── Anthropic client protocol (so tests can inject a fake) ────────────


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...


# ── Result types ───────────────────────────────────────────────────────


class ToolUseScanError(Exception):
    """Raised when the loop cannot produce a parseable feature list."""


# ── Helpers ────────────────────────────────────────────────────────────


def _block_type(block: Any) -> str:
    return getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else "")


def _block_text(block: Any) -> str:
    if isinstance(block, dict):
        return block.get("text", "")
    return getattr(block, "text", "") or ""


def _tool_use_fields(block: Any) -> tuple[str, str, dict]:
    """Extract (id, name, input) from a tool_use content block."""
    if isinstance(block, dict):
        return block.get("id", ""), block.get("name", ""), block.get("input", {}) or {}
    return (
        getattr(block, "id", ""),
        getattr(block, "name", ""),
        getattr(block, "input", {}) or {},
    )


def _final_text(content: list[Any]) -> str:
    """Concatenate all text blocks from an assistant response."""
    parts: list[str] = []
    for block in content or []:
        if _block_type(block) == "text":
            parts.append(_block_text(block))
    return "\n".join(p for p in parts if p)


def _parse_features_json(text: str) -> dict | None:
    """Extract the features dict from the model's final text response.

    Mirrors the resilience logic in :mod:`faultline.llm.sonnet_scanner`
    (raw JSON → fenced block → first balanced ``{...}``).
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    import re

    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _content_to_serializable(content: list[Any]) -> list[dict]:
    """Convert SDK content blocks to the dict shape required when echoing
    them back as the assistant's prior message in a follow-up call."""
    out: list[dict] = []
    for block in content or []:
        btype = _block_type(block)
        if btype == "text":
            out.append({"type": "text", "text": _block_text(block)})
        elif btype == "tool_use":
            tid, name, inp = _tool_use_fields(block)
            out.append({"type": "tool_use", "id": tid, "name": name, "input": inp})
        elif btype:
            # Unknown block type — pass through verbatim if dict, else skip.
            if isinstance(block, dict):
                out.append(block)
    return out


# ── Main entry point ───────────────────────────────────────────────────


def tool_use_scan(
    *,
    package_name: str,
    files: list[str],
    repo_root: Path,
    client: _AnthropicLike,
    model: str = DEFAULT_MODEL,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_tool_call: Any = None,
    tracker: "CostTracker | None" = None,
    cost_label: str = "tool-use-scan",
    system_prompt: str | None = None,
) -> dict | None:
    """Run a tool-augmented scan for a single package.

    Args:
        package_name: Display name shown to the LLM.
        files: Source files in this package (already bucketized to
            ``Bucket.SOURCE``). Every file must end up in exactly one
            output feature.
        repo_root: Repository root for tool dispatch (path safety).
        client: Anthropic client (or any object with the same
            ``messages.create`` shape). Tests inject a fake.
        model: Model id. Defaults to Sonnet.
        tool_budget: Max tool calls in this session (default 15).
        max_tokens: Max tokens per ``messages.create`` call.
        on_tool_call: Optional callable
            ``(name, input_dict, result_str) -> None`` invoked for each
            tool dispatch. Useful for telemetry / cost tracking.

    Returns:
        Parsed dict ``{"features": [...]}`` on success, or None if the
        loop finished without a parseable answer.
    """
    if not files:
        return {"features": []}

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _build_user_prompt(package_name, files)},
    ]
    tool_calls_made = 0

    while True:
        # If we've hit the budget, instruct the model to wrap up by
        # making the next call without tools available.
        tools_for_call = TOOL_SCHEMAS if tool_calls_made < tool_budget else []

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt or _SYSTEM_PROMPT,
            "messages": messages,
        }
        if tools_for_call:
            kwargs["tools"] = tools_for_call
        else:
            # Last-call nudge: append a system-style user message telling
            # the model the budget is exhausted.
            messages.append({
                "role": "user",
                "content": (
                    "Tool budget exhausted. Return the final feature JSON "
                    "now using only what you have already learned."
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
                    label=cost_label,
                )
                tracker.check_budget()

        # Append assistant turn so the next request includes it.
        messages.append({"role": "assistant", "content": _content_to_serializable(content)})

        if stop_reason != "tool_use":
            text = _final_text(content)
            parsed = _parse_features_json(text)
            if parsed is None:
                logger.warning(
                    "tool_use_scan(%s): could not parse final JSON from %d chars "
                    "(stop_reason=%s). Head: %r ... Tail: %r",
                    package_name, len(text), stop_reason,
                    text[:300], text[-300:],
                )
                return None
            return parsed

        # stop_reason == "tool_use": dispatch every tool_use block and
        # build a single user turn with all tool_result blocks.
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if _block_type(block) != "tool_use":
                continue
            tool_id, tool_name, tool_input = _tool_use_fields(block)
            tool_calls_made += 1
            try:
                result_str = dispatch_tool(tool_name, tool_input, repo_root)
            except Exception as exc:  # defensive — dispatch_tool itself shouldn't raise
                result_str = f"ERROR: tool dispatch crashed: {exc}"
            is_error = result_str.startswith("ERROR:")

            if on_tool_call is not None:
                try:
                    on_tool_call(tool_name, tool_input, result_str)
                except Exception:
                    logger.debug("on_tool_call hook raised; ignoring", exc_info=True)

            logger.info(
                "tool_use_scan(%s): %s(%s) → %d chars%s",
                package_name, tool_name,
                ", ".join(f"{k}={v!r}" for k, v in tool_input.items())[:120],
                len(result_str),
                " [ERROR]" if is_error else "",
            )

            block_payload: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_str,
            }
            if is_error:
                block_payload["is_error"] = True
            tool_results.append(block_payload)

        if not tool_results:
            # Defensive: model said tool_use but emitted no tool_use blocks.
            logger.warning("tool_use_scan(%s): tool_use stop with no blocks", package_name)
            return None

        messages.append({"role": "user", "content": tool_results})
        # Loop continues — next call sends tool_results, gets next response.


# ── Workspace adapter ──────────────────────────────────────────────────


def tool_use_scan_package(
    *,
    package_name: str,
    files: list[str],
    repo_root: Path,
    pkg_prefix: str = "",
    api_key: str | None = None,
    model: str | None = None,
    tracker: "CostTracker | None" = None,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
) -> "DeepScanResult | None":
    """Drop-in replacement for ``deep_scan(package_mode=True)``.

    ``files`` are package-relative (already prefix-stripped by
    ``deep_scan_workspace``). Internally we re-prefix to repo-relative
    so the LLM's tools resolve against real files; output paths get
    stripped back to package-relative so the existing workspace merge
    logic works unchanged.

    Returns a :class:`DeepScanResult` carrying features and descriptions.
    Flows are intentionally empty — Sprint 4 covers tool-augmented flow
    detection.
    """
    import os

    import anthropic

    from .sonnet_scanner import DeepScanResult

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("tool_use_scan_package(%s): no API key available", package_name)
        return None

    resolved_model = model or DEFAULT_MODEL
    client = anthropic.Anthropic(api_key=key)

    # Re-prefix to repo-relative for the LLM (so its tools can read the
    # actual files). Drop entries that are already absolute / unrelated.
    repo_rel_files = [pkg_prefix + f if pkg_prefix else f for f in files]

    try:
        parsed = tool_use_scan(
            package_name=package_name,
            files=repo_rel_files,
            repo_root=repo_root,
            client=client,
            model=resolved_model,
            tool_budget=tool_budget,
            tracker=tracker,
            cost_label=f"tool-use:{package_name}",
        )
    except anthropic.AuthenticationError as exc:
        logger.error("tool_use_scan_package(%s): auth error: %s", package_name, exc)
        return None
    except anthropic.RateLimitError as exc:
        logger.warning("tool_use_scan_package(%s): rate limit: %s", package_name, exc)
        return None

    if not parsed:
        return None

    features: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}

    for feat in parsed.get("features", []) or []:
        name = (feat.get("name") or "").strip()
        paths = feat.get("paths") or []
        if not name or not paths:
            continue
        # Strip pkg_prefix back off so the workspace merge code can
        # re-apply it uniformly. Files outside the package (LLM strayed)
        # are dropped — they would have broken the prefix invariant.
        rel_paths: list[str] = []
        for p in paths:
            if pkg_prefix and p.startswith(pkg_prefix):
                rel_paths.append(p[len(pkg_prefix):])
            elif not pkg_prefix:
                rel_paths.append(p)
            else:
                logger.debug(
                    "tool_use_scan_package(%s): dropping out-of-package path %s",
                    package_name, p,
                )
        if not rel_paths:
            continue
        features[name] = sorted(set(rel_paths))
        desc = (feat.get("description") or "").strip()
        if desc:
            descriptions[name] = desc

    if not features:
        return None

    return DeepScanResult(
        features=features,
        flows={},
        descriptions=descriptions,
        flow_descriptions={},
        cost_summary=tracker.summary() if tracker is not None else None,
    )
