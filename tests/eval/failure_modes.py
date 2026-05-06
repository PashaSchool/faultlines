"""Sprint 16 Day 4 — failure mode taxonomy + LLM classifier.

A coverage number alone is shallow — knowing 87% means 13% of expected
features were missed but says nothing about *how* they were missed.
This module classifies each miss into a named failure mode so we get
a breakdown like:

    13% miss rate split into:
      5× OVER_CLUSTERING   (auth + signin merged into i18n)
      4× UNDER_CLUSTERING  (one feature split into multiple)
      2× GENERIC_NAMING    (returned ``utils`` instead of business name)
      2× HALLUCINATED      (no files match the expected feature)

That breakdown is the actual interview material — concrete names of
concrete failure modes a real production system exhibits.

Public surface
==============

    FailureMode               — Enum of 6 categories.
    classify_miss(miss, ...)  — single Haiku call returns a FailureMode.
    classify_all(misses, ...) — batched, returns dict[expected → mode].
    summarise(classified)     — Counter of mode → count.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 2_048
DEFAULT_CACHE_DIR = Path.home() / ".faultline" / "eval-cache"
_CACHE_VERSION = 1


class FailureMode(enum.Enum):
    """Six categories cover ~95% of attribution misses we've seen
    across S12-S15. ``UNCATEGORISED`` is the safety valve for cases
    where the classifier itself fails or returns low-confidence."""

    OVER_CLUSTERING = "Two distinct expected features merged into one detected feature"
    UNDER_CLUSTERING = "One expected feature was split across multiple detected ones"
    GENERIC_NAMING = "Detected name is path-shaped (utils, components, types) rather than business-shaped"
    HALLUCINATED = "Detected feature has no files matching the expected concept"
    MISSING_KEY = "Major README feature was not detected at all"
    LANGUAGE_BIAS = "Detection works on TS/Python but degrades on Go/Rust/etc."
    UNCATEGORISED = "Classifier could not assign a confident mode"


_VALID_MODE_NAMES: frozenset[str] = frozenset(m.name for m in FailureMode)


@dataclass
class ClassifiedMiss:
    """One miss + its assigned failure mode + reasoning."""

    expected: str
    detected: str | None
    mode: FailureMode
    reasoning: str = ""


# ── Anthropic protocol ─────────────────────────────────────────────────


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...  # pragma: no cover


# ── System prompt ──────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You classify why a feature-detection tool missed an expected feature.

For each MISS you receive: the expected feature name, the detected
feature it was best-mapped to (or NONE), the full list of detected
features in this repo, and a sample of file paths.

Pick the single failure mode that best explains the miss:

  OVER_CLUSTERING   — Two or more distinct expected features got
                      merged under one detected name. Usually the
                      detected name is broad ("workflow-app") and
                      swallows tighter expected concepts.

  UNDER_CLUSTERING  — One expected feature was fragmented across
                      multiple detected names. Detected list will
                      have 2-3 narrow names that together equal the
                      expected concept.

  GENERIC_NAMING    — A matching feature exists but was given a
                      path-shaped name like ``utils``, ``components``,
                      ``types``, ``core`` instead of the business
                      domain name.

  HALLUCINATED      — Detected feature exists in the list but its
                      file paths don't actually correspond to the
                      expected concept.

  MISSING_KEY       — Expected feature has no remote candidate in
                      the detected list at all.

  LANGUAGE_BIAS     — Detection clearly degraded for non-TS/Python
                      languages — Go / Rust / Elixir paths got
                      lumped into shared-infra or generic buckets.

  UNCATEGORISED     — The miss doesn't fit any mode confidently.

OUTPUT (JSON only, no prose, no fences):
{
  "mode": "OVER_CLUSTERING" | "UNDER_CLUSTERING" | "GENERIC_NAMING" |
          "HALLUCINATED" | "MISSING_KEY" | "LANGUAGE_BIAS" | "UNCATEGORISED",
  "reasoning": "<one short sentence>"
}

Be specific. "Generic name" without saying which name = UNCATEGORISED.
"""


# ── Cache helpers ──────────────────────────────────────────────────────


def _cache_key(expected: str, detected: str | None,
               detected_features: list[str]) -> str:
    payload = json.dumps(
        [expected, detected or "", sorted(detected_features)],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / "failure-modes.json"


def _load_cache(cache_dir: Path | None, key: str) -> dict | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != _CACHE_VERSION:
        return None
    return data.get("entries", {}).get(key)


def _save_cache(cache_dir: Path | None, key: str, payload: dict) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir)
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = {}
        if existing.get("version") != _CACHE_VERSION:
            existing = {"version": _CACHE_VERSION, "entries": {}}
        existing.setdefault("entries", {})[key] = payload
        path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    except OSError as exc:
        logger.warning("failure_modes: cache save failed (%s)", exc)


# ── Helpers ────────────────────────────────────────────────────────────


def _parse_response(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _coerce_mode(payload: dict) -> tuple[FailureMode, str]:
    raw = (payload.get("mode") or "").strip().upper()
    reasoning = (payload.get("reasoning") or "").strip()
    if raw not in _VALID_MODE_NAMES:
        return FailureMode.UNCATEGORISED, reasoning or "judge returned invalid mode"
    return FailureMode[raw], reasoning


# ── Classification ─────────────────────────────────────────────────────


def classify_miss(
    expected: str,
    detected: str | None,
    detected_features: list[str],
    sample_paths: list[str] | None = None,
    *,
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> ClassifiedMiss:
    """Classify a single miss. Returns ``UNCATEGORISED`` on any error
    so the caller never crashes the pipeline."""
    key = _cache_key(expected, detected, detected_features)
    cached = _load_cache(cache_dir, key)
    if cached is not None:
        mode_name = cached.get("mode", "UNCATEGORISED")
        mode = (
            FailureMode[mode_name]
            if mode_name in _VALID_MODE_NAMES
            else FailureMode.UNCATEGORISED
        )
        return ClassifiedMiss(
            expected=expected, detected=detected, mode=mode,
            reasoning=cached.get("reasoning", ""),
        )

    if client is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return ClassifiedMiss(expected, detected, FailureMode.UNCATEGORISED)
        try:
            from anthropic import Anthropic
        except ImportError:
            return ClassifiedMiss(expected, detected, FailureMode.UNCATEGORISED)
        client = Anthropic(api_key=api_key)

    user_msg = json.dumps({
        "expected": expected,
        "detected": detected or "NONE",
        "all_detected": sorted(detected_features),
        "sample_paths": (sample_paths or [])[:20],
    }, indent=2, ensure_ascii=False)

    try:
        response = client.messages.create(
            model=model,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failure_modes: API call failed (%s)", exc)
        return ClassifiedMiss(expected, detected, FailureMode.UNCATEGORISED)

    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")

    parsed = _parse_response(text)
    if not parsed:
        return ClassifiedMiss(expected, detected, FailureMode.UNCATEGORISED)
    mode, reasoning = _coerce_mode(parsed)

    _save_cache(cache_dir, key, {"mode": mode.name, "reasoning": reasoning})

    return ClassifiedMiss(
        expected=expected, detected=detected, mode=mode, reasoning=reasoning,
    )


def classify_all(
    misses: Iterable[tuple[str, str | None]],
    detected_features: list[str],
    sample_paths: list[str] | None = None,
    *,
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> list[ClassifiedMiss]:
    """Classify a batch of misses. One Haiku call per miss; cache hit
    on repeats. Returns the list in input order."""
    out: list[ClassifiedMiss] = []
    for expected, detected in misses:
        out.append(classify_miss(
            expected=expected, detected=detected,
            detected_features=detected_features,
            sample_paths=sample_paths,
            client=client, api_key=api_key, model=model, cache_dir=cache_dir,
        ))
    return out


def summarise(classified: list[ClassifiedMiss]) -> Counter:
    """Mode → count breakdown for PR-comment rendering."""
    return Counter(m.mode for m in classified)
