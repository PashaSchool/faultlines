"""Sprint 16 Day 2 — LLM-as-judge for feature / flow eval.

Standard pattern in modern AI evals (RAGAS, OpenAI / Anthropic
evals): use a model to score predictions against ground truth via
**semantic** matching. ``http-client`` matches ``request-handler``;
``user-auth`` matches ``authentication-and-account-access``; etc.

Without semantic matching, exact-string scoring would punish every
naming variation and the harness would be uselessly noisy. With it,
naming becomes orthogonal to coverage.

Public surface
==============

    JudgeResult                     — coverage, precision, F1, matches
    judge_run(expected, detected,
              client, repo) -> JudgeResult

The function takes a Haiku client + the two name lists, returns
metrics. Cache verdicts by ``hash(expected + detected)`` so re-runs
of the same state cost nothing.

The default model is ``claude-haiku-4-5`` (fast, cheap, deterministic
with temperature=0). Override via ``model=`` for testing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 4_096
DEFAULT_CACHE_DIR = Path.home() / ".faultline" / "eval-cache"
_CACHE_VERSION = 1

# Match-quality buckets returned by the judge. ``exact`` and ``partial``
# both count toward coverage; ``none`` doesn't.
_VALID_QUALITIES: frozenset[str] = frozenset({"exact", "partial", "none"})


@dataclass
class Match:
    """One pairing of expected → detected with a quality verdict."""

    expected: str
    detected: str | None
    quality: str  # exact / partial / none

    @property
    def is_hit(self) -> bool:
        return self.quality in {"exact", "partial"}


@dataclass
class JudgeResult:
    """Output of one judge_run call."""

    coverage: float          # hits / |expected|
    precision: float         # hits / |detected| (after dedup)
    f1: float
    matches: list[Match] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)  # detected but unexpected
    cache_hit: bool = False

    def hits(self) -> int:
        return sum(1 for m in self.matches if m.is_hit)


# ── Anthropic client protocol so tests can inject a fake ──────────────


class _AnthropicLike(Protocol):
    @property
    def messages(self) -> Any: ...  # pragma: no cover


# ── System prompt ─────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You evaluate a feature-detection tool against a ground-truth list
curated by the open-source repository's maintainers.

For each EXPECTED feature, find the BEST matching DETECTED feature
or return "NONE" if no equivalent exists. Use SEMANTIC matching:

  - "http-client" matches "request-handling", "axios-core",
    "api-client". The names differ but the domain is the same.
  - "user-auth" matches "authentication-and-account-access" or
    "auth". Different framing, same surface.
  - "interceptors" matches "request-interceptors" exact;
    NOT "interceptor-chain-builder" (too specific) UNLESS that's
    literally what the repo calls the feature.

QUALITY BUCKETS:
  "exact"   — names + scope are equivalent.
  "partial" — overlapping but the detected name is broader / narrower.
  "none"    — no detected feature corresponds to this expected one.

For ambiguous cases, prefer "partial" over "exact" — over-claiming
hurts precision later. When in doubt, return "none".

OUTPUT (JSON only, no prose, no markdown fences):
{
  "matches": [
    {"expected": "<name>",
     "detected": "<best_match_or_NONE>",
     "quality": "exact" | "partial" | "none"}
  ],
  "extras": ["<detected_features_not_matched_to_anything_expected>"]
}

Cover EVERY expected entry exactly once. ``extras`` lists detected
names that didn't get matched to any expected (so precision can be
computed correctly).
"""


# ── Helpers ───────────────────────────────────────────────────────────


def _cache_key(expected: list[str], detected: list[str]) -> str:
    """Stable hash over the two name sets. Reorder-tolerant."""
    payload = json.dumps(
        [sorted(expected), sorted(detected)], ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(cache_dir: Path, repo: str) -> Path:
    safe_repo = re.sub(r"[^a-z0-9]+", "-", repo.lower()).strip("-") or "unknown"
    return cache_dir / f"judge-{safe_repo}.json"


def _load_cache(
    cache_dir: Path | None, repo: str, key: str,
) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, repo)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("version") != _CACHE_VERSION:
        return None
    return data.get("entries", {}).get(key)


def _save_cache(
    cache_dir: Path | None, repo: str, key: str, payload: dict,
) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, repo)
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
        logger.warning("judge: cache save failed (%s)", exc)


def _parse_response(text: str) -> dict[str, Any] | None:
    """Extract the JSON envelope. None on parse failure."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _coerce_match(entry: Any, expected_set: set[str]) -> Match | None:
    if not isinstance(entry, dict):
        return None
    expected = (entry.get("expected") or "").strip()
    detected_raw = entry.get("detected")
    detected = (
        detected_raw.strip() if isinstance(detected_raw, str) else None
    )
    if detected and detected.upper() == "NONE":
        detected = None
    quality = (entry.get("quality") or "").strip().lower()
    if expected not in expected_set or quality not in _VALID_QUALITIES:
        return None
    return Match(expected=expected, detected=detected, quality=quality)


def _build_metrics(
    matches: list[Match], detected: list[str],
) -> tuple[float, float, float]:
    """Coverage = hits / |expected|. Precision = hits / |detected dedup|.

    Both caps at 1.0. Returns (coverage, precision, f1).
    """
    if not matches:
        return (0.0, 0.0, 0.0)
    hits = sum(1 for m in matches if m.is_hit)
    coverage = hits / len(matches)
    n_detected = len(set(detected))
    precision = (hits / n_detected) if n_detected else 0.0
    if coverage + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * coverage * precision / (coverage + precision)
    return (coverage, precision, f1)


# ── Public entry point ────────────────────────────────────────────────


def judge_run(
    expected: Iterable[str],
    detected: Iterable[str],
    *,
    repo: str = "unknown",
    client: _AnthropicLike | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> JudgeResult:
    """Score detected features against expected via Haiku semantic match.

    Returns JudgeResult. Returns a coverage=0/precision=0 fallback when
    Haiku is unavailable (no API key, no client) so callers can still
    produce reports — just zero-scored.
    """
    expected_list = sorted({s.strip() for s in expected if s and s.strip()})
    detected_list = sorted({s.strip() for s in detected if s and s.strip()})
    if not expected_list:
        return JudgeResult(0.0, 0.0, 0.0)

    key = _cache_key(expected_list, detected_list)
    cached = _load_cache(cache_dir, repo, key)
    if cached is not None:
        matches = [
            Match(**m) for m in cached.get("matches", [])
            if isinstance(m, dict)
        ]
        extras = list(cached.get("extras", []))
        cov, prec, f1 = _build_metrics(matches, detected_list)
        return JudgeResult(cov, prec, f1, matches, extras, cache_hit=True)

    if client is None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("judge: no API key — returning zero-score fallback")
            return JudgeResult(0.0, 0.0, 0.0)
        try:
            from anthropic import Anthropic
        except ImportError:
            logger.warning("judge: anthropic package missing")
            return JudgeResult(0.0, 0.0, 0.0)
        client = Anthropic(api_key=api_key)

    user_msg = json.dumps({
        "expected": expected_list,
        "detected": detected_list,
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
        logger.warning("judge: API call failed (%s)", exc)
        return JudgeResult(0.0, 0.0, 0.0)

    text = ""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") == "text":
            text += getattr(block, "text", "")

    parsed = _parse_response(text) or {}
    raw_matches = parsed.get("matches") or []
    expected_set = set(expected_list)
    matches: list[Match] = []
    for entry in raw_matches:
        coerced = _coerce_match(entry, expected_set)
        if coerced is not None:
            matches.append(coerced)

    # Fill in any expected features the judge silently dropped.
    seen_expected = {m.expected for m in matches}
    for exp in expected_list:
        if exp not in seen_expected:
            matches.append(Match(expected=exp, detected=None, quality="none"))

    extras_raw = parsed.get("extras") or []
    extras = [s for s in extras_raw if isinstance(s, str) and s.strip()]

    cov, prec, f1 = _build_metrics(matches, detected_list)

    _save_cache(cache_dir, repo, key, {
        "matches": [
            {"expected": m.expected, "detected": m.detected, "quality": m.quality}
            for m in matches
        ],
        "extras": extras,
    })

    return JudgeResult(cov, prec, f1, matches, extras)
