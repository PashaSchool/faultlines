"""Five metrics for the detection benchmark harness.

All metrics return floats in [0.0, 1.0] (except :func:`generic_name_rate`,
which is also [0,1] but interpreted as "lower is better" rather than
"higher is better"). Empty inputs return 0.0 for the recall / precision
family and 0.0 for the rate (no generics if you have no features).
"""

from __future__ import annotations

import re
from typing import Iterable

from .loader import ExpectedAttribution, ExpectedFeature, ExpectedFlow


# ── Constants ────────────────────────────────────────────────────────


GENERIC_NAME_BLOCKLIST: frozenset[str] = frozenset({
    "lib", "utils", "shared", "core", "general", "misc", "common",
    "helpers", "types", "ui", "api", "main", "base",
})


def _normalize(name: str) -> str:
    """Lowercase + strip a name to compare across casing / whitespace."""
    return name.strip().lower()


def _flow_normalize(name: str) -> str:
    """Flow names compared after dehyphen + lowercase, like the docs say."""
    return _normalize(name).replace("-", " ").replace("_", " ")


# Tokens we consider "stop words" — they don't contribute meaning to
# token-set comparison so two names that only differ by these still
# match.
_NAME_STOP_TOKENS: frozenset[str] = frozenset({
    "and", "or", "of", "the", "a", "an",
    "management", "service", "module", "system", "feature",
    "user",  # "user-authentication" vs "authentication"
})


def _name_tokens(name: str) -> frozenset[str]:
    """Token-set view of a feature name.

    Splits on ``/``, ``-``, ``_`` and drops stop tokens. Used by
    :func:`_names_equivalent` to consider names like
    ``team-and-organisation-management`` and
    ``organisation-and-team-management`` equivalent (word-order swap)
    and ``user-authentication`` ≈ ``authentication`` (stop word).
    """
    raw = re.split(r"[/\-_]+", _normalize(name))
    return frozenset(t for t in raw if t and t not in _NAME_STOP_TOKENS)


def _name_matches(expected: str, detected: str) -> bool:
    """True when ``detected`` represents the same domain as ``expected``.

    Four rules, applied in order:

      1. Exact name match (``billing`` == ``billing``).
      2. Prefix match: detected starts with ``expected + "/"`` or
         vice versa. Lets ``document-signing`` count as found when
         the engine produces ``document-signing/recipient-signing``
         (Sprint 3 sub-decomposition adds granularity below the
         canonical name).
      3. Token-set equality after stop-word filter. Handles word-
         order swaps ("team-and-organisation-management" ≡
         "organisation-and-team-management") and pure-stop-word
         differences ("user-authentication" ≡ "authentication").
      4. Token subset (after stop-word filter): expected ⊆ detected,
         and expected has ≥2 meaningful tokens. Lets
         ``envelope-management`` match ``envelope-document-management``
         — detected is MORE specific than the canonical, which is fine
         for recall. The 2-token floor prevents single-token expected
         names from matching unrelated detected names that happen to
         contain that one token (e.g. expected ``email`` would
         otherwise match unrelated ``ee/organisation-email-domains``).
         For single-token canonicals the user must add explicit
         aliases in ``.faultline.yaml``.
    """
    e = _normalize(expected)
    d = _normalize(detected)
    if e == d:
        return True
    if d.startswith(e + "/") or e.startswith(d + "/"):
        return True
    et, dt = _name_tokens(expected), _name_tokens(detected)
    if not et or not dt:
        return False
    if et == dt:
        return True
    if len(et) >= 2 and et.issubset(dt):
        return True
    return False


def _detected_names(detected: dict | Iterable[str]) -> set[str]:
    """Extract feature names from either a feature_map dict or any
    iterable of names."""
    if isinstance(detected, dict):
        return {_normalize(n) for n in detected.keys()}
    return {_normalize(n) for n in detected}


# ── 1. Feature recall ────────────────────────────────────────────────


def feature_recall(
    expected: list[ExpectedFeature],
    detected: dict | Iterable[str],
) -> float:
    """Fraction of expected features matched by some detected feature.

    Match rules (see :func:`_name_matches`): exact, prefix
    (``document-signing`` matches ``document-signing/X`` from
    sub-decomposition), or token-set equality after stop-word filter.

    >>> from faultline.benchmark.loader import ExpectedFeature
    >>> e = [ExpectedFeature(name="auth", aliases=("api/auth",)),
    ...      ExpectedFeature(name="billing")]
    >>> feature_recall(e, {"auth": [], "x": []})
    0.5
    >>> feature_recall(e, {"api/auth": [], "billing": []})
    1.0
    >>> feature_recall(e, {"auth/sign-in": [], "billing": []})  # prefix
    1.0
    """
    if not expected:
        return 0.0
    have_names = (
        list(detected.keys()) if isinstance(detected, dict)
        else list(detected)
    )
    matched = 0
    for feat in expected:
        for n in feat.all_names:
            if any(_name_matches(n, d) for d in have_names):
                matched += 1
                break
    return matched / len(expected)


# ── 2. Feature precision ─────────────────────────────────────────────


def feature_precision(
    expected: list[ExpectedFeature],
    detected: dict | Iterable[str],
    *,
    excluded: Iterable[str] = ("documentation", "shared-infra", "examples"),
) -> float:
    """Fraction of detected features that map to some expected feature.

    Match rules from :func:`_name_matches` apply. Synthetic buckets in
    ``excluded`` are skipped from both numerator and denominator.

    >>> from faultline.benchmark.loader import ExpectedFeature
    >>> e = [ExpectedFeature(name="auth")]
    >>> feature_precision(e, {"auth": [], "phantom": []})
    0.5
    """
    have_raw = (
        list(detected.keys()) if isinstance(detected, dict)
        else list(detected)
    )
    excluded_norm = {_normalize(x) for x in excluded}
    have = [n for n in have_raw if _normalize(n) not in excluded_norm]
    if not have:
        return 0.0

    expected_names: list[str] = []
    for feat in expected:
        expected_names.extend(feat.all_names)

    matched = sum(
        1 for d in have
        if any(_name_matches(e, d) for e in expected_names)
    )
    return matched / len(have)


# ── 3. Flow recall ───────────────────────────────────────────────────


def flow_recall(
    expected: list[ExpectedFlow],
    detected_flows: dict[str, list[str]] | Iterable[str],
) -> float:
    """Fraction of expected flows that appear (after normalize) in
    detected flows.

    ``detected_flows`` may be the full ``DeepScanResult.flows`` dict
    (we flatten across features) or a flat iterable of flow names.

    >>> from faultline.benchmark.loader import ExpectedFlow
    >>> e = [ExpectedFlow(name="create-document"),
    ...      ExpectedFlow(name="cancel-subscription")]
    >>> flow_recall(e, {"f1": ["create-document"], "f2": ["other"]})
    0.5
    """
    if not expected:
        return 0.0
    if isinstance(detected_flows, dict):
        flat: list[str] = []
        for flows in detected_flows.values():
            flat.extend(flows or [])
    else:
        flat = list(detected_flows)
    have = {_flow_normalize(f) for f in flat}
    matched = sum(1 for ef in expected if _flow_normalize(ef.name) in have)
    return matched / len(expected)


# ── 4. File attribution accuracy ─────────────────────────────────────


def attribution_accuracy(
    samples: list[ExpectedAttribution],
    detected: dict[str, list[str]],
    *,
    expected_features: list[ExpectedFeature] | None = None,
) -> float:
    """For each sampled file, check that it ended up in the expected
    feature (or one of its aliases).

    ``expected_features`` is optional but recommended: if provided,
    we honour the alias graph so a file landing in an alias of the
    expected name still counts.

    >>> from faultline.benchmark.loader import (
    ...     ExpectedAttribution, ExpectedFeature,
    ... )
    >>> samples = [
    ...     ExpectedAttribution(path="a.ts", expected="auth"),
    ...     ExpectedAttribution(path="b.ts", expected="billing"),
    ... ]
    >>> detected = {"auth": ["a.ts"], "billing": ["b.ts"]}
    >>> attribution_accuracy(samples, detected)
    1.0
    """
    if not samples:
        return 0.0

    # Build a (path → detected_feature_name) lookup.
    where: dict[str, str] = {}
    for feat_name, paths in (detected or {}).items():
        for p in paths or []:
            # First feature wins on duplicates.
            where.setdefault(p, feat_name)

    # Build alias map: every name → its canonical expected name.
    canonical: dict[str, str] = {}
    if expected_features:
        for feat in expected_features:
            for n in feat.all_names:
                canonical[_normalize(n)] = _normalize(feat.name)

    correct = 0
    for s in samples:
        actual = where.get(s.path)
        if actual is None:
            continue
        actual_canonical = canonical.get(_normalize(actual), _normalize(actual))
        expected_canonical = canonical.get(
            _normalize(s.expected), _normalize(s.expected),
        )
        if actual_canonical == expected_canonical:
            correct += 1
    return correct / len(samples)


# ── 5. Generic-name rate ─────────────────────────────────────────────


def generic_name_rate(
    detected: dict | Iterable[str],
    *,
    excluded: Iterable[str] = ("documentation", "shared-infra", "examples"),
) -> float:
    """Fraction of detected feature names whose last path segment is
    in :data:`GENERIC_NAME_BLOCKLIST`.

    >>> generic_name_rate({"auth": [], "lib": [], "billing": []})
    0.3333333333333333
    """
    names = list(detected.keys()) if isinstance(detected, dict) else list(detected)
    excluded_norm = {_normalize(x) for x in excluded}
    names = [n for n in names if _normalize(n) not in excluded_norm]
    if not names:
        return 0.0
    bad = 0
    for n in names:
        last = _normalize(n).rsplit("/", 1)[-1].rstrip("s")
        if last in GENERIC_NAME_BLOCKLIST or last + "s" in GENERIC_NAME_BLOCKLIST:
            bad += 1
    return bad / len(names)
