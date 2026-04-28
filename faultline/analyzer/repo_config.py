"""Repo-level configuration loader.

Reads an optional ``.faultline.yaml`` (or ``.faultline.yml``) from the
analyzed repo's root and applies its rules to the post-dedup feature
map. Lets users encode their **own** product taxonomy without
touching faultline source code.

Schema (every key optional)::

    # .faultline.yaml — repo-level Faultlines config
    features:
      billing-and-subscriptions:
        description: Stripe-based billing and subscription management.
        variants:
          - lib/billing-and-subscriptions
          - ee/stripe-billing
          - trpc/enterprise-billing-and-identity

      embedded-signing:
        variants:
          - remix/embedded-signing-authoring
          - lib/embedded-signing

    skip_features:
      - tsconfig
      - tailwind-config

    force_merges:
      - into: design-system
        from:
          - ui/primitive-components
          - ui-primitives
        description: Reusable UI primitives shared across the app.

Behaviour:

  - Each ``features`` entry: any detected feature whose name matches
    one of ``variants`` is renamed to the canonical key. Optionally
    sets / overwrites the description.
  - ``skip_features``: detected features matching one of these names
    are dropped entirely.
  - ``force_merges``: behaves like a manual Sprint 2 dedup op; files
    are unioned into ``into``.

Failure mode: malformed config raises a clear error early. Missing
file is silent — config is opt-in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import yaml
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for repo_config; install with `pip install pyyaml`."
    ) from _exc

if TYPE_CHECKING:  # pragma: no cover
    from faultline.llm.sonnet_scanner import DeepScanResult


logger = logging.getLogger(__name__)


# Filenames searched, in priority order, at the repo root.
_CONFIG_FILENAMES: tuple[str, ...] = (
    ".faultline.yaml",
    ".faultline.yml",
    "faultline.config.yaml",
    "faultline.config.yml",
)


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class FeatureRule:
    """One canonical-feature entry from the repo config."""

    canonical: str
    description: str = ""
    variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForcedMerge:
    """One ``force_merges`` entry."""

    into: str
    sources: tuple[str, ...] = ()
    description: str = ""


@dataclass
class RepoConfig:
    """In-memory view of a repo's ``.faultline.yaml``.

    Every field defaults to empty so a partially-populated config is
    still useful. ``source_path`` records where the config was loaded
    from, for log lines.
    """

    features: list[FeatureRule] = field(default_factory=list)
    skip_features: list[str] = field(default_factory=list)
    force_merges: list[ForcedMerge] = field(default_factory=list)
    source_path: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.features or self.skip_features or self.force_merges)


# ── Loader ───────────────────────────────────────────────────────────


def find_repo_config(repo_root: Path | str) -> Path | None:
    """Return the first existing config filename under ``repo_root``."""
    root = Path(repo_root)
    for name in _CONFIG_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def load_repo_config(repo_root: Path | str) -> RepoConfig | None:
    """Load the repo's ``.faultline.yaml`` if present.

    Returns ``None`` when no config file exists. Returns a populated
    :class:`RepoConfig` on success. Raises ``ValueError`` on malformed
    YAML so the user gets an early signal instead of silent drift.
    """
    path = find_repo_config(repo_root)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}: cannot read ({exc})") from exc

    data = yaml.safe_load(text)
    if data is None:
        return RepoConfig(source_path=str(path))
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level must be a mapping, got {type(data).__name__}"
        )

    features = _parse_features(data.get("features"), path)
    skip = _parse_skip(data.get("skip_features"), path)
    forced = _parse_force_merges(data.get("force_merges"), path)

    cfg = RepoConfig(
        features=features,
        skip_features=skip,
        force_merges=forced,
        source_path=str(path),
    )
    logger.info(
        "repo_config: loaded %s — %d feature rules, %d skips, %d force-merges",
        path, len(features), len(skip), len(forced),
    )
    return cfg


def _parse_features(raw, source: Path) -> list[FeatureRule]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: 'features' must be a mapping")

    out: list[FeatureRule] = []
    seen: set[str] = set()
    for canonical, body in raw.items():
        canonical = str(canonical).strip()
        if not canonical:
            raise ValueError(f"{source}: empty canonical feature name")
        if canonical in seen:
            raise ValueError(f"{source}: duplicate canonical feature {canonical!r}")
        seen.add(canonical)
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise ValueError(
                f"{source}: feature {canonical!r} body must be a mapping"
            )
        variants_raw = body.get("variants") or []
        if not isinstance(variants_raw, list):
            raise ValueError(
                f"{source}: variants for {canonical!r} must be a list"
            )
        variants = tuple(
            v for v in (str(x).strip() for x in variants_raw) if v
        )
        out.append(FeatureRule(
            canonical=canonical,
            description=str(body.get("description") or "").strip(),
            variants=variants,
        ))
    return out


def _parse_skip(raw, source: Path) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{source}: 'skip_features' must be a list")
    return [s for s in (str(x).strip() for x in raw) if s]


def _parse_force_merges(raw, source: Path) -> list[ForcedMerge]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{source}: 'force_merges' must be a list")
    out: list[ForcedMerge] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: force_merges entry must be a mapping")
        into = str(entry.get("into") or "").strip()
        if not into:
            raise ValueError(f"{source}: force_merge missing 'into'")
        srcs_raw = entry.get("from") or []
        if not isinstance(srcs_raw, list):
            raise ValueError(f"{source}: force_merge 'from' must be a list")
        srcs = tuple(s for s in (str(x).strip() for x in srcs_raw) if s)
        if len(srcs) < 1:
            raise ValueError(
                f"{source}: force_merge into {into!r} needs at least 1 source"
            )
        out.append(ForcedMerge(
            into=into,
            sources=srcs,
            description=str(entry.get("description") or "").strip(),
        ))
    return out


# ── Apply ────────────────────────────────────────────────────────────


def apply_repo_config(
    result: "DeepScanResult",
    config: RepoConfig | None,
) -> "DeepScanResult":
    """Apply the user config to a DeepScanResult in place AND return it.

    Order of operations:

      1. ``force_merges`` — union sources into ``into``, drop sources.
      2. ``features`` (canonical aliasing) — rename detected variants
         to their canonical name, set/override description.
      3. ``skip_features`` — drop detected features whose name appears
         in the skip list.

    Each stage logs its effect. Missing sources / detected names are
    skipped silently — config drift over time is expected and we
    don't want stale entries to break scans.
    """
    if not result or not config or config.is_empty:
        return result

    # ── 1. force_merges ────────────────────────────────────────────
    for fm in config.force_merges:
        # Only sources that actually exist in the current map.
        live = [s for s in fm.sources if s in result.features]
        if not live:
            logger.info(
                "repo_config: force_merge into %r — no live sources, skipped",
                fm.into,
            )
            continue
        # Build union files dict to dedup while preserving order.
        unioned: dict[str, None] = {}
        for src in live:
            for f in result.features.get(src, []):
                unioned[f] = None
        if fm.into in result.features:
            for f in result.features[fm.into]:
                unioned[f] = None

        # Pop sources (skip popping ``into`` even if listed there).
        for src in live:
            if src == fm.into:
                continue
            result.features.pop(src, None)
            result.descriptions.pop(src, None)
            result.flows.pop(src, None)
            result.flow_descriptions.pop(src, None)

        result.features[fm.into] = sorted(unioned)
        if fm.description:
            result.descriptions[fm.into] = fm.description
        logger.info(
            "repo_config: force_merge → %r (%d files; sources=%s)",
            fm.into, len(result.features[fm.into]), live,
        )

    # ── 2. canonical aliasing ──────────────────────────────────────
    for rule in config.features:
        for variant in rule.variants:
            if variant not in result.features or variant == rule.canonical:
                continue
            # If the canonical already exists, union files into it.
            if rule.canonical in result.features:
                merged: dict[str, None] = {
                    f: None for f in result.features[rule.canonical]
                }
                for f in result.features.pop(variant):
                    merged[f] = None
                result.features[rule.canonical] = sorted(merged)
            else:
                result.features[rule.canonical] = result.features.pop(variant)
            # Carry description / flows / flow_descs to canonical key.
            if variant in result.descriptions:
                desc = result.descriptions.pop(variant)
                result.descriptions.setdefault(rule.canonical, desc)
            if variant in result.flows:
                result.flows.setdefault(
                    rule.canonical, [],
                ).extend(result.flows.pop(variant))
                # Dedup flow names while keeping order
                seen: set[str] = set()
                deduped = []
                for fn in result.flows[rule.canonical]:
                    if fn not in seen:
                        seen.add(fn)
                        deduped.append(fn)
                result.flows[rule.canonical] = deduped
            if variant in result.flow_descriptions:
                target = result.flow_descriptions.setdefault(rule.canonical, {})
                for fn, desc in result.flow_descriptions.pop(variant).items():
                    target.setdefault(fn, desc)
            logger.info(
                "repo_config: alias %r → %r", variant, rule.canonical,
            )
        # Override description if explicitly set.
        if rule.description and rule.canonical in result.features:
            result.descriptions[rule.canonical] = rule.description

    # ── 3. skip_features ───────────────────────────────────────────
    for skip in config.skip_features:
        if skip in result.features:
            del result.features[skip]
            result.descriptions.pop(skip, None)
            result.flows.pop(skip, None)
            result.flow_descriptions.pop(skip, None)
            logger.info("repo_config: skipped feature %r", skip)

    return result
