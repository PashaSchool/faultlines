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


# Synthetic buckets — never auto-locked, never aliased away.
# Inlined here (rather than imported from llm.dedup) so this
# module stays free of an analyzer → llm dependency.
_PROTECTED_NAMES: frozenset[str] = frozenset({
    "documentation",
    "shared-infra",
    "examples",
})


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

    ``auto_aliases`` is engine-managed (Improvement #4): after each
    successful scan we write the names of stable detected features
    into a separate top-level ``auto_aliases:`` section. Subsequent
    runs lock those names against Sprint 5 critique renaming so the
    same feature keeps the same label scan-to-scan.
    """

    features: list[FeatureRule] = field(default_factory=list)
    skip_features: list[str] = field(default_factory=list)
    force_merges: list[ForcedMerge] = field(default_factory=list)
    auto_aliases: list[FeatureRule] = field(default_factory=list)
    source_path: str = ""

    @property
    def is_empty(self) -> bool:
        return not (
            self.features or self.skip_features
            or self.force_merges or self.auto_aliases
        )

    def all_canonical_names(self) -> frozenset[str]:
        """Names from BOTH user-managed features and auto_aliases.

        Used by the pipeline to lock Sprint 5 critique against
        renaming any stable name.
        """
        names: set[str] = {r.canonical for r in self.features}
        names.update(r.canonical for r in self.auto_aliases)
        return frozenset(names)


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
    auto = _parse_features(data.get("auto_aliases"), path)

    cfg = RepoConfig(
        features=features,
        skip_features=skip,
        force_merges=forced,
        auto_aliases=auto,
        source_path=str(path),
    )
    logger.info(
        "repo_config: loaded %s — %d user features, %d auto_aliases, "
        "%d skips, %d force-merges",
        path, len(features), len(auto), len(skip), len(forced),
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


# ── Auto-save (Improvement #4) ────────────────────────────────────────


# Features below this size are too noisy to lock in — they're often
# small ancillary helpers that legitimately get renamed across runs
# as the engine learns more about the codebase. Locking them too
# eagerly would freeze in early-iteration mistakes.
_AUTO_LOCK_MIN_FILES = 10


def auto_save_canonicals(
    repo_root: Path | str,
    detected: dict[str, list[str]],
    descriptions: dict[str, str] | None = None,
    *,
    write_if_missing: bool = False,
) -> int:
    """Write stable canonical feature names back to ``.faultline.yaml``.

    For each detected feature that is:
      - not synthetic (``documentation`` / ``shared-infra`` /
        ``examples``)
      - at or above :data:`_AUTO_LOCK_MIN_FILES` files
      - not already declared in user-managed ``features:``

    ...append a stub under ``auto_aliases:`` with the engine's
    description. Subsequent scans see the name in
    :meth:`RepoConfig.all_canonical_names`, lock it against Sprint
    5 critique, and the label sticks.

    Behaviour:
      - When the repo already has a ``.faultline.yaml`` the function
        merges into it (preserving all user content). Only the
        ``auto_aliases:`` block is rewritten.
      - When no config exists, no file is created unless
        ``write_if_missing=True`` (default False — we don't want to
        litter every scanned repo with a config file the user didn't
        ask for).
      - Returns the count of NEW canonicals written. ``0`` is the
        common steady-state once the repo has stabilised.

    Failure modes (write errors, malformed existing YAML) are logged
    and the function returns ``0`` — auto-save is best-effort, never
    blocks the scan.
    """
    if not detected:
        return 0

    repo_root = Path(repo_root)
    config_path = find_repo_config(repo_root)
    if config_path is None:
        if not write_if_missing:
            return 0
        config_path = repo_root / ".faultline.yaml"

    descriptions = descriptions or {}

    # Load existing content so we preserve user edits.
    try:
        existing = (
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_path.exists() else None
        ) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "repo_config: cannot read %s for auto-save (%s) — skipping",
            config_path, exc,
        )
        return 0
    if not isinstance(existing, dict):
        logger.warning(
            "repo_config: %s top-level is not a mapping — skipping auto-save",
            config_path,
        )
        return 0

    # Names the user has explicitly declared — never auto-write
    # over them.
    user_names: set[str] = set()
    raw_user = existing.get("features") or {}
    if isinstance(raw_user, dict):
        user_names.update(str(k).strip() for k in raw_user.keys())

    # Existing auto_aliases — preserve descriptions where the engine
    # didn't supply a fresh one this run.
    raw_auto = existing.get("auto_aliases") or {}
    prev_auto: dict[str, dict] = {}
    if isinstance(raw_auto, dict):
        prev_auto = {
            str(k).strip(): (v if isinstance(v, dict) else {})
            for k, v in raw_auto.items()
        }

    # Build append-only auto_aliases: start from the previous file
    # so canonical names accumulate across runs (a name detected in
    # run N stays locked for run N+1 even if that run's Sonnet call
    # invents a slightly different name). Drift across consecutive
    # scans converges to zero this way: every new name added becomes
    # the lock for the next scan via the token-match + parent-collapse
    # passes in ``apply_repo_config``.
    new_auto: dict[str, dict] = {}
    new_count = 0
    # 1) Preserve every previously-saved auto entry not now claimed
    #    by user features. Description is kept; variants too.
    for name, prev in prev_auto.items():
        if name in user_names:
            continue
        entry: dict[str, object] = {}
        if isinstance(prev, dict):
            if prev.get("description"):
                entry["description"] = prev["description"]
            prev_variants = prev.get("variants")
            if isinstance(prev_variants, list) and prev_variants:
                entry["variants"] = list(prev_variants)
        new_auto[name] = entry
    # 2) Add fresh names from this run that aren't already locked.
    for name, files in detected.items():
        if name in _PROTECTED_NAMES:
            continue
        if name in user_names:
            continue
        if len(files) < _AUTO_LOCK_MIN_FILES:
            continue
        if name in new_auto:
            # Already preserved — refresh description if we have one.
            if descriptions.get(name):
                new_auto[name]["description"] = descriptions[name]
            continue
        prev = prev_auto.get(name) or {}
        desc = descriptions.get(name) or prev.get("description") or ""
        entry = {}
        if desc:
            entry["description"] = desc
        prev_variants = prev.get("variants")
        if isinstance(prev_variants, list) and prev_variants:
            entry["variants"] = list(prev_variants)
        new_auto[name] = entry
        new_count += 1

    # Rebuild the file: keep all user keys verbatim; overwrite only
    # ``auto_aliases``.
    output = dict(existing)
    if new_auto:
        output["auto_aliases"] = new_auto
    elif "auto_aliases" in output:
        del output["auto_aliases"]

    try:
        text = (
            "# Faultlines repo config. User-managed sections are "
            "preserved across\n"
            "# scans; the ``auto_aliases:`` block at the bottom is "
            "engine-managed —\n"
            "# Faultlines rewrites it after each scan to lock canonical "
            "feature\n"
            "# names against Sprint 5 critique renaming. Safe to delete "
            "or move\n"
            "# entries from auto_aliases up into ``features:`` to take "
            "ownership.\n\n"
        )
        text += yaml.safe_dump(
            output, sort_keys=False, allow_unicode=True, indent=2,
        )
        config_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "repo_config: cannot write %s for auto-save (%s) — skipping",
            config_path, exc,
        )
        return 0

    logger.info(
        "repo_config: auto-save wrote %d new canonical(s) to %s "
        "(total auto_aliases now: %d)",
        new_count, config_path, len(new_auto),
    )
    return new_count


# ── Apply ────────────────────────────────────────────────────────────


def _token_match_canonical(
    variant: str, canonicals: list[str],
) -> str | None:
    """Return the canonical name whose tokens match ``variant``, or None.

    Uses ``benchmark.metrics._name_matches`` (bidirectional), which
    treats ``email`` ≡ ``email-notifications`` and
    ``prisma-database`` ≡ ``prisma`` while keeping
    ``user-authentication`` distinct from ``team-management``.

    Sub-features (anything containing ``/``) are excluded — we only
    rename top-level parents to top-level canonicals here.
    """
    if "/" in variant:
        return None
    try:
        from faultline.benchmark.metrics import _name_tokens
    except Exception:  # noqa: BLE001
        return None
    vt = _name_tokens(variant)
    if not vt:
        return None
    for canonical in canonicals:
        if "/" in canonical:
            continue
        ct = _name_tokens(canonical)
        if not ct:
            continue
        # Token-set equal, OR one is a non-empty subset of the other
        # (covers ``prisma`` ↔ ``prisma-database``, ``email`` ↔
        # ``email-notifications``, and word-order swaps). Single-
        # token canonicals are allowed here — user has explicitly
        # opted in by listing them in ``.faultline.yaml``, so the
        # over-matching risk that gates ``_name_matches`` doesn't
        # apply.
        if vt == ct or vt.issubset(ct) or ct.issubset(vt):
            return canonical
    return None


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
    # User-managed ``features`` rules apply first (they may rename
    # current variants). ``auto_aliases`` follow with the same
    # apply logic but lower priority — if both define the same
    # canonical name the user version wins (already deduped at
    # apply time because the variant is gone after the first pass).
    for rule in list(config.features) + list(config.auto_aliases):
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

    # ── 2.4. parent-collapse for canonical top-level names ─────────
    # ``deep_scan_workspace`` runs a per-package Sonnet call that
    # may split a package into 2-8 sub-features (``prisma/schema``,
    # ``prisma/seed``, ``prisma/client``). If the user has the bare
    # parent (``prisma``) in ``.faultline.yaml`` they've explicitly
    # said "treat this as one feature" — collapse the children back
    # into the parent before sub_decompose / critique even see them.
    canonicals_set = config.all_canonical_names()
    for canonical in canonicals_set:
        if "/" in canonical:
            continue
        children = [
            n for n in result.features
            if n.startswith(canonical + "/")
        ]
        if not children:
            continue
        merged: dict[str, None] = {
            f: None for f in result.features.get(canonical, [])
        }
        for child in children:
            for f in result.features.pop(child, []):
                merged[f] = None
            result.descriptions.pop(child, None)
            if child in result.flows:
                result.flows.setdefault(canonical, []).extend(
                    result.flows.pop(child),
                )
            if child in result.flow_descriptions:
                target = result.flow_descriptions.setdefault(canonical, {})
                for fn, desc in result.flow_descriptions.pop(child).items():
                    target.setdefault(fn, desc)
        result.features[canonical] = sorted(merged)
        logger.info(
            "repo_config: collapsed %d children into %r",
            len(children), canonical,
        )

    # ── 2.5. token-set fallback aliasing ───────────────────────────
    # Catches drift the explicit variant lists miss. If a detected
    # feature name shares a strong token-set match with a canonical
    # name (e.g. ``prisma-database`` ↔ ``prisma``,
    # ``email`` ↔ ``email-notifications``) and the canonical doesn't
    # already exist in the result, rename to the canonical. This
    # closes the residual drift coming from the initial Sonnet call,
    # which doesn't see ``.faultline.yaml`` and invents trivially
    # different names each run.
    canonicals = list(config.all_canonical_names())
    for variant in list(result.features.keys()):
        if variant in canonicals:
            continue
        match = _token_match_canonical(variant, canonicals)
        if not match or match in result.features:
            continue
        result.features[match] = result.features.pop(variant)
        if variant in result.descriptions:
            result.descriptions[match] = result.descriptions.pop(variant)
        if variant in result.flows:
            result.flows[match] = result.flows.pop(variant)
        if variant in result.flow_descriptions:
            result.flow_descriptions[match] = result.flow_descriptions.pop(variant)
        logger.info(
            "repo_config: token-match alias %r → %r", variant, match,
        )

    # ── 3. skip_features ───────────────────────────────────────────
    for skip in config.skip_features:
        if skip in result.features:
            del result.features[skip]
            result.descriptions.pop(skip, None)
            result.flows.pop(skip, None)
            result.flow_descriptions.pop(skip, None)
            logger.info("repo_config: skipped feature %r", skip)

    return result
