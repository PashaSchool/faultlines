"""Discover canonical-feature suggestions from repo signals.

Improvement #8 — bootstrap a ``.faultline.yaml`` skeleton from
existing repo conventions instead of asking users to author one
from scratch.

Two signal sources today:

  - **Workspace package names** (``package.json``,
    ``pyproject.toml``, ``Cargo.toml``). Reuses
    :func:`faultline.analyzer.workspace.detect_workspace` so we
    don't duplicate parsing.
  - **CODEOWNERS** (root, ``.github/`` or ``docs/``). Paths grouped
    by their owning team are a strong signal that those paths
    form one product feature.

The discoverer never modifies repo files. Output is a list of
:class:`FeatureRule` suggestions; the CLI ``suggest-config``
subcommand prints them in YAML form for the user to review and
optionally drop into ``.faultline.yaml``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .repo_config import FeatureRule, _PROTECTED_NAMES


logger = logging.getLogger(__name__)


_CODEOWNERS_LOCATIONS: tuple[str, ...] = (
    "CODEOWNERS",
    ".github/CODEOWNERS",
    "docs/CODEOWNERS",
)

# CODEOWNERS lines are "<path-glob>  @owner1 @owner2 ..." with
# # comments. Owners are GitHub handles (``@user``) or team handles
# (``@org/team``). We match the first owner per line.
_RE_CODEOWNERS_LINE = re.compile(
    r"^\s*(?P<path>[^\s#]+)\s+(?P<owners>@[\S]+(?:\s+@[\S]+)*)\s*(?:#.*)?$"
)


def _read_codeowners(repo_root: Path) -> str | None:
    for rel in _CODEOWNERS_LOCATIONS:
        p = repo_root / rel
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
    return None


def _team_to_feature_name(owner: str) -> str:
    """Strip ``@org/`` prefix and convert ``-team`` suffix to clean
    canonical form.

    >>> _team_to_feature_name("@documenso/billing-team")
    'billing'
    >>> _team_to_feature_name("@billing-team")
    'billing'
    >>> _team_to_feature_name("@auth")
    'auth'
    """
    name = owner.lstrip("@")
    if "/" in name:
        name = name.split("/", 1)[1]
    # Drop common team-suffix words
    for suffix in ("-team", "-engineering", "-eng", "-squad", "-pod"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def discover_from_codeowners(repo_root: Path | str) -> list[FeatureRule]:
    """Parse CODEOWNERS into per-team :class:`FeatureRule` suggestions.

    Each unique team produces one FeatureRule whose ``canonical`` is
    the cleaned team name and whose ``variants`` is empty (the
    engine matches via path prefix overlap downstream).

    Returns empty list when no CODEOWNERS file is found or all
    lines are comments.
    """
    text = _read_codeowners(Path(repo_root))
    if not text:
        return []

    by_team: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _RE_CODEOWNERS_LINE.match(line)
        if not m:
            continue
        path = m.group("path")
        first_owner = m.group("owners").split()[0]
        team = _team_to_feature_name(first_owner)
        if not team or team in _PROTECTED_NAMES:
            continue
        by_team.setdefault(team, []).append(path)

    out: list[FeatureRule] = []
    for team, paths in sorted(by_team.items()):
        out.append(FeatureRule(
            canonical=team,
            description=f"Suggested from CODEOWNERS — paths owned by team.",
            variants=tuple(sorted(set(paths))),
        ))
    return out


def discover_from_workspace(
    repo_root: Path | str,
    source_files: list[str],
) -> list[FeatureRule]:
    """Use :mod:`faultline.analyzer.workspace` to extract package
    names as canonical-feature suggestions.

    Returns one :class:`FeatureRule` per workspace package whose
    ``canonical`` is the package's short name (last slash segment
    of ``package.json`` ``name``). ``variants`` is empty — the
    engine's workspace detection already maps files into packages,
    so the alias rule is a backup for cases where Sprint 5 critique
    renames the feature away from the package name.
    """
    from .workspace import detect_workspace

    try:
        info = detect_workspace(str(repo_root), source_files)
    except Exception:  # noqa: BLE001 — defensive on weird configs
        return []
    if not info or not info.detected:
        return []

    out: list[FeatureRule] = []
    seen: set[str] = set()
    for pkg in info.packages:
        name = pkg.name.strip()
        if not name or name in _PROTECTED_NAMES or name in seen:
            continue
        seen.add(name)
        out.append(FeatureRule(
            canonical=name,
            description=f"Suggested from workspace package at {pkg.path}",
            variants=(),
        ))
    return out


def discover_aliases(
    repo_root: Path | str,
    source_files: list[str],
) -> list[FeatureRule]:
    """Combine all signal sources and return deduped suggestions.

    Same canonical from multiple sources is merged: workspace
    description wins (it's more concrete than a CODEOWNERS team
    label), variants unioned across sources.
    """
    by_name: dict[str, FeatureRule] = {}

    for rule in (
        list(discover_from_workspace(repo_root, source_files))
        + list(discover_from_codeowners(repo_root))
    ):
        existing = by_name.get(rule.canonical)
        if existing is None:
            by_name[rule.canonical] = rule
            continue
        # Merge variants
        merged_variants = tuple(sorted(set(existing.variants) | set(rule.variants)))
        # Prefer existing description (workspace beats CODEOWNERS
        # because it's first in our concat above).
        by_name[rule.canonical] = FeatureRule(
            canonical=existing.canonical,
            description=existing.description or rule.description,
            variants=merged_variants,
        )

    return list(by_name.values())
