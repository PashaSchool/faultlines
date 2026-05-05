"""Sprint 8 Day 2 — resolve consumer features for aggregator files.

Given a set of files that belong to a classified ``shared-aggregator``
feature (DTOs, shared UI primitives, schemas, etc.), figure out which
PRODUCT features actually use those files. The Day 4 redistribution
step uses this map to attach each aggregator file as a
``shared_participant`` of every consuming feature.

Primary signal: the symbol graph (Sprint 7 infrastructure) already
records which files import which, in both directions. Walking
``SymbolGraph.reverse[file]`` gives every importer of a target file
in the same repo. We map each importer back to its owning feature.

Secondary signal: when the symbol graph has no reverse edges for an
aggregator file (e.g. dynamic imports, unresolved namespace re-
exports, untyped JS, code in languages we don't yet parse), we fall
back to a filename-token heuristic — split the file basename into
semantic tokens and look for features whose owned files reference
those tokens by name.

Last resort: an aggregator file with NO resolvable consumer at all
gets reported with an empty list. Day 4 folds those orphans into
``shared-infra`` rather than silently losing them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faultline.analyzer.symbol_graph import SymbolGraph
    from faultline.llm.sonnet_scanner import DeepScanResult

logger = logging.getLogger(__name__)


# Aggregator-file basenames often look like
#   "embed-login-body.dto.ts"
#   "BillingPlanCard.tsx"
#   "user-profile.types.ts"
# The semantic content lives in the leading hyphen/camel-cased name;
# trailing ``.dto`` / ``.types`` / ``.schema`` etc. are noise.
_NOISE_SUFFIXES: frozenset[str] = frozenset({
    "dto", "dtos", "type", "types", "schema", "schemas", "interface",
    "interfaces", "model", "models", "contract", "contracts",
    "request", "response", "config", "constants", "props", "spec",
    "test", "stories", "fixture", "fixtures", "mock", "mocks",
    "index", "main", "init", "common", "util", "utils", "helper",
    "helpers",
})

# Tokens that carry no business meaning even when they appear in a
# filename — strip them before matching against feature paths.
_NOISE_TOKENS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "for", "with", "from", "and", "or",
    "is", "in", "on", "by",
})


def _normalize_token(token: str) -> str:
    """Lowercase and collapse internal punctuation to plain alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _filename_tokens(file_path: str) -> set[str]:
    """Extract semantic tokens from an aggregator file's basename.

    ``packages/api-types/dto/auth/embed-login-body.dto.ts`` →
        {"embed", "login", "body", "auth"}
    ``packages/shared-ui/src/BillingPlanCard.tsx`` →
        {"billing", "plan", "card"}
    """
    name = Path(file_path).stem  # drops the trailing extension only
    # Strip trailing noise suffix (``.dto`` from ``embed-login.dto``)
    if "." in name:
        head, tail = name.rsplit(".", 1)
        if tail.lower() in _NOISE_SUFFIXES:
            name = head
    # CamelCase → space-separated; then split on hyphens / underscores
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    raw_parts = re.split(r"[\s\-_/]+", spaced)
    parent = Path(file_path).parent.name  # one parent folder may carry domain
    raw_parts.append(parent)
    out: set[str] = set()
    for part in raw_parts:
        norm = _normalize_token(part)
        if not norm or len(norm) < 3:
            continue
        if norm in _NOISE_SUFFIXES or norm in _NOISE_TOKENS:
            continue
        out.add(norm)
    return out


def _build_file_to_feature(result: "DeepScanResult") -> dict[str, str]:
    """Map every owned file path to its current feature name.

    Files that belong to the aggregator we're resolving are NOT
    excluded here — callers do their own filtering when needed
    (we exclude aggregator features in ``find_consumers`` so a
    file's own bucket isn't reported as its own consumer).
    """
    out: dict[str, str] = {}
    for feat_name, paths in result.features.items():
        for p in paths:
            out[p] = feat_name
    return out


def _filename_fallback_consumers(
    file_path: str,
    result: "DeepScanResult",
    excluded_feature: str,
) -> list[str]:
    """Fallback when the symbol graph has no reverse edges.

    Splits the aggregator file's basename into semantic tokens and
    returns every feature whose name contains a substring match.
    A weaker signal than imports — used only when imports are silent.
    """
    tokens = _filename_tokens(file_path)
    if not tokens:
        return []
    consumers: list[str] = []
    for feat_name in result.features:
        if feat_name == excluded_feature:
            continue
        norm_name = _normalize_token(feat_name)
        if not norm_name:
            continue
        if any(t in norm_name or norm_name in t for t in tokens if len(t) >= 4):
            consumers.append(feat_name)
    return consumers


def find_consumers(
    file_paths: list[str],
    *,
    aggregator_feature: str,
    result: "DeepScanResult",
    symbol_graph: "SymbolGraph | None",
) -> dict[str, list[str]]:
    """For each file in ``file_paths``, return the list of features
    whose OWNED files import it.

    Uses ``symbol_graph.reverse`` when available (the strong signal),
    falls back to filename tokens (the weak signal). A file with no
    resolvable consumer at all maps to an empty list — Day 4 folds
    those into ``shared-infra``.

    ``aggregator_feature`` is the name of the feature that currently
    owns the input files (so we don't report the file's own bucket
    as its own consumer).
    """
    file_to_feature = _build_file_to_feature(result)

    out: dict[str, list[str]] = {}
    fallback_used = 0
    no_signal = 0

    for path in file_paths:
        seen: set[str] = set()

        if symbol_graph is not None:
            for edge in symbol_graph.callers_of(path):
                importer = edge.target_file
                feat = file_to_feature.get(importer)
                if feat and feat != aggregator_feature:
                    seen.add(feat)

        if not seen:
            # Symbol graph silent — try filename heuristic
            fallback = _filename_fallback_consumers(
                path, result, aggregator_feature,
            )
            if fallback:
                fallback_used += 1
                seen.update(fallback)
            else:
                no_signal += 1

        out[path] = sorted(seen)

    if fallback_used or no_signal:
        logger.info(
            "aggregator_consumers(%s): %d via imports, %d via filename "
            "fallback, %d unresolved",
            aggregator_feature,
            len(file_paths) - fallback_used - no_signal,
            fallback_used,
            no_signal,
        )

    return out
