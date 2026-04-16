"""Symbol-level incremental refresh.

After the metric refresh in cache/refresh.py completes, this module
can do a cheap second pass:

  1. Compare stored symbol_hashes vs the current AST extraction.
  2. Cleanup: drop attributions for symbols that no longer exist.
  3. Re-attribute: for features that gained new symbols, run one
     targeted LLM call to place the new symbols into existing flows.
  4. Body-only changes: leave attributions alone (the function still
     does "the same thing", just implemented differently).

Keeps all existing flow + symbol attribution on unchanged code,
making refresh cheap even after large PRs that touched implementation
details.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faultline.analyzer.ast_extractor import FileSignature, extract_signatures
from faultline.cache.hashing import compute_symbol_hashes
from faultline.llm.cost import CostTracker
from faultline.models.types import FeatureMap, SymbolAttribution

logger = logging.getLogger(__name__)


@dataclass
class SymbolDelta:
    """What changed in one file's symbols since the last scan."""
    file_path: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    body_modified: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def is_structurally_changed(self) -> bool:
        """True when symbols were added or removed — needs re-attribution."""
        return bool(self.added or self.removed)


@dataclass
class SymbolRefreshReport:
    deltas: list[SymbolDelta] = field(default_factory=list)
    files_checked: int = 0
    files_with_changes: int = 0
    symbols_added: int = 0
    symbols_removed: int = 0
    symbols_body_modified: int = 0
    features_reattributed: int = 0
    attributions_removed: int = 0

    def summary(self) -> str:
        return (
            f"Symbol refresh: {self.files_checked} files checked, "
            f"{self.symbols_added} added, {self.symbols_removed} removed, "
            f"{self.symbols_body_modified} bodies modified. "
            f"{self.features_reattributed} feature(s) re-attributed, "
            f"{self.attributions_removed} stale attribution(s) cleaned."
        )


def refresh_symbol_attributions(
    feature_map: FeatureMap,
    repo_path: str,
    *,
    files_to_check: list[str] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    tracker: CostTracker | None = None,
) -> SymbolRefreshReport:
    """Incrementally update symbol attributions.

    Args:
        feature_map: Mutated in place.
        repo_path: Root of the git repo.
        files_to_check: Subset of files to scan. Defaults to all
            files referenced by any feature.
        api_key / model: Passed to the symbol attribution LLM.
        tracker: Shared cost tracker.

    Returns:
        Report with per-file deltas and aggregate counts.
    """
    if files_to_check is None:
        files_to_check = sorted({p for f in feature_map.features for p in f.paths})

    if not files_to_check:
        return SymbolRefreshReport()

    report = SymbolRefreshReport(files_checked=len(files_to_check))

    # 1. Extract current symbols and hashes
    current_sigs = extract_signatures(files_to_check, repo_path)
    current_hashes = _hash_current_symbols(current_sigs, repo_path)
    old_hashes = feature_map.symbol_hashes or {}

    # 2. Diff per file
    deltas_by_file: dict[str, SymbolDelta] = {}
    for path in files_to_check:
        delta = _compute_delta(path, old_hashes.get(path, {}), current_hashes.get(path, {}))
        deltas_by_file[path] = delta
        report.deltas.append(delta)
        report.symbols_added += len(delta.added)
        report.symbols_removed += len(delta.removed)
        report.symbols_body_modified += len(delta.body_modified)
        if delta.added or delta.removed or delta.body_modified:
            report.files_with_changes += 1

    # 3. Cleanup: drop attributions for removed symbols
    report.attributions_removed = _cleanup_stale_attributions(feature_map, deltas_by_file)

    # 4. Re-attribute: for each feature that has any structurally-changed
    #    file, run one targeted LLM call to place the NEW symbols.
    structurally_changed_files = {
        path for path, d in deltas_by_file.items() if d.is_structurally_changed
    }
    if structurally_changed_files and api_key is not None:
        report.features_reattributed = _reattribute_changed_features(
            feature_map=feature_map,
            signatures=current_sigs,
            changed_files=structurally_changed_files,
            deltas_by_file=deltas_by_file,
            api_key=api_key,
            model=model,
            tracker=tracker,
        )

    # 5. Persist new hashes back to the map
    feature_map.symbol_hashes = current_hashes

    return report


def _hash_current_symbols(
    signatures: dict[str, FileSignature],
    repo_path: str,
) -> dict[str, dict[str, str]]:
    """{path: {symbol_name: body_hash}} for all files with symbol_ranges."""
    root = Path(repo_path)
    result: dict[str, dict[str, str]] = {}
    for path, sig in signatures.items():
        if not sig.symbol_ranges:
            continue
        # If the signature already carries the source, skip re-read
        source = sig.source
        if not source:
            try:
                source = (root / path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
        if not source:
            continue
        result[path] = compute_symbol_hashes(path, source, sig.symbol_ranges)
    return result


def _compute_delta(
    file_path: str,
    old: dict[str, str],
    new: dict[str, str],
) -> SymbolDelta:
    old_keys = set(old)
    new_keys = set(new)
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    body_modified = sorted(
        name for name in old_keys & new_keys
        if old[name] != new[name]
    )
    unchanged = sorted(
        name for name in old_keys & new_keys
        if old[name] == new[name]
    )
    return SymbolDelta(
        file_path=file_path,
        added=added,
        removed=removed,
        body_modified=body_modified,
        unchanged=unchanged,
    )


def _cleanup_stale_attributions(
    feature_map: FeatureMap,
    deltas_by_file: dict[str, SymbolDelta],
) -> int:
    """Remove symbols from attributions if they no longer exist in the file.

    Returns the number of symbols removed across all attributions.
    """
    removed_by_file = {
        path: set(d.removed) for path, d in deltas_by_file.items() if d.removed
    }
    if not removed_by_file:
        return 0

    total_removed = 0

    for feature in feature_map.features:
        # Clean feature-level shared_attributions (types/interfaces).
        total_removed += _clean_attribution_list(
            feature.shared_attributions, removed_by_file,
        )
        # Clean each flow.
        for flow in feature.flows:
            total_removed += _clean_attribution_list(
                flow.symbol_attributions, removed_by_file,
            )

    return total_removed


def _clean_attribution_list(
    attributions: list[SymbolAttribution],
    removed_by_file: dict[str, set[str]],
) -> int:
    """Mutates the list in place, dropping removed symbols. Returns count removed."""
    total = 0
    i = 0
    while i < len(attributions):
        att = attributions[i]
        dead = removed_by_file.get(att.file_path, set())
        if not dead:
            i += 1
            continue
        before = len(att.symbols)
        att.symbols = [s for s in att.symbols if s not in dead]
        total += before - len(att.symbols)
        if not att.symbols:
            attributions.pop(i)
            continue
        i += 1
    return total


def _reattribute_changed_features(
    *,
    feature_map: FeatureMap,
    signatures: dict[str, FileSignature],
    changed_files: set[str],
    deltas_by_file: dict[str, SymbolDelta],
    api_key: str,
    model: str | None,
    tracker: CostTracker | None,
) -> int:
    """For each feature that touches a structurally-changed file, run
    a single attribution call (via the existing symbols/attribution
    module) that operates on the union of added + already-attributed
    symbols. This keeps LLM cost low — one call per feature, not per flow.
    """
    from faultline.symbols.attribution import attribute_symbols_to_flows
    from faultline.symbols.extractor import extract_file_symbols

    file_symbols = extract_file_symbols(signatures)
    affected_count = 0

    for feature in feature_map.features:
        if not feature.flows:
            continue
        if not any(p in changed_files for p in feature.paths):
            continue

        # Wipe the flow-level attributions that match any changed file.
        # Then let the LLM rebuild them for just those files.
        for flow in feature.flows:
            flow.symbol_attributions = [
                att for att in flow.symbol_attributions
                if att.file_path not in changed_files
            ]

        # Subset file_symbols to just what this feature touches, so
        # attribution prompt stays small.
        subset: dict[str, Any] = {
            p: file_symbols[p] for p in feature.paths
            if p in file_symbols
        }
        if not subset:
            continue

        kwargs: dict[str, Any] = {"api_key": api_key, "tracker": tracker}
        if model:
            kwargs["model"] = model
        try:
            attribute_symbols_to_flows(feature, subset, **kwargs)
            affected_count += 1
        except Exception as exc:
            logger.warning("re-attribution failed for %s: %s", feature.name, exc)

    return affected_count
