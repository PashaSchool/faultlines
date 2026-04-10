"""Shared file attribution — maps symbols from shared files to features.

A shared file is one that appears in 2+ features' paths. For each feature,
this module determines which exported symbols the feature actually imports,
and computes the line ranges those symbols occupy. This enables per-feature
health scoring weighted by the fraction of the file each feature truly owns.
"""
from faultline.analyzer.ast_extractor import FileSignature
from faultline.models.types import SymbolAttribution, SymbolRange


def build_shared_attributions(
    feature_paths: dict[str, list[str]],
    symbol_imports: dict[str, dict[str, set[str]]],
    signatures: dict[str, FileSignature],
) -> dict[str, list[SymbolAttribution]]:
    """Computes per-feature symbol attributions for shared files.

    Args:
        feature_paths: feature_name → list of file paths.
        symbol_imports: importer_file → {imported_file → {symbol names}}.
        signatures: file_path → FileSignature (with symbol_ranges populated).

    Returns:
        feature_name → list of SymbolAttribution for shared files in that feature.
    """
    # Find shared files: files that appear in 2+ features
    file_to_features: dict[str, set[str]] = {}
    for feat_name, paths in feature_paths.items():
        for p in paths:
            file_to_features.setdefault(p, set()).add(feat_name)

    shared_files = {p for p, feats in file_to_features.items() if len(feats) > 1}
    if not shared_files:
        return {}

    result: dict[str, list[SymbolAttribution]] = {}

    for shared_path in shared_files:
        sig = signatures.get(shared_path)
        if not sig or not sig.symbol_ranges:
            continue

        total_lines = sig.source.count("\n") + 1 if sig.source else 1
        symbol_map = {sr.name: sr for sr in sig.symbol_ranges}

        for feat_name in file_to_features[shared_path]:
            feat_files = feature_paths.get(feat_name, [])
            imported_symbols = _collect_imported_symbols(
                feat_files, shared_path, symbol_imports,
            )

            if not imported_symbols:
                # Feature owns this file but no file in it imports from it —
                # LLM assigned it directly. Attribute all symbols.
                imported_symbols = set(symbol_map.keys())

            # Map symbol names to line ranges
            ranges = []
            symbols_found = []
            for sym_name in imported_symbols:
                if sym_name == "*":
                    # Namespace import — attribute all symbols
                    ranges = [(sr.start_line, sr.end_line) for sr in sig.symbol_ranges]
                    symbols_found = [sr.name for sr in sig.symbol_ranges]
                    break
                sr = symbol_map.get(sym_name)
                if sr:
                    ranges.append((sr.start_line, sr.end_line))
                    symbols_found.append(sym_name)

            if not ranges:
                continue

            merged = _merge_line_ranges(ranges)
            attributed = _lines_in_ranges(merged)

            attribution = SymbolAttribution(
                file_path=shared_path,
                symbols=sorted(symbols_found),
                line_ranges=merged,
                attributed_lines=attributed,
                total_file_lines=total_lines,
            )
            result.setdefault(feat_name, []).append(attribution)

    return result


def _collect_imported_symbols(
    feature_files: list[str],
    shared_file: str,
    symbol_imports: dict[str, dict[str, set[str]]],
) -> set[str]:
    """Collects all symbol names that any file in the feature imports from shared_file."""
    symbols: set[str] = set()
    for f in feature_files:
        file_imports = symbol_imports.get(f, {})
        if shared_file in file_imports:
            symbols.update(file_imports[shared_file])
    return symbols


def _merge_line_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merges overlapping or adjacent line ranges into non-overlapping spans."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _lines_in_ranges(ranges: list[tuple[int, int]]) -> int:
    """Counts total lines across merged ranges."""
    return sum(end - start + 1 for start, end in ranges)
