"""Symbol extraction helpers.

Wraps the existing ast_extractor to produce a clean per-file symbol
listing, split into flow-eligible and feature-only (types/interfaces).
"""

from __future__ import annotations

from dataclasses import dataclass

from faultline.analyzer.ast_extractor import FileSignature

# SymbolRange.kind values that attribute to flows.
# Everything else (type, interface, enum, reexport) stays at feature
# level as shared context.
_FLOW_ELIGIBLE_KINDS = {"function", "class", "const"}


@dataclass
class FileSymbols:
    """Symbols in a single file, split by attribution scope."""
    path: str
    flow_symbols: list[str]     # functions/classes/consts — can go to specific flows
    feature_symbols: list[str]  # types/interfaces/enums — feature-level only


def extract_file_symbols(signatures: dict[str, FileSignature]) -> dict[str, FileSymbols]:
    """Split each file's exports into flow-eligible vs feature-only symbols.

    Args:
        signatures: Output from ast_extractor.extract_signatures().

    Returns:
        dict mapping file_path → FileSymbols.
    """
    result: dict[str, FileSymbols] = {}

    for path, sig in signatures.items():
        flow_syms: list[str] = []
        feature_syms: list[str] = []

        # Use symbol_ranges (richer, has kind) when available.
        if sig.symbol_ranges:
            for sym in sig.symbol_ranges:
                if sym.kind in _FLOW_ELIGIBLE_KINDS:
                    flow_syms.append(sym.name)
                else:
                    # type, interface, enum, reexport
                    feature_syms.append(sym.name)
        else:
            # Python fallback — no symbol_ranges yet. Treat all exports
            # as flow-eligible since AST doesn't separate them.
            flow_syms = list(sig.exports)

        result[path] = FileSymbols(
            path=path,
            flow_symbols=_dedupe(flow_syms),
            feature_symbols=_dedupe(feature_syms),
        )

    return result


def _dedupe(items: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
