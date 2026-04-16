"""Symbol extraction helpers.

Wraps the existing ast_extractor to produce a clean per-file symbol
listing, split into flow-eligible and feature-only (types/interfaces),
with per-symbol line ranges so attribution can deep-link to exact code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from faultline.analyzer.ast_extractor import FileSignature
from faultline.symbols.roles import classify

# SymbolRange.kind values that attribute to flows.
# Everything else (type, interface, enum, reexport) stays at feature
# level as shared context.
_FLOW_ELIGIBLE_KINDS = {"function", "class", "const"}


@dataclass
class FileSymbols:
    """Symbols in a single file, split by attribution scope.

    ``symbol_lines`` is the source of truth for SymbolAttribution.line_ranges
    — the LLM never sees lines, it only emits names; ranges are looked up
    here when applying the attribution back to the feature map.

    ``symbol_roles`` is the deterministic role classification (state,
    loading-state, ui-component, ...) computed up-front so attribution
    can stamp roles without re-inspecting source.
    """
    path: str
    flow_symbols: list[str]     # functions/classes/consts — can go to specific flows
    feature_symbols: list[str]  # types/interfaces/enums — feature-level only
    symbol_lines: dict[str, tuple[int, int]] = field(default_factory=dict)
    total_file_lines: int = 0
    symbol_roles: dict[str, str] = field(default_factory=dict)


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
        symbol_lines: dict[str, tuple[int, int]] = {}
        symbol_roles: dict[str, str] = {}
        has_routes = bool(sig.routes)

        # Use symbol_ranges (richer, has kind + line info) when available.
        if sig.symbol_ranges:
            for sym in sig.symbol_ranges:
                if sym.kind in _FLOW_ELIGIBLE_KINDS:
                    flow_syms.append(sym.name)
                else:
                    # type, interface, enum, reexport
                    feature_syms.append(sym.name)
                # Keep ranges for ALL kinds — feature-level attributions
                # need them too (so dashboards can deep-link to a type def).
                symbol_lines[sym.name] = (sym.start_line, sym.end_line)
                symbol_roles[sym.name] = classify(
                    sym.name, sym.kind, path, has_routes,
                )
        else:
            # Python fallback — no symbol_ranges yet. Treat all exports
            # as flow-eligible since AST doesn't separate them.
            flow_syms = list(sig.exports)
            for name in flow_syms:
                # Without kind info, default to "function" — the role
                # classifier degrades gracefully (still catches handler/
                # validator/loading patterns from name).
                symbol_roles[name] = classify(name, "function", path, has_routes)

        # Total file lines for "x of y lines attributed" stats.
        total_lines = 0
        if sig.source:
            total_lines = sig.source.count("\n") + 1
        elif symbol_lines:
            total_lines = max((end for _, end in symbol_lines.values()), default=0)

        result[path] = FileSymbols(
            path=path,
            flow_symbols=_dedupe(flow_syms),
            feature_symbols=_dedupe(feature_syms),
            symbol_lines=symbol_lines,
            total_file_lines=total_lines,
            symbol_roles=symbol_roles,
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
