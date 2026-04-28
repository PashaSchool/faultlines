"""Symbol-level import graph (Sprint 7 Day 1).

Builds a callgraph from a repo's TS / JS source files so flow tracing
(Sprint 7) can BFS from a route handler down to all UI / state /
API / schema files that participate in a single user-facing flow.

Two outputs:

  - ``SymbolGraph.forward[file]`` — list of imports as
    ``ImportEdge(target_file, target_symbol)`` resolved to actual
    repo-relative paths.
  - ``SymbolGraph.reverse[file]`` — inverse map; everyone who
    imports any symbol from ``file``.

Plus per-file metadata that the flow walker needs:
  - ``exports[file]`` — list of exported ``SymbolRange`` so we can
    match an entry-point ``file:line`` to its enclosing symbol.
  - ``symbol_ranges[file]`` — every top-level symbol's line span,
    used for participant attribution.

Reuses the existing regex AST extractor and tsconfig path resolver
— no new parser. Python files use the same plumbing but with their
own (existing) signature extractor; for Sprint 7 the BFS is
TS/JS-only by default since most modern web frontends are JS.

This module is pure local analysis. No LLM calls.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .ast_extractor import (
    SymbolRange,
    extract_named_imports,
    extract_signatures,
)


# Default imports: ``import Foo from './foo'`` or
# ``import Foo, { Bar } from './foo'``. ``extract_named_imports``
# captures the named cluster but NOT the leading default symbol —
# we extract it here. Side-effect-only imports (``import './x'``)
# resolve to a target file but record the synthetic ``"@import"``
# symbol so the BFS can still traverse the edge.
_RE_DEFAULT_IMPORT = re.compile(
    r"^\s*import\s+([A-Za-z_$][\w$]*)\s*(?:,\s*\{[^}]*\}\s*)?from\s*['\"]"
    r"([^'\"]+)['\"]",
    re.MULTILINE,
)
# Combined import: ``import Default, { Named1, Named2 } from './x'``.
# extract_named_imports' base regex requires whitespace-only between
# ``import`` and ``{`` and misses this shape; we extract the named
# cluster here.
_RE_COMBINED_IMPORT = re.compile(
    r"^\s*import\s+[A-Za-z_$][\w$]*\s*,\s*\{([^}]+)\}\s*from\s*['\"]"
    r"([^'\"]+)['\"]",
    re.MULTILINE,
)
_RE_SIDE_EFFECT_IMPORT = re.compile(
    r"^\s*import\s+['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)
from .import_graph import (
    _resolve_import,
    detect_monorepo_packages,
    load_tsconfig_paths,
)


logger = logging.getLogger(__name__)


_TS_JS_EXTENSIONS: frozenset[str] = frozenset({
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
})


# ── Data shapes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImportEdge:
    """One import edge from an importer file to a target symbol.

    ``target_symbol`` is the imported name as it appears in the
    importer (post-rename for ``import { X as Y }``, this is ``X``).
    A namespace import (``import * as ns``) records ``"*"``.
    """

    target_file: str
    target_symbol: str


@dataclass
class SymbolGraph:
    """File-level symbol graph for a single repo."""

    exports: dict[str, list[SymbolRange]] = field(default_factory=dict)
    symbol_ranges: dict[str, list[SymbolRange]] = field(default_factory=dict)
    forward: dict[str, list[ImportEdge]] = field(default_factory=dict)
    reverse: dict[str, list[ImportEdge]] = field(default_factory=dict)

    def imports_from(self, file: str) -> list[ImportEdge]:
        """Return resolved imports declared in ``file``."""
        return list(self.forward.get(file, []))

    def callers_of(self, file: str) -> list[ImportEdge]:
        """Return files that import any symbol from ``file``."""
        return list(self.reverse.get(file, []))

    def find_enclosing_symbol(
        self, file: str, line: int,
    ) -> SymbolRange | None:
        """Return the SymbolRange of ``file`` containing ``line``.

        When multiple ranges overlap (e.g. a function within a class),
        returns the smallest enclosing range — the most specific
        match.
        """
        ranges = self.symbol_ranges.get(file, [])
        candidates = [
            r for r in ranges
            if r.start_line <= line <= r.end_line
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.end_line - r.start_line)


# ── Builder ─────────────────────────────────────────────────────────


def build_symbol_graph(
    repo_root: str | Path,
    source_files: list[str],
) -> SymbolGraph:
    """Construct the symbol graph from a repo's source files.

    Args:
        repo_root: Absolute path to the repo root.
        source_files: List of repo-relative file paths to analyse.
            Non-TS/JS files are silently skipped (their imports are
            not part of the BFS).

    Returns:
        Populated :class:`SymbolGraph`. Missing files / parse errors
        contribute empty entries; the graph never raises on bad
        input — Sprint 7 must keep working even on partial repos.
    """
    repo_root = str(repo_root)
    file_set = set(source_files)
    alias_map = load_tsconfig_paths(repo_root)
    try:
        monorepo_packages = detect_monorepo_packages(repo_root)
    except Exception:  # noqa: BLE001 — defensive on partial repos
        monorepo_packages = set()

    signatures = extract_signatures(source_files, repo_root)

    graph = SymbolGraph()

    # Carry exports + symbol ranges over from the AST extractor.
    for path, sig in signatures.items():
        if sig.symbol_ranges:
            graph.symbol_ranges[path] = list(sig.symbol_ranges)
        # Build a synthetic SymbolRange list for ``exports`` so the
        # flow walker can match imported names back to a file's
        # exports without re-parsing. When no ranges are available
        # (e.g. Python regex extractor that doesn't compute spans),
        # synthesise a single-line placeholder.
        ranges_by_name: dict[str, SymbolRange] = {
            r.name: r for r in (sig.symbol_ranges or [])
        }
        per_file: list[SymbolRange] = []
        for name in sig.exports or []:
            if name in ranges_by_name:
                per_file.append(ranges_by_name[name])
            else:
                per_file.append(SymbolRange(
                    name=name, start_line=1, end_line=1,
                    kind="export",
                ))
        if per_file:
            graph.exports[path] = per_file

    # Walk imports per file and resolve them.
    for path in source_files:
        suffix = Path(path).suffix.lower()
        if suffix not in _TS_JS_EXTENSIONS:
            continue
        sig = signatures.get(path)
        # ``extract_signatures`` only stores source for TS/JS files —
        # use it when present, otherwise re-read.
        source = sig.source if sig else _read_safe(repo_root, path)
        if not source:
            continue

        # Collect (module, symbol) pairs from all import shapes.
        per_module: dict[str, set[str]] = {}
        for module, symbols in extract_named_imports(source).items():
            per_module.setdefault(module, set()).update(symbols)
        for m in _RE_DEFAULT_IMPORT.finditer(source):
            sym = m.group(1)
            module = m.group(2)
            if module.startswith(".") or module.startswith("@/") or module.startswith("~/"):
                per_module.setdefault(module, set()).add(sym)
        for m in _RE_COMBINED_IMPORT.finditer(source):
            names_str = m.group(1)
            module = m.group(2)
            if not (module.startswith(".") or module.startswith("@/") or module.startswith("~/")):
                continue
            for tok in names_str.split(","):
                parts = tok.strip().split(" as ")
                original = parts[0].strip()
                if original:
                    per_module.setdefault(module, set()).add(original)
        for m in _RE_SIDE_EFFECT_IMPORT.finditer(source):
            module = m.group(1)
            if module.startswith(".") or module.startswith("@/") or module.startswith("~/"):
                per_module.setdefault(module, set()).add("@import")

        edges: list[ImportEdge] = []
        for module, symbols in per_module.items():
            target = _resolve_import(
                path, module, file_set,
                alias_map=alias_map,
                monorepo_packages=monorepo_packages,
            )
            if not target:
                continue
            for sym in symbols:
                edges.append(ImportEdge(
                    target_file=target, target_symbol=sym,
                ))
        if edges:
            graph.forward[path] = edges
            for e in edges:
                # Reverse map: e.target_file gets a back-edge with
                # ``target_file=path`` (the importer).
                graph.reverse.setdefault(e.target_file, []).append(
                    ImportEdge(target_file=path, target_symbol=e.target_symbol)
                )

    logger.info(
        "symbol_graph: %d files with exports, %d files with imports, "
        "%d total forward edges",
        len(graph.exports), len(graph.forward),
        sum(len(v) for v in graph.forward.values()),
    )
    return graph


def _read_safe(repo_root: str, rel_path: str) -> str | None:
    abs_path = Path(repo_root) / rel_path
    try:
        return abs_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
