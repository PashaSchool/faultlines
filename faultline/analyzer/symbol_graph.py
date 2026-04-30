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
# Re-export shapes:
#   ``export { X, Y } from './m'``
#   ``export { X as Z } from './m'``
#   ``export * from './m'``
#   ``export * as ns from './m'``
# All count as "this file imports those symbols and re-exposes them"
# for BFS purposes — flows that touch the re-exporting barrel reach
# through to the original definitions.
_RE_NAMED_REEXPORT = re.compile(
    r"^\s*export\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)
_RE_STAR_REEXPORT = re.compile(
    r"^\s*export\s*\*(?:\s+as\s+\w+)?\s+from\s*['\"]([^'\"]+)['\"]",
    re.MULTILINE,
)
# Dynamic imports: ``const X = await import('./y')`` or
# ``import('./y').then(...)``.
_RE_DYNAMIC_IMPORT = re.compile(
    r"\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
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

_PY_EXTENSIONS: frozenset[str] = frozenset({".py"})
_GO_EXTENSIONS: frozenset[str] = frozenset({".go"})
_RS_EXTENSIONS: frozenset[str] = frozenset({".rs"})

# ``from .module import X`` / ``from ..parent import X`` / ``from a.b import X``
_RE_PY_FROM_IMPORT = re.compile(
    r"^[ \t]*from[ \t]+(?P<mod>\.+|[\w.]+(?:\.\w+)*)[ \t]+import[ \t]+"
    r"(?P<names>\([^)]*\)|[^\n#]+)",
    re.MULTILINE,
)
# ``import package.sub`` / ``import package.sub as alias``
_RE_PY_BARE_IMPORT = re.compile(
    r"^[ \t]*import[ \t]+(?P<mod>[\w.]+(?:\.\w+)*)(?:[ \t]+as[ \t]+\w+)?",
    re.MULTILINE,
)


def _extract_py_imports(source: str) -> dict[str, set[str]]:
    """Parse Python source → ``{module: {symbol_or_'*'_or_'@import'}}``.

    Conventions used to mirror the TS/JS shape downstream BFS expects:
      - Specific names (``from x import a, b``) → set of names
      - Bare ``import x`` → ``{"@import"}`` (side-effect-equivalent;
        we know the file participates but no symbol attribution).
      - ``from x import *`` → ``{"*"}`` (namespace).
    """
    out: dict[str, set[str]] = {}
    for m in _RE_PY_FROM_IMPORT.finditer(source):
        mod = m.group("mod")
        names_raw = m.group("names").replace("(", "").replace(")", "")
        names: set[str] = set()
        for tok in names_raw.split(","):
            t = tok.strip().split(" as ")[0].strip()
            if not t:
                continue
            if t == "*":
                names.add("*")
            else:
                names.add(t)
        if names:
            out.setdefault(mod, set()).update(names)
    for m in _RE_PY_BARE_IMPORT.finditer(source):
        mod = m.group("mod")
        out.setdefault(mod, set()).add("@import")
    return out


def _resolve_py_module(
    importer_file: str,
    module: str,
    file_set: set[str],
) -> str | None:
    """Resolve a Python import target to a repo-relative file path.

    Handles three cases:
      1. Relative imports (``.foo``, ``..bar.baz``) — counted dots
         walk the importer's directory tree.
      2. Absolute intra-repo imports (``apps.api.foo``) — try every
         common package-root prefix and look for matching .py.
      3. ``from . import x`` (lone dot) — module is in the same dir
         as importer.

    Returns ``None`` for stdlib / third-party / unresolvable imports.
    The BFS tolerates this — only resolved edges contribute.
    """
    importer_dir = str(Path(importer_file).parent).replace("\\", "/")
    if importer_dir == ".":
        importer_dir = ""

    # Relative ``.``, ``..`` etc.
    if module.startswith("."):
        # Count leading dots — N dots = N-1 levels up.
        dots = len(module) - len(module.lstrip("."))
        rest = module[dots:]
        # Walk up dots-1 levels from importer_dir.
        parts = importer_dir.split("/") if importer_dir else []
        if dots - 1 > len(parts):
            return None
        base_parts = parts[: len(parts) - (dots - 1)] if dots > 1 else parts
        rest_parts = rest.split(".") if rest else []
        candidate_parts = base_parts + rest_parts
        return _try_resolve_parts(candidate_parts, file_set)

    # Absolute. Try matching against the file_set as-is, plus every
    # common src/apps prefix.
    parts = module.split(".")
    direct = _try_resolve_parts(parts, file_set)
    if direct:
        return direct
    # Common monorepo prefixes — try each.
    for prefix in ("apps", "src", "lib", "packages"):
        cand = _try_resolve_parts([prefix] + parts, file_set)
        if cand:
            return cand
    return None


def _try_resolve_parts(parts: list[str], file_set: set[str]) -> str | None:
    """Given path parts, look for ``parts.py`` or ``parts/__init__.py``.

    Both shapes are valid Python module locations. Returns the
    matching file from ``file_set`` if either exists.
    """
    if not parts:
        return None
    base = "/".join(parts)
    cand_file = f"{base}.py"
    cand_init = f"{base}/__init__.py"
    if cand_file in file_set:
        return cand_file
    if cand_init in file_set:
        return cand_init
    return None


# ── Go imports (P6a) ────────────────────────────────────────────


_RE_GO_SINGLE_IMPORT = re.compile(
    r'^[ \t]*import[ \t]+(?:[\w.]+[ \t]+)?"(?P<path>[^"]+)"',
    re.MULTILINE,
)
_RE_GO_BLOCK_IMPORT = re.compile(
    r"^[ \t]*import[ \t]*\(\s*(?P<body>[^)]*)\)",
    re.MULTILINE | re.DOTALL,
)
_RE_GO_BLOCK_LINE = re.compile(
    r'^\s*(?:[\w.]+\s+)?"(?P<path>[^"]+)"',
    re.MULTILINE,
)


def _go_module_path(repo_root: str | Path) -> str | None:
    """Read ``go.mod`` and return the declared module path."""
    p = Path(repo_root) / "go.mod"
    if not p.is_file():
        return None
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("module "):
                return line.split(None, 1)[1].strip()
    except OSError:
        return None
    return None


def _extract_go_imports(source: str) -> set[str]:
    """Parse a Go source file → set of import paths."""
    out: set[str] = set()
    for m in _RE_GO_SINGLE_IMPORT.finditer(source):
        out.add(m.group("path"))
    for block in _RE_GO_BLOCK_IMPORT.finditer(source):
        for m in _RE_GO_BLOCK_LINE.finditer(block.group("body")):
            out.add(m.group("path"))
    return out


def _resolve_go_module(
    module_path: str,
    file_set: set[str],
    repo_module: str | None,
) -> str | None:
    """Resolve a Go import to a representative ``.go`` file in the repo.

    Go's "one package per dir" rule means we pick any non-test
    ``.go`` file inside the matching directory — BFS will treat
    that as the package entry point.
    """
    if not repo_module:
        return None
    if module_path == repo_module:
        return _first_go_file_in_dir("", file_set)
    if module_path.startswith(repo_module + "/"):
        rel = module_path[len(repo_module) + 1:]
        return _first_go_file_in_dir(rel, file_set)
    return None


def _first_go_file_in_dir(
    rel_dir: str, file_set: set[str],
) -> str | None:
    """Return any non-test ``.go`` file directly in ``rel_dir``."""
    prefix = (rel_dir.rstrip("/") + "/") if rel_dir else ""
    for f in sorted(file_set):
        if not f.startswith(prefix):
            continue
        rest = f[len(prefix):]
        if "/" in rest:
            continue
        if f.endswith(".go") and not f.endswith("_test.go"):
            return f
    return None


# ── Rust imports (P6b) ──────────────────────────────────────────


# ``use crate::auth::Login;`` / ``use crate::auth::*;`` — intra-crate
# ``use super::sibling::Foo;`` / ``use self::nested::Bar;``
# ``mod foo;`` — declares a child module (resolve to foo.rs or foo/mod.rs)
_RE_RS_USE = re.compile(
    r"^\s*(?:pub\s+)?use[ \t]+"
    r"(?P<full>(?:crate|super|self)(?:::(?:\w+|\*|\{[^}]*\}))+)\s*;",
    re.MULTILINE,
)
_RE_RS_MOD = re.compile(
    r"^\s*(?:pub\s+)?mod[ \t]+(?P<name>\w+)\s*;",
    re.MULTILINE,
)


def _extract_rs_imports(source: str) -> set[str]:
    """Parse Rust source → set of intra-crate MODULE paths.

    For ``use crate::auth::login::Handler;`` the captured ``full`` is
    ``crate::auth::login::Handler`` — we strip the trailing element
    (the imported symbol) to keep just the module path
    ``crate::auth::login``. Glob (``::*``) and brace (``::{...}``)
    forms have NO trailing symbol; the path is what's before them.

    Third-party crate imports (``use serde::Deserialize``) don't
    start with ``crate``/``super``/``self`` so the regex never
    matches.
    """
    out: set[str] = set()
    for m in _RE_RS_USE.finditer(source):
        full = m.group("full")
        # ``full`` looks like ``crate::a::b::FINAL`` where FINAL is
        # either a symbol name, ``*``, or ``{...}``. Drop the FINAL
        # segment — what's left is the module path.
        # Find the last ``::`` and strip everything from there.
        idx = full.rfind("::")
        if idx == -1:
            continue
        module = full[:idx]
        if module:
            out.add(module)
    return out


def _extract_rs_mods(source: str) -> set[str]:
    """Parse ``mod foo;`` declarations — these declare a child module
    of the current file."""
    return {m.group("name") for m in _RE_RS_MOD.finditer(source)}


def _resolve_rs_use(
    importer_file: str,
    use_path: str,
    file_set: set[str],
    crate_root: str,
) -> str | None:
    """Resolve a Rust ``use`` path to a file in the crate.

    Args:
        importer_file: e.g. ``src/auth/login.rs``
        use_path: e.g. ``crate::api::handler``
        crate_root: e.g. ``src`` (where ``lib.rs`` / ``main.rs`` lives)
    """
    parts = use_path.split("::")
    if not parts:
        return None
    head = parts[0]
    rest = parts[1:]

    if head == "crate":
        # Walk from crate_root.
        base_parts = [crate_root] if crate_root else []
    elif head == "super":
        # One dir up from importer.
        importer_parts = importer_file.split("/")
        # Drop file basename + one parent.
        if len(importer_parts) < 2:
            return None
        base_parts = importer_parts[:-2]
    elif head == "self":
        # Same dir as importer.
        importer_parts = importer_file.split("/")
        base_parts = importer_parts[:-1]
    else:
        return None

    # Drop the trailing element — Rust ``use crate::a::b::Foo`` imports
    # a SYMBOL ``Foo`` from MODULE ``crate::a::b``. We resolve to the
    # module file.
    module_parts = rest[:-1] if rest else []
    return _try_resolve_rs_parts(base_parts + module_parts, file_set)


def _try_resolve_rs_parts(
    parts: list[str], file_set: set[str],
) -> str | None:
    """``a/b/c.rs`` or ``a/b/c/mod.rs`` shaped lookup."""
    if not parts:
        return None
    base = "/".join(parts)
    cand_file = f"{base}.rs"
    cand_mod = f"{base}/mod.rs"
    if cand_file in file_set:
        return cand_file
    if cand_mod in file_set:
        return cand_mod
    return None


def _detect_rust_crate_root(file_set: set[str]) -> str:
    """Find ``src`` if it exists; otherwise empty string (root layout)."""
    if any(f.startswith("src/") for f in file_set):
        return "src"
    return ""


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
    *,
    include_http_edges: bool = True,
) -> SymbolGraph:
    """Construct the symbol graph from a repo's source files.

    Args:
        repo_root: Absolute path to the repo root.
        source_files: List of repo-relative file paths to analyse.
            Non-TS/JS files are silently skipped (their imports are
            not part of the BFS).
        include_http_edges: When True (default), build a route
            registry via :mod:`faultline.analyzer.url_route_resolver`
            and add **virtual** ImportEdges from every client file
            making a fetch / axios / tRPC call to the server file
            handling the matched route. The synthetic
            ``target_symbol`` is ``"@http"`` so callers can
            distinguish HTTP edges from real imports. Set False for
            a pure static-import view (e.g. unit tests).

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

    # Walk Python imports first — same edge shape as TS/JS so BFS
    # treats them uniformly. Skip files whose source we already
    # parsed via signatures.
    for path in source_files:
        suffix = Path(path).suffix.lower()
        if suffix not in _PY_EXTENSIONS:
            continue
        source = _read_safe(repo_root, path)
        if not source:
            continue
        per_module = _extract_py_imports(source)
        edges: list[ImportEdge] = []
        for module, symbols in per_module.items():
            target = _resolve_py_module(path, module, file_set)
            if not target:
                continue
            for sym in symbols:
                edges.append(ImportEdge(
                    target_file=target, target_symbol=sym,
                ))
        if edges:
            graph.forward.setdefault(path, []).extend(edges)
            for e in edges:
                graph.reverse.setdefault(e.target_file, []).append(
                    ImportEdge(target_file=path, target_symbol=e.target_symbol)
                )

    # Walk Go imports — same edge shape, "@import" symbol since Go
    # imports a package not a specific symbol.
    repo_module = _go_module_path(repo_root)
    if repo_module:
        for path in source_files:
            if Path(path).suffix.lower() not in _GO_EXTENSIONS:
                continue
            source = _read_safe(repo_root, path)
            if not source:
                continue
            for module_path in _extract_go_imports(source):
                target = _resolve_go_module(module_path, file_set, repo_module)
                if not target or target == path:
                    continue
                edge = ImportEdge(target_file=target, target_symbol="@import")
                graph.forward.setdefault(path, []).append(edge)
                graph.reverse.setdefault(target, []).append(
                    ImportEdge(target_file=path, target_symbol="@import")
                )

    # Walk Rust ``use`` statements + ``mod`` declarations.
    rs_files = [f for f in source_files if Path(f).suffix.lower() in _RS_EXTENSIONS]
    if rs_files:
        crate_root = _detect_rust_crate_root(file_set)
        for path in rs_files:
            source = _read_safe(repo_root, path)
            if not source:
                continue
            uses = _extract_rs_imports(source)
            mods = _extract_rs_mods(source)
            edges: list[ImportEdge] = []
            for use_path in uses:
                target = _resolve_rs_use(path, use_path, file_set, crate_root)
                if target and target != path:
                    edges.append(ImportEdge(
                        target_file=target, target_symbol="@import",
                    ))
            # ``mod foo;`` declares child module → either ``foo.rs``
            # in the same dir or ``foo/mod.rs``.
            importer_dir = str(Path(path).parent).replace("\\", "/")
            if importer_dir == ".":
                importer_dir = ""
            for mod_name in mods:
                cand_file = (
                    f"{importer_dir}/{mod_name}.rs"
                    if importer_dir else f"{mod_name}.rs"
                )
                cand_mod = (
                    f"{importer_dir}/{mod_name}/mod.rs"
                    if importer_dir else f"{mod_name}/mod.rs"
                )
                target = (
                    cand_file if cand_file in file_set
                    else cand_mod if cand_mod in file_set
                    else None
                )
                if target and target != path:
                    edges.append(ImportEdge(
                        target_file=target, target_symbol="@import",
                    ))
            for e in edges:
                graph.forward.setdefault(path, []).append(e)
                graph.reverse.setdefault(e.target_file, []).append(
                    ImportEdge(target_file=path, target_symbol="@import")
                )

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
        # Named re-exports — pass each symbol through to BFS.
        for m in _RE_NAMED_REEXPORT.finditer(source):
            names_str = m.group(1)
            module = m.group(2)
            if not (module.startswith(".") or module.startswith("@/") or module.startswith("~/")):
                continue
            for tok in names_str.split(","):
                parts = tok.strip().split(" as ")
                original = parts[0].strip()
                if original:
                    per_module.setdefault(module, set()).add(original)
        # Star re-exports — record as namespace ("*") so BFS pulls
        # every export of the target.
        for m in _RE_STAR_REEXPORT.finditer(source):
            module = m.group(1)
            if module.startswith(".") or module.startswith("@/") or module.startswith("~/"):
                per_module.setdefault(module, set()).add("*")
        # Dynamic imports — treated as side-effect for BFS purposes
        # since we can't statically know which symbol is destructured
        # off the resolved module.
        for m in _RE_DYNAMIC_IMPORT.finditer(source):
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

    # Improvement #6: layer in URL/tRPC HTTP-boundary edges so
    # Sprint 7 BFS traces from a UI component through the API
    # client to the actual server handler in one walk.
    if include_http_edges:
        try:
            _add_http_edges(graph, repo_root, source_files)
        except Exception as exc:  # noqa: BLE001 — opportunistic
            logger.warning(
                "symbol_graph: HTTP edge layering failed (%s)", exc,
            )

    return graph


def _add_http_edges(
    graph: SymbolGraph,
    repo_root: str | Path,
    source_files: list[str],
) -> None:
    """Layer fetch/axios/tRPC client→server edges onto ``graph``.

    For each ``ClientCall`` that resolves to one or more server
    files, add an :class:`ImportEdge` with the synthetic symbol
    ``"@http"`` so the BFS walker reaches the server via the
    runtime call. Edges are added to BOTH ``forward`` and
    ``reverse`` so the graph is consistent.

    Self-edges (call resolves to its own file — possible on small
    Next.js apps where the same file declares both server route
    and client fetch) are skipped to avoid loops.
    """
    from .url_route_resolver import build_route_registry, resolve_call_to_routes

    registry = build_route_registry(repo_root, source_files)
    if not registry.calls or not registry.routes:
        return

    added = 0
    for call in registry.calls:
        targets = resolve_call_to_routes(call, registry)
        for server_file in targets:
            if server_file == call.file:
                continue
            edge = ImportEdge(target_file=server_file, target_symbol="@http")
            # Avoid duplicate edges (same call site can appear
            # multiple times in a file; one edge is enough).
            forward_list = graph.forward.setdefault(call.file, [])
            if edge not in forward_list:
                forward_list.append(edge)
                graph.reverse.setdefault(server_file, []).append(
                    ImportEdge(target_file=call.file, target_symbol="@http")
                )
                added += 1

    if added:
        logger.info(
            "symbol_graph: layered %d HTTP edge(s) "
            "(client → server via fetch/axios/tRPC)",
            added,
        )


def _read_safe(repo_root: str, rel_path: str) -> str | None:
    abs_path = Path(repo_root) / rel_path
    try:
        return abs_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
