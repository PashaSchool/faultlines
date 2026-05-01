"""Map test files to the symbols they exercise.

Powers symbol-scoped test coverage: for each ``(source_file, symbol)``
pair, list which test files cover it. Used by Sprint 1 Day 9 scoring
to compute per-flow / per-symbol test-presence signal.

Approach (per user spec, "C — mix"):

  1. **Import-based primary** — parse a test file's imports via the
     existing ``SymbolGraph``. Each ``ImportEdge(target_file,
     target_symbol)`` says: this test file pulls in ``target_symbol``
     from ``target_file``. We record this as
     ``(target_file, target_symbol) -> test_file``.

     Namespace imports (``import * as X``) and side-effect imports
     (``import './setup'``) attach the test to ALL exports of the
     target file (they exercise every symbol potentially).

  2. **Filename fallback** — when a test file has no resolvable
     imports (or the target file isn't in the graph), match by name:

       - ``utils.test.ts`` / ``utils.spec.ts`` → ``utils.ts``
       - ``test_auth.py`` / ``auth_test.py`` → ``auth.py``
       - ``auth_test.go`` → ``auth.go``
       - ``tests/parser.rs`` → ``src/parser.rs`` (same crate)

     The fallback attaches the test to ALL symbols of the matched
     file. Less precise than imports but better than nothing.

Both passes contribute to the same output. A test file can map to
many symbols; one symbol can be exercised by many tests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .symbol_graph import SymbolGraph
from .validation import is_test_file

logger = logging.getLogger(__name__)


_TEST_NAME_PATTERNS = [
    # JS/TS: utils.test.ts, utils.spec.tsx, utils.e2e.js
    re.compile(r"^(?P<base>.+?)\.(test|spec|e2e)\.(ts|tsx|js|jsx|mjs)$"),
    # Python: test_utils.py, utils_test.py
    re.compile(r"^test_(?P<base>.+?)\.py$"),
    re.compile(r"^(?P<base>.+?)_test\.py$"),
    # Go: foo_test.go
    re.compile(r"^(?P<base>.+?)_test\.go$"),
    # Rust: integration tests in tests/X.rs map to src/X.rs by convention
    re.compile(r"^(?P<base>.+?)\.rs$"),  # only when path includes /tests/
]

_SOURCE_EXT_BY_TEST_KIND = {
    "ts": [".ts", ".tsx"],
    "tsx": [".ts", ".tsx"],
    "js": [".js", ".jsx", ".mjs"],
    "jsx": [".js", ".jsx"],
    "mjs": [".mjs", ".js"],
    "py": [".py"],
    "go": [".go"],
    "rs": [".rs"],
}


@dataclass
class TestMap:
    """Test → symbol attribution.

    ``by_symbol`` is the primary output: for each ``(file, symbol)``,
    the test files that exercise it. Symbols not in the map have no
    known test coverage.

    ``by_file`` is a fallback view used when a symbol-level lookup
    misses (e.g., the symbol wasn't in the graph). Tests attached
    via filename match populate this directly.

    ``unmapped_tests`` lists test files that produced no mappings —
    typically smoke tests, e2e suites that drive the UI without
    importing source modules, or tests for files outside the graph.
    """
    by_symbol: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    by_file: dict[str, list[str]] = field(default_factory=dict)
    unmapped_tests: list[str] = field(default_factory=list)

    def tests_for_symbol(self, file: str, symbol: str) -> list[str]:
        return list(self.by_symbol.get((file, symbol), []))

    def tests_for_file(self, file: str) -> list[str]:
        """All tests touching the file at any granularity."""
        out: set[str] = set(self.by_file.get(file, []))
        for (f, _sym), tests in self.by_symbol.items():
            if f == file:
                out.update(tests)
        return sorted(out)

    def has_test(self, file: str, symbol: str | None = None) -> bool:
        if symbol is not None:
            if (file, symbol) in self.by_symbol:
                return True
        return file in self.by_file or any(
            f == file for (f, _) in self.by_symbol
        )


def apply_test_attribution(
    feature_map: "FeatureMap",  # type: ignore[name-defined]
    test_map: TestMap,
) -> None:
    """Populate ``Flow.test_files`` and ``SymbolAttribution.shared_with_flows``.

    For each flow, collect test files that exercise any of the flow's
    ``symbol_attributions``. Falls back to file-level test mapping when
    the symbol isn't in ``test_map.by_symbol``.

    For each ``SymbolAttribution`` (on flows AND on features), compute
    the set of OTHER flow names within the same feature that also
    reference at least one of the attribution's symbols. Stores the
    sorted list on ``shared_with_flows`` for UI badging.

    Mutates ``feature_map`` in place. Idempotent — running twice
    overwrites with the same data.
    """
    for feature in feature_map.features:
        # ── Per-flow: test_files ────────────────────────────────
        # Build symbol → flows index for this feature so we can
        # answer "which flows of this feature reference symbol X".
        symbol_to_flows: dict[tuple[str, str], list[str]] = {}
        for flow in feature.flows:
            for attr in flow.symbol_attributions:
                for sym in attr.symbols:
                    key = (attr.file_path, sym)
                    symbol_to_flows.setdefault(key, []).append(flow.name)

        for flow in feature.flows:
            tests: set[str] = set()
            for attr in flow.symbol_attributions:
                for sym in attr.symbols:
                    tests.update(test_map.tests_for_symbol(attr.file_path, sym))
                # Also include any file-level fallback tests
                tests.update(test_map.by_file.get(attr.file_path, []))
            flow.test_files = sorted(tests)
            flow.test_file_count = len(tests)

            # ── shared_with_flows on flow.symbol_attributions ──
            for attr in flow.symbol_attributions:
                shared: set[str] = set()
                for sym in attr.symbols:
                    other_flows = symbol_to_flows.get(
                        (attr.file_path, sym), [],
                    )
                    shared.update(o for o in other_flows if o != flow.name)
                attr.shared_with_flows = sorted(shared)

        # ── shared_with_flows on feature.shared_attributions ──
        for attr in feature.shared_attributions:
            shared: set[str] = set()
            for sym in attr.symbols:
                shared.update(symbol_to_flows.get((attr.file_path, sym), []))
            attr.shared_with_flows = sorted(shared)


def build_test_map(
    all_files: list[str],
    graph: SymbolGraph,
) -> TestMap:
    """Build a ``TestMap`` for the repo.

    Args:
        all_files: All source files (test + production). Tests are
            detected via ``validation.is_test_file``.
        graph: A pre-built ``SymbolGraph`` for the repo.

    The graph drives the import-based attribution. The full file
    list drives the filename fallback.
    """
    test_files = [f for f in all_files if is_test_file(f)]
    source_files = [f for f in all_files if not is_test_file(f)]
    source_set = set(source_files)

    tm = TestMap()
    pass1_hit_tests: set[str] = set()

    # ── Pass 1: import-based ───────────────────────────────────────
    for tf in test_files:
        edges = graph.imports_from(tf)
        if not edges:
            continue
        attached = False
        for edge in edges:
            target = edge.target_file
            if target not in source_set:
                continue
            sym = edge.target_symbol
            if sym in {"*", "@import"}:
                # Namespace / side-effect — attach to all exports
                exports = graph.exports.get(target, [])
                if exports:
                    for sr in exports:
                        _add_symbol(tm, target, sr.name, tf)
                        attached = True
                else:
                    _add_file(tm, target, tf)
                    attached = True
            else:
                _add_symbol(tm, target, sym, tf)
                attached = True
        if attached:
            pass1_hit_tests.add(tf)

    # ── Pass 2: filename fallback for tests pass 1 didn't catch ──
    for tf in test_files:
        if tf in pass1_hit_tests:
            continue
        target = _filename_match(tf, source_set)
        if target is None:
            tm.unmapped_tests.append(tf)
            continue
        _add_file(tm, target, tf)
        # Also attach to all exports if available so symbol-level
        # lookups hit too.
        for sr in graph.exports.get(target, []):
            _add_symbol(tm, target, sr.name, tf)

    logger.info(
        "test_mapper: %d test files → import-mapped %d, filename-mapped %d, unmapped %d",
        len(test_files),
        len(pass1_hit_tests),
        len(test_files) - len(pass1_hit_tests) - len(tm.unmapped_tests),
        len(tm.unmapped_tests),
    )
    return tm


# ── Internals ─────────────────────────────────────────────────────


def _add_symbol(tm: TestMap, file: str, symbol: str, test: str) -> None:
    key = (file, symbol)
    bucket = tm.by_symbol.setdefault(key, [])
    if test not in bucket:
        bucket.append(test)


def _add_file(tm: TestMap, file: str, test: str) -> None:
    bucket = tm.by_file.setdefault(file, [])
    if test not in bucket:
        bucket.append(test)


def _filename_match(test_path: str, source_set: set[str]) -> str | None:
    """Find the production file a test file most likely covers.

    Tries (in order):
      1. ``<dir>/<base>.<ext>`` — sibling source file
      2. Walk up: try parent dirs replacing ``tests/`` segment with ``src/``
      3. Glob for ``<base>.<ext>`` anywhere in the repo (last resort)
    """
    p = PurePosixPath(test_path)
    name = p.name
    parent = str(p.parent) if str(p.parent) != "." else ""

    base, exts = _strip_test_suffix(name)
    if base is None or not exts:
        return None

    # 1. Sibling: same directory, base + each candidate ext
    for ext in exts:
        candidate = f"{parent}/{base}{ext}" if parent else f"{base}{ext}"
        if candidate in source_set:
            return candidate

    # 2. Replace `/tests/` or `/__tests__/` with `/src/` in path
    for needle in ("/tests/", "/__tests__/", "/test/", "/spec/"):
        if needle in test_path:
            for ext in exts:
                candidate_path = test_path.replace(needle, "/src/")
                # Strip the original suffix and try base
                cand_dir = str(PurePosixPath(candidate_path).parent)
                candidate = f"{cand_dir}/{base}{ext}"
                if candidate in source_set:
                    return candidate

    # 3. Rust convention: tests/foo.rs → src/foo.rs at crate root
    if test_path.endswith(".rs"):
        if "/tests/" in test_path or test_path.startswith("tests/"):
            base_clean = base
            cand = f"src/{base_clean}.rs"
            if cand in source_set:
                return cand

    # 4. Last resort — basename match anywhere (might be wrong but better
    # than nothing for orphan tests). Only use for first hit.
    for ext in exts:
        target_name = f"{base}{ext}"
        for src in source_set:
            if Path(src).name == target_name:
                return src

    return None


def _strip_test_suffix(filename: str) -> tuple[str | None, list[str]]:
    """Return ``(base, [candidate_source_extensions])`` or ``(None, [])``.

    For ``utils.test.tsx`` → ``("utils", [".ts", ".tsx"])``.
    For ``test_auth.py`` → ``("auth", [".py"])``.
    """
    for pat in _TEST_NAME_PATTERNS:
        m = pat.match(filename)
        if not m:
            continue
        base = m.group("base")
        # Detect the ext from the original filename to choose candidates
        if filename.endswith(".rs"):
            return base, [".rs"]
        if filename.endswith(".go"):
            return base, [".go"]
        if filename.endswith(".py"):
            return base, [".py"]
        # JS/TS family
        for ext in (".tsx", ".ts", ".jsx", ".js", ".mjs"):
            if filename.endswith(ext):
                # Map jsx test → js source candidates etc.
                kind = ext.lstrip(".")
                return base, _SOURCE_EXT_BY_TEST_KIND.get(kind, [ext])
        return base, []
    return None, []
