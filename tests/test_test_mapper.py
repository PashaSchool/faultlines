"""Tests for analyzer.test_mapper — test→symbol mapping (Sprint 1 Day 3)."""

from __future__ import annotations

from faultline.analyzer.symbol_graph import (
    ImportEdge,
    SymbolGraph,
)
from faultline.analyzer.test_mapper import (
    _filename_match,
    _strip_test_suffix,
    build_test_map,
)
from faultline.models.types import SymbolRange


def _graph(
    *,
    forward: dict[str, list[ImportEdge]] | None = None,
    exports: dict[str, list[SymbolRange]] | None = None,
) -> SymbolGraph:
    return SymbolGraph(
        forward=forward or {},
        exports=exports or {},
    )


def _sym(name: str, start: int = 1, end: int = 10) -> SymbolRange:
    return SymbolRange(name=name, start_line=start, end_line=end)


# ── Helpers tests ────────────────────────────────────────────────


class TestStripTestSuffix:
    def test_jest_test(self):
        base, exts = _strip_test_suffix("utils.test.ts")
        assert base == "utils"
        assert ".ts" in exts and ".tsx" in exts

    def test_spec_tsx(self):
        base, exts = _strip_test_suffix("Button.spec.tsx")
        assert base == "Button"
        assert ".tsx" in exts

    def test_python_test_underscore_prefix(self):
        base, exts = _strip_test_suffix("test_auth.py")
        assert base == "auth"
        assert exts == [".py"]

    def test_python_test_underscore_suffix(self):
        base, exts = _strip_test_suffix("auth_test.py")
        assert base == "auth"
        assert exts == [".py"]

    def test_go(self):
        base, exts = _strip_test_suffix("auth_test.go")
        assert base == "auth"
        assert exts == [".go"]

    def test_non_test_file(self):
        base, exts = _strip_test_suffix("README.md")
        # README matches none of the patterns → (None, [])
        assert base is None or exts == []


class TestFilenameMatch:
    def test_sibling_match(self):
        sources = {"src/utils.ts", "src/auth.ts"}
        assert _filename_match("src/utils.test.ts", sources) == "src/utils.ts"

    def test_python_underscore_prefix(self):
        sources = {"app/auth.py", "app/utils.py"}
        assert _filename_match("app/test_auth.py", sources) == "app/auth.py"

    def test_go_test_sibling(self):
        sources = {"pkg/auth.go"}
        assert _filename_match("pkg/auth_test.go", sources) == "pkg/auth.go"

    def test_rust_tests_dir_to_src(self):
        sources = {"src/parser.rs"}
        assert _filename_match("tests/parser.rs", sources) == "src/parser.rs"

    def test_tests_dir_to_src_dir(self):
        sources = {"src/components/Button.tsx"}
        m = _filename_match("src/components/__tests__/Button.test.tsx", sources)
        # Falls through filename-match basename last-resort; correct file found
        assert m == "src/components/Button.tsx"

    def test_no_match(self):
        sources = {"src/utils.ts"}
        assert _filename_match("orphan/auth.test.ts", sources) is None


# ── End-to-end mapping tests ─────────────────────────────────────


class TestImportBased:
    def test_named_import_attaches_to_symbol(self):
        # auth.test.ts imports { login } from "./auth"
        graph = _graph(
            forward={
                "src/auth.test.ts": [ImportEdge(target_file="src/auth.ts", target_symbol="login")],
            },
            exports={"src/auth.ts": [_sym("login"), _sym("logout")]},
        )
        all_files = ["src/auth.ts", "src/auth.test.ts"]
        tm = build_test_map(all_files, graph)
        assert tm.tests_for_symbol("src/auth.ts", "login") == ["src/auth.test.ts"]
        # Logout NOT attached — only login was imported
        assert tm.tests_for_symbol("src/auth.ts", "logout") == []

    def test_namespace_import_attaches_to_all_exports(self):
        graph = _graph(
            forward={
                "src/utils.test.ts": [ImportEdge(target_file="src/utils.ts", target_symbol="*")],
            },
            exports={"src/utils.ts": [_sym("formatDate"), _sym("parseDate"), _sym("addDays")]},
        )
        all_files = ["src/utils.ts", "src/utils.test.ts"]
        tm = build_test_map(all_files, graph)
        for sym in ("formatDate", "parseDate", "addDays"):
            assert tm.tests_for_symbol("src/utils.ts", sym) == ["src/utils.test.ts"]

    def test_multiple_tests_one_symbol(self):
        graph = _graph(
            forward={
                "tests/a.test.ts": [ImportEdge(target_file="src/auth.ts", target_symbol="login")],
                "tests/b.test.ts": [ImportEdge(target_file="src/auth.ts", target_symbol="login")],
            },
            exports={"src/auth.ts": [_sym("login")]},
        )
        all_files = ["src/auth.ts", "tests/a.test.ts", "tests/b.test.ts"]
        tm = build_test_map(all_files, graph)
        tests = tm.tests_for_symbol("src/auth.ts", "login")
        assert set(tests) == {"tests/a.test.ts", "tests/b.test.ts"}

    def test_one_test_multiple_symbols(self):
        graph = _graph(
            forward={
                "src/auth.test.ts": [
                    ImportEdge(target_file="src/auth.ts", target_symbol="login"),
                    ImportEdge(target_file="src/auth.ts", target_symbol="logout"),
                ],
            },
            exports={"src/auth.ts": [_sym("login"), _sym("logout")]},
        )
        all_files = ["src/auth.ts", "src/auth.test.ts"]
        tm = build_test_map(all_files, graph)
        assert tm.tests_for_symbol("src/auth.ts", "login") == ["src/auth.test.ts"]
        assert tm.tests_for_symbol("src/auth.ts", "logout") == ["src/auth.test.ts"]


class TestFilenameFallback:
    def test_falls_back_when_no_imports(self):
        # No edges in graph for the test file → filename match
        graph = _graph(
            exports={"src/auth.ts": [_sym("login"), _sym("logout")]},
        )
        all_files = ["src/auth.ts", "src/auth.test.ts"]
        tm = build_test_map(all_files, graph)
        # Symbol-level attach via exports
        assert tm.tests_for_symbol("src/auth.ts", "login") == ["src/auth.test.ts"]
        assert tm.tests_for_symbol("src/auth.ts", "logout") == ["src/auth.test.ts"]
        # File-level too
        assert "src/auth.test.ts" in tm.tests_for_file("src/auth.ts")

    def test_orphan_test_in_unmapped(self):
        graph = _graph()
        all_files = ["src/auth.ts", "tests/standalone.test.ts"]
        tm = build_test_map(all_files, graph)
        # No name match for "standalone" → unmapped
        assert "tests/standalone.test.ts" in tm.unmapped_tests

    def test_python_test_no_imports(self):
        graph = _graph(
            exports={"app/billing.py": [_sym("charge"), _sym("refund")]},
        )
        all_files = ["app/billing.py", "tests/test_billing.py"]
        tm = build_test_map(all_files, graph)
        assert tm.tests_for_symbol("app/billing.py", "charge") == ["tests/test_billing.py"]


class TestEndToEnd:
    def test_mixed_pass1_pass2(self):
        # Two test files: one has imports (precise), one falls back to filename
        graph = _graph(
            forward={
                "src/auth.test.ts": [
                    ImportEdge(target_file="src/auth.ts", target_symbol="login"),
                ],
            },
            exports={
                "src/auth.ts": [_sym("login"), _sym("logout")],
                "src/utils.ts": [_sym("formatDate")],
            },
        )
        all_files = [
            "src/auth.ts",
            "src/utils.ts",
            "src/auth.test.ts",  # pass 1 — only login
            "src/utils.test.ts",  # pass 2 — all utils exports
        ]
        tm = build_test_map(all_files, graph)
        # Pass 1: only login attached for auth
        assert tm.tests_for_symbol("src/auth.ts", "login") == ["src/auth.test.ts"]
        assert tm.tests_for_symbol("src/auth.ts", "logout") == []
        # Pass 2: filename fallback covered all utils
        assert tm.tests_for_symbol("src/utils.ts", "formatDate") == ["src/utils.test.ts"]

    def test_has_test_helper(self):
        graph = _graph(
            exports={"src/auth.ts": [_sym("login")]},
        )
        all_files = ["src/auth.ts", "src/auth.test.ts"]
        tm = build_test_map(all_files, graph)
        assert tm.has_test("src/auth.ts", "login")
        assert not tm.has_test("src/utils.ts", "anything")
