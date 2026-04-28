"""Unit tests for ``faultline.analyzer.symbol_graph``."""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.analyzer.symbol_graph import (
    ImportEdge,
    SymbolGraph,
    build_symbol_graph,
)


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# ── SymbolGraph plumbing ──────────────────────────────────────────


class TestSymbolGraphAccessors:
    def test_imports_from_returns_list(self):
        g = SymbolGraph(forward={
            "a.ts": [ImportEdge("b.ts", "X")],
        })
        assert g.imports_from("a.ts") == [ImportEdge("b.ts", "X")]
        assert g.imports_from("missing.ts") == []

    def test_callers_of(self):
        g = SymbolGraph(reverse={
            "b.ts": [ImportEdge("a.ts", "X")],
        })
        assert g.callers_of("b.ts") == [ImportEdge("a.ts", "X")]
        assert g.callers_of("missing.ts") == []


# ── find_enclosing_symbol ─────────────────────────────────────────


class TestFindEnclosingSymbol:
    def test_returns_smallest_containing_range(self):
        from faultline.models.types import SymbolRange
        g = SymbolGraph(symbol_ranges={
            "x.ts": [
                SymbolRange(name="OuterClass", start_line=1, end_line=200, kind="class"),
                SymbolRange(name="innerFunc", start_line=20, end_line=40, kind="function"),
            ],
        })
        # Line 25 is in both — should pick innerFunc (smaller)
        result = g.find_enclosing_symbol("x.ts", 25)
        assert result is not None and result.name == "innerFunc"

    def test_no_match(self):
        from faultline.models.types import SymbolRange
        g = SymbolGraph(symbol_ranges={
            "x.ts": [SymbolRange(name="f", start_line=10, end_line=20, kind="function")],
        })
        assert g.find_enclosing_symbol("x.ts", 5) is None

    def test_missing_file(self):
        g = SymbolGraph()
        assert g.find_enclosing_symbol("nope.ts", 1) is None


# ── build_symbol_graph (real fixture repos) ───────────────────────


class TestBuildGraph:
    def test_single_file_no_imports(self, tmp_path: Path):
        _write(tmp_path, "x.ts", "export const x = 1;\n")
        g = build_symbol_graph(tmp_path, ["x.ts"])
        assert "x.ts" in g.exports
        assert g.exports["x.ts"][0].name == "x"
        assert g.forward.get("x.ts") in (None, [])
        assert g.reverse.get("x.ts") in (None, [])

    def test_relative_import_resolves(self, tmp_path: Path):
        _write(tmp_path, "src/a.ts", "export const A = 1;\n")
        _write(tmp_path, "src/b.ts", "import { A } from './a';\nexport const B = A;\n")
        g = build_symbol_graph(tmp_path, ["src/a.ts", "src/b.ts"])
        # b.ts forward → a.ts:A
        edges = g.forward.get("src/b.ts") or []
        assert any(e.target_file == "src/a.ts" and e.target_symbol == "A" for e in edges)
        # a.ts reverse → b.ts (caller)
        callers = g.reverse.get("src/a.ts") or []
        assert any(c.target_file == "src/b.ts" for c in callers)

    def test_named_import_with_alias(self, tmp_path: Path):
        _write(tmp_path, "a.ts", "export const A = 1;\n")
        _write(tmp_path, "b.ts", "import { A as Renamed } from './a';\nexport const B = Renamed;\n")
        g = build_symbol_graph(tmp_path, ["a.ts", "b.ts"])
        edges = g.forward.get("b.ts") or []
        # Original name is preserved in target_symbol
        assert any(e.target_symbol == "A" for e in edges)

    def test_namespace_import_uses_star(self, tmp_path: Path):
        _write(tmp_path, "a.ts", "export const A = 1;\n")
        _write(tmp_path, "b.ts", "import * as ns from './a';\nexport const B = ns.A;\n")
        g = build_symbol_graph(tmp_path, ["a.ts", "b.ts"])
        edges = g.forward.get("b.ts") or []
        assert any(e.target_symbol == "*" for e in edges)

    def test_default_import(self, tmp_path: Path):
        _write(tmp_path, "comp.tsx", "export default function Comp() {}\n")
        _write(tmp_path, "page.tsx",
               "import Comp from './comp';\nexport default () => <Comp />;\n")
        g = build_symbol_graph(tmp_path, ["comp.tsx", "page.tsx"])
        edges = g.forward.get("page.tsx") or []
        assert any(
            e.target_file == "comp.tsx" and e.target_symbol == "Comp"
            for e in edges
        )

    def test_default_plus_named_import(self, tmp_path: Path):
        _write(tmp_path, "lib.ts",
               "export default function fmt() {}\nexport const X = 1;\n")
        _write(tmp_path, "user.ts",
               "import fmt, { X } from './lib';\nexport const z = fmt(X);\n")
        g = build_symbol_graph(tmp_path, ["lib.ts", "user.ts"])
        edges = g.forward.get("user.ts") or []
        symbols = {e.target_symbol for e in edges if e.target_file == "lib.ts"}
        assert {"fmt", "X"}.issubset(symbols)

    def test_side_effect_import(self, tmp_path: Path):
        _write(tmp_path, "init.ts", "console.log('init');\n")
        _write(tmp_path, "main.ts", "import './init';\nexport const x = 1;\n")
        g = build_symbol_graph(tmp_path, ["init.ts", "main.ts"])
        edges = g.forward.get("main.ts") or []
        # @import sentinel records the side-effect edge so BFS can traverse
        assert any(
            e.target_file == "init.ts" and e.target_symbol == "@import"
            for e in edges
        )

    def test_named_reexport(self, tmp_path: Path):
        # Barrel pattern: index.ts re-exports symbols from sibling
        # files. A consumer of the barrel should reach the original.
        _write(tmp_path, "auth/login.ts", "export const login = () => {};\n")
        _write(tmp_path, "auth/index.ts", "export { login } from './login';\n")
        _write(tmp_path, "consumer.ts",
               "import { login } from './auth';\nexport const x = login;\n")
        g = build_symbol_graph(tmp_path, [
            "auth/login.ts", "auth/index.ts", "consumer.ts",
        ])
        # auth/index.ts should have an edge to auth/login.ts via the
        # re-export.
        edges = g.forward.get("auth/index.ts") or []
        assert any(
            e.target_file == "auth/login.ts" and e.target_symbol == "login"
            for e in edges
        )

    def test_named_reexport_with_alias(self, tmp_path: Path):
        _write(tmp_path, "a.ts", "export const A = 1;\n")
        _write(tmp_path, "barrel.ts",
               "export { A as Renamed } from './a';\n")
        g = build_symbol_graph(tmp_path, ["a.ts", "barrel.ts"])
        edges = g.forward.get("barrel.ts") or []
        # Original name preserved (we record the source symbol)
        assert any(e.target_symbol == "A" for e in edges)

    def test_star_reexport(self, tmp_path: Path):
        _write(tmp_path, "lib.ts",
               "export const X = 1;\nexport const Y = 2;\n")
        _write(tmp_path, "barrel.ts", "export * from './lib';\n")
        g = build_symbol_graph(tmp_path, ["lib.ts", "barrel.ts"])
        edges = g.forward.get("barrel.ts") or []
        # Star re-export → namespace edge
        assert any(
            e.target_file == "lib.ts" and e.target_symbol == "*"
            for e in edges
        )

    def test_star_reexport_with_namespace_name(self, tmp_path: Path):
        _write(tmp_path, "ns.ts", "export const X = 1;\n")
        _write(tmp_path, "barrel.ts", "export * as theNs from './ns';\n")
        g = build_symbol_graph(tmp_path, ["ns.ts", "barrel.ts"])
        edges = g.forward.get("barrel.ts") or []
        assert any(e.target_file == "ns.ts" for e in edges)

    def test_dynamic_import(self, tmp_path: Path):
        _write(tmp_path, "lazy.ts", "export default function Lazy() {}\n")
        _write(tmp_path, "loader.ts",
               "export const loadLazy = () => import('./lazy');\n")
        g = build_symbol_graph(tmp_path, ["lazy.ts", "loader.ts"])
        edges = g.forward.get("loader.ts") or []
        # Dynamic imports captured as @import (side-effect-style)
        assert any(
            e.target_file == "lazy.ts" and e.target_symbol == "@import"
            for e in edges
        )

    def test_http_edge_fetch(self, tmp_path: Path):
        # Improvement #6: build_symbol_graph layers HTTP edges
        # (client → server) on top of static imports.
        _write(tmp_path, "server/api.ts",
               "app.post('/api/documents', createDocument);\n")
        _write(tmp_path, "client/upload.tsx",
               "const r = await fetch('/api/documents', {method: 'POST'});\n")
        g = build_symbol_graph(tmp_path, [
            "server/api.ts", "client/upload.tsx",
        ])
        edges = g.forward.get("client/upload.tsx") or []
        assert any(
            e.target_file == "server/api.ts" and e.target_symbol == "@http"
            for e in edges
        )
        # Reverse map populated too
        rev = g.reverse.get("server/api.ts") or []
        assert any(
            e.target_file == "client/upload.tsx" and e.target_symbol == "@http"
            for e in rev
        )

    def test_http_edge_trpc(self, tmp_path: Path):
        _write(tmp_path, "trpc/router.ts",
               "export const documentsRouter = router({\n"
               "  send: protectedProcedure.input(z.any()).mutation(({input}) => {}),\n"
               "});\n")
        _write(tmp_path, "ui/SendButton.tsx",
               "const send = trpc.documents.send.useMutation();\n")
        g = build_symbol_graph(tmp_path, [
            "trpc/router.ts", "ui/SendButton.tsx",
        ])
        edges = g.forward.get("ui/SendButton.tsx") or []
        assert any(
            e.target_file == "trpc/router.ts" and e.target_symbol == "@http"
            for e in edges
        )

    def test_http_edges_can_be_disabled(self, tmp_path: Path):
        _write(tmp_path, "server/api.ts",
               "app.get('/api/x', h);\n")
        _write(tmp_path, "client/x.ts",
               "fetch('/api/x');\n")
        g = build_symbol_graph(
            tmp_path, ["server/api.ts", "client/x.ts"],
            include_http_edges=False,
        )
        # Pure static-import view — no HTTP edges
        for edges in g.forward.values():
            assert not any(e.target_symbol == "@http" for e in edges)

    def test_http_edge_self_skipped(self, tmp_path: Path):
        # Single Next.js app router file declaring both server and
        # consumed by itself — self-edge would create a BFS loop.
        _write(tmp_path, "app/api/users/route.ts",
               "export async function GET() {\n"
               "  await fetch('/api/users');\n"
               "  return new Response();\n"
               "}\n")
        g = build_symbol_graph(tmp_path, ["app/api/users/route.ts"])
        edges = g.forward.get("app/api/users/route.ts") or []
        # No self-loop
        assert all(e.target_file != "app/api/users/route.ts" for e in edges)

    def test_third_party_import_skipped(self, tmp_path: Path):
        _write(tmp_path, "x.ts",
               "import express from 'express';\nexport const app = express();\n")
        g = build_symbol_graph(tmp_path, ["x.ts"])
        # 'express' is bare → not resolved → no forward edge
        assert g.forward.get("x.ts") in (None, [])

    def test_multiple_imports_same_file(self, tmp_path: Path):
        _write(tmp_path, "lib.ts", "export const A = 1;\nexport const B = 2;\nexport const C = 3;\n")
        _write(tmp_path, "user.ts",
               "import { A, B } from './lib';\nimport { C as Cee } from './lib';\n"
               "export const z = A + B + Cee;\n")
        g = build_symbol_graph(tmp_path, ["lib.ts", "user.ts"])
        edges = g.forward.get("user.ts") or []
        symbols = {e.target_symbol for e in edges if e.target_file == "lib.ts"}
        assert {"A", "B", "C"}.issubset(symbols)

    def test_path_alias_via_tsconfig(self, tmp_path: Path):
        _write(tmp_path, "tsconfig.json", """{
            "compilerOptions": {
                "baseUrl": ".",
                "paths": {"@/*": ["src/*"]}
            }
        }""")
        _write(tmp_path, "src/auth/login.ts", "export const login = () => {};\n")
        _write(tmp_path, "src/app/page.tsx", "import { login } from '@/auth/login';\nexport default () => login;\n")
        g = build_symbol_graph(tmp_path, [
            "src/auth/login.ts", "src/app/page.tsx", "tsconfig.json",
        ])
        edges = g.forward.get("src/app/page.tsx") or []
        assert any(e.target_file == "src/auth/login.ts" for e in edges)

    def test_unreadable_file_does_not_crash(self, tmp_path: Path):
        # File listed but doesn't exist on disk
        g = build_symbol_graph(tmp_path, ["does-not-exist.ts"])
        assert g.exports == {}
        assert g.forward == {}

    def test_python_files_skipped_in_forward(self, tmp_path: Path):
        _write(tmp_path, "a.py", "from b import x\n")
        _write(tmp_path, "b.py", "x = 1\n")
        g = build_symbol_graph(tmp_path, ["a.py", "b.py"])
        # Python isn't TS/JS — forward map omits a.py
        assert "a.py" not in g.forward

    def test_export_ranges_carried_over(self, tmp_path: Path):
        _write(tmp_path, "x.ts",
               "export function login() {\n  return 1;\n}\n"
               "export const xyz = 5;\n")
        g = build_symbol_graph(tmp_path, ["x.ts"])
        names = {r.name for r in g.exports.get("x.ts", [])}
        assert "login" in names
        assert "xyz" in names

    def test_self_import_does_not_explode(self, tmp_path: Path):
        # Pathological: file imports from itself (rare but happens)
        _write(tmp_path, "x.ts", "import { Y } from './x';\nexport const Y = 1;\n")
        g = build_symbol_graph(tmp_path, ["x.ts"])
        # Either resolves to itself or doesn't — must not raise
        assert isinstance(g.forward.get("x.ts", []), list)


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_input(self, tmp_path: Path):
        g = build_symbol_graph(tmp_path, [])
        assert g.exports == {}
        assert g.forward == {}
        assert g.reverse == {}

    def test_files_with_no_exports(self, tmp_path: Path):
        _write(tmp_path, "x.ts", "const private_var = 1;\n")
        g = build_symbol_graph(tmp_path, ["x.ts"])
        # No exports recorded, no error
        assert g.exports.get("x.ts") in (None, [])

    def test_relative_dot_dot_above_root(self, tmp_path: Path):
        _write(tmp_path, "a.ts", "import { X } from '../outside';\n")
        g = build_symbol_graph(tmp_path, ["a.ts"])
        # Resolves above root → no edge
        assert g.forward.get("a.ts") in (None, [])

    def test_nonexistent_local_import_dropped(self, tmp_path: Path):
        _write(tmp_path, "a.ts", "import { Y } from './missing';\n")
        g = build_symbol_graph(tmp_path, ["a.ts"])
        # Target file not in source list → no edge
        assert g.forward.get("a.ts") in (None, [])
