"""Unit tests for ``faultline.analyzer.flow_tracer``."""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.analyzer.flow_tracer import (
    DEFAULT_DEPTH,
    TracedFlow,
    TracedParticipant,
    _resolve_target_symbols,
    trace_flow,
)
from faultline.analyzer.symbol_graph import (
    ImportEdge,
    SymbolGraph,
    build_symbol_graph,
)
from faultline.models.types import SymbolRange


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# ── _resolve_target_symbols ──────────────────────────────────────


class TestResolveTargetSymbols:
    def test_named_match(self):
        g = SymbolGraph(exports={
            "a.ts": [
                SymbolRange(name="X", start_line=1, end_line=5, kind="const"),
                SymbolRange(name="Y", start_line=10, end_line=15, kind="const"),
            ]
        })
        edge = ImportEdge(target_file="a.ts", target_symbol="X")
        out = _resolve_target_symbols(g, edge)
        assert len(out) == 1 and out[0].name == "X"

    def test_namespace_returns_all_exports(self):
        g = SymbolGraph(exports={
            "a.ts": [
                SymbolRange(name="X", start_line=1, end_line=5, kind="const"),
                SymbolRange(name="Y", start_line=10, end_line=15, kind="const"),
            ]
        })
        edge = ImportEdge(target_file="a.ts", target_symbol="*")
        names = {s.name for s in _resolve_target_symbols(g, edge)}
        assert names == {"X", "Y"}

    def test_side_effect_returns_empty(self):
        g = SymbolGraph(exports={"a.ts": [
            SymbolRange(name="X", start_line=1, end_line=5, kind="const"),
        ]})
        edge = ImportEdge(target_file="a.ts", target_symbol="@import")
        assert _resolve_target_symbols(g, edge) == []

    def test_unknown_symbol(self):
        g = SymbolGraph(exports={"a.ts": [
            SymbolRange(name="X", start_line=1, end_line=5, kind="const"),
        ]})
        edge = ImportEdge(target_file="a.ts", target_symbol="Missing")
        assert _resolve_target_symbols(g, edge) == []


# ── trace_flow on synthetic graphs ───────────────────────────────


class TestTraceFlowSynthetic:
    def _g(self) -> SymbolGraph:
        # entry → a → b
        # entry → c
        # entry → d (side-effect)
        g = SymbolGraph(
            exports={
                "entry.ts": [SymbolRange(name="run", start_line=1, end_line=10, kind="function")],
                "a.ts": [SymbolRange(name="aFn", start_line=1, end_line=5, kind="function")],
                "b.ts": [SymbolRange(name="bFn", start_line=1, end_line=5, kind="function")],
                "c.ts": [SymbolRange(name="cFn", start_line=1, end_line=5, kind="function")],
                "d.ts": [SymbolRange(name="init", start_line=1, end_line=2, kind="function")],
            },
            symbol_ranges={
                "entry.ts": [SymbolRange(name="run", start_line=1, end_line=10, kind="function")],
            },
            forward={
                "entry.ts": [
                    ImportEdge("a.ts", "aFn"),
                    ImportEdge("c.ts", "cFn"),
                    ImportEdge("d.ts", "@import"),
                ],
                "a.ts": [ImportEdge("b.ts", "bFn")],
            },
        )
        return g

    def test_basic_bfs(self):
        g = self._g()
        flow = trace_flow(g, "entry.ts", entry_line=5)
        files = {p.file for p in flow.participants}
        assert files == {"entry.ts", "a.ts", "b.ts", "c.ts", "d.ts"}

    def test_resolves_entry_symbol(self):
        g = self._g()
        flow = trace_flow(g, "entry.ts", entry_line=3)
        assert flow.entry_symbol is not None
        assert flow.entry_symbol.name == "run"

    def test_no_entry_line_walks_anyway(self):
        g = self._g()
        flow = trace_flow(g, "entry.ts", entry_line=0)
        assert flow.entry_symbol is None
        # BFS still runs — entry is not anchored to one symbol
        assert len(flow.participants) >= 4

    def test_depth_cap(self):
        g = self._g()
        flow = trace_flow(g, "entry.ts", entry_line=5, depth=1)
        files = {p.file for p in flow.participants}
        # depth=1 → entry + direct imports, but NOT b (which is 2 hops)
        assert "entry.ts" in files
        assert "a.ts" in files
        assert "b.ts" not in files

    def test_depth_zero_only_entry(self):
        g = self._g()
        flow = trace_flow(g, "entry.ts", entry_line=5, depth=0)
        assert {p.file for p in flow.participants} == {"entry.ts"}

    def test_side_effect_flag(self):
        g = self._g()
        flow = trace_flow(g, "entry.ts", entry_line=5)
        d = next(p for p in flow.participants if p.file == "d.ts")
        assert d.side_effect_only is True
        assert d.symbols == []

    def test_named_participant_has_target_symbol(self):
        g = self._g()
        flow = trace_flow(g, "entry.ts", entry_line=5)
        a = next(p for p in flow.participants if p.file == "a.ts")
        assert any(s.name == "aFn" for s in a.symbols)

    def test_namespace_import_pulls_all_exports(self):
        g = SymbolGraph(
            exports={
                "ns.ts": [
                    SymbolRange(name="X", start_line=1, end_line=5, kind="const"),
                    SymbolRange(name="Y", start_line=6, end_line=10, kind="const"),
                ],
                "entry.ts": [],
            },
            forward={"entry.ts": [ImportEdge("ns.ts", "*")]},
        )
        flow = trace_flow(g, "entry.ts")
        ns = next(p for p in flow.participants if p.file == "ns.ts")
        names = {s.name for s in ns.symbols}
        assert names == {"X", "Y"}

    def test_self_import_does_not_loop(self):
        g = SymbolGraph(
            exports={"x.ts": [
                SymbolRange(name="X", start_line=1, end_line=5, kind="const"),
            ]},
            forward={"x.ts": [ImportEdge("x.ts", "X")]},
        )
        flow = trace_flow(g, "x.ts")
        assert {p.file for p in flow.participants} == {"x.ts"}

    def test_diamond_dependency_dedups(self):
        # entry → a → c
        # entry → b → c
        g = SymbolGraph(
            exports={
                "entry.ts": [], "a.ts": [], "b.ts": [],
                "c.ts": [SymbolRange(name="C", start_line=1, end_line=5, kind="const")],
            },
            forward={
                "entry.ts": [ImportEdge("a.ts", "*"), ImportEdge("b.ts", "*")],
                "a.ts": [ImportEdge("c.ts", "C")],
                "b.ts": [ImportEdge("c.ts", "C")],
            },
        )
        flow = trace_flow(g, "entry.ts")
        c_participants = [p for p in flow.participants if p.file == "c.ts"]
        assert len(c_participants) == 1
        # Only one C symbol despite being reached twice
        assert len([s for s in c_participants[0].symbols if s.name == "C"]) == 1

    def test_max_participants_cap(self):
        # Build a wide fan-out: entry imports 100 distinct files
        exports = {f"f{i}.ts": [
            SymbolRange(name=f"f{i}Fn", start_line=1, end_line=2, kind="function"),
        ] for i in range(100)}
        exports["entry.ts"] = []
        forward = {"entry.ts": [
            ImportEdge(f"f{i}.ts", f"f{i}Fn") for i in range(100)
        ]}
        g = SymbolGraph(exports=exports, forward=forward)
        flow = trace_flow(g, "entry.ts", max_participants=10)
        assert len(flow.participants) <= 10

    def test_unknown_entry_file(self):
        g = SymbolGraph()
        flow = trace_flow(g, "missing.ts")
        # entry is recorded as one participant even though graph
        # doesn't know it — keeps Sprint 7 stable on partial graphs
        assert len(flow.participants) == 1
        assert flow.participants[0].file == "missing.ts"
        assert flow.participants[0].symbols == []


# ── trace_flow on real fixture repo ─────────────────────────────


class TestTraceFlowFixtureRepo:
    def test_traces_through_two_hops(self, tmp_path: Path):
        _write(tmp_path, "lib/api.ts",
               "export async function fetchUser() { return {}; }\n")
        _write(tmp_path, "store/user-store.ts",
               "import { fetchUser } from '../lib/api';\n"
               "export function useUserStore() { return fetchUser; }\n")
        _write(tmp_path, "components/UserCard.tsx",
               "import { useUserStore } from '../store/user-store';\n"
               "export default function UserCard() {\n"
               "  return useUserStore();\n"
               "}\n")
        _write(tmp_path, "routes/user.tsx",
               "import UserCard from '../components/UserCard';\n"
               "export default function UserRoute() { return <UserCard />; }\n")
        files = [
            "lib/api.ts", "store/user-store.ts",
            "components/UserCard.tsx", "routes/user.tsx",
        ]
        graph = build_symbol_graph(tmp_path, files)
        flow = trace_flow(graph, "routes/user.tsx", entry_line=2, depth=3)
        files_reached = {p.file for p in flow.participants}
        assert files_reached == set(files)

    def test_depth_limits_real_repo(self, tmp_path: Path):
        _write(tmp_path, "a.ts", "export const A = 1;\n")
        _write(tmp_path, "b.ts", "import { A } from './a';\nexport const B = A;\n")
        _write(tmp_path, "c.ts", "import { B } from './b';\nexport const C = B;\n")
        _write(tmp_path, "d.ts", "import { C } from './c';\nexport const D = C;\n")
        graph = build_symbol_graph(tmp_path, ["a.ts", "b.ts", "c.ts", "d.ts"])
        flow = trace_flow(graph, "d.ts", entry_line=2, depth=2)
        files = {p.file for p in flow.participants}
        # depth=2 from d → c (depth 1) → b (depth 2). a is depth 3.
        assert "d.ts" in files
        assert "c.ts" in files
        assert "b.ts" in files
        assert "a.ts" not in files


# ── trace_flow_callgraph orchestrator ────────────────────────────


class TestTraceFlowCallgraph:
    def _build_documenso_like_repo(self, tmp_path: Path) -> None:
        # API server route handler
        _write(tmp_path, "apps/web/app/api/users/route.ts",
               "export async function POST() { return new Response(); }\n")
        # State store
        _write(tmp_path, "apps/web/stores/user-store.ts",
               "import { create } from 'zustand';\n"
               "export const useUserStore = create(() => ({}));\n")
        # UI component
        _write(tmp_path, "apps/web/components/UserForm.tsx",
               "import { useUserStore } from '../stores/user-store';\n"
               "export default function UserForm() {\n"
               "  return useUserStore();\n"
               "}\n")
        # Schema
        _write(tmp_path, "packages/prisma/schema.prisma",
               "model User { id Int @id }\n")
        # Entry point — a Remix route component
        _write(tmp_path, "apps/web/app/users/page.tsx",
               "import UserForm from '../../components/UserForm';\n"
               "export default function Page() {\n"
               "  return <UserForm />;\n"
               "}\n")

    def test_orchestrator_traces_all_flows(self, tmp_path: Path):
        from faultline.analyzer.flow_tracer import trace_flow_callgraph
        from faultline.llm.sonnet_scanner import DeepScanResult

        self._build_documenso_like_repo(tmp_path)

        result = DeepScanResult(
            features={"users": [
                "apps/web/app/users/page.tsx",
                "apps/web/components/UserForm.tsx",
                "apps/web/stores/user-store.ts",
                "apps/web/app/api/users/route.ts",
                "packages/prisma/schema.prisma",
            ]},
            flows={"users": ["create-user"]},
            flow_descriptions={"users": {
                "create-user": "User submits form. (entry: apps/web/app/users/page.tsx:2)",
            }},
        )

        trace = trace_flow_callgraph(result, tmp_path, depth=3)
        assert "users" in trace
        assert "create-user" in trace["users"]
        participants = trace["users"]["create-user"]
        files = {p.file for p in participants}

        # Entry + reachable callees
        assert "apps/web/app/users/page.tsx" in files
        assert "apps/web/components/UserForm.tsx" in files
        assert "apps/web/stores/user-store.ts" in files

        # Layers populated by the classifier
        layers = {p.layer for p in participants}
        assert "ui" in layers
        assert "state" in layers

    def test_no_flows_returns_empty(self, tmp_path: Path):
        from faultline.analyzer.flow_tracer import trace_flow_callgraph
        from faultline.llm.sonnet_scanner import DeepScanResult
        result = DeepScanResult(features={"x": []}, flows={})
        assert trace_flow_callgraph(result, tmp_path) == {}

    def test_skips_flows_without_entry_trail(self, tmp_path: Path):
        from faultline.analyzer.flow_tracer import trace_flow_callgraph
        from faultline.llm.sonnet_scanner import DeepScanResult
        _write(tmp_path, "x.ts", "export const X = 1;\n")
        result = DeepScanResult(
            features={"f": ["x.ts"]},
            flows={"f": ["bare-flow"]},
            flow_descriptions={"f": {"bare-flow": "no entry suffix"}},
        )
        trace = trace_flow_callgraph(result, tmp_path)
        # Flow without (entry: ...) trail not included
        assert "f" not in trace
