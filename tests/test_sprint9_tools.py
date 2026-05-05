"""Sprint 9 Day 1 — unit tests for the import-graph tools.

These tools let the agentic classifier investigate aggregator
candidates by following imports, finding consumers, and summarizing
features. No real LLM call is exercised here; the agent itself is
Day 2 work.
"""

from __future__ import annotations

from faultline.analyzer.symbol_graph import ImportEdge, SymbolGraph
from faultline.llm.sonnet_scanner import DeepScanResult
from faultline.llm.tools import (
    consumers_of,
    feature_summary,
    imports_of,
)


def _graph_with(
    forward: dict[str, list[str]] | None = None,
    reverse: dict[str, list[str]] | None = None,
) -> SymbolGraph:
    """Build a SymbolGraph populated only with the directed edges
    we care about for tool tests."""
    g = SymbolGraph()
    if forward:
        for src, targets in forward.items():
            g.forward[src] = [
                ImportEdge(target_file=t, target_symbol="*") for t in targets
            ]
    if reverse:
        for tgt, importers in reverse.items():
            g.reverse[tgt] = [
                ImportEdge(target_file=imp, target_symbol="*") for imp in importers
            ]
    return g


# ── imports_of ───────────────────────────────────────────────────────


class TestImportsOf:
    def test_returns_outgoing_edges(self):
        g = _graph_with(forward={
            "src/auth/login.tsx": [
                "packages/api-types/dto/auth/login.dto.ts",
                "packages/shared-ui/Button.tsx",
            ],
        })
        result = imports_of(g, "src/auth/login.tsx")
        assert "packages/api-types/dto/auth/login.dto.ts" in result
        assert "packages/shared-ui/Button.tsx" in result
        assert "2 edges" in result

    def test_leaf_file_reports_no_outgoing(self):
        g = _graph_with()  # empty graph
        result = imports_of(g, "packages/api-types/dto/types.ts")
        assert "no outgoing" in result.lower()
        assert "leaf file" in result.lower()

    def test_missing_graph_returns_error_string(self):
        result = imports_of(None, "any.ts")
        assert result.startswith("ERROR")
        assert "symbol graph" in result.lower()

    def test_missing_path_returns_error_string(self):
        g = _graph_with()
        assert imports_of(g, "").startswith("ERROR")

    def test_dedupes_target_files(self):
        # Same target imported via two symbols → one line out
        g = SymbolGraph()
        g.forward["src/x.ts"] = [
            ImportEdge(target_file="lib/y.ts", target_symbol="A"),
            ImportEdge(target_file="lib/y.ts", target_symbol="B"),
        ]
        result = imports_of(g, "src/x.ts")
        assert result.count("→ lib/y.ts") == 1


# ── consumers_of ─────────────────────────────────────────────────────


class TestConsumersOf:
    def test_returns_incoming_edges_with_arrow_direction(self):
        # ← arrow means "this file is imported BY"
        g = _graph_with(reverse={
            "packages/api-types/dto/user.dto.ts": [
                "src/auth/login.tsx",
                "src/billing/plan.tsx",
                "src/settings/profile.tsx",
            ],
        })
        result = consumers_of(g, "packages/api-types/dto/user.dto.ts")
        assert "src/auth/login.tsx" in result
        assert "src/billing/plan.tsx" in result
        assert "src/settings/profile.tsx" in result
        assert "← " in result  # incoming arrow direction
        assert "3 edges" in result

    def test_unused_file_explicit_message(self):
        g = _graph_with()
        result = consumers_of(g, "packages/dto/orphan.ts")
        assert "no consumers" in result.lower()
        assert "unused" in result.lower() or "untracked" in result.lower()

    def test_missing_graph_returns_error_string(self):
        assert consumers_of(None, "x.ts").startswith("ERROR")

    def test_dedupes_importers(self):
        g = SymbolGraph()
        g.reverse["pkg/dto/x.ts"] = [
            ImportEdge(target_file="src/auth/login.tsx", target_symbol="A"),
            ImportEdge(target_file="src/auth/login.tsx", target_symbol="B"),
        ]
        result = consumers_of(g, "pkg/dto/x.ts")
        assert result.count("← src/auth/login.tsx") == 1


# ── feature_summary ──────────────────────────────────────────────────


class TestFeatureSummary:
    def _ds(self, **kwargs) -> DeepScanResult:
        return DeepScanResult(
            features=kwargs.get("features", {}),
            flows=kwargs.get("flows", {}),
            descriptions=kwargs.get("descriptions", {}),
        )

    def test_summary_for_real_feature(self):
        result = self._ds(
            features={"Authentication": [
                "src/auth/login.tsx", "src/auth/signup.tsx",
            ]},
            flows={"Authentication": ["log-in", "sign-up"]},
            descriptions={"Authentication": "User login and registration."},
        )
        out = feature_summary(result, "Authentication")
        assert "Authentication" in out
        assert "Files: 2" in out
        assert "Flows: 2" in out
        assert "User login and registration" in out
        assert "src/auth/login.tsx" in out
        assert "log-in" in out

    def test_summary_for_missing_feature(self):
        result = self._ds(features={"Auth": ["a.ts"]})
        out = feature_summary(result, "DoesNotExist")
        assert out.startswith("ERROR")
        assert "no feature" in out.lower()

    def test_summary_caps_path_samples(self):
        # 20 paths, but only first 8 shown — keeps prompt budget bounded
        result = self._ds(features={
            "Big": [f"src/file_{i}.ts" for i in range(20)],
        })
        out = feature_summary(result, "Big")
        assert "src/file_0.ts" in out
        assert "src/file_7.ts" in out
        assert "src/file_8.ts" not in out
        assert "Files: 20" in out  # full count still reported

    def test_missing_scan_result_returns_error_string(self):
        assert feature_summary(None, "Auth").startswith("ERROR")
