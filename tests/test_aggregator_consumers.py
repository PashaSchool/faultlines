"""Sprint 8 Day 2 — tests for aggregator file → consumer feature mapping."""

from __future__ import annotations

from faultline.analyzer.aggregator_consumers import (
    _filename_tokens,
    _filename_fallback_consumers,
    find_consumers,
)
from faultline.analyzer.symbol_graph import ImportEdge, SymbolGraph
from faultline.llm.sonnet_scanner import DeepScanResult


def _ds(features: dict[str, list[str]]) -> DeepScanResult:
    return DeepScanResult(features=features)


def _graph_with_reverse(
    edges: dict[str, list[str]],
) -> SymbolGraph:
    """Build a SymbolGraph where ``edges[file] = [importer1, importer2]``.

    Each importer becomes an ImportEdge whose ``target_file`` is the
    importer (matches the existing SymbolGraph schema where
    ``callers_of(X)`` returns edges pointing at importing files).
    """
    g = SymbolGraph()
    for target, importers in edges.items():
        g.reverse[target] = [
            ImportEdge(target_file=imp, target_symbol="*")
            for imp in importers
        ]
    return g


# ── _filename_tokens ─────────────────────────────────────────────────


class TestFilenameTokens:
    def test_dto_file_strips_dto_suffix(self):
        tokens = _filename_tokens(
            "packages/api-types/dto/auth/embed-login-body.dto.ts"
        )
        # ``.dto`` and ``.ts`` stripped; meaningful parts kept; parent
        # folder ``auth`` included as domain hint
        assert "embed" in tokens
        assert "login" in tokens
        assert "auth" in tokens
        assert "dto" not in tokens

    def test_camelcase_filename(self):
        tokens = _filename_tokens(
            "packages/shared-ui/src/BillingPlanCard.tsx"
        )
        assert "billing" in tokens
        assert "plan" in tokens
        assert "card" in tokens

    def test_short_tokens_dropped(self):
        # ``.dto`` stripped, leaving ``a-b-x``; all three tokens are
        # under 3 chars after normalization → all dropped
        tokens = _filename_tokens("a/b-x.dto.ts")
        assert tokens == set()

    def test_noise_tokens_dropped(self):
        tokens = _filename_tokens("packages/types/index.types.ts")
        # ``index``, ``types`` both filtered as noise; parent ``types``
        # also noise — empty result
        assert tokens == set()


# ── _filename_fallback_consumers ─────────────────────────────────────


class TestFilenameFallback:
    def test_matches_feature_by_substring(self):
        result = _ds({
            "Authentication": ["src/auth/login.tsx"],
            "Workflows": ["src/workflows/index.tsx"],
            "Dto": ["packages/api-types/dto/auth/login.dto.ts"],
        })
        consumers = _filename_fallback_consumers(
            "packages/api-types/dto/auth/login.dto.ts",
            result,
            excluded_feature="Dto",
        )
        # ``login`` and ``auth`` tokens match feature name "Authentication"
        # via substring (auth ⊂ authentication)
        assert "Authentication" in consumers
        assert "Dto" not in consumers
        assert "Workflows" not in consumers

    def test_excludes_self_feature(self):
        result = _ds({
            "Auth": ["src/auth/login.tsx"],
        })
        # File belongs to Auth itself — must not list as own consumer
        consumers = _filename_fallback_consumers(
            "packages/dto/auth/login.dto.ts",
            result,
            excluded_feature="Auth",
        )
        assert "Auth" not in consumers


# ── find_consumers — primary signal (symbol graph) ───────────────────


class TestFindConsumersFromSymbolGraph:
    def test_dto_file_imported_by_auth_feature(self):
        # Aggregator owns ``packages/api-types/dto/auth.dto.ts``.
        # Symbol graph says ``apps/cli/auth/login.controller.ts`` imports it.
        # That importer's feature is ``Authentication`` → expect Auth
        # in consumers.
        result = _ds({
            "Dto": ["packages/api-types/dto/auth.dto.ts"],
            "Authentication": [
                "apps/cli/auth/login.controller.ts",
                "apps/web/auth/login.tsx",
            ],
        })
        graph = _graph_with_reverse({
            "packages/api-types/dto/auth.dto.ts": [
                "apps/cli/auth/login.controller.ts",
            ],
        })
        consumers = find_consumers(
            ["packages/api-types/dto/auth.dto.ts"],
            aggregator_feature="Dto",
            result=result,
            symbol_graph=graph,
        )
        assert consumers["packages/api-types/dto/auth.dto.ts"] == [
            "Authentication",
        ]

    def test_dto_file_imported_by_multiple_features(self):
        result = _ds({
            "Dto": ["packages/api-types/dto/user.dto.ts"],
            "Authentication": ["apps/web/auth/login.tsx"],
            "Settings": ["apps/web/settings/profile.tsx"],
            "Billing": ["apps/web/billing/plan.tsx"],
        })
        graph = _graph_with_reverse({
            "packages/api-types/dto/user.dto.ts": [
                "apps/web/auth/login.tsx",
                "apps/web/settings/profile.tsx",
                "apps/web/billing/plan.tsx",
            ],
        })
        consumers = find_consumers(
            ["packages/api-types/dto/user.dto.ts"],
            aggregator_feature="Dto",
            result=result,
            symbol_graph=graph,
        )
        # All three consuming features listed, alphabetically
        assert consumers["packages/api-types/dto/user.dto.ts"] == [
            "Authentication", "Billing", "Settings",
        ]

    def test_excludes_aggregator_self_imports(self):
        # ``dto/user.dto.ts`` imports ``dto/base.dto.ts``. Both are in
        # the same Dto aggregator — that intra-aggregator edge must
        # NOT appear as a consumer.
        result = _ds({
            "Dto": [
                "packages/api-types/dto/user.dto.ts",
                "packages/api-types/dto/base.dto.ts",
            ],
            "Authentication": ["apps/web/auth/login.tsx"],
        })
        graph = _graph_with_reverse({
            "packages/api-types/dto/base.dto.ts": [
                "packages/api-types/dto/user.dto.ts",  # same Dto bucket
                "apps/web/auth/login.tsx",              # real consumer
            ],
        })
        consumers = find_consumers(
            ["packages/api-types/dto/base.dto.ts"],
            aggregator_feature="Dto",
            result=result,
            symbol_graph=graph,
        )
        assert consumers["packages/api-types/dto/base.dto.ts"] == [
            "Authentication",
        ]


# ── find_consumers — fallback path (no graph or no edges) ────────────


class TestFindConsumersFallback:
    def test_no_graph_uses_filename_heuristic(self):
        result = _ds({
            "Dto": ["packages/api-types/dto/auth/login.dto.ts"],
            "Authentication": ["apps/web/auth/login.tsx"],
        })
        consumers = find_consumers(
            ["packages/api-types/dto/auth/login.dto.ts"],
            aggregator_feature="Dto",
            result=result,
            symbol_graph=None,
        )
        # No graph → fallback to filename tokens; "auth" matches
        # "Authentication"
        assert "Authentication" in consumers[
            "packages/api-types/dto/auth/login.dto.ts"
        ]

    def test_graph_with_no_reverse_edges_uses_fallback(self):
        result = _ds({
            "Dto": ["packages/api-types/dto/billing.dto.ts"],
            "Billing": ["apps/web/billing/plan.tsx"],
        })
        graph = SymbolGraph()  # empty — no reverse edges
        consumers = find_consumers(
            ["packages/api-types/dto/billing.dto.ts"],
            aggregator_feature="Dto",
            result=result,
            symbol_graph=graph,
        )
        # Fallback finds Billing via filename token "billing"
        assert "Billing" in consumers[
            "packages/api-types/dto/billing.dto.ts"
        ]

    def test_unresolved_file_returns_empty_list(self):
        # File whose name has no semantic tokens AND no graph edges.
        # Day 4 will fold these into shared-infra rather than guessing.
        result = _ds({
            "Dto": ["packages/api-types/dto/x.dto.ts"],
            "Authentication": ["apps/web/auth/login.tsx"],
        })
        consumers = find_consumers(
            ["packages/api-types/dto/x.dto.ts"],
            aggregator_feature="Dto",
            result=result,
            symbol_graph=SymbolGraph(),
        )
        assert consumers["packages/api-types/dto/x.dto.ts"] == []
