"""Regression tests for Fixable-accuracy improvements (Fix #1, #2).

These reproduce two failure patterns observed in the n8n / strapi eval:

  * Class A — duplicate display_name across packages (e.g. "Credentials"
    appears 3× from packages/@n8n/cli, packages/@n8n/core, packages/nodes-base).
  * Class B — tiny-and-cold features (≤3 files, <30 commits, no flows)
    surviving deep_scan_workspace's per-package output (Di, Push, Chat,
    Codemirror Lang, TypeScript Config).

The fixes must NOT touch:

  * Hot small features — ≥30 commits OR ≥1 flow even with 1-3 files
    (e.g. n8n's Workflows 3f/300c, Ndv 3f/148c, Execution 1f/112c).
  * Large features — anything ≥4 files keeps surviving regardless of
    commit count (we trust the splitter's verdict on real-sized features).
"""

from __future__ import annotations

from datetime import datetime, timezone

from faultline.llm.pipeline import (
    _collapse_same_name_features,
    _flow_tokens,
    _name_matches_flow,
    _reattribute_flows_by_name_match,
)
from faultline.llm.sonnet_scanner import DeepScanResult
from faultline.analyzer.features import _drop_noise_features
from faultline.models.types import Feature


def _ds(features: dict[str, list[str]], descriptions: dict[str, str] | None = None) -> DeepScanResult:
    return DeepScanResult(
        features=features,
        descriptions=descriptions or {n: f"{n} description" for n in features},
    )


def _feat(name: str, paths: list[str], commits: int) -> Feature:
    return Feature(
        name=name,
        paths=paths,
        authors=["alice@example.com"],
        total_commits=commits,
        bug_fixes=0,
        bug_fix_ratio=0.0,
        last_modified=datetime.now(tz=timezone.utc),
        health_score=80.0,
    )


# ── Fix #1 — same-name auto-merge ────────────────────────────────────


class TestCollapseSameName:
    def test_three_credentials_features_disjoint_renamed(self):
        """Sprint 14 — n8n's Credentials × 3 across packages. Paths are
        path-disjoint (cli / core / nodes-base), so we PREFIX-RENAME
        each to keep them separate. Different packages, different
        business concerns. The old behaviour (always merge) is wrong."""
        result = _ds({
            "Credentials": ["packages/@n8n/cli/credentials.ts"],
            "credentials": [
                "packages/@n8n/core/credentials/loader.ts",
                "packages/@n8n/core/credentials/manager.ts",
            ],
            "CREDENTIALS": [
                "packages/nodes-base/Credentials.node.ts",
                "packages/nodes-base/CredentialsHelper.ts",
                "packages/nodes-base/credentials_test.ts",
            ],
            "Workflow": ["packages/workflow/index.ts", "packages/workflow/runner.ts"],
        })

        merged = _collapse_same_name_features(result)

        # Three Credentials variants stay distinct after path-aware rename.
        cred_keys = [
            n for n in merged.features
            if n.lower().endswith("credentials") or "credentials" in n.lower()
        ]
        assert len(cred_keys) == 3, f"expected 3 distinct Credentials, got {cred_keys}"
        # Workflow feature unchanged.
        assert len(merged.features["Workflow"]) == 2

    def test_no_duplicates_returns_input_unchanged(self):
        """No same-name pairs → no-op."""
        result = _ds({
            "Auth": ["a/login.ts"],
            "Billing": ["b/stripe.ts"],
            "Editor": ["c/edit.ts"],
        })
        merged = _collapse_same_name_features(result)
        assert dict(merged.features) == dict(result.features)

    def test_paths_deduplicated_and_sorted(self):
        """Determinism: same input → byte-identical output."""
        result = _ds({
            "Errors": ["pkg/a/errors.ts", "pkg/c/errors.ts"],
            "errors": ["pkg/a/errors.ts", "pkg/b/errors.ts"],  # overlap on pkg/a
        })
        merged = _collapse_same_name_features(result)
        assert len(merged.features) == 1
        canonical_paths = list(merged.features.values())[0]
        # Deduped + sorted
        assert canonical_paths == sorted(set([
            "pkg/a/errors.ts", "pkg/b/errors.ts", "pkg/c/errors.ts",
        ]))

    def test_descriptions_carry_over_when_paths_overlap(self):
        """Sprint 14 — descriptions follow the canonical only when
        paths overlap (true merge). With overlap, behaviour unchanged."""
        result = _ds(
            {
                "Admin": ["shared.ts"],
                "ADMIN": ["b.ts", "c.ts", "shared.ts"],  # overlap on shared.ts
            },
            descriptions={"Admin": "small one", "ADMIN": "big one"},
        )
        merged = _collapse_same_name_features(result)
        assert len(merged.features) == 1
        canonical = list(merged.features.keys())[0]
        # Canonical = ADMIN (3 paths), description follows.
        assert canonical == "ADMIN"
        assert merged.descriptions[canonical] == "big one"

    def test_synthetic_buckets_protected(self):
        """``shared-infra`` and ``documentation`` never get merged with each
        other or with anything else, even if names happened to collide."""
        result = _ds({
            "shared-infra": ["a.ts"],
            "documentation": ["docs/x.md"],
        })
        merged = _collapse_same_name_features(result)
        assert "shared-infra" in merged.features

    def test_slash_prefixed_subdecompose_children_disjoint_renamed(self):
        """Sprint 14 — slash-prefixed sub_decompose children with
        path-disjoint locations (nodes-base vs frontend) get
        prefix-renamed instead of merged. Different deployable
        targets shouldn't be flattened together."""
        result = _ds({
            "credentials": [
                "packages/nodes-base/credentials/AwsAssumeRole.credentials.ts",
                "packages/nodes-base/credentials/Bitbucket.credentials.ts",
            ],
            "n8n/credentials": [
                "packages/frontend/editor-ui/src/features/credentials/quickConnect.api.ts",
            ],
            "Workflow Editor": ["apps/web/workflow-editor/canvas.tsx"],
        })
        merged = _collapse_same_name_features(result)
        # Both Credentials variants stay distinct after rename
        cred_keys = [
            n for n in merged.features if "credentials" in n.lower()
        ]
        assert len(cred_keys) == 2
        # Workflow Editor untouched
        assert "Workflow Editor" in merged.features

    def test_post_pipeline_disjoint_duplicates_get_renamed(self):
        """Sprint 14 — plane's Issues lives in two different deployable
        apps (web vs space). Path-disjoint, so the path-aware
        collapse renames instead of merging. This is a known plane
        scatter case the user explicitly flagged in S14 plan."""
        result = _ds({
            "Issues": [
                "apps/web/components/issues/header.tsx",
                "apps/web/core/issues/list.tsx",
            ],
            "issues": [
                "apps/space/components/issues/board.tsx",
            ],
            "Workflow Editor": ["apps/web/workflow-editor/canvas.tsx"],
        })
        merged = _collapse_same_name_features(result)
        # Both Issues stay distinct after rename
        issues_keys = [n for n in merged.features if "issues" in n.lower()]
        assert len(issues_keys) == 2
        # Workflow Editor untouched
        assert "Workflow Editor" in merged.features


# ── Fix #2 — commit-aware noise drop ─────────────────────────────────


class TestDropNoiseFeatures:
    def test_tiny_cold_features_dropped_to_shared_infra(self):
        """Di (1f/14c), Push (3f/14c), Chat (2f/6c) — drop. Editor stays."""
        features = [
            _feat("Editor", [f"editor/file{i}.ts" for i in range(50)], 800),
            _feat("Di", ["packages/di/index.ts"], 14),
            _feat("Push", ["push/a.ts", "push/b.ts", "push/c.ts"], 14),
            _feat("Chat", ["chat/a.ts", "chat/b.ts"], 6),
        ]
        flows_by_feature = {"Editor": ["edit-document"]}

        kept = _drop_noise_features(features, flows_by_feature)

        names = [f.name for f in kept]
        assert "Editor" in names
        assert "Di" not in names
        assert "Push" not in names
        assert "Chat" not in names
        # Their paths fold into shared-infra
        infra = next(f for f in kept if f.name == "shared-infra")
        assert "packages/di/index.ts" in infra.paths
        assert "push/a.ts" in infra.paths
        assert "chat/a.ts" in infra.paths

    def test_hot_small_features_survive(self):
        """ESCAPE HATCH: ≥30 commits keeps the feature even with 1-3 files.
        Reproduces n8n's Workflows 3f/300c, Execution 1f/112c, Ndv 3f/148c."""
        features = [
            _feat("Workflows", ["frontend/workflows.tsx", "frontend/list.tsx", "frontend/detail.tsx"], 300),
            _feat("Execution", ["cli/execute.ts"], 112),
            _feat("Ndv", ["frontend/Ndv.vue", "frontend/NdvInput.vue", "frontend/NdvOutput.vue"], 148),
            _feat("Cold", ["x/y.ts"], 5),  # control: should be dropped
        ]
        kept = _drop_noise_features(features, {})

        names = [f.name for f in kept]
        assert "Workflows" in names
        assert "Execution" in names
        assert "Ndv" in names
        assert "Cold" not in names

    def test_features_with_flows_survive(self):
        """A feature with flows is hot by definition, even at 1f/2c."""
        features = [
            _feat("AuthLogin", ["auth/login.ts"], 2),
        ]
        kept = _drop_noise_features(features, {"AuthLogin": ["log-in-flow"]})
        assert [f.name for f in kept] == ["AuthLogin"]

    def test_no_noise_returns_input(self):
        features = [
            _feat("Big", [f"a/{i}.ts" for i in range(20)], 100),
        ]
        kept = _drop_noise_features(features, {})
        assert len(kept) == 1
        assert kept[0].name == "Big"

    def test_synthetic_buckets_never_dropped(self):
        """``shared-infra`` and ``documentation`` are never noise-targets
        even if they happen to be small at scan time."""
        features = [
            _feat("shared-infra", ["x.ts"], 1),
            _feat("documentation", ["docs/x.md"], 1),
        ]
        kept = _drop_noise_features(features, {})
        names = [f.name for f in kept]
        assert "shared-infra" in names
        assert "documentation" in names

    def test_dropped_paths_dedupe_in_shared_infra(self):
        """Re-folding into shared-infra must not duplicate paths if shared-infra
        already exists with overlapping content."""
        features = [
            _feat("shared-infra", ["common/a.ts", "common/b.ts"], 50),
            _feat("Tiny", ["common/a.ts", "common/c.ts"], 3),  # overlaps on a.ts
        ]
        kept = _drop_noise_features(features, {})
        infra = next(f for f in kept if f.name == "shared-infra")
        assert infra.paths.count("common/a.ts") == 1
        assert sorted(infra.paths) == ["common/a.ts", "common/b.ts", "common/c.ts"]


# ── Stage 2.6 — flow re-attribution by name-token match ─────────────


class TestFlowTokens:
    def test_splits_on_spaces_hyphens_underscores(self):
        assert _flow_tokens("Authenticate with Password") == {"authenticate", "with", "password"}
        assert _flow_tokens("manage-roles-flow") == {"manage", "roles", "flow"}
        assert _flow_tokens("create_workflow_flow") == {"create", "workflow", "flow"}

    def test_normalizes_to_lowercase(self):
        assert "authenticate" in _flow_tokens("Authenticate")


class TestNameMatchesFlow:
    def test_full_token_match_scores_2(self):
        assert _name_matches_flow("Auth", _flow_tokens("Authenticate Password")) >= 1
        # Specifically: "auth" is substring of "authenticate" — score 1 (loose)
        # AND if flow has bare token "auth" — score 2

    def test_substring_match_scores_1(self):
        # "auth" ⊂ "authenticate" → score 1 (loose)
        score = _name_matches_flow("Auth", {"authenticate", "password"})
        assert score == 1

    def test_exact_token_scores_2(self):
        # Feature "Auth" exactly matches token "auth"
        score = _name_matches_flow("Auth", {"auth", "log", "in"})
        assert score == 2

    def test_short_feature_name_no_match(self):
        # "AI" is too short (3 chars) to count as a meaningful match
        # — would otherwise match every flow with an A in it
        score = _name_matches_flow("AI", {"manage", "ai", "model"})
        assert score == 0


class TestReattributeFlows:
    def test_moves_auth_flow_from_studio_to_auth(self):
        """Supabase real case: Auth feature has 38 files but 0 flows;
        ``Authenticate with Password`` flow stranded on Vue Blocks
        because login form's entry point lives there."""
        result = DeepScanResult(
            features={
                "Auth": ["studio/auth/RateLimits.tsx"],
                "Vue Blocks": ["vue-blocks/login-form.vue"],
            },
            flows={
                "Vue Blocks": ["Authenticate with Password"],
            },
            flow_descriptions={
                "Vue Blocks": {"Authenticate with Password": "Login form."},
            },
        )
        _reattribute_flows_by_name_match(result)
        # Flow moved to Auth
        assert "Authenticate with Password" in result.flows.get("Auth", [])
        assert "Authenticate with Password" not in result.flows.get("Vue Blocks", [])
        # Description migrated too
        assert (
            result.flow_descriptions.get("Auth", {}).get("Authenticate with Password")
            == "Login form."
        )

    def test_does_not_move_when_current_owner_already_matches(self):
        """If current owner's name already strongly matches flow tokens,
        don't move (avoid bouncing flows around)."""
        result = DeepScanResult(
            features={
                "Auth": ["src/auth/login.tsx"],
                "Authentication": ["src/auth/signup.tsx"],
            },
            flows={
                "Auth": ["Login Flow"],  # current owner is Auth, "auth"
                                          # is substring of nothing in "login flow"
            },
        )
        _reattribute_flows_by_name_match(result)
        # No move — neither feature has stronger match than current
        assert "Login Flow" in result.flows.get("Auth", [])

    def test_protected_buckets_dont_steal_flows(self):
        result = DeepScanResult(
            features={
                "Auth": ["src/auth/login.tsx"],
                "shared-infra": ["build.config.ts"],
            },
            flows={
                "Auth": ["Build Auth System"],  # "build" might match infra
            },
        )
        _reattribute_flows_by_name_match(result)
        # shared-infra is protected; flow stays on Auth
        assert "Build Auth System" in result.flows.get("Auth", [])

    def test_moves_multiple_flows_in_one_pass(self):
        result = DeepScanResult(
            features={
                "Auth": ["a.ts"],
                "Billing": ["b.ts"],
                "Studio": ["s.ts"],
            },
            flows={
                "Studio": [
                    "Manage Auth Configuration",
                    "View Billing Account",
                    "Open Studio Dashboard",  # stays — "studio" ⊂ "studio"
                ],
            },
        )
        _reattribute_flows_by_name_match(result)
        assert "Manage Auth Configuration" in result.flows.get("Auth", [])
        assert "View Billing Account" in result.flows.get("Billing", [])
        assert "Open Studio Dashboard" in result.flows.get("Studio", [])

    def test_short_feature_names_dont_steal(self):
        """``AI`` (2 chars) is too short to be a reliable match —
        otherwise it would steal flows whose tokens contain 'ai' as
        a substring (e.g. ``main``, ``email``)."""
        result = DeepScanResult(
            features={
                "AI": ["a.ts"],
                "Email": ["e.ts"],
            },
            flows={
                "Email": ["Send Email"],  # AI's "ai" ⊂ "email" — but AI too short
            },
        )
        _reattribute_flows_by_name_match(result)
        assert "Send Email" in result.flows.get("Email", [])
        assert result.flows.get("AI", []) == []
