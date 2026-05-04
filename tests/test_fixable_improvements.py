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

from faultline.llm.pipeline import _collapse_same_name_features
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
    def test_three_credentials_features_collapse_to_one(self):
        """n8n's Credentials × 3 case: same display_name across 3 packages."""
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

        # Three Credentials variants collapse to ONE entry. Workflow untouched.
        cred_keys = [n for n in merged.features if n.lower() == "credentials"]
        assert len(cred_keys) == 1, f"expected 1 Credentials, got {cred_keys}"

        # All 6 original credential paths preserved on the canonical entry.
        canonical = merged.features[cred_keys[0]]
        assert len(canonical) == 6
        assert "packages/nodes-base/Credentials.node.ts" in canonical
        assert "packages/@n8n/cli/credentials.ts" in canonical

        # Canonical name = the variant with the most paths (3 files).
        assert cred_keys[0] == "CREDENTIALS"

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

    def test_descriptions_carry_over_from_canonical(self):
        result = _ds(
            {
                "Admin": ["a.ts"],
                "ADMIN": ["b.ts", "c.ts", "d.ts"],
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
        assert "documentation" in merged.features


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
