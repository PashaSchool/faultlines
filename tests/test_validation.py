"""Unit tests for faultline.analyzer.validation.

All test inputs are either synthetic fixtures or real file paths observed
in the Day 1 baseline runs (fastapi tutorial explosion, cal.com
vitest-mocks leak, gin root bucket). Regressions here directly correspond
to acceptance criteria C2, C3, and F in REWRITE_PLAN.md.
"""

import pytest

from faultline.analyzer.validation import (
    CANONICAL_SHARED_INFRA,
    canonical_bucket_name,
    drop_phantom_features,
    filter_test_files,
    is_documentation_file,
    is_test_feature_name,
    is_test_file,
    partition_docs_vs_code,
)


# ── is_test_file ─────────────────────────────────────────────────────────


class TestIsTestFile:
    """Test the is_test_file predicate on real paths from baseline repos."""

    @pytest.mark.parametrize(
        "path",
        [
            "src/auth/login.test.ts",
            "packages/core/__tests__/parser.ts",
            "tests/unit/test_git.py",
            "e2e/checkout.spec.ts",
            "apps/web/__mocks__/api.ts",
            "packages/embeds/vitest-mocks/handler.ts",
            "src/components/Button.spec.tsx",
            "pkg/gin_test.go",
            "tests/__fixtures__/sample.json",
            "cypress/e2e/login.cy.ts",
        ],
    )
    def test_detects_test_files(self, path: str) -> None:
        assert is_test_file(path), f"expected {path!r} to be a test file"

    @pytest.mark.parametrize(
        "path",
        [
            "src/auth/login.ts",
            "src/attestation/handler.ts",   # "attest" contains "test" — must not match
            "src/contest/round.ts",          # "contest" contains "test" — must not match
            "packages/core/parser.ts",
            "README.md",
            "docs/architecture.md",
            "src/components/Button.tsx",
        ],
    )
    def test_rejects_production_files(self, path: str) -> None:
        assert not is_test_file(path), f"expected {path!r} NOT to be a test file"


# ── is_test_feature_name ─────────────────────────────────────────────────


class TestIsTestFeatureName:
    @pytest.mark.parametrize(
        "name",
        ["tests", "__tests__", "vitest-mocks", "e2e", "__fixtures__", "Cypress", "SPEC"],
    )
    def test_detects_test_feature_names(self, name: str) -> None:
        assert is_test_feature_name(name)

    @pytest.mark.parametrize(
        "name",
        ["authentication", "billing", "user-onboarding", "shared-infra"],
    )
    def test_rejects_real_feature_names(self, name: str) -> None:
        assert not is_test_feature_name(name)


# ── is_documentation_file ────────────────────────────────────────────────


class TestIsDocumentationFile:
    """Regression for Day 1 fastapi/trpc findings — docs/tutorials must collapse."""

    @pytest.mark.parametrize(
        "path",
        [
            # fastapi baseline — 21 phantom features from tutorial dirs
            "docs_src/tutorial001_py310/main.py",
            "docs_src/tutorial002_an_py310/app.py",
            "docs_src/main.py",
            "docs/reference/index.md",
            # trpc baseline — www/* split into 8 features
            "www/versioned/docs.mdx",
            "www/client/index.tsx",
            "www/og-image/page.tsx",
            "www/website/landing.tsx",
            # Generic examples
            "examples/basic/server.ts",
            "examples/next-edge-runtime/page.tsx",
            "example/hello-world/main.rs",
            "demo/quickstart.ts",
            "playground/sandbox.ts",
            "tutorials/getting-started/step-01.md",
            "site/content/blog/post-1.md",
            "apps/docs/next.config.ts",   # nested inside apps/
        ],
    )
    def test_detects_doc_files(self, path: str) -> None:
        assert is_documentation_file(path), f"expected {path!r} to be documentation"

    @pytest.mark.parametrize(
        "path",
        [
            "src/auth/login.ts",
            "apps/web/pages/index.tsx",
            "packages/core/parser.ts",
            "fastapi/routing.py",
            "README.md",                         # at root is not in a docs dir
            "pkg/gin.go",
        ],
    )
    def test_rejects_production_code(self, path: str) -> None:
        assert not is_documentation_file(path), f"expected {path!r} NOT to be documentation"


# ── canonical_bucket_name ────────────────────────────────────────────────


class TestCanonicalBucketName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("root", CANONICAL_SHARED_INFRA),          # gin baseline
            ("init", CANONICAL_SHARED_INFRA),          # fastapi baseline
            ("__init__", CANONICAL_SHARED_INFRA),
            ("main", CANONICAL_SHARED_INFRA),
            ("shared", CANONICAL_SHARED_INFRA),
            ("SHARED", CANONICAL_SHARED_INFRA),
            ("/root/", CANONICAL_SHARED_INFRA),        # stripped
            ("authentication", "authentication"),
            ("user-onboarding", "user-onboarding"),
            ("billing", "billing"),
        ],
    )
    def test_canonicalizes(self, raw: str, expected: str) -> None:
        assert canonical_bucket_name(raw) == expected


# ── drop_phantom_features ────────────────────────────────────────────────


class TestDropPhantomFeatures:
    def test_drops_empty_feature(self) -> None:
        result = drop_phantom_features({
            "auth": ["src/auth/a.ts"],
            "ghost": [],
        })
        assert result == {"auth": ["src/auth/a.ts"]}

    def test_drops_test_feature(self) -> None:
        result = drop_phantom_features({
            "auth": ["src/auth/a.ts"],
            "vitest-mocks": ["packages/embeds/vitest-mocks/handler.ts"],
            "tests": ["tests/unit/a.test.ts"],
        })
        assert "vitest-mocks" not in result
        assert "tests" not in result
        assert "auth" in result

    def test_preserves_real_features(self) -> None:
        features = {
            "authentication": ["src/auth/login.ts"],
            "billing": ["src/billing/stripe.ts"],
            "shared-infra": ["src/utils/time.ts"],
        }
        assert drop_phantom_features(features) == features


# ── filter_test_files ────────────────────────────────────────────────────


class TestFilterTestFiles:
    def test_preserves_order_and_filters(self) -> None:
        inputs = [
            "src/auth/login.ts",
            "src/auth/login.test.ts",
            "packages/core/parser.ts",
            "packages/core/__tests__/parser.ts",
            "README.md",
        ]
        expected = [
            "src/auth/login.ts",
            "packages/core/parser.ts",
            "README.md",
        ]
        assert filter_test_files(inputs) == expected


# ── partition_docs_vs_code ───────────────────────────────────────────────


class TestPartitionDocsVsCode:
    def test_fastapi_regression_shape(self) -> None:
        """fastapi: 21 tutorials under docs_src/ must collapse to docs bucket."""
        inputs = [
            "fastapi/routing.py",
            "fastapi/applications.py",
            "docs_src/tutorial001_py310/main.py",
            "docs_src/tutorial002_py310/app.py",
            "docs_src/tutorial003_py310/main.py",
            "tests/test_routing.py",
            "README.md",
        ]
        code, docs = partition_docs_vs_code(inputs)
        assert "fastapi/routing.py" in code
        assert "fastapi/applications.py" in code
        assert "tests/test_routing.py" in code   # tests still in code bucket — separate filter
        assert len(docs) == 3
        for path in docs:
            assert path.startswith("docs_src/")

    def test_trpc_regression_shape(self) -> None:
        """trpc: all www/* files collapse into docs bucket, not 8 separate features."""
        inputs = [
            "packages/server/src/index.ts",
            "packages/client/src/index.ts",
            "www/versioned/docs.mdx",
            "www/client/page.tsx",
            "www/server/middleware.ts",
            "www/documentation/index.mdx",
            "www/blog/post-1.mdx",
            "www/website/landing.tsx",
            "www/og-image/page.tsx",
            "www/shared-infra/config.ts",
        ]
        code, docs = partition_docs_vs_code(inputs)
        assert len(code) == 2
        assert len(docs) == 8
        assert all(d.startswith("www/") for d in docs)

    def test_empty(self) -> None:
        assert partition_docs_vs_code([]) == ([], [])
