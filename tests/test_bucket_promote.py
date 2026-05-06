"""Sprint 14 Day 1 — bucket_promote tests."""

from __future__ import annotations

from faultline.analyzer.ast_extractor import FileSignature
from faultline.analyzer.bucket_promote import (
    MIN_INBOUND,
    promote_imported_docs_infra,
)
from faultline.analyzer.bucketizer import Bucket


def _sig(path: str, *, imports=(), exports=(), routes=()) -> FileSignature:
    return FileSignature(
        path=path,
        exports=list(exports),
        routes=list(routes),
        imports=list(imports),
    )


def test_promotes_heavily_imported_schema():
    """A schema file imported by 3+ SOURCE files moves to SOURCE."""
    partition = {
        Bucket.SOURCE: ["src/a.ts", "src/b.ts", "src/c.ts", "src/d.ts"],
        Bucket.DOCUMENTATION: ["schema/types.ts"],
        Bucket.INFRASTRUCTURE: [],
        Bucket.TESTS: [],
        Bucket.GENERATED: [],
    }
    sigs = {
        "src/a.ts": _sig("src/a.ts", imports=["schema/types.ts"], exports=["Foo"]),
        "src/b.ts": _sig("src/b.ts", imports=["schema/types.ts"], exports=["Bar"]),
        "src/c.ts": _sig("src/c.ts", imports=["schema/types.ts"], exports=["Baz"]),
        "src/d.ts": _sig("src/d.ts", imports=[], exports=["Qux"]),
        "schema/types.ts": _sig("schema/types.ts", exports=["UserType"]),
    }
    docs_p, infra_p = promote_imported_docs_infra(partition, sigs)
    assert docs_p == 1
    assert "schema/types.ts" in partition[Bucket.SOURCE]
    assert "schema/types.ts" not in partition[Bucket.DOCUMENTATION]


def test_skips_below_threshold():
    """Only 2 inbound — not enough."""
    partition = {
        Bucket.SOURCE: ["src/a.ts", "src/b.ts"],
        Bucket.DOCUMENTATION: ["schema/types.ts"],
        Bucket.INFRASTRUCTURE: [],
        Bucket.TESTS: [],
        Bucket.GENERATED: [],
    }
    sigs = {
        "src/a.ts": _sig("src/a.ts", imports=["schema/types.ts"], exports=["x"]),
        "src/b.ts": _sig("src/b.ts", imports=["schema/types.ts"], exports=["y"]),
        "schema/types.ts": _sig("schema/types.ts", exports=["UserType"]),
    }
    docs_p, _ = promote_imported_docs_infra(partition, sigs)
    assert docs_p == 0
    assert "schema/types.ts" in partition[Bucket.DOCUMENTATION]


def test_skips_dts_ambient_declarations():
    """`.d.ts` files stay in INFRA even with many imports — no runtime."""
    partition = {
        Bucket.SOURCE: [f"src/{c}.ts" for c in "abcd"],
        Bucket.DOCUMENTATION: [],
        Bucket.INFRASTRUCTURE: ["types/global.d.ts"],
        Bucket.TESTS: [],
        Bucket.GENERATED: [],
    }
    sigs = {
        f"src/{c}.ts": _sig(f"src/{c}.ts", imports=["types/global.d.ts"]) for c in "abcd"
    }
    sigs["types/global.d.ts"] = _sig("types/global.d.ts", exports=[])
    _, infra_p = promote_imported_docs_infra(partition, sigs)
    assert infra_p == 0
    assert "types/global.d.ts" in partition[Bucket.INFRASTRUCTURE]


def test_skips_pure_type_only_files():
    """File with no exports/routes considered pure-type — skipped."""
    partition = {
        Bucket.SOURCE: [f"src/{c}.ts" for c in "abcd"],
        Bucket.DOCUMENTATION: ["schema/empty.ts"],
        Bucket.INFRASTRUCTURE: [],
        Bucket.TESTS: [],
        Bucket.GENERATED: [],
    }
    sigs = {f"src/{c}.ts": _sig(f"src/{c}.ts", imports=["schema/empty.ts"]) for c in "abcd"}
    sigs["schema/empty.ts"] = _sig("schema/empty.ts")  # no exports
    docs_p, _ = promote_imported_docs_infra(partition, sigs)
    assert docs_p == 0


def test_skips_markdown_images():
    """Non-source extensions never promote."""
    partition = {
        Bucket.SOURCE: [f"src/{c}.ts" for c in "abcd"],
        Bucket.DOCUMENTATION: ["docs/api.md"],
        Bucket.INFRASTRUCTURE: ["assets/logo.png"],
        Bucket.TESTS: [],
        Bucket.GENERATED: [],
    }
    sigs = {
        f"src/{c}.ts": _sig(f"src/{c}.ts", imports=["docs/api.md", "assets/logo.png"])
        for c in "abcd"
    }
    sigs["docs/api.md"] = _sig("docs/api.md", exports=["foo"])
    docs_p, infra_p = promote_imported_docs_infra(partition, sigs)
    assert docs_p == 0 and infra_p == 0


def test_promotes_infra_config_with_logic():
    """An INFRA file with real exports that 3+ source files import gets promoted."""
    partition = {
        Bucket.SOURCE: [f"src/{c}.ts" for c in "abcd"],
        Bucket.DOCUMENTATION: [],
        Bucket.INFRASTRUCTURE: ["config/feature-flags.ts"],
        Bucket.TESTS: [],
        Bucket.GENERATED: [],
    }
    sigs = {
        f"src/{c}.ts": _sig(f"src/{c}.ts", imports=["config/feature-flags.ts"])
        for c in "abcd"
    }
    sigs["config/feature-flags.ts"] = _sig(
        "config/feature-flags.ts",
        exports=["isFeatureEnabled", "FEATURE_FLAGS"],
    )
    _, infra_p = promote_imported_docs_infra(partition, sigs)
    assert infra_p == 1
    assert "config/feature-flags.ts" in partition[Bucket.SOURCE]


def test_no_signatures_returns_zero():
    partition = {b: [] for b in Bucket}
    docs_p, infra_p = promote_imported_docs_infra(partition, None)
    assert docs_p == 0 and infra_p == 0
