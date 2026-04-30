"""Tests for the Rust import resolver in symbol_graph (P6b)."""

from __future__ import annotations

from faultline.analyzer.symbol_graph import (
    _detect_rust_crate_root,
    _extract_rs_imports,
    _extract_rs_mods,
    _resolve_rs_use,
    build_symbol_graph,
)


# ── Parsing ─────────────────────────────────────────────────────


def test_extract_use_crate_path():
    src = "use crate::auth::login::Handler;\n"
    assert _extract_rs_imports(src) == {"crate::auth::login": {"Handler"}}


def test_extract_use_with_glob():
    src = "use crate::handlers::*;\n"
    assert _extract_rs_imports(src) == {"crate::handlers": {"*"}}


def test_extract_use_with_braces():
    src = "use crate::api::{post, get, put};\n"
    assert _extract_rs_imports(src) == {"crate::api": {"post", "get", "put"}}


def test_extract_use_super_self():
    src = "use super::sibling::Foo;\nuse self::nested::Bar;\n"
    out = _extract_rs_imports(src)
    assert out["super::sibling"] == {"Foo"}
    assert out["self::nested"] == {"Bar"}


def test_extract_pub_use():
    src = "pub use crate::types::User;\n"
    assert _extract_rs_imports(src) == {"crate::types": {"User"}}


def test_extract_ignores_external_crates():
    src = "use serde::Deserialize;\nuse tokio::runtime::Runtime;\n"
    # External crates have no ``crate``/``super``/``self`` prefix, so
    # the regex (which anchors to those) returns nothing.
    assert _extract_rs_imports(src) == {}


def test_extract_mod_declarations():
    src = "mod auth;\npub mod billing;\nfn main() {}\n"
    assert _extract_rs_mods(src) == {"auth", "billing"}


# ── Crate root detection ────────────────────────────────────────


def test_detect_crate_root_with_src():
    files = {"src/main.rs", "src/lib.rs", "Cargo.toml"}
    assert _detect_rust_crate_root(files) == "src"


def test_detect_crate_root_flat_layout():
    files = {"main.rs", "Cargo.toml"}
    assert _detect_rust_crate_root(files) == ""


# ── Resolution ──────────────────────────────────────────────────


def test_resolve_crate_path_to_file():
    files = {
        "src/main.rs",
        "src/lib.rs",
        "src/auth/login.rs",
        "src/auth/mod.rs",
    }
    target = _resolve_rs_use(
        "src/main.rs", "crate::auth::login::Handler", files, "src",
    )
    assert target == "src/auth/login.rs"


def test_resolve_crate_path_to_mod_rs():
    files = {"src/main.rs", "src/auth/mod.rs"}
    target = _resolve_rs_use(
        "src/main.rs", "crate::auth::Login", files, "src",
    )
    assert target == "src/auth/mod.rs"


def test_resolve_super_walks_up():
    files = {
        "src/auth/login.rs",
        "src/serializers.rs",
    }
    target = _resolve_rs_use(
        "src/auth/login.rs", "super::serializers::Foo", files, "src",
    )
    assert target == "src/serializers.rs"


def test_resolve_self_in_same_dir():
    files = {
        "src/auth/login.rs",
        "src/auth/helpers.rs",
    }
    target = _resolve_rs_use(
        "src/auth/login.rs", "self::helpers::utils", files, "src",
    )
    assert target == "src/auth/helpers.rs"


def test_resolve_external_crate_returns_none():
    files = {"src/main.rs"}
    assert _resolve_rs_use(
        "src/main.rs", "serde::Deserialize", files, "src",
    ) is None


# ── End-to-end ─────────────────────────────────────────────────


def test_build_symbol_graph_traces_rust_imports(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text(
        '[package]\nname = "myapp"\nversion = "0.1.0"\n', encoding="utf-8",
    )
    src = repo / "src"
    src.mkdir()
    (src / "main.rs").write_text(
        "mod handlers;\nuse crate::handlers::Handler;\nfn main() {}\n",
        encoding="utf-8",
    )
    (src / "handlers.rs").write_text(
        "pub struct Handler {}\n", encoding="utf-8",
    )

    graph = build_symbol_graph(
        str(repo),
        ["Cargo.toml", "src/main.rs", "src/handlers.rs"],
        include_http_edges=False,
    )
    main_edges = graph.forward.get("src/main.rs") or []
    # ``mod handlers;`` AND ``use crate::handlers::Handler;`` both
    # point at handlers.rs — we get one or two edges.
    assert any(e.target_file == "src/handlers.rs" for e in main_edges)
    # Reverse map populated.
    assert any(
        e.target_file == "src/main.rs"
        for e in (graph.reverse.get("src/handlers.rs") or [])
    )
