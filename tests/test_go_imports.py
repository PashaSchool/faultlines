"""Tests for the Go import resolver in symbol_graph (P6a)."""

from __future__ import annotations

from faultline.analyzer.symbol_graph import (
    _extract_go_imports,
    _first_go_file_in_dir,
    _go_module_path,
    _resolve_go_module,
    build_symbol_graph,
)


# ── Parsing ─────────────────────────────────────────────────────


def test_extract_single_line_import():
    src = 'package gin\nimport "fmt"\n'
    assert _extract_go_imports(src) == {"fmt"}


def test_extract_aliased_single_line():
    src = 'package gin\nimport j "encoding/json"\n'
    assert _extract_go_imports(src) == {"encoding/json"}


def test_extract_block_imports():
    src = (
        "package gin\n"
        "import (\n"
        '    "fmt"\n'
        '    "github.com/foo/bar"\n'
        '    _ "github.com/baz/qux"  // side effect\n'
        '    j "encoding/json"\n'
        ")\n"
    )
    out = _extract_go_imports(src)
    assert out == {
        "fmt", "github.com/foo/bar",
        "github.com/baz/qux", "encoding/json",
    }


def test_extract_handles_no_imports():
    src = "package gin\n\nfunc Foo() {}\n"
    assert _extract_go_imports(src) == set()


# ── go.mod parsing ──────────────────────────────────────────────


def test_go_module_path_reads_go_mod(tmp_path):
    (tmp_path / "go.mod").write_text(
        "module github.com/gin-gonic/gin\n\ngo 1.21\n", encoding="utf-8",
    )
    assert _go_module_path(tmp_path) == "github.com/gin-gonic/gin"


def test_go_module_path_returns_none_when_missing(tmp_path):
    assert _go_module_path(tmp_path) is None


# ── Resolution ──────────────────────────────────────────────────


def test_resolve_returns_none_for_stdlib():
    files = {"context.go"}
    assert _resolve_go_module("fmt", files, "github.com/x/y") is None
    assert _resolve_go_module("encoding/json", files, "github.com/x/y") is None


def test_resolve_returns_none_for_third_party():
    files = {"context.go"}
    assert _resolve_go_module(
        "github.com/other/lib", files, "github.com/x/y",
    ) is None


def test_resolve_flat_layout_module_self():
    """gin: ``module github.com/gin-gonic/gin`` and root ``.go`` files.
    An import of ``github.com/gin-gonic/gin`` resolves to a root file."""
    files = {"context.go", "gin.go", "router.go", "binding/binding.go"}
    target = _resolve_go_module(
        "github.com/gin-gonic/gin", files,
        "github.com/gin-gonic/gin",
    )
    assert target in files
    # Must be a root file, not under binding/
    assert "/" not in target


def test_resolve_subdir_under_module():
    files = {
        "context.go",
        "binding/binding.go",
        "binding/json.go",
    }
    target = _resolve_go_module(
        "github.com/gin-gonic/gin/binding", files,
        "github.com/gin-gonic/gin",
    )
    assert target in {"binding/binding.go", "binding/json.go"}


def test_extract_imports_with_aliases():
    from faultline.analyzer.symbol_graph import (
        _extract_go_imports_with_aliases,
    )
    src = (
        'package main\n'
        'import (\n'
        '    "fmt"\n'
        '    j "encoding/json"\n'
        '    _ "github.com/myapp/init"\n'
        '    "github.com/myapp/handlers"\n'
        ')\n'
    )
    out = _extract_go_imports_with_aliases(src)
    assert out["fmt"] == "fmt"  # default = last segment
    assert out["encoding/json"] == "j"  # explicit alias
    assert out["github.com/myapp/init"] == "_"  # side-effect
    assert out["github.com/myapp/handlers"] == "handlers"


def test_extract_call_sites_finds_used_symbols():
    from faultline.analyzer.symbol_graph import _extract_go_call_sites
    src = (
        'h := handlers.NewHandler()\n'
        'handlers.Run()\n'
        'handlers.Shutdown()\n'
        'fmt.Println("hello")\n'
        'user.Name = "x"\n'  # struct field — alias not in import map
    )
    alias_to_path = {"handlers": "github.com/myapp/handlers"}
    calls = _extract_go_call_sites(src, alias_to_path)
    assert calls == {
        "github.com/myapp/handlers": {"NewHandler", "Run", "Shutdown"},
    }


def test_call_sites_skip_unexported_symbols():
    """Lowercase symbols (Go's unexported) are ignored — they can't
    be referenced from another package by definition."""
    from faultline.analyzer.symbol_graph import _extract_go_call_sites
    src = "handlers.run()\nhandlers.private()\nhandlers.Public()\n"
    alias_to_path = {"handlers": "github.com/myapp/handlers"}
    calls = _extract_go_call_sites(src, alias_to_path)
    # Only ``Public`` (capitalized) is detected.
    assert calls == {"github.com/myapp/handlers": {"Public"}}


def test_first_go_file_skips_test_files():
    files = {"foo_test.go", "foo.go"}
    assert _first_go_file_in_dir("", files) == "foo.go"


# ── End-to-end ─────────────────────────────────────────────────


def test_build_symbol_graph_traces_go_imports(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "go.mod").write_text(
        "module example.com/myapp\n", encoding="utf-8",
    )
    (repo / "main.go").write_text(
        'package main\n'
        'import (\n'
        '    "example.com/myapp/handlers"\n'
        '    "fmt"\n'
        ')\n'
        'func main() { handlers.Run(); fmt.Println("ok") }\n',
        encoding="utf-8",
    )
    (repo / "handlers").mkdir()
    (repo / "handlers" / "run.go").write_text(
        'package handlers\nfunc Run() {}\n', encoding="utf-8",
    )

    graph = build_symbol_graph(
        str(repo),
        ["main.go", "handlers/run.go"],
        include_http_edges=False,
    )
    main_edges = graph.forward.get("main.go") or []
    # Must have an edge to handlers/run.go
    assert any(e.target_file == "handlers/run.go" for e in main_edges)
    # No edges for stdlib ``fmt``.
    assert not any(e.target_file == "fmt" for e in main_edges)
    # Reverse map populated.
    assert any(
        e.target_file == "main.go"
        for e in (graph.reverse.get("handlers/run.go") or [])
    )
