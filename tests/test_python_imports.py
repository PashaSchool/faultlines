"""Tests for the Python import resolver in symbol_graph (P1)."""

from __future__ import annotations

from faultline.analyzer.symbol_graph import (
    _extract_py_imports,
    _resolve_py_module,
    build_symbol_graph,
)


# ── Parsing ─────────────────────────────────────────────────────


def test_extract_from_import_single():
    src = "from .module import foo\n"
    out = _extract_py_imports(src)
    assert out == {".module": {"foo"}}


def test_extract_from_import_multiple_names():
    src = "from a.b import x, y, z\n"
    out = _extract_py_imports(src)
    assert out["a.b"] == {"x", "y", "z"}


def test_extract_from_import_aliased():
    src = "from foo import bar as baz, qux\n"
    out = _extract_py_imports(src)
    assert out["foo"] == {"bar", "qux"}


def test_extract_from_import_parenthesized():
    src = (
        "from a.b import (\n"
        "    x,\n"
        "    y,\n"
        ")\n"
    )
    out = _extract_py_imports(src)
    assert out["a.b"] == {"x", "y"}


def test_extract_star_import():
    src = "from foo import *\n"
    out = _extract_py_imports(src)
    assert out == {"foo": {"*"}}


def test_extract_bare_import():
    src = "import os\nimport package.sub\n"
    out = _extract_py_imports(src)
    assert "os" in out
    assert out["os"] == {"@import"}
    assert "package.sub" in out


def test_extract_relative_dots():
    src = "from .. import sibling\nfrom ...top import thing\n"
    out = _extract_py_imports(src)
    assert ".." in out and "sibling" in out[".."]
    # Three dots followed by ``top`` parses as one composite module
    # name — ``...top`` (3 levels up + ``top`` package).
    assert "...top" in out and "thing" in out["...top"]


# ── Resolution ──────────────────────────────────────────────────


def test_resolve_relative_single_dot():
    files = {"apps/api/views/foo.py", "apps/api/views/bar.py"}
    target = _resolve_py_module("apps/api/views/foo.py", ".bar", files)
    assert target == "apps/api/views/bar.py"


def test_resolve_relative_two_dots_up():
    files = {
        "apps/api/views/foo.py",
        "apps/api/serializers.py",
    }
    target = _resolve_py_module(
        "apps/api/views/foo.py", "..serializers", files,
    )
    assert target == "apps/api/serializers.py"


def test_resolve_resolves_to_init():
    files = {
        "apps/api/views/foo.py",
        "apps/api/utils/__init__.py",
    }
    target = _resolve_py_module(
        "apps/api/views/foo.py", "..utils", files,
    )
    assert target == "apps/api/utils/__init__.py"


def test_resolve_absolute_with_apps_prefix():
    files = {
        "apps/api/views/foo.py",
        "apps/api/serializers.py",
    }
    # Code says ``from api.serializers import X`` — resolves under
    # ``apps/`` prefix.
    target = _resolve_py_module(
        "apps/api/views/foo.py", "api.serializers", files,
    )
    assert target == "apps/api/serializers.py"


def test_resolve_returns_none_for_stdlib():
    files = {"src/app.py"}
    assert _resolve_py_module("src/app.py", "os", files) is None
    assert _resolve_py_module("src/app.py", "json", files) is None


def test_resolve_returns_none_for_third_party():
    files = {"src/app.py"}
    assert _resolve_py_module("src/app.py", "django.urls", files) is None


# ── End-to-end on a synthetic repo ──────────────────────────────


def test_build_symbol_graph_traces_python_import_chain(tmp_path):
    """Three Python files chained via relative imports produce a
    connected graph with forward + reverse edges."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text(
        "from .views import handler\n", encoding="utf-8",
    )
    (repo / "src" / "views.py").write_text(
        "from .models import Item\n"
        "def handler(req):\n    return Item.list()\n",
        encoding="utf-8",
    )
    (repo / "src" / "models.py").write_text(
        "class Item:\n    @staticmethod\n    def list():\n        return []\n",
        encoding="utf-8",
    )

    graph = build_symbol_graph(
        str(repo),
        ["src/main.py", "src/views.py", "src/models.py"],
        include_http_edges=False,
    )
    # main.py should have an outgoing edge to views.py.
    main_edges = graph.forward.get("src/main.py") or []
    assert any(
        e.target_file == "src/views.py" and e.target_symbol == "handler"
        for e in main_edges
    )
    # views.py should have outgoing to models.py.
    views_edges = graph.forward.get("src/views.py") or []
    assert any(e.target_file == "src/models.py" for e in views_edges)
    # Reverse map: models.py is imported by views.py.
    rev = graph.reverse.get("src/models.py") or []
    assert any(e.target_file == "src/views.py" for e in rev)
