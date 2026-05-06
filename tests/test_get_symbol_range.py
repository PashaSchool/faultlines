"""Sprint 12 Day 4 — get_symbol_range / list_exported_symbols tests."""

from __future__ import annotations

from faultline.analyzer.ast_extractor import (
    get_symbol_range,
    list_exported_symbols,
)


# ── TS/JS ─────────────────────────────────────────────────────────────


_TS_SOURCE = """\
import { foo } from './foo';

export const TIMEOUT = 5000;

export function login(email: string, password: string) {
  return api.post('/login', { email, password });
}

export function logout() {
  return api.post('/logout');
}

export class AuthService {
  signin() {}
  signout() {}
}
"""


def test_get_symbol_range_ts_function():
    r = get_symbol_range("web/auth.ts", _TS_SOURCE, "login")
    assert r is not None
    assert r.kind == "function"
    assert r.start_line >= 5
    assert r.end_line > r.start_line


def test_get_symbol_range_ts_class():
    r = get_symbol_range("web/auth.ts", _TS_SOURCE, "AuthService")
    assert r is not None
    assert r.kind == "class"


def test_get_symbol_range_ts_const():
    r = get_symbol_range("web/auth.ts", _TS_SOURCE, "TIMEOUT")
    assert r is not None
    assert r.kind == "const"


def test_get_symbol_range_unknown_symbol():
    assert get_symbol_range("web/auth.ts", _TS_SOURCE, "doesNotExist") is None


def test_list_exported_symbols_ts():
    syms = list_exported_symbols("web/auth.ts", _TS_SOURCE)
    names = {s.name for s in syms}
    assert {"TIMEOUT", "login", "logout", "AuthService"} <= names


# ── Python ────────────────────────────────────────────────────────────


_PY_SOURCE = '''\
from typing import Any


class AuthHandler:
    def login(self, email: str) -> Any:
        return {"ok": True}


def reset_password(email: str) -> None:
    print("reset")


def _internal_helper() -> None:
    pass
'''


def test_get_symbol_range_python_class():
    r = get_symbol_range("server/auth.py", _PY_SOURCE, "AuthHandler")
    assert r is not None
    assert r.kind == "class"


def test_get_symbol_range_python_function():
    r = get_symbol_range("server/auth.py", _PY_SOURCE, "reset_password")
    assert r is not None
    assert r.kind == "function"


def test_get_symbol_range_python_skips_private():
    """Python parser may skip _-prefixed (private). Acceptable."""
    # We don't assert one way; just document current behaviour.
    r = get_symbol_range("server/auth.py", _PY_SOURCE, "_internal_helper")
    assert r is None or r.kind == "function"


# ── Edge cases ────────────────────────────────────────────────────────


def test_get_symbol_range_empty_symbol_returns_none():
    assert get_symbol_range("foo.ts", "export const X = 1;", "") is None


def test_get_symbol_range_unsupported_extension():
    assert get_symbol_range("foo.go", "package main\nfunc Login() {}", "Login") is None


def test_get_symbol_range_no_extension():
    assert get_symbol_range("Makefile", "all:\n\techo hi", "all") is None


def test_list_exported_symbols_unsupported_returns_empty():
    assert list_exported_symbols("foo.go", "package main") == []


def test_get_symbol_range_jsx_alias():
    src = "export function Button() {}"
    r = get_symbol_range("ui/Button.jsx", src, "Button")
    assert r is not None
    assert r.kind == "function"
