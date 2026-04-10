"""Tests for the regex-based TypeScript/JavaScript signature extractor."""
import textwrap
from pathlib import Path
import pytest

from faultline.analyzer.ast_extractor import (
    extract_signatures,
    FileSignature,
    _parse_file,
)


# ---------------------------------------------------------------------------
# Unit tests for _parse_file (no filesystem needed)
# ---------------------------------------------------------------------------

def test_extracts_named_function_export():
    source = "export function LoginForm() { return null; }"
    sig = _parse_file("components/LoginForm.tsx", source)
    assert "LoginForm" in sig.exports


def test_extracts_async_function_export():
    source = "export async function fetchUser(id: string) {}"
    sig = _parse_file("api/users.ts", source)
    assert "fetchUser" in sig.exports


def test_extracts_const_export():
    source = "export const useAuth = () => { return {}; };"
    sig = _parse_file("hooks/useAuth.ts", source)
    assert "useAuth" in sig.exports


def test_extracts_class_export():
    source = "export class AuthService { login() {} }"
    sig = _parse_file("services/auth.ts", source)
    assert "AuthService" in sig.exports


def test_extracts_default_function_export():
    source = "export default function CheckoutPage() { return null; }"
    sig = _parse_file("pages/checkout.tsx", source)
    assert "CheckoutPage" in sig.exports


def test_extracts_reexport_block():
    source = "export { LoginForm, useAuth, AuthService as Auth };"
    sig = _parse_file("index.ts", source)
    assert "LoginForm" in sig.exports
    assert "useAuth" in sig.exports
    assert "Auth" in sig.exports        # re-export alias


def test_extracts_nextjs_app_router_methods():
    source = textwrap.dedent("""\
        export async function GET(request: Request) {}
        export async function POST(request: Request) {}
    """)
    sig = _parse_file("app/api/auth/login/route.ts", source)
    assert any("GET" in r for r in sig.routes)
    assert any("POST" in r for r in sig.routes)


def test_extracts_nextjs_page_data_fetchers():
    source = textwrap.dedent("""\
        export async function getServerSideProps(context) { return { props: {} }; }
    """)
    sig = _parse_file("pages/dashboard.tsx", source)
    assert "getServerSideProps" in sig.routes


def test_extracts_express_routes():
    source = textwrap.dedent("""\
        router.get('/users', getUsers);
        router.post('/users', createUser);
        app.delete('/users/:id', deleteUser);
    """)
    sig = _parse_file("routes/users.ts", source)
    route_strs = " ".join(sig.routes)
    assert "GET" in route_strs and "/users" in route_strs
    assert "POST" in route_strs
    assert "DELETE" in route_strs


def test_extracts_relative_imports():
    source = textwrap.dedent("""\
        import { useAuth } from './useAuth';
        import { api } from '@/lib/api';
        import React from 'react';
    """)
    sig = _parse_file("components/Login.tsx", source)
    assert "./useAuth" in sig.imports
    assert "@/lib/api" in sig.imports
    assert "react" not in sig.imports       # node_modules excluded


def test_python_file_returns_empty_signature():
    sig = _parse_file("services/auth.py", "def login(): pass")
    assert sig.is_empty()


def test_non_supported_file_skipped_by_extract_signatures(tmp_path):
    rs_file = tmp_path / "auth.rs"
    rs_file.write_text("fn login() {}")
    result = extract_signatures(["auth.rs"], str(tmp_path))
    assert "auth.rs" not in result


def test_python_file_included_by_extract_signatures(tmp_path):
    py_file = tmp_path / "auth.py"
    py_file.write_text("def login(): pass")
    result = extract_signatures(["auth.py"], str(tmp_path))
    assert "auth.py" in result
    assert "login" in result["auth.py"].exports


def test_extract_signatures_reads_real_file(tmp_path):
    ts_file = tmp_path / "LoginForm.tsx"
    ts_file.write_text("export function LoginForm() { return null; }")
    result = extract_signatures(["LoginForm.tsx"], str(tmp_path))
    assert "LoginForm.tsx" in result
    assert "LoginForm" in result["LoginForm.tsx"].exports


def test_missing_file_is_silently_skipped(tmp_path):
    result = extract_signatures(["nonexistent.ts"], str(tmp_path))
    assert result == {}


def test_file_with_no_exports_excluded_from_result(tmp_path):
    ts_file = tmp_path / "types.ts"
    ts_file.write_text("interface User { id: string; name: string; }")
    result = extract_signatures(["types.ts"], str(tmp_path))
    # No exports/routes/imports → excluded from results
    assert "types.ts" not in result


def test_to_prompt_line_formats_correctly():
    sig = FileSignature(
        path="api/auth.ts",
        exports=["login", "logout"],
        routes=["POST /api/login"],
    )
    line = sig.to_prompt_line()
    assert "api/auth.ts" in line
    assert "login" in line
    assert "POST /api/login" in line
