"""Tests for ``faultline.analyzer.url_route_resolver`` (Improvement #6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from faultline.analyzer.url_route_resolver import (
    ClientCall,
    RouteRegistration,
    RouteRegistry,
    _normalize_path,
    build_route_registry,
    resolve_call_to_routes,
)


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# ── _normalize_path ──────────────────────────────────────────────


class TestNormalizePath:
    def test_lowercases(self):
        assert _normalize_path("/Users") == "/users"

    def test_trailing_slash_stripped(self):
        assert _normalize_path("/users/") == "/users"

    def test_leading_slash_added(self):
        assert _normalize_path("users") == "/users"

    def test_express_param(self):
        assert _normalize_path("/users/:id") == "/users/:*"

    def test_nextjs_bracket_param(self):
        assert _normalize_path("/users/[id]") == "/users/:*"

    def test_curly_param(self):
        assert _normalize_path("/users/{id}/posts") == "/users/:*/posts"

    def test_catchall_bracket(self):
        assert _normalize_path("/files/[...slug]") == "/files/:*"


# ── server-route extraction ─────────────────────────────────────


class TestServerRouteExtraction:
    def test_nextjs_app_router_methods(self, tmp_path: Path):
        _write(tmp_path, "app/users/route.ts",
               "export async function GET() { return new Response(); }\n"
               "export async function POST() { return new Response(); }\n")
        reg = build_route_registry(tmp_path, ["app/users/route.ts"])
        methods = {r.method for r in reg.routes if r.framework == "nextjs-app"}
        assert methods == {"GET", "POST"}
        patterns = {r.pattern for r in reg.routes if r.framework == "nextjs-app"}
        assert "/users" in patterns

    def test_nextjs_app_router_with_param(self, tmp_path: Path):
        _write(tmp_path, "app/users/[id]/route.ts",
               "export async function GET() { return new Response(); }\n")
        reg = build_route_registry(tmp_path, ["app/users/[id]/route.ts"])
        assert any(r.pattern == "/users/:*" for r in reg.routes)

    def test_pages_api_route(self, tmp_path: Path):
        _write(tmp_path, "pages/api/users/login.ts",
               "export default function handler(req, res) { res.json({}); }\n")
        reg = build_route_registry(tmp_path, ["pages/api/users/login.ts"])
        assert any(r.pattern == "/api/users/login" and r.framework == "nextjs-pages"
                   for r in reg.routes)

    def test_express_routes(self, tmp_path: Path):
        _write(tmp_path, "server.ts",
               "import express from 'express';\n"
               "const app = express();\n"
               "app.get('/users', listUsers);\n"
               "app.post('/users/:id', updateUser);\n"
               "app.use('/middleware', mw);\n")
        reg = build_route_registry(tmp_path, ["server.ts"])
        # use() is middleware, not a route → excluded
        methods = [(r.method, r.pattern) for r in reg.routes if r.framework == "express"]
        assert ("GET", "/users") in methods
        assert ("POST", "/users/:*") in methods
        assert all(p != "/middleware" for _, p in methods)

    def test_fastapi_decorator(self, tmp_path: Path):
        _write(tmp_path, "api/users.py",
               "from fastapi import APIRouter\n"
               "router = APIRouter()\n"
               "@router.get('/users')\n"
               "async def list_users(): return []\n"
               "@router.post('/users/{user_id}')\n"
               "async def update_user(user_id: int): pass\n")
        reg = build_route_registry(tmp_path, ["api/users.py"])
        methods = [(r.method, r.pattern) for r in reg.routes if r.framework == "fastapi"]
        assert ("GET", "/users") in methods
        assert ("POST", "/users/:*") in methods

    def test_trpc_procedures(self, tmp_path: Path):
        _write(tmp_path, "trpc/users.ts",
               "export const usersRouter = router({\n"
               "  list: protectedProcedure.query(({ ctx }) => ctx.users),\n"
               "  create: protectedProcedure.input(z.object({})).mutation(({ input }) => input),\n"
               "});\n")
        reg = build_route_registry(tmp_path, ["trpc/users.ts"])
        names = {r.pattern for r in reg.routes if r.framework == "trpc"}
        assert "trpc:list" in names
        assert "trpc:create" in names

    def test_trpc_standalone_procedure_with_route_suffix(self, tmp_path: Path):
        # Documenso-style: one file = one procedure, exported with
        # ``Route`` suffix. Client calls strip the suffix.
        _write(tmp_path, "trpc/recipient/find-suggestions.ts",
               "import { authenticatedProcedure } from '../trpc';\n"
               "export const findRecipientSuggestionsRoute = authenticatedProcedure\n"
               "  .input(ZGetRecipientSuggestionsRequestSchema)\n"
               "  .output(ZGetRecipientSuggestionsResponseSchema)\n"
               "  .query(async ({ input, ctx }) => { return []; });\n")
        reg = build_route_registry(tmp_path, ["trpc/recipient/find-suggestions.ts"])
        names = {r.pattern for r in reg.routes if r.framework == "trpc"}
        # Suffix stripped — client-call ``findRecipientSuggestions``
        # matches without needing the ``Route`` part
        assert "trpc:findRecipientSuggestions" in names

    def test_trpc_standalone_with_authenticated_procedure(self, tmp_path: Path):
        # ``authenticatedProcedure`` is documenso's flavour
        _write(tmp_path, "trpc/billing/checkout.ts",
               "export const checkoutRoute = authenticatedProcedure\n"
               "  .mutation(({ ctx }) => null);\n")
        reg = build_route_registry(tmp_path, ["trpc/billing/checkout.ts"])
        names = {r.pattern for r in reg.routes if r.framework == "trpc"}
        assert "trpc:checkout" in names

    def test_trpc_standalone_without_suffix(self, tmp_path: Path):
        # Plain ``export const send = procedure...`` no suffix —
        # client calls match the bare name.
        _write(tmp_path, "trpc/send.ts",
               "export const send = procedure.mutation(() => {});\n")
        reg = build_route_registry(tmp_path, ["trpc/send.ts"])
        names = {r.pattern for r in reg.routes if r.framework == "trpc"}
        assert "trpc:send" in names

    def test_trpc_handler_suffix(self, tmp_path: Path):
        _write(tmp_path, "trpc/x.ts",
               "export const createUserHandler = procedure.mutation(() => {});\n")
        reg = build_route_registry(tmp_path, ["trpc/x.ts"])
        names = {r.pattern for r in reg.routes if r.framework == "trpc"}
        assert "trpc:createUser" in names

    def test_trpc_no_double_emit(self, tmp_path: Path):
        # File has BOTH a router block AND a standalone export with
        # the same name. We should emit it once.
        _write(tmp_path, "trpc/dup.ts",
               "export const usersRouter = router({\n"
               "  list: procedure.query(() => []),\n"
               "});\n"
               "export const list = procedure.query(() => []);\n")
        reg = build_route_registry(tmp_path, ["trpc/dup.ts"])
        list_routes = [r for r in reg.routes if r.pattern == "trpc:list"]
        assert len(list_routes) == 1

    def test_no_route_in_plain_file(self, tmp_path: Path):
        _write(tmp_path, "lib/utils.ts", "export const x = 1;\n")
        reg = build_route_registry(tmp_path, ["lib/utils.ts"])
        assert reg.routes == []


# ── client-call extraction ──────────────────────────────────────


class TestClientCallExtraction:
    def test_fetch_path(self, tmp_path: Path):
        _write(tmp_path, "client.ts",
               "const r = await fetch('/api/users');\n"
               "const p = await fetch('/api/users', { method: 'POST' });\n")
        reg = build_route_registry(tmp_path, ["client.ts"])
        kinds = [(c.method, c.pattern) for c in reg.calls if c.kind == "fetch"]
        assert ("*", "/api/users") in kinds
        assert ("POST", "/api/users") in kinds

    def test_axios_methods(self, tmp_path: Path):
        _write(tmp_path, "api-client.ts",
               "axios.get('/api/users');\n"
               "axios.post('/api/users', data);\n"
               "axios.delete('/api/users/123');\n")
        reg = build_route_registry(tmp_path, ["api-client.ts"])
        triples = [(c.method, c.pattern, c.kind) for c in reg.calls]
        assert ("GET", "/api/users", "axios") in triples
        assert ("POST", "/api/users", "axios") in triples
        assert ("DELETE", "/api/users/123", "axios") in triples

    def test_external_urls_skipped(self, tmp_path: Path):
        _write(tmp_path, "client.ts",
               "fetch('https://api.example.com/users');\n"
               "axios.get('http://other.com/x');\n")
        reg = build_route_registry(tmp_path, ["client.ts"])
        # External URLs (don't start with /) ignored
        assert reg.calls == []

    def test_trpc_client(self, tmp_path: Path):
        _write(tmp_path, "page.tsx",
               "const { data } = trpc.users.list.useQuery();\n"
               "const m = trpc.users.create.useMutation();\n"
               "const c = trpcClient.documents.send.mutation();\n")
        reg = build_route_registry(tmp_path, ["page.tsx"])
        patterns = {c.pattern for c in reg.calls if c.kind == "trpc"}
        assert "trpc:list" in patterns
        assert "trpc:create" in patterns
        assert "trpc:send" in patterns


# ── resolve_call_to_routes ──────────────────────────────────────


class TestResolveCallToRoutes:
    def test_exact_match(self):
        reg = RouteRegistry(routes=[
            RouteRegistration(file="server.ts", method="GET", pattern="/users", framework="express"),
        ])
        call = ClientCall(file="client.ts", method="GET", pattern="/users", kind="fetch")
        assert resolve_call_to_routes(call, reg) == ["server.ts"]

    def test_method_mismatch(self):
        reg = RouteRegistry(routes=[
            RouteRegistration(file="server.ts", method="GET", pattern="/users", framework="express"),
        ])
        call = ClientCall(file="client.ts", method="POST", pattern="/users", kind="fetch")
        assert resolve_call_to_routes(call, reg) == []

    def test_wildcard_method(self):
        reg = RouteRegistry(routes=[
            RouteRegistration(file="server.ts", method="*", pattern="/users", framework="nextjs-pages"),
        ])
        call = ClientCall(file="client.ts", method="POST", pattern="/users", kind="fetch")
        assert resolve_call_to_routes(call, reg) == ["server.ts"]

    def test_parametric_route(self):
        reg = RouteRegistry(routes=[
            RouteRegistration(file="server.ts", method="GET", pattern="/users/:*", framework="express"),
        ])
        call = ClientCall(file="client.ts", method="GET", pattern="/users/123", kind="fetch")
        assert resolve_call_to_routes(call, reg) == ["server.ts"]

    def test_no_match(self):
        reg = RouteRegistry(routes=[
            RouteRegistration(file="server.ts", method="GET", pattern="/posts", framework="express"),
        ])
        call = ClientCall(file="client.ts", method="GET", pattern="/users", kind="fetch")
        assert resolve_call_to_routes(call, reg) == []

    def test_trpc_procedure_match(self):
        reg = RouteRegistry(routes=[
            RouteRegistration(file="trpc/users.ts", method="*", pattern="trpc:list", framework="trpc"),
        ])
        call = ClientCall(file="page.tsx", method="*", pattern="trpc:list", kind="trpc")
        assert resolve_call_to_routes(call, reg) == ["trpc/users.ts"]

    def test_multiple_servers_for_same_path(self):
        reg = RouteRegistry(routes=[
            RouteRegistration(file="legacy.ts", method="GET", pattern="/users", framework="express"),
            RouteRegistration(file="modern.ts", method="GET", pattern="/users", framework="nextjs-app"),
        ])
        call = ClientCall(file="c.ts", method="GET", pattern="/users", kind="fetch")
        files = resolve_call_to_routes(call, reg)
        assert set(files) == {"legacy.ts", "modern.ts"}


# ── Integration: end-to-end client → server resolution ─────────


class TestEndToEnd:
    def test_full_pipeline_express(self, tmp_path: Path):
        # Server: Express handler
        _write(tmp_path, "backend/server.ts",
               "app.post('/api/documents', createDocument);\n")
        # Client: fetch call
        _write(tmp_path, "frontend/upload.tsx",
               "const r = await fetch('/api/documents', { method: 'POST' });\n")
        reg = build_route_registry(tmp_path, [
            "backend/server.ts", "frontend/upload.tsx",
        ])
        # Find the fetch call and resolve it
        calls = [c for c in reg.calls if c.file == "frontend/upload.tsx"]
        assert len(calls) == 1
        targets = resolve_call_to_routes(calls[0], reg)
        assert targets == ["backend/server.ts"]

    def test_full_pipeline_trpc(self, tmp_path: Path):
        _write(tmp_path, "trpc/router.ts",
               "export const documentsRouter = router({\n"
               "  send: protectedProcedure.input(z.object({})).mutation(({input}) => {}),\n"
               "});\n")
        _write(tmp_path, "ui/SendButton.tsx",
               "const send = trpc.documents.send.useMutation();\n")
        reg = build_route_registry(tmp_path, [
            "trpc/router.ts", "ui/SendButton.tsx",
        ])
        client_calls = [c for c in reg.calls if c.kind == "trpc"]
        assert len(client_calls) == 1
        targets = resolve_call_to_routes(client_calls[0], reg)
        assert targets == ["trpc/router.ts"]
