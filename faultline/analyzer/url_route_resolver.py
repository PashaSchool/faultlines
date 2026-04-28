"""Fetch URL ↔ server route mapping (Improvement #6).

Sprint 7's static-import call-graph cannot cross the
client-server boundary: ``fetch('/api/users')`` reaches no file
via imports because the URL is resolved at runtime by the
browser/HTTP layer. Result: flow traces stop at the API client
and never show the actual server handler.

This module fills that gap. Two passes:

  1. Build a **route registry** for the repo: every server-side
     handler keyed by (HTTP method, normalized URL pattern).
     Sources:
       - Next.js file-based routing (``app/**/route.ts``,
         ``app/**/page.tsx``, ``pages/api/**``)
       - Express / Hono ``app.get('/path', ...)`` etc.
       - FastAPI / Flask ``@app.get('/path')`` etc.
       - tRPC ``procedure.query/mutation`` (matched by procedure
         name, not URL)

  2. Walk every source file and extract **client URL calls**:
     ``fetch('/api/users')``, ``axios.get('/api/users')``,
     ``useQuery({queryKey: ['users']})`` patterns,
     ``trpc.users.list.useQuery()``.

The output is a list of synthetic edges
``(client_file, server_file)`` that the Sprint 7 BFS walker can
overlay on the static import graph. With these edges in place,
a flow that starts at a UI component traces all the way through
the HTTP boundary to the server handler.

Status: Day 1 — extractor + registry only. Day 2 wires it into
``symbol_graph`` as virtual edges. No LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


# ── Data shapes ─────────────────────────────────────────────────


@dataclass(frozen=True)
class RouteRegistration:
    """One server-side route handler discovered in the repo."""

    file: str
    method: str  # GET / POST / PUT / PATCH / DELETE / "*" for any
    pattern: str  # normalized URL or tRPC procedure path
    framework: str  # nextjs-app | nextjs-pages | express | fastapi | trpc
    line: int = 0


@dataclass(frozen=True)
class ClientCall:
    """One client-side URL or tRPC call discovered in source."""

    file: str
    method: str  # GET / POST / ... / "*" when method unknown
    pattern: str
    kind: str  # "fetch" | "axios" | "trpc" | "react-query"
    line: int = 0


@dataclass
class RouteRegistry:
    """All server routes + client calls for a repo."""

    routes: list[RouteRegistration] = field(default_factory=list)
    calls: list[ClientCall] = field(default_factory=list)

    def server_files(self) -> set[str]:
        return {r.file for r in self.routes}

    def client_files(self) -> set[str]:
        return {c.file for c in self.calls}


# ── Server-route extractors ─────────────────────────────────────


# Next.js app router: app/**/route.{ts,js} declares HTTP handlers
# inline by exporting GET, POST, ... functions. Filename signals
# "is a route"; content signals "which methods".
_RE_APP_ROUTER_FILE = re.compile(
    r"(?:^|/)app/(.+)/route\.(?:ts|js|mjs)$"
)
_RE_APP_ROUTER_METHODS = re.compile(
    r"export\s+(?:async\s+)?function\s+"
    r"(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b"
)
# Pages router: pages/api/**/*.ts (each file is one handler).
_RE_PAGES_API_FILE = re.compile(r"(?:^|/)pages/api/(.+)\.(?:ts|js)$")

# Express / Hono / Koa: ``app.get('/path', handler)`` etc.
_RE_EXPRESS_ROUTE = re.compile(
    r"\b(?:app|router|api)\.(?P<method>get|post|put|patch|delete|all|use|head|options)"
    r"\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
)

# FastAPI / Flask / Starlette decorators
_RE_FASTAPI_ROUTE = re.compile(
    r"@(?:app|router|blueprint)\.(?P<method>get|post|put|patch|delete|api_route|route)"
    r"\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
)

# tRPC v10/v11 procedure declarations:
#   userRouter = router({ list: protectedProcedure.query(...) })
# We extract the local symbol name (``list``) — Day 2 will join
# this against client-side ``trpc.users.list.useQuery()`` calls.
_RE_TRPC_PROCEDURE = re.compile(
    r"(?P<name>[A-Za-z_]\w*)\s*:\s*(?:t\.)?"
    r"(?:protected|public)?[Pp]rocedure[\s\S]{0,200}?"
    r"\.(?:query|mutation|subscription)\s*\("
)


def _normalize_path(path: str) -> str:
    """Lowercase + collapse trailing slashes + parameterize.

    ``/users/[id]`` and ``/users/:id`` and ``/users/{id}`` all
    normalize to the same canonical form so client/server matching
    works regardless of framework convention.
    """
    p = path.strip().lower().rstrip("/")
    if not p.startswith("/"):
        p = "/" + p
    # :param → :*  /  [param] → :*  /  {param} → :*
    p = re.sub(r":[a-z0-9_-]+", ":*", p, flags=re.IGNORECASE)
    p = re.sub(r"\[\.{0,3}[a-z0-9_-]+\]", ":*", p, flags=re.IGNORECASE)
    p = re.sub(r"\{[a-z0-9_-]+\}", ":*", p, flags=re.IGNORECASE)
    return p


def _routes_from_nextjs_app_file(
    rel_path: str, source: str,
) -> list[RouteRegistration]:
    m = _RE_APP_ROUTER_FILE.search(rel_path)
    if not m:
        return []
    pattern_raw = "/" + m.group(1)
    pattern = _normalize_path(pattern_raw)
    methods = {match.group(1).upper() for match in _RE_APP_ROUTER_METHODS.finditer(source)}
    if not methods:
        # File exists but no handler exported — could be a layout-
        # adjacent route file. Register as wildcard so the BFS
        # still has SOMETHING to connect.
        methods = {"*"}
    out: list[RouteRegistration] = []
    for method in methods:
        out.append(RouteRegistration(
            file=rel_path, method=method, pattern=pattern,
            framework="nextjs-app",
        ))
    return out


def _routes_from_pages_api_file(rel_path: str) -> list[RouteRegistration]:
    m = _RE_PAGES_API_FILE.search(rel_path)
    if not m:
        return []
    pattern_raw = "/api/" + m.group(1)
    return [RouteRegistration(
        file=rel_path,
        method="*",  # Pages API handler dispatches by req.method internally
        pattern=_normalize_path(pattern_raw),
        framework="nextjs-pages",
    )]


def _routes_from_express_or_fastapi(
    rel_path: str, source: str,
) -> list[RouteRegistration]:
    out: list[RouteRegistration] = []
    for regex, framework in (
        (_RE_EXPRESS_ROUTE, "express"),
        (_RE_FASTAPI_ROUTE, "fastapi"),
    ):
        for m in regex.finditer(source):
            method = m.group("method").upper()
            if method == "USE":
                continue  # middleware, not a route
            if method == "API_ROUTE" or method == "ROUTE":
                method = "*"
            line = source.count("\n", 0, m.start()) + 1
            out.append(RouteRegistration(
                file=rel_path, method=method,
                pattern=_normalize_path(m.group("path")),
                framework=framework, line=line,
            ))
    return out


def _routes_from_trpc(
    rel_path: str, source: str,
) -> list[RouteRegistration]:
    # Heuristic: only inspect files that look like tRPC routers
    # (contain ``router({`` or ``createTRPCRouter``).
    if "router({" not in source and "createTRPCRouter" not in source:
        return []
    out: list[RouteRegistration] = []
    for m in _RE_TRPC_PROCEDURE.finditer(source):
        name = m.group("name")
        # tRPC procedures are addressed by name, not URL. Use a
        # synthetic path ``trpc:<router-segment>/<procedure>``.
        # For Day 1 we just record the procedure name; Day 2's
        # client-side matcher will compare on procedure name
        # within the same file scope.
        line = source.count("\n", 0, m.start()) + 1
        out.append(RouteRegistration(
            file=rel_path, method="*",
            pattern=f"trpc:{name}",
            framework="trpc", line=line,
        ))
    return out


# ── Client-call extractors ──────────────────────────────────────


_RE_FETCH_CALL = re.compile(
    r"\bfetch\s*\(\s*['\"`](?P<path>[^'\"`]+)['\"`]"
    r"(?:\s*,\s*\{\s*method\s*:\s*['\"](?P<method>[A-Z]+)['\"])?",
)
_RE_AXIOS_CALL = re.compile(
    r"\baxios\.(?P<method>get|post|put|patch|delete|head|options)"
    r"\s*\(\s*['\"`](?P<path>[^'\"`]+)['\"`]",
)
_RE_TRPC_CLIENT = re.compile(
    r"\btrpc(?:Client)?\.(?P<router>[A-Za-z_]\w*)\.(?P<proc>[A-Za-z_]\w*)"
    r"\.(?:useQuery|useMutation|useInfiniteQuery|query|mutation)\s*\(",
)


def _calls_from_source(rel_path: str, source: str) -> list[ClientCall]:
    out: list[ClientCall] = []
    for m in _RE_FETCH_CALL.finditer(source):
        path = m.group("path")
        if not path.startswith("/"):
            continue  # external URL, ignore
        method = (m.group("method") or "*").upper()
        line = source.count("\n", 0, m.start()) + 1
        out.append(ClientCall(
            file=rel_path, method=method,
            pattern=_normalize_path(path),
            kind="fetch", line=line,
        ))
    for m in _RE_AXIOS_CALL.finditer(source):
        path = m.group("path")
        if not path.startswith("/"):
            continue
        line = source.count("\n", 0, m.start()) + 1
        out.append(ClientCall(
            file=rel_path,
            method=m.group("method").upper(),
            pattern=_normalize_path(path),
            kind="axios", line=line,
        ))
    for m in _RE_TRPC_CLIENT.finditer(source):
        line = source.count("\n", 0, m.start()) + 1
        out.append(ClientCall(
            file=rel_path, method="*",
            pattern=f"trpc:{m.group('proc')}",
            kind="trpc", line=line,
        ))
    return out


# ── Builder ─────────────────────────────────────────────────────


_TS_JS_EXTS = frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})
_PY_EXTS = frozenset({".py"})


def build_route_registry(
    repo_root: str | Path,
    source_files: list[str],
) -> RouteRegistry:
    """Scan source files and produce a :class:`RouteRegistry`.

    No LLM calls. Best-effort: unreadable files are silently
    skipped; the registry still works on whatever it could parse.
    """
    repo_root = Path(repo_root)
    registry = RouteRegistry()

    for rel_path in source_files:
        suffix = Path(rel_path).suffix.lower()
        if suffix not in _TS_JS_EXTS and suffix not in _PY_EXTS:
            continue

        try:
            source = (repo_root / rel_path).read_text(
                encoding="utf-8", errors="ignore",
            )
        except OSError:
            continue

        # Server-side routes
        if suffix in _TS_JS_EXTS:
            registry.routes.extend(_routes_from_nextjs_app_file(rel_path, source))
            registry.routes.extend(_routes_from_pages_api_file(rel_path))
            registry.routes.extend(_routes_from_express_or_fastapi(rel_path, source))
            registry.routes.extend(_routes_from_trpc(rel_path, source))
        elif suffix in _PY_EXTS:
            registry.routes.extend(_routes_from_express_or_fastapi(rel_path, source))

        # Client-side calls (TS/JS only — Python doesn't ship
        # frontend code that fetches in the relevant sense).
        if suffix in _TS_JS_EXTS:
            registry.calls.extend(_calls_from_source(rel_path, source))

    logger.info(
        "url_route_resolver: %d routes, %d client calls "
        "across %d source files",
        len(registry.routes), len(registry.calls), len(source_files),
    )
    return registry


# ── Edge resolver (for symbol_graph integration in Day 2) ──────


def resolve_call_to_routes(
    call: ClientCall,
    registry: RouteRegistry,
) -> list[str]:
    """Match one ClientCall against the route registry.

    Returns the list of server **file paths** the call could
    reach. Multiple matches are possible when the same path is
    handled by routes in different layers (e.g. an Express app
    AND a Next.js app router on different deployments — rare
    but legal).

    Match rules (in order):
      - Exact pattern AND method match.
      - Exact pattern, method "*" on either side.
      - Pattern with a parameter (``:*``) on the server side
        matching a literal value on the client (``/users/123``
        client → ``/users/:*`` server).
    """
    out: set[str] = set()
    call_pattern = call.pattern
    call_method = call.method

    for route in registry.routes:
        if route.method != "*" and call_method != "*" and route.method != call_method:
            continue
        if route.pattern == call_pattern:
            out.add(route.file)
            continue
        # Parametric match — substitute :* segments with [^/]+ then
        # compare the whole pattern.
        if ":*" in route.pattern:
            parts = route.pattern.split(":*")
            regex = "^" + r"[^/]+".join(re.escape(p) for p in parts) + "$"
            if re.match(regex, call_pattern):
                out.add(route.file)

    return sorted(out)
