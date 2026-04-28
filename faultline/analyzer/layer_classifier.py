"""Layer classifier (Sprint 7 Day 3).

Classify a source file into one of six **layers** so the dashboard
can render a flow as a top-to-bottom slice (UI → state → API client
→ API server → schema). The classifier is two stages:

    1. Path patterns — match against known framework conventions
       (Next.js / Remix / tRPC / FastAPI / Prisma / Django / Vue).
       Settles ~80-90% of files cheaply.
    2. Content patterns — when path doesn't decide, scan for
       state-store factories (createSlice / atom / create<>),
       schema definitions, route decorators, etc.

Files we can't classify confidently fall back to ``support`` —
shared utilities visible in the trace but visually muted.

A future stage (Day 3 polish or Day 4) wires a Haiku tiebreaker for
genuinely ambiguous files; the LLM call is gated and capped.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal


logger = logging.getLogger(__name__)


Layer = Literal["ui", "state", "api-client", "api-server", "schema", "support"]


# ── Path patterns ─────────────────────────────────────────────────


# Order matters — first match wins. Keep more specific patterns
# above more general ones.
_PATH_PATTERNS: tuple[tuple[Layer, re.Pattern], ...] = (
    # ── Schema (most specific) ────────────────────────────────────
    ("schema", re.compile(r"(?:^|/)(?:prisma/schema\.prisma|schema\.prisma)$")),
    ("schema", re.compile(r"(?:^|/)prisma/migrations/")),
    ("schema", re.compile(r"(?:^|/)migrations/.*\.(?:sql|py|ts|js)$")),
    ("schema", re.compile(r"(?:^|/)db/.*schema\.(?:ts|js|sql)$")),
    ("schema", re.compile(r"(?:^|/)drizzle\.(?:config|schema)\.")),
    ("schema", re.compile(r"(?:^|/)models/.*\.py$")),  # SQLAlchemy
    ("schema", re.compile(r"(?:^|/)models\.py$")),       # Django flat-app style
    # Python data-model conventions
    ("schema", re.compile(r"(?:^|/)schemas?/.*\.py$")),
    ("schema", re.compile(r"(?:^|/)types\.py$")),         # Pydantic types module
    ("schema", re.compile(r"(?:^|/)entities/.*\.py$")),   # SQLAlchemy entities

    # ── API server (route handlers, procedures) ───────────────────
    # Next.js app router route handlers
    ("api-server", re.compile(r"(?:^|/)app/.+/route\.(?:ts|js|mjs)$")),
    # Next.js pages router api endpoints
    ("api-server", re.compile(r"(?:^|/)pages/api/")),
    # Remix loaders / actions live in routes/*.tsx — UI-side, not
    # api-server. So we don't put a routes/* api rule here.
    # tRPC server-side
    ("api-server", re.compile(r"(?:^|/)trpc/(?:server|router)/")),
    ("api-server", re.compile(r"(?:^|/)server/(?:trpc|router)/")),
    # Server-only helpers
    ("api-server", re.compile(r"(?:^|/)server-only/")),
    ("api-server", re.compile(r"(?:^|/)lib/.+/server-only/")),
    ("api-server", re.compile(r"(?:^|/)server\.(?:ts|js|py)$")),
    # FastAPI / Flask / NestJS conventions
    ("api-server", re.compile(r"(?:^|/)routers/.+\.py$")),
    ("api-server", re.compile(r"(?:^|/)controllers/")),
    ("api-server", re.compile(r"(?:^|/)resolvers/")),
    ("api-server", re.compile(r"(?:^|/)endpoints/")),
    ("api-server", re.compile(r"(?:^|/)views/.*\.py$")),  # Django views/
    ("api-server", re.compile(r"(?:^|/)views\.py$")),      # Django flat views.py
    ("api-server", re.compile(r"(?:^|/)urls\.py$")),       # Django urls.py
    # Python CLI entry points — Click/Typer/argparse apps
    ("api-server", re.compile(r"(?:^|/)cli\.py$")),
    ("api-server", re.compile(r"(?:^|/)cli/.+\.py$")),
    ("api-server", re.compile(r"(?:^|/)__main__\.py$")),
    ("api-server", re.compile(r"(?:^|/)cmd/.+\.(?:py|go|rs)$")),
    ("api-server", re.compile(r"(?:^|/)main\.(?:py|go|rs)$")),
    # FastAPI / Starlette / Flask app modules
    ("api-server", re.compile(r"(?:^|/)app\.py$")),
    ("api-server", re.compile(r"(?:^|/)server\.py$")),
    ("api-server", re.compile(r"(?:^|/)api/.+\.py$")),

    # ── State (stores, contexts, hooks-as-state) ─────────────────
    ("state", re.compile(r"(?:^|/)stores?/")),
    ("state", re.compile(r"(?:^|/)slices?/")),
    ("state", re.compile(r"(?:^|/)atoms?/")),  # Jotai / Recoil
    ("state", re.compile(r"(?:^|/)recoil/")),
    ("state", re.compile(r"(?:^|/)redux/")),
    ("state", re.compile(r"(?:^|/)zustand/")),
    ("state", re.compile(r"(?:^|/)contexts?/.*\.(?:tsx?|jsx?)$")),
    ("state", re.compile(r"[A-Za-z]+Store\.(?:ts|tsx|js|jsx)$")),
    ("state", re.compile(r"[A-Za-z]+Slice\.(?:ts|tsx|js|jsx)$")),
    ("state", re.compile(r"[A-Za-z]+Context\.(?:ts|tsx|js|jsx)$")),
    ("state", re.compile(r"[A-Za-z]+Provider\.(?:ts|tsx|js|jsx)$")),

    # ── API client (data fetching) ────────────────────────────────
    ("api-client", re.compile(r"(?:^|/)trpc/client/")),
    ("api-client", re.compile(r"(?:^|/)client/(?:api|trpc)/")),
    ("api-client", re.compile(r"(?:^|/)api/.+\.(?:ts|js|tsx|jsx)$")),
    ("api-client", re.compile(r"(?:^|/)lib/api[/\.]")),
    ("api-client", re.compile(r"[A-Za-z]+Client\.(?:ts|js)$")),

    # ── UI ────────────────────────────────────────────────────────
    ("ui", re.compile(r"(?:^|/)app/.+/(?:page|layout|error|loading|not-found)\.(?:tsx?|jsx?)$")),
    ("ui", re.compile(r"(?:^|/)pages/.*\.(?:tsx?|jsx?)$")),
    ("ui", re.compile(r"(?:^|/)routes/.*\.(?:tsx?|jsx?)$")),  # Remix
    ("ui", re.compile(r"(?:^|/)components/")),
    ("ui", re.compile(r"(?:^|/)views/.*\.(?:tsx?|jsx?)$")),
    ("ui", re.compile(r"(?:^|/)widgets/")),
    ("ui", re.compile(r"(?:^|/)layouts/")),
    ("ui", re.compile(r"(?:^|/)templates/")),
    ("ui", re.compile(r"(?:^|/)dialogs/")),
    ("ui", re.compile(r"(?:^|/)forms/")),
    # Design-system / primitives — common in monorepos
    # (shadcn, radix wrappers, custom DS).
    ("ui", re.compile(r"(?:^|/)primitives/")),
    ("ui", re.compile(r"(?:^|/)design-system/")),
    ("ui", re.compile(r"(?:^|/)ds/")),
    ("ui", re.compile(r"(?:^|/)kit/.+\.(?:tsx?|jsx?)$")),
    # Filename-suffix UI conventions
    ("ui", re.compile(r"[A-Za-z]+(?:Form|Dialog|Modal|Card|Page|Sheet|Drawer|"
                      r"Panel|Banner|Toast|Tooltip|Menu|Tab|List|Item)"
                      r"\.(?:tsx?|jsx?)$")),

    # Hooks live close to state but most are mixed; classify as
    # state when used as state — content regex catches the rest.
    ("state", re.compile(r"(?:^|/)hooks?/")),
)


def classify_path(path: str) -> Layer | None:
    """Apply path patterns; return ``None`` when nothing matches."""
    for layer, regex in _PATH_PATTERNS:
        if regex.search(path):
            return layer
    return None


# ── Content patterns ──────────────────────────────────────────────


_CONTENT_STATE = re.compile(
    r"\b(?:createSlice|configureStore|atom\s*\(|atomWithDefault\s*\(|"
    r"create\s*<[^>]*>\s*\(|writable\s*\(|readable\s*\(|"
    r"derived\s*\(|createContext\s*<|useContext\s*\(|"
    r"defineStore\s*\()",
)
_CONTENT_API_SERVER = re.compile(
    r"(?:@(?:app|router|blueprint)\.(?:get|post|put|patch|delete|api_route)|"
    r"\b(?:app|router|api)\.(?:get|post|put|patch|delete|all|use)\s*\(|"
    r"\bprocedure\.(?:query|mutation|subscription)\s*\(|"
    r"\bprotectedProcedure\.|\bpublicProcedure\.|"
    r"\b(?:loader|action)\s*=\s*async|"
    # Python CLI frameworks — these declare command handlers, the
    # closest Python analog to a route handler.
    r"@click\.(?:command|group)\s*\(|"
    r"@(?:typer_)?app\.command\s*\(|"
    r"\btyper\.Typer\s*\(|"
    r"\bclick\.command\s*\(\s*\)|"
    # FastAPI / APIRouter Python decorators
    r"@(?:[a-z_]+_)?(?:router|api|app)\.(?:get|post|put|patch|delete)\s*\(|"
    # Argparse subcommands
    r"\.add_subparsers\s*\(|"
    r"\bargparse\.ArgumentParser\s*\()",
)
_CONTENT_API_CLIENT = re.compile(
    r"\b(?:useQuery|useMutation|useInfiniteQuery|useSubscription|"
    r"useSWR|useSWRImmutable|axios\.(?:get|post|put|patch|delete)|"
    r"trpc\.[A-Za-z]+\.(?:useQuery|useMutation))",
)
_CONTENT_SCHEMA = re.compile(
    r"(?:^|\n)\s*model\s+\w+\s*\{|"
    r"@(?:Entity|Schema)\s*\(|"
    r"sequelize\.define\s*\(|"
    r"class\s+\w+\s*\([^)]*models\.Model[^)]*\)|"
    # Pydantic v2 BaseModel
    r"class\s+\w+\s*\(\s*BaseModel\s*\)|"
    r"class\s+\w+\s*\([^)]*BaseModel[^)]*\)|"
    # SQLAlchemy 2 declarative
    r"\bMapped\s*\[|"
    r"class\s+\w+\s*\(\s*DeclarativeBase\s*\)|"
    r"class\s+\w+\s*\(\s*Base\s*\):",
    re.MULTILINE,
)
# JSX hint — file returns/renders JSX. Used as a last-ditch UI
# signal when path patterns and other content patterns don't fire
# (e.g. an .ipynb-style module or an unusual project layout).
# We require BOTH a JSX-tag-shaped opener AND a React-ish import
# to avoid false positives on string templates.
_CONTENT_JSX_TAG = re.compile(r"</?[A-Z][A-Za-z0-9]*[\s/>]")
_CONTENT_REACT_IMPORT = re.compile(
    r"\bfrom\s+['\"]react['\"]|\bfrom\s+['\"]react/[a-z]|\bjsx\b|\bjsxs\b"
)


def classify_content(content: str) -> Layer | None:
    """Apply content patterns to a file's source.

    Returns ``None`` when no pattern fires. When multiple patterns
    fire (rare), priority is schema > api-server > state > api-
    client — the more specific layer wins.
    """
    if not content:
        return None
    if _CONTENT_SCHEMA.search(content):
        return "schema"
    if _CONTENT_API_SERVER.search(content):
        return "api-server"
    if _CONTENT_STATE.search(content):
        return "state"
    if _CONTENT_API_CLIENT.search(content):
        return "api-client"
    if (
        _CONTENT_JSX_TAG.search(content)
        and _CONTENT_REACT_IMPORT.search(content)
    ):
        return "ui"
    return None


# ── Combined ──────────────────────────────────────────────────────


def classify_file(
    path: str,
    content: str | None = None,
) -> Layer:
    """Classify ``path`` into a single :data:`Layer`.

    Path patterns are tried first (cheap, framework-aware). When
    they don't decide, falls back to content scanning. When content
    is also unhelpful (or not provided), returns ``"support"``.

    >>> classify_file("apps/web/app/users/page.tsx")
    'ui'
    >>> classify_file("apps/web/app/api/users/route.ts")
    'api-server'
    >>> classify_file("packages/prisma/schema.prisma")
    'schema'
    >>> classify_file("packages/store/userStore.ts")
    'state'
    """
    by_path = classify_path(path)
    if by_path is not None:
        return by_path
    if content is not None:
        by_content = classify_content(content)
        if by_content is not None:
            return by_content
    return "support"


def classify_files(
    repo_root: str | Path,
    paths: list[str],
    *,
    read_content: bool = True,
) -> dict[str, Layer]:
    """Batch classifier with optional content reads.

    When ``read_content`` is False we only consult path patterns
    (fast, no I/O). Useful for snapshot reports where touching the
    filesystem is undesirable.
    """
    out: dict[str, Layer] = {}
    root = Path(repo_root)
    for p in paths:
        if read_content:
            try:
                source = (root / p).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                source = ""
        else:
            source = ""
        out[p] = classify_file(p, source if source else None)
    return out
