"""Read-only tools exposed to the per-package LLM during Sprint 1
tool-augmented detection.

Four tools, all bounded so a runaway loop cannot DOS the host or
explode the token budget:

  - ``read_file_head``    head of a file, line-numbered
  - ``list_directory``    source-only listing (uses bucketizer)
  - ``grep_pattern``      regex search across source files
  - ``get_file_commits``  recent commit messages touching a file

The dispatcher (:func:`dispatch_tool`) takes a tool name + input dict
+ the repository root and returns a string suitable for use as the
``content`` of an Anthropic ``tool_result`` block.

Path safety: every tool resolves user input against ``repo_root`` and
rejects anything that escapes via ``..`` or symlinks. Errors are
returned as strings, not raised — the LLM receives them as a tool
result so it can adapt.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..analyzer.bucketizer import Bucket, classify_file


# ── Limits (kept in code, not magic constants in dispatch) ──────────────

MAX_READ_LINES = 100
MAX_READ_BYTES = 8 * 1024  # 8 KB

MAX_LIST_ENTRIES = 50

MAX_GREP_RESULTS = 30
GREP_TIMEOUT_SECONDS = 5.0
GREP_MAX_FILES_SCANNED = 5000

MAX_COMMIT_LIMIT = 20


# ── Anthropic tool schema definitions ──────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file_head",
        "description": (
            "Read the first N lines of a source file relative to the "
            "repository root. Useful for inferring what a file does "
            "from its imports, top-level declarations, or docstrings. "
            "Output is line-numbered and truncated at 100 lines or 8KB."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to repo root, e.g. 'apps/web/lib/billing.ts'.",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to read (default 50, max 100).",
                    "default": 50,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List source files in a directory (excludes tests, docs, "
            "generated, and infrastructure files). Useful for seeing "
            "what's nearby a file you're investigating. Output is up "
            "to 50 entries, one per line."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dirpath": {
                    "type": "string",
                    "description": "Directory relative to repo root. Use '.' for root.",
                },
            },
            "required": ["dirpath"],
        },
    },
    {
        "name": "grep_pattern",
        "description": (
            "Search for a regex across source files. Returns matching "
            "lines with file:line prefixes. Use to trace concepts that "
            "may not be obvious from filenames (e.g. search 'Stripe' to "
            "find billing files even outside a billing/ folder). "
            "Capped at 30 matches and 5 seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regex (re.IGNORECASE applied).",
                },
                "path_glob": {
                    "type": "string",
                    "description": "Optional path prefix to limit the search (e.g. 'packages/auth'). Defaults to whole repo.",
                    "default": "",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "find_route_handlers",
        "description": (
            "Scan source files for HTTP route handler declarations "
            "across common frameworks: Next.js (app/page.tsx, "
            "app/route.ts, pages/api), Express/Hono (app.get/post/...), "
            "FastAPI (@app.get/@router.post), tRPC "
            "(procedure.query/mutation), Remix (loader/action exports). "
            "Returns up to 50 matches as 'path:line: snippet'. Useful "
            "for grounding flow detection in real entry points."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path_glob": {
                    "type": "string",
                    "description": "Optional path prefix to limit the search (e.g. 'apps/web'). Defaults to whole repo.",
                    "default": "",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_event_handlers",
        "description": (
            "Scan source files for event-driven entry points: DOM "
            "addEventListener, EventEmitter on('event',...), webhook "
            "handlers (Stripe Webhook.constructEvent etc.), queue/"
            "worker subscriptions, chat-bot handlers (bot.on/client.on). "
            "Returns up to 50 matches as 'path:line: snippet'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path_glob": {
                    "type": "string",
                    "description": "Optional path prefix to limit the search.",
                    "default": "",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_file_commits",
        "description": (
            "Return the last N commit messages that touched a file. "
            "Commit messages often name the work in business terms "
            "even when the filename is generic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to repo root.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of commits (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": ["path"],
        },
    },
]


# ── Sprint 9 aggregator-specific tool schemas ──────────────────────────
#
# Kept separate from TOOL_SCHEMAS because they need _symbol_graph and
# _scan_result injection at dispatch time and aren't useful to the
# Sprint 1 per-package naming flow. The agentic classifier
# (faultline.llm.aggregator_agent) merges these with TOOL_SCHEMAS
# when building its message list.

AGGREGATOR_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "imports_of",
        "description": (
            "Return the files that the given file imports (outgoing "
            "edges in the symbol graph). Use this to trace what a "
            "file depends on. Empty result means a leaf file with no "
            "tracked outgoing imports — usually a type-only or asset "
            "file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to repo root.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "consumers_of",
        "description": (
            "Return the files that import FROM the given file "
            "(incoming edges). This is the strongest signal for "
            "shared-aggregator classification: a DTO file imported "
            "by 12 distinct features tells you it's shared "
            "infrastructure, not a feature on its own. Use this to "
            "investigate ambiguous packages like 'contracts', "
            "'shared-ui', 'types' before deciding."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to repo root.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "feature_summary",
        "description": (
            "Return a concise stats card for one feature in the "
            "current scan: file count, flow count, description, "
            "sample paths and flow names. Use this before "
            "investigating a feature to decide whether it's worth "
            "deeper exploration (large + active = product feature; "
            "small + cold + generic name = likely dev-internal)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Feature name as it appears in the current scan.",
                },
            },
            "required": ["name"],
        },
    },
]


# ── Path safety ────────────────────────────────────────────────────────


class ToolError(Exception):
    """Raised internally; converted to user-visible error string by dispatcher."""


def _safe_resolve(repo_root: Path, user_path: str) -> Path:
    """Resolve ``user_path`` against ``repo_root`` and ensure it stays inside.

    Rejects absolute paths, ``..`` traversal, and symlinks that point
    outside the repo. Returns the resolved absolute path.
    """
    if not user_path or user_path.strip() == "":
        raise ToolError("path is empty")

    candidate = (repo_root / user_path).resolve()
    root_resolved = repo_root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise ToolError(f"path escapes repo root: {user_path}")
    return candidate


# ── Tool implementations ───────────────────────────────────────────────


def read_file_head(repo_root: Path, path: str, lines: int = 50) -> str:
    """Return the head of a file, line-numbered, truncated.

    Errors return a string starting with ``ERROR:`` rather than raising.
    """
    try:
        target = _safe_resolve(repo_root, path)
    except ToolError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: file not found: {path}"
    if not target.is_file():
        return f"ERROR: not a regular file: {path}"

    lines_capped = max(1, min(int(lines), MAX_READ_LINES))

    try:
        raw = target.read_bytes()
    except OSError as exc:
        return f"ERROR: read failed: {exc}"

    truncated_bytes = False
    if len(raw) > MAX_READ_BYTES:
        raw = raw[:MAX_READ_BYTES]
        truncated_bytes = True

    text = raw.decode("utf-8", errors="replace")
    file_lines = text.splitlines()
    truncated_lines = len(file_lines) > lines_capped
    file_lines = file_lines[:lines_capped]

    width = len(str(len(file_lines))) if file_lines else 1
    body = "\n".join(f"{i+1:>{width}}: {ln}" for i, ln in enumerate(file_lines))

    if truncated_lines or truncated_bytes:
        body += f"\n[FILE TRUNCATED — showed {len(file_lines)} lines]"
    return body or "[EMPTY FILE]"


def list_directory(repo_root: Path, dirpath: str) -> str:
    """List source files in a directory (one per line, capped at 50)."""
    target_user = "." if not dirpath or dirpath.strip() == "" else dirpath
    try:
        target = _safe_resolve(repo_root, target_user)
    except ToolError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: directory not found: {dirpath}"
    if not target.is_dir():
        return f"ERROR: not a directory: {dirpath}"

    root_resolved = repo_root.resolve()
    entries: list[str] = []
    subdirs: list[str] = []

    try:
        children = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    except OSError as exc:
        return f"ERROR: list failed: {exc}"

    for child in children:
        try:
            rel = child.resolve().relative_to(root_resolved).as_posix()
        except (ValueError, OSError):
            continue
        if child.is_dir():
            if classify_file(rel + "/_probe").value in {"generated", "tests"}:
                continue
            subdirs.append(rel + "/")
        elif child.is_file():
            if classify_file(rel) is Bucket.SOURCE:
                entries.append(rel)
        if len(entries) + len(subdirs) >= MAX_LIST_ENTRIES:
            break

    parts: list[str] = []
    if subdirs:
        parts.append("DIRECTORIES:")
        parts.extend(subdirs)
    if entries:
        parts.append("SOURCE FILES:")
        parts.extend(entries)
    if not parts:
        return "[EMPTY — no source files in this directory]"
    return "\n".join(parts)


def grep_pattern(repo_root: Path, pattern: str, path_glob: str = "") -> str:
    """Regex-search source files, returning ``path:lineno: line`` matches."""
    if not pattern:
        return "ERROR: empty pattern"
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"ERROR: invalid regex: {exc}"

    if path_glob:
        try:
            scan_root = _safe_resolve(repo_root, path_glob)
        except ToolError as exc:
            return f"ERROR: {exc}"
        if not scan_root.exists():
            return f"ERROR: path not found: {path_glob}"
    else:
        scan_root = repo_root.resolve()

    root_resolved = repo_root.resolve()
    matches: list[str] = []
    files_scanned = 0
    deadline = time.monotonic() + GREP_TIMEOUT_SECONDS

    walker = [scan_root] if scan_root.is_file() else scan_root.rglob("*")

    for entry in walker:
        if time.monotonic() > deadline:
            matches.append("[TIMEOUT — stopped scanning]")
            break
        if len(matches) >= MAX_GREP_RESULTS:
            break
        if files_scanned >= GREP_MAX_FILES_SCANNED:
            matches.append("[FILE LIMIT — stopped scanning]")
            break
        if not entry.is_file():
            continue
        try:
            rel = entry.resolve().relative_to(root_resolved).as_posix()
        except (ValueError, OSError):
            continue
        if classify_file(rel) is not Bucket.SOURCE:
            continue
        files_scanned += 1
        try:
            with entry.open("r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if compiled.search(line):
                        matches.append(f"{rel}:{lineno}: {line.rstrip()[:200]}")
                        if len(matches) >= MAX_GREP_RESULTS:
                            break
        except OSError:
            continue

    if not matches:
        return f"[NO MATCHES — scanned {files_scanned} files]"
    return "\n".join(matches)


_ROUTE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Next.js / Remix file-based routing — matched on path conventions
    # rather than file content; we find these via filename match below.
    # Express + Hono
    ("express-or-hono-method",
     r"\b(app|router|api)\.(get|post|put|patch|delete|all|use)\s*\("),
    # FastAPI / Flask decorators
    ("fastapi-flask-route",
     r"@(?:app|router|blueprint)\.(?:get|post|put|patch|delete|api_route|route)\s*\("),
    # tRPC
    ("trpc-procedure",
     r"\b(?:procedure|protectedProcedure|publicProcedure|t\.procedure)\s*[.\n]\s*(?:query|mutation|subscription)\s*\("),
    # Remix loaders/actions (exported)
    ("remix-loader-action",
     r"^\s*export\s+(?:const|async\s+function)\s+(?:loader|action)\b"),
    # Django urls.py path()
    ("django-path",
     r"\bpath\s*\(\s*['\"][^'\"]*['\"]\s*,\s*[A-Za-z_]"),
    # NestJS decorators
    ("nestjs-route",
     r"@(?:Get|Post|Put|Patch|Delete|All)\s*\("),
)


_EVENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("addEventListener", r"\.addEventListener\s*\(\s*['\"]"),
    ("emitter-on",       r"\.on\s*\(\s*['\"][a-zA-Z._-]+['\"]"),
    ("queue-worker",     r"\b(?:queue|worker|consumer|subscriber)\.(?:process|on|subscribe|consume)\s*\("),
    ("stripe-webhook",   r"\bstripe\.webhooks\.constructEvent\s*\("),
    ("bot-on",           r"\b(?:bot|client|app)\.on\s*\(\s*['\"][a-zA-Z._-]+['\"]"),
    ("react-hook-effect",
     r"\buseEffect\s*\(\s*\(\s*\)\s*=>\s*\{\s*[A-Za-z_].*\.(?:addEventListener|subscribe)"),
)


_NEXTJS_FILENAME_RE = re.compile(
    r"(?:^|/)(?:app|src/app)/.*?(?:page|route|layout|error|loading)\.(?:tsx?|jsx?)$"
)
_NEXTJS_PAGES_API_RE = re.compile(r"(?:^|/)pages/api/.*\.(?:tsx?|jsx?)$")


def _scan_patterns(
    repo_root: Path,
    path_glob: str,
    patterns: tuple[tuple[str, str], ...],
    *,
    extra_filename_matchers: tuple[re.Pattern, ...] = (),
) -> str:
    """Shared scanner for route + event tools.

    Walks the source tree (filtered by bucketizer), applies regex
    patterns, returns up to 50 matches as ``path:line: snippet``.
    Reused by ``find_route_handlers`` / ``find_event_handlers``.
    """
    if path_glob:
        try:
            scan_root = _safe_resolve(repo_root, path_glob)
        except ToolError as exc:
            return f"ERROR: {exc}"
        if not scan_root.exists():
            return f"ERROR: path not found: {path_glob}"
    else:
        scan_root = repo_root.resolve()

    root_resolved = repo_root.resolve()
    compiled = [(name, re.compile(p, re.MULTILINE)) for name, p in patterns]
    matches: list[str] = []
    files_scanned = 0
    deadline = time.monotonic() + GREP_TIMEOUT_SECONDS

    walker = [scan_root] if scan_root.is_file() else scan_root.rglob("*")
    for entry in walker:
        if time.monotonic() > deadline:
            matches.append("[TIMEOUT — stopped scanning]")
            break
        if len(matches) >= 50:
            break
        if files_scanned >= GREP_MAX_FILES_SCANNED:
            matches.append("[FILE LIMIT — stopped scanning]")
            break
        if not entry.is_file():
            continue
        try:
            rel = entry.resolve().relative_to(root_resolved).as_posix()
        except (ValueError, OSError):
            continue
        if classify_file(rel) is not Bucket.SOURCE:
            continue
        files_scanned += 1

        # Filename-based detection (Next.js file routing).
        for fm_re in extra_filename_matchers:
            if fm_re.search(rel):
                matches.append(f"{rel}:1: [route-file]")
                if len(matches) >= 50:
                    return "\n".join(matches)
                break

        try:
            text = entry.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, regex in compiled:
            for m in regex.finditer(text):
                lineno = text.count("\n", 0, m.start()) + 1
                line = text[
                    text.rfind("\n", 0, m.start()) + 1 : text.find("\n", m.end())
                    if text.find("\n", m.end()) != -1 else len(text)
                ]
                matches.append(f"{rel}:{lineno}: [{label}] {line.strip()[:160]}")
                if len(matches) >= 50:
                    return "\n".join(matches)

    if not matches:
        return f"[NO MATCHES — scanned {files_scanned} files]"
    return "\n".join(matches)


def find_route_handlers(repo_root: Path, path_glob: str = "") -> str:
    return _scan_patterns(
        repo_root, path_glob, _ROUTE_PATTERNS,
        extra_filename_matchers=(_NEXTJS_FILENAME_RE, _NEXTJS_PAGES_API_RE),
    )


def find_event_handlers(repo_root: Path, path_glob: str = "") -> str:
    return _scan_patterns(repo_root, path_glob, _EVENT_PATTERNS)


def get_file_commits(repo_root: Path, path: str, limit: int = 5) -> str:
    """Return the last N commit subjects that touched ``path``."""
    try:
        target = _safe_resolve(repo_root, path)
    except ToolError as exc:
        return f"ERROR: {exc}"

    if not target.exists():
        return f"ERROR: file not found: {path}"

    limit_capped = max(1, min(int(limit), MAX_COMMIT_LIMIT))

    try:
        from git import InvalidGitRepositoryError, Repo
    except ImportError:
        return "ERROR: gitpython not available"

    try:
        repo = Repo(repo_root, search_parent_directories=False)
    except InvalidGitRepositoryError:
        return "ERROR: not a git repository"

    rel = target.resolve().relative_to(repo_root.resolve()).as_posix()
    try:
        commits = list(repo.iter_commits(paths=rel, max_count=limit_capped))
    except Exception as exc:  # GitCommandError or similar
        return f"ERROR: git log failed: {exc}"

    if not commits:
        return "[NO COMMITS — file untracked or no history]"

    out: list[str] = []
    for c in commits:
        subject = c.message.splitlines()[0] if c.message else ""
        out.append(f"{c.hexsha[:8]} {subject}")
    return "\n".join(out)


# ── Dispatcher ─────────────────────────────────────────────────────────


def dispatch_tool(
    name: str,
    tool_input: dict[str, Any],
    repo_root: Path,
) -> str:
    """Run a tool by name and return its string result.

    Never raises — every error path returns a string starting with
    ``ERROR:`` so the LLM can be told what went wrong via tool_result.
    """
    if not isinstance(tool_input, dict):
        return f"ERROR: tool input must be a dict, got {type(tool_input).__name__}"

    if name == "read_file_head":
        path = tool_input.get("path", "")
        lines = tool_input.get("lines", 50)
        return read_file_head(repo_root, path, lines)

    if name == "list_directory":
        dirpath = tool_input.get("dirpath", ".")
        return list_directory(repo_root, dirpath)

    if name == "grep_pattern":
        pattern = tool_input.get("pattern", "")
        path_glob = tool_input.get("path_glob", "")
        return grep_pattern(repo_root, pattern, path_glob)

    if name == "get_file_commits":
        path = tool_input.get("path", "")
        limit = tool_input.get("limit", 5)
        return get_file_commits(repo_root, path, limit)

    if name == "find_route_handlers":
        return find_route_handlers(repo_root, tool_input.get("path_glob", ""))

    if name == "find_event_handlers":
        return find_event_handlers(repo_root, tool_input.get("path_glob", ""))

    # Sprint 9: import-graph tools for agentic aggregator detection.
    # The agent uses these to investigate which OTHER files / features
    # consume a given file before deciding if it's shared infrastructure.
    if name == "imports_of":
        path = tool_input.get("path", "")
        graph = tool_input.get("_symbol_graph")  # injected by caller
        return imports_of(graph, path)

    if name == "consumers_of":
        path = tool_input.get("path", "")
        graph = tool_input.get("_symbol_graph")  # injected by caller
        return consumers_of(graph, path)

    if name == "feature_summary":
        feature_name = tool_input.get("name", "")
        result = tool_input.get("_scan_result")  # injected by caller
        return feature_summary(result, feature_name)

    return f"ERROR: unknown tool: {name}"


# ── Sprint 9 import-graph tools ────────────────────────────────────────


def imports_of(symbol_graph: Any, path: str) -> str:
    """Return files that ``path`` imports (forward edges from Sprint 7
    SymbolGraph).

    Output is line-per-edge: ``<target_file>``. Empty result is a
    real signal — the file imports nothing tracked, so it's likely
    a leaf (type definition, constant, raw asset).
    """
    if symbol_graph is None:
        return "ERROR: symbol graph not available; pipeline did not build one"
    if not path:
        return "ERROR: path is required"
    try:
        edges = symbol_graph.imports_from(path)
    except (AttributeError, KeyError):
        return f"ERROR: file not in symbol graph: {path}"
    if not edges:
        return f"(no outgoing imports from {path} — leaf file)"
    out = [f"Imports from {path} ({len(edges)} edges):"]
    seen = set()
    for edge in edges:
        target = getattr(edge, "target_file", None)
        if target and target not in seen:
            seen.add(target)
            out.append(f"  → {target}")
    return "\n".join(out)


def consumers_of(symbol_graph: Any, path: str) -> str:
    """Return files that import FROM ``path`` (reverse edges).

    This is the strongest signal for aggregator classification:
    a DTO file imported by 12 different features' code is exactly
    what we want to redistribute.
    """
    if symbol_graph is None:
        return "ERROR: symbol graph not available; pipeline did not build one"
    if not path:
        return "ERROR: path is required"
    try:
        edges = symbol_graph.callers_of(path)
    except (AttributeError, KeyError):
        return f"ERROR: file not in symbol graph: {path}"
    if not edges:
        return f"(no consumers of {path} — file is unused or only consumed by untracked code)"
    out = [f"Consumers of {path} ({len(edges)} edges):"]
    seen = set()
    for edge in edges:
        importer = getattr(edge, "target_file", None)
        if importer and importer not in seen:
            seen.add(importer)
            out.append(f"  ← {importer}")
    return "\n".join(out)


def feature_summary(scan_result: Any, name: str) -> str:
    """Return a concise summary of one feature in the in-flight scan.

    The agent uses this to decide whether a feature is large/active
    (likely product) vs small/cold (likely dev-internal) before
    investigating with the file-level tools.
    """
    if scan_result is None:
        return "ERROR: scan result not available"
    if not name:
        return "ERROR: name is required"
    paths = scan_result.features.get(name)
    if paths is None:
        return f"ERROR: no feature named {name}"
    flows = scan_result.flows.get(name, []) if hasattr(scan_result, "flows") else []
    desc = (
        scan_result.descriptions.get(name, "")
        if hasattr(scan_result, "descriptions") else ""
    )
    out = [
        f"Feature: {name}",
        f"  Files: {len(paths)}",
        f"  Flows: {len(flows)}",
    ]
    if desc:
        out.append(f"  Description: {desc[:120]}")
    if paths:
        out.append("  Sample paths (first 8):")
        for p in paths[:8]:
            out.append(f"    {p}")
    if flows:
        out.append("  Flow names (first 8):")
        for f in flows[:8]:
            out.append(f"    {f}")
    return "\n".join(out)
