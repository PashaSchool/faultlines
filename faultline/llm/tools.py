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

    return f"ERROR: unknown tool: {name}"
