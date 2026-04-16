"""Content and symbol hashing helpers.

Per-file content hash is used to skip re-parsing files that didn't
change (even if git says they did — e.g. reverted commits).

Per-symbol body hash is used for the most aggressive cache: if a
function's body hasn't changed, its flow attribution stays valid
without any LLM call.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from faultline.models.types import SymbolRange


def hash_file(abs_path: Path) -> str | None:
    """SHA-256 of file content. Returns None for missing/unreadable files."""
    try:
        data = abs_path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def hash_files(files: list[str], repo_path: str) -> dict[str, str]:
    """Compute content hashes for a list of relative file paths."""
    root = Path(repo_path)
    result: dict[str, str] = {}
    for rel in files:
        h = hash_file(root / rel)
        if h is not None:
            result[rel] = h
    return result


def changed_files(
    old_hashes: dict[str, str],
    new_hashes: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    """Returns (modified, added, removed) relative to old hashes."""
    old_keys = set(old_hashes.keys())
    new_keys = set(new_hashes.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys

    modified = {
        path for path in old_keys & new_keys
        if old_hashes[path] != new_hashes[path]
    }

    return modified, added, removed


def hash_symbol_body(source: str, symbol: SymbolRange) -> str:
    """SHA-256 of a symbol's source lines (1-indexed, inclusive)."""
    lines = source.splitlines()
    start = max(0, symbol.start_line - 1)
    end = min(len(lines), symbol.end_line)
    body = "\n".join(lines[start:end]).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def compute_symbol_hashes(
    file_path: str,
    source: str,
    symbols: list[SymbolRange],
) -> dict[str, str]:
    """{symbol_name: hash_of_body} for one file."""
    return {sym.name: hash_symbol_body(source, sym) for sym in symbols}
