"""Line-level git blame index with persistent SQLite cache.

For each (file, line) pair, records which commit last touched that line.
Powers symbol-scoped scoring: instead of "this file has 50 commits", we
can ask "which commits touched lines L5-L25 of this file" — i.e. only
the commits that actually changed the symbol we care about.

The cache lives in ``<repo>/.faultline/cache/blame.sqlite`` and is
incremental: re-indexing a file only re-runs ``git blame`` when the
file's HEAD commit has changed since last index.

Graceful degradation: every public method returns success/failure
(or ``None``) instead of raising. Callers tier-down to file-level
scoring when blame data is unavailable.
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_BLAME_TIMEOUT_S = 60.0


@dataclass(frozen=True, slots=True)
class BlameLine:
    line: int
    commit_sha: str


@dataclass(frozen=True, slots=True)
class IndexStats:
    indexed: int = 0      # files newly indexed (or refreshed)
    cached: int = 0       # files already up-to-date
    failed: int = 0       # files that failed (timeout / binary / not in git)
    failures: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            object.__setattr__(self, "failures", [])


class BlameIndex:
    """Persistent line-level blame index.

    Usage:
        idx = BlameIndex(repo_root)
        idx.index_file("src/utils.ts")
        commits = idx.commits_touching_lines("src/utils.ts", 5, 25)
        # → set of commit SHAs that touch any of lines 5..25 in utils.ts
    """

    def __init__(
        self,
        repo_root: Path,
        cache_dir: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.cache_dir = (
            cache_dir
            if cache_dir is not None
            else (self.repo_root / ".faultline" / "cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "blame.sqlite"
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BlameIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Schema ────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS blame_lines (
              file_path TEXT NOT NULL,
              line      INTEGER NOT NULL,
              commit_sha TEXT NOT NULL,
              PRIMARY KEY (file_path, line)
            );
            CREATE INDEX IF NOT EXISTS idx_blame_file ON blame_lines (file_path);
            CREATE INDEX IF NOT EXISTS idx_blame_sha ON blame_lines (commit_sha);

            CREATE TABLE IF NOT EXISTS file_state (
              file_path  TEXT PRIMARY KEY,
              head_sha   TEXT NOT NULL,
              indexed_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # ── Public API ────────────────────────────────────────────────────

    def index_file(self, rel_path: str) -> bool:
        """Index a single file. Returns True on success (or already cached).

        Never raises. On failure, logs a debug-level message and
        returns False. The caller should fall back to file-level
        scoring for this file.
        """
        head_sha = self._git_head_for_file(rel_path)
        if head_sha is None:
            return False  # file not tracked / git error
        if self._is_up_to_date(rel_path, head_sha):
            return True
        try:
            blame_rows = self._run_blame(rel_path)
        except Exception as exc:  # noqa: BLE001 — broad on purpose
            logger.debug("blame_index: %s — blame failed (%s)", rel_path, exc)
            return False
        self._store_blame(rel_path, head_sha, blame_rows)
        return True

    def index_files(self, rel_paths: list[str]) -> IndexStats:
        """Index a batch of files. Returns aggregate stats."""
        indexed = 0
        cached = 0
        failed = 0
        failures: list[str] = []
        for path in rel_paths:
            head_sha = self._git_head_for_file(path)
            if head_sha is None:
                failed += 1
                failures.append(f"{path}: not in git")
                continue
            if self._is_up_to_date(path, head_sha):
                cached += 1
                continue
            try:
                blame_rows = self._run_blame(path)
            except subprocess.TimeoutExpired:
                failed += 1
                failures.append(f"{path}: timeout")
                continue
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append(f"{path}: {exc}")
                continue
            self._store_blame(path, head_sha, blame_rows)
            indexed += 1
        return IndexStats(
            indexed=indexed, cached=cached, failed=failed, failures=failures,
        )

    def commits_touching_lines(
        self, rel_path: str, start_line: int, end_line: int,
    ) -> set[str] | None:
        """Returns commit SHAs that last-touched any line in [start, end].

        Returns ``None`` when the file is not yet indexed (caller should
        decide whether to index on-demand or fall back to file-level).
        Returns an empty set when the range has no blame data.
        """
        if not self._has_blame(rel_path):
            return None
        rows = self._conn.execute(
            "SELECT DISTINCT commit_sha FROM blame_lines "
            "WHERE file_path = ? AND line BETWEEN ? AND ?",
            (rel_path, start_line, end_line),
        ).fetchall()
        return {r[0] for r in rows}

    def commits_touching_file(self, rel_path: str) -> set[str] | None:
        """All commit SHAs that touch any line of the file."""
        if not self._has_blame(rel_path):
            return None
        rows = self._conn.execute(
            "SELECT DISTINCT commit_sha FROM blame_lines WHERE file_path = ?",
            (rel_path,),
        ).fetchall()
        return {r[0] for r in rows}

    def is_indexed(self, rel_path: str) -> bool:
        return self._has_blame(rel_path)

    # ── Internals ─────────────────────────────────────────────────────

    def _has_blame(self, rel_path: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM file_state WHERE file_path = ? LIMIT 1",
            (rel_path,),
        ).fetchone()
        return row is not None

    def _is_up_to_date(self, rel_path: str, head_sha: str) -> bool:
        row = self._conn.execute(
            "SELECT head_sha FROM file_state WHERE file_path = ?",
            (rel_path,),
        ).fetchone()
        return row is not None and row[0] == head_sha

    def _git_head_for_file(self, rel_path: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", "-C", str(self.repo_root), "log", "-n", "1",
                 "--format=%H", "--", rel_path],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        sha = r.stdout.strip()
        return sha or None

    def _run_blame(self, rel_path: str) -> list[BlameLine]:
        """Run ``git blame --porcelain`` and parse line→sha mapping."""
        r = subprocess.run(
            ["git", "-C", str(self.repo_root), "blame", "--porcelain",
             "--", rel_path],
            capture_output=True, text=True, timeout=_BLAME_TIMEOUT_S,
        )
        if r.returncode != 0:
            raise RuntimeError(f"blame returned {r.returncode}: {r.stderr.strip()}")
        return _parse_porcelain(r.stdout)

    def _store_blame(
        self, rel_path: str, head_sha: str, rows: list[BlameLine],
    ) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM blame_lines WHERE file_path = ?", (rel_path,),
            )
            self._conn.executemany(
                "INSERT INTO blame_lines (file_path, line, commit_sha) "
                "VALUES (?, ?, ?)",
                ((rel_path, r.line, r.commit_sha) for r in rows),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO file_state "
                "(file_path, head_sha, indexed_at) VALUES (?, ?, ?)",
                (
                    rel_path,
                    head_sha,
                    datetime.now(tz=timezone.utc).isoformat(),
                ),
            )


# ── Porcelain parser ────────────────────────────────────────────────


def _parse_porcelain(text: str) -> list[BlameLine]:
    """Parse ``git blame --porcelain`` output into line→sha rows.

    Porcelain format:
      <sha> <orig_line> <final_line> [<group_size>]
      author Name
      author-mail <...>
      ...
      filename <path>
      \t<content>

    The first line of each block carries the SHA and final-line
    number. Repeated lines from the same commit only carry the SHA
    + line numbers (no header repeated). We track the "current SHA"
    across blocks to handle this.
    """
    out: list[BlameLine] = []
    current_sha: str | None = None
    expecting_content = False
    final_line: int | None = None

    for raw in text.splitlines():
        if raw.startswith("\t"):
            # content line — emit using last (sha, final_line)
            if current_sha is not None and final_line is not None:
                out.append(BlameLine(line=final_line, commit_sha=current_sha))
            expecting_content = False
            current_sha = None
            final_line = None
            continue

        parts = raw.split(" ", 3)
        if (
            len(parts) >= 3
            and len(parts[0]) == 40
            and parts[0].replace("-", "").isalnum()
        ):
            # Header line: sha orig_line final_line [count]
            try:
                current_sha = parts[0]
                final_line = int(parts[2])
                expecting_content = True
            except ValueError:
                current_sha = None
                final_line = None
                expecting_content = False
        # Other header lines (author, etc.) just consumed; we don't
        # need their data for this index.
    return out
