"""Git-diff utility for incremental scans.

Standalone helper. Does NOT touch the existing pipeline modules —
``pipeline.run()``, ``deep_scan``, ``deep_scan_workspace`` are
unchanged. The incremental orchestrator (Stage 3) is the only
caller.

Computes the set of source files that changed between a prior
scan's commit (``last_sha``) and the working tree. Used to decide
which features need re-analysis.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class GitDiff:
    """Changed/added/removed files between two refs.

    Sets are repo-relative POSIX paths matching what the pipeline
    feeds Sonnet, so we can intersect directly with feature.paths
    without normalization.
    """

    base_sha: str | None
    head_sha: str | None
    added: set[str] = field(default_factory=set)
    modified: set[str] = field(default_factory=set)
    deleted: set[str] = field(default_factory=set)
    # When True the caller should fall back to a full scan: either no
    # base ref was supplied, or git refused to diff for some reason.
    fallback_full_scan: bool = False

    @property
    def changed_paths(self) -> set[str]:
        """Files the incremental pipeline must consider — added or
        modified. Deletions handled separately (we drop them from
        carry-forward, but they don't trigger re-analysis themselves)."""
        return self.added | self.modified

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.deleted)


def compute_git_diff(
    repo_root: str | Path,
    base_sha: str | None,
    *,
    head_sha: str = "HEAD",
) -> GitDiff:
    """Run ``git diff --name-status base..head`` and return a
    :class:`GitDiff`.

    Returns a diff with ``fallback_full_scan=True`` when:
      - ``base_sha`` is missing (first incremental run on a repo)
      - ``base_sha`` doesn't exist in the repo (force-pushed, rebased,
        clone too shallow)
      - the repo isn't actually a git repo
      - git itself errors out (binary missing, corrupt repo)

    Uncommitted working-tree changes are included via a separate
    ``git diff --name-status HEAD`` call so a developer running
    ``faultline analyze --incremental`` mid-edit still re-analyses
    files they're actively touching.
    """
    repo_root = Path(repo_root).resolve()
    diff = GitDiff(base_sha=base_sha, head_sha=None)

    if not (repo_root / ".git").exists() and not _is_git_dir(repo_root):
        logger.info("git_diff: %s is not a git repo — fallback to full", repo_root)
        diff.fallback_full_scan = True
        return diff

    if not base_sha:
        logger.info("git_diff: no base_sha — fallback to full scan")
        diff.fallback_full_scan = True
        return diff

    if not _ref_exists(repo_root, base_sha):
        logger.info(
            "git_diff: base ref %s not present locally (shallow clone?) "
            "— fallback to full",
            base_sha[:12],
        )
        diff.fallback_full_scan = True
        return diff

    head_resolved = _resolve_ref(repo_root, head_sha)
    diff.head_sha = head_resolved

    # Committed changes between base and HEAD.
    committed = _diff_name_status(repo_root, f"{base_sha}..{head_sha}")
    if committed is None:
        diff.fallback_full_scan = True
        return diff
    _merge_diff_lines(diff, committed)

    # Uncommitted changes (staged + unstaged) on top of HEAD.
    uncommitted = _diff_name_status(repo_root, "HEAD")
    if uncommitted:
        _merge_diff_lines(diff, uncommitted)

    logger.info(
        "git_diff: %s..%s — +%d ~%d -%d",
        (base_sha or "?")[:12],
        (head_resolved or head_sha)[:12],
        len(diff.added), len(diff.modified), len(diff.deleted),
    )
    return diff


# ── Internals ───────────────────────────────────────────────────


def _is_git_dir(path: Path) -> bool:
    """``git rev-parse --is-inside-work-tree`` — handles worktrees
    where ``.git`` is a file, and any other non-standard layout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path, capture_output=True, text=True, timeout=10,
        )
        return out.returncode == 0 and out.stdout.strip() == "true"
    except (OSError, subprocess.TimeoutExpired):
        return False


def _ref_exists(repo_root: Path, ref: str) -> bool:
    try:
        out = subprocess.run(
            ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        return out.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _resolve_ref(repo_root: Path, ref: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", ref],
            cwd=repo_root, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _diff_name_status(repo_root: Path, range_arg: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "diff", "--name-status", "-z", range_arg],
            cwd=repo_root, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("git_diff: %s failed (%s)", range_arg, exc)
        return None
    if out.returncode != 0:
        logger.warning(
            "git_diff: %s exited %d: %s",
            range_arg, out.returncode, out.stderr.strip()[:200],
        )
        return None
    # ``-z`` emits NUL-separated records: STATUS\tPATH\0[STATUS\tPATH\0...]
    # For renames (R/C) the format is STATUS\0PATH_OLD\0PATH_NEW\0.
    return out.stdout.split("\x00")


def _merge_diff_lines(diff: GitDiff, parts: list[str]) -> None:
    """Walk the ``-z`` stream from ``git diff --name-status``.

    Format: ``STATUS\\0PATH\\0STATUS\\0PATH\\0...``. For renames /
    copies (``R100``, ``C75``) it's ``STATUS\\0OLD_PATH\\0NEW_PATH\\0``.
    """
    i = 0
    while i < len(parts):
        status = parts[i]
        if not status:
            i += 1
            continue
        code = status[0]
        if code in ("R", "C"):
            old = parts[i + 1] if i + 1 < len(parts) else ""
            new = parts[i + 2] if i + 2 < len(parts) else ""
            i += 3
            if old:
                diff.deleted.add(old)
            if new:
                diff.modified.add(new)
            continue
        # Single-path entries: STATUS\0PATH.
        path = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2
        if not path:
            continue
        if code == "A":
            diff.added.add(path)
        elif code in ("M", "T"):
            diff.modified.add(path)
        elif code == "D":
            diff.deleted.add(path)
        # Unknown statuses (U for unmerged, X for unknown) are skipped.
