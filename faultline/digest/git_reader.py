"""
Standalone git reader for daily digest.

Zero imports from faultline core — fully self-contained.
Reads commits merged to main/master for a given date.
"""

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from git import Repo, InvalidGitRepositoryError


@dataclass
class DigestCommit:
    sha: str
    message: str
    author: str
    date: str  # ISO format
    files_changed: list[str] = field(default_factory=list)
    is_bug_fix: bool = False
    pr_number: int | None = None


_BUG_FIX_RE = re.compile(
    r"\bfix\b|\bbug\b|\bhotfix\b|\bpatch\b|\brevert\b|\bregression\b"
    r"|\bcrash\b|\berror\b|\bbroken\b|\bissue\b|\bdefect\b",
    re.IGNORECASE,
)
_FALSE_POSITIVE_RE = re.compile(
    r"\bfix\s+(?:typo|lint|format\w*|style|import|indent|spacing|test|docs?|merge|ci|build)\b",
    re.IGNORECASE,
)
_PR_MERGE_RE = re.compile(r"Merge pull request #(\d+)", re.IGNORECASE)
_PR_SQUASH_RE = re.compile(r"\(#(\d+)\)\s*(?:\n|$)")


def _is_bug_fix(message: str) -> bool:
    if not _BUG_FIX_RE.search(message):
        return False
    if _FALSE_POSITIVE_RE.search(message):
        return False
    return True


def _extract_pr(message: str) -> int | None:
    m = _PR_MERGE_RE.search(message)
    if m:
        return int(m.group(1))
    m = _PR_SQUASH_RE.search(message)
    if m:
        return int(m.group(1))
    return None


def _get_remote_url(repo: Repo) -> str:
    try:
        url = repo.remotes.origin.url
        if url.startswith("git@"):
            url = re.sub(r"^git@([^:]+):", r"https://\1/", url)
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except Exception:
        return ""


def get_daily_commits(
    repo_path: str,
    date: str,
    branch: str | None = None,
) -> dict:
    """Get all commits merged to main/master on a given date.

    Args:
        repo_path: Path to the git repository.
        date: Date string in YYYY-MM-DD format.
        branch: Branch name. Auto-detects main/master if None.

    Returns:
        Dict with repo info and commits list, ready for JSON serialization.
    """
    repo = Repo(repo_path)
    remote_url = _get_remote_url(repo)

    # Resolve branch
    if not branch:
        for candidate in ("main", "master"):
            if candidate in [ref.name for ref in repo.heads]:
                branch = candidate
                break
        if not branch:
            branch = repo.active_branch.name

    # Parse date range
    target = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    since = target.replace(hour=0, minute=0, second=0)
    until = target.replace(hour=23, minute=59, second=59)

    # Get commits on the branch for that date
    commits: list[DigestCommit] = []
    for git_commit in repo.iter_commits(
        branch,
        since=since.isoformat(),
        until=until.isoformat(),
        first_parent=True,  # only merged commits, skip merge bases
    ):
        msg = git_commit.message.strip()
        first_line = msg.split("\n")[0]

        files = []
        try:
            if git_commit.parents:
                diff = git_commit.parents[0].diff(git_commit)
                files = [d.a_path or d.b_path for d in diff if d.a_path or d.b_path]
            else:
                files = list(git_commit.stats.files.keys())
        except Exception:
            pass

        commits.append(DigestCommit(
            sha=git_commit.hexsha[:8],
            message=first_line,
            author=git_commit.author.name or git_commit.author.email,
            date=git_commit.committed_datetime.isoformat(),
            files_changed=files,
            is_bug_fix=_is_bug_fix(first_line),
            pr_number=_extract_pr(msg),
        ))

    # Group by author
    authors: dict[str, int] = {}
    for c in commits:
        authors[c.author] = authors.get(c.author, 0) + 1

    # Group files by top-level directory
    dir_changes: dict[str, int] = {}
    for c in commits:
        for f in c.files_changed:
            parts = Path(f).parts
            top_dir = parts[0] if len(parts) > 1 else "(root)"
            dir_changes[top_dir] = dir_changes.get(top_dir, 0) + 1

    repo_name = Path(repo_path).resolve().name

    return {
        "repo_name": repo_name,
        "repo_path": str(Path(repo_path).resolve()),
        "remote_url": remote_url,
        "branch": branch,
        "date": date,
        "total_commits": len(commits),
        "bug_fixes": sum(1 for c in commits if c.is_bug_fix),
        "authors": dict(sorted(authors.items(), key=lambda x: -x[1])),
        "dir_changes": dict(sorted(dir_changes.items(), key=lambda x: -x[1])),
        "commits": [asdict(c) for c in commits],
    }
