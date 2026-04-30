import re
from datetime import datetime, timezone
from pathlib import Path

from git import Repo, InvalidGitRepositoryError
from rich.progress import Progress, SpinnerColumn, TextColumn

from faultline.models.types import Commit, FileBlame

# Regex to extract PR numbers from commit messages.
# Handles two common GitHub merge strategies:
#   "Merge pull request #123 from branch"  (merge commit)
#   "fix: something (#123)"                (squash merge)
_PR_MERGE_RE  = re.compile(r"Merge pull request #(\d+)", re.IGNORECASE)
_PR_SQUASH_RE = re.compile(r"\(#(\d+)\)\s*(?:\n|$)")

# Regex patterns that identify bug fix commits
BUG_FIX_PATTERNS = [
    r"\bfix\b", r"\bbug\b", r"\bhotfix\b", r"\bpatch\b",
    r"\brevert\b", r"\bregression\b", r"\bcrash\b", r"\berror\b",
    r"\bbroken\b", r"\bissue\b", r"\bdefect\b",
    r"\bresolve\b", r"\btimeout\b", r"\bnull\s*(?:pointer|check|ref)\b",
    r"\bundefined\b", r"\bNaN\b", r"\brace\s*condition\b",
    r"\bdeadlock\b", r"\bmemory\s*leak\b", r"\boverflow\b",
]

BUG_FIX_REGEX = re.compile("|".join(BUG_FIX_PATTERNS), re.IGNORECASE)

# Patterns that indicate a "fix" commit is NOT a real bug fix
_FALSE_POSITIVE_PATTERNS = [
    r"\bfix\s+(?:typo|lint|format\w*|style|import|indent|spacing|whitespace)\b",
    r"\bfix\s+(?:test|spec|mock|snapshot)\b",
    r"\bfix\s+(?:docs?|readme|comment|changelog)\b",
    r"\bfix\s+(?:merge|conflict|rebase)\b",
    r"\bfix\s+(?:ci|pipeline|build|deploy)\b",
]
_FALSE_POSITIVE_REGEX = re.compile("|".join(_FALSE_POSITIVE_PATTERNS), re.IGNORECASE)

# Approximate seconds per commit based on profiling (git stats I/O)
_SECONDS_PER_COMMIT = 0.008
# Approximate seconds per LLM call for flow detection (one call per feature)
_SECONDS_PER_FLOW_FEATURE = 4
# Rough estimate of features for flow duration estimate (before actual detection)
_ESTIMATED_FLOW_FEATURES = 20
DEFAULT_MAX_COMMITS = 5_000


def is_bug_fix(message: str) -> bool:
    if not BUG_FIX_REGEX.search(message):
        return False
    # Exclude false positives: "fix typo", "fix lint", "fix test", etc.
    if _FALSE_POSITIVE_REGEX.search(message):
        return False
    return True


def extract_pr_number(message: str) -> int | None:
    """Extracts a GitHub PR number from a commit message, or None if not found."""
    m = _PR_MERGE_RE.search(message)
    if m:
        return int(m.group(1))
    m = _PR_SQUASH_RE.search(message)
    if m:
        return int(m.group(1))
    return None


def get_remote_url(repo: Repo) -> str:
    """
    Returns the GitHub base URL for the repo (without .git suffix), or empty string.
    Normalizes SSH remotes to HTTPS: git@github.com:org/repo.git → https://github.com/org/repo
    """
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


def load_repo(path: str) -> Repo:
    try:
        repo = Repo(path, search_parent_directories=True)
    except InvalidGitRepositoryError:
        raise ValueError(f"'{path}' is not a git repository")

    if repo.head.is_detached:
        return repo

    try:
        repo.head.commit
    except ValueError:
        raise ValueError(
            f"Repository at '{path}' has no commits yet. "
            "Make at least one commit before running analysis."
        )

    return repo


def estimate_commits(repo: Repo, days: int, max_commits: int = DEFAULT_MAX_COMMITS) -> int:
    """
    Quickly estimates the number of commits in the date range.
    Uses git rev-list --count which is near-instant regardless of repo size.
    """
    since = f"--since={days} days ago"
    try:
        count = repo.git.rev_list("--count", since, "HEAD")
        return min(int(count), max_commits)
    except Exception:
        return 0


def estimate_duration(commit_count: int, use_llm: bool = False, use_flows: bool = False) -> str:
    """Returns a human-readable time estimate for the analysis."""
    seconds = commit_count * _SECONDS_PER_COMMIT
    if use_llm:
        seconds += 5  # avg LLM API round-trip for feature detection
    if use_flows:
        seconds += _ESTIMATED_FLOW_FEATURES * _SECONDS_PER_FLOW_FEATURE  # ~20 features * 4 sec

    if seconds < 10:
        return "< 10 sec"
    elif seconds < 60:
        return f"~ {int(seconds)} sec"
    else:
        minutes = seconds / 60
        return f"~ {minutes:.1f} min"


def get_commits(repo: Repo, days: int = 365, max_commits: int = DEFAULT_MAX_COMMITS) -> list[Commit]:
    """Returns all commits from the last N days (up to max_commits)."""
    commits = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Reading git history...", total=None)

        for commit in repo.iter_commits(max_count=max_commits):
            commit_date = datetime.fromtimestamp(
                commit.committed_date, tz=timezone.utc
            )

            # Stop when commits are older than the requested date range
            age_days = (datetime.now(tz=timezone.utc) - commit_date).days
            if age_days > days:
                break

            files_changed = list(commit.stats.files.keys())

            msg = commit.message.strip()
            commits.append(Commit(
                sha=commit.hexsha[:8],
                message=msg,
                author=str(commit.author.name),
                date=commit_date,
                files_changed=files_changed,
                is_bug_fix=is_bug_fix(msg),
                pr_number=extract_pr_number(msg),
            ))

    return commits


def get_file_blame(repo: Repo, file_path: str) -> FileBlame | None:
    """Returns blame information for a file."""
    try:
        blame = repo.blame("HEAD", file_path)
        authors = set()
        latest_date = None

        for blame_commit, _ in blame:
            authors.add(str(blame_commit.author.name))
            commit_date = datetime.fromtimestamp(
                blame_commit.committed_date, tz=timezone.utc
            )
            if latest_date is None or commit_date > latest_date:
                latest_date = commit_date

        return FileBlame(
            path=file_path,
            authors=list(authors),
            last_modified=latest_date or datetime.now(tz=timezone.utc),
            total_commits=len(blame),
        )
    except Exception:
        return None


def get_tracked_files(repo: Repo, src: str | None = None) -> list[str]:
    """
    Returns all tracked files in the repository.

    Args:
        repo: GitPython Repo instance.
        src: Optional subdirectory to restrict analysis to (e.g. 'src/').
             Files outside this path are excluded.
    """
    skip_extensions = {
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
        ".pdf", ".zip", ".tar", ".gz", ".lock", ".sum",
        ".woff", ".woff2", ".ttf", ".eot", ".map",
    }
    skip_dirs = {
        # Package managers / dependencies
        "node_modules", "vendor", "venv", ".venv",
        # Build output
        "dist", "build", ".next", "out", "coverage", "storybook-static",
        # Git internals
        ".git",
        # Python cache
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        # Tooling / CI — not source code
        ".github", ".husky", ".storybook", ".circleci",
        # AI assistants / IDE configs leaking project-internal docs
        # into the scan (caused superset to detect a bogus
        # ``ai-assistant-protocol`` feature from .claude/projects/...).
        ".claude", ".cursor", ".idea", ".vscode", ".aider",
        # Lockfiles directories (some repos)
        ".turbo", ".cache", ".parcel-cache",
        # Test fixtures that produce noise
        ".devcontainer",
    }
    skip_filenames = {
        # Config and lockfiles at any depth
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        ".eslintrc", ".prettierrc", ".editorconfig",
    }

    src_prefix = Path(src) if src else None

    files = []
    for item in repo.tree().traverse():
        if item.type != "blob":
            continue

        path = Path(item.path)

        # Filter to subdirectory if --src is specified
        if src_prefix and not path.is_relative_to(src_prefix):
            continue

        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix.lower() in skip_extensions:
            continue
        if path.name in skip_filenames:
            continue

        files.append(item.path)

    return files
