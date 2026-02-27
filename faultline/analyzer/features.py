import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from faultline.models.types import Commit, Feature, FeatureMap, Flow, PullRequest, TimelinePoint

_MAX_FILES_PER_BULK_COMMIT = 30  # commits touching more files than this are excluded (bulk ops)
_MIN_COCHANGE_COMMITS = 2        # minimum shared commits for a pair to count
_MIN_COCHANGE_SCORE = 0.25       # minimum Jaccard coupling score
_MAX_COCHANGE_PAIRS = 40         # max pairs to return (sorted by score desc)


def compute_cochange(commits: list[Commit]) -> list[tuple[str, str, float]]:
    """
    Computes file co-change coupling from commit history.

    Returns top file pairs sorted by coupling score (descending).
    Coupling score (Jaccard) = commits_touching_both / commits_touching_either.
    Scores >= 0.25 are strong signals that files belong to the same feature.

    Commits touching more than _MAX_FILES_PER_BULK_COMMIT files are excluded
    to avoid noise from bulk operations (formatting, large refactors).
    """
    filtered = [c for c in commits if len(c.files_changed) <= _MAX_FILES_PER_BULK_COMMIT]

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    file_counts: dict[str, int] = defaultdict(int)

    for commit in filtered:
        files = commit.files_changed
        for f in files:
            file_counts[f] += 1
        for f1, f2 in combinations(sorted(files), 2):
            pair_counts[(f1, f2)] += 1

    results: list[tuple[str, str, float]] = []
    for (f1, f2), count in pair_counts.items():
        if count < _MIN_COCHANGE_COMMITS:
            continue
        union = file_counts[f1] + file_counts[f2] - count
        score = count / union if union > 0 else 0.0
        if score >= _MIN_COCHANGE_SCORE:
            results.append((f1, f2, round(score, 2)))

    return sorted(results, key=lambda x: x[2], reverse=True)[:_MAX_COCHANGE_PAIRS]


def detect_features_from_structure(files: list[str]) -> dict[str, list[str]]:
    """
    Detects features based on directory structure.
    This is a heuristic fallback — LLM analysis provides richer results.

    Examples:
        src/auth/login.py      → feature: "auth"
        src/payments/stripe.py → feature: "payments"
        src/api/users.py       → feature: "api"
    """
    features: dict[str, list[str]] = defaultdict(list)

    for file_path in files:
        parts = Path(file_path).parts
        feature_name = _extract_feature_name(parts)
        features[feature_name].append(file_path)

    return dict(features)


def _extract_feature_name(parts: tuple[str, ...]) -> str:
    """
    Extracts a feature name from a file path.

    Logic:
    - src/auth/login.py       → "auth"
    - app/api/payments/...    → "payments"
    - lib/utils/helpers.py    → "utils"
    - index.py                → "root"
    """
    # Skip common top-level wrapper directories (not feature names themselves)
    skip_prefixes = {
        # Generic source roots
        "src", "app", "lib", "pkg", "internal", "core",
        # Frontend structural directories — not business features
        "views", "pages", "screens", "routes", "containers",
        "components", "layouts", "features",
    }

    for i, part in enumerate(parts[:-1]):  # Exclude the filename
        if part.lower() not in skip_prefixes:
            return part.lower()

    # File is at the repo root
    return "root"


_TEST_FILE_RE = re.compile(
    r"\.(test|spec)\.(ts|tsx|js|jsx|py)$"
    r"|/__tests__/"
    r"|/tests?/"
    r"|/test_[^/]+\.py$"
    r"|/[^/]+_test\.py$",
    re.IGNORECASE,
)


def _is_test_file(path: str) -> bool:
    """Returns True if the file path looks like a test file."""
    normalized = "/" + path.replace("\\", "/")
    return bool(_TEST_FILE_RE.search(normalized))


def _is_test_commit(commit: Commit, flow_paths: set[str]) -> bool:
    """
    Returns True if the commit adds/modifies test files for this flow's directory tree.
    Checks whether any changed file is a test file alongside the flow's source files.
    """
    # Derive the directories that contain flow source files
    flow_dirs = {str(Path(p).parent) for p in flow_paths}

    for f in commit.files_changed:
        if not _is_test_file(f):
            continue
        # Counts as a test commit if the test file is in or near a flow directory
        test_dir = str(Path(f).parent)
        if test_dir in flow_dirs or any(test_dir.startswith(d) for d in flow_dirs):
            return True
    return False


def _build_weekly_timeline(
    commits: list[Commit],
    flow_paths: set[str],
) -> list[TimelinePoint]:
    """
    Groups commits by ISO week and counts total, bug-fix, and test commits.
    Returns points sorted chronologically (oldest first).
    """
    weekly: dict[str, dict[str, int]] = defaultdict(lambda: {
        "total": 0, "bug_fix": 0, "test": 0,
    })

    for c in commits:
        iso = c.date.isocalendar()
        label = f"{iso.year}-W{iso.week:02d}"
        weekly[label]["total"] += 1
        if c.is_bug_fix:
            weekly[label]["bug_fix"] += 1
        if _is_test_commit(c, flow_paths):
            weekly[label]["test"] += 1

    return [
        TimelinePoint(
            date=label,
            total_commits=v["total"],
            bug_fix_commits=v["bug_fix"],
            test_commits=v["test"],
        )
        for label, v in sorted(weekly.items())
    ]


def _collect_prs(commits: list[Commit], remote_url: str) -> list[PullRequest]:
    """Returns deduplicated PullRequest objects from bug-fix commits that have a PR number."""
    seen: set[int] = set()
    prs: list[PullRequest] = []
    for c in commits:
        if not c.is_bug_fix or c.pr_number is None or c.pr_number in seen:
            continue
        seen.add(c.pr_number)
        url = f"{remote_url}/pull/{c.pr_number}" if remote_url else ""
        prs.append(PullRequest(
            number=c.pr_number,
            url=url,
            title=c.message.split("\n")[0][:120],
            author=c.author,
            date=c.date,
        ))
    return sorted(prs, key=lambda p: p.date, reverse=True)


def build_feature_map(
    repo_path: str,
    commits: list[Commit],
    feature_paths: dict[str, list[str]],
    days: int,
    remote_url: str = "",
) -> FeatureMap:
    """Builds a FeatureMap by joining commits with detected features."""

    feature_commits: dict[str, list[Commit]] = defaultdict(list)
    feature_authors: dict[str, set[str]] = defaultdict(set)
    feature_last_modified: dict[str, datetime] = {}

    for commit in commits:
        touched_features = set()

        for file_path in commit.files_changed:
            # Find which feature this file belongs to
            for feature_name, paths in feature_paths.items():
                if file_path in paths:
                    touched_features.add(feature_name)
                    break

        for feature_name in touched_features:
            feature_commits[feature_name].append(commit)
            feature_authors[feature_name].add(commit.author)

            # Track the most recent modification date
            if feature_name not in feature_last_modified or \
               commit.date > feature_last_modified[feature_name]:
                feature_last_modified[feature_name] = commit.date

    features = []
    for feature_name, paths in feature_paths.items():
        commits_for_feature = feature_commits.get(feature_name, [])
        total = len(commits_for_feature)
        bug_fixes = sum(1 for c in commits_for_feature if c.is_bug_fix)
        bug_fix_ratio = bug_fixes / total if total > 0 else 0.0

        features.append(Feature(
            name=feature_name,
            paths=paths,
            authors=sorted(feature_authors.get(feature_name, set())),
            total_commits=total,
            bug_fixes=bug_fixes,
            bug_fix_ratio=round(bug_fix_ratio, 3),
            last_modified=feature_last_modified.get(
                feature_name,
                datetime.now(tz=timezone.utc)
            ),
            health_score=_calculate_health(bug_fix_ratio, total),
            bug_fix_prs=_collect_prs(commits_for_feature, remote_url),
        ))

    return FeatureMap(
        repo_path=repo_path,
        remote_url=remote_url,
        analyzed_at=datetime.now(tz=timezone.utc),
        total_commits=len(commits),
        date_range_days=days,
        features=features,
    )


def build_flows_metrics(
    commits: list[Commit],
    flow_file_mappings: dict[str, list[str]],
    remote_url: str = "",
    coverage_data: dict[str, float] | None = None,
) -> list[Flow]:
    """
    Builds Flow objects with commit metrics, mirroring build_feature_map logic.

    Args:
        commits: All commits for the parent feature.
        flow_file_mappings: Dict of flow_name → list of file paths.

    Returns:
        List of Flow objects with health scores and bug fix metrics.
    """
    flow_commits: dict[str, list[Commit]] = defaultdict(list)
    flow_authors: dict[str, set[str]] = defaultdict(set)
    flow_last_modified: dict[str, datetime] = {}

    for commit in commits:
        touched_flows: set[str] = set()
        for file_path in commit.files_changed:
            for flow_name, paths in flow_file_mappings.items():
                if file_path in paths:
                    touched_flows.add(flow_name)
                    break
        for flow_name in touched_flows:
            flow_commits[flow_name].append(commit)
            flow_authors[flow_name].add(commit.author)
            if flow_name not in flow_last_modified or \
               commit.date > flow_last_modified[flow_name]:
                flow_last_modified[flow_name] = commit.date

    # Pre-compute flow directories for test-commit detection
    flow_dirs: dict[str, set[str]] = {
        flow_name: {str(Path(p).parent) for p in paths}
        for flow_name, paths in flow_file_mappings.items()
    }

    # Collect test-only commits per flow (commits that touch adjacent test files
    # but may not touch any flow source file — e.g. standalone test additions)
    flow_test_only_commits: dict[str, list[Commit]] = defaultdict(list)
    seen_shas: dict[str, set[str]] = defaultdict(set)  # avoid duplicates with flow_commits

    for commit in commits:
        for flow_name, dirs in flow_dirs.items():
            if commit.sha in seen_shas[flow_name]:
                continue
            for f in commit.files_changed:
                if _is_test_file(f) and str(Path(f).parent) in dirs:
                    flow_test_only_commits[flow_name].append(commit)
                    seen_shas[flow_name].add(commit.sha)
                    break

    flows = []
    for flow_name, paths in flow_file_mappings.items():
        commits_for_flow = flow_commits.get(flow_name, [])
        total = len(commits_for_flow)
        bug_fixes = sum(1 for c in commits_for_flow if c.is_bug_fix)
        bug_fix_ratio = bug_fixes / total if total > 0 else 0.0

        paths_set = set(paths)
        # Count test files: those explicitly in flow paths + adjacent test files
        # touched by any commit (from test-only commits collected above)
        adjacent_test_files: set[str] = set()
        for c in flow_test_only_commits.get(flow_name, []):
            for f in c.files_changed:
                if _is_test_file(f):
                    adjacent_test_files.add(f)
        test_file_count = (
            sum(1 for p in paths if _is_test_file(p)) + len(adjacent_test_files)
        )

        # Merge source-file commits + test-only commits for the timeline
        seen_for_timeline = {c.sha for c in commits_for_flow}
        timeline_commits = list(commits_for_flow) + [
            c for c in flow_test_only_commits.get(flow_name, [])
            if c.sha not in seen_for_timeline
        ]
        weekly_points = _build_weekly_timeline(timeline_commits, paths_set)

        # Bus factor: authors with ≥20% of flow commits
        threshold = max(1, total * 0.2)
        author_counts: dict[str, int] = {}
        for c in commits_for_flow:
            author_counts[c.author] = author_counts.get(c.author, 0) + 1
        bus_factor = max(1, sum(1 for cnt in author_counts.values() if cnt >= threshold))

        # Health trend: first-half vs second-half bug_fix_ratio delta (positive = improving)
        health_trend: float | None = None
        if len(weekly_points) >= 4:
            mid = len(weekly_points) // 2

            def _bug_ratio(pts: list[TimelinePoint]) -> float:
                total_c = sum(p.total_commits for p in pts)
                return sum(p.bug_fix_commits for p in pts) / total_c if total_c > 0 else 0.0

            health_trend = round(_bug_ratio(weekly_points[:mid]) - _bug_ratio(weekly_points[mid:]), 3)

        # Hotspot files: source files with >40% bug_fix_ratio and ≥3 commits
        file_total: dict[str, int] = {}
        file_bugs: dict[str, int] = {}
        for c in commits_for_flow:
            for f in c.files_changed:
                if f in paths_set and not _is_test_file(f):
                    file_total[f] = file_total.get(f, 0) + 1
                    if c.is_bug_fix:
                        file_bugs[f] = file_bugs.get(f, 0) + 1

        hotspot_files = sorted(
            [f for f, t in file_total.items() if t >= 3 and file_bugs.get(f, 0) / t > 0.4],
            key=lambda f: -(file_bugs.get(f, 0) / file_total[f]),
        )[:5]

        # Coverage: avg line coverage % across non-test source files
        coverage_pct: float | None = None
        if coverage_data:
            coverages = []
            for p in paths:
                if _is_test_file(p):
                    continue
                for cov_path, pct in coverage_data.items():
                    if cov_path.endswith(p) or p.endswith(cov_path.lstrip("/")):
                        coverages.append(pct)
                        break
            if coverages:
                coverage_pct = round(sum(coverages) / len(coverages), 1)

        flows.append(Flow(
            name=flow_name,
            paths=paths,
            authors=sorted(flow_authors.get(flow_name, set())),
            total_commits=total,
            bug_fixes=bug_fixes,
            bug_fix_ratio=round(bug_fix_ratio, 3),
            last_modified=flow_last_modified.get(
                flow_name,
                datetime.now(tz=timezone.utc),
            ),
            health_score=_calculate_health(bug_fix_ratio, total),
            bug_fix_prs=_collect_prs(commits_for_flow, remote_url),
            test_file_count=test_file_count,
            weekly_points=weekly_points,
            bus_factor=bus_factor,
            health_trend=health_trend,
            hotspot_files=hotspot_files,
            coverage_pct=coverage_pct,
        ))

    return flows


def _calculate_health(bug_fix_ratio: float, total_commits: int) -> float:
    """
    Calculates a health score from 0 to 100.
    100 = healthy, 0 = high technical debt.

    Formula:
    - Base score decreases with bug fix ratio (ratio 0.5 → score 0)
    - Activity factor adds confidence for well-tested features
    """
    if total_commits == 0:
        return 100.0

    base_score = max(0.0, 100.0 - (bug_fix_ratio * 200))
    activity_factor = min(1.0, total_commits / 50)

    return round(base_score * activity_factor + base_score * (1 - activity_factor) * 0.8, 1)
