"""Deterministic feature detection via co-change community detection.

Files that frequently change together in git history are grouped into the same
feature using Union-Find. This is the primary detection algorithm.

Same git history → same groups every time (100% deterministic).

When --llm is enabled, the LLM *names* the groups (but does not determine them).
Results are cached, so repeated runs return identical names.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

from faultline.models.types import Commit

# Minimum commits in history to trust co-change signal.
# Below this threshold the caller falls back to directory heuristics.
_MIN_COMMITS_FOR_COCHANGE = 50

# Jaccard coupling threshold for merging two files into the same feature.
# Jaccard = commits_touching_both / commits_touching_either.
# 0.20 means "these files change together in ≥20% of commits that touch either one".
_COCHANGE_THRESHOLD = 0.20

# Commits that touch more files than this are excluded (bulk ops / large refactors).
_MAX_FILES_PER_COMMIT = 30

# A file must appear in at least this many commits to participate in coupling.
# Files edited only once produce noisy pairs.
_MIN_FILE_COMMITS = 2

# Directory names that are generic structural wrappers, not business feature names.
_SKIP_DIRS = {
    "src", "app", "lib", "pkg", "internal", "core",
    "views", "pages", "screens", "routes", "containers",
    "components", "layouts", "features",
}


class _UnionFind:
    """Path-compressed, union-by-rank disjoint set data structure."""

    def __init__(self, nodes: list[str]) -> None:
        self._parent: dict[str, str] = {n: n for n in nodes}
        self._rank: dict[str, int] = defaultdict(int)

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            clusters[self.find(node)].append(node)
        return dict(clusters)


def detect_features_from_cochange(
    files: list[str],
    commits: list[Commit],
) -> dict[str, list[str]] | None:
    """Groups files into features based on co-change patterns.

    Returns None when there are fewer than _MIN_COMMITS_FOR_COCHANGE commits —
    the caller should fall back to directory-based heuristics in that case.
    Also returns None if the resulting mapping is empty (no co-change signal).

    The returned dict maps feature_name → list of file paths.
    Names are directory-derived; pass the result to name_clusters_llm() or
    name_clusters_ollama() to replace them with semantic business domain names.

    Args:
        files: Tracked file paths (relative, with path prefix already stripped).
        commits: Commit history for the analysis window.
    """
    if len(commits) < _MIN_COMMITS_FOR_COCHANGE:
        return None

    file_set = set(files)

    # Index: file → set of commit SHAs (non-bulk commits only)
    file_commits: dict[str, set[str]] = defaultdict(set)
    for commit in commits:
        if len(commit.files_changed) > _MAX_FILES_PER_COMMIT:
            continue
        for f in commit.files_changed:
            if f in file_set:
                file_commits[f].add(commit.sha)

    # Inverted index: commit SHA → files (for efficient O(k²) pair counting)
    commit_to_files: dict[str, list[str]] = defaultdict(list)
    for f, shas in file_commits.items():
        if len(shas) < _MIN_FILE_COMMITS:
            continue  # too few appearances — unreliable signal
        for sha in shas:
            commit_to_files[sha].append(f)

    # Count co-occurrences for each file pair
    pair_both: dict[tuple[str, str], int] = defaultdict(int)
    for touched in commit_to_files.values():
        if len(touched) < 2:
            continue
        for f1, f2 in combinations(sorted(touched), 2):
            pair_both[(f1, f2)] += 1

    # Union-Find: merge files whose Jaccard score meets the threshold
    uf = _UnionFind(files)
    for (f1, f2), both in pair_both.items():
        a = len(file_commits.get(f1, set()))
        b = len(file_commits.get(f2, set()))
        denom = a + b - both
        if denom > 0 and both / denom >= _COCHANGE_THRESHOLD:
            uf.union(f1, f2)

    result = _finalize_clusters(uf.groups())
    return result if result else None


def _finalize_clusters(
    raw_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Converts raw Union-Find groups to named feature clusters.

    Multi-file clusters receive a directory-derived name.
    Singleton clusters are merged into a same-directory cluster if one exists,
    or grouped together under a shared directory name.
    """
    multi: dict[str, list[str]] = {}
    singletons: list[str] = []

    for members in raw_groups.values():
        members_sorted = sorted(members)
        if len(members_sorted) >= 2:
            name = _cluster_name(members_sorted)
            name = _unique_name(name, multi)
            multi[name] = members_sorted
        else:
            singletons.extend(members_sorted)

    # Build dir → cluster index so singletons can be absorbed
    dir_to_cluster: dict[str, str] = {}
    for cluster_name, members in multi.items():
        for f in members:
            dir_to_cluster[str(Path(f).parent)] = cluster_name

    # Assign singletons: merge into same-dir cluster or bucket by dir name
    dir_orphans: dict[str, list[str]] = defaultdict(list)
    for f in singletons:
        d = str(Path(f).parent)
        if d in dir_to_cluster:
            multi[dir_to_cluster[d]].append(f)
        else:
            dir_orphans[_feature_name_from_path(f)].append(f)

    for name, fs in dir_orphans.items():
        name = _unique_name(name, multi)
        multi[name] = sorted(fs)

    return multi


def _cluster_name(files: list[str]) -> str:
    """Derives a cluster name from the most common meaningful directory component."""
    counts: dict[str, int] = defaultdict(int)
    for f in files:
        counts[_feature_name_from_path(f)] += 1
    return max(counts, key=lambda k: counts[k])


def _feature_name_from_path(path: str) -> str:
    """Extracts the first non-generic directory component as a feature name."""
    for part in Path(path).parts[:-1]:
        if part.lower() not in _SKIP_DIRS:
            return part.lower()
    return "root"


def _unique_name(name: str, existing: dict) -> str:
    """Returns a unique name by appending a numeric suffix if needed."""
    if name not in existing:
        return name
    suffix = 2
    while f"{name}-{suffix}" in existing:
        suffix += 1
    return f"{name}-{suffix}"
