"""Import-graph feature clustering.

Parses import/require statements to build a dependency graph between files.
Files connected through imports form the same feature cluster.

This is the primary grouping algorithm — it captures structural code
relationships independent of git history. Works best for TS/JS/TSX/JSX
projects where import chains naturally reflect feature boundaries.

Hub detection: files imported by many others (shared utilities, type definitions)
are excluded as union bridges to prevent one giant cluster.

Same codebase → same groups every time (100% deterministic).
"""

import os
from collections import defaultdict
from pathlib import Path

from faultline.analyzer.ast_extractor import FileSignature

# Extensions to try when resolving a bare import path ("./auth" → "auth.ts" etc.)
_EXTENSIONS_TO_TRY = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]

# Index file names to try for directory imports ("./auth" → "auth/index.ts" etc.)
_INDEX_FILES = [f"index{ext}" for ext in _EXTENSIONS_TO_TRY]

# Alias prefixes treated as internal project imports
_ALIAS_PREFIXES = ("@/", "~/", "#/")

# Base hub threshold — files imported by more than this many distinct files are "hubs"
# (shared utilities, type barrels). Hub files are excluded from Union-Find bridges
# so they don't merge unrelated features into one giant cluster.
# Scales with codebase size: max(8, file_count // 30).
_BASE_IMPORT_FANIN = 8

# If a Union-Find cluster grows beyond this fraction of all files, it's a sign that
# import chains have connected unrelated modules. In that case the cluster is split
# back into directory-based sub-clusters so LLM receives smaller, focused groups.
_MAX_CLUSTER_FRACTION = 0.25

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
            self._parent[x] = self.find(self._parent[x])
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


def build_import_clusters(
    files: list[str],
    signatures: dict[str, FileSignature],
) -> dict[str, list[str]]:
    """Groups files into clusters based on import dependency relationships.

    Algorithm:
    1. Resolve each import statement to an actual file path in the project.
    2. Count how many files each resolved target is imported by (fan-in).
    3. Build edges, excluding hub files (fan-in > _MAX_IMPORT_FANIN) as bridges.
    4. Union-Find finds connected components — each component is a cluster.
    5. Singleton clusters are absorbed into same-directory clusters.

    Args:
        files: All tracked file paths (relative to project root / --src).
        signatures: Output of extract_signatures() — contains import paths.

    Returns:
        dict mapping cluster_name → list of file paths.
        Names are directory-derived; pass to merge_and_name_clusters_llm()
        for semantic business names.
    """
    if not files:
        return {}

    file_set = set(files)

    # Phase 1: collect all import edges and count fan-in per target file
    edges: list[tuple[str, str]] = []
    fanin: dict[str, int] = defaultdict(int)

    for rel_path, sig in signatures.items():
        if rel_path not in file_set:
            continue
        for import_path in sig.imports:
            resolved = _resolve_import(rel_path, import_path, file_set)
            if resolved and resolved != rel_path:
                fanin[resolved] += 1
                edges.append((rel_path, resolved))

    # Phase 2: Union-Find — skip hub files as bridges
    max_fanin = max(_BASE_IMPORT_FANIN, len(files) // 30)
    uf = _UnionFind(files)
    for importer, imported in edges:
        if fanin[imported] <= max_fanin:
            uf.union(importer, imported)

    raw_groups = uf.groups()

    # Phase 3: Split oversized clusters back into directory sub-clusters.
    # A single import chain can accidentally merge 80%+ of a large codebase.
    # Any cluster exceeding _MAX_CLUSTER_FRACTION of all files is too broad —
    # split it by directory so LLM receives focused, manageable groups.
    max_size = max(20, int(len(files) * _MAX_CLUSTER_FRACTION))
    split_groups: dict[str, list[str]] = {}
    for root, members in raw_groups.items():
        if len(members) <= max_size:
            split_groups[root] = members
        else:
            # Re-bucket by directory: treat as if import graph didn't connect them
            by_dir: dict[str, list[str]] = defaultdict(list)
            for f in members:
                by_dir[str(Path(f).parent)].append(f)
            for dir_path, dir_files in by_dir.items():
                split_groups[dir_path] = dir_files

    return _finalize_clusters(split_groups)


def _resolve_import(
    importer: str,
    import_path: str,
    file_set: set[str],
) -> str | None:
    """Resolves an import path to an actual file in the project.

    Handles:
    - Relative imports: ./foo, ../bar/baz
    - Alias imports: @/foo, ~/bar (mapped to root and src/ root)
    - Bare imports that might be internal: skipped (likely node_modules)

    Returns None if the path resolves outside the file set.
    """
    if import_path.startswith("./") or import_path.startswith("../"):
        importer_dir = str(Path(importer).parent)
        raw = os.path.normpath(os.path.join(importer_dir, import_path))
        # Normalize to forward slashes; skip if resolution went above project root
        base = raw.replace("\\", "/").lstrip("/")
        if base.startswith(".."):
            return None
        return _try_extensions(base, file_set)

    for alias in _ALIAS_PREFIXES:
        if import_path.startswith(alias):
            remainder = import_path[len(alias):]
            # Try as-is (when --src strips the prefix) and with src/ prefix
            for base in (remainder, f"src/{remainder}"):
                result = _try_extensions(base, file_set)
                if result:
                    return result
            return None

    return None  # third-party package — skip


def _try_extensions(base: str, file_set: set[str]) -> str | None:
    """Tries a path with common TS/JS extensions and as a directory index file.

    Returns the first match found in file_set, or None.
    """
    # Exact match (path already has an extension)
    if base in file_set:
        return base

    # Try appending extensions
    for ext in _EXTENSIONS_TO_TRY:
        candidate = base + ext
        if candidate in file_set:
            return candidate

    # Try as a directory index import
    for index_name in _INDEX_FILES:
        candidate = f"{base}/{index_name}"
        if candidate in file_set:
            return candidate

    return None


def _finalize_clusters(
    raw_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Converts raw Union-Find groups to named feature clusters.

    Multi-file clusters get a directory-derived name.
    Singletons are merged into a same-directory cluster when one exists,
    or bucketed under a shared directory name.
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

    # Absorb singletons into same-dir cluster or bucket by dir name
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
    """Returns a unique name, appending a numeric suffix if needed."""
    if name not in existing:
        return name
    suffix = 2
    while f"{name}-{suffix}" in existing:
        suffix += 1
    return f"{name}-{suffix}"
