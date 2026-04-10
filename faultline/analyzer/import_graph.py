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

# Directories whose files act as architectural bridges between features.
# Files in these dirs get a LOWER hub threshold (_SHARED_DIR_FANIN) because
# they're designed to be shared — even moderate fan-in means they connect
# unrelated features and should not bridge clusters.
_SHARED_DIR_NAMES = frozenset({
    "shared", "common", "utils", "utilities", "helpers",
    "hooks", "customHooks", "custom-hooks",
    "lib", "stores", "context", "providers",
    "template", "templates", "base",
})

# Fan-in threshold for files in shared directories — much lower than the base.
_SHARED_DIR_FANIN = 3

# If a Union-Find cluster grows beyond this fraction of all files, it's a sign that
# import chains have connected unrelated modules. In that case the cluster is split
# back into directory-based sub-clusters so LLM receives smaller, focused groups.
_MAX_CLUSTER_FRACTION = 0.25

# Absolute cap on cluster size — even in huge repos, a single cluster
# should never exceed this. Prevents "compiler" (3600 files) scenarios.
_MAX_CLUSTER_ABSOLUTE = 300

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


# Minimum files in a directory subtree to be considered a domain boundary
_MIN_DOMAIN_SIZE = 30


def scan_domains(files: list[str]) -> dict[str, str]:
    """Pre-scans file structure to find natural business domain boundaries.

    Walks the directory tree and identifies directories with _MIN_DOMAIN_SIZE+
    files that have business-meaningful names. These become hard boundaries
    that import graph clustering cannot cross.

    Returns:
        dict mapping file_path → domain_key for every file.
        Files in the same domain share a domain_key.
    """
    # Count files per business domain key
    domain_counts: dict[str, list[str]] = defaultdict(list)
    for f in files:
        key = _business_domain_key(f)
        domain_counts[key].append(f)

    # Domains with enough files become hard boundaries;
    # small domains get merged into a catch-all
    file_to_domain: dict[str, str] = {}
    for domain_key, domain_files in domain_counts.items():
        if len(domain_files) >= _MIN_DOMAIN_SIZE:
            for f in domain_files:
                file_to_domain[f] = domain_key
        else:
            # Small domain — will be clustered freely by import graph
            for f in domain_files:
                file_to_domain[f] = "__open__"

    return file_to_domain


def load_tsconfig_paths(repo_root: str) -> dict[str, str]:
    """Reads tsconfig.json and returns path alias → directory mappings.

    Supports JSONC (tsconfig allows comments and trailing commas).
    Searches for tsconfig.json in repo_root and src/.

    Returns:
        dict mapping alias prefix → resolved directory.
        Example: {"@/": "src/", "~lib/": "lib/"}
    """
    import re as _re

    candidates = [
        os.path.join(repo_root, "tsconfig.json"),
        os.path.join(repo_root, "src", "tsconfig.json"),
        os.path.join(repo_root, "tsconfig.base.json"),
        os.path.join(repo_root, "src", "tsconfig.base.json"),
    ]

    for tsconfig_path in candidates:
        if not os.path.isfile(tsconfig_path):
            continue
        try:
            with open(tsconfig_path) as f:
                content = f.read()
            # Strip single-line and multi-line comments
            content = _re.sub(r"//[^\n]*", "", content)
            content = _re.sub(r"/\*.*?\*/", "", content, flags=_re.DOTALL)
            # Strip trailing commas before } or ]
            content = _re.sub(r",\s*([}\]])", r"\1", content)

            import json as _json
            config = _json.loads(content)
            compiler_opts = config.get("compilerOptions", {})
            base_url = compiler_opts.get("baseUrl", ".")
            paths = compiler_opts.get("paths", {})

            if not paths:
                continue

            tsconfig_dir = os.path.dirname(tsconfig_path)
            result: dict[str, str] = {}

            for alias_pattern, targets in paths.items():
                if not targets or not alias_pattern.endswith("/*"):
                    continue
                alias_prefix = alias_pattern[:-1]  # "@/*" → "@/"
                target = targets[0]  # Take first target
                if target.endswith("/*"):
                    target = target[:-1]  # "./src/*" → "./src/"
                resolved = os.path.normpath(os.path.join(tsconfig_dir, base_url, target))
                resolved = os.path.relpath(resolved, repo_root).replace("\\", "/")
                if not resolved.endswith("/"):
                    resolved += "/"
                result[alias_prefix] = resolved

            if result:
                return result
        except Exception:
            continue

    return {}


def build_import_clusters(
    files: list[str],
    signatures: dict[str, FileSignature],
    tsconfig_paths: dict[str, str] | None = None,
    monorepo_packages: set[str] | None = None,
) -> dict[str, list[str]]:
    """Groups files into clusters based on import dependency relationships.

    Algorithm:
    0. Pre-scan: identify domain boundaries (dirs with 30+ files).
    1. Resolve each import statement to an actual file path in the project.
    2. Count how many files each resolved target is imported by (fan-in).
    3. Build edges, excluding hub files and cross-domain imports as bridges.
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

    # Phase 0: Pre-scan domain boundaries
    file_domains = scan_domains(files)

    # Phase 1: collect all import edges and count fan-in per target file
    edges: list[tuple[str, str]] = []
    fanin: dict[str, int] = defaultdict(int)
    alias_map = tsconfig_paths or {}

    for rel_path, sig in signatures.items():
        if rel_path not in file_set:
            continue
        for import_path in sig.imports:
            resolved = _resolve_import(rel_path, import_path, file_set, alias_map, monorepo_packages)
            if resolved and resolved != rel_path:
                fanin[resolved] += 1
                edges.append((rel_path, resolved))

    # Phase 2: Union-Find — skip hub files and cross-domain imports as bridges
    max_fanin = max(_BASE_IMPORT_FANIN, len(files) // 30)

    # Pre-compute which files live in shared/common directories
    shared_files: set[str] = set()
    for f in files:
        parts = Path(f).parts
        if any(p.lower() in _SHARED_DIR_NAMES for p in parts[:-1]):
            shared_files.add(f)

    uf = _UnionFind(files)
    for importer, imported in edges:
        # Never bridge across domain boundaries
        if file_domains.get(importer) != file_domains.get(imported):
            continue

        threshold = _SHARED_DIR_FANIN if imported in shared_files else max_fanin
        if fanin[imported] <= threshold:
            uf.union(importer, imported)

    raw_groups = uf.groups()

    # Phase 3: Split oversized clusters back into directory sub-clusters.
    # A single import chain can accidentally merge 80%+ of a large codebase.
    # Any cluster exceeding _MAX_CLUSTER_FRACTION of all files is too broad —
    # split by business-domain directory so LLM receives 15-30 medium clusters
    # instead of 300+ micro-clusters that get over-merged back.
    max_size = min(
        _MAX_CLUSTER_ABSOLUTE,
        max(20, int(len(files) * _MAX_CLUSTER_FRACTION)),
    )
    split_groups: dict[str, list[str]] = {}
    for root, members in raw_groups.items():
        if len(members) <= max_size:
            split_groups[root] = members
        else:
            # Re-bucket by business-domain directory (not leaf directory).
            # Uses the first meaningful dir component to create medium-sized groups.
            by_domain: dict[str, list[str]] = defaultdict(list)
            for f in members:
                domain = _business_domain_key(f)
                by_domain[domain].append(f)
            for domain_key, domain_files in by_domain.items():
                split_groups[domain_key] = domain_files

    return _finalize_clusters(split_groups)


def compute_cluster_edges(
    cluster_mapping: dict[str, list[str]],
    signatures: dict[str, FileSignature],
    file_set: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    monorepo_packages: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Computes import connections between clusters.

    Returns:
        {cluster_name: {other_cluster_name: connection_count}}
        where connection_count is the number of file-level import edges
        from files in cluster_name to files in other_cluster_name.
    """
    if not cluster_mapping:
        return {}

    if file_set is None:
        file_set = {f for fs in cluster_mapping.values() for f in fs}

    # Build file → cluster index
    file_to_cluster: dict[str, str] = {}
    for cluster_name, files in cluster_mapping.items():
        for f in files:
            file_to_cluster[f] = cluster_name

    # Count cross-cluster imports
    cross_edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rel_path, sig in signatures.items():
        if rel_path not in file_to_cluster:
            continue
        src_cluster = file_to_cluster[rel_path]
        for import_path in sig.imports:
            resolved = _resolve_import(
                rel_path, import_path, file_set,
                alias_map=alias_map or {},
                monorepo_packages=monorepo_packages,
            )
            if not resolved or resolved not in file_to_cluster:
                continue
            dst_cluster = file_to_cluster[resolved]
            if dst_cluster != src_cluster:
                cross_edges[src_cluster][dst_cluster] += 1

    return dict(cross_edges)


def compute_internal_cohesion(
    cluster_mapping: dict[str, list[str]],
    signatures: dict[str, FileSignature],
    file_set: set[str] | None = None,
    alias_map: dict[str, str] | None = None,
    monorepo_packages: set[str] | None = None,
) -> dict[str, float]:
    """Computes internal import density per cluster.

    Returns:
        {cluster_name: density} where density = internal_edges / (n * (n-1))
        for clusters with n >= 2 files. Single-file clusters get density 1.0.
    """
    if not cluster_mapping:
        return {}

    if file_set is None:
        file_set = {f for fs in cluster_mapping.values() for f in fs}

    file_to_cluster: dict[str, str] = {}
    for cluster_name, files in cluster_mapping.items():
        for f in files:
            file_to_cluster[f] = cluster_name

    internal_edges: dict[str, int] = defaultdict(int)

    for rel_path, sig in signatures.items():
        if rel_path not in file_to_cluster:
            continue
        src_cluster = file_to_cluster[rel_path]
        for import_path in sig.imports:
            resolved = _resolve_import(
                rel_path, import_path, file_set,
                alias_map=alias_map or {},
                monorepo_packages=monorepo_packages,
            )
            if resolved and resolved in file_to_cluster and file_to_cluster[resolved] == src_cluster:
                internal_edges[src_cluster] += 1

    result: dict[str, float] = {}
    for name, files in cluster_mapping.items():
        n = len(files)
        if n <= 1:
            result[name] = 1.0
        else:
            max_possible = n * (n - 1)
            result[name] = internal_edges.get(name, 0) / max_possible if max_possible > 0 else 0.0

    return result


# Generic structural directories — skip these when looking for business domain
_STRUCTURAL_DIRS = {
    "src", "app", "lib", "pkg", "internal", "core",
    "views", "pages", "screens", "routes", "containers",
    "components", "layouts", "features", "modules",
    "hooks", "customHooks", "custom-hooks",
    "shared", "common", "utils", "helpers",
    "stores", "context", "providers", "services",
    "types", "models", "schemas", "constants",
    "assets", "styles", "images", "fonts",
    "test", "tests", "__tests__", "testing",
    "stories", "storybook",
    "new", "old", "legacy", "deprecated",
    # Monorepo containers
    "packages", "apps", "workspaces", "projects",
}


def _business_domain_key(filepath: str) -> str:
    """Extracts a business-domain grouping key from a file path.

    Skips generic structural directories (src/, views/, components/, hooks/, shared/)
    and returns the first THREE meaningful directory components joined with '/'.
    Uses 3 levels to handle nested monorepos (compiler/packages/<name>/src/...).

    Examples:
        src/views/NDR/AutoSegmentation/tables/Foo.tsx           → NDR/AutoSegmentation
        src/components/shared/TrafficPatternsTable.tsx           → components/shared
        compiler/packages/babel-plugin/src/HIR/Build.ts         → compiler/babel-plugin/HIR
        packages/react-devtools-shared/src/backend/renderer.js  → react-devtools-shared/backend
    """
    parts = Path(filepath).parts[:-1]  # exclude filename
    meaningful: list[str] = []

    for part in parts:
        if part.lower() in _STRUCTURAL_DIRS:
            continue
        meaningful.append(part)
        if len(meaningful) >= 3:
            break

    if meaningful:
        return "/".join(meaningful)

    # All dirs are structural — use the deepest two path components
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    elif parts:
        return parts[0]
    return "root"


def _resolve_import(
    importer: str,
    import_path: str,
    file_set: set[str],
    alias_map: dict[str, str] | None = None,
    monorepo_packages: set[str] | None = None,
) -> str | None:
    """Resolves an import path to an actual file in the project.

    Handles:
    - Relative imports: ./foo, ../bar/baz
    - Alias imports: @/foo, ~/bar (mapped to root and src/ root)
    - tsconfig paths: custom aliases from tsconfig.json
    - Monorepo bare imports: 'react-reconciler/src/...' → 'packages/react-reconciler/src/...'
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

    # Try tsconfig path aliases first (most specific)
    if alias_map:
        for alias_prefix, dir_prefix in alias_map.items():
            if import_path.startswith(alias_prefix):
                remainder = import_path[len(alias_prefix):]
                base = dir_prefix + remainder
                result = _try_extensions(base, file_set)
                if result:
                    return result

    # Try built-in alias prefixes (@/, ~/, #/)
    for alias in _ALIAS_PREFIXES:
        if import_path.startswith(alias):
            remainder = import_path[len(alias):]
            # Try as-is (when --src strips the prefix) and with src/ prefix
            for base in (remainder, f"src/{remainder}"):
                result = _try_extensions(base, file_set)
                if result:
                    return result
            return None

    # Try monorepo bare import: 'shared/foo' → 'packages/shared/src/foo'
    if monorepo_packages:
        result = _resolve_monorepo_import(import_path, file_set, monorepo_packages)
        if result:
            return result

    return None  # third-party package — skip


def _resolve_monorepo_import(
    import_path: str,
    file_set: set[str],
    package_names: set[str],
) -> str | None:
    """Resolves a bare monorepo import to a file in the project.

    Handles patterns like:
        'react-reconciler/src/ReactFiber' → 'packages/react-reconciler/src/ReactFiber.js'
        'shared/ReactVersion'             → 'packages/shared/ReactVersion.js'
    """
    # Split: 'react-reconciler/src/Foo' → ('react-reconciler', 'src/Foo')
    # Also handle: 'shared/Foo' → ('shared', 'Foo')
    parts = import_path.split("/", 1)
    if not parts:
        return None

    pkg_name = parts[0]
    remainder = parts[1] if len(parts) > 1 else ""

    if pkg_name not in package_names:
        return None

    # Try common monorepo layouts:
    # packages/<pkg>/src/<remainder>
    # packages/<pkg>/<remainder>
    candidates = []
    if remainder:
        candidates.append(f"packages/{pkg_name}/src/{remainder}")
        candidates.append(f"packages/{pkg_name}/{remainder}")
    else:
        candidates.append(f"packages/{pkg_name}/src/index")
        candidates.append(f"packages/{pkg_name}/index")

    for base in candidates:
        result = _try_extensions(base, file_set)
        if result:
            return result

    return None


def detect_monorepo_packages(repo_root: str) -> set[str]:
    """Detects monorepo package names by scanning packages/, apps/ directories.

    Returns a set of package directory names that can be used as bare import prefixes.
    """
    package_names: set[str] = set()
    for container in ("packages", "apps", "modules", "services"):
        container_dir = os.path.join(repo_root, container)
        if not os.path.isdir(container_dir):
            continue
        for entry in os.listdir(container_dir):
            entry_path = os.path.join(container_dir, entry)
            if os.path.isdir(entry_path):
                package_names.add(entry)

    return package_names


# .js → .ts extension swaps for TypeScript projects that use .js in imports
# (moduleResolution: "node16" / "nodenext" / "bundler")
_JS_TO_TS_SWAPS = {
    ".js": [".ts", ".tsx"],
    ".jsx": [".tsx", ".ts"],
    ".mjs": [".mts", ".ts"],
    ".cjs": [".cts", ".ts"],
}


def _try_extensions(base: str, file_set: set[str]) -> str | None:
    """Tries a path with common TS/JS extensions and as a directory index file.

    Also handles .js→.ts resolution for TypeScript projects that write
    imports with .js extensions (e.g. VSCode, modern ESM projects).

    Returns the first match found in file_set, or None.
    """
    # Exact match (path already has an extension)
    if base in file_set:
        return base

    # Try .js → .ts swap (import says './foo.js' but file is 'foo.ts')
    for js_ext, ts_candidates in _JS_TO_TS_SWAPS.items():
        if base.endswith(js_ext):
            stem = base[: -len(js_ext)]
            for ts_ext in ts_candidates:
                candidate = stem + ts_ext
                if candidate in file_set:
                    return candidate

    # Try appending extensions (bare import without extension)
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


def resolve_symbol_imports(
    signatures: dict[str, FileSignature],
    alias_map: dict[str, str] | None = None,
    monorepo_packages: set[str] | None = None,
) -> dict[str, dict[str, set[str]]]:
    """Resolves which symbols each file imports from other project files.

    Returns:
        {importer_file: {imported_file: {symbol_name, ...}}}
        For namespace imports (import * as X), the symbol set contains "*".
    """
    from faultline.analyzer.ast_extractor import extract_named_imports

    file_set = set(signatures.keys())
    result: dict[str, dict[str, set[str]]] = {}

    for file_path, sig in signatures.items():
        if not sig.source:
            continue
        named = extract_named_imports(sig.source)
        if not named:
            continue

        file_imports: dict[str, set[str]] = {}
        for module_path, symbols in named.items():
            resolved = _resolve_import(
                file_path, module_path, file_set,
                alias_map=alias_map,
                monorepo_packages=monorepo_packages,
            )
            if resolved and resolved in signatures:
                file_imports.setdefault(resolved, set()).update(symbols)

        if file_imports:
            result[file_path] = file_imports

    return result
