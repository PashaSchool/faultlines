"""Workspace / monorepo package detection.

Detects package manager workspace configurations and enumerates
sub-packages so each can be analyzed independently for better
feature detection accuracy on large monorepos.

Supported:
  - pnpm (pnpm-workspace.yaml)
  - npm/yarn (package.json workspaces)
  - Turborepo (turbo.json)
  - Nx (nx.json + project.json)
  - Lerna (lerna.json)
  - Cargo (Cargo.toml [workspace])
  - Go (go.work)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path


@dataclass
class WorkspacePackage:
    """A single package/app within a monorepo."""

    name: str
    path: str  # relative to repo root, e.g. "packages/auth"
    files: list[str] = field(default_factory=list)


@dataclass
class WorkspaceInfo:
    """Result of workspace detection."""

    detected: bool
    manager: str  # "pnpm" | "npm" | "yarn" | "turbo" | "nx" | "lerna" | "cargo" | "go" | "none"
    packages: list[WorkspacePackage] = field(default_factory=list)
    root_files: list[str] = field(default_factory=list)  # files not in any package


def detect_workspace(repo_root: str, files: list[str]) -> WorkspaceInfo:
    """Detect workspace configuration and enumerate packages.

    Args:
        repo_root: Absolute path to the repository root.
        files: List of file paths relative to analysis root.

    Returns:
        WorkspaceInfo with detected packages and their files.
    """
    root = Path(repo_root)

    # Try each workspace type in priority order
    for detector in [
        _detect_pnpm,
        _detect_npm_yarn,
        _detect_turbo,
        _detect_nx,
        _detect_lerna,
        _detect_cargo,
        _detect_go,
    ]:
        info = detector(root, files)
        if info and info.detected:
            # Assign files to packages
            _assign_files_to_packages(info, files)
            return info

    return WorkspaceInfo(detected=False, manager="none")


def _assign_files_to_packages(info: WorkspaceInfo, files: list[str]) -> None:
    """Assign each file to its package, or to root_files if unmatched."""
    # Sort packages by path length descending so deeper paths match first
    sorted_pkgs = sorted(info.packages, key=lambda p: len(p.path), reverse=True)

    for f in files:
        matched = False
        for pkg in sorted_pkgs:
            prefix = pkg.path + "/"
            if f.startswith(prefix) or f == pkg.path:
                pkg.files.append(f)
                matched = True
                break
        if not matched:
            info.root_files.append(f)

    # Remove empty packages
    info.packages = [p for p in info.packages if p.files]


def _resolve_globs(root: Path, patterns: list[str]) -> list[WorkspacePackage]:
    """Resolve workspace glob patterns to actual directories."""
    packages: list[WorkspacePackage] = []
    seen: set[str] = set()

    for pattern in patterns:
        pattern = pattern.rstrip("/")

        if "*" in pattern or "?" in pattern:
            # Glob pattern like "packages/*" or "apps/**"
            parent = pattern.split("*")[0].rstrip("/")
            parent_path = root / parent
            if not parent_path.is_dir():
                continue

            # Single level glob
            for child in sorted(parent_path.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    rel = str(child.relative_to(root))
                    if fnmatch(rel, pattern) and rel not in seen:
                        name = _package_name(child)
                        packages.append(WorkspacePackage(name=name, path=rel))
                        seen.add(rel)
        else:
            # Exact path
            exact = root / pattern
            if exact.is_dir() and pattern not in seen:
                name = _package_name(exact)
                packages.append(WorkspacePackage(name=name, path=pattern))
                seen.add(pattern)

    return packages


def _package_name(pkg_dir: Path) -> str:
    """Extract a human-readable package name."""
    # Try reading package.json name
    pkg_json = pkg_dir / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text())
            name = data.get("name", "")
            if name:
                # Strip scope: @myorg/auth → auth
                return name.split("/")[-1]
        except (json.JSONDecodeError, OSError):
            pass

    # Try reading Cargo.toml name
    cargo = pkg_dir / "Cargo.toml"
    if cargo.exists():
        try:
            for line in cargo.read_text().splitlines():
                m = re.match(r'^name\s*=\s*"(.+)"', line)
                if m:
                    return m.group(1)
        except OSError:
            pass

    # Fallback to directory name
    return pkg_dir.name


# ── Workspace type detectors ──


def _detect_pnpm(root: Path, files: list[str]) -> WorkspaceInfo | None:
    """Detect pnpm workspace (pnpm-workspace.yaml)."""
    ws_file = root / "pnpm-workspace.yaml"
    if not ws_file.exists():
        return None

    try:
        content = ws_file.read_text()
        patterns = _parse_yaml_list(content, "packages")
        if not patterns:
            return None

        packages = _resolve_globs(root, patterns)
        return WorkspaceInfo(detected=True, manager="pnpm", packages=packages)
    except OSError:
        return None


def _detect_npm_yarn(root: Path, files: list[str]) -> WorkspaceInfo | None:
    """Detect npm/yarn workspaces (package.json workspaces field)."""
    pkg_json = root / "package.json"
    if not pkg_json.exists():
        return None

    try:
        data = json.loads(pkg_json.read_text())
        workspaces = data.get("workspaces")
        if not workspaces:
            return None

        # yarn can have { packages: [...] } or [...]
        if isinstance(workspaces, dict):
            patterns = workspaces.get("packages", [])
        elif isinstance(workspaces, list):
            patterns = workspaces
        else:
            return None

        if not patterns:
            return None

        manager = "yarn" if (root / "yarn.lock").exists() else "npm"
        packages = _resolve_globs(root, patterns)
        return WorkspaceInfo(detected=True, manager=manager, packages=packages)
    except (json.JSONDecodeError, OSError):
        return None


def _detect_turbo(root: Path, files: list[str]) -> WorkspaceInfo | None:
    """Detect Turborepo — uses npm/yarn/pnpm workspaces underneath."""
    if not (root / "turbo.json").exists():
        return None

    # Turbo relies on the package manager's workspace config
    info = _detect_pnpm(root, files) or _detect_npm_yarn(root, files)
    if info and info.detected:
        info.manager = "turbo"
    return info


def _detect_nx(root: Path, files: list[str]) -> WorkspaceInfo | None:
    """Detect Nx workspace."""
    if not (root / "nx.json").exists():
        return None

    packages: list[WorkspacePackage] = []

    # Nx projects can be in apps/, libs/, packages/ directories
    for search_dir in ["apps", "libs", "packages", "modules"]:
        search_path = root / search_dir
        if search_path.is_dir():
            for child in sorted(search_path.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    # Verify it's an Nx project (has project.json or package.json)
                    if (child / "project.json").exists() or (child / "package.json").exists():
                        name = _package_name(child)
                        packages.append(WorkspacePackage(
                            name=name,
                            path=str(child.relative_to(root)),
                        ))

    if not packages:
        return None

    return WorkspaceInfo(detected=True, manager="nx", packages=packages)


def _detect_lerna(root: Path, files: list[str]) -> WorkspaceInfo | None:
    """Detect Lerna monorepo."""
    lerna_file = root / "lerna.json"
    if not lerna_file.exists():
        return None

    try:
        data = json.loads(lerna_file.read_text())
        patterns = data.get("packages", ["packages/*"])
        packages = _resolve_globs(root, patterns)
        return WorkspaceInfo(detected=True, manager="lerna", packages=packages)
    except (json.JSONDecodeError, OSError):
        return None


def _detect_cargo(root: Path, files: list[str]) -> WorkspaceInfo | None:
    """Detect Cargo workspace (Cargo.toml [workspace])."""
    cargo = root / "Cargo.toml"
    if not cargo.exists():
        return None

    try:
        content = cargo.read_text()
        if "[workspace]" not in content:
            return None

        # Parse members from [workspace] section
        patterns: list[str] = []
        in_members = False
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("members"):
                in_members = True
                # Handle inline: members = ["crate-a", "crate-b"]
                m = re.search(r'\[(.+)\]', line)
                if m:
                    for item in m.group(1).split(","):
                        item = item.strip().strip('"').strip("'")
                        if item:
                            patterns.append(item)
                    in_members = False
                continue
            if in_members:
                if line == "]":
                    in_members = False
                    continue
                item = line.strip(",").strip().strip('"').strip("'")
                if item:
                    patterns.append(item)

        if not patterns:
            return None

        packages = _resolve_globs(root, patterns)
        return WorkspaceInfo(detected=True, manager="cargo", packages=packages)
    except OSError:
        return None


def _detect_go(root: Path, files: list[str]) -> WorkspaceInfo | None:
    """Detect Go workspace (go.work)."""
    go_work = root / "go.work"
    if not go_work.exists():
        return None

    try:
        content = go_work.read_text()
        patterns: list[str] = []
        in_use = False
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("use ("):
                in_use = True
                continue
            if line == ")" and in_use:
                in_use = False
                continue
            if line.startswith("use ") and not in_use:
                patterns.append(line.split()[1])
            elif in_use and line:
                patterns.append(line)

        if not patterns:
            return None

        # Go workspace paths are relative dirs
        packages: list[WorkspacePackage] = []
        for p in patterns:
            p = p.strip().rstrip("/")
            pkg_path = root / p
            if pkg_path.is_dir():
                packages.append(WorkspacePackage(name=pkg_path.name, path=p))

        return WorkspaceInfo(detected=True, manager="go", packages=packages)
    except OSError:
        return None


# ── Utility ──


def _parse_yaml_list(content: str, key: str) -> list[str]:
    """Minimal YAML parser — extracts a list under a given key.

    Avoids requiring PyYAML as a dependency. Handles:
        packages:
          - "apps/*"
          - "packages/*"
    """
    items: list[str] = []
    in_key = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            in_key = True
            continue
        if in_key:
            if stripped.startswith("- "):
                item = stripped[2:].strip().strip('"').strip("'")
                if item:
                    items.append(item)
            elif stripped and not stripped.startswith("#"):
                break  # new key started
    return items
