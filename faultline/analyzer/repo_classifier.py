"""Classifies repository structure type to optimize feature detection strategy.

Detects whether a repo is organized by business features (e.g. src/auth/, src/payments/)
or by technical layers (e.g. src/server/, src/client/, src/lib/) so LLM prompts
can be adapted for better feature grouping.

Also detects whether a repo is a *library* (consumed as a dependency) versus
an *application* (runs end-user flows). Libraries should not produce user-flow
detections since their "users" are developers invoking APIs, not end users
clicking through screens.
"""

from pathlib import Path, PurePosixPath


# Technical layer directory names — when >50% of top-level dirs match,
# the repo is likely layer-organized rather than feature-organized.
LAYER_KEYWORDS = frozenset({
    # Architecture layers
    "server", "client", "backend", "frontend", "web", "mobile",
    # Code organization
    "lib", "libs", "utils", "utilities", "helpers", "common", "shared",
    "core", "internal", "runtime", "vendor", "bundled", "compiled",
    # Technical concerns
    "types", "typings", "models", "schemas", "interfaces",
    "config", "configs", "configuration", "constants",
    "middleware", "interceptors", "guards", "decorators",
    "providers", "adapters", "drivers", "connectors",
    # Build / tooling
    "scripts", "tools", "tooling", "build", "dist", "out",
    "generated", "codegen", "proto", "protos",
    # Testing
    "test", "tests", "__tests__", "testing", "testutils",
    "fixtures", "mocks", "__mocks__",
    # Assets
    "assets", "static", "public", "media", "images", "fonts", "styles",
    # Docs
    "docs", "documentation", "examples", "samples",
})

# Monorepo markers — presence of these directories signals a monorepo.
MONOREPO_MARKERS = frozenset({
    "packages", "apps", "modules", "services", "workspaces", "projects",
})


class RepoStructure:
    """Result of repo structure classification."""

    def __init__(
        self,
        layout: str,
        top_dirs: list[str],
        layer_ratio: float,
        monorepo_root: str | None = None,
        is_library: bool = False,
        library_signals: list[str] | None = None,
    ):
        self.layout = layout              # "feature" | "layer" | "monorepo"
        self.top_dirs = top_dirs          # top-level meaningful dirs
        self.layer_ratio = layer_ratio    # fraction of dirs matching LAYER_KEYWORDS
        self.monorepo_root = monorepo_root  # e.g. "packages" if monorepo
        self.is_library = is_library      # True if repo is a consumable library, not an app
        self.library_signals = library_signals or []  # human-readable reasons for is_library verdict

    def __repr__(self) -> str:
        return (
            f"RepoStructure(layout={self.layout!r}, layer_ratio={self.layer_ratio:.0%}, "
            f"dirs={len(self.top_dirs)}, is_library={self.is_library})"
        )


def classify_repo(
    files: list[str],
    repo_root: str | Path | None = None,
) -> RepoStructure:
    """Classify repository structure from its file list.

    Args:
        files: List of file paths relative to analysis root (path_prefix already stripped).
        repo_root: Optional filesystem path to the repo. When provided, enables
            library-vs-application detection by reading package.json, pyproject.toml,
            go.mod, and config files. Existing callers that omit this argument
            get `is_library=False` (backward compatible).

    Returns:
        RepoStructure with layout type and metadata.
    """
    top_dirs = _extract_top_dirs(files)

    is_library = False
    library_signals: list[str] = []
    if repo_root is not None:
        is_library, library_signals = detect_library(Path(repo_root), files)

    if not top_dirs:
        return RepoStructure(
            layout="feature",
            top_dirs=[],
            layer_ratio=0.0,
            is_library=is_library,
            library_signals=library_signals,
        )

    # Check for monorepo markers first
    monorepo_root = _detect_monorepo(top_dirs)
    if monorepo_root:
        return RepoStructure(
            layout="monorepo",
            top_dirs=top_dirs,
            layer_ratio=0.0,
            monorepo_root=monorepo_root,
            is_library=is_library,
            library_signals=library_signals,
        )

    # Calculate layer ratio
    layer_dirs = [d for d in top_dirs if d.lower() in LAYER_KEYWORDS]
    layer_ratio = len(layer_dirs) / len(top_dirs) if top_dirs else 0.0

    layout = "layer" if layer_ratio > 0.50 else "feature"

    return RepoStructure(
        layout=layout,
        top_dirs=top_dirs,
        layer_ratio=layer_ratio,
        is_library=is_library,
        library_signals=library_signals,
    )


# ── Library detection ─────────────────────────────────────────────────────

# App-builder config files. Presence in a "production" directory (not
# examples/, www/, docs/, etc.) signals the repo is an application.
_APP_CONFIG_PATTERNS = (
    "next.config.js", "next.config.mjs", "next.config.ts", "next.config.cjs",
    "vite.config.js", "vite.config.mjs", "vite.config.ts", "vite.config.cjs",
    "nuxt.config.js", "nuxt.config.ts",
    "astro.config.js", "astro.config.mjs", "astro.config.ts",
    "remix.config.js", "remix.config.cjs",
    "gatsby-config.js", "gatsby-config.ts",
    "svelte.config.js",
    "angular.json",
    "vue.config.js",
    "expo.json",
    "app.json",  # React Native / Expo
)

# Directory name segments that indicate "this is demo/docs, not the actual product".
# An app-config file found only inside these directories does NOT make the repo an app.
_NON_PRODUCTION_DIRS = frozenset({
    "examples", "example", "demo", "demos", "playground", "playgrounds",
    "sandbox", "sandboxes", "tests", "test", "__tests__", "e2e",
    "www", "docs", "documentation", "site", "website", "landing",
})

# Python application entry-point filenames that, when found at repo root,
# indicate the repo is an application rather than a library.
_PY_APP_ENTRYPOINTS = (
    "main.py", "app.py", "manage.py",  # Django/Flask/generic
    "asgi.py", "wsgi.py",               # ASGI/WSGI servers
    "__main__.py",                       # package with main
    "run.py", "server.py",              # common convention
)


def _has_pyproject_console_script(pyproject_path: Path) -> bool:
    """True when ``pyproject.toml`` registers any console script.

    Looks at both standard ``[project.scripts]`` (PEP 621) and
    ``[tool.poetry.scripts]`` (Poetry). A non-empty mapping under
    either is enough — we don't try to validate the entry-point
    targets.

    Tolerates malformed TOML / missing tomllib (fallback to a
    shallow regex scan) so the classifier never crashes on weird
    pyproject files.
    """
    try:
        import tomllib  # Python 3.11+
        try:
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError):
            return False
        proj_scripts = (data.get("project") or {}).get("scripts") or {}
        if isinstance(proj_scripts, dict) and proj_scripts:
            return True
        poetry_scripts = (
            ((data.get("tool") or {}).get("poetry") or {}).get("scripts") or {}
        )
        return isinstance(poetry_scripts, dict) and bool(poetry_scripts)
    except ImportError:  # pragma: no cover — Python < 3.11 fallback
        try:
            text = pyproject_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        # Shallow regex: header line + at least one ``key = "..."``
        # entry beneath it before the next ``[`` header.
        import re as _re
        for header in (r"\[project\.scripts\]", r"\[tool\.poetry\.scripts\]"):
            block = _re.search(rf"{header}\n((?:[^\[]+\n)+)", text)
            if block and _re.search(r"^\s*\w[\w-]*\s*=", block.group(1), _re.MULTILINE):
                return True
        return False


def detect_library(
    repo_root: Path,
    files: list[str],
) -> tuple[bool, list[str]]:
    """Detect whether a repo is a library (consumed as dep) vs an application.

    Multi-language heuristic covering JS/TS, Python, Go, and Rust. Returns
    ``(is_library, signals)`` where signals is a human-readable list of the
    evidence collected — useful for logging and for explaining verdicts to
    users who disagree with the classification.

    The verdict is conservative: when signals are mixed or absent the repo
    is classified as an application (``is_library=False``), because wrongly
    suppressing flow detection on an app is worse than wrongly generating
    flows on a library.

    Args:
        repo_root: Filesystem path to the repository root.
        files: Tracked file paths relative to repo_root (used to detect
            non-production config locations without re-walking the tree).

    Returns:
        Tuple of (is_library, signals). Signals are always returned even
        when is_library is False, so callers can log the reasoning.
    """
    signals: list[str] = []
    app_votes = 0
    lib_votes = 0

    # ── JS/TS signals ─────────────────────────────────────────────────
    # Scan for app-builder configs. Only count those NOT inside
    # examples/, www/, docs/, etc.
    production_configs: list[str] = []
    non_production_configs: list[str] = []
    for f in files:
        basename = PurePosixPath(f).name
        if basename not in _APP_CONFIG_PATTERNS:
            continue
        parts = {p.lower() for p in PurePosixPath(f).parts[:-1]}
        if parts & _NON_PRODUCTION_DIRS:
            non_production_configs.append(f)
        else:
            production_configs.append(f)

    if production_configs:
        app_votes += 2
        signals.append(
            f"found {len(production_configs)} app-builder config(s) in production "
            f"locations (e.g. {production_configs[0]})"
        )
    if non_production_configs and not production_configs:
        lib_votes += 1
        signals.append(
            f"app-builder configs found only in examples/docs/www "
            f"({len(non_production_configs)} file(s)) — treated as library"
        )

    # package.json at root — look for library-like fields
    pkg_json_path = repo_root / "package.json"
    if pkg_json_path.exists():
        try:
            import json
            pkg = json.loads(pkg_json_path.read_text())
        except (json.JSONDecodeError, OSError):
            pkg = {}
        if pkg:
            if pkg.get("main") or pkg.get("exports") or pkg.get("module"):
                lib_votes += 1
                signals.append("package.json declares main/exports/module (library entry points)")
            if pkg.get("private") is False:
                lib_votes += 1
                signals.append("package.json has private=false (published package)")
            # scripts.start/dev running a server is a weak app signal
            scripts = pkg.get("scripts") or {}
            app_cmds = ("next ", "next dev", "vite", "remix dev", "nuxt dev", "astro dev")
            for cmd_name in ("start", "dev"):
                cmd = str(scripts.get(cmd_name, ""))
                if any(cmd.startswith(ac) or f" {ac}" in cmd for ac in app_cmds):
                    app_votes += 1
                    signals.append(f"package.json scripts.{cmd_name} runs an app dev server")
                    break

    # ── Python signals ────────────────────────────────────────────────
    pyproject = repo_root / "pyproject.toml"
    setup_py = repo_root / "setup.py"
    if pyproject.exists() or setup_py.exists():
        # Has a packaging manifest — check for app entry points at
        # the repo root AND for registered console scripts in
        # pyproject.toml. CLI tools (faultline, click, typer apps,
        # etc.) typically register entry points in
        # ``[project.scripts]`` / ``[tool.poetry.scripts]`` instead
        # of dropping a main.py at repo root, so the file-only
        # check produces a false-positive library verdict on every
        # such CLI.
        has_root_entrypoint = any(
            (repo_root / name).exists() for name in _PY_APP_ENTRYPOINTS
        )
        has_console_script = (
            _has_pyproject_console_script(pyproject)
            if pyproject.exists() else False
        )
        if has_root_entrypoint or has_console_script:
            app_votes += 1
            if has_console_script:
                signals.append(
                    "pyproject.toml registers console script(s) — treated as application"
                )
            else:
                signals.append("python app entrypoint found at repo root")
        else:
            lib_votes += 2
            signals.append(
                "pyproject.toml/setup.py present with no main.py/app.py/manage.py "
                "at repo root and no console scripts in [project.scripts]"
            )

    # ── Go signals ────────────────────────────────────────────────────
    if (repo_root / "go.mod").exists():
        has_main_go = (repo_root / "main.go").exists()
        has_cmd_dir = (repo_root / "cmd").is_dir()
        if not has_main_go and not has_cmd_dir:
            lib_votes += 2
            signals.append("go.mod present with no main.go and no cmd/ directory")
        else:
            app_votes += 1
            signals.append("go.mod with main.go or cmd/ present")

    # ── Rust signals ──────────────────────────────────────────────────
    cargo_toml = repo_root / "Cargo.toml"
    if cargo_toml.exists():
        try:
            text = cargo_toml.read_text()
        except OSError:
            text = ""
        has_lib = "[lib]" in text
        has_bin = "[[bin]]" in text or (repo_root / "src" / "main.rs").exists()
        if has_lib and not has_bin:
            lib_votes += 2
            signals.append("Cargo.toml has [lib] and no [[bin]]/main.rs")
        elif has_bin:
            app_votes += 1
            signals.append("Cargo.toml has [[bin]] or src/main.rs")

    # ── Verdict ───────────────────────────────────────────────────────
    is_library = lib_votes > app_votes
    if not signals:
        signals.append("no language-specific signals detected — defaulting to application")

    return is_library, signals


def _extract_top_dirs(files: list[str]) -> list[str]:
    """Get unique first-level directory names from file paths."""
    dirs: set[str] = set()
    for f in files:
        parts = PurePosixPath(f).parts
        if len(parts) > 1:
            dirs.add(parts[0])
    return sorted(dirs)


def _detect_monorepo(top_dirs: list[str]) -> str | None:
    """Check if any top-level dir is a monorepo marker."""
    for d in top_dirs:
        if d.lower() in MONOREPO_MARKERS:
            return d
    return None


def build_layer_context(structure: RepoStructure) -> str:
    """Build extra LLM prompt context for layer-organized repos.

    Returns a string to be appended to the LLM system prompt that instructs
    it to look for cross-cutting business features across technical layers.
    """
    if structure.layout == "feature":
        return ""

    if structure.layout == "monorepo":
        return (
            "\n\n## REPOSITORY STRUCTURE: MONOREPO\n"
            f"This repository is a monorepo with packages under `{structure.monorepo_root}/`. "
            "Each package may contain its own feature set. "
            "Group by business feature ACROSS packages when the same domain spans multiple packages. "
            "A package that is a standalone product should be treated as a single feature or "
            "split into sub-features based on its internal structure."
        )

    # layout == "layer"
    return (
        "\n\n## REPOSITORY STRUCTURE: TECHNICAL LAYERS\n"
        "This codebase is organized by TECHNICAL LAYERS (server/, client/, lib/, shared/, etc.) "
        "rather than by business features. The top-level directories represent architectural "
        "boundaries, NOT business domains.\n\n"
        "To find the real business features, you MUST:\n"
        "1. Look DEEPER — at 2nd and 3rd level subdirectories across all layers.\n"
        "   Example: server/router/ + client/router/ + shared/router/ = one feature \"routing\".\n"
        "2. Cross-cut across layers — files in server/auth/, client/auth/, shared/auth/ "
        "all belong to the same \"auth\" feature.\n"
        "3. Use subdirectory names as feature signals, not top-level directory names.\n"
        "4. Look at sample filenames for domain hints (e.g. image-optimizer.ts → \"image-optimization\").\n\n"
        "WRONG: \"server-runtime\" (technical layer as feature)\n"
        "RIGHT: \"app-router\", \"image-optimization\", \"middleware\", \"dev-overlay\" (business capabilities)"
    )
