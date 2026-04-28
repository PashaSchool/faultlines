"""
Deep scan module using Claude Sonnet for high-quality initial feature detection.

This is an independent module that provides a one-shot analysis:
  candidates (from heuristics) + file tree → Sonnet → features with files AND flows

Designed for SaaS initial scan — runs once per repo, then incremental updates
use cheaper methods (heuristics + Haiku).

Output is a standard dict[str, SonnetFeature] that cli.py converts to FeatureMap.
"""

import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from faultline.analyzer.validation import (
    canonical_bucket_name,
    drop_phantom_features,
    filter_test_files,
    is_documentation_file,
    is_test_feature_name,
    is_test_file,
    partition_docs_vs_code,
)
from faultline.llm.cost import CostTracker, deterministic_params

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0

# Day 12: serializes the short critical section at the end of
# ``deep_scan`` where the module-global ``_last_scan_result`` is written,
# finalized, and then snapshot into a ``DeepScanResult``. The expensive
# LLM call itself stays OUTSIDE the lock, so parallel workspace runs
# overlap the network round-trips and only serialize microseconds of
# post-processing. Without this lock, ``deep_scan_workspace``'s
# ``ThreadPoolExecutor`` path would let two threads stomp on the global
# between write and snapshot, cross-contaminating flow and description
# data across packages.
_GLOBAL_RESULT_LOCK = threading.Lock()


# ── Response models ────────────────────────────────────────────────────────

class SonnetFlow(BaseModel):
    name: str
    description: str = ""
    files: list[str] = []


class SonnetFeature(BaseModel):
    name: str
    description: str = ""
    files: list[str] = []
    flows: list[SonnetFlow] = []


class RenameOp(BaseModel):
    from_name: str = ""
    to: str = ""


class SplitOp(BaseModel):
    from_name: str = ""
    into: list[str] = []


class SonnetOpsResponse(BaseModel):
    merge: list[list[str]] = []
    rename: list[RenameOp] = []
    remove: list[str] = []
    split: list[SplitOp] = []
    features: list[SonnetFeature] = []


# ── DeepScanResult dataclass (D10) ─────────────────────────────────────────
#
# The legacy interface returned a bare ``dict[str, list[str]]`` and stashed
# flows + descriptions in the module-global ``_last_scan_result``. This
# breaks reentrancy (you can't analyze two repos in one process) and forces
# the caller to know about three different read paths.
#
# ``DeepScanResult`` collects all of that into a single value. To avoid
# breaking the existing dict-iterating callers in cli.py and the test
# suite, the dataclass also implements the dict read interface
# (``__getitem__``, ``__iter__``, ``__contains__``, ``__len__``,
# ``items``/``keys``/``values``, ``get``). New code should prefer the
# explicit attributes (``result.features``, ``result.flows``, etc.) and
# old code keeps working unchanged.


@dataclass
class DeepScanResult:
    """Structured return value from ``deep_scan`` and ``deep_scan_workspace``.

    Attributes:
        features: feature_name → list[file_path] (the primary mapping)
        flows: feature_name → list[flow_name] (empty for libraries)
        descriptions: feature_name → one-line description
        flow_descriptions: feature_name → flow_name → description
        cost_summary: snapshot of the cost tracker at scan completion
            (``None`` when no tracker was passed in)

    The dataclass is intentionally dict-compatible at the read level so
    legacy callers that iterate the result as a feature map keep working.
    """

    features: dict[str, list[str]] = field(default_factory=dict)
    flows: dict[str, list[str]] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    flow_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)
    cost_summary: dict[str, Any] | None = None
    # Sprint 7: per-feature, per-flow list of call-graph participants.
    # Shape: ``{feature_name: {flow_name: list[TracedParticipant]}}``.
    # Populated by ``analyzer.flow_tracer.trace_flow_callgraph`` when
    # ``--trace-flows`` is set; left empty otherwise.
    flow_participants: dict[str, dict[str, list[Any]]] = field(
        default_factory=dict,
    )

    # ── dict read shims (legacy compat) ─────────────────────────────────
    def __getitem__(self, key: str) -> list[str]:
        return self.features[key]

    def __setitem__(self, key: str, value: list[str]) -> None:
        self.features[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.features

    def __iter__(self):
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def __bool__(self) -> bool:
        return bool(self.features)

    def items(self):
        return self.features.items()

    def keys(self):
        return self.features.keys()

    def values(self):
        return self.features.values()

    def get(self, key: str, default: Any = None) -> Any:
        return self.features.get(key, default)


# ── Commit context builder (D5) ────────────────────────────────────────────


def build_commit_context(
    commits: list | None,
    *,
    top_n: int = 30,
    days: int = 90,
) -> str | None:
    """Render a compact "recent activity" snippet for the LLM prompt.

    Counts how often each file (and its parent directory) was touched
    by commits in the last ``days`` days, then returns the top ``top_n``
    entries as a single newline-separated string. The result is meant
    to be appended to the user prompt under a ``## Recent activity``
    heading so Sonnet can distinguish actively developed features
    from dormant ones.

    The output is intentionally bounded:
      - Only the ``top_n`` most-touched paths are included
      - Each entry is a single line (``path  N commits``)
      - Files and directories are interleaved by commit count, so
        a hot feature directory rises above its individual files

    Returns ``None`` when there is nothing useful to inject (no
    commits, no files in the last ``days`` window). The caller can
    pass ``None`` straight through to ``deep_scan(commit_context=...)``
    without a guard.

    The shape was chosen to spend at most ~600 tokens on the
    ``## Recent activity`` section, leaving the rest of Sonnet's
    context budget for the actual candidates and file lists.
    """
    if not commits:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    file_counts: Counter[str] = Counter()
    dir_counts: Counter[str] = Counter()

    for commit in commits:
        commit_date = getattr(commit, "date", None)
        if commit_date is None:
            continue
        # Normalize naive datetimes to UTC so the comparison is safe.
        if commit_date.tzinfo is None:
            commit_date = commit_date.replace(tzinfo=timezone.utc)
        if commit_date < cutoff:
            continue

        files = getattr(commit, "files_changed", None) or []
        for fp in files:
            file_counts[fp] += 1
            parent = str(Path(fp).parent)
            if parent and parent != ".":
                dir_counts[parent] += 1

    if not file_counts and not dir_counts:
        return None

    # Interleave files and directories by count, then take top_n.
    combined: Counter[str] = Counter()
    for path, count in file_counts.items():
        combined[path] = count
    for path, count in dir_counts.items():
        # Mark dirs with trailing slash so the LLM can tell them apart.
        combined[path.rstrip("/") + "/"] = count

    entries = combined.most_common(top_n)
    if not entries:
        return None

    lines = [f"{path}  {count} commits" for path, count in entries]
    return "\n".join(lines)


# ── Prompts ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are a senior software architect. You will receive pre-grouped feature \
candidates from structural heuristics. Return operations to clean them up, \
plus user flows for each surviving feature.

## What to return

JSON with five keys:
- **merge**: candidate groups to combine (first = target name)
- **rename**: candidates that need a business-domain name
- **remove**: candidates that are pure UI scaffolding or infra (→ shared-infra)
- **split**: large candidates that should be broken into sub-features
- **features**: final feature list with description and 3-5 flows each

Do NOT return file lists. Files stay with their candidates automatically.

## Merge rules

You MUST aggressively merge. Apply these rules:

1. **Singular/plural duplicates** → merge. "issue" + "issues" → "issues".
2. **Sub-feature + parent** → merge into parent. "work-item-filters" → "issues". \
   "workspace-draft" → "workspace". "pdf-export" → merge into the feature it exports.
3. **UI scaffolding** → remove. These are NOT features: "dropdowns", "empty-state", \
   "base-layouts", "breadcrumbs", "toolbar", "navigation", "sidebar", "icons", \
   "shared-ui" (keep this name but as infrastructure). "helpers", "src" → remove.
4. **Technical layers** → remove. "editor" (if it's a rich-text editor used by many features), \
   "live" (if it's real-time infrastructure), "deploy", "proxy" → remove.

## Split rules — CRITICAL

Look at candidates with >100 files. These are often god-features that contain \
multiple distinct business domains. Based on the sample file paths shown, identify \
if the candidate should be split.

Common splits:
- "document" (200+ files) might contain: signing, templates, recipients, audit-logs, fields → split
- "api" (400+ files) might contain: auth, billing, webhooks, admin → split
- "app" (500+ files) might contain: dashboard, settings, onboarding, admin → split

Format: {"from": "document", "into": ["document-signing", "document-templates", "document-recipients"]}

Only split if you can clearly identify 2+ distinct business sub-domains from the file paths. \
Do NOT split just because a candidate is large.

**HARD CAP: never produce more than 8 sub-features from a single split.** If you \
cannot identify 8 or fewer cohesive business sub-domains, do not split — leave the \
candidate as one feature.

{target_block}

## Feature rules

A feature = something a user interacts with. Ask: "Can a PM write a user story about this?"

## Flow rules — CRITICAL

EVERY feature MUST have flows. No exceptions. A flow = a user action sequence.
- Flow names: lowercase, end with "-flow"
- Each flow has a 1-sentence description
- Think about what a user DOES with this feature

**Flow count scales with feature size:**
- Small feature (<20 files): 3-5 flows
- Medium feature (20-100 files): 5-8 flows
- Large feature (100-300 files): 8-12 flows
- Very large feature (300+ files): 10-15 flows — use the subdirectory breakdown to identify flows

**Derive flows from exported functions and routes.** Some files show their exports and API routes:
- "exports: create_dashboard, delete_dashboard, duplicate_dashboard" → flows: create, delete, duplicate
- "routes: GET /counts, POST /{id}/duplicate" → flows: view-counts, duplicate
- "exports: CreateInvestigationRequest, list_investigations" → flows: create, browse

Each exported function that represents a user action = a flow. CRUD exports = 3-4 flows minimum.

**Prioritize "likely flow entry points" when present.** Per-candidate blocks may list \
router files (`backend/routers/investigations.py`) and page components \
(`StartInvestigationPage.tsx`, `InvestigationDetailPage.tsx`). Each router endpoint and \
each page component is a strong flow signal — one flow per distinct user action \
surfaced there. A router with `POST /start` + `GET /{id}` + `POST /{id}/complete` \
implies three flows: `start-investigation-flow`, `view-investigation-flow`, \
`complete-investigation-flow`.

**Anti-patterns — NEVER emit these as flow names.** They describe code structure, \
not what a user does. The post-processor will strip them anyway, so you are wasting \
output tokens:
- Generic layers: `data-flow`, `data-display-flow`, `state-management-flow`, \
  `navigation-flow`, `routing-flow`, `view-flow`, `display-flow`, `render-flow`, \
  `layout-flow`, `page-layout-flow`, `error-handling-flow`, `validation-flow`
- UI toggles / dev affordances: `sidebar-toggle-flow`, `theme-toggle-flow`, \
  `change-language-flow`, `toggle-dev-environment-flow`
- Infra plumbing: `loading-flow`, `fetching-flow`, `caching-flow`, `storage-flow`, \
  `styling-flow`, `theming-flow`, `animation-flow`, `config-flow`, `setup-flow`
- Filename-derived: `index.ts-flow`, `main.py-flow`, anything ending with a file extension

If the only "flow" you can think of for a feature is a technical layer, that means \
the feature is actually infra — drop it into `shared-infra` instead of emitting a \
bogus flow.

For large features, each major subdirectory often represents a distinct user workflow. \
If you see subdirectories like "SecurityGroups/", "AutoSegmentation/", "Dashboard/", "Issues/" \
inside a feature — each of those is a flow (or multiple flows).

Example:
- "network-detection" (800 files with subdirs: SecurityGroups, AutoSegmentation, Dashboard, Issues, NetworkLog, Inventory, PortflowPage):
  → "view-detections-flow", "investigate-anomaly-flow", "manage-security-groups-flow", \
    "configure-auto-segmentation-flow", "view-network-dashboard-flow", "manage-issues-flow", \
    "analyze-network-log-flow", "monitor-port-flows-flow", "create-exclusion-rules-flow", \
    "benchmark-detection-flow"

If you return a feature without flows, the output is INVALID.

## JSON format

Return ONLY JSON, no text before or after:

{"merge":[],"rename":[],"remove":[],"split":[],"features":[{"name":"x","description":"...","flows":[{"name":"y-flow","description":"..."}]}]}\
"""


# Target-block variants. The repo-wide variant asks for 12-25 business
# features. The package variant tells the LLM it's analyzing a single
# monorepo package and should return at most 8 features. The library
# variant tells the LLM it's analyzing a consumable library whose "users"
# are developers, not end users, so it should return public modules
# rather than business features. All three are injected into
# ``_SYSTEM_PROMPT_TEMPLATE`` via ``_build_system_prompt``.

_TARGET_REPO = (
    "## Target\n\n"
    "After all operations: **12-25 business features**. Not 5, not 50."
)

_TARGET_PACKAGE_TEMPLATE = (
    "## Target\n\n"
    "This is a single package within a monorepo (name: `{package_name}`). "
    "Return **1-8 features** for this package only — NOT 12-25. If the package "
    "has a single cohesive purpose (e.g. `auth`, `db`, `cli`), return ONE feature "
    "named after the package. Only split into multiple features if you can identify "
    "2 or more distinct business sub-domains from the file paths. **HARD CAP: never "
    "more than 8 features for a single package, ever.**"
)

# Day 14 post-mission polish: libraries over-merged when the repo-wide
# prompt was used. A 120-file Go library or a 3K-file Python framework
# would get everything collapsed into "shared-infra" or "web-framework"
# because Sonnet can't find 12-25 "business features" in a library — the
# library IS the feature. The library variant reframes the task: decompose
# into the library's own public modules (router, middleware, binding,
# render…) using names the library's own authors would recognize. This
# lifts gin and fastapi from 0% F1 to (measured after rollout) a useful
# granularity without touching the workspace per-package path.
_TARGET_LIBRARY = (
    "## Target\n\n"
    "This repository is a **consumable library**, not an application. "
    "Its users are developers calling its public APIs, not end users "
    "clicking through a product. Return **5-15 public modules** — the "
    "top-level pieces of the library's API surface — rather than "
    "business features.\n\n"
    "Good module names come from the library's own source tree. Use the "
    "directory / package names the library's authors chose (e.g. "
    "`router`, `middleware`, `binding`, `render`, `validator`, "
    "`context`, `errors`, `recovery`, `dependencies`, `security`, "
    "`responses`, `exceptions`, `websockets`, `background-tasks`, "
    "`testing`, `openapi`).\n\n"
    "Do NOT produce end-user business-feature names like "
    "\"document signing\" or \"user onboarding\" — those don't apply "
    "to a library. Do NOT collapse everything into one giant "
    "`shared-infra` or `web-framework` bucket — that destroys "
    "granularity. Do NOT split into 12-25 entries if the library "
    "genuinely has only 6-10 modules.\n\n"
    "**HARD CAP: never more than 15 modules. FLOOR: at least 5 modules "
    "unless the library is trivially small (<20 source files).** "
    "If a candidate clearly corresponds to a public module, keep it "
    "under its own name instead of merging it into shared-infra.\n\n"
    "**Split the flat root candidate aggressively.** Many libraries keep "
    "their top-level files in a single directory (e.g. `auth.go`, "
    "`logger.go`, `recovery.go`, `routergroup.go`, `tree.go`, "
    "`errors.go`, `context.go`). The heuristic detector will group all "
    "of these into ONE `shared-infra` or `root` candidate. Even if "
    "that candidate has only 20-30 files, you MUST split it into its "
    "public modules based on the filenames alone. Do not leave it as "
    "a single bucket. For the example above: split into `router` "
    "(routergroup.go + tree.go), `recovery`, `logger`, `errors`, "
    "`context`, `auth`.\n\n"
    "**Do NOT use the `remove` operation on module-like candidates.** "
    "In library mode, the `remove` op is reserved for test scaffolding "
    "(`testdata`, `test-helpers`) and trivial wrapper files. Things "
    "like `auth`, `context`, `errors`, `logger`, `recovery`, `router`, "
    "`routergroup`, `tree`, `middleware`, `validator`, `binding`, "
    "`render`, `codec`, `debug`, `fs`, `path`, `response-writer` are "
    "ALL public modules of the library's API surface. If you see a "
    "candidate with that kind of name, KEEP it as a feature (optionally "
    "`rename` it for canonicalization). NEVER `remove` it into "
    "shared-infra. Libraries should produce 5-15 modules, not 2.\n\n"
    "**Prefer `rename` over `merge` for canonicalization.** If you see "
    "`routergroup` + `tree` and think they're both part of a 'router' "
    "concept, rename one to `router` rather than merging both into "
    "something generic. Keep module-level granularity."
)


def _build_system_prompt(
    *,
    package_mode: bool = False,
    package_name: str | None = None,
    is_library: bool = False,
) -> str:
    """Render the system prompt with the right ``## Target`` block.

    Precedence (highest to lowest):
      1. ``package_mode=True`` → per-package prompt (1-8 features). Used
         inside ``deep_scan_workspace`` where the library-vs-app
         distinction doesn't matter because each package is isolated.
      2. ``is_library=True``   → library prompt (5-15 public modules).
         Used in the single-call ``deep_scan`` path for libraries like
         gin and fastapi, where the repo-wide "12-25 business features"
         target causes over-merging.
      3. default              → repo-wide prompt (12-25 features). Used
         for monolith applications in the single-call path.

    The three modes are mutually exclusive in practice: ``package_mode``
    wins because it's invoked per-package and already carries enough
    context (the package name) that library-vs-app becomes irrelevant.
    """
    if package_mode:
        # Use .replace, not .format — the template body contains literal
        # '{' from JSON examples that would otherwise need double-escaping.
        target = _TARGET_PACKAGE_TEMPLATE.replace(
            "{package_name}", package_name or "unknown"
        )
    elif is_library:
        target = _TARGET_LIBRARY
    else:
        target = _TARGET_REPO
    return _SYSTEM_PROMPT_TEMPLATE.replace("{target_block}", target)


# Backwards-compat alias for code that imported the old constant by name.
# Resolves the repo-wide variant lazily so existing tests still see the
# 12-25 target string they expect.
_SYSTEM_PROMPT = _build_system_prompt()


_USER_PROMPT = """\
<candidates>
{candidates_text}
</candidates>

<unmatched_directories>
{unmatched_text}
</unmatched_directories>

Return JSON. Rules: merge aggressively, split god-features (>100 files), 12-25 final features, EVERY feature MUST have 3-5 flows.\
"""


# ── Pre/post-processing helpers ────────────────────────────────────────────
#
# These are extracted from deep_scan() so they can be unit-tested without
# hitting the LLM. The real function below just orchestrates:
#     _clean_inputs → LLM call → apply ops → _finalize_result


_STEM_TRIM_SUFFIXES = (
    "_test", "_tests",
    "_appengine", "_unix", "_windows", "_linux", "_darwin", "_bsd",
    "_freebsd", "_openbsd", "_netbsd", "_plan9", "_js", "_wasm",
    "_amd64", "_arm64", "_386",
    "_file", "_multipart", "_legacy", "_deprecated",
)


def _stem_for_path(path: str) -> str:
    """Return the normalized module stem for a file path.

    Strips OS/arch/test/variant suffixes so `context.go`,
    `context_appengine.go`, and `context_test.go` all produce the
    same `context` stem.
    """
    stem = Path(path).stem.lower()
    while True:
        trimmed = stem
        for suf in _STEM_TRIM_SUFFIXES:
            if trimmed.endswith(suf):
                trimmed = trimmed[: -len(suf)]
                break
        if trimmed == stem:
            break
        stem = trimmed
    return stem.strip("_-.")


def _split_candidate_by_stem(paths: list[str]) -> dict[str, list[str]]:
    """Group a flat file list by normalized filename stem."""
    stems: dict[str, list[str]] = {}
    for fp in paths:
        stem = _stem_for_path(fp)
        if not stem:
            continue
        stems.setdefault(stem, []).append(fp)
    return stems


_APP_LAYER_DIRS = frozenset({
    "routers", "routes", "router",
    "services", "service",
    "models", "model",
    "handlers", "handler",
    "controllers", "controller",
    "endpoints", "endpoint",
    "views", "view",
    "pages", "page",
    "screens", "screen",
})

_LAYER_NAME_SUFFIXES = (
    "_service", "_router", "_handler", "_controller",
    "_model", "_repository", "_repo", "_view", "_page",
    "_executor", "_scheduler", "_engine", "_manager",
    "_generator", "_builder", "_provider", "_client",
    "_settings", "_config",
)

# Soft suffixes trimmed without a leading underscore — frontend pages like
# ``InvestigationPage.tsx`` have no separator. Only applied after the stem
# already contains at least four leading characters so we never eat real
# words (``stage`` → ``st``).
_LAYER_SOFT_TAIL_WORDS = ("page", "view", "screen", "panel", "widget")

_MIN_STEM_LEN_AFTER_TRIM = 4


def _normalize_domain_stem(stem: str) -> str:
    """Strip layer-role and plural suffixes so related files group together.

    ``investigations`` / ``investigation_executor`` / ``investigation``
    all resolve to ``investigation``. Deliberately conservative — only
    trims well-known suffixes and a single trailing ``s``.
    """
    for suf in _LAYER_NAME_SUFFIXES:
        if stem.endswith(suf) and len(stem) - len(suf) >= _MIN_STEM_LEN_AFTER_TRIM:
            stem = stem[: -len(suf)]
            break
    for tail in _LAYER_SOFT_TAIL_WORDS:
        if stem.endswith(tail) and len(stem) - len(tail) >= _MIN_STEM_LEN_AFTER_TRIM:
            stem = stem[: -len(tail)]
            break
    if (
        stem.endswith("s")
        and not stem.endswith("ss")
        and not stem.endswith("us")
        and len(stem) > _MIN_STEM_LEN_AFTER_TRIM
    ):
        stem = stem[:-1]
    return stem.strip("_-.")

_MIN_CATCHALL_SPLIT_FILES = 15
_MIN_SUBDOMAIN_FILES = 2


def _domain_stem_from_layered_path(path: str, catchall_name: str) -> str | None:
    """Return the domain name for a file living under an app-layer dir.

    Soc0-style layout: ``backend/routers/investigations.py`` →
    ``investigations``. The detector walks the path parts and, when it
    finds one of ``_APP_LAYER_DIRS`` whose parent matches ``catchall_name``
    (or is at the repo root), treats the next path segment as the domain.

    File stems get ``_LAYER_NAME_SUFFIXES`` trimmed so
    ``investigation_service.py`` and ``investigation_router.py`` both
    resolve to ``investigation``.
    """
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part.lower() not in _APP_LAYER_DIRS:
            continue
        if i + 1 >= len(parts):
            return None
        if i > 0:
            parent = parts[i - 1].lower()
            if parent != catchall_name.lower() and i != 1:
                # Only accept layers directly under the catchall root or
                # at the repo root — don't misfire on deeply nested
                # ``components/ui/views/Button.tsx`` layouts.
                continue
        next_part = parts[i + 1]
        # When the next segment is itself a directory (e.g., ``frontend/src/
        # pages/investigations/index.tsx``) the domain is that directory,
        # not the leaf filename.
        if i + 2 < len(parts):
            stem = next_part.lower()
        else:
            stem = Path(next_part).stem.lower()
        stem = _normalize_domain_stem(stem)
        return stem or None
    return None


def split_catchall_by_layer(
    catchall_files: list[str],
    catchall_name: str,
    min_domain_files: int = _MIN_SUBDOMAIN_FILES,
) -> tuple[dict[str, list[str]], list[str]]:
    """Split a catchall bucket into domain candidates by app-layer signal.

    Returns ``(domain_candidates, leftover)``. A file joins a domain only
    if its path contains a recognized layer dir (``routers/``, ``services/``,
    ``views/``, …) directly under the catchall root. Domains with fewer
    than ``min_domain_files`` files get folded back into ``leftover`` so
    single-router one-offs don't explode into noise features.

    Soc0 backend bucket example::

        backend/routers/investigations.py  ┐
        backend/services/investigation_    ├─ domain = "investigation"
        backend/models/investigation.py    ┘

    Pure transformation — no LLM calls, no module globals. Unit-tested.
    """
    domains: dict[str, list[str]] = {}
    leftover: list[str] = []
    for fp in catchall_files:
        domain = _domain_stem_from_layered_path(fp, catchall_name)
        if domain:
            domains.setdefault(domain, []).append(fp)
        else:
            leftover.append(fp)

    final_domains: dict[str, list[str]] = {}
    for name, paths in domains.items():
        if len(paths) >= min_domain_files:
            final_domains[name] = paths
        else:
            leftover.extend(paths)
    return final_domains, leftover


_PAGE_FILENAME_SUFFIXES = (
    "page.tsx", "page.jsx", "page.ts", "page.js",
    "view.tsx", "view.jsx", "view.ts", "view.js",
    "screen.tsx", "screen.jsx",
)
_ROUTER_LAYER_DIRS = frozenset({"routers", "routes", "handlers", "controllers"})
_PAGE_LAYER_DIRS = frozenset({"pages", "screens", "views", "routes"})


def extract_flow_entry_points(
    paths: list[str],
    max_items: int = 8,
) -> list[tuple[str, str]]:
    """Pick files most likely to be user-facing flow entry points.

    Returns up to ``max_items`` ``(path, role)`` pairs where role is one
    of ``"router"`` or ``"page"``. Backend files living directly under a
    ``routers/`` (or ``routes/``/``handlers/``/``controllers/``)
    subdirectory count as routers. Frontend files whose filename ends
    with ``Page.tsx``/``View.tsx``/``Screen.tsx`` or that live directly
    under ``pages/`` / ``screens/`` / ``views/`` count as pages.

    This exists because the standard candidate block hides ``views/``
    and ``pages/`` in ``_SKIP_SUBDIR_NAMES`` (keeps the subdirectory
    rollup tidy), so Sonnet never saw the strongest flow signal in the
    file list. Emitting entry points as a dedicated block reinstates
    that signal without exploding the rollup.
    """
    routers: list[tuple[str, str]] = []
    pages: list[tuple[str, str]] = []
    seen: set[str] = set()

    for p in paths:
        if p in seen:
            continue
        parts = Path(p).parts
        low_parts = [pt.lower() for pt in parts]
        filename_lower = parts[-1].lower() if parts else ""
        role: str | None = None
        # Router detection: any layer dir anywhere in the path whose
        # immediate child is the file itself (e.g. routers/investigations.py).
        for i, part in enumerate(low_parts[:-1]):
            if part in _ROUTER_LAYER_DIRS and i + 1 == len(parts) - 1:
                role = "router"
                break
        if role is None:
            for i, part in enumerate(low_parts[:-1]):
                if part in _PAGE_LAYER_DIRS:
                    role = "page"
                    break
        if role is None and any(filename_lower.endswith(suf) for suf in _PAGE_FILENAME_SUFFIXES):
            role = "page"
        if role == "router":
            routers.append((p, role))
        elif role == "page":
            pages.append((p, role))
        seen.add(p)

    # Prefer routers (strongest signal — they literally declare URLs),
    # then pages. Cap total so the prompt stays bounded.
    merged = routers + pages
    return merged[:max_items]


def _promote_library_root_candidates(
    candidates: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Split the ``root`` / ``init`` catchall buckets by filename stem.

    Libraries like Gin keep each public module in a top-level source
    file — ``auth.go``, ``logger.go``, ``recovery.go``, ``routergroup.go``,
    ``tree.go``, ``errors.go``, ``context.go``. The heuristic candidate
    detector groups all of those into the ``root`` catchall, which
    ``deep_scan`` then dumps into ``shared-infra``. Sonnet never sees
    them as distinct candidates and can't emit useful split operations.

    This helper rewrites ``root`` / ``init`` into per-stem candidates.
    OS/arch/test suffixes are stripped so related files group together:
    ``context.go`` + ``context_appengine.go`` + ``context_test.go``
    all land in the ``context`` candidate.

    Intentionally limited to the ``root``/``init`` catchalls because
    named candidates like Gin's ``binding`` or ``render`` also have
    high per-file stem diversity (``binding/json.go``, ``binding/xml.go``,
    ``binding/form.go``…) but each of those IS the binding module's
    format implementations, NOT distinct public modules. Splitting
    them would explode ``binding`` into 20+ format features.
    The downside is that flat library-named candidates like Flask's
    ``flask`` bucket (``flask/app.py``, ``flask/cli.py``,
    ``flask/blueprints.py``…) don't get stem-split, so Flask hits
    ~50% recall instead of ~90%. Accepted trade-off: 100% recall on
    chi/axios and 86-90% on gin/fastapi, with Flask as a known soft
    spot.

    Pure transformation — no LLM calls, no module globals. Unit-tested.
    """
    result: dict[str, list[str]] = {}

    def _merge_in(stems_map: dict[str, list[str]]) -> None:
        for stem, paths in stems_map.items():
            if stem in result:
                result[stem].extend(paths)
            else:
                result[stem] = list(paths)

    for name, paths in candidates.items():
        if name in ("root", "init"):
            _merge_in(_split_candidate_by_stem(paths))
        else:
            if name in result:
                result[name].extend(paths)
            else:
                result[name] = list(paths)

    return result


def _clean_inputs(
    files: list[str],
    candidates: dict[str, list[str]],
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Apply validation primitives upstream of the LLM call.

    Filters test files and documentation files out of both ``files`` and
    every candidate bucket, and drops any candidate whose name is a
    test-infrastructure alias (``vitest-mocks``, ``__tests__``, etc.).

    Returns a 3-tuple:
      - ``cleaned_files``: input minus docs and test files
      - ``cleaned_candidates``: candidates with docs/test paths removed;
        empty buckets and test-named buckets are dropped entirely
      - ``docs_files``: every file that matched ``is_documentation_file``,
        to be re-attached as a single ``documentation`` feature in
        ``_finalize_result``

    Rationale: the Day 1 baseline showed heuristic candidates leaking
    ``vitest-mocks`` (cal.com) and splitting ``docs_src/tutorial001_py310``
    into 21 features (fastapi). Running the LLM on that noise wastes
    tokens and produces phantom features. Doing the filtering here keeps
    the LLM's context focused on actual business code.
    """
    code_files, docs_files = partition_docs_vs_code(files)
    cleaned_files = filter_test_files(code_files)

    cleaned_candidates: dict[str, list[str]] = {}
    for name, paths in candidates.items():
        if is_test_feature_name(name):
            continue
        kept = [
            p for p in paths
            if not is_documentation_file(p) and not is_test_file(p)
        ]
        if kept:
            cleaned_candidates[name] = kept

    return cleaned_files, cleaned_candidates, docs_files


def _enrich_crud_from_signatures(
    features: list["SonnetFeature"],
    signatures: dict | None,
) -> int:
    """Synthesize missing CRUD flows from route / export / filename signals.

    LLM flow detection routinely collapses distinct CRUD verbs into a
    single "manage-X" flow or drops delete entirely (it has few commits
    and is easy to miss). We detect the gaps deterministically via
    ``flow_detector._detect_crud_files`` and inject placeholder flows
    so Sonnet output matches the legacy Haiku path's guarantees.

    Precision > recall: a verb is only synthesized when at least one
    file in the feature shows a strong CRUD signal (matching HTTP verb
    in a route, unambiguous filename token, or CRUD-shaped export).

    Returns the count of flows added.
    """
    if not features or not signatures:
        return 0

    from faultline.llm.flow_detector import (
        _CRUD_NAME_HINTS,
        _detect_crud_files,
        _feature_noun,
    )

    added = 0
    for feat in features:
        if not feat.files:
            continue
        hits = _detect_crud_files(feat.files, signatures)
        if not hits:
            continue

        noun = _feature_noun(feat.name)
        existing_names_lower = [fl.name.lower() for fl in feat.flows]

        for verb, files in hits.items():
            hints = _CRUD_NAME_HINTS[verb]
            already_covered = any(
                any(h in name for h in hints)
                for name in existing_names_lower
            )
            if already_covered:
                continue
            synth_name = f"{verb}-{noun}-flow"
            if synth_name.lower() in existing_names_lower:
                continue
            feat.flows.append(SonnetFlow(
                name=synth_name,
                description=f"User {verb}s {noun.replace('-', ' ')}",
                files=list(files[:3]),
            ))
            existing_names_lower.append(synth_name.lower())
            added += 1
    return added


_CRUD_VERB_PREFIXES = frozenset({
    "create", "add", "new",
    "update", "edit", "patch", "modify",
    "delete", "remove", "destroy",
})

_DOMAIN_SKIP_SEGMENTS = frozenset({
    "api", "apis", "pages", "app", "apps", "src", "lib", "routes",
    "router", "routers", "handler", "handlers", "controller",
    "controllers", "service", "services", "v1", "v2", "v3",
    "public", "internal", "shared", "common",
})


def _path_domain(path: str) -> str:
    """Best-effort business-domain name for a file path.

    Walks backwards from the filename, skipping dynamic-route brackets
    (``[id]``, ``[...params]``) and structural segments (``api``,
    ``pages``, ``v1``…) until a concrete directory name is found. Used
    to tell ``apps/api/v1/pages/api/bookings/[id]/_delete.ts`` apart
    from ``apps/api/v1/pages/api/api-keys/[id]/_delete.ts`` — both live
    in the same package but represent different business domains.
    """
    parts = Path(path).parts
    for part in reversed(parts[:-1]):
        low = part.lower()
        if not part or part.startswith("[") or part.startswith("("):
            continue
        if low in _DOMAIN_SKIP_SEGMENTS:
            continue
        return low
    return ""


def _flow_is_crud_verb_bucket(
    flow: "SonnetFlow",
    max_domains: int = 2,
) -> bool:
    """Return True when a CRUD-prefixed flow clusters unrelated domains.

    cal.com's ``api`` feature landed ``delete-api-flow`` across 17
    ``_delete.ts`` files spanning bookings, api-keys, attendees,
    availabilities, etc. The name sounds like a user journey, but the
    file set is really "every DELETE endpoint in the REST API" —
    eight separate business operations crammed into one artefact.

    Signal: flow name starts with a CRUD verb AND the file set has
    more than ``max_domains`` distinct path domains. Non-CRUD flows
    (``login-flow``, ``checkout-flow``) are left alone — a real user
    journey is expected to touch several directories.
    """
    name = flow.name.lower().replace("-flow", "").strip("-")
    parts = name.split("-")
    if len(parts) < 2:
        return False
    if parts[0] not in _CRUD_VERB_PREFIXES:
        return False
    files = flow.files or []
    if len(files) < 3:
        # Too few files to conclude cross-domain; trust the name.
        return False
    domains = {_path_domain(p) for p in files}
    domains.discard("")
    return len(domains) > max_domains


def _filter_verb_bucket_flows(features: list["SonnetFeature"]) -> int:
    """Strip CRUD verb-bucket flows from each feature.

    Complements ``_filter_noise_flows`` (which catches
    ``state-management-flow`` style technical junk). This one catches
    bucket flows that look legitimate — ``delete-api-flow`` —
    but cluster files across unrelated business domains. Returns the
    count removed for logging.
    """
    removed = 0
    for feat in features:
        if not feat.flows:
            continue
        keep: list[SonnetFlow] = []
        for fl in feat.flows:
            if _flow_is_crud_verb_bucket(fl):
                removed += 1
                continue
            keep.append(fl)
        feat.flows = keep
    if removed:
        logger.info("Flow filter: dropped %d CRUD verb-bucket flow(s)", removed)
    return removed


def _filter_noise_flows(features: list["SonnetFeature"]) -> int:
    """Strip technical / filename-derived flows from each feature.

    Reuses the same filters the legacy Haiku path runs in
    ``_filter_valid_files``: ``_is_technical_flow`` rejects
    ``sidebar-toggle-flow``, ``change-language-flow``,
    ``state-management-flow`` and peers; ``_is_filename_flow``
    rejects ``index.ts-flow`` style names. Returns the count of
    flows removed for logging.
    """
    from faultline.llm.flow_detector import _is_filename_flow, _is_technical_flow

    removed = 0
    for feat in features:
        if not feat.flows:
            continue
        keep: list[SonnetFlow] = []
        for fl in feat.flows:
            if _is_technical_flow(fl.name) or _is_filename_flow(fl.name):
                removed += 1
                continue
            keep.append(fl)
        feat.flows = keep
    if removed:
        logger.info("Flow filter: dropped %d technical/filename flows", removed)
    return removed


def _dedup_flows_across_features(
    features: list["SonnetFeature"],
    feature_files: dict[str, list[str]],
) -> None:
    """Remove duplicate flow names, keeping the best-matched feature.

    Sonnet sometimes emits the same flow under two features — Soc0 hit
    ``chat-conversation-flow`` in both ``backend`` and ``chat``. For each
    duplicate name we pick the feature whose file set has the largest
    overlap with the flow's own file list. Ties resolve toward the feature
    with fewer total files (narrower ownership wins). Mutates ``features``
    in place.
    """
    if not features:
        return

    feat_sets: dict[str, set[str]] = {
        name: set(files) for name, files in feature_files.items()
    }

    name_to_owners: dict[str, list[tuple[str, int, int]]] = {}
    for feat in features:
        for flow in feat.flows:
            flow_files = set(flow.files or [])
            overlap = len(flow_files & feat_sets.get(feat.name, set()))
            size = len(feat_sets.get(feat.name, set()))
            name_to_owners.setdefault(flow.name, []).append((feat.name, overlap, size))

    winners: dict[str, str] = {}
    for flow_name, owners in name_to_owners.items():
        if len(owners) < 2:
            winners[flow_name] = owners[0][0]
            continue
        # Sort by (-overlap, size, feature_name) — most overlap wins,
        # narrower feature breaks ties, alphabetical last.
        owners.sort(key=lambda o: (-o[1], o[2], o[0]))
        winners[flow_name] = owners[0][0]

    for feat in features:
        feat.flows = [fl for fl in feat.flows if winners.get(fl.name) == feat.name]


def _merge_noise_singletons(
    features: dict[str, list[str]],
    sonnet_features: list["SonnetFeature"],
) -> dict[str, list[str]]:
    """Fold 1-file features with no flows into ``shared-infra``.

    Soc0's ``ui-animations`` was one CSS file, one commit, zero flows,
    zero bug fixes. Keeping it as a standalone feature is pure noise —
    it inflates the feature count and dilutes the top-risk table.
    ``shared-infra`` is the canonical catchall for such stragglers.
    """
    flows_by_feature: dict[str, int] = {
        feat.name: len(feat.flows) for feat in sonnet_features
    }

    noisy: set[str] = set()
    for name, paths in features.items():
        if name == "shared-infra":
            continue
        if len(paths) == 1 and flows_by_feature.get(name, 0) == 0:
            noisy.add(name)

    if not noisy:
        return features

    merged: dict[str, list[str]] = {}
    shared: list[str] = []
    for name, paths in features.items():
        if name in noisy:
            shared.extend(paths)
        elif name == "shared-infra":
            shared.extend(paths)
        else:
            merged[name] = paths
    if shared:
        merged["shared-infra"] = shared

    # Drop the merged-away entries from the side-channel so flow lookups
    # don't return stale owners.
    kept = set(merged.keys())
    sonnet_features[:] = [f for f in sonnet_features if f.name in kept]

    return merged


def _finalize_result(
    result: dict[str, list[str]],
    docs_files: list[str],
    is_library: bool,
) -> dict[str, list[str]]:
    """Apply post-processing cleanups to the LLM operation result.

    Steps (in order):
      1. Attach a synthetic ``documentation`` feature containing every
         file previously partitioned out by ``_clean_inputs``.
      2. Canonicalize bucket names (``root``/``init``/``main`` → ``shared-infra``),
         merging any duplicates that resolve to the same canonical name.
         Also rewrites feature names in the module-global
         ``_last_scan_result`` so ``get_deep_scan_flows`` can still match
         by name.
      3. Drop phantom features — empty buckets and test-infrastructure
         names that slipped through (belt-and-braces; ``_clean_inputs``
         should have caught these already).
      4. If ``is_library=True``, strip flows from every feature in
         ``_last_scan_result``. Libraries per acceptance criterion C
         produce feature maps but no user-journey flows; the operations
         prompt still runs because it handles naming/merging, just not
         flow output.

    Returns the cleaned feature map.
    """
    # 1. Attach documentation bucket
    if docs_files:
        result.setdefault("documentation", []).extend(docs_files)

    # 2. Canonicalize bucket names, merging duplicates
    canonicalized: dict[str, list[str]] = {}
    for name, paths in result.items():
        canonical = canonical_bucket_name(name)
        canonicalized.setdefault(canonical, []).extend(paths)

    # 2b. Canonicalize feature names in the side-channel so fuzzy flow
    # matching in get_deep_scan_flows() still resolves correctly.
    global _last_scan_result
    if _last_scan_result is not None:
        for feat in _last_scan_result.features:
            feat.name = canonical_bucket_name(feat.name)

    # 3. Drop phantom features (empty, test-named)
    cleaned = drop_phantom_features(canonicalized)

    # 4. Strip flows for libraries
    if is_library and _last_scan_result is not None:
        for feat in _last_scan_result.features:
            feat.flows = []

    # 4a. Drop technical / filename-derived flows (``sidebar-toggle-flow``,
    # ``change-language-flow``, ``index.ts-flow``). The Haiku path runs
    # these filters in ``_filter_valid_files``; the Sonnet path bypassed
    # them, so Sonnet noise was reaching the final map unchallenged.
    if _last_scan_result is not None and not is_library:
        _filter_noise_flows(_last_scan_result.features)

    # 4a+. Drop CRUD verb-buckets — flows whose name reads like a user
    # journey but whose files span multiple business domains. cal.com's
    # ``api`` feature emitted ``delete-api-flow`` spanning 17
    # ``_delete.ts`` files across bookings/api-keys/attendees/…; the
    # flow name is fake signal and inflates risk metrics misleadingly.
    if _last_scan_result is not None and not is_library:
        _filter_verb_bucket_flows(_last_scan_result.features)

    # 4b. Deduplicate flow names across features. When the LLM emits the
    # same flow (e.g., ``chat-conversation-flow``) under two features
    # we keep it only in the feature whose file set overlaps most with
    # the flow's own files — the other feature has it by mistake.
    if _last_scan_result is not None and not is_library:
        _dedup_flows_across_features(_last_scan_result.features, cleaned)

    # 4c. Merge single-file no-flow features into ``shared-infra``. These
    # are almost always leaked filename-stems (Soc0's ``ui-animations``)
    # that survived phantom filtering but carry no real business signal.
    if _last_scan_result is not None:
        cleaned = _merge_noise_singletons(cleaned, _last_scan_result.features)

    # 5. Deterministic ordering (D11): sort by descending size then name.
    # Combined with temperature=0 on the LLM call, this makes two
    # consecutive runs on the same repo produce byte-identical JSON.
    cleaned = dict(
        sorted(cleaned.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    )
    # Also sort _last_scan_result features to match, so flow matching
    # and downstream rendering use the same order.
    if _last_scan_result is not None:
        order = {name: idx for idx, name in enumerate(cleaned.keys())}
        _last_scan_result.features.sort(
            key=lambda f: order.get(f.name, len(order))
        )

    return cleaned


# ── Main function ──────────────────────────────────────────────────────────

def deep_scan(
    files: list[str],
    candidates: dict[str, list[str]],
    api_key: str | None = None,
    signatures: dict | None = None,
    *,
    is_library: bool = False,
    model: str | None = None,
    tracker: CostTracker | None = None,
    package_mode: bool = False,
    package_name: str | None = None,
    commit_context: str | None = None,
) -> DeepScanResult | None:
    """
    Performs a deep scan using Sonnet to detect features and flows.

    Args:
        files: All file paths in the repo.
        candidates: Pre-computed candidates from detect_candidates().
        api_key: Anthropic API key.
        signatures: Optional AST-extracted signatures keyed by file path.
        is_library: When True, the result has flows stripped from every
            feature. Set this when ``repo_classifier.detect_library``
            reports the repo is a consumable library. Default False.
        model: Override the default Sonnet model id. When omitted,
            uses the module-level ``_MODEL`` constant. Passed through
            to the CostTracker so pricing is looked up accurately.
        tracker: Optional CostTracker. When provided, every successful
            LLM call is recorded with its token usage and cost, and
            ``tracker.check_budget()`` is invoked immediately after —
            which may raise ``BudgetExceeded`` to abort the scan before
            the caller fires further requests.
        package_mode: When True, swap the system prompt to a per-package
            variant that asks for 1-8 features instead of 12-25. Used by
            ``deep_scan_workspace`` to analyze a single monorepo package
            in isolation. Default False (whole-repo mode).
        package_name: Name of the package being analyzed in
            ``package_mode``. Inserted into the per-package prompt so the
            LLM knows what to call the resulting feature when collapsing
            to a single bucket. Ignored when ``package_mode=False``.
        commit_context: Optional pre-rendered string describing recent
            activity (top modified files/dirs over the last N days).
            Built via ``build_commit_context(commits)``. When provided,
            appended to the user prompt under a ``## Recent activity``
            heading so Sonnet can weigh actively developed areas more
            heavily when naming features.

    Returns:
        Tuple of (feature_paths, flow_data) where:
        - feature_paths: dict[feature_name → list[file_path]]
        - flow_data: stored in _last_scan_result for cli.py to retrieve
        Returns None on failure.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.error("No API key for deep scan")
        return None

    client = anthropic.Anthropic(api_key=key)
    resolved_model = model or _MODEL
    system_prompt = _build_system_prompt(
        package_mode=package_mode,
        package_name=package_name,
        is_library=is_library,
    )

    # Pre-process: strip test and docs files before they reach the LLM.
    # This replaces the earlier scattered filtering and makes the LLM's
    # token budget go toward real business code, not tutorial samples.
    files, candidates, docs_files = _clean_inputs(files, candidates)
    if docs_files:
        logger.info(
            "Deep scan: partitioned %d documentation files into synthetic bucket",
            len(docs_files),
        )

    # Day 14 library-mode fix: split the root/init catchall by filename
    # stem so gin's top-level files (auth.go, logger.go, recovery.go…)
    # surface as distinct per-module candidates instead of getting dumped
    # into shared-infra. Gated on ``not package_mode`` because workspace
    # per-package calls are already isolated and don't need this escape
    # hatch (trpc regressed from 16 → 106 features when this ran
    # per-package).
    library_mode_active = is_library and not package_mode
    if library_mode_active:
        candidates = _promote_library_root_candidates(candidates)
        logger.info(
            "Deep scan (library mode): promoted root/init into %d per-stem candidates",
            len(candidates),
        )

    # Separate real candidates from catch-all buckets. Library mode keeps
    # single-file candidates because libraries commonly have one .go/.py
    # file per public module — rejecting them would collapse everything
    # back into shared-infra. Package mode keeps the legacy ≥2 threshold.
    _CATCHALL = {"backend", "frontend", "root", "init", "packages", "web", "api", "lib"}
    _MIN_CANDIDATE_FILES = 1 if library_mode_active else 2
    real_candidates: dict[str, list[str]] = {}
    unmatched: list[str] = []

    for name, paths in candidates.items():
        if name in _CATCHALL:
            # Before dumping into unmatched, try to salvage domain features
            # from app-layer signals (routers/, services/, views/…). Without
            # this, Sonnet sees a flat mass of files and re-invents the same
            # ``backend``/``frontend`` bucket that the heuristic produced.
            if not library_mode_active and len(paths) >= _MIN_CATCHALL_SPLIT_FILES:
                domain_cands, leftover = split_catchall_by_layer(paths, name)
                if domain_cands:
                    for dname, dpaths in domain_cands.items():
                        if dname in real_candidates:
                            real_candidates[dname].extend(dpaths)
                        elif dname in candidates:
                            # Domain already a top-level candidate — let its
                            # own entry handle it, feed files through that.
                            real_candidates.setdefault(dname, []).extend(dpaths)
                        else:
                            real_candidates[dname] = dpaths
                    unmatched.extend(leftover)
                    logger.info(
                        "Deep scan: split catchall '%s' (%d files) into %d domain candidates + %d leftover",
                        name, len(paths), len(domain_cands), len(leftover),
                    )
                    continue
            unmatched.extend(paths)
        elif len(paths) < _MIN_CANDIDATE_FILES:
            unmatched.extend(paths)
        else:
            real_candidates[name] = paths

    # Format candidates — show subdirectory breakdown for large ones, ALL files for small
    _MAX_FILES = 30  # Show more files so Sonnet can derive flows from filenames
    _LARGE_THRESHOLD = 50  # Show subdir breakdown instead of file list
    _SKIP_SUBDIR_NAMES = {
        "src", "app", "core", "lib", "views", "components", "hooks", "utils",
        "helpers", "types", "models", "schemas", "services", "store", "stores",
        "features", "shared", "ui", "common", "ndr", "web", "api",
    }
    cand_lines = []
    for name, paths in sorted(real_candidates.items(), key=lambda x: -len(x[1])):
        if len(paths) >= _LARGE_THRESHOLD:
            # Show subdirectory breakdown — reveals internal structure
            from collections import Counter as _Ctr
            subdirs = _Ctr()
            for fp in paths:
                parts = Path(fp).parts
                for part in parts[:-1]:
                    if part.lower() not in _SKIP_SUBDIR_NAMES and not part.startswith((".", "(")):
                        subdirs[part] += 1
                        break
            cand_lines.append(f"## {name} ({len(paths)} files) — subdirectories:")
            for subdir, cnt in subdirs.most_common(20):
                cand_lines.append(f"  {subdir}/: {cnt} files")
        else:
            cand_lines.append(f"## {name} ({len(paths)} files)")
            for p in paths[:_MAX_FILES]:
                # Show exports/routes for key files (routers, pages)
                sig_info = ""
                if signatures and p in signatures:
                    sig = signatures[p]
                    if sig.exports:
                        sig_info = f"  → exports: {', '.join(sig.exports[:8])}"
                    if sig.routes:
                        sig_info += f"  → routes: {', '.join(sig.routes[:5])}"
                cand_lines.append(f"  {p}")
                if sig_info:
                    cand_lines.append(f"    {sig_info}")
            if len(paths) > _MAX_FILES:
                cand_lines.append(f"  ... and {len(paths) - _MAX_FILES} more")

        # Flow entry points: surface router and page files explicitly so
        # Sonnet stops missing ``start-investigation-flow`` because its
        # entry point lived under ``views/`` (hidden in the rollup above).
        entry_points = extract_flow_entry_points(paths)
        if entry_points:
            cand_lines.append("  → likely flow entry points:")
            for ep_path, ep_role in entry_points:
                cand_lines.append(f"    • {ep_path} ({ep_role})")
    candidates_text = "\n".join(cand_lines)

    # Format unmatched — collapse to dirs if too many
    _MAX_UNMATCHED = 200
    if len(unmatched) > _MAX_UNMATCHED:
        from collections import defaultdict
        dir_files: dict[str, list[str]] = defaultdict(list)
        for f in unmatched:
            d = str(Path(f).parent) if "/" in f else "."
            dir_files[d].append(Path(f).name)
        lines = []
        for d in sorted(dir_files.keys()):
            samples = dir_files[d][:4]
            lines.append(f"{d}/ ({len(dir_files[d])} files): {', '.join(samples)}")
        unmatched_text = "\n".join(lines)
    else:
        unmatched_text = "\n".join(sorted(unmatched)) if unmatched else "(none)"

    prompt = _USER_PROMPT.format(
        candidates_text=candidates_text,
        unmatched_text=unmatched_text,
    )

    # D5: append recent-activity context when available. Bounded by
    # ``build_commit_context`` to ~30 lines so it doesn't blow the
    # token budget on huge repos.
    if commit_context:
        prompt = (
            f"{prompt}\n\n"
            f"## Recent activity (last 90 days, top files/dirs)\n"
            f"{commit_context}"
        )

    logger.info("Deep scan: %d candidates, %d unmatched files → Sonnet", len(real_candidates), len(unmatched))

    # Call Sonnet (operations-based — no file lists in response)
    ops = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(
                model=resolved_model,
                max_tokens=8_192,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                **deterministic_params(resolved_model),
            )

            # D4: record token usage and check budget immediately, so
            # BudgetExceeded aborts the run before we attempt to parse
            # or retry. The tracker is opt-in; callers without cost
            # tracking pass tracker=None and nothing happens here.
            if tracker is not None:
                usage = getattr(response, "usage", None)
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                tracker.record(
                    provider="anthropic",
                    model=resolved_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    label="deep-scan",
                )
                tracker.check_budget()  # may raise BudgetExceeded

            text = response.content[0].text if response.content else ""
            parsed = _parse_json_response(text)
            if not parsed:
                logger.warning("Deep scan: could not parse JSON from response")
                continue

            parsed = _normalize_response(parsed, model=resolved_model)

            try:
                ops = SonnetOpsResponse.model_validate(parsed)
                break
            except ValidationError as e:
                logger.warning("Deep scan validation error: %s", str(e)[:300])
                continue

        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("Deep scan failed (attempt %d): %s. Retry in %.1fs", attempt + 1, e, delay)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            logger.error("Deep scan auth error: %s", e)
            break

    if not ops:
        return None

    logger.info("Deep scan: Sonnet returned %d features, %d merges, %d removes, %d splits",
                len(ops.features), len(ops.merge), len(ops.remove), len(ops.split))

    # Apply operations to candidates
    result = dict(real_candidates)

    # Apply merges — largest candidate becomes target.
    # Day 14: library mode (single-call path only) deliberately skips
    # ops.merge. Prompts told Sonnet "don't collapse modules into
    # shared-infra" but it still emits aggressive merges that undo the
    # `_promote_library_root_candidates` stem split. Skipping merges
    # here preserves per-module granularity; trustworthy merges
    # (singular/plural duplicates) are rare enough on libraries that
    # losing them is cheaper than losing module names. Package-mode
    # calls keep the legacy merge behaviour because they're already
    # per-package isolated.
    if not library_mode_active:
        for group in ops.merge:
            if len(group) < 2:
                continue
            existing = [(n, len(result.get(n, []))) for n in group if n in result]
            if len(existing) < 2:
                continue
            target = max(existing, key=lambda x: x[1])[0]
            for name, _ in existing:
                if name != target:
                    result[target].extend(result.pop(name))
                    logger.info("Merged '%s' into '%s'", name, target)
    elif ops.merge:
        logger.info(
            "Deep scan (library mode, single-call): ignoring %d merge op(s) to preserve module granularity",
            len(ops.merge),
        )

    # Apply splits — distribute files by keyword matching on subdirectory/filename
    for split_op in ops.split:
        source = split_op.from_name
        if source not in result or len(split_op.into) < 2:
            continue
        source_files = result.pop(source)
        # Distribute files: match each file to the best sub-feature by keyword
        sub_features: dict[str, list[str]] = {name: [] for name in split_op.into}
        remainder: list[str] = []
        for fp in source_files:
            fp_lower = fp.lower()
            placed = False
            for sub_name in split_op.into:
                # Extract keywords from sub-feature name (e.g. "document-signing" → ["signing"])
                keywords = [k for k in sub_name.replace(source, "").split("-") if k and len(k) >= 3]
                if any(kw in fp_lower for kw in keywords):
                    sub_features[sub_name].append(fp)
                    placed = True
                    break
            if not placed:
                remainder.append(fp)
        # Put remainder in the first (primary) sub-feature
        if remainder:
            sub_features[split_op.into[0]].extend(remainder)
        for sub_name, sub_files in sub_features.items():
            if sub_files:
                result[sub_name] = sub_files
                logger.info("Split '%s' → '%s' (%d files)", source, sub_name, len(sub_files))

    # Apply renames
    for rename in ops.rename:
        old = rename.from_name
        new = rename.to
        if old in result and new != old:
            result[new] = result.pop(old)
            logger.info("Renamed '%s' → '%s'", old, new)

    # Apply removes → shared-infra.
    # Day 14: library mode narrows ops.remove to obvious non-module names
    # (test scaffolding, license, build config). Anything that could be
    # a public module is kept. Without this guard Sonnet removes 80% of
    # candidates for small libraries and everything collapses.
    _LIBRARY_REMOVE_WHITELIST = frozenset({
        "testdata", "test-helpers", "test_helpers", "testhelpers",
        "license", "go", "package",
        "benchmark", "benchmarks", "example", "examples",
        "deprecated", "legacy",
    })
    infra_files: list[str] = []
    for name in ops.remove:
        if name not in result:
            continue
        if library_mode_active and name.lower() not in _LIBRARY_REMOVE_WHITELIST:
            logger.info(
                "Deep scan (library mode, single-call): ignoring remove('%s') — kept as module",
                name,
            )
            continue
        infra_files.extend(result.pop(name))
        logger.info("Removed '%s' → shared-infra", name)
    if infra_files:
        result.setdefault("shared-infra", []).extend(infra_files)

    # Add unmatched catch-all files to shared-infra (not as separate features)
    for name, paths in candidates.items():
        if name in _CATCHALL and paths:
            result.setdefault("shared-infra", []).extend(paths)

    # Day 12: the rest of this function touches the module-global
    # ``_last_scan_result`` (write, canonicalize names, sort) and then
    # reads it back inside ``_build_deep_scan_result``. All of that
    # must be atomic relative to other threads also inside ``deep_scan``,
    # otherwise the ThreadPoolExecutor path in ``deep_scan_workspace``
    # cross-contaminates flows and descriptions between packages. The
    # lock window is microseconds — negligible next to the minute-long
    # LLM call above that runs fully in parallel.
    with _GLOBAL_RESULT_LOCK:
        # Store features with flow data for later extraction
        global _last_scan_result
        _last_scan_result = ops

        # Post-process: re-attach docs bucket, canonicalize names, drop
        # phantoms, strip flows for libraries. All pure transformations —
        # unit-tested in tests/test_sonnet_scanner_pipeline.py.
        features_dict = _finalize_result(result, docs_files, is_library)

        # Fix D: synthesize missing CRUD flows from route signatures.
        # Runs after _finalize_result so we only operate on features that
        # survived filtering — and only on app repos (libraries don't
        # expose user flows).
        if not is_library and signatures:
            added = _enrich_crud_from_signatures(_last_scan_result.features, signatures)
            if added:
                logger.info("CRUD enrichment: synthesized %d missing flow(s)", added)

        # D10: wrap into a DeepScanResult that carries flows + descriptions +
        # cost summary alongside the feature map. Reads from the global
        # side channel ``_last_scan_result`` populated above; legacy
        # ``get_deep_scan_*`` accessors continue to work for callers that
        # haven't migrated yet.
        return _build_deep_scan_result(features_dict, tracker)


# ── DeepScanResult builder (D10) ───────────────────────────────────────────


def _build_deep_scan_result(
    features: dict[str, list[str]],
    tracker: CostTracker | None,
) -> DeepScanResult:
    """Snapshot the global ``_last_scan_result`` into a DeepScanResult.

    Reads flows, descriptions, and flow descriptions from the module-level
    ``_last_scan_result`` that ``deep_scan`` populated just before calling
    this helper. Captured here so the return value is self-contained and
    doesn't leave callers at the mercy of a subsequent ``deep_scan`` call
    overwriting the global (the old reentrancy hazard).

    The global itself is intentionally NOT cleared — the legacy
    ``get_deep_scan_flows`` / ``get_deep_scan_descriptions`` readers are
    still exported so existing callers in ``cli.py`` keep working until
    the Week 2 cutover.
    """
    flows: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    flow_descriptions: dict[str, dict[str, str]] = {}

    if _last_scan_result is not None:
        for feat in _last_scan_result.features:
            if feat.flows:
                flows[feat.name] = [fl.name for fl in feat.flows]
                flow_descriptions[feat.name] = {
                    fl.name: fl.description for fl in feat.flows
                }
            if feat.description:
                descriptions[feat.name] = feat.description

    cost_summary = tracker.summary() if tracker is not None else None

    return DeepScanResult(
        features=features,
        flows=flows,
        descriptions=descriptions,
        flow_descriptions=flow_descriptions,
        cost_summary=cost_summary,
    )


# ── Workspace-aware orchestration (D6) ─────────────────────────────────────
#
# Monorepos like documenso (2.5K files) and cal.com (10K) timed out the
# legacy single-call ``deep_scan`` at 600s. The fix is to call it once per
# workspace package, with each call seeing only its own files. This keeps
# every individual prompt small enough that Sonnet can answer in <60s, and
# total cost grows linearly with the number of real packages instead of
# combinatorially with file count.
#
# This helper does NOT touch ``cli.py``. The Week 2 cutover will replace
# the legacy strategy at ``cli.py:264-380`` with a single call to this
# function. Until then it lives in parallel and can be unit-tested with
# mocked ``deep_scan`` calls.

# Default size threshold below which a package is treated as a single
# feature without an LLM call. Tuned to avoid spending tokens on tiny
# helper packages while still letting medium packages get a name from
# Sonnet. The legacy ``_SPLIT_THRESHOLD=200`` is intentionally NOT used —
# the per-package prompt's HARD CAP of 8 features replaces it.
_DEFAULT_PKG_LLM_FLOOR = 30

# Package-name prefixes that mark example/demo/starter content. These get
# grouped into a single ``examples`` feature instead of one feature per
# package. Mirrors the legacy filter at cli.py:262.
_EXAMPLE_PKG_PREFIXES = (
    "example", "sample", "demo", "template", "starter", "tutorial",
)

# Day 12: how many per-package ``deep_scan`` calls to run concurrently
# inside ``deep_scan_workspace``. Four is chosen to stay well under
# Anthropic's default-tier 50 req/min rate limit while still cutting
# wall-clock time by ~4x for large monorepos like cal.com (15+ large
# packages × ~60s each = 15 min serial → ~4 min parallel).
_MAX_WORKERS = 4

# Sprint 1 size guard. Above this file count, the tool_use_scan final
# JSON consistently truncates at max_tokens (every path is echoed in
# the answer; ~25K output tokens for 2K paths blows the SDK's no-stream
# 21333 ceiling). 800 sits below the truncation knee on Sonnet 4.6
# while still covering 95% of real-world packages. Oversized packages
# route through the no-tools deep_scan path even when --tool-use is set.
_TOOL_USE_MAX_FILES = 800


def deep_scan_workspace(
    workspace_info,  # WorkspaceInfo from faultline.analyzer.workspace
    *,
    api_key: str | None = None,
    model: str | None = None,
    signatures: dict | None = None,
    is_library: bool = False,
    tracker: CostTracker | None = None,
    min_files_for_llm: int = _DEFAULT_PKG_LLM_FLOOR,
    commit_context: str | None = None,
    max_workers: int = _MAX_WORKERS,
    use_tools: bool = False,
    repo_root=None,  # pathlib.Path; required when use_tools=True
) -> DeepScanResult | None:
    """Run ``deep_scan`` once per workspace package and merge the results.

    For each package in ``workspace_info.packages``:

      - Test packages (``tests``, ``e2e``, ``__tests__``…) are skipped
        entirely. Tests are never a feature.
      - Example/demo packages (``examples/foo``, ``demo-app``…) are pooled
        into a single synthetic ``examples`` feature. None of them get
        their own LLM call.
      - Packages smaller than ``min_files_for_llm`` become one feature
        named after the package, with no LLM call. The package name is
        already a cohesive label (it came from package.json).
      - Larger packages get their own ``deep_scan(package_mode=True)``
        invocation. The returned features are re-prefixed with the
        package name (``{pkg.name}/{sub-feature}``) when more than one
        sub-feature comes back; a single sub-feature collapses to the
        bare package name to avoid noise like ``auth/auth``.

    The shared ``CostTracker`` is threaded through every per-package
    call so the total cost reported at the end of the run is the sum
    across all packages. ``BudgetExceeded`` from any one call propagates
    out immediately so the caller can stop the scan before firing more
    requests.

    ``workspace_info.root_files`` (anything that wasn't claimed by a
    package — typically CI config, root package.json, README) becomes
    the ``shared-infra`` feature.

    Returns the merged feature → file mapping, or ``None`` if the
    workspace had no usable packages at all.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from faultline.analyzer.features import detect_candidates
    from faultline.analyzer.validation import (
        canonical_bucket_name,
        is_test_feature_name,
    )
    from faultline.llm.cost import BudgetExceeded

    if workspace_info is None or not workspace_info.packages:
        return None

    raw_mapping: dict[str, list[str]] = {}
    merged_flows: dict[str, list[str]] = {}
    merged_descriptions: dict[str, str] = {}
    merged_flow_descriptions: dict[str, dict[str, str]] = {}
    examples_files: list[str] = []

    # Largest packages first so a budget abort kills the expensive ones
    # before we waste calls on small packages.
    packages = sorted(workspace_info.packages, key=lambda p: -len(p.files))

    # ── Phase 1: serial pre-processing ─────────────────────────────────
    # Filter tests/examples, short-circuit small packages, build
    # per-package candidates. This is all pure Python (no I/O) so
    # parallelism wouldn't help. Output is a list of "llm_jobs" ready
    # for the thread pool.
    llm_jobs: list[tuple] = []  # (pkg, pkg_prefix, pkg_files_rel, pkg_sigs, pkg_candidates)

    for pkg in packages:
        if not pkg.files:
            continue

        name_lower = pkg.name.lower()

        # Test packages: skipped entirely (tests are never a feature).
        if is_test_feature_name(pkg.name):
            logger.info("workspace: skip test package %s (%d files)", pkg.name, len(pkg.files))
            continue

        # Example / demo / tutorial packages: pooled into one bucket.
        if any(name_lower.startswith(prefix) for prefix in _EXAMPLE_PKG_PREFIXES):
            examples_files.extend(pkg.files)
            logger.info("workspace: pool %s (%d files) → examples", pkg.name, len(pkg.files))
            continue

        # Small packages: 1 feature, no LLM call. The package name is
        # already a good label so spending tokens here is pure waste.
        if len(pkg.files) < min_files_for_llm:
            raw_mapping[pkg.name] = list(pkg.files)
            logger.info(
                "workspace: %s (%d files) → 1 feature (under LLM floor)",
                pkg.name,
                len(pkg.files),
            )
            continue

        # Large package: prepare the inputs for a per-package LLM call.
        # Strip the package path prefix so files passed to deep_scan are
        # relative to the package root. This keeps the candidate detector
        # focused and the LLM prompt readable.
        pkg_prefix = pkg.path.rstrip("/") + "/" if pkg.path else ""
        if pkg_prefix:
            pkg_files_rel = [
                f[len(pkg_prefix):] for f in pkg.files if f.startswith(pkg_prefix)
            ]
        else:
            pkg_files_rel = list(pkg.files)

        if not pkg_files_rel:
            raw_mapping[pkg.name] = list(pkg.files)
            continue

        pkg_sigs: dict | None = None
        if signatures:
            pkg_sigs = {
                f[len(pkg_prefix):]: sig
                for f, sig in signatures.items()
                if pkg_prefix and f.startswith(pkg_prefix)
            }
            if not pkg_sigs:
                pkg_sigs = None

        try:
            pkg_candidates = detect_candidates(pkg_files_rel)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "workspace: detect_candidates failed for %s (%s) — fallback 1 feature",
                pkg.name, exc,
            )
            raw_mapping[pkg.name] = list(pkg.files)
            continue

        llm_jobs.append((pkg, pkg_prefix, pkg_files_rel, pkg_sigs, pkg_candidates))

    # ── Phase 2: parallel LLM calls ────────────────────────────────────
    # Submit each prepared job to a ThreadPoolExecutor. The LLM round-trip
    # is ~60s per package; running ``max_workers`` in parallel cuts
    # wall-clock by that factor. Results are collected and merged in
    # the main thread after all futures finish so the merge logic stays
    # single-threaded and deterministic.
    #
    # Error handling: ``BudgetExceeded`` from any future is propagated
    # out of the whole function immediately so the caller can stop the
    # scan. Any other exception is logged and the package falls back to
    # a single-feature entry.

    def _run_pkg(
        pkg, pkg_prefix, pkg_files_rel, pkg_sigs, pkg_candidates,
    ) -> tuple:
        """Thread-pool worker: returns (pkg, pkg_prefix, sub_result_or_None)."""
        try:
            # Size guard: tool_use_scan asks the model to echo every
            # file path in the final JSON. Above ~800 files the output
            # truncates at max_tokens before the JSON closes, parse
            # fails, and the package falls back to a single feature —
            # which is worse than the no-tools deep_scan that handles
            # large packages just fine. Sprint 3 (sub-decomposition)
            # is the real fix; until then route oversized packages
            # through the no-tools path even when --tool-use is set.
            tool_use_active = use_tools and len(pkg_files_rel) <= _TOOL_USE_MAX_FILES
            if use_tools and not tool_use_active:
                logger.info(
                    "workspace: %s (%d files) exceeds tool-use size limit (%d) — "
                    "falling back to no-tools deep_scan for this package",
                    pkg.name, len(pkg_files_rel), _TOOL_USE_MAX_FILES,
                )
            if tool_use_active:
                if repo_root is None:
                    raise ValueError(
                        "deep_scan_workspace: use_tools=True requires repo_root"
                    )
                from faultline.llm.tool_use_scan import tool_use_scan_package
                sub_result = tool_use_scan_package(
                    package_name=pkg.name,
                    files=pkg_files_rel,
                    repo_root=repo_root,
                    pkg_prefix=pkg_prefix,
                    api_key=api_key,
                    model=model,
                    tracker=tracker,
                )
            else:
                sub_result = deep_scan(
                    pkg_files_rel,
                    pkg_candidates,
                    api_key=api_key,
                    signatures=pkg_sigs,
                    is_library=is_library,
                    model=model,
                    tracker=tracker,
                    package_mode=True,
                    package_name=pkg.name,
                    commit_context=commit_context,
                )
        except BudgetExceeded:
            raise
        except Exception as exc:
            logger.warning(
                "workspace: deep_scan raised %s for %s — fallback 1 feature",
                type(exc).__name__, pkg.name,
            )
            return (pkg, pkg_prefix, None)
        return (pkg, pkg_prefix, sub_result)

    effective_workers = max(1, min(max_workers, len(llm_jobs) or 1))
    sub_results: list[tuple] = []

    if llm_jobs:
        logger.info(
            "workspace: %d packages queued for parallel LLM (max_workers=%d)",
            len(llm_jobs), effective_workers,
        )
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_run_pkg, *job): job[0] for job in llm_jobs}
            try:
                for fut in as_completed(futures):
                    sub_results.append(fut.result())
            except BudgetExceeded:
                # Cancel any unstarted futures and propagate. Running
                # futures can't be interrupted mid-LLM-call but the
                # budget abort stops everything queued after them.
                for f in futures:
                    f.cancel()
                raise

    # ── Phase 3: deterministic serial merge ───────────────────────────
    # Process results in the same largest-first order as the jobs were
    # submitted so the final feature ordering is stable across runs.
    sub_results.sort(
        key=lambda t: (-len(t[0].files), t[0].name),
    )

    for pkg, pkg_prefix, sub_result in sub_results:
        if not sub_result:
            logger.warning(
                "workspace: deep_scan returned no features for %s — fallback 1 feature",
                pkg.name,
            )
            raw_mapping[pkg.name] = list(pkg.files)
            continue

        # ``sub_result`` is a DeepScanResult after D10. Use ``.features``
        # for re-prefixing and pull flows/descriptions for the merged
        # workspace-level result.
        sub_mapping = sub_result.features

        # Per-package coverage enforcement. The LLM commonly:
        #   (a) returns sub-features that cover only a subset of the
        #       package's files (typical: 70-80% on a 200-file package);
        #   (b) labels some sub-features with infra-ish aliases like
        #       "lib", "utils", "shared", which canonicalize to the
        #       global shared-infra bucket and bleed package code out.
        #
        # Both leak real business code into a generic infra label and
        # destroy the user's mental model: code in apps/web/onboarding/
        # belongs to the web package, not to repo-level infra. Now that
        # the top-level bucketizer materializes shared-infra from real
        # repo configs, per-package shared-infra has no reason to exist.
        #
        # Fix: pop sub-features whose name canonicalizes to shared-infra,
        # add their files to the leftover set, then attach the combined
        # leftover to the largest remaining ("primary") sub-feature.
        infra_sub_files: list[str] = []
        for sub_name in list(sub_mapping.keys()):
            if canonical_bucket_name(sub_name) == "shared-infra":
                infra_sub_files.extend(sub_mapping.pop(sub_name))

        pkg_rel_paths = {
            f[len(pkg_prefix):] if pkg_prefix and f.startswith(pkg_prefix) else f
            for f in pkg.files
        }
        attributed_rel = set()
        for files in sub_mapping.values():
            attributed_rel.update(files)
        leftover_rel = sorted(
            (pkg_rel_paths - attributed_rel) | set(infra_sub_files)
        )
        if leftover_rel:
            primary = max(
                sub_mapping.items(),
                key=lambda kv: len(kv[1]),
                default=None,
            )
            if primary is not None:
                primary_name, primary_files = primary
                sub_mapping[primary_name] = list(primary_files) + leftover_rel
                logger.warning(
                    "workspace: %s — %d files unattributed (or labelled "
                    "infra-ish) by per-package LLM, attached to primary "
                    "sub-feature '%s' (%d → %d)",
                    pkg.name,
                    len(leftover_rel),
                    primary_name,
                    len(primary_files),
                    len(primary_files) + len(leftover_rel),
                )
            else:
                sub_mapping[pkg.name] = list(leftover_rel)
                logger.warning(
                    "workspace: %s — %d files unattributed and no "
                    "sub-feature to absorb them; kept under package name",
                    pkg.name,
                    len(leftover_rel),
                )

        # Re-prefix file paths back to repo-relative form, building a
        # mapping from sub-feature name (in sub_result) → final feature
        # key (in raw_mapping) so we can re-key flows/descriptions to
        # match what cli.py will read.
        sub_to_final: dict[str, str] = {}

        if len(sub_mapping) == 1:
            # Single sub-feature → bare package name (avoids "auth/auth").
            only_name, only_files = next(iter(sub_mapping.items()))
            raw_mapping[pkg.name] = [pkg_prefix + f for f in only_files]
            sub_to_final[only_name] = pkg.name
        else:
            for sub_name, sub_files in sub_mapping.items():
                # Canonical infra names from the per-package call go into
                # the global shared-infra bucket, not pkg.name/shared-infra.
                if canonical_bucket_name(sub_name) == "shared-infra":
                    raw_mapping.setdefault("shared-infra", []).extend(
                        pkg_prefix + f for f in sub_files
                    )
                    sub_to_final[sub_name] = "shared-infra"
                    continue
                final_key = f"{pkg.name}/{sub_name}"
                raw_mapping[final_key] = [pkg_prefix + f for f in sub_files]
                sub_to_final[sub_name] = final_key

        # Merge flows and descriptions under the final feature keys.
        for sub_name, final_key in sub_to_final.items():
            if sub_name in sub_result.flows and final_key not in merged_flows:
                merged_flows[final_key] = sub_result.flows[sub_name]
            if sub_name in sub_result.descriptions and final_key not in merged_descriptions:
                merged_descriptions[final_key] = sub_result.descriptions[sub_name]
            if sub_name in sub_result.flow_descriptions and final_key not in merged_flow_descriptions:
                merged_flow_descriptions[final_key] = sub_result.flow_descriptions[sub_name]
        logger.info("workspace: %s → %d feature(s)", pkg.name, len(sub_mapping))

    if examples_files:
        raw_mapping["examples"] = examples_files

    if workspace_info.root_files:
        raw_mapping.setdefault("shared-infra", []).extend(workspace_info.root_files)

    if not raw_mapping:
        return None

    # Deterministic ordering, matches single-call deep_scan (D11).
    sorted_features = dict(
        sorted(raw_mapping.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    )

    return DeepScanResult(
        features=sorted_features,
        flows=merged_flows,
        descriptions=merged_descriptions,
        flow_descriptions=merged_flow_descriptions,
        cost_summary=tracker.summary() if tracker is not None else None,
    )


def _normalize_response(data: dict, model: str | None = None) -> dict:
    """Normalizes LLM response to expected SonnetOpsResponse format.

    Handles variations in how Sonnet returns features (dict vs list)
    while preserving merge/rename/remove operations.

    ``model`` controls model-specific normalizations. Opus 4.7 returns
    ``merge`` as ``[{target, source}]`` dicts rather than the schema's
    ``list[list[str]]``; we flatten that shape when the caller tells us
    the response came from an Opus model. Sonnet/Haiku stay untouched.
    """
    result = dict(data)
    is_opus = bool(model and model.startswith("claude-opus"))

    features = result.get("features", [])

    # If features is a dict keyed by name → convert to list
    if isinstance(features, dict):
        normalized = []
        for name, value in features.items():
            if isinstance(value, dict):
                feat = {"name": name, **value}
                # Normalize flows if dict
                if "flows" in feat and isinstance(feat["flows"], dict):
                    feat["flows"] = [
                        {"name": fn, **(fv if isinstance(fv, dict) else {})}
                        for fn, fv in feat["flows"].items()
                    ]
                normalized.append(feat)
        result["features"] = normalized

    # Normalize rename items — Sonnet may return:
    # [{"from": "x", "to": "y"}] or [["old", "new"]] or [{"from_name": "x", "to": "y"}]
    renames = result.get("rename", [])
    normalized_renames = []
    for r in renames:
        if isinstance(r, dict):
            if "from" in r and "from_name" not in r:
                r["from_name"] = r.pop("from")
            normalized_renames.append(r)
        elif isinstance(r, list) and len(r) == 2:
            normalized_renames.append({"from_name": r[0], "to": r[1]})
    result["rename"] = normalized_renames

    # Normalize split items (same "from" → "from_name" issue)
    for s in result.get("split", []):
        if isinstance(s, dict) and "from" in s and "from_name" not in s:
            s["from_name"] = s.pop("from")

    # Opus-only: flatten dict-shaped merge ops into the schema's
    # ``list[list[str]]``. Opus 4.7 emits
    # ``[{"target": "insights", "source": ["insight_subscriptions"]}]``
    # which pydantic rejects; Sonnet/Haiku honour the list-of-lists
    # schema directly. Target goes first so downstream code that looks
    # at group ordering sees Opus's intended merge target up front,
    # even though the merge applier picks the largest feature as target
    # anyway.
    if is_opus:
        merges = result.get("merge", [])
        normalized_merges: list[list[str]] = []
        for m in merges:
            if isinstance(m, list):
                normalized_merges.append([str(x) for x in m if isinstance(x, (str, int))])
            elif isinstance(m, dict):
                target = m.get("target")
                sources = m.get("source") or m.get("sources") or []
                if isinstance(sources, str):
                    sources = [sources]
                group = []
                if isinstance(target, str):
                    group.append(target)
                if isinstance(sources, list):
                    group.extend(s for s in sources if isinstance(s, str))
                if len(group) >= 2:
                    normalized_merges.append(group)
        result["merge"] = normalized_merges

    return result


def _parse_json_response(text: str) -> dict | None:
    """Extracts JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


# ── Flow extraction ────────────────────────────────────────────────────────

_last_scan_result: SonnetOpsResponse | None = None


def get_deep_scan_flows() -> dict[str, list[str]]:
    """
    Returns flow names per feature from the last deep scan.

    Uses fuzzy matching to map Sonnet feature names to candidate names,
    since they may differ (Sonnet says "documents", candidate = "document").

    Returns:
        dict[feature_name → list[flow_name]]
    """
    if not _last_scan_result:
        return {}

    result: dict[str, list[str]] = {}
    for feat in _last_scan_result.features:
        if feat.flows:
            result[feat.name] = [fl.name for fl in feat.flows]

    return result


def match_flows_to_features(
    flow_data: dict[str, list[str]],
    feature_names: list[str],
) -> dict[str, list[str]]:
    """Maps Sonnet flow data to actual feature names via fuzzy matching.

    Sonnet may use "documents" but the feature is named "document".
    """
    result: dict[str, list[str]] = {}
    used_sonnet_names: set[str] = set()

    for feat_name in feature_names:
        # Exact match
        if feat_name in flow_data:
            result[feat_name] = flow_data[feat_name]
            used_sonnet_names.add(feat_name)
            continue

        # Fuzzy: singular/plural, substring
        for sonnet_name, flows in flow_data.items():
            if sonnet_name in used_sonnet_names:
                continue
            if (feat_name in sonnet_name or sonnet_name in feat_name
                    or feat_name.rstrip("s") == sonnet_name.rstrip("s")):
                result[feat_name] = flows
                used_sonnet_names.add(sonnet_name)
                break

    return result


def get_deep_scan_descriptions() -> dict[str, str]:
    """Returns feature descriptions from the last deep scan."""
    if not _last_scan_result:
        return {}
    return {f.name: f.description for f in _last_scan_result.features if f.description}


def get_deep_scan_flow_descriptions() -> dict[str, dict[str, str]]:
    """Returns flow descriptions from the last deep scan."""
    if not _last_scan_result:
        return {}
    result: dict[str, dict[str, str]] = {}
    for feat in _last_scan_result.features:
        if feat.flows:
            result[feat.name] = {fl.name: fl.description for fl in feat.flows}
    return result
