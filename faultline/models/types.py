from pydantic import BaseModel
from datetime import datetime


class TimelinePoint(BaseModel):
    date: str          # ISO week label "YYYY-Www"
    total_commits: int
    bug_fix_commits: int
    test_commits: int


class Commit(BaseModel):
    sha: str
    message: str
    author: str
    date: datetime
    files_changed: list[str]
    is_bug_fix: bool = False
    pr_number: int | None = None


class PullRequest(BaseModel):
    number: int
    url: str          # full GitHub PR URL, empty string if remote unknown
    title: str        # first line of the commit message
    author: str
    date: datetime


class FileBlame(BaseModel):
    path: str
    authors: list[str]
    last_modified: datetime
    total_commits: int


class Flow(BaseModel):
    name: str                  # "checkout-flow", "login-flow"
    display_name: str | None = None  # Title Case label for UI ("Checkout")
    description: str | None = None
    participants: list["FlowParticipant"] = []
    # Sprint 4: tool-augmented flow detection grounds every flow in
    # a real route handler / event subscription. These fields point
    # at the file (and optional line) where that flow's user journey
    # begins. Both are None for flows produced by the legacy Haiku
    # detector (which doesn't record entry points).
    entry_point_file: str | None = None
    entry_point_line: int | None = None
    paths: list[str]           # files belonging to this flow
    authors: list[str]
    total_commits: int
    bug_fixes: int
    bug_fix_ratio: float
    last_modified: datetime
    health_score: float        # 0-100, higher is better
    bug_fix_prs: list[PullRequest] = []
    test_file_count: int = 0   # number of test files associated with this flow
    test_files: list[str] = []  # actual test file paths (Sprint 2 Day 10)
    weekly_points: list[TimelinePoint] = []  # weekly activity timeline
    bus_factor: int = 1                      # authors with ≥20% of flow commits
    health_trend: float | None = None        # first_half_bug_ratio - second_half; positive = improving
    hotspot_files: list[str] = []            # source files with >40% bug_fix_ratio (≥3 commits)
    coverage_pct: float | None = None        # avg line coverage % across source files; None if unavailable
    symbol_attributions: list["SymbolAttribution"] = []  # symbols (functions/classes) that belong to this flow — populated when --symbols is enabled


class SymbolRange(BaseModel):
    name: str              # exported symbol name, e.g. "FEATURE_FLAGS"
    start_line: int        # 1-indexed, inclusive
    end_line: int          # 1-indexed, inclusive
    kind: str = "const"    # "const", "function", "class", "type", "enum", "reexport"


class FlowParticipant(BaseModel):
    """One file that participates in a flow's call-graph reach.

    Sprint 7 ``trace_flow_callgraph`` populates these from the
    symbol-import graph BFS + layer classifier. ``layer`` is one of
    ``ui`` / ``state`` / ``api-client`` / ``api-server`` /
    ``schema`` / ``support``.
    """

    path: str
    layer: str = "support"
    depth: int = 0
    side_effect_only: bool = False
    symbols: list[SymbolRange] = []
    role: str | None = None  # optional human-readable role hint


class SharedParticipant(BaseModel):
    """A file a feature USES but does not own (Sprint 8 / 9).

    When the aggregator classifier deletes a "shared-aggregator"
    feature (a multi-domain DTOs package, a shared-UI primitives lib),
    each of its files is redistributed as a ``SharedParticipant`` on
    every product feature that imports it. The same file can appear
    on N features — that's the point: a Button.tsx used by 10
    features should show up on all 10.

    Distinguished from ``Feature.paths`` (which stays 1:1 file→feature
    for owned source code, used by blame / commit-attribution).

    ``role``:
        - ``consumer``: feature imports the file from elsewhere
        - ``co-owner``: redistribution found multiple consumers and
          the file is essential to several
    ``line_weight`` (0.0–1.0) carries forward existing line-scoped
    attribution. Defaults to 1.0 when fine-grained data isn't
    available. ``origin_feature`` retains the deleted aggregator's
    name so the dashboard can show provenance ("from: Shared API
    Schemas") without losing it.
    """

    file_path: str
    role: str = "consumer"  # "consumer" | "co-owner"
    line_weight: float = 1.0
    origin_feature: str | None = None


class SymbolAttribution(BaseModel):
    file_path: str                          # the shared file
    symbols: list[str]                      # symbol names attributed to this feature
    line_ranges: list[tuple[int, int]]      # merged non-overlapping (start, end) spans
    attributed_lines: int                   # total lines across all ranges
    total_file_lines: int                   # total lines in the file
    roles: dict[str, str] = {}              # {symbol_name: role}; role in {entry,handler,validator,data-fetch,state,loading-state,error-state,side-effect,ui-component,helper,type}
    # Sprint 2 Day 10: multi-attribution badge data — names of OTHER
    # flows (within the same feature) that also reference at least one
    # of these symbols. UI surfaces "shared with N flows" so a reader
    # knows the attributed code participates in multiple journeys.
    # The bug-ratio / coverage credit is NOT split — every flow gets
    # full credit per user spec ("a"). The badge is purely a hint.
    shared_with_flows: list[str] = []


class Feature(BaseModel):
    name: str
    # Title Case display label derived from ``name`` (or set explicitly
    # by post-processing). Dashboards and reports show this; ``name``
    # stays a stable slug used for dedup / config lookups / IDs.
    display_name: str | None = None
    description: str | None = None  # LLM-generated semantic description
    paths: list[str]          # directories/files belonging to this feature
    authors: list[str]        # contributors
    total_commits: int
    bug_fixes: int            # number of bug fix commits
    bug_fix_ratio: float      # bug_fixes / total_commits
    last_modified: datetime
    health_score: float       # 0-100, higher is better
    flows: list[Flow] = []    # populated when --flows flag is used
    bug_fix_prs: list[PullRequest] = []
    coverage_pct: float | None = None  # avg line coverage % across source files; None if unavailable
    shared_attributions: list[SymbolAttribution] = []  # symbol-scoped data for shared files
    # Refactor Day 1: participants — every file (with line ranges and
    # role) imported transitively from any of this feature's source
    # files. Built by analyzer.feature_participants.build_feature_participants
    # via SymbolGraph BFS. Replaces ``shared_attributions`` as the
    # primary attachment surface for line-scoped scoring; the older
    # field stays populated for back-compat callers but the
    # cross-feature gate ("file in 2+ features") no longer disables
    # symbol-scoped health and coverage.
    participants: list["FlowParticipant"] = []
    symbol_health_score: float | None = None           # health score weighted by symbol line ranges
    # Sprint 8/9: files this feature CONSUMES from a deleted aggregator
    # (DTO packages, shared-UI primitives, schema crates). Same file
    # can appear on multiple features. ``paths`` stays the 1:1
    # owned-file list for blame / commit-attribution; this list is
    # the additive N:M overlay.
    shared_participants: list[SharedParticipant] = []


class FeatureMap(BaseModel):
    repo_path: str
    remote_url: str = ""      # GitHub base URL, e.g. https://github.com/org/repo
    analyzed_at: datetime
    total_commits: int
    date_range_days: int
    features: list[Feature]
    last_scanned_sha: str = ""               # git HEAD at scan time — used for incremental refresh
    file_hashes: dict[str, str] = {}         # {rel_path: sha256_of_content} — skip re-parse when file unchanged
    symbol_hashes: dict[str, dict[str, str]] = {}  # {rel_path: {symbol_name: sha256_of_body}} — per-symbol cache for incremental LLM skip

    def sorted_by_risk(self) -> list[Feature]:
        """Returns features sorted from highest to lowest risk."""
        return sorted(self.features, key=lambda f: f.bug_fix_ratio, reverse=True)
