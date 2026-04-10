"""Base protocol and models for analytics integrations."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class PageMetrics(BaseModel):
    """Aggregated traffic metrics for a single page/route."""

    route: str
    pageviews: int = 0
    unique_visitors: int = 0
    avg_session_duration_sec: float = 0.0
    bounce_rate: float | None = None


class ErrorMetrics(BaseModel):
    """Aggregated error metrics for a single route or component."""

    route: str
    error_count: int = 0
    unique_errors: int = 0
    top_errors: list[ErrorEntry] = []


class ErrorEntry(BaseModel):
    """Single error type with count and optional link."""

    title: str
    count: int
    url: str = ""  # link to Sentry issue / PostHog error page


class ImpactScore(BaseModel):
    """Computed impact combining health score with analytics data."""

    flow_name: str
    health_score: float
    pageviews: int
    error_count: int
    impact_level: str  # "critical" | "high" | "medium" | "low" | "healthy"
    score: float  # 0-100, lower = more urgent


# Fix forward reference
ErrorMetrics.model_rebuild()


@runtime_checkable
class AnalyticsProvider(Protocol):
    """Protocol that all analytics providers must implement.

    Each provider connects to an external analytics service and returns
    standardized metrics that FeatureMap can correlate with code features.
    """

    @property
    def name(self) -> str:
        """Provider identifier, e.g. 'posthog', 'sentry', 'ga4'."""
        ...

    async def validate_connection(self) -> bool:
        """Test that API credentials are valid. Returns True if OK."""
        ...

    async def get_page_traffic(
        self, days: int = 30
    ) -> list[PageMetrics]:
        """Return traffic metrics per page/route for the last N days."""
        ...

    async def get_error_counts(
        self, days: int = 30
    ) -> list[ErrorMetrics]:
        """Return error metrics per route for the last N days.

        Not all providers support this (e.g. GA4 doesn't).
        Return empty list if unsupported.
        """
        ...


def compute_impact_scores(
    flows: list[dict],
    traffic: list[PageMetrics],
    errors: list[ErrorMetrics],
) -> list[ImpactScore]:
    """Combine health scores with analytics to produce impact scores.

    Args:
        flows: list of dicts with 'name', 'health_score', 'paths' keys
        traffic: page traffic from analytics provider
        errors: error counts from analytics provider

    Returns:
        Impact scores sorted by urgency (most critical first).
    """
    traffic_by_route = {pm.route: pm for pm in traffic}
    errors_by_route = {em.route: em for em in errors}

    scores: list[ImpactScore] = []

    for flow in flows:
        flow_name = flow["name"]
        health = flow["health_score"]
        flow_paths = flow.get("paths", [])

        total_views = 0
        total_errors = 0

        for path in flow_paths:
            route = _path_to_route(path)
            if route in traffic_by_route:
                total_views += traffic_by_route[route].pageviews
            if route in errors_by_route:
                total_errors += errors_by_route[route].error_count

        score = _calculate_score(health, total_views, total_errors)
        level = _score_to_level(score)

        scores.append(
            ImpactScore(
                flow_name=flow_name,
                health_score=health,
                pageviews=total_views,
                error_count=total_errors,
                impact_level=level,
                score=score,
            )
        )

    return sorted(scores, key=lambda s: s.score)


def _path_to_route(file_path: str) -> str:
    """Convert a file path to a URL route for matching.

    Examples:
        src/app/dashboard/settings/page.tsx -> /dashboard/settings
        pages/api/webhooks/github.ts -> /api/webhooks/github
        src/routes/checkout/+page.svelte -> /checkout
    """
    route = file_path

    # Strip common prefixes
    for prefix in ("src/app/", "src/pages/", "app/", "pages/", "src/routes/"):
        if route.startswith(prefix):
            route = route[len(prefix):]
            break

    # Strip file extensions and index/page markers
    for suffix in (
        "/page.tsx", "/page.ts", "/page.jsx", "/page.js",
        "/+page.svelte", "/+page.ts",
        "/index.tsx", "/index.ts", "/index.jsx", "/index.js",
        ".tsx", ".ts", ".jsx", ".js", ".svelte",
    ):
        if route.endswith(suffix):
            route = route[: -len(suffix)]
            break

    # Ensure leading slash
    if not route.startswith("/"):
        route = "/" + route

    # Clean trailing slash
    if route != "/" and route.endswith("/"):
        route = route.rstrip("/")

    return route


def _calculate_score(health: float, pageviews: int, errors: int) -> float:
    """Lower score = more urgent to fix.

    Formula weights:
    - health_score: 40% (inverted — low health = low score)
    - traffic_factor: 35% (more traffic = lower score = more urgent)
    - error_factor: 25% (more errors = lower score = more urgent)
    """
    health_component = health  # 0-100, higher is better

    # Normalize traffic: log scale, cap at 100K views
    import math
    traffic_norm = min(math.log10(max(pageviews, 1)) / 5.0, 1.0) * 100
    traffic_component = 100 - traffic_norm  # invert: more traffic = lower

    # Normalize errors: log scale, cap at 10K errors
    error_norm = min(math.log10(max(errors, 1)) / 4.0, 1.0) * 100
    error_component = 100 - error_norm

    return (
        health_component * 0.40
        + traffic_component * 0.35
        + error_component * 0.25
    )


def _score_to_level(score: float) -> str:
    """Convert numeric score to human-readable impact level."""
    if score < 25:
        return "critical"
    if score < 45:
        return "high"
    if score < 65:
        return "medium"
    if score < 80:
        return "low"
    return "healthy"
