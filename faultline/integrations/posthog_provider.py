"""PostHog analytics provider for FeatureMap."""

from __future__ import annotations

import httpx

from faultline.integrations.base import (
    AnalyticsProvider,
    ErrorEntry,
    ErrorMetrics,
    PageMetrics,
)


class PostHogProvider:
    """Fetches traffic and error metrics from PostHog API.

    Supports both PostHog Cloud and self-hosted instances.

    Usage:
        provider = PostHogProvider(
            api_key="phx_...",
            project_id="12345",
            host="https://app.posthog.com",  # or self-hosted URL
        )
        if await provider.validate_connection():
            traffic = await provider.get_page_traffic(days=30)
    """

    name = "posthog"

    def __init__(
        self,
        api_key: str,
        project_id: str,
        host: str = "https://app.posthog.com",
    ) -> None:
        self._api_key = api_key
        self._project_id = project_id
        self._host = host.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=f"{self._host}/api/projects/{self._project_id}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )

    async def validate_connection(self) -> bool:
        """Verify API key and project access."""
        try:
            resp = await self._client.get("/")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def get_page_traffic(self, days: int = 30) -> list[PageMetrics]:
        """Query PostHog for $pageview events grouped by current_url."""
        query = {
            "query": {
                "kind": "HogQLQuery",
                "query": (
                    "SELECT "
                    "  properties.$current_url AS url, "
                    "  count() AS views, "
                    "  count(DISTINCT person_id) AS visitors, "
                    "  avg(properties.$session_duration) AS avg_duration "
                    "FROM events "
                    "WHERE event = '$pageview' "
                    f"  AND timestamp > now() - INTERVAL {days} DAY "
                    "GROUP BY url "
                    "ORDER BY views DESC "
                    "LIMIT 500"
                ),
            }
        }

        try:
            resp = await self._client.post("/query/", json=query)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return []

        results: list[PageMetrics] = []
        for row in data.get("results", []):
            url, views, visitors, avg_dur = row
            route = _extract_route(url or "")
            if not route:
                continue
            results.append(
                PageMetrics(
                    route=route,
                    pageviews=int(views or 0),
                    unique_visitors=int(visitors or 0),
                    avg_session_duration_sec=float(avg_dur or 0),
                )
            )

        return _dedupe_routes(results)

    async def get_error_counts(self, days: int = 30) -> list[ErrorMetrics]:
        """Query PostHog for $exception events grouped by current_url."""
        query = {
            "query": {
                "kind": "HogQLQuery",
                "query": (
                    "SELECT "
                    "  properties.$current_url AS url, "
                    "  count() AS errors, "
                    "  count(DISTINCT properties.$exception_message) AS unique_errs, "
                    "  groupArray(10)(properties.$exception_message) AS messages "
                    "FROM events "
                    "WHERE event = '$exception' "
                    f"  AND timestamp > now() - INTERVAL {days} DAY "
                    "GROUP BY url "
                    "ORDER BY errors DESC "
                    "LIMIT 200"
                ),
            }
        }

        try:
            resp = await self._client.post("/query/", json=query)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return []

        results: list[ErrorMetrics] = []
        for row in data.get("results", []):
            url, errors, unique_errs, messages = row
            route = _extract_route(url or "")
            if not route:
                continue

            top_errors = [
                ErrorEntry(title=msg, count=1)
                for msg in (messages or [])
                if msg
            ]

            results.append(
                ErrorMetrics(
                    route=route,
                    error_count=int(errors or 0),
                    unique_errors=int(unique_errs or 0),
                    top_errors=top_errors,
                )
            )

        return results

    async def close(self) -> None:
        await self._client.aclose()


def _extract_route(full_url: str) -> str:
    """Extract path from a full URL, stripping query params and hash."""
    from urllib.parse import urlparse

    if not full_url:
        return ""

    parsed = urlparse(full_url)
    path = parsed.path or "/"

    # Normalize trailing slash
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return path


def _dedupe_routes(metrics: list[PageMetrics]) -> list[PageMetrics]:
    """Merge metrics for the same route (different query params, etc.)."""
    merged: dict[str, PageMetrics] = {}

    for pm in metrics:
        if pm.route in merged:
            existing = merged[pm.route]
            merged[pm.route] = PageMetrics(
                route=pm.route,
                pageviews=existing.pageviews + pm.pageviews,
                unique_visitors=existing.unique_visitors + pm.unique_visitors,
                avg_session_duration_sec=(
                    existing.avg_session_duration_sec + pm.avg_session_duration_sec
                ) / 2,
            )
        else:
            merged[pm.route] = pm

    return sorted(merged.values(), key=lambda m: m.pageviews, reverse=True)
