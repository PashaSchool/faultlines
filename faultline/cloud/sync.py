"""HTTP client for the Faultlines cloud API.

All functions are silent no-ops when ``FAULTLINE_API_KEY`` is unset —
that's the documented opt-in surface. Errors are logged but never raised
to callers; cloud sync should never break a local CLI run or an MCP tool
call.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE = "https://faultlines.dev/api/cloud"


def _api_base() -> str:
    return os.environ.get("FAULTLINE_API_BASE", _DEFAULT_API_BASE).rstrip("/")


def _api_key() -> str | None:
    key = os.environ.get("FAULTLINE_API_KEY", "").strip()
    return key or None


def push_feature_map(feature_map: Any) -> dict[str, Any] | None:
    """Upload a feature map to the SaaS dashboard.

    Args:
        feature_map: pydantic FeatureMap or a dict — anything with
            ``model_dump(mode="json")`` or already a dict.

    Returns:
        Server response dict on success, None when no API key is set
        or the call fails.
    """
    key = _api_key()
    if not key:
        return None

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — cannot push to cloud")
        return None

    if hasattr(feature_map, "model_dump"):
        payload = feature_map.model_dump(mode="json")
    elif isinstance(feature_map, dict):
        payload = feature_map
    else:
        logger.warning("push_feature_map: unsupported feature_map type")
        return None

    try:
        response = httpx.post(
            f"{_api_base()}/scans",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            content=json.dumps(payload, default=str),
            timeout=60.0,
        )
        if response.status_code >= 400:
            logger.warning(
                "cloud push failed: %d %s", response.status_code, response.text[:200],
            )
            return None
        return response.json()
    except Exception as exc:
        logger.warning("cloud push error: %s", exc)
        return None


def send_mcp_events_batch(events: list[dict[str, Any]]) -> int:
    """Send a batch of MCP tool-call events to the dashboard.

    Returns the number of events the server accepted, or 0 on no-op /
    failure. Caller is responsible for batching — typical batch is 50–500
    events flushed every minute.
    """
    if not events:
        return 0

    key = _api_key()
    if not key:
        return 0

    try:
        import httpx
    except ImportError:
        return 0

    try:
        response = httpx.post(
            f"{_api_base()}/usage",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={"events": events},
            timeout=15.0,
        )
        if response.status_code >= 400:
            logger.warning(
                "cloud usage push failed: %d %s",
                response.status_code, response.text[:200],
            )
            return 0
        data = response.json()
        return int(data.get("accepted", 0))
    except Exception as exc:
        logger.warning("cloud usage push error: %s", exc)
        return 0
