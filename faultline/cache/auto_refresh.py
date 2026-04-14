"""MCP-triggered background auto-refresh.

When an AI agent queries the MCP server, this module can opportunistically
check whether the feature map is stale and, if so, trigger a background
refresh thread. The current query still returns immediately with a
warning — the next query sees the refreshed data.

Opt-in via ``FAULTLINE_AUTO_REFRESH=1`` environment variable. Default
is off so nothing happens without the user's explicit decision.

Safety:
  - File-based lock with PID + timestamp (survives process crashes)
  - Throttle: minimum 5 minutes between refresh attempts per repo
  - Background thread catches all exceptions and logs them
  - No LLM calls by default (metric refresh only)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_THROTTLE_SECONDS = 300  # 5 minutes
_LOCK_STALE_SECONDS = 600        # auto-release stale locks after 10 min
_LOCK_DIR = Path.home() / ".faultline"


class RefreshLock:
    """File-based lock preventing concurrent refreshes across processes.

    Lock file format: JSON with pid and timestamp. Stale locks (owned by
    dead processes or older than _LOCK_STALE_SECONDS) are auto-released.
    """

    def __init__(self, repo_slug: str) -> None:
        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        self.path = _LOCK_DIR / f".refresh-{repo_slug}.lock"

    def acquire(self) -> bool:
        """Returns True if lock acquired, False if another process holds it."""
        if self.path.exists():
            if not self._is_stale():
                return False
            self._release_silently()

        try:
            self.path.write_text(json.dumps({
                "pid": os.getpid(),
                "timestamp": time.time(),
            }))
        except OSError as exc:
            logger.warning("could not acquire refresh lock: %s", exc)
            return False
        return True

    def release(self) -> None:
        try:
            if self.path.exists():
                content = json.loads(self.path.read_text())
                if content.get("pid") == os.getpid():
                    self.path.unlink()
        except (OSError, json.JSONDecodeError):
            self._release_silently()

    def _release_silently(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass

    def _is_stale(self) -> bool:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return True

        ts = data.get("timestamp", 0)
        if time.time() - ts > _LOCK_STALE_SECONDS:
            return True

        pid = data.get("pid")
        if pid and not _pid_alive(pid):
            return True

        return False


def _pid_alive(pid: int) -> bool:
    """Check whether a PID is still running (Unix/macOS)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────
# Throttle state — tracks last refresh attempt per repo slug
# ──────────────────────────────────────────────────────────────────────

_throttle_state: dict[str, float] = {}
_throttle_lock = threading.Lock()


def should_refresh(repo_slug: str, throttle_seconds: int | None = None) -> bool:
    """True if we haven't attempted a refresh for this repo in the last
    throttle_seconds. Does NOT consult git — caller is responsible for
    checking freshness first.
    """
    throttle = throttle_seconds if throttle_seconds is not None else _DEFAULT_THROTTLE_SECONDS
    with _throttle_lock:
        last = _throttle_state.get(repo_slug, 0.0)
        if time.time() - last < throttle:
            return False
        _throttle_state[repo_slug] = time.time()
    return True


def reset_throttle(repo_slug: str) -> None:
    """Force the next attempt to not be throttled. Test helper."""
    with _throttle_lock:
        _throttle_state.pop(repo_slug, None)


# ──────────────────────────────────────────────────────────────────────
# Background refresh trigger
# ──────────────────────────────────────────────────────────────────────


def trigger_background_refresh(
    feature_map_path: Path,
    repo_path: str,
    *,
    throttle_seconds: int | None = None,
) -> bool:
    """Kick off a background refresh if not throttled and not already running.

    Args:
        feature_map_path: The JSON file to refresh.
        repo_path: Path to the git repository.
        throttle_seconds: Override the default 5-minute throttle.

    Returns:
        True if a refresh thread was started, False if throttled / locked.
    """
    repo_slug = _slug(repo_path)

    if not should_refresh(repo_slug, throttle_seconds):
        return False

    lock = RefreshLock(repo_slug)
    if not lock.acquire():
        return False

    thread = threading.Thread(
        target=_background_refresh_task,
        args=(feature_map_path, repo_path, lock),
        name=f"faultline-refresh-{repo_slug}",
        daemon=True,
    )
    thread.start()
    return True


def _background_refresh_task(
    feature_map_path: Path,
    repo_path: str,
    lock: RefreshLock,
) -> None:
    """Runs in a background thread. Never blocks the MCP server."""
    try:
        from faultline.cache.refresh import refresh_feature_map
        from faultline.models.types import FeatureMap
        from faultline.output.writer import write_feature_map

        fm = FeatureMap.model_validate_json(feature_map_path.read_text())
        result = refresh_feature_map(fm, repo_path)

        if not result.freshness_before.is_stale:
            logger.info("auto-refresh: no changes for %s", feature_map_path.name)
            return

        # Write back to the same file to avoid file proliferation.
        write_feature_map(result.updated_map, str(feature_map_path))
        logger.info(
            "auto-refresh: updated %s (%d commits behind → fresh)",
            feature_map_path.name, result.freshness_before.commits_behind,
        )
    except Exception as exc:
        logger.warning("auto-refresh failed: %s", exc)
    finally:
        lock.release()


def _slug(repo_path: str) -> str:
    """Stable identifier for a repo path (used for lock + throttle keys)."""
    import hashlib
    return hashlib.sha256(repo_path.encode("utf-8")).hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
# Public API for MCP server integration
# ──────────────────────────────────────────────────────────────────────


def maybe_trigger_refresh(feature_map_path: Path, fm_data: dict) -> bool:
    """Called from the MCP server's load-map path.

    Returns True if a refresh was triggered. The caller should add a
    warning to the response so the AI agent knows the current data
    may be stale but is being updated.
    """
    if os.environ.get("FAULTLINE_AUTO_REFRESH") not in ("1", "true", "yes"):
        return False

    repo_path = fm_data.get("repo_path", "")
    if not repo_path:
        return False

    # Cheap freshness check — compare stored SHA to current HEAD via subprocess
    scanned_sha = fm_data.get("last_scanned_sha", "")
    if not scanned_sha:
        return False

    import subprocess
    try:
        current = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, text=True, timeout=2,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False

    if current == scanned_sha:
        return False  # already fresh

    # Stale → trigger background refresh (throttled)
    return trigger_background_refresh(feature_map_path, repo_path)
