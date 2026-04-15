"""Thread-safe MCP event buffer with periodic flush.

The MCP server enqueues one event per tool call. A background thread
flushes the queue every 60 seconds (or when it hits 200 events) by
posting a batch to ``/api/cloud/usage``.

Stays a no-op when ``FAULTLINE_API_KEY`` is unset — never spawns the
background thread, never blocks tool calls.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)

_FLUSH_INTERVAL_SECONDS = 60
_MAX_BATCH_SIZE = 200
_MAX_QUEUE_SIZE = 5000  # protect memory if API is unreachable for hours


class _EventBuffer:
    def __init__(self) -> None:
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=_MAX_QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started_lock = threading.Lock()

    def enqueue(self, event: dict[str, Any]) -> None:
        if not os.environ.get("FAULTLINE_API_KEY"):
            return
        try:
            self._queue.put_nowait(event)
        except Exception:
            # Queue full — drop the event silently to avoid blocking the MCP call
            return
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        with self._started_lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._flush_loop, name="faultline-cloud-flush", daemon=True,
            )
            self._thread.start()
            atexit.register(self._flush_remaining)

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(_FLUSH_INTERVAL_SECONDS)
            self._flush_once()

    def _flush_once(self) -> int:
        batch: list[dict[str, Any]] = []
        while len(batch) < _MAX_BATCH_SIZE:
            try:
                batch.append(self._queue.get_nowait())
            except Empty:
                break
        if not batch:
            return 0
        try:
            from faultline.cloud.sync import send_mcp_events_batch
            return send_mcp_events_batch(batch)
        except Exception as exc:
            logger.warning("event flush failed: %s", exc)
            return 0

    def _flush_remaining(self) -> None:
        # Called on interpreter shutdown — best-effort drain.
        self._stop.set()
        self._flush_once()


_buffer = _EventBuffer()


def record_mcp_event(
    *,
    tool_name: str,
    query_arg: str | None = None,
    files_returned: int | None = None,
    tokens_saved: int = 0,
) -> None:
    """Record one MCP tool call. Safe to call from anywhere — never blocks."""
    _buffer.enqueue({
        "tool_name": tool_name,
        "query_arg": query_arg,
        "files_returned": files_returned,
        "tokens_saved": tokens_saved,
        "occurred_at": datetime.now(tz=timezone.utc).isoformat(),
    })
