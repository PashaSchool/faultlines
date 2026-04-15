"""Tests for faultline/cloud/."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from faultline.cloud import event_buffer
from faultline.cloud.sync import push_feature_map, send_mcp_events_batch


@pytest.fixture(autouse=True)
def _reset_buffer(monkeypatch):
    monkeypatch.delenv("FAULTLINE_API_KEY", raising=False)
    monkeypatch.delenv("FAULTLINE_API_BASE", raising=False)
    # Drain any leftover queue between tests
    while not event_buffer._buffer._queue.empty():
        event_buffer._buffer._queue.get_nowait()
    yield


class TestPushFeatureMap:
    def test_no_op_without_api_key(self) -> None:
        result = push_feature_map({"repo_path": "/x", "features": []})
        assert result is None

    def test_pushes_dict_when_api_key_set(self, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_API_KEY", "fl_test")
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True, "scan_id": "abc"}
            mock_post.return_value = mock_response
            result = push_feature_map({"repo_path": "/x", "features": []})
        assert result == {"ok": True, "scan_id": "abc"}
        call = mock_post.call_args
        assert call.kwargs["headers"]["Authorization"] == "Bearer fl_test"

    def test_returns_none_on_4xx(self, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_API_KEY", "fl_test")
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Invalid API key"
            mock_post.return_value = mock_response
            result = push_feature_map({"repo_path": "/x"})
        assert result is None

    def test_uses_custom_api_base(self, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_API_KEY", "fl_test")
        monkeypatch.setenv("FAULTLINE_API_BASE", "http://localhost:3000/api/cloud")
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True}
            mock_post.return_value = mock_response
            push_feature_map({"repo_path": "/x"})
        url = mock_post.call_args.args[0]
        assert url.startswith("http://localhost:3000/api/cloud/scans")


class TestSendMcpEventsBatch:
    def test_empty_batch_no_op(self) -> None:
        assert send_mcp_events_batch([]) == 0

    def test_no_op_without_api_key(self) -> None:
        assert send_mcp_events_batch([{"tool_name": "x", "occurred_at": "2026-01-01"}]) == 0

    def test_sends_batch_and_returns_accepted_count(self, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_API_KEY", "fl_test")
        with patch("httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"accepted": 3}
            mock_post.return_value = mock_response
            count = send_mcp_events_batch([
                {"tool_name": "find_feature", "occurred_at": "2026-01-01"},
                {"tool_name": "get_hotspots", "occurred_at": "2026-01-01"},
                {"tool_name": "list_features", "occurred_at": "2026-01-01"},
            ])
        assert count == 3
        body = mock_post.call_args.kwargs["json"]
        assert "events" in body and len(body["events"]) == 3


class TestEventBuffer:
    def test_enqueue_no_op_without_key(self) -> None:
        event_buffer.record_mcp_event(tool_name="x", tokens_saved=10)
        assert event_buffer._buffer._queue.qsize() == 0

    def test_enqueue_when_api_key_set(self, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_API_KEY", "fl_test")
        # Stop the background thread so we don't actually try to flush
        with patch.object(event_buffer._buffer, "_ensure_thread"):
            event_buffer.record_mcp_event(
                tool_name="find_feature",
                query_arg="auth",
                files_returned=5,
                tokens_saved=27000,
            )
        assert event_buffer._buffer._queue.qsize() == 1
        evt = event_buffer._buffer._queue.get_nowait()
        assert evt["tool_name"] == "find_feature"
        assert evt["query_arg"] == "auth"
        assert evt["files_returned"] == 5
        assert evt["tokens_saved"] == 27000
        assert "occurred_at" in evt

    def test_flush_once_sends_to_cloud(self, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_API_KEY", "fl_test")
        # Pre-populate the queue
        for i in range(5):
            event_buffer._buffer._queue.put_nowait({
                "tool_name": f"tool_{i}",
                "occurred_at": "2026-01-01",
                "tokens_saved": 100,
            })
        with patch("faultline.cloud.sync.send_mcp_events_batch") as mock_send:
            mock_send.return_value = 5
            count = event_buffer._buffer._flush_once()
        assert count == 5
        assert event_buffer._buffer._queue.qsize() == 0

    def test_flush_empty_returns_zero(self) -> None:
        assert event_buffer._buffer._flush_once() == 0
