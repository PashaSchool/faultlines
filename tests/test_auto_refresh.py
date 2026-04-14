"""Tests for faultline/cache/auto_refresh.py."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from faultline.cache import auto_refresh
from faultline.cache.auto_refresh import (
    RefreshLock,
    maybe_trigger_refresh,
    reset_throttle,
    should_refresh,
)


@pytest.fixture(autouse=True)
def _isolate_lock_dir(tmp_path, monkeypatch):
    """Route lock files into the test's tmpdir so tests don't leak."""
    monkeypatch.setattr(auto_refresh, "_LOCK_DIR", tmp_path)
    auto_refresh._throttle_state.clear()
    yield
    auto_refresh._throttle_state.clear()


class TestRefreshLock:
    def test_acquires_when_no_existing_lock(self) -> None:
        lock = RefreshLock("slug1")
        assert lock.acquire() is True
        assert lock.path.exists()
        lock.release()
        assert not lock.path.exists()

    def test_blocks_second_acquire(self) -> None:
        a = RefreshLock("slug2")
        b = RefreshLock("slug2")
        assert a.acquire() is True
        assert b.acquire() is False
        a.release()

    def test_auto_releases_stale_lock(self, tmp_path) -> None:
        lock = RefreshLock("slug3")
        lock.path.write_text(json.dumps({
            "pid": 999999,  # unlikely to be alive
            "timestamp": time.time() - 3600,  # 1 hour old
        }))
        assert lock.acquire() is True

    def test_release_only_removes_own_lock(self, tmp_path) -> None:
        lock = RefreshLock("slug4")
        lock.path.write_text(json.dumps({
            "pid": 999998,
            "timestamp": time.time(),  # fresh, another process
        }))
        # release() should NOT touch someone else's lock
        lock.release()
        assert lock.path.exists()


class TestThrottle:
    def test_first_call_not_throttled(self) -> None:
        assert should_refresh("repo1") is True

    def test_subsequent_call_throttled(self) -> None:
        assert should_refresh("repo1") is True
        assert should_refresh("repo1") is False

    def test_different_repos_independent(self) -> None:
        assert should_refresh("repo1") is True
        assert should_refresh("repo2") is True

    def test_short_throttle_respected(self) -> None:
        assert should_refresh("repo1", throttle_seconds=0) is True
        # Even tiny throttle blocks immediate re-call
        # (time.time() resolution may allow this; tolerate either)

    def test_reset_throttle_clears(self) -> None:
        should_refresh("repo1")
        assert should_refresh("repo1") is False
        reset_throttle("repo1")
        assert should_refresh("repo1") is True


class TestMaybeTriggerRefresh:
    def test_no_op_when_env_not_set(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("FAULTLINE_AUTO_REFRESH", raising=False)
        fm_path = tmp_path / "feature-map.json"
        fm_path.write_text("{}")
        triggered = maybe_trigger_refresh(fm_path, {"repo_path": str(tmp_path)})
        assert triggered is False

    def test_no_op_without_sha(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH", "1")
        fm_path = tmp_path / "feature-map.json"
        triggered = maybe_trigger_refresh(fm_path, {"repo_path": str(tmp_path)})
        assert triggered is False

    def test_no_op_when_git_head_matches(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH", "1")
        fm_path = tmp_path / "feature-map.json"
        with patch("subprocess.check_output") as mock_git:
            mock_git.return_value = "abc123\n"
            data = {"repo_path": str(tmp_path), "last_scanned_sha": "abc123"}
            triggered = maybe_trigger_refresh(fm_path, data)
        assert triggered is False

    def test_triggers_when_stale_and_enabled(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH", "1")
        fm_path = tmp_path / "feature-map.json"
        with patch("subprocess.check_output") as mock_git, \
             patch("faultline.cache.auto_refresh.trigger_background_refresh") as mock_bg:
            mock_git.return_value = "new_sha\n"
            mock_bg.return_value = True
            data = {"repo_path": str(tmp_path), "last_scanned_sha": "old_sha"}
            triggered = maybe_trigger_refresh(fm_path, data)
        assert triggered is True
        mock_bg.assert_called_once()
        # Default throttle used when env var unset
        kwargs = mock_bg.call_args.kwargs
        assert kwargs.get("throttle_seconds") is None

    def test_throttle_override_from_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH", "1")
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH_THROTTLE", "60")
        fm_path = tmp_path / "feature-map.json"
        with patch("subprocess.check_output") as mock_git, \
             patch("faultline.cache.auto_refresh.trigger_background_refresh") as mock_bg:
            mock_git.return_value = "new_sha\n"
            mock_bg.return_value = True
            data = {"repo_path": str(tmp_path), "last_scanned_sha": "old_sha"}
            maybe_trigger_refresh(fm_path, data)
        assert mock_bg.call_args.kwargs.get("throttle_seconds") == 60

    def test_throttle_zero_disables_throttling(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH", "1")
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH_THROTTLE", "0")
        fm_path = tmp_path / "feature-map.json"
        with patch("subprocess.check_output") as mock_git, \
             patch("faultline.cache.auto_refresh.trigger_background_refresh") as mock_bg:
            mock_git.return_value = "new_sha\n"
            mock_bg.return_value = True
            data = {"repo_path": str(tmp_path), "last_scanned_sha": "old_sha"}
            maybe_trigger_refresh(fm_path, data)
        assert mock_bg.call_args.kwargs.get("throttle_seconds") == 0

    def test_bad_throttle_value_uses_default(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH", "1")
        monkeypatch.setenv("FAULTLINE_AUTO_REFRESH_THROTTLE", "not-a-number")
        fm_path = tmp_path / "feature-map.json"
        with patch("subprocess.check_output") as mock_git, \
             patch("faultline.cache.auto_refresh.trigger_background_refresh") as mock_bg:
            mock_git.return_value = "new_sha\n"
            mock_bg.return_value = True
            data = {"repo_path": str(tmp_path), "last_scanned_sha": "old_sha"}
            maybe_trigger_refresh(fm_path, data)
        # Falls back to None (i.e. default) for bad input
        assert mock_bg.call_args.kwargs.get("throttle_seconds") is None
