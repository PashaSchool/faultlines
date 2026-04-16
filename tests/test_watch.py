"""Tests for faultline/watch/daemon.py (paths that don't require a real daemon)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from faultline.watch import daemon


@pytest.fixture(autouse=True)
def _isolate_pid_dir(tmp_path, monkeypatch):
    """Route PID files into tmp."""
    monkeypatch.setattr(daemon, "_PID_DIR", tmp_path / "watchers")
    yield


class TestInterestingPath:
    def test_source_files_are_interesting(self) -> None:
        assert daemon._interesting_path("src/foo.py") is True
        assert daemon._interesting_path("lib/bar.ts") is True
        assert daemon._interesting_path("pkg/baz.go") is True
        assert daemon._interesting_path("x.rs") is True

    def test_ignored_dirs_filtered(self) -> None:
        assert daemon._interesting_path("node_modules/pkg/index.js") is False
        assert daemon._interesting_path(".git/HEAD") is False
        assert daemon._interesting_path("__pycache__/foo.pyc") is False
        assert daemon._interesting_path(".venv/lib/site.py") is False
        assert daemon._interesting_path("dist/bundle.js") is False

    def test_non_source_files_ignored(self) -> None:
        assert daemon._interesting_path("README.md") is False
        assert daemon._interesting_path("package.json") is False
        assert daemon._interesting_path("image.png") is False
        assert daemon._interesting_path("Makefile") is False


class TestPidTracking:
    def test_write_and_read_pid(self, tmp_path) -> None:
        repo = str(tmp_path / "repo")
        daemon._write_pid(repo, 12345)
        info = daemon._read_pid(repo)
        assert info is not None
        assert info["pid"] == 12345
        assert info["repo_path"] == str(Path(repo).resolve())

    def test_clear_pid(self, tmp_path) -> None:
        repo = str(tmp_path / "repo")
        daemon._write_pid(repo, 12345)
        daemon._clear_pid(repo)
        assert daemon._read_pid(repo) is None

    def test_slug_is_stable_per_path(self, tmp_path) -> None:
        repo = str(tmp_path / "repo")
        assert daemon._slug(repo) == daemon._slug(repo)

    def test_slug_different_for_different_paths(self, tmp_path) -> None:
        assert daemon._slug(str(tmp_path / "a")) != daemon._slug(str(tmp_path / "b"))


class TestWatcherStatus:
    def test_returns_not_running_when_no_pid(self, tmp_path) -> None:
        status = daemon.watcher_status(str(tmp_path / "repo"))
        assert status.running is False
        assert status.pid is None

    def test_returns_running_for_live_pid(self, tmp_path) -> None:
        repo = str(tmp_path / "repo")
        import os
        daemon._write_pid(repo, os.getpid())  # our own pid is alive
        status = daemon.watcher_status(repo)
        assert status.running is True
        assert status.pid == os.getpid()

    def test_cleans_up_dead_pid(self, tmp_path) -> None:
        repo = str(tmp_path / "repo")
        daemon._write_pid(repo, 999998)  # likely dead
        status = daemon.watcher_status(repo)
        assert status.running is False
        assert daemon._read_pid(repo) is None  # auto-cleaned


class TestStopDaemon:
    def test_returns_false_when_not_running(self, tmp_path) -> None:
        assert daemon.stop_daemon(str(tmp_path / "repo")) is False

    def test_sends_sigterm_to_running_pid(self, tmp_path) -> None:
        import os
        import signal as _sig
        repo = str(tmp_path / "repo")
        daemon._write_pid(repo, os.getpid())
        with patch("os.kill") as mock_kill:
            result = daemon.stop_daemon(repo)
        assert result is True
        # One os.kill(pid, 0) for liveness check + one with SIGTERM
        sigterm_calls = [c for c in mock_kill.call_args_list if c.args[1] == _sig.SIGTERM]
        assert len(sigterm_calls) == 1


class TestRunRefresh:
    def test_writes_updated_map_when_stale(self, tmp_path) -> None:
        from datetime import datetime, timezone
        from faultline.cache.refresh import RefreshResult
        from faultline.cache.freshness import FreshnessReport
        from faultline.models.types import FeatureMap

        fm_path = tmp_path / "feature-map.json"
        fm = FeatureMap(
            repo_path="/tmp/x",
            analyzed_at=datetime.now(tz=timezone.utc),
            total_commits=1,
            date_range_days=30,
            features=[],
        )
        fm_path.write_text(fm.model_dump_json())

        updated = fm.model_copy()
        updated.last_scanned_sha = "new_sha"
        mock_result = RefreshResult(
            updated_map=updated,
            freshness_before=FreshnessReport(
                is_stale=True,
                current_sha="new_sha",
                scanned_sha="old_sha",
                commits_behind=3,
                changed_files_count=5,
                has_new_files=False,
                days_since_scan=1,
            ),
            files_truly_modified=5,
            files_added=0,
            files_removed=0,
        )

        with patch("faultline.cache.refresh.refresh_feature_map") as mock_refresh:
            mock_refresh.return_value = mock_result
            daemon._run_refresh(fm_path, "/tmp/x")

        # File should now contain new_sha
        written = json.loads(fm_path.read_text())
        assert written.get("last_scanned_sha") == "new_sha"

    def test_no_op_when_fresh(self, tmp_path) -> None:
        from datetime import datetime, timezone
        from faultline.cache.refresh import RefreshResult
        from faultline.cache.freshness import FreshnessReport
        from faultline.models.types import FeatureMap

        fm_path = tmp_path / "feature-map.json"
        fm = FeatureMap(
            repo_path="/tmp/x",
            analyzed_at=datetime.now(tz=timezone.utc),
            total_commits=1,
            date_range_days=30,
            features=[],
        )
        original = fm.model_dump_json()
        fm_path.write_text(original)

        mock_result = RefreshResult(
            updated_map=fm,
            freshness_before=FreshnessReport(
                is_stale=False,
                current_sha="same",
                scanned_sha="same",
                commits_behind=0,
                changed_files_count=0,
                has_new_files=False,
                days_since_scan=0,
            ),
            files_truly_modified=0,
            files_added=0,
            files_removed=0,
        )

        with patch("faultline.cache.refresh.refresh_feature_map") as mock_refresh:
            mock_refresh.return_value = mock_result
            daemon._run_refresh(fm_path, "/tmp/x")

        # File should be unchanged
        assert fm_path.read_text() == original


class TestResolveMapPath:
    def test_explicit_path_used_when_exists(self, tmp_path) -> None:
        p = tmp_path / "custom.json"
        p.write_text("{}")
        result = daemon._resolve_map_path(str(p), str(tmp_path))
        assert result == p

    def test_missing_explicit_returns_none(self, tmp_path) -> None:
        missing = tmp_path / "nope.json"
        result = daemon._resolve_map_path(str(missing), str(tmp_path))
        assert result is None
