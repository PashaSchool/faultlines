"""Unit tests for the new ``faultline.llm.pipeline.run`` entry point.

The pipeline wrapper is the Day 8 deliverable of the rewrite mission.
It replaces the five-strategy conditional at ``cli.py:264-483`` with a
single dispatch that picks between ``deep_scan_workspace`` (monorepo)
and ``deep_scan`` (monolith) based on workspace state.

These tests exercise the wrapper with mocked ``deep_scan`` /
``deep_scan_workspace`` so they stay hermetic (no LLM, no filesystem).
They cover:

  - Dispatch rules: empty / None / single-package workspace → single
    call; two+ packages → workspace path
  - ``is_library`` threading from ``RepoStructure`` into both paths
  - Commit context built once and forwarded unchanged
  - Tracker threading
  - Model and api_key forwarding
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from faultline.analyzer.repo_classifier import RepoStructure
from faultline.analyzer.workspace import WorkspaceInfo, WorkspacePackage
from faultline.llm.cost import CostTracker
from faultline.llm.pipeline import run
from faultline.llm.sonnet_scanner import DeepScanResult


# ── Fixtures ────────────────────────────────────────────────────────────


def _mk_structure(is_library: bool = False) -> RepoStructure:
    return RepoStructure(
        layout="feature",
        top_dirs=["src"],
        layer_ratio=0.1,
        is_library=is_library,
    )


def _mk_workspace(num_packages: int, files_per_pkg: int = 50) -> WorkspaceInfo:
    packages = [
        WorkspacePackage(
            name=f"pkg{i}",
            path=f"packages/pkg{i}",
            files=[f"packages/pkg{i}/src/file{j}.ts" for j in range(files_per_pkg)],
        )
        for i in range(num_packages)
    ]
    return WorkspaceInfo(
        detected=num_packages > 0,
        manager="pnpm",
        packages=packages,
        root_files=["package.json", "pnpm-workspace.yaml"],
    )


def _ds_result(features: dict[str, list[str]]) -> DeepScanResult:
    return DeepScanResult(features=features)


class _FakeCommit:
    """Stand-in for the Commit pydantic model used by build_commit_context."""
    def __init__(self, date, files_changed):
        self.date = date
        self.files_changed = files_changed


def _recent_commits(paths: list[str]) -> list[_FakeCommit]:
    return [
        _FakeCommit(
            date=datetime.now(timezone.utc) - timedelta(days=1),
            files_changed=paths,
        ),
    ]


# ── Dispatch rules ──────────────────────────────────────────────────────


class TestDispatchRules:
    @patch("faultline.llm.pipeline.deep_scan")
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_two_plus_packages_uses_workspace_path(
        self, mock_workspace: MagicMock, mock_single: MagicMock,
    ) -> None:
        mock_workspace.return_value = _ds_result({"pkg0": ["a.ts"]})
        run(
            analysis_files=["a.ts"],
            workspace=_mk_workspace(num_packages=3),
            repo_structure=_mk_structure(),
            api_key="sk-ant-test",
        )
        assert mock_workspace.call_count == 1
        assert mock_single.call_count == 0

    @patch("faultline.llm.pipeline.deep_scan")
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_single_package_workspace_uses_single_call(
        self, mock_workspace: MagicMock, mock_single: MagicMock,
    ) -> None:
        """One package isn't really a monorepo — single-call wins."""
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        run(
            analysis_files=["a.ts", "b.ts"],
            workspace=_mk_workspace(num_packages=1),
            repo_structure=_mk_structure(),
            api_key="sk-ant-test",
        )
        assert mock_workspace.call_count == 0
        assert mock_single.call_count == 1

    @patch("faultline.llm.pipeline.deep_scan")
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_no_workspace_uses_single_call(
        self, mock_workspace: MagicMock, mock_single: MagicMock,
    ) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        run(
            analysis_files=["src/a.ts", "src/b.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            api_key="sk-ant-test",
        )
        assert mock_workspace.call_count == 0
        assert mock_single.call_count == 1

    @patch("faultline.llm.pipeline.deep_scan")
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_undetected_workspace_uses_single_call(
        self, mock_workspace: MagicMock, mock_single: MagicMock,
    ) -> None:
        """detected=False wins even if packages happen to be present."""
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        ws = WorkspaceInfo(
            detected=False, manager="none", packages=[], root_files=[],
        )
        run(
            analysis_files=["src/a.ts"],
            workspace=ws,
            repo_structure=_mk_structure(),
            api_key="sk-ant-test",
        )
        assert mock_workspace.call_count == 0
        assert mock_single.call_count == 1


# ── is_library threading ────────────────────────────────────────────────


class TestIsLibraryThreading:
    @patch("faultline.llm.pipeline.deep_scan")
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_library_flag_reaches_workspace_path(
        self, mock_workspace: MagicMock, mock_single: MagicMock,
    ) -> None:
        mock_workspace.return_value = _ds_result({"pkg0": ["a.ts"]})
        run(
            analysis_files=["a.ts"],
            workspace=_mk_workspace(num_packages=3),
            repo_structure=_mk_structure(is_library=True),
            api_key="sk-ant-test",
        )
        assert mock_workspace.call_args.kwargs["is_library"] is True

    @patch("faultline.llm.pipeline.deep_scan")
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_library_flag_reaches_single_call_path(
        self, mock_workspace: MagicMock, mock_single: MagicMock,
    ) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(is_library=True),
            api_key="sk-ant-test",
        )
        assert mock_single.call_args.kwargs["is_library"] is True

    @patch("faultline.llm.pipeline.deep_scan")
    def test_missing_repo_structure_assumes_application(
        self, mock_single: MagicMock,
    ) -> None:
        """None repo_structure must not crash and must default to app mode."""
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=None,
            api_key="sk-ant-test",
        )
        assert mock_single.call_args.kwargs["is_library"] is False


# ── Commit context ──────────────────────────────────────────────────────


class TestCommitContextThreading:
    @patch("faultline.llm.pipeline.deep_scan")
    def test_no_commits_means_no_context(self, mock_single: MagicMock) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            commits=None,
            api_key="sk-ant-test",
        )
        assert mock_single.call_args.kwargs["commit_context"] is None

    @patch("faultline.llm.pipeline.deep_scan")
    def test_recent_commits_build_context_block(
        self, mock_single: MagicMock,
    ) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        run(
            analysis_files=["src/auth/login.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            commits=_recent_commits(["src/auth/login.ts", "src/auth/signup.ts"]),
            api_key="sk-ant-test",
        )
        ctx = mock_single.call_args.kwargs["commit_context"]
        assert ctx is not None
        assert "src/auth/login.ts" in ctx

    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_context_threads_to_workspace_path(
        self, mock_workspace: MagicMock,
    ) -> None:
        mock_workspace.return_value = _ds_result({"pkg0": ["a.ts"]})
        run(
            analysis_files=["a.ts"],
            workspace=_mk_workspace(num_packages=3),
            repo_structure=_mk_structure(),
            commits=_recent_commits(["packages/pkg0/src/x.ts"]),
            api_key="sk-ant-test",
        )
        ctx = mock_workspace.call_args.kwargs["commit_context"]
        assert ctx is not None
        assert "packages/pkg0/src/x.ts" in ctx

    @patch("faultline.llm.pipeline.deep_scan")
    def test_top_n_bound_respected(self, mock_single: MagicMock) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        commits = [
            _FakeCommit(
                date=datetime.now(timezone.utc) - timedelta(days=1),
                files_changed=[f"src/feat{i}/file.ts"],
            )
            for i in range(50)
        ]
        run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            commits=commits,
            commit_context_top_n=3,
            api_key="sk-ant-test",
        )
        ctx = mock_single.call_args.kwargs["commit_context"]
        assert ctx is not None
        assert len(ctx.splitlines()) == 3


# ── Tracker + forwarding ────────────────────────────────────────────────


class TestTrackerAndForwarding:
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_tracker_threads_to_workspace(
        self, mock_workspace: MagicMock,
    ) -> None:
        mock_workspace.return_value = _ds_result({"pkg0": ["a.ts"]})
        tracker = CostTracker()
        run(
            analysis_files=["a.ts"],
            workspace=_mk_workspace(num_packages=3),
            repo_structure=_mk_structure(),
            tracker=tracker,
            api_key="sk-ant-test",
        )
        assert mock_workspace.call_args.kwargs["tracker"] is tracker

    @patch("faultline.llm.pipeline.deep_scan")
    def test_tracker_threads_to_single_call(self, mock_single: MagicMock) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        tracker = CostTracker()
        run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            tracker=tracker,
            api_key="sk-ant-test",
        )
        assert mock_single.call_args.kwargs["tracker"] is tracker

    @patch("faultline.llm.pipeline.deep_scan")
    def test_model_override_forwards(self, mock_single: MagicMock) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            model="claude-haiku-4-5",
            api_key="sk-ant-test",
        )
        assert mock_single.call_args.kwargs["model"] == "claude-haiku-4-5"

    @patch("faultline.llm.pipeline.deep_scan")
    def test_signatures_forward_to_single_call(
        self, mock_single: MagicMock,
    ) -> None:
        mock_single.return_value = _ds_result({"auth": ["a.ts"]})
        sigs = {"src/a.ts": object()}
        run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            signatures=sigs,
            api_key="sk-ant-test",
        )
        assert mock_single.call_args.kwargs["signatures"] is sigs

    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_signatures_forward_to_workspace(
        self, mock_workspace: MagicMock,
    ) -> None:
        mock_workspace.return_value = _ds_result({"pkg0": ["a.ts"]})
        sigs = {"packages/pkg0/src/a.ts": object()}
        run(
            analysis_files=["a.ts"],
            workspace=_mk_workspace(num_packages=3),
            repo_structure=_mk_structure(),
            signatures=sigs,
            api_key="sk-ant-test",
        )
        assert mock_workspace.call_args.kwargs["signatures"] is sigs


# ── Return value ────────────────────────────────────────────────────────


class TestReturnValue:
    @patch("faultline.llm.pipeline.deep_scan_workspace")
    def test_returns_deep_scan_result_from_workspace(
        self, mock_workspace: MagicMock,
    ) -> None:
        expected = _ds_result({"pkg0": ["a.ts"]})
        mock_workspace.return_value = expected
        result = run(
            analysis_files=["a.ts"],
            workspace=_mk_workspace(num_packages=3),
            repo_structure=_mk_structure(),
            api_key="sk-ant-test",
        )
        assert result is expected
        assert isinstance(result, DeepScanResult)

    @patch("faultline.llm.pipeline.deep_scan")
    def test_returns_deep_scan_result_from_single_call(
        self, mock_single: MagicMock,
    ) -> None:
        expected = _ds_result({"auth": ["a.ts"]})
        mock_single.return_value = expected
        result = run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            api_key="sk-ant-test",
        )
        assert result is expected

    @patch("faultline.llm.pipeline.deep_scan")
    def test_propagates_none_return(self, mock_single: MagicMock) -> None:
        """If the underlying scan returns None (LLM failure), run() passes
        it through so the caller can surface the error."""
        mock_single.return_value = None
        result = run(
            analysis_files=["src/a.ts"],
            workspace=None,
            repo_structure=_mk_structure(),
            api_key="sk-ant-test",
        )
        assert result is None
