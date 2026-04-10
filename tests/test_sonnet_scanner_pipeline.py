"""Unit tests for the pre/post-processing helpers in sonnet_scanner.

These cover deltas D1, D2, D3, D4, D8, D9, D11, D12 from
docs/rewrite/sonnet_scanner_delta.md without calling the LLM. The
helpers are intentionally extracted from ``deep_scan`` so the
validation primitives can be verified here with no network, no
API key, and sub-millisecond runtime.

Scenarios are modeled on the real Day 1 baseline failures:
  - fastapi: 21 phantom features from docs_src/tutorial00N paths
  - trpc:    www/* split into 8 features, 43 library flows
  - cal.com: vitest-mocks leaked as a feature
  - gin:     "root" bucket leaked as a feature
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import faultline.llm.sonnet_scanner as scanner
from faultline.llm.cost import BudgetExceeded, CostTracker
from faultline.analyzer.workspace import WorkspaceInfo, WorkspacePackage
from faultline.llm.sonnet_scanner import (
    DeepScanResult,
    SonnetFeature,
    SonnetFlow,
    SonnetOpsResponse,
    _build_system_prompt,
    _clean_inputs,
    _finalize_result,
    _promote_library_root_candidates,
    build_commit_context,
    deep_scan,
    deep_scan_workspace,
)


class TestPromoteLibraryRootCandidates:
    """Day 14 library-mode fix: split root catchall by filename stem."""

    def test_gin_root_splits_by_stem(self) -> None:
        """Typical gin case: 'root' catchall holds top-level .go files that
        should each become their own per-stem candidate."""
        candidates = {
            "root": [
                "auth.go", "context.go", "context_appengine.go",
                "errors.go", "logger.go", "recovery.go",
                "routergroup.go", "tree.go",
            ],
            "binding": ["binding/json.go", "binding/xml.go"],
        }
        result = _promote_library_root_candidates(candidates)
        assert "root" not in result
        assert "binding" in result  # untouched
        assert result["auth"] == ["auth.go"]
        assert result["errors"] == ["errors.go"]
        assert result["logger"] == ["logger.go"]
        assert result["recovery"] == ["recovery.go"]
        assert result["routergroup"] == ["routergroup.go"]
        assert result["tree"] == ["tree.go"]
        # context.go + context_appengine.go should collapse to "context"
        assert set(result["context"]) == {"context.go", "context_appengine.go"}

    def test_init_catchall_also_split(self) -> None:
        """Python-style catchall 'init' gets the same treatment as 'root'."""
        candidates = {
            "init": ["routing.py", "security.py", "middleware.py"],
        }
        result = _promote_library_root_candidates(candidates)
        assert "init" not in result
        assert result["routing"] == ["routing.py"]
        assert result["security"] == ["security.py"]
        assert result["middleware"] == ["middleware.py"]

    def test_no_root_catchall_is_noop(self) -> None:
        """When there's nothing to split, return candidates unchanged."""
        candidates = {
            "binding": ["binding/json.go"],
            "render": ["render/json.go"],
        }
        result = _promote_library_root_candidates(candidates)
        assert result == candidates

    def test_stem_collision_with_existing_candidate(self) -> None:
        """If a stem matches an existing subdir candidate, merge — never
        drop files on the floor."""
        candidates = {
            "root": ["render.go"],
            "render": ["render/json.go", "render/xml.go"],
        }
        result = _promote_library_root_candidates(candidates)
        assert "root" not in result
        assert sorted(result["render"]) == ["render.go", "render/json.go", "render/xml.go"]

    def test_trims_os_arch_suffixes(self) -> None:
        """Files differing only by OS/arch/test suffix should group together."""
        candidates = {
            "root": [
                "fs.go", "fs_linux.go", "fs_windows.go", "fs_darwin.go",
                "fs_test.go",
            ],
        }
        result = _promote_library_root_candidates(candidates)
        assert "root" not in result
        assert "fs" in result
        assert len(result["fs"]) == 5

    def test_handles_empty_root(self) -> None:
        candidates = {"root": [], "binding": ["binding/json.go"]}
        result = _promote_library_root_candidates(candidates)
        assert "root" not in result
        assert result["binding"] == ["binding/json.go"]


def _ds_result(features: dict[str, list[str]]) -> DeepScanResult:
    """Build a minimal DeepScanResult for mocked deep_scan returns."""
    return DeepScanResult(features=features)


# ── _clean_inputs: D2, D3 ────────────────────────────────────────────────


class TestCleanInputs:
    def test_filters_test_files_from_files_list(self) -> None:
        files = [
            "src/auth/login.ts",
            "src/auth/login.test.ts",
            "tests/unit/parser.py",
            "README.md",
        ]
        cleaned_files, _, _ = _clean_inputs(files, {})
        assert "src/auth/login.ts" in cleaned_files
        assert "README.md" in cleaned_files
        assert "src/auth/login.test.ts" not in cleaned_files
        assert "tests/unit/parser.py" not in cleaned_files

    def test_partitions_docs_files(self) -> None:
        files = [
            "fastapi/routing.py",
            "docs_src/tutorial001_py310/main.py",
            "docs_src/tutorial002_py310/app.py",
            "www/blog/post-1.mdx",
        ]
        cleaned_files, _, docs_files = _clean_inputs(files, {})
        assert cleaned_files == ["fastapi/routing.py"]
        assert len(docs_files) == 3
        assert all(("docs_src" in f) or ("www/" in f) for f in docs_files)

    def test_fastapi_regression_all_tutorials_go_to_docs(self) -> None:
        """fastapi baseline: 21 docs_src/tutorial00N dirs must land in docs bucket."""
        files = [
            "fastapi/routing.py",
            "fastapi/applications.py",
        ] + [
            f"docs_src/tutorial00{i}_py310/main.py"
            for i in range(1, 10)
        ]
        cleaned_files, _, docs_files = _clean_inputs(files, {})
        assert set(cleaned_files) == {"fastapi/routing.py", "fastapi/applications.py"}
        assert len(docs_files) == 9
        for f in docs_files:
            assert f.startswith("docs_src/")

    def test_drops_test_feature_name_from_candidates(self) -> None:
        """cal.com regression: vitest-mocks must not survive as a candidate."""
        candidates = {
            "authentication": ["src/auth/login.ts", "src/auth/signup.ts"],
            "vitest-mocks": ["packages/embeds/vitest-mocks/handler.ts"],
            "__tests__": ["packages/core/__tests__/parser.ts"],
        }
        _, cleaned_candidates, _ = _clean_inputs([], candidates)
        assert "authentication" in cleaned_candidates
        assert "vitest-mocks" not in cleaned_candidates
        assert "__tests__" not in cleaned_candidates

    def test_removes_docs_paths_from_candidate_buckets(self) -> None:
        """A candidate that mixes code + docs must keep only code paths."""
        candidates = {
            "auth": [
                "src/auth/login.ts",
                "src/auth/signup.ts",
                "docs/auth/guide.md",         # should be stripped
                "examples/auth/demo.ts",       # should be stripped
            ],
        }
        _, cleaned, _ = _clean_inputs([], candidates)
        assert cleaned["auth"] == ["src/auth/login.ts", "src/auth/signup.ts"]

    def test_removes_test_paths_from_candidate_buckets(self) -> None:
        candidates = {
            "auth": [
                "src/auth/login.ts",
                "src/auth/login.test.ts",
                "src/auth/__tests__/signup.ts",
            ],
        }
        _, cleaned, _ = _clean_inputs([], candidates)
        assert cleaned["auth"] == ["src/auth/login.ts"]

    def test_drops_candidates_that_become_empty(self) -> None:
        """A candidate whose every path is test/docs is dropped entirely."""
        candidates = {
            "auth": ["src/auth/login.ts"],
            "ghost": [
                "src/ghost/a.test.ts",
                "src/ghost/b.test.ts",
            ],
        }
        _, cleaned, _ = _clean_inputs([], candidates)
        assert "auth" in cleaned
        assert "ghost" not in cleaned

    def test_preserves_ordering_within_buckets(self) -> None:
        """_clean_inputs must not shuffle paths within a candidate."""
        candidates = {
            "auth": [
                "src/auth/z-signup.ts",
                "src/auth/a-login.ts",
                "src/auth/m-session.ts",
            ],
        }
        _, cleaned, _ = _clean_inputs([], candidates)
        assert cleaned["auth"] == [
            "src/auth/z-signup.ts",
            "src/auth/a-login.ts",
            "src/auth/m-session.ts",
        ]

    def test_empty_inputs(self) -> None:
        cleaned_files, cleaned_candidates, docs_files = _clean_inputs([], {})
        assert cleaned_files == []
        assert cleaned_candidates == {}
        assert docs_files == []


# ── _finalize_result: D1, D8, D9 ─────────────────────────────────────────


def _make_ops(features: list[tuple[str, list[str]]]) -> SonnetOpsResponse:
    """Build a SonnetOpsResponse with named features and dummy flows."""
    return SonnetOpsResponse(
        features=[
            SonnetFeature(
                name=name,
                description=f"desc for {name}",
                flows=[SonnetFlow(name=f"{name}-flow", description=f"{name} action")],
            )
            for name, _ in features
        ]
    )


class TestFinalizeResult:
    def setup_method(self) -> None:
        # Reset the module-global side channel between tests
        scanner._last_scan_result = None

    def test_attaches_documentation_bucket(self) -> None:
        """D2: docs files partitioned in _clean_inputs get reattached here."""
        result = {"auth": ["src/auth/login.ts"]}
        docs = ["docs/guide.md", "examples/sample.ts"]
        cleaned = _finalize_result(result, docs_files=docs, is_library=False)
        assert "documentation" in cleaned
        assert cleaned["documentation"] == ["docs/guide.md", "examples/sample.ts"]
        assert cleaned["auth"] == ["src/auth/login.ts"]

    def test_merges_docs_into_existing_bucket(self) -> None:
        """If LLM already created a 'documentation' feature, merge instead of clobbering."""
        result = {"documentation": ["docs/existing.md"]}
        docs = ["docs/new.md"]
        cleaned = _finalize_result(result, docs_files=docs, is_library=False)
        assert cleaned["documentation"] == ["docs/existing.md", "docs/new.md"]

    def test_no_docs_bucket_when_no_docs_files(self) -> None:
        result = {"auth": ["src/auth/login.ts"]}
        cleaned = _finalize_result(result, docs_files=[], is_library=False)
        assert "documentation" not in cleaned

    def test_canonicalizes_root_to_shared_infra(self) -> None:
        """gin regression: 'root' must become 'shared-infra'."""
        result = {
            "binding": ["binding/a.go"],
            "root": ["main.go", "helpers.go"],
        }
        cleaned = _finalize_result(result, docs_files=[], is_library=True)
        assert "root" not in cleaned
        assert "shared-infra" in cleaned
        assert set(cleaned["shared-infra"]) == {"main.go", "helpers.go"}

    def test_merges_canonical_duplicates(self) -> None:
        """If both 'root' and 'init' appear, they both land in shared-infra."""
        result = {
            "auth": ["src/auth/a.ts"],
            "root": ["main.ts"],
            "init": ["bootstrap.ts"],
            "shared-infra": ["utils/time.ts"],
        }
        cleaned = _finalize_result(result, docs_files=[], is_library=False)
        assert "root" not in cleaned
        assert "init" not in cleaned
        assert sorted(cleaned["shared-infra"]) == sorted([
            "main.ts", "bootstrap.ts", "utils/time.ts",
        ])

    def test_drops_phantom_empty_feature(self) -> None:
        result = {
            "auth": ["src/auth/login.ts"],
            "ghost": [],
        }
        cleaned = _finalize_result(result, docs_files=[], is_library=False)
        assert "ghost" not in cleaned
        assert "auth" in cleaned

    def test_drops_phantom_test_named_feature(self) -> None:
        """Belt-and-braces: even if a test-named feature slips past
        _clean_inputs (e.g. added by the LLM), drop it here."""
        result = {
            "auth": ["src/auth/login.ts"],
            "__tests__": ["something.ts"],
        }
        cleaned = _finalize_result(result, docs_files=[], is_library=False)
        assert "__tests__" not in cleaned
        assert "auth" in cleaned

    def test_is_library_strips_flows_from_side_channel(self) -> None:
        """D1: library repos must have 0 flows regardless of what Sonnet returned."""
        scanner._last_scan_result = _make_ops([
            ("binding", ["b.go"]),
            ("router", ["r.go"]),
        ])
        result = {"binding": ["b.go"], "router": ["r.go"]}

        _finalize_result(result, docs_files=[], is_library=True)

        assert scanner._last_scan_result is not None
        for feat in scanner._last_scan_result.features:
            assert feat.flows == []

    def test_non_library_preserves_flows(self) -> None:
        scanner._last_scan_result = _make_ops([("auth", ["a.ts"])])
        result = {"auth": ["a.ts"]}

        _finalize_result(result, docs_files=[], is_library=False)

        assert scanner._last_scan_result is not None
        assert len(scanner._last_scan_result.features[0].flows) == 1

    def test_canonicalizes_side_channel_feature_names(self) -> None:
        """get_deep_scan_flows matches by name; canonicalized result must
        stay in sync with _last_scan_result feature names."""
        scanner._last_scan_result = _make_ops([("root", ["main.go"])])
        result = {"root": ["main.go"]}

        cleaned = _finalize_result(result, docs_files=[], is_library=False)

        assert "shared-infra" in cleaned
        assert scanner._last_scan_result is not None
        assert scanner._last_scan_result.features[0].name == "shared-infra"

    def test_works_without_side_channel(self) -> None:
        """_finalize_result must not crash when _last_scan_result is None
        (e.g. when the LLM call itself failed but we still want to clean)."""
        scanner._last_scan_result = None
        result = {"auth": ["a.ts"], "root": ["main.ts"]}
        cleaned = _finalize_result(result, docs_files=[], is_library=True)
        assert "auth" in cleaned
        assert "shared-infra" in cleaned


# ── End-to-end flow on the extracted pipeline ───────────────────────────


class TestEndToEndWithoutLLM:
    """Simulate the pipeline around the LLM call: _clean_inputs → fake ops
    application → _finalize_result. Verifies the two helpers compose."""

    def setup_method(self) -> None:
        scanner._last_scan_result = None

    def test_fastapi_shape_without_llm(self) -> None:
        """fastapi: 3 code files + 9 tutorial files → 1 code feature + 1 docs
        feature, regardless of what the (mocked) LLM would do."""
        files = [
            "fastapi/routing.py",
            "fastapi/applications.py",
            "fastapi/dependencies.py",
        ] + [f"docs_src/tutorial00{i}_py310/main.py" for i in range(1, 10)]
        candidates = {
            "fastapi": ["fastapi/routing.py", "fastapi/applications.py", "fastapi/dependencies.py"],
        }

        cleaned_files, cleaned_candidates, docs_files = _clean_inputs(files, candidates)
        assert len(cleaned_files) == 3
        assert len(docs_files) == 9
        assert "fastapi" in cleaned_candidates

        # Pretend the LLM returned the 'fastapi' feature as-is
        scanner._last_scan_result = _make_ops([("fastapi", cleaned_candidates["fastapi"])])
        fake_result = dict(cleaned_candidates)

        final = _finalize_result(fake_result, docs_files=docs_files, is_library=True)
        assert set(final.keys()) == {"fastapi", "documentation"}
        assert len(final["documentation"]) == 9
        # Library → flows stripped
        assert scanner._last_scan_result is not None
        assert scanner._last_scan_result.features[0].flows == []


# ── D11: deterministic sort ──────────────────────────────────────────────


class TestDeterministicSort:
    def setup_method(self) -> None:
        scanner._last_scan_result = None

    def test_sorts_by_size_descending(self) -> None:
        """Largest feature first, then smaller — stable across runs."""
        result = {
            "small": ["a.ts"],
            "huge": ["1.ts", "2.ts", "3.ts", "4.ts"],
            "medium": ["x.ts", "y.ts"],
        }
        cleaned = _finalize_result(result, docs_files=[], is_library=False)
        assert list(cleaned.keys()) == ["huge", "medium", "small"]

    def test_sorts_by_name_when_sizes_tie(self) -> None:
        """Two features with identical file counts: alphabetical name wins."""
        result = {
            "zebra": ["z1.ts", "z2.ts"],
            "apple": ["a1.ts", "a2.ts"],
            "mango": ["m1.ts", "m2.ts"],
        }
        cleaned = _finalize_result(result, docs_files=[], is_library=False)
        assert list(cleaned.keys()) == ["apple", "mango", "zebra"]

    def test_two_runs_produce_identical_order(self) -> None:
        """Regression: shuffled input → same output order every call."""
        a = _finalize_result(
            {"b": ["1"], "a": ["1", "2"], "c": ["1", "2", "3"]},
            docs_files=[],
            is_library=False,
        )
        b = _finalize_result(
            {"c": ["1", "2", "3"], "a": ["1", "2"], "b": ["1"]},
            docs_files=[],
            is_library=False,
        )
        assert list(a.keys()) == list(b.keys())
        assert list(a.keys()) == ["c", "a", "b"]

    def test_sorts_side_channel_to_match(self) -> None:
        """_last_scan_result feature ordering must mirror the result dict
        so downstream flow matching iterates them in the same order."""
        scanner._last_scan_result = _make_ops([
            ("small", ["s.ts"]),
            ("huge", ["1.ts", "2.ts", "3.ts"]),
            ("medium", ["x.ts", "y.ts"]),
        ])
        result = {
            "small": ["s.ts"],
            "huge": ["1.ts", "2.ts", "3.ts"],
            "medium": ["x.ts", "y.ts"],
        }
        _finalize_result(result, docs_files=[], is_library=False)
        names = [f.name for f in scanner._last_scan_result.features]
        assert names == ["huge", "medium", "small"]


# ── D4 + D12: tracker wiring and configurable model ─────────────────────


def _make_llm_response(json_text: str, input_tokens: int, output_tokens: int) -> MagicMock:
    """Build a fake Anthropic response with .content[0].text and .usage.*"""
    resp = MagicMock()
    resp.content = [MagicMock()]
    resp.content[0].text = json_text
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    return resp


_MINIMAL_OPS_JSON = (
    '{"merge":[],"rename":[],"remove":[],"split":[],'
    '"features":[{"name":"auth","description":"d","flows":[]}]}'
)


class TestCostTrackerWiring:
    def setup_method(self) -> None:
        scanner._last_scan_result = None

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_records_tokens_when_tracker_provided(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, input_tokens=12_345, output_tokens=678,
        )

        tracker = CostTracker()
        deep_scan(
            files=["src/auth/login.ts", "src/auth/signup.ts"],
            candidates={"auth": ["src/auth/login.ts", "src/auth/signup.ts"]},
            api_key="sk-ant-test",
            tracker=tracker,
        )

        assert tracker.call_count == 1
        rec = tracker.records[0]
        assert rec.provider == "anthropic"
        assert rec.input_tokens == 12_345
        assert rec.output_tokens == 678
        assert rec.label == "deep-scan"

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_no_tracker_means_no_recording(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        # Should not raise even though tracker is None
        result = deep_scan(
            files=["src/auth/login.ts", "src/auth/signup.ts"],
            candidates={"auth": ["src/auth/login.ts", "src/auth/signup.ts"]},
            api_key="sk-ant-test",
        )
        assert result is not None

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_budget_exceeded_raises_and_aborts(self, mock_cls: MagicMock) -> None:
        """A tight budget should trip check_budget() and bubble up."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        # Huge token counts to guarantee the budget is blown.
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON,
            input_tokens=10_000_000,
            output_tokens=1_000_000,
        )

        tracker = CostTracker(max_cost=0.01)
        with pytest.raises(BudgetExceeded):
            deep_scan(
                files=["src/a.ts"],
                candidates={"auth": ["src/a.ts", "src/b.ts"]},
                api_key="sk-ant-test",
                tracker=tracker,
            )

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_handles_missing_usage_gracefully(self, mock_cls: MagicMock) -> None:
        """If the Anthropic SDK response has no .usage (old SDK, mocks),
        tracker should record 0 tokens instead of crashing."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        bad_resp = MagicMock()
        bad_resp.content = [MagicMock()]
        bad_resp.content[0].text = _MINIMAL_OPS_JSON
        # Remove usage attribute entirely
        del bad_resp.usage
        mock_client.messages.create.return_value = bad_resp

        tracker = CostTracker()
        deep_scan(
            files=["src/a.ts"],
            candidates={"auth": ["src/a.ts", "src/b.ts"]},
            api_key="sk-ant-test",
            tracker=tracker,
        )
        assert tracker.call_count == 1
        assert tracker.records[0].input_tokens == 0
        assert tracker.records[0].output_tokens == 0


class TestModelOverride:
    def setup_method(self) -> None:
        scanner._last_scan_result = None

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_default_model_used_when_not_specified(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        deep_scan(
            files=["src/a.ts"],
            candidates={"auth": ["src/a.ts", "src/b.ts"]},
            api_key="sk-ant-test",
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == scanner._MODEL

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_custom_model_threads_through(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        tracker = CostTracker()
        deep_scan(
            files=["src/a.ts"],
            candidates={"auth": ["src/a.ts", "src/b.ts"]},
            api_key="sk-ant-test",
            model="claude-haiku-4-5",
            tracker=tracker,
        )

        # Sent to Anthropic
        assert mock_client.messages.create.call_args.kwargs["model"] == "claude-haiku-4-5"
        # Recorded in tracker with the override (so pricing lookup hits Haiku, not Sonnet)
        assert tracker.records[0].model == "claude-haiku-4-5"


# ── D7: package-mode prompt and split hard cap ──────────────────────────


class TestPackageModePrompt:
    def test_repo_prompt_targets_12_25_features(self) -> None:
        """Default mode keeps the legacy 12-25 target language."""
        prompt = _build_system_prompt()
        assert "12-25" in prompt
        assert "single package within a monorepo" not in prompt

    def test_package_prompt_targets_1_8_features(self) -> None:
        """Package mode swaps in the 1-8 target and HARD CAP language."""
        prompt = _build_system_prompt(package_mode=True, package_name="auth")
        assert "1-8 features" in prompt
        assert "HARD CAP" in prompt
        assert "`auth`" in prompt
        # Repo target must be gone
        assert "12-25 business features" not in prompt

    def test_package_prompt_handles_missing_name(self) -> None:
        """Defensive: package_mode without a name should still render."""
        prompt = _build_system_prompt(package_mode=True)
        assert "1-8 features" in prompt
        assert "unknown" in prompt

    def test_split_rule_has_hard_cap(self) -> None:
        """D7: every prompt variant carries the 8-sub-feature hard cap."""
        for prompt in (
            _build_system_prompt(),
            _build_system_prompt(package_mode=True, package_name="x"),
            _build_system_prompt(is_library=True),
        ):
            assert "HARD CAP" in prompt
            assert "8 sub-features" in prompt

    def test_library_prompt_targets_5_15_modules(self) -> None:
        """Day 14 library-mode fix: libraries ask for 5-15 public modules,
        not 12-25 business features."""
        prompt = _build_system_prompt(is_library=True)
        assert "5-15 public modules" in prompt
        assert "consumable library" in prompt
        # Must name common library modules to prime the LLM
        assert "router" in prompt
        assert "middleware" in prompt
        # Must explicitly forbid the over-merge failure mode
        assert "shared-infra" in prompt
        # Must not use the repo-wide target
        assert "12-25 business features" not in prompt
        # Must not apply the package-mode language
        assert "single package within a monorepo" not in prompt

    def test_package_mode_beats_library_mode(self) -> None:
        """Precedence: package_mode wins because each per-package call
        is already isolated enough that library-vs-app is irrelevant."""
        prompt = _build_system_prompt(
            package_mode=True,
            package_name="auth",
            is_library=True,
        )
        # Package-mode target should be active
        assert "1-8 features" in prompt
        assert "`auth`" in prompt
        # Library-mode target should NOT leak in
        assert "5-15 public modules" not in prompt
        assert "consumable library" not in prompt

    def test_default_still_repo_mode(self) -> None:
        """Default (no flags) keeps the 12-25 target language."""
        prompt = _build_system_prompt()
        assert "12-25 business features" in prompt
        assert "5-15 public modules" not in prompt
        assert "single package within a monorepo" not in prompt

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_deep_scan_uses_library_prompt_when_is_library(
        self, mock_cls: MagicMock,
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        deep_scan(
            files=["src/router.go", "src/middleware.go"],
            candidates={"router": ["src/router.go"], "middleware": ["src/middleware.go"]},
            api_key="sk-ant-test",
            is_library=True,
        )

        sent_system = mock_client.messages.create.call_args.kwargs["system"]
        assert "5-15 public modules" in sent_system
        assert "consumable library" in sent_system

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_deep_scan_uses_package_prompt_when_requested(
        self, mock_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        deep_scan(
            files=["src/auth/login.ts", "src/auth/signup.ts"],
            candidates={"auth": ["src/auth/login.ts", "src/auth/signup.ts"]},
            api_key="sk-ant-test",
            package_mode=True,
            package_name="auth",
        )

        sent_system = mock_client.messages.create.call_args.kwargs["system"]
        assert "1-8 features" in sent_system
        assert "`auth`" in sent_system


# ── D6: deep_scan_workspace orchestration ───────────────────────────────


def _ws(packages: list[WorkspacePackage], root_files: list[str] | None = None) -> WorkspaceInfo:
    """Build a WorkspaceInfo for tests."""
    return WorkspaceInfo(
        detected=True,
        manager="pnpm",
        packages=packages,
        root_files=root_files or [],
    )


def _pkg(name: str, path: str, n_files: int) -> WorkspacePackage:
    """Build a WorkspacePackage with N synthetic files under its path."""
    files = [f"{path}/src/file{i}.ts" for i in range(n_files)]
    return WorkspacePackage(name=name, path=path, files=files)


class TestDeepScanWorkspace:
    def setup_method(self) -> None:
        scanner._last_scan_result = None

    def test_returns_none_for_empty_workspace(self) -> None:
        ws = _ws(packages=[])
        assert deep_scan_workspace(ws, api_key="sk-ant-test") is None

    def test_returns_none_when_workspace_info_is_none(self) -> None:
        assert deep_scan_workspace(None, api_key="sk-ant-test") is None

    def test_skips_test_packages_entirely(self) -> None:
        """Packages named tests/e2e/__tests__ never reach the LLM."""
        ws = _ws(packages=[
            _pkg("tests", "packages/tests", 100),
            _pkg("e2e", "packages/e2e", 50),
            _pkg("auth", "packages/auth", 5),  # small → no LLM call
        ])
        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
        ) as mock_ds:
            result = deep_scan_workspace(ws, api_key="sk-ant-test")

        assert mock_ds.call_count == 0
        assert result is not None
        assert "tests" not in result
        assert "e2e" not in result
        assert "auth" in result

    def test_pools_example_packages_into_examples_feature(self) -> None:
        """example-* / demo-* / tutorial-* packages share one bucket."""
        ws = _ws(packages=[
            _pkg("example-todos", "examples/todos", 20),
            _pkg("demo-app", "examples/demo", 15),
            _pkg("auth", "packages/auth", 5),
        ])
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            result = deep_scan_workspace(ws, api_key="sk-ant-test")

        assert mock_ds.call_count == 0
        assert "examples" in result
        assert len(result["examples"]) == 35  # 20 + 15
        assert "example-todos" not in result
        assert "demo-app" not in result

    def test_small_package_becomes_one_feature_without_llm(self) -> None:
        """Packages under min_files_for_llm get the package name verbatim."""
        ws = _ws(packages=[_pkg("cli", "packages/cli", 8)])
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            result = deep_scan_workspace(
                ws, api_key="sk-ant-test", min_files_for_llm=30,
            )

        assert mock_ds.call_count == 0
        assert "cli" in result
        assert len(result["cli"]) == 8

    def test_large_package_calls_deep_scan_in_package_mode(self) -> None:
        """≥ min_files_for_llm triggers a per-package deep_scan call."""
        ws = _ws(packages=[_pkg("dashboard", "apps/dashboard", 100)])

        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            mock_ds.return_value = _ds_result(
                {"dashboard": [f"src/file{i}.ts" for i in range(100)]}
            )
            deep_scan_workspace(ws, api_key="sk-ant-test", min_files_for_llm=30)

        assert mock_ds.call_count == 1
        kwargs = mock_ds.call_args.kwargs
        assert kwargs["package_mode"] is True
        assert kwargs["package_name"] == "dashboard"

    def test_single_subfeature_collapses_to_bare_package_name(self) -> None:
        """deep_scan returns one feature → result keyed by package name only.

        Avoids ugly names like ``auth/auth``."""
        ws = _ws(packages=[_pkg("auth", "packages/auth", 50)])
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            mock_ds.return_value = _ds_result({
                "auth": [f"src/file{i}.ts" for i in range(50)],
            })
            result = deep_scan_workspace(ws, api_key="sk-ant-test", min_files_for_llm=30)

        assert "auth" in result
        assert "auth/auth" not in result
        # Files are repo-relative again
        assert result["auth"][0].startswith("packages/auth/")

    def test_multi_subfeature_prefixes_with_package_name(self) -> None:
        """When the LLM returns N>1 features, they get ``{pkg}/{name}`` keys."""
        ws = _ws(packages=[_pkg("api", "apps/api", 200)])
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            mock_ds.return_value = _ds_result({
                "auth": [f"src/file{i}.ts" for i in range(60)],
                "billing": [f"src/file{i}.ts" for i in range(60, 130)],
                "webhooks": [f"src/file{i}.ts" for i in range(130, 200)],
            })
            result = deep_scan_workspace(ws, api_key="sk-ant-test", min_files_for_llm=30)

        assert "api/auth" in result
        assert "api/billing" in result
        assert "api/webhooks" in result
        # All paths should be re-prefixed with the package path
        for files in result.values():
            for f in files:
                assert f.startswith("apps/api/")

    def test_per_package_shared_infra_merges_into_global(self) -> None:
        """If a package call returns a 'shared-infra' sub-feature, it goes
        into the global shared-infra bucket, not pkg.name/shared-infra."""
        ws = _ws(
            packages=[_pkg("api", "apps/api", 200)],
            root_files=["package.json", ".github/workflows/ci.yml"],
        )
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            mock_ds.return_value = _ds_result({
                "auth": [f"src/file{i}.ts" for i in range(180)],
                "shared-infra": [f"src/file{i}.ts" for i in range(180, 200)],
            })
            result = deep_scan_workspace(ws, api_key="sk-ant-test", min_files_for_llm=30)

        assert "api/shared-infra" not in result
        assert "shared-infra" in result
        # Package's shared-infra files + the workspace root files
        assert any("apps/api/" in f for f in result["shared-infra"])
        assert "package.json" in result["shared-infra"]
        assert ".github/workflows/ci.yml" in result["shared-infra"]

    def test_root_files_become_shared_infra(self) -> None:
        ws = _ws(
            packages=[_pkg("auth", "packages/auth", 5)],
            root_files=["pnpm-workspace.yaml", "turbo.json"],
        )
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            result = deep_scan_workspace(ws, api_key="sk-ant-test")

        assert mock_ds.call_count == 0
        assert "shared-infra" in result
        assert sorted(result["shared-infra"]) == ["pnpm-workspace.yaml", "turbo.json"]

    def test_threads_tracker_through_all_calls(self) -> None:
        """The same CostTracker instance is passed to every per-package call."""
        ws = _ws(packages=[
            _pkg("api", "apps/api", 100),
            _pkg("web", "apps/web", 80),
        ])
        tracker = CostTracker()
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            mock_ds.return_value = _ds_result({"x": ["src/file0.ts"]})
            deep_scan_workspace(
                ws, api_key="sk-ant-test", tracker=tracker, min_files_for_llm=30,
            )

        assert mock_ds.call_count == 2
        for call in mock_ds.call_args_list:
            assert call.kwargs["tracker"] is tracker

    def test_budget_exceeded_propagates(self) -> None:
        """A BudgetExceeded from any per-package call must abort the scan."""
        ws = _ws(packages=[
            _pkg("api", "apps/api", 200),
            _pkg("web", "apps/web", 200),
        ])

        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            mock_ds.side_effect = BudgetExceeded(spent=1.0, limit=0.5)
            with pytest.raises(BudgetExceeded):
                deep_scan_workspace(
                    ws, api_key="sk-ant-test", min_files_for_llm=30,
                )

        # The largest package is processed first; sort kicks in regardless of order
        assert mock_ds.call_count == 1

    def test_processes_largest_packages_first(self) -> None:
        """Sort: biggest package first so budget aborts cheaper later calls."""
        ws = _ws(packages=[
            _pkg("small", "packages/small", 40),
            _pkg("huge", "packages/huge", 500),
            _pkg("medium", "packages/medium", 100),
        ])
        seen_order: list[str] = []
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            def capture(*args, **kwargs):
                seen_order.append(kwargs["package_name"])
                return _ds_result({kwargs["package_name"]: ["src/file0.ts"]})
            mock_ds.side_effect = capture
            deep_scan_workspace(ws, api_key="sk-ant-test", min_files_for_llm=30)

        assert seen_order == ["huge", "medium", "small"]

    def test_falls_back_when_deep_scan_returns_none(self) -> None:
        """If a per-package LLM call fails (returns None), the package
        becomes one feature instead of disappearing from the map."""
        ws = _ws(packages=[_pkg("api", "apps/api", 100)])
        with patch("faultline.llm.sonnet_scanner.deep_scan") as mock_ds:
            mock_ds.return_value = None
            result = deep_scan_workspace(ws, api_key="sk-ant-test", min_files_for_llm=30)

        assert "api" in result
        assert len(result["api"]) == 100

    def test_documenso_shape(self) -> None:
        """End-to-end shape: documenso-style monorepo. 5 packages, mix of
        skip/small/large. Verifies the orchestrator produces ≤ ~10 features
        without timing out (vs the 91-feature legacy baseline)."""
        ws = _ws(
            packages=[
                _pkg("web", "apps/web", 600),               # large → LLM
                _pkg("marketing", "apps/marketing", 200),   # large → LLM
                _pkg("ee", "packages/ee", 150),             # large → LLM
                _pkg("lib", "packages/lib", 12),            # small → 1 feature
                _pkg("tsconfig", "packages/tsconfig", 4),   # small → 1 feature
                _pkg("tests", "packages/tests", 80),        # skip
                _pkg("example-app", "examples/app", 30),    # examples
            ],
            root_files=["turbo.json", "package.json"],
        )

        def fake_deep_scan(files, candidates, **kwargs):
            name = kwargs["package_name"]
            # Pretend the LLM returned one feature for each large package.
            return _ds_result({name: list(files)})

        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
            side_effect=fake_deep_scan,
        ) as mock_ds:
            result = deep_scan_workspace(ws, api_key="sk-ant-test", min_files_for_llm=30)

        # 3 large packages → 3 LLM calls (web, marketing, ee)
        assert mock_ds.call_count == 3
        # No test/example packages survive as features
        assert "tests" not in result
        assert "example-app" not in result
        # Pooled examples bucket present
        assert "examples" in result
        # All 3 large packages, both small packages, examples, shared-infra
        expected_keys = {
            "web", "marketing", "ee",
            "lib", "tsconfig",
            "examples", "shared-infra",
        }
        assert set(result.keys()) == expected_keys
        # Way under the 91-feature legacy baseline.
        assert len(result) <= 10


# ── D5: commit context builder + injection ──────────────────────────────


class _FakeCommit:
    """Light stand-in for the Commit pydantic model used in tests."""
    def __init__(self, date, files_changed):
        self.date = date
        self.files_changed = files_changed


class TestBuildCommitContext:
    def test_returns_none_for_empty_commits(self) -> None:
        assert build_commit_context(None) is None
        assert build_commit_context([]) is None

    def test_excludes_commits_outside_window(self) -> None:
        """Commits older than ``days`` must not contribute."""
        old = _FakeCommit(
            date=datetime.now(timezone.utc) - timedelta(days=120),
            files_changed=["src/old/legacy.ts"],
        )
        recent = _FakeCommit(
            date=datetime.now(timezone.utc) - timedelta(days=5),
            files_changed=["src/auth/login.ts"],
        )
        ctx = build_commit_context([old, recent], days=90)
        assert ctx is not None
        assert "src/auth/login.ts" in ctx
        assert "legacy" not in ctx

    def test_aggregates_per_file_and_per_dir(self) -> None:
        """Both files and parent dirs should appear, dirs marked with /."""
        commits = [
            _FakeCommit(
                date=datetime.now(timezone.utc) - timedelta(days=1),
                files_changed=["src/auth/login.ts", "src/auth/signup.ts"],
            ),
            _FakeCommit(
                date=datetime.now(timezone.utc) - timedelta(days=2),
                files_changed=["src/auth/login.ts"],
            ),
        ]
        ctx = build_commit_context(commits, days=30)
        assert ctx is not None
        # Files
        assert "src/auth/login.ts  2 commits" in ctx
        assert "src/auth/signup.ts  1 commits" in ctx
        # Directory rolled up — marked with trailing slash
        assert "src/auth/  3 commits" in ctx

    def test_top_n_caps_output_size(self) -> None:
        commits = [
            _FakeCommit(
                date=datetime.now(timezone.utc) - timedelta(days=1),
                files_changed=[f"src/feat{i}/index.ts"],
            )
            for i in range(50)
        ]
        ctx = build_commit_context(commits, top_n=5, days=30)
        assert ctx is not None
        assert len(ctx.splitlines()) == 5

    def test_handles_naive_datetimes(self) -> None:
        """gitpython sometimes hands us naive datetimes; must not crash."""
        naive = _FakeCommit(
            date=datetime.utcnow() - timedelta(days=1),
            files_changed=["src/x.ts"],
        )
        ctx = build_commit_context([naive], days=30)
        assert ctx is not None
        assert "src/x.ts" in ctx

    def test_returns_none_when_only_old_commits(self) -> None:
        old = _FakeCommit(
            date=datetime.now(timezone.utc) - timedelta(days=400),
            files_changed=["src/old.ts"],
        )
        assert build_commit_context([old], days=90) is None


class TestCommitContextInjection:
    def setup_method(self) -> None:
        scanner._last_scan_result = None

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_no_context_means_no_recent_activity_section(
        self, mock_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        deep_scan(
            files=["src/a.ts"],
            candidates={"auth": ["src/a.ts", "src/b.ts"]},
            api_key="sk-ant-test",
        )
        sent_user = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "## Recent activity" not in sent_user

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_context_appears_in_user_prompt(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        deep_scan(
            files=["src/a.ts"],
            candidates={"auth": ["src/a.ts", "src/b.ts"]},
            api_key="sk-ant-test",
            commit_context="src/auth/login.ts  12 commits\nsrc/auth/  18 commits",
        )
        sent_user = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "## Recent activity" in sent_user
        assert "src/auth/login.ts  12 commits" in sent_user
        assert "src/auth/  18 commits" in sent_user


# ── D10: DeepScanResult dataclass and dict shims ────────────────────────


class TestDeepScanResultShims:
    def test_features_attribute_is_primary(self) -> None:
        r = DeepScanResult(features={"auth": ["a.ts"]})
        assert r.features == {"auth": ["a.ts"]}

    def test_dict_membership(self) -> None:
        r = DeepScanResult(features={"auth": ["a.ts"]})
        assert "auth" in r
        assert "billing" not in r

    def test_dict_indexing(self) -> None:
        r = DeepScanResult(features={"auth": ["a.ts", "b.ts"]})
        assert r["auth"] == ["a.ts", "b.ts"]

    def test_dict_iteration(self) -> None:
        r = DeepScanResult(features={"a": ["1"], "b": ["2"]})
        assert sorted(list(r)) == ["a", "b"]
        assert sorted(r.keys()) == ["a", "b"]

    def test_dict_items_and_values(self) -> None:
        r = DeepScanResult(features={"a": ["1"], "b": ["2", "3"]})
        assert dict(r.items()) == {"a": ["1"], "b": ["2", "3"]}
        assert sorted([len(v) for v in r.values()]) == [1, 2]

    def test_len(self) -> None:
        assert len(DeepScanResult(features={})) == 0
        assert len(DeepScanResult(features={"a": ["1"], "b": ["2"]})) == 2

    def test_truthiness(self) -> None:
        assert not DeepScanResult(features={})
        assert DeepScanResult(features={"a": ["1"]})

    def test_get_with_default(self) -> None:
        r = DeepScanResult(features={"auth": ["a.ts"]})
        assert r.get("auth") == ["a.ts"]
        assert r.get("missing", "fallback") == "fallback"


class TestDeepScanReturnsDeepScanResult:
    def setup_method(self) -> None:
        scanner._last_scan_result = None

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_deep_scan_returns_dataclass(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        result = deep_scan(
            files=["src/auth/a.ts", "src/auth/b.ts"],
            candidates={"auth": ["src/auth/a.ts", "src/auth/b.ts"]},
            api_key="sk-ant-test",
        )
        assert isinstance(result, DeepScanResult)
        assert "auth" in result
        # Description from minimal ops json: name="auth", description="d"
        assert result.descriptions.get("auth") == "d"

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_deep_scan_includes_cost_summary_when_tracker_used(
        self, mock_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 1000, 500,
        )

        tracker = CostTracker()
        result = deep_scan(
            files=["src/a.ts"],
            candidates={"auth": ["src/a.ts", "src/b.ts"]},
            api_key="sk-ant-test",
            tracker=tracker,
        )
        assert isinstance(result, DeepScanResult)
        assert result.cost_summary is not None

    @patch("faultline.llm.sonnet_scanner.anthropic.Anthropic")
    def test_deep_scan_no_tracker_means_none_cost(
        self, mock_cls: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_llm_response(
            _MINIMAL_OPS_JSON, 100, 50,
        )

        result = deep_scan(
            files=["src/a.ts"],
            candidates={"auth": ["src/a.ts", "src/b.ts"]},
            api_key="sk-ant-test",
        )
        assert result.cost_summary is None

    def test_workspace_returns_dataclass(self) -> None:
        ws = _ws(packages=[_pkg("auth", "packages/auth", 5)])
        result = deep_scan_workspace(ws, api_key="sk-ant-test")
        assert isinstance(result, DeepScanResult)

    def test_workspace_aggregates_flows_from_per_package_calls(self) -> None:
        """Per-package DeepScanResult flows must surface in the merged result
        under the final feature key (with package prefix when split)."""
        ws = _ws(packages=[_pkg("api", "apps/api", 200)])

        per_pkg = DeepScanResult(
            features={
                "auth": [f"src/file{i}.ts" for i in range(120)],
                "billing": [f"src/file{i}.ts" for i in range(120, 200)],
            },
            flows={
                "auth": ["login-flow", "signup-flow"],
                "billing": ["checkout-flow"],
            },
            descriptions={
                "auth": "Authentication and session management",
                "billing": "Subscriptions and invoicing",
            },
        )
        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
            return_value=per_pkg,
        ):
            result = deep_scan_workspace(
                ws, api_key="sk-ant-test", min_files_for_llm=30,
            )

        assert isinstance(result, DeepScanResult)
        # Multi-feature → keys are prefixed
        assert "api/auth" in result.flows
        assert "api/billing" in result.flows
        assert result.flows["api/auth"] == ["login-flow", "signup-flow"]
        assert "api/auth" in result.descriptions

    def test_workspace_parallel_submits_all_large_packages(self) -> None:
        """Day 12: parallel path submits every large package exactly once
        to deep_scan, and small/test/example packages bypass the pool."""
        ws = _ws(
            packages=[
                _pkg("api", "apps/api", 150),
                _pkg("web", "apps/web", 100),
                _pkg("ee", "packages/ee", 80),
                _pkg("tiny", "packages/tiny", 5),          # under floor
                _pkg("tests", "packages/tests", 60),        # test filter
                _pkg("example-a", "examples/a", 50),        # example pool
            ],
        )
        call_names: list[str] = []

        def fake_deep_scan(files, candidates, **kwargs):
            call_names.append(kwargs["package_name"])
            return _ds_result({kwargs["package_name"]: list(files)})

        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
            side_effect=fake_deep_scan,
        ) as mock_ds:
            result = deep_scan_workspace(
                ws, api_key="sk-ant-test", min_files_for_llm=30,
            )

        assert mock_ds.call_count == 3
        assert sorted(call_names) == ["api", "ee", "web"]
        # All three large packages + tiny + examples + no tests
        assert "api" in result.features
        assert "web" in result.features
        assert "ee" in result.features
        assert "tiny" in result.features
        assert "examples" in result.features
        assert "tests" not in result.features

    def test_workspace_parallel_max_workers_1_serializes(self) -> None:
        """max_workers=1 runs one LLM call at a time but produces the
        same final result as the default."""
        ws = _ws(
            packages=[
                _pkg("api", "apps/api", 100),
                _pkg("web", "apps/web", 100),
            ],
        )
        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
        ) as mock_ds:
            mock_ds.side_effect = lambda *args, **kw: _ds_result(
                {kw["package_name"]: ["src/file0.ts"]}
            )
            result = deep_scan_workspace(
                ws, api_key="sk-ant-test", min_files_for_llm=30, max_workers=1,
            )
        assert mock_ds.call_count == 2
        assert set(result.keys()) == {"api", "web"}

    def test_workspace_parallel_budget_exceeded_propagates(self) -> None:
        """A BudgetExceeded from any worker tears down the whole scan."""
        ws = _ws(
            packages=[
                _pkg("api", "apps/api", 100),
                _pkg("web", "apps/web", 100),
                _pkg("ee", "packages/ee", 100),
            ],
        )
        call_count = {"n": 0}

        def fake(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise BudgetExceeded(spent=1.0, limit=0.5)
            return _ds_result({kwargs["package_name"]: ["src/file0.ts"]})

        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
            side_effect=fake,
        ):
            with pytest.raises(BudgetExceeded):
                deep_scan_workspace(
                    ws, api_key="sk-ant-test", min_files_for_llm=30,
                )

    def test_workspace_parallel_deterministic_ordering(self) -> None:
        """Two runs with the same inputs must produce identical ordered
        feature lists even when futures complete out of order."""
        import time

        ws = _ws(
            packages=[
                _pkg("alpha", "packages/alpha", 150),
                _pkg("bravo", "packages/bravo", 100),
                _pkg("charlie", "packages/charlie", 50),
            ],
        )

        def slow_fake(files, candidates, **kwargs):
            # Inject variable delay so completion order differs from
            # submission order.
            name = kwargs["package_name"]
            delay = {"alpha": 0.03, "bravo": 0.01, "charlie": 0.02}[name]
            time.sleep(delay)
            return _ds_result({name: list(files)})

        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
            side_effect=slow_fake,
        ):
            r1 = deep_scan_workspace(
                ws, api_key="sk-ant-test", min_files_for_llm=30,
            )
        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
            side_effect=slow_fake,
        ):
            r2 = deep_scan_workspace(
                ws, api_key="sk-ant-test", min_files_for_llm=30,
            )

        assert list(r1.features.keys()) == list(r2.features.keys())
        # Sort is (-size, name); alpha is largest, then bravo, then charlie
        assert list(r1.features.keys()) == ["alpha", "bravo", "charlie"]

    def test_workspace_collapsed_single_feature_keeps_flows(self) -> None:
        """Single-feature collapse: flows attached under the bare package
        name, not the LLM-returned sub-feature name."""
        ws = _ws(packages=[_pkg("auth", "packages/auth", 60)])
        per_pkg = DeepScanResult(
            features={"authentication": [f"src/file{i}.ts" for i in range(60)]},
            flows={"authentication": ["login-flow"]},
            descriptions={"authentication": "Login and signup"},
        )
        with patch(
            "faultline.llm.sonnet_scanner.deep_scan",
            return_value=per_pkg,
        ):
            result = deep_scan_workspace(
                ws, api_key="sk-ant-test", min_files_for_llm=30,
            )

        assert "auth" in result.features
        assert result.flows.get("auth") == ["login-flow"]
        assert result.descriptions.get("auth") == "Login and signup"
