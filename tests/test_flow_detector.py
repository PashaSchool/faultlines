"""Tests for LLM-based flow detection within features."""
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from faultline.analyzer.ast_extractor import FileSignature
from faultline.llm.flow_detector import (
    _FlowDetectionResponse,
    _FlowFileMapping,
    _filter_valid_files,
    _build_signatures_text,
    detect_flows_llm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_flow_response(flows_data: list[dict]) -> MagicMock:
    """Builds a mock parsed response from the Anthropic client."""
    parsed = _FlowDetectionResponse(
        flows=[
            _FlowFileMapping(flow_name=f["name"], files=f["files"])
            for f in flows_data
        ]
    )
    mock_response = MagicMock()
    mock_response.parsed_output = parsed
    return mock_response


def make_signature(path: str, exports=None, routes=None) -> FileSignature:
    return FileSignature(
        path=path,
        exports=exports or [],
        routes=routes or [],
    )


# ---------------------------------------------------------------------------
# _filter_valid_files
# ---------------------------------------------------------------------------

def test_filter_removes_hallucinated_files():
    flows = [
        _FlowFileMapping(flow_name="login-flow", files=["real.ts", "fake.ts"]),
    ]
    result = _filter_valid_files(flows, allowed_files={"real.ts"})
    assert result[0].files == ["real.ts"]


def test_filter_removes_flow_with_no_valid_files():
    flows = [
        _FlowFileMapping(flow_name="ghost-flow", files=["nonexistent.ts"]),
        _FlowFileMapping(flow_name="real-flow", files=["real.ts"]),
    ]
    result = _filter_valid_files(flows, allowed_files={"real.ts"})
    assert len(result) == 1
    assert result[0].flow_name == "real-flow"


def test_filter_keeps_all_valid_flows():
    flows = [
        _FlowFileMapping(flow_name="login-flow", files=["login.ts"]),
        _FlowFileMapping(flow_name="checkout-flow", files=["checkout.ts"]),
    ]
    result = _filter_valid_files(flows, allowed_files={"login.ts", "checkout.ts"})
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _build_signatures_text
# ---------------------------------------------------------------------------

def test_signatures_text_includes_exports_and_routes():
    sigs = {
        "api/auth.ts": make_signature("api/auth.ts", exports=["login"], routes=["POST /api/login"]),
    }
    text = _build_signatures_text(["api/auth.ts"], sigs)
    assert "login" in text
    assert "POST /api/login" in text


def test_signatures_text_shows_placeholder_for_missing_sig():
    text = _build_signatures_text(["unknown.ts"], {})
    assert "unknown.ts" in text
    assert "no signatures" in text


def test_signatures_text_trims_imports_for_large_features():
    """Features with >30 files should keep relative imports but drop external ones."""
    files = [f"file_{i}.ts" for i in range(35)]
    sigs = {
        "file_0.ts": make_signature("file_0.ts", exports=["Foo"]),
    }
    # Relative import (kept) + external import (dropped)
    sigs["file_0.ts"].imports = ["./bar", "lodash"]

    text = _build_signatures_text(files, sigs)
    assert "./bar" in text        # relative imports kept (flow-critical)
    assert "lodash" not in text   # external imports dropped for large features


# ---------------------------------------------------------------------------
# detect_flows_llm
# ---------------------------------------------------------------------------

@patch("faultline.llm.flow_detector.anthropic.Anthropic")
def test_returns_empty_when_no_api_key(mock_cls, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = detect_flows_llm("payments", ["stripe.ts"], {})
    assert result == []
    mock_cls.assert_not_called()


@patch("faultline.llm.flow_detector.anthropic.Anthropic")
def test_returns_empty_when_no_files(mock_cls):
    result = detect_flows_llm("payments", [], {}, api_key="sk-ant-test")
    assert result == []
    mock_cls.assert_not_called()


@patch("faultline.llm.flow_detector.anthropic.Anthropic")
def test_returns_flow_mappings_on_success(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.return_value = make_mock_flow_response([
        {"name": "checkout-flow", "files": ["checkout.ts", "payment.ts"]},
        {"name": "refund-flow", "files": ["refund.ts"]},
    ])

    result = detect_flows_llm(
        feature_name="payments",
        feature_files=["checkout.ts", "payment.ts", "refund.ts"],
        signatures={},
        api_key="sk-ant-test",
    )

    assert len(result) == 2
    flow_names = [r.flow_name for r in result]
    assert "checkout-flow" in flow_names
    assert "refund-flow" in flow_names


@patch("faultline.llm.flow_detector.anthropic.Anthropic")
def test_fallback_on_authentication_error(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.side_effect = anthropic.AuthenticationError(
        message="Invalid key",
        response=MagicMock(status_code=401),
        body={},
    )
    result = detect_flows_llm("auth", ["login.ts"], {}, api_key="sk-ant-test")
    assert result == []


@patch("faultline.llm.flow_detector.anthropic.Anthropic")
def test_fallback_on_rate_limit_error(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.side_effect = anthropic.RateLimitError(
        message="Rate limited",
        response=MagicMock(status_code=429),
        body={},
    )
    result = detect_flows_llm("auth", ["login.ts"], {}, api_key="sk-ant-test")
    assert result == []


@patch("faultline.llm.flow_detector.anthropic.Anthropic")
def test_hallucinated_files_are_filtered(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.return_value = make_mock_flow_response([
        {"name": "login-flow", "files": ["login.ts", "invented_by_llm.ts"]},
    ])

    result = detect_flows_llm(
        feature_name="auth",
        feature_files=["login.ts"],
        signatures={},
        api_key="sk-ant-test",
    )

    assert len(result) == 1
    assert result[0].files == ["login.ts"]
    assert "invented_by_llm.ts" not in result[0].files
