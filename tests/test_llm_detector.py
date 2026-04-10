from unittest.mock import MagicMock, patch

import anthropic
import pytest

from faultline.llm.detector import (
    _FeatureDetectionResponse,
    _FeatureFileMapping,
    _MAX_FILES_FOR_DETECTION,
    detect_features_llm,
)


def make_mock_response(features_data: list[dict]) -> MagicMock:
    """Helper to build a mock parsed response from the Anthropic client."""
    parsed = _FeatureDetectionResponse(
        features=[
            _FeatureFileMapping(feature_name=f["name"], files=f["files"])
            for f in features_data
        ]
    )
    mock_response = MagicMock()
    mock_response.parsed_output = parsed
    return mock_response


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_returns_empty_when_no_api_key(mock_cls, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = detect_features_llm(["src/auth.py"])

    assert result == {}
    mock_cls.assert_not_called()


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_returns_empty_when_files_empty(mock_cls):
    result = detect_features_llm([], api_key="sk-ant-test")

    assert result == {}
    mock_cls.assert_not_called()


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_returns_feature_mapping_on_success(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.return_value = make_mock_response([
        {"name": "user-auth", "files": ["src/auth/login.py", "src/auth/session.py"]},
        {"name": "payments", "files": ["src/payments/stripe.py"]},
    ])

    result = detect_features_llm(
        ["src/auth/login.py", "src/auth/session.py", "src/payments/stripe.py"],
        api_key="sk-ant-test",
    )

    assert result == {
        "user-auth": ["src/auth/login.py", "src/auth/session.py"],
        "payments": ["src/payments/stripe.py"],
    }


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_fallback_on_authentication_error(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.side_effect = anthropic.AuthenticationError(
        message="Invalid API key",
        response=MagicMock(status_code=401),
        body={},
    )

    result = detect_features_llm(["src/auth.py"], api_key="sk-ant-test")

    assert result == {}


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_fallback_on_rate_limit_error(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.side_effect = anthropic.RateLimitError(
        message="Rate limit",
        response=MagicMock(status_code=429),
        body={},
    )

    result = detect_features_llm(["src/auth.py"], api_key="sk-ant-test")

    assert result == {}


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_fallback_on_api_connection_error(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.side_effect = anthropic.APIConnectionError(
        request=MagicMock(),
    )

    result = detect_features_llm(["src/auth.py"], api_key="sk-ant-test")

    assert result == {}


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_filters_unknown_files_from_response(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.return_value = make_mock_response([
        {
            "name": "user-auth",
            "files": ["src/auth/login.py", "src/auth/unknown.py"],
        },
    ])

    result = detect_features_llm(
        ["src/auth/login.py"],
        api_key="sk-ant-test",
    )

    assert result == {"user-auth": ["src/auth/login.py"]}
    assert "src/auth/unknown.py" not in result.get("user-auth", [])


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_collapses_to_dirs_for_large_repos(mock_cls):
    """For repos > _DIR_COLLAPSE_THRESHOLD files, sends unique dirs instead of individual files."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.return_value = make_mock_response([])

    # 600 files spread across 10 feature directories
    all_files = [
        f"feature_{i}/component_{j}.tsx"
        for i in range(10)
        for j in range(60)
    ]
    detect_features_llm(all_files, api_key="sk-ant-test")

    call_args = mock_client.messages.parse.call_args
    user_message = call_args.kwargs["messages"][0]["content"]
    sent_lines = [line for line in user_message.split("\n") if line.strip()]

    # Should send directories (feature_0, feature_1, ...) not 600 individual files
    assert len(sent_lines) < len(all_files), "Should collapse to dirs, not send all files"
    # Dir-collapse format: "  feature_0 → component_0.tsx, component_1.tsx, ..."
    assert any("feature_0" in line and "→" in line for line in sent_lines)


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_reads_api_key_from_env_var(mock_cls, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.return_value = make_mock_response([
        {"name": "core", "files": ["src/main.py"]},
    ])

    result = detect_features_llm(["src/main.py"])

    mock_cls.assert_called_once_with(api_key="sk-ant-env-key")
    assert result == {"core": ["src/main.py"]}


@patch("faultline.llm.detector.anthropic.Anthropic")
def test_features_with_no_valid_files_are_excluded(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.parse.return_value = make_mock_response([
        {"name": "user-auth", "files": ["src/auth/login.py"]},
        {"name": "ghost-feature", "files": ["nonexistent/a.py", "nonexistent/b.py"]},
    ])

    result = detect_features_llm(
        ["src/auth/login.py"],
        api_key="sk-ant-test",
    )

    assert "user-auth" in result
    assert "ghost-feature" not in result
