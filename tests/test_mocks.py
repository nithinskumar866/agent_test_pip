"""Tests for tool mocking context managers and registry."""

from agenttest.mocks import (
    MOCK_REGISTRY,
    mock_tool_failure,
    mock_tool_latency,
    mock_tool_response,
)


def test_mock_tool_failure_adds_to_registry():
    assert "web_search" not in MOCK_REGISTRY
    with mock_tool_failure("web_search", "Custom failure"):
        assert "web_search" in MOCK_REGISTRY
        assert MOCK_REGISTRY["web_search"]["type"] == "failure"
        assert MOCK_REGISTRY["web_search"]["error"] == "Custom failure"


def test_mock_tool_failure_cleans_up_after_exit():
    assert "temp_tool" not in MOCK_REGISTRY
    with mock_tool_failure("temp_tool"):
        assert "temp_tool" in MOCK_REGISTRY
    assert "temp_tool" not in MOCK_REGISTRY


def test_mock_tool_response_adds_to_registry():
    with mock_tool_response("calculator", {"result": 42}):
        assert "calculator" in MOCK_REGISTRY
        assert MOCK_REGISTRY["calculator"]["type"] == "response"
        assert MOCK_REGISTRY["calculator"]["data"] == {"result": 42}
    assert "calculator" not in MOCK_REGISTRY


def test_mock_tool_latency_adds_to_registry():
    with mock_tool_latency("database", 500):
        assert "database" in MOCK_REGISTRY
        assert MOCK_REGISTRY["database"]["type"] == "latency"
        assert MOCK_REGISTRY["database"]["delay"] == 0.5
    assert "database" not in MOCK_REGISTRY


def test_nested_mocks_work_correctly():
    with mock_tool_failure("tool_a"):
        with mock_tool_response("tool_b", "ok"):
            assert "tool_a" in MOCK_REGISTRY
            assert "tool_b" in MOCK_REGISTRY
        assert "tool_b" not in MOCK_REGISTRY
        assert "tool_a" in MOCK_REGISTRY
    assert "tool_a" not in MOCK_REGISTRY
