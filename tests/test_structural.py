"""Tests for pure Python structural assertion functions."""

from agenttest.assertions.structural import (
    assert_gracefully_degraded,
    assert_latency_under,
    assert_output_contains,
    assert_output_is_valid_json,
    assert_output_matches_regex,
    assert_steps_at_most,
    assert_tokens_under,
    assert_tool_was_called,
    assert_tool_was_not_called,
    assert_word_count_between,
)
from agenttest.core.result import AgentResult


def test_word_count_between():
    assert assert_word_count_between("Hello world from agenttest", 2, 5) is True
    assert assert_word_count_between("Hello world", 5, 10) is False


def test_steps_at_most():
    assert assert_steps_at_most(3, 5) is True
    assert assert_steps_at_most(10, 5) is False


def test_tool_was_called():
    tools = ["search", "calculator"]
    assert assert_tool_was_called(tools, "search") is True
    assert assert_tool_was_called(tools, "database") is False


def test_tool_was_not_called():
    tools = ["search", "calculator"]
    assert assert_tool_was_not_called(tools, "database") is True
    assert assert_tool_was_not_called(tools, "search") is False


def test_latency_under():
    assert assert_latency_under(150.0, 500.0) is True
    assert assert_latency_under(600.0, 500.0) is False


def test_tokens_under():
    assert assert_tokens_under(120, 200) is True
    assert assert_tokens_under(250, 200) is False


def test_output_contains():
    assert assert_output_contains("The capital of France is Paris.", "Paris") is True
    assert assert_output_contains("The capital of France is Paris.", "London") is False


def test_output_matches_regex():
    assert assert_output_matches_regex("Score: 95%", r"Score: \d+%") is True
    assert assert_output_matches_regex("Score: N/A", r"Score: \d+%") is False


def test_output_is_valid_json():
    assert assert_output_is_valid_json('{"key": "value", "status": 200}') is True
    assert assert_output_is_valid_json('{"key": invalid json}') is False


def test_gracefully_degraded():
    res_degraded = AgentResult(
        success=True,
        output="Fallback response",
        steps_taken=1,
        tools_called=[],
        tokens_used=10,
        latency_ms=50.0,
        gracefully_degraded=True,
    )
    res_normal = AgentResult(
        success=True,
        output="Normal response",
        steps_taken=1,
        tools_called=[],
        tokens_used=10,
        latency_ms=50.0,
        gracefully_degraded=False,
    )
    assert assert_gracefully_degraded(res_degraded) is True
    assert assert_gracefully_degraded(res_normal) is False
