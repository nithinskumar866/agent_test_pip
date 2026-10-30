"""Tests for structural assertions (raise on failure) and their bool predicates."""

import pytest

from agenttest import run, tool
from agenttest.assertions import structural as s
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


def _result(**overrides):
    fields = dict(success=True, output="Fallback response", steps_taken=1, tools_called=[],
                  tokens_used=10, latency_ms=50.0, gracefully_degraded=False)
    fields.update(overrides)
    return AgentResult(**fields)


def test_word_count_between():
    assert assert_word_count_between("Hello world from agenttest", 2, 5) is True
    with pytest.raises(AssertionError, match="Expected 5–10 words, got 2"):
        assert_word_count_between("Hello world", 5, 10)
    assert s.word_count_between("Hello world", 5, 10) is False


def test_steps_at_most():
    assert assert_steps_at_most(3, 5) is True
    with pytest.raises(AssertionError, match="at most 5 steps, agent took 10"):
        assert_steps_at_most(10, 5)
    assert s.steps_at_most(10, 5) is False


def test_tool_was_called():
    tools = ["search", "calculator"]
    assert assert_tool_was_called(tools, "search") is True
    with pytest.raises(AssertionError, match=r"'database' to be called; tools called: \['search', 'calculator'\]"):
        assert_tool_was_called(tools, "database")
    assert s.tool_was_called(tools, "database") is False


def test_tool_was_not_called():
    tools = ["search", "calculator"]
    assert assert_tool_was_not_called(tools, "database") is True
    with pytest.raises(AssertionError, match="'search' not to be called; it was called 1 time"):
        assert_tool_was_not_called(tools, "search")
    assert s.tool_was_not_called(tools, "search") is False


def test_latency_under():
    assert assert_latency_under(150.0, 500.0) is True
    with pytest.raises(AssertionError, match="under 500 ms, got 600.0 ms"):
        assert_latency_under(600.0, 500.0)
    assert s.latency_under(600.0, 500.0) is False


def test_tokens_under():
    assert assert_tokens_under(120, 200) is True
    with pytest.raises(AssertionError, match="fewer than 200 tokens, used 250"):
        assert_tokens_under(250, 200)
    assert s.tokens_under(250, 200) is False


def test_output_contains():
    assert assert_output_contains("The capital of France is Paris.", "Paris") is True
    with pytest.raises(AssertionError, match="contain 'London'"):
        assert_output_contains("The capital of France is Paris.", "London")
    assert s.output_contains("The capital of France is Paris.", "London") is False


def test_output_matches_regex():
    assert assert_output_matches_regex("Score: 95%", r"Score: \d+%") is True
    with pytest.raises(AssertionError, match="Score: N/A"):
        assert_output_matches_regex("Score: N/A", r"Score: \d+%")
    assert s.output_matches_regex("Score: N/A", r"Score: \d+%") is False


def test_output_is_valid_json():
    assert assert_output_is_valid_json('{"key": "value", "status": 200}') is True
    with pytest.raises(AssertionError, match="Expected valid JSON output"):
        assert_output_is_valid_json('{"key": invalid json}')
    assert s.output_is_valid_json('{"key": invalid json}') is False


def test_gracefully_degraded():
    res_degraded = _result(gracefully_degraded=True)
    res_normal = _result(output="Normal response")
    assert assert_gracefully_degraded(res_degraded) is True
    with pytest.raises(AssertionError, match="no tool failed during the run"):
        assert_gracefully_degraded(res_normal)
    assert s.gracefully_degraded(res_normal) is False


# --- new: messages and AgentResult inputs ------------------------------------

def test_failure_message_includes_agent_output():
    result = _result(output="I am not sure.", steps_taken=7)
    with pytest.raises(AssertionError) as exc_info:
        assert_steps_at_most(result, 3)
    assert "output: 'I am not sure.'" in str(exc_info.value)


def test_long_output_is_truncated_in_message():
    with pytest.raises(AssertionError) as exc_info:
        assert_output_contains("word " * 500, "missing")
    message = str(exc_info.value)
    assert "more chars]" in message
    assert len(message) < 600


def test_assertions_accept_agent_results():
    result = _result(output='{"a": 1}', tools_called=["search"], tokens_used=5, latency_ms=10.0)
    assert assert_tool_was_called(result, "search")
    assert assert_tokens_under(result, 6)
    assert assert_latency_under(result, 11)
    assert assert_output_is_valid_json(result)


def test_gracefully_degraded_explains_a_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @tool
    def flaky():
        raise TimeoutError("down")

    result = run(lambda: flaky())
    with pytest.raises(AssertionError, match="the agent crashed: TimeoutError: down"):
        assert_gracefully_degraded(result)


def test_gracefully_degraded_explains_missing_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @tool
    def flaky():
        raise TimeoutError("down")

    def agent():
        try:
            flaky()
        except TimeoutError:
            return ""

    with pytest.raises(AssertionError, match=r"tools \['flaky'\] failed but the agent produced no output"):
        assert_gracefully_degraded(run(agent))
