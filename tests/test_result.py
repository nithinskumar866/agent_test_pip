"""Tests for AgentResult model, multi-provider judge routing, and errors."""

import os
from unittest.mock import patch

import pytest

from agenttest.config import AgentTestConfigError
from agenttest.core.result import AgentResult, _clear_judge_cache, _judge_llm


@pytest.fixture(autouse=True)
def clear_cache():
    _clear_judge_cache()
    yield
    _clear_judge_cache()


def test_agent_result_construction():
    res = AgentResult(
        success=True,
        output="Paris is the capital of France.",
        steps_taken=2,
        tools_called=["search"],
        tokens_used=150,
        latency_ms=450.0,
        gracefully_degraded=False,
    )
    assert res.success is True
    assert res.output == "Paris is the capital of France."
    assert res.steps_taken == 2
    assert res.tools_called == ["search"]
    assert res.tokens_used == 150
    assert res.latency_ms == 450.0
    assert res.gracefully_degraded is False
    assert res.error is None


def test_word_count_between_passes():
    res = AgentResult(
        success=True,
        output="One two three four five",
        steps_taken=1,
        tools_called=[],
        tokens_used=10,
        latency_ms=100.0,
        gracefully_degraded=False,
    )
    assert res.word_count_between(1, 10) is True


def test_word_count_between_fails():
    res = AgentResult(
        success=True,
        output="One two three four five",
        steps_taken=1,
        tools_called=[],
        tokens_used=10,
        latency_ms=100.0,
        gracefully_degraded=False,
    )
    assert res.word_count_between(6, 10) is False
    assert res.word_count_between(1, 4) is False


def test_missing_api_key_raises_config_error():
    res = AgentResult(
        success=True,
        output="Test output",
        steps_taken=1,
        tools_called=[],
        tokens_used=10,
        latency_ms=100.0,
        gracefully_degraded=False,
    )
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(AgentTestConfigError) as exc_info:
            res.contains_fact("Paris")
        assert "ANTHROPIC_API_KEY environment variable not set" in str(exc_info.value)


def test_hallucination_score_missing_key():
    res = AgentResult(
        success=True,
        output="Test output",
        steps_taken=1,
        tools_called=[],
        tokens_used=10,
        latency_ms=100.0,
        gracefully_degraded=False,
    )
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(AgentTestConfigError):
            _ = res.hallucination_score


def test_unknown_judge_provider_raises_error():
    with patch("agenttest.core.result._load_config", return_value={"judge_provider": "unknown_llm"}):
        with pytest.raises(AgentTestConfigError) as exc_info:
            _judge_llm("test prompt")
        assert "Unknown judge_provider: 'unknown_llm'" in str(exc_info.value)


def test_judge_provider_none_raises_helpful_error():
    with patch("agenttest.core.result._load_config", return_value={"judge_provider": "none"}):
        with pytest.raises(AgentTestConfigError) as exc_info:
            _judge_llm("test prompt")
        assert "LLM judge is disabled (judge_provider=none)" in str(exc_info.value)
