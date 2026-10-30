"""Tests for LLM-judged assertions, with the judge replaced by a stub."""

from unittest.mock import patch

import pytest

from agenttest import AgentTestJudgeError
from agenttest.assertions import semantic as sem
from agenttest.core.result import AgentResult, _clear_judge_cache


@pytest.fixture(autouse=True)
def clear_cache():
    _clear_judge_cache()
    yield
    _clear_judge_cache()


def judge(answer):
    """Patch the judge to always return ``answer``; yields the prompts it saw."""
    prompts = []

    def fake(prompt):
        prompts.append(prompt)
        return answer

    return patch("agenttest.assertions.semantic._judge_llm", side_effect=fake), prompts


def _result(output="Paris is the capital of France."):
    return AgentResult(success=True, output=output, steps_taken=1, tools_called=[],
                       tokens_used=0, latency_ms=1.0, gracefully_degraded=False)


@pytest.mark.parametrize(
    ("assertion", "arg"),
    [
        (sem.assert_contains_fact, "Paris"),
        (sem.assert_tone_matches, "professional"),
        (sem.assert_answers_question, "What is the capital?"),
        (sem.assert_language_matches, "English"),
        (sem.assert_eval_prompt, "mentions a city"),
    ],
)
def test_yes_no_assertions_pass_and_fail(assertion, arg):
    patcher, prompts = judge("YES")
    with patcher:
        assert assertion(_result(), arg) is True
    assert arg in prompts[0]
    assert "Paris is the capital of France." in prompts[0]

    patcher, _ = judge("NO, it does not.")
    with patcher:
        with pytest.raises(AssertionError) as exc_info:
            assertion(_result(), arg)
    message = str(exc_info.value)
    assert "judge answered: 'NO, it does not.'" in message
    assert "output: 'Paris is the capital of France.'" in message


@pytest.mark.parametrize("answer", ["yes", "Yes.", "**YES**", "  YES - clearly"])
def test_verdict_parsing_accepts_common_yes_forms(answer):
    patcher, _ = judge(answer)
    with patcher:
        assert sem.contains_fact("text", "claim") is True


@pytest.mark.parametrize("answer", ["Maybe", "I think so", "", "YESTERDAY"])
def test_unparseable_verdict_raises_judge_error(answer):
    patcher, _ = judge(answer)
    with patcher:
        with pytest.raises(AgentTestJudgeError, match="did not answer YES or NO"):
            sem.contains_fact("text", "claim")


@pytest.mark.parametrize(("answer", "score"), [("0.05", 0.05), ("Score: 0.3", 0.3), ("1", 1.0), ("0", 0.0), (".7", 0.7)])
def test_hallucination_score_parsing(answer, score):
    patcher, _ = judge(answer)
    with patcher:
        assert sem.hallucination_score("text") == pytest.approx(score)


@pytest.mark.parametrize("answer", ["high", "1.5", "10", "-0.2 maybe"])
def test_unparseable_or_out_of_range_score_raises(answer):
    patcher, _ = judge(answer)
    with patcher:
        with pytest.raises(AgentTestJudgeError, match="score between 0.0 and 1.0"):
            sem.hallucination_score("text")


def test_assert_hallucination_score_threshold():
    patcher, _ = judge("0.05")
    with patcher:
        assert sem.assert_hallucination_score(_result(), max_score=0.1) is True
    patcher, _ = judge("0.6")
    with patcher:
        with pytest.raises(AssertionError, match="0.60 exceeds the allowed 0.10"):
            sem.assert_hallucination_score(_result(), max_score=0.1)


def test_assert_hallucination_score_below_is_strict():
    patcher, _ = judge("0.2")
    with patcher:
        with pytest.raises(AssertionError, match="not below the threshold 0.20"):
            sem.assert_hallucination_score_below(_result(), 0.2)


def test_invalid_max_score():
    with pytest.raises(ValueError):
        sem.assert_hallucination_score("text", max_score=2)


def test_agent_result_methods_share_semantic_implementation():
    patcher, prompts = judge("YES")
    with patcher:
        result = _result()
        assert result.contains_fact("Paris") is True
        assert result.tone_matches("neutral") is True
    assert "factual claim: 'Paris'" in prompts[0]


def test_agent_result_hallucination_score_cached_and_strict():
    patcher, prompts = judge("0.25")
    with patcher:
        result = _result()
        assert result.hallucination_score == 0.25
        assert result.hallucination_score == 0.25
    assert len(prompts) == 1

    patcher, _ = judge("no idea")
    with patcher:
        with pytest.raises(AgentTestJudgeError):
            _ = _result().hallucination_score
