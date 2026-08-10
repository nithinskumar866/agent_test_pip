"""Semantic assertions backed by LLM evaluation (LLM-as-a-judge).
"""

from __future__ import annotations

from typing import Any

from agenttest.core.result import _judge_llm

__all__ = [
    "assert_contains_fact",
    "assert_hallucination_score",
    "assert_hallucination_score_below",
    "assert_tone_matches",
    "assert_answers_question",
    "assert_language_matches",
    "assert_eval_prompt",
]


def _extract_text(target: Any) -> str:
    if hasattr(target, "output"):
        return str(target.output)
    return str(target)


def assert_contains_fact(output: Any, claim: str) -> bool:
    text = _extract_text(output)
    prompt = (
        f"Does the following text contain the factual claim: "
        f"'{claim}'?\n\nText: {text}\n\n"
        f"Answer only YES or NO."
    )
    result = _judge_llm(prompt)
    return result.upper().startswith("YES")


def assert_hallucination_score(output: Any) -> float:
    text = _extract_text(output)
    prompt = (
        f"Rate how likely this text contains hallucinations "
        f"or factual inaccuracies on a scale from 0.0 (completely "
        f"accurate) to 1.0 (severe hallucinations).\n\n"
        f"Text: {text}\n\n"
        f"Return only a single float number between 0.0 and 1.0. "
        f"Nothing else."
    )
    result = _judge_llm(prompt)
    try:
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except ValueError:
        return 0.5


def assert_hallucination_score_below(result: Any, threshold: float) -> None:
    score = assert_hallucination_score(result)
    if score >= threshold:
        raise AssertionError(
            f"Hallucination check failed: Score {score:.2f} >= threshold {threshold:.2f}."
        )


def assert_tone_matches(output: Any, tone: str) -> bool:
    text = _extract_text(output)
    prompt = (
        f"Does the following text match a '{tone}' tone?\n\n"
        f"Text: {text}\n\n"
        f"Answer only YES or NO."
    )
    result = _judge_llm(prompt)
    return result.upper().startswith("YES")


def assert_answers_question(output: Any, question: str) -> bool:
    text = _extract_text(output)
    prompt = (
        f"Does the following text adequately answer this question: "
        f"'{question}'?\n\n"
        f"Text: {text}\n\n"
        f"Answer only YES or NO."
    )
    result = _judge_llm(prompt)
    return result.upper().startswith("YES")


def assert_language_matches(output: Any, language: str) -> bool:
    text = _extract_text(output)
    prompt = (
        f"Is the following text written in {language}?\n\n"
        f"Text: {text}\n\n"
        f"Answer only YES or NO."
    )
    result = _judge_llm(prompt)
    return result.upper().startswith("YES")


def assert_eval_prompt(result: Any, instruction: str) -> None:
    text = _extract_text(result)
    prompt = (
        f"Does this text satisfy the following criteria: '{instruction}'? "
        f"Text: {text}. Answer only YES or NO."
    )
    answer = _judge_llm(prompt)
    if not answer.strip().upper().startswith("YES"):
        raise AssertionError(
            f"Evaluation prompt check failed: Criteria '{instruction}' was not satisfied.\n"
            f"Judge response: {answer!r}"
        )
