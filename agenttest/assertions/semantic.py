"""Semantic checks backed by an LLM judge (LLM-as-a-judge).

Same two layers as the structural checks:

* **predicates** (``contains_fact``, ``tone_matches``, ...) return a bool, and
  ``hallucination_score`` returns a float in [0, 1];
* **assertions** (``assert_contains_fact``, ...) raise ``AssertionError``
  including the judge's raw answer, and return ``True`` when they pass.

A judge answer that cannot be parsed raises
:class:`~agenttest.AgentTestJudgeError` rather than being guessed at.
"""

from __future__ import annotations

import re
from typing import Any, Final

from agenttest.assertions.structural import excerpt, fail
from agenttest.config import AgentTestJudgeError
from agenttest.core.result import _judge_llm

__all__ = [
    # predicates
    "contains_fact",
    "hallucination_score",
    "tone_matches",
    "answers_question",
    "language_matches",
    "satisfies",
    # assertions
    "assert_contains_fact",
    "assert_hallucination_score",
    "assert_hallucination_score_below",
    "assert_tone_matches",
    "assert_answers_question",
    "assert_language_matches",
    "assert_eval_prompt",
]

_VERDICT: Final[re.Pattern[str]] = re.compile(r"^\W*(YES|NO)\b", re.IGNORECASE)
_SCORE: Final[re.Pattern[str]] = re.compile(r"(?<![\d.-])(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")


def _text(target: Any) -> str:
    if hasattr(target, "output"):
        return str(target.output)
    return str(target)


def _ask_yes_no(question: str, text: str) -> tuple[bool, str]:
    prompt = f"{question}\n\nText: {text}\n\nAnswer only YES or NO."
    answer = _judge_llm(prompt)
    match = _VERDICT.match(answer)
    if match is None:
        raise AgentTestJudgeError(f"Judge did not answer YES or NO: {answer!r}")
    return match.group(1).upper() == "YES", answer


def _parse_score(answer: str) -> float:
    match = _SCORE.search(answer)
    if match is None:
        raise AgentTestJudgeError(f"Judge did not return a score between 0.0 and 1.0: {answer!r}")
    return float(match.group())


# --- prompts (one place) -----------------------------------------------------

def _fact_question(claim: str) -> str:
    return f"Does the following text contain the factual claim: '{claim}'?"


def _tone_question(tone: str) -> str:
    return f"Does the following text match a '{tone}' tone?"


def _answers_question(question: str) -> str:
    return f"Does the following text adequately answer this question: '{question}'?"


def _language_question(language: str) -> str:
    return f"Is the following text written in {language}?"


def _criteria_question(instruction: str) -> str:
    return f"Does the following text satisfy these criteria: '{instruction}'?"


# --- predicates --------------------------------------------------------------

def contains_fact(output: Any, claim: str) -> bool:
    return _ask_yes_no(_fact_question(claim), _text(output))[0]


def tone_matches(output: Any, tone: str) -> bool:
    return _ask_yes_no(_tone_question(tone), _text(output))[0]


def answers_question(output: Any, question: str) -> bool:
    return _ask_yes_no(_answers_question(question), _text(output))[0]


def language_matches(output: Any, language: str) -> bool:
    return _ask_yes_no(_language_question(language), _text(output))[0]


def satisfies(output: Any, instruction: str) -> bool:
    return _ask_yes_no(_criteria_question(instruction), _text(output))[0]


def hallucination_score(output: Any) -> float:
    """Judge-rated likelihood (0.0–1.0) that the output contains hallucinations."""
    prompt = (
        "Rate how likely this text contains hallucinations or factual inaccuracies "
        "on a scale from 0.0 (completely accurate) to 1.0 (severe hallucinations).\n\n"
        f"Text: {_text(output)}\n\n"
        "Return only a single float number between 0.0 and 1.0. Nothing else."
    )
    return _parse_score(_judge_llm(prompt))


# --- assertions --------------------------------------------------------------

def _assert_yes(question: str, output: Any, expectation: str) -> bool:
    text = _text(output)
    verdict, answer = _ask_yes_no(question, text)
    if not verdict:
        fail(f"{expectation}\n  judge answered: {answer!r}\n  output: {excerpt(text)!r}")
    return True


def assert_contains_fact(output: Any, claim: str) -> bool:
    return _assert_yes(_fact_question(claim), output, f"Expected output to contain the fact {claim!r}.")


def assert_tone_matches(output: Any, tone: str) -> bool:
    return _assert_yes(_tone_question(tone), output, f"Expected output to have a {tone!r} tone.")


def assert_answers_question(output: Any, question: str) -> bool:
    return _assert_yes(_answers_question(question), output, f"Expected output to answer {question!r}.")


def assert_language_matches(output: Any, language: str) -> bool:
    return _assert_yes(_language_question(language), output, f"Expected output to be written in {language}.")


def assert_eval_prompt(result: Any, instruction: str) -> bool:
    return _assert_yes(_criteria_question(instruction), result, f"Expected output to satisfy: {instruction!r}.")


def assert_hallucination_score(output: Any, max_score: float = 0.1) -> bool:
    """Assert the judge's hallucination score is at most ``max_score``."""
    if not 0.0 <= max_score <= 1.0:
        raise ValueError("max_score must be between 0.0 and 1.0")
    score = hallucination_score(output)
    if score > max_score:
        fail(f"Hallucination score {score:.2f} exceeds the allowed {max_score:.2f}.\n"
             f"  output: {excerpt(_text(output))!r}")
    return True


def assert_hallucination_score_below(result: Any, threshold: float) -> bool:
    """Assert the hallucination score is strictly below ``threshold``."""
    score = hallucination_score(result)
    if score >= threshold:
        fail(f"Hallucination score {score:.2f} is not below the threshold {threshold:.2f}.\n"
             f"  output: {excerpt(_text(result))!r}")
    return True
