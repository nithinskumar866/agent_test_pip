"""Pure Python structural assertions for agent results. Zero network or LLM calls.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "assert_word_count_between",
    "assert_steps_at_most",
    "assert_steps_taken_max",
    "assert_tool_was_called",
    "assert_tool_was_not_called",
    "assert_latency_under",
    "assert_latency_below",
    "assert_tokens_under",
    "assert_output_contains",
    "assert_output_matches_regex",
    "assert_regex_match",
    "assert_output_is_valid_json",
    "assert_valid_json",
    "assert_gracefully_degraded",
]


def _extract_text(target: Any) -> str:
    if hasattr(target, "output"):
        return str(target.output)
    return str(target)


def assert_word_count_between(output: Any, min_count: int, max_count: int) -> bool:
    text = _extract_text(output)
    count = len(text.split())
    return min_count <= count <= max_count


def assert_steps_at_most(steps_taken: Any, max_steps: int) -> bool:
    steps = steps_taken.steps_taken if hasattr(steps_taken, "steps_taken") else int(steps_taken)
    return steps <= max_steps


def assert_steps_taken_max(result: Any, max_steps: int) -> bool:
    return assert_steps_at_most(result, max_steps)


def assert_tool_was_called(tools_called: Any, tool_name: str) -> bool:
    tools = tools_called.tools_called if hasattr(tools_called, "tools_called") else tools_called
    return tool_name in tools


def assert_tool_was_not_called(tools_called: Any, tool_name: str) -> bool:
    tools = tools_called.tools_called if hasattr(tools_called, "tools_called") else tools_called
    return tool_name not in tools


def assert_latency_under(latency_ms: Any, threshold_ms: float) -> bool:
    lat = latency_ms.latency_ms if hasattr(latency_ms, "latency_ms") else float(latency_ms)
    return lat < threshold_ms


def assert_latency_below(result: Any, max_ms: float) -> bool:
    return assert_latency_under(result, max_ms)


def assert_tokens_under(tokens_used: Any, max_tokens: int) -> bool:
    tok = tokens_used.tokens_used if hasattr(tokens_used, "tokens_used") else int(tokens_used)
    return tok < max_tokens


def assert_output_contains(output: Any, substring: str) -> bool:
    text = _extract_text(output)
    return substring.lower() in text.lower()


def assert_output_matches_regex(output: Any, pattern: str) -> bool:
    text = _extract_text(output)
    return bool(re.search(pattern, text))


def assert_regex_match(result: Any, pattern: str) -> bool:
    return assert_output_matches_regex(result, pattern)


def assert_output_is_valid_json(output: Any) -> bool:
    text = _extract_text(output)
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def assert_valid_json(result: Any) -> bool:
    return assert_output_is_valid_json(result)


def assert_gracefully_degraded(result: Any) -> bool:
    return getattr(result, "gracefully_degraded", False) is True
