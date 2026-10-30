"""Pure Python structural checks for agent results. Zero network or LLM calls.

Two layers:

* **predicates** (``word_count_between``, ``tool_was_called``, ...) return a bool;
* **assertions** (``assert_word_count_between``, ...) raise ``AssertionError``
  with a message that says what was expected, what was found, and shows the
  output — and return ``True`` when they pass.

Every function accepts either an :class:`~agenttest.AgentResult` or the raw
value it inspects (a string, a list of tool names, a number).
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    # predicates
    "word_count_between",
    "steps_at_most",
    "tool_was_called",
    "tool_was_not_called",
    "latency_under",
    "tokens_under",
    "output_contains",
    "output_matches_regex",
    "output_is_valid_json",
    "gracefully_degraded",
    # assertions
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

_EXCERPT_CHARS = 300


def _text(target: Any) -> str:
    if hasattr(target, "output"):
        return str(target.output)
    return str(target)


def _field(target: Any, name: str, convert: Any) -> Any:
    return getattr(target, name) if hasattr(target, name) else convert(target)


def _tools(target: Any) -> list[str]:
    return list(getattr(target, "tools_called", target))


def excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    """The output, shortened for failure messages."""
    text = text.strip()
    return text if len(text) <= limit else f"{text[:limit]}… [{len(text) - limit} more chars]"


def fail(message: str, target: Any = None) -> None:
    """Raise ``AssertionError``, appending the agent output when there is one."""
    if target is not None and hasattr(target, "output"):
        message = f"{message}\n  output: {excerpt(str(target.output))!r}"
    raise AssertionError(message)


# --- predicates --------------------------------------------------------------

def word_count_between(output: Any, min_count: int, max_count: int) -> bool:
    return min_count <= len(_text(output).split()) <= max_count


def steps_at_most(steps_taken: Any, max_steps: int) -> bool:
    return _field(steps_taken, "steps_taken", int) <= max_steps


def tool_was_called(tools_called: Any, tool_name: str) -> bool:
    return tool_name in _tools(tools_called)


def tool_was_not_called(tools_called: Any, tool_name: str) -> bool:
    return tool_name not in _tools(tools_called)


def latency_under(latency_ms: Any, threshold_ms: float) -> bool:
    return _field(latency_ms, "latency_ms", float) < threshold_ms


def tokens_under(tokens_used: Any, max_tokens: int) -> bool:
    return _field(tokens_used, "tokens_used", int) < max_tokens


def output_contains(output: Any, substring: str) -> bool:
    return substring.lower() in _text(output).lower()


def output_matches_regex(output: Any, pattern: str) -> bool:
    return re.search(pattern, _text(output)) is not None


def output_is_valid_json(output: Any) -> bool:
    try:
        json.loads(_text(output))
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    return True


def gracefully_degraded(result: Any) -> bool:
    return getattr(result, "gracefully_degraded", False) is True


# --- assertions --------------------------------------------------------------

def assert_word_count_between(output: Any, min_count: int, max_count: int) -> bool:
    count = len(_text(output).split())
    if not min_count <= count <= max_count:
        fail(f"Expected {min_count}–{max_count} words, got {count}.", output)
    return True


def assert_steps_at_most(steps_taken: Any, max_steps: int) -> bool:
    steps = _field(steps_taken, "steps_taken", int)
    if steps > max_steps:
        fail(f"Expected at most {max_steps} steps, agent took {steps}.", steps_taken)
    return True


def assert_steps_taken_max(result: Any, max_steps: int) -> bool:
    return assert_steps_at_most(result, max_steps)


def assert_tool_was_called(tools_called: Any, tool_name: str) -> bool:
    tools = _tools(tools_called)
    if tool_name not in tools:
        fail(f"Expected tool {tool_name!r} to be called; tools called: {tools or 'none'}.", tools_called)
    return True


def assert_tool_was_not_called(tools_called: Any, tool_name: str) -> bool:
    tools = _tools(tools_called)
    if tool_name in tools:
        fail(f"Expected tool {tool_name!r} not to be called; it was called {tools.count(tool_name)} time(s).",
             tools_called)
    return True


def assert_latency_under(latency_ms: Any, threshold_ms: float) -> bool:
    latency = _field(latency_ms, "latency_ms", float)
    if latency >= threshold_ms:
        fail(f"Expected latency under {threshold_ms:g} ms, got {latency:.1f} ms.", latency_ms)
    return True


def assert_latency_below(result: Any, max_ms: float) -> bool:
    return assert_latency_under(result, max_ms)


def assert_tokens_under(tokens_used: Any, max_tokens: int) -> bool:
    tokens = _field(tokens_used, "tokens_used", int)
    if tokens >= max_tokens:
        fail(f"Expected fewer than {max_tokens} tokens, used {tokens}.", tokens_used)
    return True


def assert_output_contains(output: Any, substring: str) -> bool:
    if not output_contains(output, substring):
        fail(f"Expected output to contain {substring!r} (case-insensitive).\n  output: {excerpt(_text(output))!r}")
    return True


def assert_output_matches_regex(output: Any, pattern: str) -> bool:
    if not output_matches_regex(output, pattern):
        fail(f"Expected output to match /{pattern}/.\n  output: {excerpt(_text(output))!r}")
    return True


def assert_regex_match(result: Any, pattern: str) -> bool:
    return assert_output_matches_regex(result, pattern)


def assert_output_is_valid_json(output: Any) -> bool:
    text = _text(output)
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        fail(f"Expected valid JSON output: {exc}.\n  output: {excerpt(text)!r}")
    return True


def assert_valid_json(result: Any) -> bool:
    return assert_output_is_valid_json(result)


def assert_gracefully_degraded(result: Any) -> bool:
    if gracefully_degraded(result):
        return True
    trace = getattr(result, "trace", None)
    failed = [s.name for s in getattr(trace, "failed_tool_calls", [])] if trace is not None else []
    if getattr(result, "success", True) is False:
        reason = f"the agent crashed: {getattr(result, 'error', None)}"
    elif not failed:
        reason = "no tool failed during the run (did the mock match a tool name?)"
    else:
        reason = f"tools {failed} failed but the agent produced no output"
    fail(f"Expected the agent to degrade gracefully, but {reason}.", result)
    return True
