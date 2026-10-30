"""Context managers that inject tool failures, canned responses and latency.

Mocks apply to tools by name, wherever the tool runs:

* functions decorated with :func:`agenttest.tool`,
* LangChain / LangGraph tools (``BaseTool``),
* OpenAI Agents SDK function tools,
* CrewAI tools,

for the duration of the ``with`` block — including inside
:func:`agenttest.run` worker threads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from agenttest.mocks._registry import MockHandle, pop_mock, push_mock

__all__ = [
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
]


@contextmanager
def _activate(handle: MockHandle) -> Iterator[MockHandle]:
    push_mock(handle)
    try:
        yield handle
    finally:
        pop_mock(handle)


@contextmanager
def mock_tool_failure(
    tool_name: str,
    error_message: str | None = None,
    *,
    exception: BaseException | type[BaseException] | None = None,
) -> Iterator[MockHandle]:
    """Make the named tool raise instead of running.

    Args:
        tool_name: Name of the tool, as the agent knows it.
        error_message: Message for the raised exception.
        exception: Exception class or instance to raise. Defaults to
            :class:`~agenttest.mocks.MockedToolError`.

    Yields:
        A :class:`~agenttest.mocks.MockHandle`; ``handle.calls`` counts hits.
    """
    message = error_message or f"Tool '{tool_name}' failed (mocked)"
    with _activate(MockHandle(tool_name, "failure", error=message, exception=exception)) as handle:
        yield handle


@contextmanager
def mock_tool_response(tool_name: str, returns: Any) -> Iterator[MockHandle]:
    """Make the named tool return ``returns`` instead of running."""
    with _activate(MockHandle(tool_name, "response", data=returns)) as handle:
        yield handle


@contextmanager
def mock_tool_latency(tool_name: str, delay_ms: int | float) -> Iterator[MockHandle]:
    """Delay the named tool by ``delay_ms`` milliseconds, then run it for real."""
    if delay_ms < 0:
        raise ValueError("delay_ms must be non-negative")
    with _activate(MockHandle(tool_name, "latency", delay=delay_ms / 1000.0)) as handle:
        yield handle
