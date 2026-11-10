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

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from agenttest.mocks._registry import REAL, MockHandle, pop_mock, push_mock

__all__ = [
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
    "mock_tool_sequence",
    "mock_tool",
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
    times: int | None = None,
) -> Iterator[MockHandle]:
    """Make the named tool raise instead of running.

    Args:
        tool_name: Name of the tool, as the agent knows it.
        error_message: Message for the raised exception.
        exception: Exception class or instance to raise. Defaults to
            :class:`~agenttest.mocks.MockedToolError`.
        times: Fail only this many calls, then let the real tool run, to test
            that an agent recovers: ``times=1`` means "fail once, then work".

    Yields:
        A :class:`~agenttest.mocks.MockHandle`; ``handle.calls`` counts hits.
    """
    message = error_message or f"Tool '{tool_name}' failed (mocked)"
    if times is not None and times < 0:
        raise ValueError("times must be non-negative")
    with _activate(MockHandle(tool_name, "failure", error=message, exception=exception, times=times)) as handle:
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


@contextmanager
def mock_tool_sequence(tool_name: str, outcomes: Iterable[Any], *, then: Any = REAL) -> Iterator[MockHandle]:
    """Script the tool's successive calls.

    Each outcome is a value to return, an exception (class or instance) to
    raise, or :data:`agenttest.REAL` to call the real tool::

        with mock_tool_sequence("search", [TimeoutError("slow"), "cached result", REAL]):
            ...

    Args:
        outcomes: One outcome per call, in order.
        then: What every later call does. Defaults to calling the real tool.
    """
    with _activate(MockHandle(tool_name, "sequence", outcomes=list(outcomes), then=then)) as handle:
        yield handle


@contextmanager
def mock_tool(tool_name: str, handler: Callable[[dict[str, Any]], Any]) -> Iterator[MockHandle]:
    """Answer the tool with ``handler(arguments)``, so the result can depend on the call.

    The handler gets the arguments as a dict. What it returns is the tool's
    result; what it raises is the tool's failure; returning
    :data:`agenttest.REAL` calls the real tool::

        with mock_tool("get_weather", lambda args: {"Paris": "rain"}[args["city"]]):
            ...
    """
    if not callable(handler):
        raise TypeError("mock_tool() needs a callable that takes the arguments dict")
    with _activate(MockHandle(tool_name, "handler", handler=handler)) as handle:
        yield handle
