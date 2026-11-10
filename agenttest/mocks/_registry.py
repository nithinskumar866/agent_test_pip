"""The tool-mock registry, and the one function every integration calls.

Mocks are keyed by tool name. Each name holds a *stack* of mocks, so nesting
two mocks of the same tool works and exiting the inner one restores the outer.

The registry is process-wide rather than per-context on purpose: frameworks
run tools on their own worker threads (LangGraph's ToolNode, CrewAI's
executor) that do not always inherit context variables, and a mock that
silently fails to apply is worse than no mock. It is guarded by a lock.

Integrations never interpret mock specs themselves — they call
:func:`apply_mock` (or :func:`apply_mock_async`) with the tool name and a
thunk that runs the real tool.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Final, Literal

__all__ = [
    "MOCK_REGISTRY",
    "MockHandle",
    "MockedToolError",
    "active_mock",
    "apply_mock",
    "apply_mock_async",
    "entering_tool",
    "in_tool_call",
    "push_mock",
    "pop_mock",
]

MockType = Literal["failure", "response", "latency", "sequence", "handler"]


class _Real:
    """Sentinel: let this call reach the real tool."""

    def __repr__(self) -> str:
        return "agenttest.REAL"


#: Use in ``mock_tool_sequence`` to let a call through to the real tool.
REAL: Final = _Real()


def tool_args(value: Any) -> dict[str, Any]:
    """A tool call's arguments as a dict, whichever framework recorded them.

    Handles keyword dicts, JSON strings (OpenAI Agents SDK) and LangChain
    ``ToolCall`` dicts. Anything else becomes ``{"input": value}``.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {"input": value}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    if isinstance(value, dict):
        if isinstance(value.get("args"), dict) and "name" in value:  # a LangChain ToolCall
            return value["args"]
        return value
    return {} if value in (None, [], ()) else {"input": value}


class MockedToolError(RuntimeError):
    """Default exception raised by :func:`agenttest.mock_tool_failure`."""


@dataclass
class MockHandle:
    """An active mock. Yielded by the ``mock_tool_*`` context managers.

    ``calls`` counts how many times the mocked tool was invoked — assert on it
    to catch a mock whose tool name never matched anything.
    """

    tool_name: str
    type: MockType
    error: str | None = None
    # builtins.type: the field named ``type`` shadows the builtin in this class body.
    exception: BaseException | builtins.type[BaseException] | None = None
    data: Any = None
    delay: float = 0.0
    #: Stop intercepting after this many calls (the real tool runs afterwards).
    times: int | None = None
    outcomes: list[Any] = field(default_factory=list)
    then: Any = REAL
    handler: Callable[[dict[str, Any]], Any] | None = None
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def called(self) -> bool:
        return self.calls > 0

    def as_dict(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"type": self.type}
        if self.type == "failure":
            spec["error"] = self.error
        elif self.type == "response":
            spec["data"] = self.data
        elif self.type == "latency":
            spec["delay"] = self.delay
        return spec

    def _take(self) -> int | None:
        """Claim this call: its 0-based position, or ``None`` once ``times`` is used up."""
        with self._lock:
            if self.times is not None and self.calls >= self.times:
                return None
            self.calls += 1
            return self.calls - 1

    def _outcome(self, position: int, input: Any) -> Any:
        """What this call does: a value to return, an exception to raise, or ``REAL``."""
        if self.type == "failure":
            return self._exception()
        if self.type == "response":
            return self.data
        if self.type == "handler":
            assert self.handler is not None
            return self.handler(tool_args(input))
        return self.outcomes[position] if position < len(self.outcomes) else self.then

    def _exception(self) -> BaseException:
        exc = self.exception
        if exc is None:
            return MockedToolError(self.error)
        if isinstance(exc, type):
            return exc(self.error)
        return exc


_STACKS: dict[str, list[MockHandle]] = {}
_LOCK = threading.Lock()


def push_mock(handle: MockHandle) -> None:
    with _LOCK:
        _STACKS.setdefault(handle.tool_name, []).append(handle)


def pop_mock(handle: MockHandle) -> None:
    with _LOCK:
        stack = _STACKS.get(handle.tool_name, [])
        if handle in stack:
            # Remove this exact handle even if exits happen out of order.
            del stack[len(stack) - 1 - stack[::-1].index(handle)]
        if not stack:
            _STACKS.pop(handle.tool_name, None)


def active_mock(tool_name: str) -> MockHandle | None:
    """The innermost active mock for ``tool_name``, if any."""
    with _LOCK:
        stack = _STACKS.get(tool_name)
        return stack[-1] if stack else None


class _RegistryView(Mapping[str, dict[str, Any]]):
    """Read-only ``{tool_name: spec_dict}`` view of the innermost active mocks."""

    def __getitem__(self, tool_name: str) -> dict[str, Any]:
        handle = active_mock(tool_name)
        if handle is None:
            raise KeyError(tool_name)
        return handle.as_dict()

    def __iter__(self) -> Iterator[str]:
        with _LOCK:
            return iter(list(_STACKS))

    def __len__(self) -> int:
        with _LOCK:
            return len(_STACKS)

    def __repr__(self) -> str:
        return f"MOCK_REGISTRY({dict(self)!r})"


#: Read-only view of active mocks, e.g. ``{"web_search": {"type": "failure", ...}}``.
MOCK_REGISTRY: Mapping[str, dict[str, Any]] = _RegistryView()


#: Tool names whose call is already being handled further up this call stack.
_IN_TOOL: ContextVar[frozenset[str]] = ContextVar("agenttest_in_tool", default=frozenset())


def in_tool_call(tool_name: str) -> bool:
    """Whether an outer layer is already handling a call to ``tool_name``.

    One tool call can pass through several interception layers (e.g. an
    ``@agenttest.tool`` function wrapped as a LangChain tool, or ``_arun``
    falling back to ``_run``). Only the outermost layer applies the mock and
    records the step.
    """
    return tool_name in _IN_TOOL.get()


@contextmanager
def entering_tool(tool_name: str) -> Iterator[None]:
    token = _IN_TOOL.set(_IN_TOOL.get() | {tool_name})
    try:
        yield
    finally:
        _IN_TOOL.reset(token)


def _raise(outcome: BaseException | type[BaseException]) -> None:
    exc = outcome() if isinstance(outcome, type) else outcome
    exc._agenttest_mocked = True  # type: ignore[attr-defined]
    raise exc


def is_mocked_failure(exc: BaseException) -> bool:
    """Whether ``exc`` was raised by a mock rather than by the real tool."""
    return getattr(exc, "_agenttest_mocked", False)


def _is_exception(value: Any) -> bool:
    return isinstance(value, BaseException) or (isinstance(value, type) and issubclass(value, BaseException))


def apply_mock(tool_name: str, call_real: Callable[[], Any], *, input: Any = None) -> Any:
    """Run a tool call, honouring any active mock for ``tool_name``.

    * failure  → raise the configured exception without calling the tool
    * response → return the canned data without calling the tool
    * latency  → sleep, then call the real tool
    * sequence → the next scripted outcome
    * handler  → whatever the function returns or raises for these arguments
    * no mock, or a mock whose ``times`` ran out → call the real tool
    """
    if in_tool_call(tool_name):
        return call_real()
    with entering_tool(tool_name):
        handle = active_mock(tool_name)
        position = handle._take() if handle is not None else None
        if handle is None or position is None:
            return call_real()
        if handle.type == "latency":
            time.sleep(handle.delay)
            return call_real()
        try:
            outcome = handle._outcome(position, input)
        except Exception as exc:  # raised by a handler
            _raise(exc)
        if outcome is REAL:
            return call_real()
        if _is_exception(outcome):
            _raise(outcome)
        return outcome


async def apply_mock_async(tool_name: str, call_real: Callable[[], Awaitable[Any]], *, input: Any = None) -> Any:
    """Async twin of :func:`apply_mock`; latency uses ``asyncio.sleep``."""
    if in_tool_call(tool_name):
        return await call_real()
    with entering_tool(tool_name):
        handle = active_mock(tool_name)
        position = handle._take() if handle is not None else None
        if handle is None or position is None:
            return await call_real()
        if handle.type == "latency":
            await asyncio.sleep(handle.delay)
            return await call_real()
        try:
            outcome = handle._outcome(position, input)
        except Exception as exc:  # raised by a handler
            _raise(exc)
        if outcome is REAL:
            return await call_real()
        if _is_exception(outcome):
            _raise(outcome)
        return outcome
