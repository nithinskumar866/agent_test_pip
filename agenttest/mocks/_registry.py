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
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

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

MockType = Literal["failure", "response", "latency"]


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
        else:
            spec["delay"] = self.delay
        return spec

    def _hit(self) -> None:
        with self._lock:
            self.calls += 1

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


def apply_mock(tool_name: str, call_real: Callable[[], Any]) -> Any:
    """Run a tool call, honouring any active mock for ``tool_name``.

    * failure  → raise the configured exception without calling the tool
    * response → return the canned data without calling the tool
    * latency  → sleep, then call the real tool
    * no mock  → call the real tool
    """
    if in_tool_call(tool_name):
        return call_real()
    with entering_tool(tool_name):
        handle = active_mock(tool_name)
        if handle is None:
            return call_real()
        handle._hit()
        if handle.type == "failure":
            raise handle._exception()
        if handle.type == "response":
            return handle.data
        time.sleep(handle.delay)
        return call_real()


async def apply_mock_async(tool_name: str, call_real: Callable[[], Awaitable[Any]]) -> Any:
    """Async twin of :func:`apply_mock`; latency uses ``asyncio.sleep``."""
    if in_tool_call(tool_name):
        return await call_real()
    with entering_tool(tool_name):
        handle = active_mock(tool_name)
        if handle is None:
            return await call_real()
        handle._hit()
        if handle.type == "failure":
            raise handle._exception()
        if handle.type == "response":
            return handle.data
        await asyncio.sleep(handle.delay)
        return await call_real()
