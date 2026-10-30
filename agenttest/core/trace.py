"""Execution traces: what an agent did during one run.

A :class:`Trace` is an ordered list of :class:`Step` records — LLM calls,
tool calls and graph-node executions. The engine opens a trace for each run
with :func:`tracing`; adapters and the :func:`tool` decorator append to
whichever trace is active in the current context.

The active trace lives in a :class:`contextvars.ContextVar`, so concurrent
runs (threads or asyncio tasks) never write into each other's trace.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, TypeVar, overload

from pydantic import BaseModel, Field

from agenttest.mocks._registry import active_mock, apply_mock, apply_mock_async, in_tool_call

__all__ = ["Step", "Trace", "tracing", "current_trace", "record_step", "run_tool", "arun_tool", "tool"]

StepKind = Literal["llm", "tool", "node"]

F = TypeVar("F", bound=Callable[..., Any])


class Step(BaseModel):
    """One unit of agent work."""

    kind: StepKind
    name: str
    input: Any = None
    output: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    #: Tool names an LLM *requested* in this step (not necessarily executed).
    tool_requests: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def failed(self) -> bool:
        return self.error is not None


class Trace(BaseModel):
    """Ordered record of the steps taken during one agent run."""

    steps: list[Step] = Field(default_factory=list)

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    @property
    def llm_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "llm"]

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool"]

    @property
    def tools_called(self) -> list[str]:
        """Names of tools that were executed, in call order (with repeats)."""
        return [s.name for s in self.tool_calls]

    @property
    def failed_tool_calls(self) -> list[Step]:
        return [s for s in self.tool_calls if s.failed]

    @property
    def tokens_used(self) -> int:
        return sum(s.tokens for s in self.steps)


_ACTIVE_TRACE: ContextVar[Trace | None] = ContextVar("agenttest_active_trace", default=None)


def current_trace() -> Trace | None:
    """Return the trace of the run in progress, or ``None`` outside a run."""
    return _ACTIVE_TRACE.get()


@contextmanager
def tracing() -> Iterator[Trace]:
    """Open a fresh trace for the duration of the block."""
    trace = Trace()
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE_TRACE.reset(token)


def record_step(kind: StepKind, name: str, **fields: Any) -> Step | None:
    """Append a step to the active trace. A no-op outside a run."""
    trace = _ACTIVE_TRACE.get()
    if trace is None:
        return None
    return trace.add(Step(kind=kind, name=name, **fields))


def _describe_call(sig: inspect.Signature | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if sig is None:
        return {"args": list(args), "kwargs": kwargs}
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}
    return dict(bound.arguments)


class _ToolCall:
    """Records one tool call as a ``tool`` step when the block exits.

    Outside a run nothing is recorded.
    """

    def __init__(self, name: str, call_input: Any) -> None:
        self._name = name
        self._input = call_input
        self._output: Any = None
        self._mock = active_mock(name)

    def __enter__(self) -> _ToolCall:
        self._start = time.perf_counter()
        return self

    def done(self, output: Any) -> Any:
        self._output = output
        return output

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        record_step(
            "tool",
            self._name,
            input=self._input,
            output=None if exc is not None else self._output,
            error=f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception) else None,
            duration_ms=(time.perf_counter() - self._start) * 1000,
            metadata={"mocked": self._mock.type} if self._mock is not None else {},
        )


def run_tool(name: str, call_real: Callable[[], Any], *, input: Any = None) -> Any:
    """Execute one tool call: apply any active mock, and record a ``tool`` step.

    The single path every integration uses for tools it intercepts. Nested
    interception of the same call (see :func:`~agenttest.mocks._registry.in_tool_call`)
    is passed straight through, so a call is mocked and recorded exactly once.
    """
    if in_tool_call(name) or current_trace() is None:
        return apply_mock(name, call_real)
    with _ToolCall(name, input) as call:
        return call.done(apply_mock(name, call_real))


async def arun_tool(name: str, call_real: Callable[[], Awaitable[Any]], *, input: Any = None) -> Any:
    """Async twin of :func:`run_tool`."""
    if in_tool_call(name) or current_trace() is None:
        return await apply_mock_async(name, call_real)
    with _ToolCall(name, input) as call:
        return call.done(await apply_mock_async(name, call_real))


@overload
def tool(fn: F, /) -> F: ...
@overload
def tool(*, name: str | None = None) -> Callable[[F], F]: ...


def tool(fn: Callable[..., Any] | None = None, /, *, name: str | None = None) -> Any:
    """Mark a plain Python function as an agent tool so its calls are traced.

    Use on the tools of framework-less agents (framework adapters capture
    tool calls on their own)::

        @agenttest.tool
        def web_search(query: str) -> str: ...

    Each call records a ``tool`` step with its arguments, result or error and
    duration, and honours any active ``mock_tool_*`` for the tool's name.
    Outside an agenttest run nothing is recorded, but mocks still apply.
    Works on both sync and ``async`` functions.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        try:
            sig: inspect.Signature | None = inspect.signature(func)
        except (TypeError, ValueError):
            sig = None

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await arun_tool(
                    tool_name, functools.partial(func, *args, **kwargs), input=_describe_call(sig, args, kwargs)
                )

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return run_tool(
                    tool_name, functools.partial(func, *args, **kwargs), input=_describe_call(sig, args, kwargs)
                )

            wrapper = sync_wrapper

        wrapper._agenttest_tool_name = tool_name  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate
