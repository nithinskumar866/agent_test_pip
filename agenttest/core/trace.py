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
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Literal, TypeVar, overload

from pydantic import BaseModel, Field

__all__ = ["Step", "Trace", "tracing", "current_trace", "record_step", "tool"]

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
    duration. Outside an agenttest run the decorator is transparent.
    Works on both sync and ``async`` functions.
    """

    def decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        try:
            sig: inspect.Signature | None = inspect.signature(func)
        except (TypeError, ValueError):
            sig = None

        def record(call_input: Any, start: float, *, output: Any = None, error: BaseException | None = None) -> None:
            record_step(
                "tool",
                tool_name,
                input=call_input,
                output=output,
                error=f"{type(error).__name__}: {error}" if error is not None else None,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if current_trace() is None:
                    return await func(*args, **kwargs)
                start, call_input = time.perf_counter(), _describe_call(sig, args, kwargs)
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    record(call_input, start, error=exc)
                    raise
                record(call_input, start, output=result)
                return result

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if current_trace() is None:
                    return func(*args, **kwargs)
                start, call_input = time.perf_counter(), _describe_call(sig, args, kwargs)
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    record(call_input, start, error=exc)
                    raise
                record(call_input, start, output=result)
                return result

            wrapper = sync_wrapper

        wrapper._agenttest_tool_name = tool_name  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate
