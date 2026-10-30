"""Tool-mock support for LangChain ``BaseTool`` (used by LangGraph's ToolNode).

Mocks have to act *inside* LangChain's tool machinery: ``BaseTool.run`` turns
whatever ``_run`` returns into a ``ToolMessage`` and routes exceptions through
callbacks and error handling, and ToolNode expects exactly that. So mocks are
applied around the concrete tool class's ``_run`` / ``_arun``:

* ``BaseTool.run`` / ``arun`` are patched for the duration of a run; the first
  time a tool class is seen, its ``_run`` / ``_arun`` get a mock-aware wrapper;
* every wrapper is removed when the run ends.

Patching the class (not the instance) keeps concurrent calls of the same tool
safe. Tool steps are not recorded here. Each ``BaseTool.run`` is logged as a
:class:`ToolRun`; the LangGraph adapter records steps from the graph's
``ToolMessage`` objects and uses the log for runs that produced none (a tool
called directly inside a node, or a failure that crashed the graph).
"""

from __future__ import annotations

import contextvars
import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from agenttest.mocks._registry import active_mock, apply_mock, apply_mock_async

__all__ = ["ToolRun", "ToolRunLog", "langchain_tool_mocks", "current_tool_log"]

_MISSING = object()


@dataclass
class ToolRun:
    """One completed ``BaseTool.run`` / ``arun`` call."""

    name: str
    input: Any
    output: Any = None
    error: str | None = None
    tool_call_id: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRunLog:
    """Thread-safe log of the LangChain tool runs made during one agent run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: list[ToolRun] = []

    def add(self, run: ToolRun) -> None:
        with self._lock:
            self._runs.append(run)

    def take(self, predicate: Any = None) -> list[ToolRun]:
        """Remove and return the runs matching ``predicate`` (all by default)."""
        with self._lock:
            taken = [r for r in self._runs if predicate is None or predicate(r)]
            self._runs = [r for r in self._runs if r not in taken]
        return taken


_LOG: contextvars.ContextVar[ToolRunLog | None] = contextvars.ContextVar("agenttest_langchain_tool_log", default=None)


def current_tool_log() -> ToolRunLog | None:
    """The log of the enclosing :func:`langchain_tool_mocks` block, if any."""
    return _LOG.get()


def _begin(tool: Any, tool_input: Any, kwargs: dict[str, Any]) -> tuple[ToolRun, float]:
    handle = active_mock(tool.name)
    run = ToolRun(
        name=tool.name,
        input=tool_input,
        tool_call_id=kwargs.get("tool_call_id"),
        metadata={"mocked": handle.type} if handle is not None else {},
    )
    return run, time.perf_counter()


def _end(log: ToolRunLog, run: ToolRun, start: float, output: Any = None, exc: BaseException | None = None) -> None:
    run.duration_ms = (time.perf_counter() - start) * 1000
    if exc is not None:
        run.error = f"{type(exc).__name__}: {exc}"
    else:
        # With a tool_call_id, BaseTool.run returns a ToolMessage.
        run.output = getattr(output, "content", output)
    log.add(run)


class _ClassPatcher:
    """Wraps ``_run`` / ``_arun`` on tool classes; restores them on ``restore()``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._saved: list[tuple[type, str, Any]] = []
        self._seen: set[type] = set()

    def ensure(self, cls: type) -> None:
        with self._lock:
            if cls in self._seen:
                return
            self._seen.add(cls)
            for attr, make in (("_run", _wrap_run), ("_arun", _wrap_arun)):
                current = getattr(cls, attr, None)
                if current is None or getattr(current, "_agenttest_mock_wrapper", False):
                    continue
                self._saved.append((cls, attr, cls.__dict__.get(attr, _MISSING)))
                setattr(cls, attr, make(current))

    def restore(self) -> None:
        with self._lock:
            for cls, attr, original in reversed(self._saved):
                if original is _MISSING:
                    delattr(cls, attr)
                else:
                    setattr(cls, attr, original)
            self._saved.clear()
            self._seen.clear()


def _wrap_run(original: Any) -> Any:
    # functools.wraps keeps the signature visible, so BaseTool.run still sees
    # (and passes) ``run_manager`` when the real ``_run`` accepts it.
    @functools.wraps(original)
    def _run(self: Any, *args: Any, **kwargs: Any) -> Any:
        return apply_mock(self.name, lambda: original(self, *args, **kwargs))

    _run._agenttest_mock_wrapper = True  # type: ignore[attr-defined]
    return _run


def _wrap_arun(original: Any) -> Any:
    @functools.wraps(original)
    async def _arun(self: Any, *args: Any, **kwargs: Any) -> Any:
        return await apply_mock_async(self.name, lambda: original(self, *args, **kwargs))

    _arun._agenttest_mock_wrapper = True  # type: ignore[attr-defined]
    return _arun


@contextmanager
def langchain_tool_mocks() -> Iterator[ToolRunLog | None]:
    """Make agenttest tool mocks apply to every LangChain tool run in the block.

    Yields the block's :class:`ToolRunLog` (``None`` without langchain-core).
    """
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        yield None
        return

    patcher = _ClassPatcher()
    log = ToolRunLog()
    original_run = BaseTool.run
    original_arun = BaseTool.arun

    @functools.wraps(original_run)
    def run(self: Any, tool_input: Any = None, *args: Any, **kwargs: Any) -> Any:
        patcher.ensure(type(self))
        record, start = _begin(self, tool_input, kwargs)
        try:
            output = original_run(self, tool_input, *args, **kwargs)
        except Exception as exc:
            _end(log, record, start, exc=exc)
            raise
        _end(log, record, start, output)
        return output

    @functools.wraps(original_arun)
    async def arun(self: Any, tool_input: Any = None, *args: Any, **kwargs: Any) -> Any:
        patcher.ensure(type(self))
        record, start = _begin(self, tool_input, kwargs)
        try:
            output = await original_arun(self, tool_input, *args, **kwargs)
        except Exception as exc:
            _end(log, record, start, exc=exc)
            raise
        _end(log, record, start, output)
        return output

    BaseTool.run = run  # type: ignore[method-assign]
    BaseTool.arun = arun  # type: ignore[method-assign]
    token = _LOG.set(log)
    try:
        yield log
    finally:
        _LOG.reset(token)
        BaseTool.run = original_run  # type: ignore[method-assign]
        BaseTool.arun = original_arun  # type: ignore[method-assign]
        patcher.restore()
