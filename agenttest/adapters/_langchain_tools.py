"""Tool-mock support for LangChain ``BaseTool`` (used by LangGraph's ToolNode).

Mocks have to act *inside* LangChain's tool machinery: ``BaseTool.run`` turns
whatever ``_run`` returns into a ``ToolMessage`` and routes exceptions through
callbacks and error handling, and ToolNode expects exactly that. So mocks are
applied around the concrete tool class's ``_run`` / ``_arun``:

* ``BaseTool.run`` / ``arun`` are patched for the duration of a run; the first
  time a tool class is seen, its ``_run`` / ``_arun`` get a mock-aware wrapper;
* every wrapper is removed when the run ends.

Patching the class (not the instance) keeps concurrent calls of the same tool
safe. Tool steps are not recorded here — the LangGraph adapter reads them from
the resulting ``ToolMessage`` objects, which also covers tools agenttest
cannot intercept.
"""

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from agenttest.mocks._registry import apply_mock, apply_mock_async

__all__ = ["langchain_tool_mocks"]

_MISSING = object()


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
def langchain_tool_mocks() -> Iterator[None]:
    """Make agenttest tool mocks apply to every LangChain tool run in the block."""
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        yield
        return

    patcher = _ClassPatcher()
    original_run = BaseTool.run
    original_arun = BaseTool.arun

    @functools.wraps(original_run)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        patcher.ensure(type(self))
        return original_run(self, *args, **kwargs)

    @functools.wraps(original_arun)
    async def arun(self: Any, *args: Any, **kwargs: Any) -> Any:
        patcher.ensure(type(self))
        return await original_arun(self, *args, **kwargs)

    BaseTool.run = run  # type: ignore[method-assign]
    BaseTool.arun = arun  # type: ignore[method-assign]
    try:
        yield
    finally:
        BaseTool.run = original_run  # type: ignore[method-assign]
        BaseTool.arun = original_arun  # type: ignore[method-assign]
        patcher.restore()
