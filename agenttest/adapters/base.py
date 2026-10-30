"""``BaseAdapter`` — the contract every framework adapter implements.

An adapter knows three things about one kind of agent:

* how to **call** it (:meth:`BaseAdapter.invoke`, and natively from async
  code with :meth:`BaseAdapter.ainvoke`),
* how to **observe** it — hooks installed for exactly one run by
  :meth:`BaseAdapter.instrument` and removed afterwards, recording steps into
  the active :class:`~agenttest.core.trace.Trace`,
* how to turn its return value into **text** (:meth:`BaseAdapter.extract_output`).

Timing, error capture, timeouts and building the
:class:`~agenttest.core.result.AgentResult` belong to the engine
(:mod:`agenttest.core.runner`), not to adapters.

Also hosts ``GenericAdapter``, used for plain Python callables.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Iterator

if TYPE_CHECKING:
    from agenttest.core.result import AgentResult

__all__ = ["BaseAdapter", "GenericAdapter", "call_agent", "entry_point", "text_of"]


def entry_point(agent: Any) -> Any:
    """The callable used to run a plain agent: its ``run``/``invoke`` method, or itself."""
    for method in ("run", "invoke"):
        bound = getattr(agent, method, None)
        if callable(bound):
            return bound
    if callable(agent):
        return agent
    raise TypeError(
        f"Don't know how to run {type(agent).__name__!r}: expected a callable "
        "or an object with a run()/invoke() method."
    )


def _is_coroutine_entry(agent: Any) -> bool:
    try:
        return inspect.iscoroutinefunction(entry_point(agent))
    except TypeError:
        return False


def call_agent(agent: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a plain agent: a callable, or an object with ``run``/``invoke``."""
    return entry_point(agent)(*args, **kwargs)


def text_of(value: Any) -> str:
    """Best-effort conversion of an agent's return value to its text output."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for attr in ("final_output", "output", "content", "text", "raw"):
        inner = getattr(value, attr, None)
        if isinstance(inner, str):
            return inner
    if isinstance(value, dict):
        for key in ("output", "answer", "result", "content"):
            if isinstance(value.get(key), str):
                return value[key]
        messages = value.get("messages")
        if isinstance(messages, list) and messages:
            return text_of(messages[-1])
    return str(value)


class BaseAdapter:
    """Base class for framework adapters. Subclasses override what differs."""

    #: Framework name used in ``agenttest.config.json``.
    name: ClassVar[str] = "generic"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        """Whether this adapter recognises ``agent`` by its type alone."""
        return False

    @contextmanager
    def instrument(self) -> Iterator[None]:
        """Install observation hooks for one run; undo them on exit."""
        yield

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the agent and return its raw result (may be awaitable)."""
        return call_agent(agent, *args, **kwargs)

    async def ainvoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the agent from async code and return its raw result.

        The default runs a plain ``async def`` agent on the event loop and
        anything else in a worker thread (with this context, so it records into
        the same trace). Adapters whose framework has a native async API
        override this to use it.
        """
        if type(self).invoke is BaseAdapter.invoke and _is_coroutine_entry(agent):
            return await call_agent(agent, *args, **kwargs)
        context = contextvars.copy_context()
        raw = await asyncio.to_thread(context.run, self.invoke, agent, *args, **kwargs)
        if inspect.isawaitable(raw):
            raw = await raw
        return raw

    def extract_output(self, raw: Any) -> str:
        """Turn the agent's raw result into its final text output."""
        return text_of(raw)

    def run_with_trace(self, agent: Any, *args: Any, **kwargs: Any) -> AgentResult:
        """Run ``agent`` with this adapter. Shortcut for ``agenttest.run(agent, adapter=self)``."""
        from agenttest.core.runner import run

        return run(agent, *args, adapter=self, **kwargs)


class GenericAdapter(BaseAdapter):
    """Adapter for plain Python agents.

    Captures the output; tool calls are captured for functions decorated
    with :func:`agenttest.tool`.
    """

    name: ClassVar[str] = "generic"
