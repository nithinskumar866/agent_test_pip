"""CrewAI adapter — runs a crew with ``kickoff`` and records its task outputs.

Each task output becomes a ``node`` step named after the agent role that
produced it; the crew's aggregate token usage is attached to the last step.
Tool calls are traced (and mocks applied) by patching CrewAI's tool classes
for the duration of the run. ``arun`` uses the native ``akickoff`` (falling
back to ``kickoff_async`` on older CrewAI).
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar
from unittest import mock

from agenttest.adapters.base import BaseAdapter, call_agent
from agenttest.core.trace import arun_tool, record_step, run_tool

__all__ = ["CrewAIAdapter"]


def _traced_invoke(original: Any) -> Any:
    @functools.wraps(original)
    def invoke(self: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        return run_tool(self.name, lambda: original(self, input, *args, **kwargs), input=input)

    return invoke


def _traced_ainvoke(original: Any) -> Any:
    @functools.wraps(original)
    async def ainvoke(self: Any, input: Any = None, *args: Any, **kwargs: Any) -> Any:
        return await arun_tool(self.name, lambda: original(self, input, *args, **kwargs), input=input)

    return ainvoke


def _defining(root: type, attr: str) -> list[type]:
    """``root`` and its subclasses that define ``attr`` themselves."""
    found, stack, seen = [], [root], set()
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if attr in cls.__dict__:
            found.append(cls)
        stack.extend(cls.__subclasses__())
    return found


def _traced_run(original: Any) -> Any:
    @functools.wraps(original)
    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        return run_tool(self.name, lambda: original(self, *args, **kwargs), input=kwargs or list(args))

    return run


def _as_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """``run(crew, {"topic": ...})`` passes the dict as ``inputs=``."""
    if args and "inputs" not in kwargs:
        kwargs = {**kwargs, "inputs": args[0]}
        args = args[1:]
    return args, kwargs


class CrewAIAdapter(BaseAdapter):
    """Adapter for CrewAI crews."""

    name: ClassVar[str] = "crewai"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        return type(agent).__module__.startswith("crewai") and hasattr(agent, "kickoff")

    @contextmanager
    def instrument(self) -> Iterator[None]:
        """Trace and mock every CrewAI tool call made during the run.

        CrewAI's ToolUsage runs tools through ``CrewStructuredTool.invoke`` /
        ``ainvoke``; ``BaseTool.run`` covers tools called directly.
        """
        targets: list[tuple[type, str, Any]] = []
        try:
            from crewai.tools.structured_tool import CrewStructuredTool

            targets += [
                (CrewStructuredTool, "invoke", _traced_invoke),
                (CrewStructuredTool, "ainvoke", _traced_ainvoke),
            ]
        except ImportError:
            pass
        try:
            from crewai.tools import BaseTool

            # Subclasses such as crewai's own ``Tool`` override ``run``.
            targets += [(cls, "run", _traced_run) for cls in _defining(BaseTool, "run")]
        except ImportError:
            pass

        with ExitStack() as stack:
            for cls, attr, make in targets:
                stack.enter_context(mock.patch.object(cls, attr, make(getattr(cls, attr))))
            yield

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(agent, "kickoff"):
            return call_agent(agent, *args, **kwargs)
        args, kwargs = _as_inputs(args, kwargs)
        result = agent.kickoff(*args, **kwargs)
        self._record(result)
        return result

    async def ainvoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        kickoff = getattr(agent, "akickoff", None) or getattr(agent, "kickoff_async", None)
        if kickoff is None:
            return await super().ainvoke(agent, *args, **kwargs)
        args, kwargs = _as_inputs(args, kwargs)
        result = await kickoff(*args, **kwargs)
        self._record(result)
        return result

    @staticmethod
    def _record(result: Any) -> None:
        steps = [
            record_step(
                "node",
                str(getattr(task, "agent", None) or "agent"),
                input=getattr(task, "description", None),
                output=getattr(task, "raw", None),
            )
            for task in getattr(result, "tasks_output", None) or []
        ]
        usage = getattr(result, "token_usage", None)
        if usage is None:
            return
        # CrewAI only reports crew-wide usage; attach it to the final step.
        target = steps[-1] if steps else record_step("node", "crew")
        if target is not None:
            target.input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            target.output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            target.metadata["aggregate_usage"] = True
