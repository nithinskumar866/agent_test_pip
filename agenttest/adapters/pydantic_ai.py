"""Pydantic AI adapter — runs an ``Agent`` and reads its message history.

Tool calls are traced (and mocks applied) through ``FunctionToolset.call_tool``
for the duration of the run. Each model response becomes an ``llm`` step with
its token usage and model name, placed before the tools it asked for.

A mocked failure behaves like any exception raised by a real tool: Pydantic AI
lets it end the run. To simulate a recoverable failure instead, mock with
``exception=ModelRetry("try again")``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar
from unittest import mock

from agenttest.adapters.base import BaseAdapter, call_agent, text_of
from agenttest.core.trace import Step, arun_tool, current_trace

__all__ = ["PydanticAIAdapter"]


def _traced_call_tool(original: Any) -> Any:
    async def call_tool(self: Any, name: str, tool_args: dict[str, Any], ctx: Any, tool: Any) -> Any:
        return await arun_tool(name, lambda: original(self, name, tool_args, ctx, tool), input=tool_args)

    return call_tool


def _record(result: Any) -> None:
    """Interleave model responses with the tool steps already recorded, in call order."""
    trace = current_trace()
    if trace is None:
        return
    pending = [s for s in trace.steps if s.kind == "tool"]
    ordered = [s for s in trace.steps if s.kind != "tool"]
    for message in result.all_messages():
        if getattr(message, "kind", None) != "response":
            continue
        calls = [p.tool_name for p in message.parts if getattr(p, "part_kind", None) == "tool-call"]
        ordered.append(Step(
            kind="llm",
            name="model_response",
            model=getattr(message, "model_name", None),
            input_tokens=int(getattr(message.usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(message.usage, "output_tokens", 0) or 0),
            tool_requests=calls,
        ))
        for name in calls:
            match = next((s for s in pending if s.name == name), None)
            if match is not None:
                pending.remove(match)
                ordered.append(match)
    trace.steps[:] = [*ordered, *pending]


class PydanticAIAdapter(BaseAdapter):
    """Adapter for Pydantic AI agents."""

    name: ClassVar[str] = "pydantic-ai"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        return type(agent).__module__.startswith("pydantic_ai") and hasattr(agent, "run_sync")

    @contextmanager
    def instrument(self) -> Iterator[None]:
        try:
            from pydantic_ai.toolsets.function import FunctionToolset
        except ImportError:
            yield
            return
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(FunctionToolset, "call_tool", _traced_call_tool(FunctionToolset.call_tool))
            )
            yield

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.can_handle(agent):
            return call_agent(agent, *args, **kwargs)
        result = agent.run_sync(*args, **kwargs)
        _record(result)
        return result

    async def ainvoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.can_handle(agent):
            return await super().ainvoke(agent, *args, **kwargs)
        result = await agent.run(*args, **kwargs)
        _record(result)
        return result

    def extract_output(self, raw: Any) -> str:
        output = getattr(raw, "output", None)
        return output if isinstance(output, str) else text_of(raw) if output is None else str(output)

    def turn_arguments(self, history: list[dict[str, str]], previous: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        kwargs = {"message_history": previous.all_messages()} if previous is not None else {}
        return (history[-1]["content"],), kwargs
