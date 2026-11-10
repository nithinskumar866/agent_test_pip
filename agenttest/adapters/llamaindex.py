"""LlamaIndex adapter — runs workflow agents (``ReActAgent``, ``FunctionAgent``, ``AgentWorkflow``).

Tool calls are traced (and mocks applied) by patching ``FunctionTool`` for the
duration of the run. LlamaIndex catches tool exceptions and shows them to the
model, so a mocked failure can be recovered from, like a real one.

LlamaIndex does not report token usage on the result, so ``tokens_used`` is 0.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar
from unittest import mock

from agenttest.adapters.base import BaseAdapter, call_agent, text_of
from agenttest.core.trace import arun_tool, run_tool

__all__ = ["LlamaIndexAdapter"]


def _tool_output(tool: Any, value: Any, kwargs: dict[str, Any]) -> Any:
    from llama_index.core.tools import ToolOutput

    return ToolOutput(content=str(value), tool_name=tool.metadata.name, raw_input={"kwargs": kwargs}, raw_output=value)


def _traced_call(original: Any) -> Any:
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        real: list[Any] = []

        def call_real() -> Any:
            real.append(original(self, *args, **kwargs))
            return real[-1].content

        value = run_tool(self.metadata.name, call_real, input=kwargs or list(args))
        return real[-1] if real else _tool_output(self, value, kwargs)

    return call


def _traced_acall(original: Any) -> Any:
    async def acall(self: Any, *args: Any, **kwargs: Any) -> Any:
        real: list[Any] = []

        async def call_real() -> Any:
            real.append(await original(self, *args, **kwargs))
            return real[-1].content

        value = await arun_tool(self.metadata.name, call_real, input=kwargs or list(args))
        return real[-1] if real else _tool_output(self, value, kwargs)

    return acall


async def _run(agent: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args and "user_msg" not in kwargs:
        kwargs = {**kwargs, "user_msg": args[0]}
        args = args[1:]
    return await agent.run(*args, **kwargs)


class LlamaIndexAdapter(BaseAdapter):
    """Adapter for LlamaIndex workflow agents."""

    name: ClassVar[str] = "llamaindex"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        return type(agent).__module__.startswith("llama_index") and hasattr(agent, "run")

    @contextmanager
    def instrument(self) -> Iterator[None]:
        try:
            from llama_index.core.tools import FunctionTool
        except ImportError:
            yield
            return
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(FunctionTool, "call", _traced_call(FunctionTool.call)))
            stack.enter_context(mock.patch.object(FunctionTool, "acall", _traced_acall(FunctionTool.acall)))
            yield

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.can_handle(agent):
            return call_agent(agent, *args, **kwargs)
        # Workflows need a running loop from the moment they start.
        return asyncio.run(_run(agent, args, kwargs))

    async def ainvoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.can_handle(agent):
            return await super().ainvoke(agent, *args, **kwargs)
        return await _run(agent, args, kwargs)

    def extract_output(self, raw: Any) -> str:
        message = getattr(raw, "response", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else text_of(raw)

    def turn_arguments(self, history: list[dict[str, str]], previous: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        from llama_index.core.llms import ChatMessage

        earlier = [ChatMessage(role=m["role"], content=m["content"]) for m in history[:-1]]
        return (history[-1]["content"],), {"chat_history": earlier}
