"""LangGraph adapter — traces a compiled graph by streaming its node updates.

Rather than replacing nodes on the user's graph (which breaks compiled
``PregelNode`` objects and leaks across runs), the adapter runs the graph with
``stream(stream_mode="updates")``: each yielded update is one node execution.
Tool executions are read from the ``ToolMessage`` objects that tool nodes add
to the state.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from agenttest.adapters.base import BaseAdapter, call_agent
from agenttest.core.trace import record_step

__all__ = ["LangGraphAdapter"]


def _messages_in(update: Any) -> list[Any]:
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list):
            return messages
        if messages is not None:
            return [messages]
    return []


def _usage_of(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


class LangGraphAdapter(BaseAdapter):
    """Adapter for compiled LangGraph graphs."""

    name: ClassVar[str] = "langgraph"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        return type(agent).__module__.startswith("langgraph") and hasattr(agent, "stream")

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(agent, "stream"):
            return call_agent(agent, *args, **kwargs)

        state: dict[str, Any] = {}
        last = time.perf_counter()
        for chunk in agent.stream(*args, stream_mode="updates", **kwargs):
            now = time.perf_counter()
            if not isinstance(chunk, dict):
                continue
            for node, update in chunk.items():
                self._record_node(node, update, (now - last) * 1000)
                if isinstance(update, dict):
                    state.update(update)
            last = now
        return state

    def _record_node(self, node: str, update: Any, duration_ms: float) -> None:
        messages = _messages_in(update)
        input_tokens = output_tokens = 0
        requested: list[str] = []
        for message in messages:
            i, o = _usage_of(message)
            input_tokens += i
            output_tokens += o
            for call in getattr(message, "tool_calls", None) or []:
                tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if tool_name:
                    requested.append(tool_name)

        record_step(
            "node",
            node,
            output=update,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_requests=requested,
        )
        for message in messages:
            if getattr(message, "type", None) == "tool":
                failed = getattr(message, "status", None) == "error"
                content = getattr(message, "content", None)
                record_step(
                    "tool",
                    getattr(message, "name", None) or "unknown",
                    output=None if failed else content,
                    error=str(content) if failed else None,
                    metadata={"tool_call_id": getattr(message, "tool_call_id", None), "node": node},
                )
