"""LangGraph adapter — traces a compiled graph by streaming its node updates.

Rather than replacing nodes on the user's graph (which breaks compiled
``PregelNode`` objects and leaks across runs), the adapter runs the graph with
``stream(stream_mode="updates")``: each yielded update is one node execution.
Tool executions are read from the ``ToolMessage`` objects that tool nodes add
to the state, and tool mocks are applied to LangChain tools during the run.

Whether a mocked tool failure crashes the graph or becomes an error
``ToolMessage`` is decided by the graph itself (``ToolNode(handle_tool_errors=...)``),
exactly as for a real failure.

LangChain tool runs that produce no ``ToolMessage`` are recorded from the
tool-run log instead: a tool called directly inside a node is recorded after
that node's step, and a failure that crashed the graph is recorded before the
run ends. ``arun`` drives the graph with ``astream``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, ClassVar, Iterator

from agenttest.adapters._langchain_tools import ToolRun, current_tool_log, langchain_tool_mocks
from agenttest.adapters.base import BaseAdapter, call_agent
from agenttest.core.trace import current_trace, record_step

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

    @contextmanager
    def instrument(self) -> Iterator[None]:
        with langchain_tool_mocks():
            yield

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(agent, "stream"):
            return call_agent(agent, *args, **kwargs)
        run = _GraphRun(self)
        try:
            for chunk in agent.stream(*args, stream_mode="updates", **kwargs):
                run.on_chunk(chunk)
        finally:
            run.close()
        return run.state

    async def ainvoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(agent, "astream"):
            return await super().ainvoke(agent, *args, **kwargs)
        run = _GraphRun(self)
        try:
            async for chunk in agent.astream(*args, stream_mode="updates", **kwargs):
                run.on_chunk(chunk)
        finally:
            run.close()
        return run.state

    def _record_node(self, node: str, update: Any, duration_ms: float) -> set[str]:
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

        tool_call_ids: set[str] = set()
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
                if getattr(message, "tool_call_id", None):
                    tool_call_ids.add(message.tool_call_id)
                failed = getattr(message, "status", None) == "error"
                content = getattr(message, "content", None)
                record_step(
                    "tool",
                    getattr(message, "name", None) or "unknown",
                    output=None if failed else content,
                    error=str(content) if failed else None,
                    metadata={"tool_call_id": getattr(message, "tool_call_id", None), "node": node},
                )
        return tool_call_ids


def _record_tool_run(run: ToolRun, **metadata: Any) -> None:
    record_step(
        "tool",
        run.name,
        input=run.input,
        output=run.output,
        error=run.error,
        duration_ms=run.duration_ms,
        metadata={"tool_call_id": run.tool_call_id, **run.metadata, **metadata},
    )


def _annotate_tool_step(run: ToolRun) -> None:
    """Add what the tool-run log knows (input, mock type) to the ToolMessage step."""
    trace = current_trace()
    if trace is None:
        return
    for step in reversed(trace.tool_calls):
        if step.metadata.get("tool_call_id") == run.tool_call_id:
            if step.input is None:
                step.input = run.input
            step.metadata.update(run.metadata)
            return


class _GraphRun:
    """Turns one stream of node updates into trace steps."""

    def __init__(self, adapter: LangGraphAdapter) -> None:
        self.adapter = adapter
        self.state: dict[str, Any] = {}
        self.log = current_tool_log()
        self.seen_ids: set[str] = set()
        self.last = time.perf_counter()

    def on_chunk(self, chunk: Any) -> None:
        now = time.perf_counter()
        if not isinstance(chunk, dict):
            return
        for node, update in chunk.items():
            self.seen_ids |= self.adapter._record_node(node, update, (now - self.last) * 1000)
            if isinstance(update, dict):
                self.state.update(update)
            # Runs without a tool_call_id never get a ToolMessage: they were
            # called directly by the node that just finished.
            if self.log is not None:
                for run in self.log.take(lambda r: r.tool_call_id is None or r.tool_call_id in self.seen_ids):
                    if run.tool_call_id is None:
                        _record_tool_run(run, node=node)
                    else:
                        _annotate_tool_step(run)
        self.last = now

    def close(self) -> None:
        """Record tool runs whose ToolMessage never arrived (e.g. the graph crashed)."""
        if self.log is None:
            return
        for run in self.log.take():
            if run.tool_call_id not in self.seen_ids:
                _record_tool_run(run, unreported=True)
