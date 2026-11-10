"""OpenAI Agents SDK adapter.

An ``agents.Agent`` is not callable — it is run with ``Runner.run_sync``. The
adapter does that, then reads the run result: ``new_items`` gives tool calls
and their outputs, ``raw_responses`` gives one ``llm`` step per model response
with its token usage.

``arun`` uses the native ``Runner.run``.

Tool mocks: for the duration of the run, each function tool reachable from
the agent has its ``on_invoke_tool`` wrapped — the agent's own tools, those
of agents it hands off to (directly or through ``handoff()``), and those of
agents exposed with ``Agent.as_tool()``. Tools on MCP servers are mocked
where the SDK invokes them, by tool name.

``run(agent, input, stream=True)`` uses ``Runner.run_streamed`` and records the
time to the first event on the first ``llm`` step (``metadata["first_event_ms"]``).

A mocked failure goes through the tool's own failure policy (default error
message, custom ``failure_error_function``, or
re-raise) — so the agent sees what it would see for a real tool error — and
is recorded on the matching tool step; a mocked failure with no matching step
(a crashed run, or a tool of a nested ``as_tool`` agent) gets a step of its own.

Any other agent (a function wrapping the SDK, for example) is called as-is.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar
from unittest import mock

from agenttest.adapters.base import BaseAdapter, call_agent, text_of
from agenttest.core.trace import current_trace, record_step
from agenttest.mocks._registry import apply_mock_async, is_mocked_failure

__all__ = ["OpenAIAdapter"]

#: Mocked failures of the run in progress, for tools only reachable at run time.
_RUN_FAILURES: ContextVar[dict[str, list[tuple[str, str]]] | None] = ContextVar("agenttest_run_failures", default=None)

#: Last-resort message for SDKs without a failure-policy helper.
_TOOL_ERROR_TEMPLATE = "An error occurred while running the tool. Please try again. Error: {error}"


def _item_type(item: Any) -> str | None:
    return getattr(item, "type", None)


def _record_run(result: Any, failures: dict[str, list[tuple[str, str]]], model: str | None = None) -> None:
    for response in getattr(result, "raw_responses", None) or []:
        usage = getattr(response, "usage", None)
        record_step(
            "llm",
            "model_response",
            model=model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            metadata={"response_id": getattr(response, "response_id", None)},
        )

    pending: dict[str, Any] = {}
    try:
        _record_items(result, failures, pending)
    finally:
        _record_leftover_failures(failures)


def _record_leftover_failures(failures: dict[str, list[tuple[str, str]]]) -> None:
    for key, queue in failures.items():
        for name, error in queue:
            record_step(
                "tool",
                name,
                error=error,
                metadata={"call_id": None if key.startswith("name:") else key, "mocked": "failure", "unreported": True},
            )
    failures.clear()


def _record_items(result: Any, failures: dict[str, list[tuple[str, str]]], pending: dict[str, Any]) -> None:
    for item in getattr(result, "new_items", None) or []:
        raw = getattr(item, "raw_item", None)
        if _item_type(item) == "tool_call_item":
            name = getattr(raw, "name", None) or "unknown"
            call_id = getattr(raw, "call_id", None) or getattr(raw, "id", None)
            error = _take_failure(failures, call_id, name)
            step = record_step(
                "tool",
                name,
                input=getattr(raw, "arguments", None),
                error=error,
                metadata={"call_id": call_id, **({"mocked": "failure"} if error else {})},
            )
            if call_id and step is not None:
                pending[call_id] = step
        elif _item_type(item) == "tool_call_output_item":
            call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
            step = pending.get(call_id) if call_id else None
            if step is not None and step.error is None:
                step.output = getattr(item, "output", None)


def _take_failure(failures: dict[str, list[tuple[str, str]]], call_id: str | None, name: str) -> str | None:
    """Pop the mocked failure for this call: by call id, else first by tool name."""
    if call_id and call_id in failures:
        return failures.pop(call_id)[0][1]
    queue = failures.get(f"name:{name}")
    return queue.pop(0)[1] if queue else None


def _handoff_target(handoff: Any) -> Any:
    if hasattr(handoff, "tools"):  # an Agent listed directly
        return handoff
    ref = getattr(handoff, "_agent_ref", None)  # a Handoff built by handoff()
    return ref() if callable(ref) else None


def _walk(agent: Any) -> list[tuple[Any, bool]]:
    """``(agent, nested)`` for every reachable agent; nested means behind ``as_tool``."""
    seen: dict[int, tuple[Any, bool]] = {}
    stack = [(agent, False)]
    while stack:
        current, nested = stack.pop()
        if current is None or (id(current) in seen and (nested or not seen[id(current)][1])):
            continue
        seen[id(current)] = (current, nested)
        stack.extend((_handoff_target(h), nested) for h in getattr(current, "handoffs", None) or [])
        stack.extend((getattr(t, "_agent_instance", None), True) for t in getattr(current, "tools", None) or [])
    return list(seen.values())


def _reachable_agents(agent: Any) -> list[Any]:
    """The agent, agents it can hand off to, and agents wrapped by its tools."""
    return [current for current, _ in _walk(agent)]


@contextmanager
def _mock_function_tools(agent: Any, failures: dict[str, list[tuple[str, str]]]) -> Iterator[None]:
    saved: list[tuple[Any, Any]] = []
    for owner, nested in _walk(agent):
        for tool in getattr(owner, "tools", None) or []:
            original = getattr(tool, "on_invoke_tool", None)
            if original is None or getattr(original, "_agenttest_mock_wrapper", False):
                continue
            saved.append((tool, original))
            tool.on_invoke_tool = _wrap_invoke(tool, original, failures, nested)
    try:
        yield
    finally:
        for tool, original in reversed(saved):
            tool.on_invoke_tool = original


async def _handle_mocked_failure(tool: Any, ctx: Any, exc: Exception) -> Any:
    """Turn a mocked failure into what the SDK would do with a real one.

    Uses the tool's own failure policy: the SDK's default error message, the
    tool's custom ``failure_error_function``, or re-raising when that is None.
    """
    try:
        from agents.tool import maybe_invoke_function_tool_failure_error_function
    except ImportError:  # older SDKs
        try:
            from agents import default_tool_error_function
        except ImportError:
            return _TOOL_ERROR_TEMPLATE.format(error=exc)
        return default_tool_error_function(ctx, exc)
    result = await maybe_invoke_function_tool_failure_error_function(function_tool=tool, context=ctx, error=exc)
    if result is None:
        raise exc
    return result


def _wrap_invoke(tool: Any, original: Any, failures: dict[str, list[tuple[str, str]]], nested: bool = False) -> Any:
    name = tool.name

    @functools.wraps(original)
    async def on_invoke_tool(ctx: Any, arguments: str) -> Any:
        try:
            output = await apply_mock_async(name, lambda: original(ctx, arguments), input=arguments)
        except Exception as exc:
            if not is_mocked_failure(exc):
                raise  # a real tool error: the SDK handles it as it normally would
            call_id = getattr(ctx, "tool_call_id", None)
            key = call_id if call_id else f"name:{name}"
            failures.setdefault(key, []).append((name, f"{type(exc).__name__}: {exc}"))
            return await _handle_mocked_failure(tool, ctx, exc)
        if nested:  # the outer run's items don't show calls made inside an as_tool agent
            record_step("tool", name, input=arguments, output=output, metadata={"nested": True})
        return output

    on_invoke_tool._agenttest_mock_wrapper = True  # type: ignore[attr-defined]
    return on_invoke_tool


@dataclass
class _Session:
    result: Any = None
    first_event_ms: float | None = None


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI Agents SDK agents."""

    name: ClassVar[str] = "openai"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        module = type(agent).__module__
        return (module == "agents" or module.startswith("agents.")) and hasattr(agent, "instructions")

    @contextmanager
    def instrument(self) -> Iterator[None]:
        """Apply mocks to MCP tools, which only exist once the run has started."""
        try:
            from agents.mcp.util import MCPUtil
        except ImportError:
            yield
            return
        original = MCPUtil.__dict__["invoke_mcp_tool"].__func__

        async def invoke(cls: Any, server: Any, tool: Any, context: Any, input_json: str, **kwargs: Any) -> Any:
            try:
                return await apply_mock_async(
                    tool.name, lambda: original(cls, server, tool, context, input_json, **kwargs), input=input_json
                )
            except Exception as exc:
                failures = _RUN_FAILURES.get()
                if is_mocked_failure(exc) and failures is not None:
                    failures.setdefault(f"name:{tool.name}", []).append((tool.name, f"{type(exc).__name__}: {exc}"))
                raise

        with mock.patch.object(MCPUtil, "invoke_mcp_tool", classmethod(invoke)):
            yield

    @contextmanager
    def _session(self, agent: Any) -> Iterator[_Session]:
        """Mocks for one run, and recording what it did, even if it crashes."""
        failures: dict[str, list[tuple[str, str]]] = {}
        token = _RUN_FAILURES.set(failures)
        session = _Session()
        try:
            with _mock_function_tools(agent, failures):
                yield session
        except Exception as exc:
            # SDK exceptions carry what ran before the failure.
            session.result = getattr(exc, "run_data", None) or session.result
            raise
        finally:
            _RUN_FAILURES.reset(token)
            model = getattr(agent, "model", None)
            _record_run(session.result, failures, model if isinstance(model, str) else None)
            trace = current_trace()
            if session.first_event_ms is not None and trace is not None and trace.llm_calls:
                trace.llm_calls[0].metadata["first_event_ms"] = session.first_event_ms

    @staticmethod
    async def _stream(session: _Session, agent: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        from agents import Runner

        start = time.perf_counter()
        session.result = Runner.run_streamed(agent, *args, **kwargs)
        async for event in session.result.stream_events():
            if session.first_event_ms is None and event.type == "raw_response_event":
                session.first_event_ms = (time.perf_counter() - start) * 1000

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.can_handle(agent):
            return call_agent(agent, *args, **kwargs)

        # The engine never calls invoke() on a running event loop (arun uses
        # ainvoke), so run_sync is always safe here.
        from agents import Runner

        stream = kwargs.pop("stream", False)
        with self._session(agent) as session:
            if stream:
                asyncio.run(self._stream(session, agent, args, kwargs))
            else:
                session.result = Runner.run_sync(agent, *args, **kwargs)
        return session.result

    async def ainvoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.can_handle(agent):
            return await super().ainvoke(agent, *args, **kwargs)

        from agents import Runner

        stream = kwargs.pop("stream", False)
        with self._session(agent) as session:
            if stream:
                await self._stream(session, agent, args, kwargs)
            else:
                session.result = await Runner.run(agent, *args, **kwargs)
        return session.result

    def turn_arguments(self, history: list[dict[str, str]], previous: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if previous is not None and hasattr(previous, "to_input_list"):
            return ([*previous.to_input_list(), history[-1]],), {}
        return (history,), {}

    def extract_output(self, raw: Any) -> str:
        final = getattr(raw, "final_output", None)
        if final is not None:
            return final if isinstance(final, str) else str(final)
        return text_of(raw)
