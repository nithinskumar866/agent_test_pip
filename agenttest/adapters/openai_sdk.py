"""OpenAI Agents SDK adapter.

An ``agents.Agent`` is not callable — it is run with ``Runner.run_sync``. The
adapter does that, then reads the run result: ``new_items`` gives tool calls
and their outputs, ``raw_responses`` gives one ``llm`` step per model response
with its token usage.

Tool mocks: for the duration of the run, each function tool reachable from
the agent (its tools, and the tools of agents it can hand off to) has its
``on_invoke_tool`` wrapped. A mocked failure goes through the tool's own
failure policy (default error message, custom ``failure_error_function``, or
re-raise) — so the agent sees what it would see for a real tool error — and
is recorded on the matching tool step.

Any other agent (a function wrapping the SDK, for example) is called as-is.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, ClassVar, Iterator

from agenttest.adapters.base import BaseAdapter, call_agent, text_of
from agenttest.core.trace import record_step
from agenttest.mocks._registry import active_mock, apply_mock_async

__all__ = ["OpenAIAdapter"]

#: Last-resort message for SDKs without a failure-policy helper.
_TOOL_ERROR_TEMPLATE = "An error occurred while running the tool. Please try again. Error: {error}"


def _item_type(item: Any) -> str | None:
    return getattr(item, "type", None)


def _record_run(result: Any, failures: dict[str, list[str]]) -> None:
    for response in getattr(result, "raw_responses", None) or []:
        usage = getattr(response, "usage", None)
        record_step(
            "llm",
            "model_response",
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            metadata={"response_id": getattr(response, "response_id", None)},
        )

    pending: dict[str, Any] = {}
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


def _take_failure(failures: dict[str, list[str]], call_id: str | None, name: str) -> str | None:
    """Pop the mocked failure for this call: by call id, else first by tool name."""
    if call_id and call_id in failures:
        return failures.pop(call_id)[0]
    queue = failures.get(f"name:{name}")
    return queue.pop(0) if queue else None


def _reachable_agents(agent: Any) -> list[Any]:
    seen: dict[int, Any] = {}
    stack = [agent]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen[id(current)] = current
        for handoff in getattr(current, "handoffs", None) or []:
            if hasattr(handoff, "tools"):  # an Agent, not a Handoff object
                stack.append(handoff)
    return list(seen.values())


@contextmanager
def _mock_function_tools(agent: Any, failures: dict[str, list[str]]) -> Iterator[None]:
    saved: list[tuple[Any, Any]] = []
    for owner in _reachable_agents(agent):
        for tool in getattr(owner, "tools", None) or []:
            original = getattr(tool, "on_invoke_tool", None)
            if original is None or getattr(original, "_agenttest_mock_wrapper", False):
                continue
            saved.append((tool, original))
            tool.on_invoke_tool = _wrap_invoke(tool, original, failures)
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


def _wrap_invoke(tool: Any, original: Any, failures: dict[str, list[str]]) -> Any:
    name = tool.name

    @functools.wraps(original)
    async def on_invoke_tool(ctx: Any, arguments: str) -> Any:
        try:
            return await apply_mock_async(name, lambda: original(ctx, arguments))
        except Exception as exc:
            handle = active_mock(name)
            if handle is None or handle.type != "failure":
                raise  # a real tool error: the SDK handles it as it normally would
            call_id = getattr(ctx, "tool_call_id", None)
            failures.setdefault(call_id if call_id else f"name:{name}", []).append(f"{type(exc).__name__}: {exc}")
            return await _handle_mocked_failure(tool, ctx, exc)

    on_invoke_tool._agenttest_mock_wrapper = True  # type: ignore[attr-defined]
    return on_invoke_tool


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI Agents SDK agents."""

    name: ClassVar[str] = "openai"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        module = type(agent).__module__
        return (module == "agents" or module.startswith("agents.")) and hasattr(agent, "instructions")

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.can_handle(agent):
            return call_agent(agent, *args, **kwargs)

        # The engine never calls invoke() on a running event loop (arun uses a
        # worker thread), so run_sync is always safe here.
        from agents import Runner

        failures: dict[str, list[str]] = {}
        with _mock_function_tools(agent, failures):
            result = Runner.run_sync(agent, *args, **kwargs)
        _record_run(result, failures)
        return result

    def extract_output(self, raw: Any) -> str:
        final = getattr(raw, "final_output", None)
        if final is not None:
            return final if isinstance(final, str) else str(final)
        return text_of(raw)
