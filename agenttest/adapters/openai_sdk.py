"""OpenAI Agents SDK adapter.

An ``agents.Agent`` is not callable — it is run with ``Runner.run`` /
``Runner.run_sync``. The adapter does that, then reads the run result:
``new_items`` gives tool calls and their outputs, ``raw_responses`` gives one
``llm`` step per model response with its token usage.

Any other agent (a function wrapping the SDK, for example) is called as-is.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agenttest.adapters.base import BaseAdapter, call_agent, text_of
from agenttest.core.trace import record_step

__all__ = ["OpenAIAdapter"]


def _item_type(item: Any) -> str | None:
    return getattr(item, "type", None)


def _record_run(result: Any) -> None:
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
            call_id = getattr(raw, "call_id", None) or getattr(raw, "id", None)
            step = record_step(
                "tool",
                getattr(raw, "name", None) or "unknown",
                input=getattr(raw, "arguments", None),
                metadata={"call_id": call_id},
            )
            if call_id and step is not None:
                pending[call_id] = step
        elif _item_type(item) == "tool_call_output_item":
            call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
            step = pending.get(call_id) if call_id else None
            if step is not None:
                step.output = getattr(item, "output", None)


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

        result = Runner.run_sync(agent, *args, **kwargs)
        _record_run(result)
        return result

    def extract_output(self, raw: Any) -> str:
        final = getattr(raw, "final_output", None)
        if final is not None:
            return final if isinstance(final, str) else str(final)
        return text_of(raw)
