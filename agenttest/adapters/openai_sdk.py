"""OpenAI SDK adapter — production trace capture for OpenAI SDK / Agents calls.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from agenttest.adapters.base import BaseAdapter
from agenttest.core.result import AgentResult

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI SDK agents and response generation."""

    def __init__(self) -> None:
        self._trace: list[dict[str, Any]] = []

    def wrap(self, agent: Any) -> Any:
        original_run = getattr(agent, "run", agent)

        if inspect.iscoroutinefunction(original_run):
            @functools.wraps(original_run)
            async def patched_run(input_text: Any, **kwargs: Any):
                result = await original_run(input_text, **kwargs)
                for event in getattr(result, "events", []):
                    if hasattr(event, "tool_calls"):
                        for tc in getattr(event, "tool_calls", []):
                            fn = getattr(tc, "function", None)
                            self._trace.append({
                                "tool": getattr(fn, "name", "unknown"),
                                "input": getattr(fn, "arguments", ""),
                                "output": "",
                            })
                final_out = getattr(result, "final_output", str(result))
                self._trace.append({
                    "agent": getattr(agent, "name", "agent"),
                    "output": str(final_out),
                })
                return result

            if hasattr(agent, "run"):
                agent.run = patched_run
                return agent
            return patched_run
        else:
            @functools.wraps(original_run)
            def patched_run_sync(*args: Any, **kwargs: Any):
                result = original_run(*args, **kwargs)
                if hasattr(result, "events"):
                    for event in getattr(result, "events", []):
                        if hasattr(event, "tool_calls"):
                            for tc in getattr(event, "tool_calls", []):
                                fn = getattr(tc, "function", None)
                                self._trace.append({
                                    "tool": getattr(fn, "name", "unknown"),
                                    "input": getattr(fn, "arguments", ""),
                                    "output": "",
                                })
                final_out = getattr(result, "final_output", str(result))
                self._trace.append({
                    "agent": getattr(agent, "name", "agent"),
                    "output": str(final_out),
                })
                return result

            if hasattr(agent, "run"):
                agent.run = patched_run_sync
                return agent
            return patched_run_sync

    def capture_trace(self) -> dict[str, Any]:
        return {"steps": self._trace}

    def build_result(self, trace: dict[str, Any], latency_ms: float) -> AgentResult:
        steps = trace.get("steps", [])
        tools_called = [
            s["tool"] for s in steps if "tool" in s
        ]
        agent_steps = [s for s in steps if "agent" in s]
        final_output = (
            agent_steps[-1]["output"] if agent_steps else (steps[-1]["output"] if steps else "")
        )
        return AgentResult(
            success=True,
            output=final_output,
            steps_taken=len(steps),
            tools_called=tools_called,
            tokens_used=0,
            latency_ms=latency_ms,
            gracefully_degraded=False,
            raw_trace=trace,
        )
