"""CrewAI adapter — production trace capture for CrewAI agent and crew runs.
"""

from __future__ import annotations

import functools
from typing import Any

from agenttest.adapters.base import BaseAdapter
from agenttest.core.result import AgentResult

__all__ = ["CrewAIAdapter"]


class CrewAIAdapter(BaseAdapter):
    """Adapter for CrewAI crews and agents."""

    def __init__(self) -> None:
        self._trace: list[dict[str, Any]] = []

    def wrap(self, crew: Any) -> Any:
        original_kickoff = getattr(crew, "kickoff", None)
        if not original_kickoff:
            return crew

        @functools.wraps(original_kickoff)
        def patched_kickoff(inputs: Any = None):
            for agent in getattr(crew, "agents", []):
                original_execute = getattr(agent, "execute_task", None)
                if not original_execute:
                    continue

                def make_traced_execute(agent_name: str, original_fn: Any):
                    @functools.wraps(original_fn)
                    def traced_execute(task: Any, context: Any = None, tools: Any = None, **kwargs: Any):
                        if context is not None or tools is not None:
                            result = original_fn(task, context, tools, **kwargs)
                        else:
                            result = original_fn(task, **kwargs)

                        task_desc = getattr(task, "description", str(task))
                        tools_used = [
                            getattr(t, "name", str(t)) for t in (tools or [])
                        ]

                        self._trace.append({
                            "agent": agent_name,
                            "task": str(task_desc),
                            "output": str(result),
                            "tools_used": tools_used,
                        })
                        return result

                    return traced_execute

                role = getattr(agent, "role", "agent")
                agent.execute_task = make_traced_execute(role, original_execute)

            return original_kickoff(inputs) if inputs is not None else original_kickoff()

        crew.kickoff = patched_kickoff
        return crew

    def capture_trace(self) -> dict[str, Any]:
        return {"steps": self._trace}

    def build_result(self, trace: dict[str, Any], latency_ms: float) -> AgentResult:
        steps = trace.get("steps", [])
        tools_called: list[str] = []
        for step in steps:
            tools_called.extend(step.get("tools_used", []))
        final_output = steps[-1]["output"] if steps else ""
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
