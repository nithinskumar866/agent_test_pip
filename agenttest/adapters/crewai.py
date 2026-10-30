"""CrewAI adapter — runs a crew with ``kickoff`` and records its task outputs.

Each task output becomes a ``node`` step named after the agent role that
produced it; the crew's aggregate token usage is attached to the last step.
Per-tool tracing for CrewAI arrives with the Phase 3 integration work.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agenttest.adapters.base import BaseAdapter, call_agent
from agenttest.core.trace import record_step

__all__ = ["CrewAIAdapter"]


class CrewAIAdapter(BaseAdapter):
    """Adapter for CrewAI crews."""

    name: ClassVar[str] = "crewai"

    @classmethod
    def can_handle(cls, agent: Any) -> bool:
        return type(agent).__module__.startswith("crewai") and hasattr(agent, "kickoff")

    def invoke(self, agent: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(agent, "kickoff"):
            return call_agent(agent, *args, **kwargs)
        if args and "inputs" not in kwargs:
            kwargs["inputs"], args = args[0], args[1:]
        result = agent.kickoff(*args, **kwargs)
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
