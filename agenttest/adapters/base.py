"""``BaseAdapter`` — the abstract contract every framework adapter implements.

Also hosts ``GenericAdapter``, the fallback used when no framework adapter
is explicitly configured or detected.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from agenttest.core.result import AgentResult

__all__ = ["BaseAdapter", "GenericAdapter"]


class BaseAdapter(ABC):
    """Abstract base adapter interface for capturing agent execution traces."""

    @abstractmethod
    def wrap(self, agent_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap an agent function to capture its execution trace."""
        pass

    @abstractmethod
    def capture_trace(self) -> dict[str, Any]:
        """Return raw execution trace after agent runs."""
        pass

    @abstractmethod
    def build_result(self, trace: dict[str, Any], latency_ms: float) -> AgentResult:
        """Convert raw trace into an AgentResult."""
        pass

    def run_with_trace(self, agent_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> AgentResult:
        """Template method — wraps run, executes function, captures trace, and builds result."""
        start = time.time()
        try:
            output = agent_fn(*args, **kwargs)
            trace = self.capture_trace()
            if not trace.get("output") and output is not None:
                trace["output"] = str(output)
            latency = (time.time() - start) * 1000
            return self.build_result(trace, latency)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return AgentResult(
                success=False,
                output="",
                steps_taken=0,
                tools_called=[],
                tokens_used=0,
                latency_ms=latency,
                gracefully_degraded=False,
                raw_trace={},
                error=str(e),
            )


class GenericAdapter(BaseAdapter):
    """Generic fallback adapter for plain Python agent functions."""

    def __init__(self) -> None:
        self._last_output: Any = None

    def wrap(self, agent_fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._last_output = agent_fn(*args, **kwargs)
            return self._last_output

        return wrapped

    def capture_trace(self) -> dict[str, Any]:
        output_str = str(self._last_output) if self._last_output is not None else ""
        return {"output": output_str, "steps": 1, "tools": []}

    def build_result(self, trace: dict[str, Any], latency_ms: float) -> AgentResult:
        output_str = trace.get("output", "")
        steps = trace.get("steps", 1)
        tools = trace.get("tools", [])
        return AgentResult(
            success=True,
            output=output_str,
            steps_taken=steps,
            tools_called=tools,
            tokens_used=0,
            latency_ms=latency_ms,
            gracefully_degraded=False,
            raw_trace=trace,
            error=None,
        )
