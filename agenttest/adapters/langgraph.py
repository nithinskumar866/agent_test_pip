"""LangGraph adapter — captures traces by wrapping graph node execution.

Intercepts LangGraph execution to record node inputs/outputs, tools invoked,
and builds an AgentResult.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from agenttest.adapters.base import BaseAdapter
from agenttest.core.result import AgentResult
from agenttest.mocks._registry import MOCK_REGISTRY

__all__ = ["LangGraphAdapter"]


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangGraph pipelines."""

    def __init__(self) -> None:
        self._trace: list[dict[str, Any]] = []

    def wrap(self, graph: Any) -> Any:
        """Wrap a LangGraph object (or runnable graph) to capture node traces."""
        if not hasattr(graph, "invoke"):
            # Plain callable wrapper fallback
            return graph

        original_invoke = graph.invoke

        @functools.wraps(original_invoke)
        def patched_invoke(state: Any, config: Any = None) -> Any:
            self._trace.clear()
            if hasattr(graph, "nodes") and isinstance(graph.nodes, dict):
                for node_name, node_runnable in graph.nodes.items():
                    original_runnable = node_runnable

                    def make_traced(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
                        def traced_fn(s: Any, c: Any = None) -> Any:
                            # Check mock registry for tool mocks
                            if name in MOCK_REGISTRY:
                                mock_info = MOCK_REGISTRY[name]
                                if mock_info.get("type") == "failure":
                                    raise RuntimeError(mock_info.get("error", f"Tool '{name}' failed"))
                                elif mock_info.get("type") == "response":
                                    res = mock_info.get("data", {})
                                    self._trace.append({"node": name, "input": dict(s) if hasattr(s, "keys") else str(s), "output": res})
                                    return res
                                elif mock_info.get("type") == "latency":
                                    time.sleep(mock_info.get("delay", 0))

                            result = fn(s, c) if c is not None else fn(s)
                            self._trace.append({
                                "node": name,
                                "input": dict(s) if hasattr(s, "keys") else str(s),
                                "output": dict(result) if hasattr(result, "keys") else str(result),
                            })
                            return result

                        return traced_fn

                    graph.nodes[node_name] = make_traced(node_name, original_runnable)

            return original_invoke(state, config) if config is not None else original_invoke(state)

        graph.invoke = patched_invoke
        return graph

    def capture_trace(self) -> dict[str, Any]:
        return {"steps": list(self._trace)}

    def build_result(self, trace: dict[str, Any], latency_ms: float) -> AgentResult:
        steps = trace.get("steps", [])
        tools_called: list[str] = []
        final_output = ""

        for step in steps:
            output = step.get("output", {})
            if isinstance(output, dict) and "tool_calls" in output:
                for t in output["tool_calls"]:
                    if isinstance(t, dict) and "name" in t:
                        tools_called.append(t["name"])
                    elif isinstance(t, str):
                        tools_called.append(t)

        if steps:
            last_output = steps[-1].get("output", {})
            if isinstance(last_output, dict):
                if "messages" in last_output and last_output["messages"]:
                    last_msg = last_output["messages"][-1]
                    final_output = getattr(last_msg, "content", str(last_msg))
                elif "output" in last_output:
                    final_output = str(last_output["output"])
                else:
                    final_output = str(last_output)
            else:
                final_output = str(last_output)

        return AgentResult(
            success=True,
            output=final_output,
            steps_taken=len(steps),
            tools_called=tools_called,
            tokens_used=0,
            latency_ms=latency_ms,
            gracefully_degraded=False,
            raw_trace=trace,
            error=None,
        )
