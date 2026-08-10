"""Anthropic SDK adapter — production trace capture for raw Anthropic SDK tool loops.
"""

from __future__ import annotations

import functools
from typing import Any

from agenttest.adapters.base import BaseAdapter
from agenttest.core.result import AgentResult

__all__ = ["AnthropicAdapter"]


class AnthropicAdapter(BaseAdapter):
    """Adapter for raw Anthropic SDK tool loops."""

    def __init__(self) -> None:
        self._trace: list[dict[str, Any]] = []

    def wrap(self, agent_fn: Any) -> Any:
        import functools

        @functools.wraps(agent_fn)
        def patched(*args: Any, **kwargs: Any):
            try:
                import anthropic

                original_create = anthropic.Anthropic.messages.create

                def traced_create(self_client: Any, **kw: Any):
                    response = original_create(self_client, **kw)
                    usage = getattr(response, "usage", None)
                    step = {
                        "model": kw.get("model", "unknown"),
                        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                        "tool_calls": [
                            getattr(b, "name", "unknown") for b in response.content
                            if getattr(b, "type", None) == "tool_use"
                        ],
                        "output": " ".join(
                            getattr(b, "text", "") for b in response.content
                            if hasattr(b, "text")
                        ),
                    }
                    self._trace.append(step)
                    return response

                import unittest.mock as mock

                with mock.patch.object(
                    anthropic.Anthropic.messages.__class__,
                    "create",
                    traced_create,
                ):
                    return agent_fn(*args, **kwargs)
            except (ImportError, AttributeError):
                res = agent_fn(*args, **kwargs)
                out_text = getattr(res, "content", str(res))
                if isinstance(out_text, list):
                    out_text = " ".join(getattr(b, "text", str(b)) for b in out_text if hasattr(b, "text"))
                self._trace.append({
                    "model": "unknown",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tool_calls": [],
                    "output": str(out_text),
                })
                return res

        return patched

    def capture_trace(self) -> dict[str, Any]:
        return {"steps": self._trace}

    def build_result(self, trace: dict[str, Any], latency_ms: float) -> AgentResult:
        steps = trace.get("steps", [])
        tools_called: list[str] = []
        total_tokens = 0
        for step in steps:
            tools_called.extend(step.get("tool_calls", []))
            total_tokens += (
                step.get("input_tokens", 0) + step.get("output_tokens", 0)
            )
        final_output = steps[-1]["output"] if steps else ""
        return AgentResult(
            success=True,
            output=final_output,
            steps_taken=len(steps),
            tools_called=tools_called,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
            gracefully_degraded=False,
            raw_trace=trace,
        )
