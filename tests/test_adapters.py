"""Tests for BaseAdapter, GenericAdapter, LangGraphAdapter, CrewAIAdapter, OpenAIAdapter, AnthropicAdapter."""

from agenttest.adapters.base import BaseAdapter, GenericAdapter
from agenttest.adapters.langgraph import LangGraphAdapter
from agenttest.core.result import AgentResult


def test_generic_adapter_run_with_trace():
    adapter = GenericAdapter()

    def my_agent(prompt: str):
        return f"Response to {prompt}"

    result = adapter.run_with_trace(my_agent, "hello")
    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.output == "Response to hello"
    assert result.steps_taken == 1
    assert result.latency_ms >= 0.0


def test_generic_adapter_handles_exceptions():
    adapter = GenericAdapter()

    def failing_agent():
        raise ValueError("Agent internal failure")

    result = adapter.run_with_trace(failing_agent)
    assert isinstance(result, AgentResult)
    assert result.success is False
    assert "Agent internal failure" in (result.error or "")


def test_base_adapter_result_has_all_fields():
    class DummyAdapter(BaseAdapter):
        def wrap(self, agent):
            return agent

        def capture_trace(self):
            return {"steps": [{"output": "dummy"}]}

        def build_result(self, trace, latency_ms):
            return AgentResult(
                success=True,
                output="dummy",
                steps_taken=1,
                tools_called=["tool1"],
                tokens_used=10,
                latency_ms=latency_ms,
                gracefully_degraded=False,
                raw_trace=trace,
            )

    adapter = DummyAdapter()
    res = adapter.build_result(adapter.capture_trace(), 50.0)
    assert res.success is True
    assert res.output == "dummy"
    assert res.steps_taken == 1
    assert res.tools_called == ["tool1"]
    assert res.tokens_used == 10
    assert res.latency_ms == 50.0
    assert res.gracefully_degraded is False


def test_langgraph_adapter_builds_correct_result():
    class DummyGraph:
        def __init__(self):
            self.nodes = {
                "node_a": lambda s: {"messages": ["step1"]},
                "node_b": lambda s: {"output": "final answer"},
            }

        def invoke(self, state, config=None):
            res1 = self.nodes["node_a"](state)
            res2 = self.nodes["node_b"](res1)
            return res2

    adapter = LangGraphAdapter()
    graph = DummyGraph()
    wrapped_graph = adapter.wrap(graph)

    res = wrapped_graph.invoke({"input": "test"})
    assert res == {"output": "final answer"}

    trace = adapter.capture_trace()
    result = adapter.build_result(trace, 150.0)

    assert result.success is True
    assert result.steps_taken == 2
    assert result.output == "final answer"
