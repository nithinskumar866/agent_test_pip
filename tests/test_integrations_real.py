"""Integration tests against the real framework SDKs (skipped when not installed).

No network and no LLM: LangGraph runs a scripted agent node, the OpenAI
Agents SDK runs a scripted ``Model``. These exercise the frameworks' real
tool-execution paths, which is where agenttest hooks in.
"""


import time
from typing import Any

import pytest

from agenttest import mock_tool_failure, mock_tool_latency, mock_tool_response, run


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# =============================================================================
# LangGraph (langgraph + langchain-core)
# =============================================================================

def _build_graph(handle_tool_errors: Any = True):
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import tool as lc_tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    @lc_tool
    def web_search(query: str) -> str:
        """Search the web."""
        return f"real results for {query}"

    def agent(state: MessagesState) -> dict:
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            prefix = "Search failed, answering from memory" if last.status == "error" else "Found"
            return {"messages": [AIMessage(content=f"{prefix}: {last.content}",
                                           usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28})]}
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "news"}, "id": "call-1"}],
                                       usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})]}

    def route(state: MessagesState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([web_search], handle_tool_errors=handle_tool_errors))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()


def test_langgraph_real_graph_is_traced():
    result = run(_build_graph(), {"messages": [("user", "latest news?")]})

    assert result.success, result.traceback
    assert result.output == "Found: real results for news"
    assert result.tools_called == ["web_search"]
    assert [s.name for s in result.trace.steps if s.kind == "node"] == ["agent", "tools", "agent"]
    assert result.tokens_used == 42


def test_langgraph_failure_mock_becomes_error_tool_message():
    with mock_tool_failure("web_search", "503 Service Unavailable") as handle:
        result = run(_build_graph(), {"messages": [("user", "news?")]})

    assert handle.calls == 1
    assert result.success, result.traceback
    assert result.output.startswith("Search failed, answering from memory")
    assert "503 Service Unavailable" in result.trace.failed_tool_calls[0].error
    assert result.gracefully_degraded is True


def test_langgraph_failure_mock_without_error_handling_crashes_graph():
    # Same as a real tool failure when the graph does not handle tool errors.
    with mock_tool_failure("web_search", "boom"):
        result = run(_build_graph(handle_tool_errors=False), {"messages": [("user", "news?")]})

    assert result.success is False
    assert "boom" in result.error
    assert result.gracefully_degraded is False


def test_langgraph_response_mock_feeds_canned_output():
    with mock_tool_response("web_search", "canned headlines"):
        result = run(_build_graph(), {"messages": [("user", "news?")]})

    assert result.output == "Found: canned headlines"
    assert result.trace.tool_calls[0].output == "canned headlines"


def test_langgraph_latency_mock_slows_real_tool():
    with mock_tool_latency("web_search", 120):
        result = run(_build_graph(), {"messages": [("user", "news?")]})

    assert result.output == "Found: real results for news"
    assert result.latency_ms >= 110


def test_langgraph_async_tool_invocation_is_mocked_once():
    pytest.importorskip("langchain_core")
    import asyncio

    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def lookup(q: str) -> str:
        """Look something up."""
        return "real"

    async def agent():
        return await lookup.ainvoke({"q": "x"})  # sync tool: _arun falls back to _run

    from agenttest import arun

    with mock_tool_latency("lookup", 1) as handle:
        result = asyncio.run(arun(agent, adapter="langgraph"))
    assert result.output == "real"
    assert handle.calls == 1


def test_langchain_tool_classes_are_restored_after_run():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import BaseTool, StructuredTool

    before = (BaseTool.run, BaseTool.arun, StructuredTool.__dict__.get("_run"), StructuredTool.__dict__.get("_arun"))
    with mock_tool_response("web_search", "x"):
        run(_build_graph(), {"messages": [("user", "q")]})
    after = (BaseTool.run, BaseTool.arun, StructuredTool.__dict__.get("_run"), StructuredTool.__dict__.get("_arun"))
    assert before == after


# =============================================================================
# OpenAI Agents SDK
# =============================================================================

def _scripted_model():
    pytest.importorskip("agents")
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

    def message(text: str) -> ResponseOutputMessage:
        return ResponseOutputMessage(
            id="msg_1", role="assistant", status="completed", type="message",
            content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
        )

    class ScriptedModel(Model):
        """Calls get_weather once, then answers with whatever the tool returned."""

        async def get_response(self, system_instructions, input, *args: Any, **kwargs: Any) -> ModelResponse:
            items = input if isinstance(input, list) else []
            outputs = [i for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"]
            if outputs:
                output = [message(f"Answer based on: {outputs[-1]['output']}")]
            else:
                output = [ResponseFunctionToolCall(type="function_call", id="fc_1", call_id="call_1",
                                                   name="get_weather", arguments='{"city": "Paris"}', status="completed")]
            return ModelResponse(output=output, usage=Usage(requests=1, input_tokens=30, output_tokens=10, total_tokens=40),
                                 response_id=None)

        def stream_response(self, *args: Any, **kwargs: Any):  # pragma: no cover - not used
            raise NotImplementedError

    return ScriptedModel()


def _weather_agent(**tool_kwargs: Any):
    pytest.importorskip("agents")
    from agents import Agent, function_tool, set_tracing_disabled

    set_tracing_disabled(True)

    @function_tool(**tool_kwargs)
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    return Agent(name="weather", instructions="Use tools.", tools=[get_weather], model=_scripted_model())


def test_openai_real_runner_is_traced():
    agent = _weather_agent()
    result = run(agent, "Weather in Paris?")

    assert result.success, result.traceback
    assert result.output == "Answer based on: sunny in Paris"
    assert result.tools_called == ["get_weather"]
    assert result.trace.tool_calls[0].output == "sunny in Paris"
    assert result.tokens_used == 80


def test_openai_failure_mock_uses_sdk_default_error_message():
    agent = _weather_agent()
    with mock_tool_failure("get_weather", "weather API down") as handle:
        result = run(agent, "Weather in Paris?")

    assert handle.calls == 1
    assert result.success, result.traceback
    from agents import default_tool_error_function

    assert result.output == f"Answer based on: {default_tool_error_function(None, RuntimeError())}"
    assert result.trace.tool_calls[0].error == "MockedToolError: weather API down"
    assert result.gracefully_degraded is True


def test_openai_failure_mock_honours_custom_failure_function():
    agent = _weather_agent(failure_error_function=lambda ctx, err: f"custom: {err}")
    with mock_tool_failure("get_weather", "quota exceeded"):
        result = run(agent, "Weather in Paris?")

    assert result.output == "Answer based on: custom: quota exceeded"


def test_openai_failure_mock_reraises_when_failure_function_is_none():
    agent = _weather_agent(failure_error_function=None)
    with mock_tool_failure("get_weather", "fatal"):
        result = run(agent, "Weather in Paris?")

    assert result.success is False
    assert "fatal" in (result.error or "") + (result.traceback or "")


def test_openai_response_mock():
    agent = _weather_agent()
    with mock_tool_response("get_weather", "snowing"):
        result = run(agent, "Weather in Paris?")

    assert result.output == "Answer based on: snowing"


def test_openai_tools_restored_after_run():
    agent = _weather_agent()
    original = agent.tools[0].on_invoke_tool
    with mock_tool_failure("get_weather"):
        run(agent, "q")
    assert agent.tools[0].on_invoke_tool is original


# =============================================================================
# CrewAI (tool layer only: a full crew needs a real LLM)
# =============================================================================

def _crew_tool():
    pytest.importorskip("crewai")
    from crewai.tools import tool as crew_tool

    @crew_tool("web_search")
    def web_search(query: str) -> str:
        """Search the web."""
        return f"real results for {query}"

    return web_search


def test_crewai_structured_tool_calls_are_traced_and_mocked():
    structured = _crew_tool().to_structured_tool()

    def agent():
        return structured.invoke({"query": "ai"})

    assert run(agent, adapter="crewai").tools_called == ["web_search"]

    with mock_tool_response("web_search", "canned") as handle:
        result = run(agent, adapter="crewai")
    assert result.output == "canned"
    assert handle.calls == 1

    with mock_tool_failure("web_search", "down"):
        result = run(agent, adapter="crewai")
    assert result.success is False
    assert result.trace.failed_tool_calls[0].name == "web_search"


def test_crewai_base_tool_run_is_mocked_once():
    crew_tool = _crew_tool()
    with mock_tool_latency("web_search", 1) as handle:
        result = run(lambda: crew_tool.run(query="x"), adapter="crewai")
    assert result.output == "real results for x"
    assert handle.calls == 1
    assert result.tools_called == ["web_search"]


def test_crewai_classes_restored_after_run():
    pytest.importorskip("crewai")
    from crewai.tools import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool

    before = (CrewStructuredTool.invoke, CrewStructuredTool.ainvoke, BaseTool.run)
    run(lambda: None, adapter="crewai")
    assert (CrewStructuredTool.invoke, CrewStructuredTool.ainvoke, BaseTool.run) == before


def test_timing_sanity():
    # Guard against the latency assertions above passing vacuously.
    start = time.perf_counter()
    time.sleep(0.01)
    assert time.perf_counter() - start >= 0.009
