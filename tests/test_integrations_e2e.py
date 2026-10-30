"""End-to-end runs of real framework agent loops (skipped when an SDK is missing).

Every framework below runs its *own* agent loop — LangGraph's prebuilt ReAct
agent, the OpenAI Agents SDK ``Runner`` with handoffs and agents-as-tools,
a full CrewAI ``Crew``, and the Anthropic client over HTTP. Only the model is
replaced (a scripted chat model, ``Model``, ``BaseLLM``, or an HTTP transport),
so nothing touches the network and the results are deterministic.
"""


import asyncio
import json
from typing import Any

import pytest

import agenttest
from agenttest import AgentTestTimeoutError, arun, mock_tool_failure, mock_tool_response, run


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# =============================================================================
# LangGraph
# =============================================================================

def _search_tool():
    pytest.importorskip("langgraph")
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def web_search(query: str) -> str:
        """Search the web."""
        return f"real results for {query}"

    return web_search


def _tool_calling_chat_model():
    """A real LangChain chat model class whose replies are scripted."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    class ToolCallingFake(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingFake":
            return self

    replies = iter([
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "ai"}, "id": "call-1"}],
                  usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}),
        AIMessage(content="Here is what I found.",
                  usage_metadata={"input_tokens": 30, "output_tokens": 6, "total_tokens": 36}),
    ])
    return ToolCallingFake(messages=replies)


def _react_agent():
    web_search = _search_tool()
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(_tool_calling_chat_model(), [web_search])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_langgraph_prebuilt_react_agent_with_real_chat_model():
    with mock_tool_response("web_search", "canned headlines") as handle:
        result = run(_react_agent(), {"messages": [("user", "news?")]})

    assert result.success, result.traceback
    assert result.output == "Here is what I found."
    assert handle.calls == 1
    assert result.tools_called == ["web_search"]
    assert result.tokens_used == 53
    step = result.trace.tool_calls[0]
    assert step.output == "canned headlines"
    assert step.metadata["mocked"] == "response"
    assert "ai" in str(step.input)


def _async_graph(handle_tool_errors: Any = True):
    web_search = _search_tool()
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    async def agent(state: MessagesState) -> dict:  # async-only: sync stream() cannot run it
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            return {"messages": [AIMessage(content=f"Found: {last.content}")]}
        return {"messages": [AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": "c1"}])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode([web_search], handle_tool_errors=handle_tool_errors))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile()


def test_langgraph_arun_streams_natively_with_astream():
    graph = _async_graph()
    with mock_tool_response("web_search", "async canned") as handle:
        result = asyncio.run(arun(graph, {"messages": [("user", "q")]}))

    assert result.success, result.traceback
    assert result.output == "Found: async canned"
    assert handle.calls == 1
    assert [s.name for s in result.trace.steps if s.kind == "node"] == ["agent", "tools", "agent"]


def test_langgraph_arun_timeout_cancels_the_graph():
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, MessagesState, StateGraph

    async def slow(state: MessagesState) -> dict:
        await asyncio.sleep(5)
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("slow", slow)
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)

    result = asyncio.run(arun(builder.compile(), {"messages": []}, timeout=0.2))
    assert result.success is False
    with pytest.raises(AgentTestTimeoutError):
        result.raise_for_error()
    assert result.latency_ms < 2000


def test_langgraph_crash_from_mocked_failure_is_still_traced():
    graph = _async_graph(handle_tool_errors=False)
    with mock_tool_failure("web_search", "boom"):
        result = asyncio.run(arun(graph, {"messages": [("user", "q")]}))

    assert result.success is False
    assert result.tools_called == ["web_search"]
    step = result.trace.failed_tool_calls[0]
    assert step.error == "MockedToolError: boom"
    assert step.metadata["unreported"] is True
    assert step.metadata["mocked"] == "failure"


def test_langgraph_tool_called_directly_inside_a_node_is_traced():
    web_search = _search_tool()
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    def research(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content=web_search.invoke({"query": "direct"}))]}

    builder = StateGraph(MessagesState)
    builder.add_node("research", research)
    builder.add_edge(START, "research")
    builder.add_edge("research", END)

    result = run(builder.compile(), {"messages": [("user", "q")]})
    assert result.output == "real results for direct"
    assert result.tools_called == ["web_search"]
    assert result.trace.tool_calls[0].metadata["node"] == "research"
    # The tool step follows the node that called it.
    assert [s.kind for s in result.trace.steps] == ["node", "tool"]


# =============================================================================
# OpenAI Agents SDK
# =============================================================================

def _scripted(decide: Any):
    """A real ``Model`` subclass; ``decide(input_items)`` returns output items."""
    pytest.importorskip("agents")
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage

    class Scripted(Model):
        async def get_response(self, system_instructions, input, *args: Any, **kwargs: Any) -> ModelResponse:
            items = input if isinstance(input, list) else []
            return ModelResponse(output=await _maybe_await(decide(items)),
                                 usage=Usage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
                                 response_id=None)

        def stream_response(self, *args: Any, **kwargs: Any):  # pragma: no cover - not used
            raise NotImplementedError

    return Scripted()


async def _maybe_await(value: Any) -> Any:
    return await value if asyncio.iscoroutine(value) else value


def _message(text: str) -> Any:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    return ResponseOutputMessage(id="msg", role="assistant", status="completed", type="message",
                                 content=[ResponseOutputText(type="output_text", text=text, annotations=[])])


def _call(name: str, call_id: str, arguments: dict[str, Any]) -> Any:
    from openai.types.responses import ResponseFunctionToolCall

    return ResponseFunctionToolCall(type="function_call", id=f"fc_{call_id}", call_id=call_id, name=name,
                                    arguments=json.dumps(arguments), status="completed")


def _outputs(items: list[Any]) -> dict[str, str]:
    return {i["call_id"]: i["output"] for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"}


def _tool_then_answer(tool_name: str, call_id: str, arguments: dict[str, Any] | None = None):
    def decide(items: list[Any]) -> list[Any]:
        outputs = _outputs(items)
        if call_id in outputs:
            return [_message(f"Answer based on: {outputs[call_id]}")]
        return [_call(tool_name, call_id, arguments or {"city": "Paris"})]

    return decide


def _weather_tool():
    from agents import function_tool, set_tracing_disabled

    set_tracing_disabled(True)

    @function_tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"sunny in {city}"

    return get_weather


def test_openai_arun_uses_native_runner(monkeypatch):
    pytest.importorskip("agents")
    from agents import Agent, Runner

    agent = Agent(name="weather", instructions="Use tools.", tools=[_weather_tool()],
                  model=_scripted(_tool_then_answer("get_weather", "c1")))

    def no_sync(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("arun must not use Runner.run_sync")

    monkeypatch.setattr(Runner, "run_sync", no_sync)
    result = asyncio.run(arun(agent, "Weather?"))

    assert result.success, result.traceback
    assert result.output == "Answer based on: sunny in Paris"
    assert result.tools_called == ["get_weather"]


def test_openai_mocks_reach_agents_behind_handoff_objects():
    pytest.importorskip("agents")
    from agents import Agent, handoff

    specialist = Agent(name="specialist", instructions="Use tools.", tools=[_weather_tool()],
                       model=_scripted(_tool_then_answer("get_weather", "w1")))
    transfer = handoff(specialist)
    triage = Agent(name="triage", instructions="Route.", handoffs=[transfer],
                   model=_scripted(lambda items: [_call(transfer.tool_name, "h1", {})]))

    with mock_tool_response("get_weather", "raining") as handle:
        result = run(triage, "Weather?")

    assert result.success, result.traceback
    assert handle.calls == 1
    assert result.output == "Answer based on: raining"
    assert "get_weather" in result.tools_called


def test_openai_mocks_reach_agents_exposed_as_tools():
    pytest.importorskip("agents")
    from agents import Agent

    forecaster = Agent(name="forecaster", instructions="Use tools.", tools=[_weather_tool()],
                       model=_scripted(_tool_then_answer("get_weather", "w1")))
    orchestrator = Agent(
        name="orchestrator", instructions="Delegate.",
        tools=[forecaster.as_tool(tool_name="ask_forecaster", tool_description="Ask the forecaster.")],
        model=_scripted(_tool_then_answer("ask_forecaster", "o1", {"input": "Weather in Paris?"})),
    )

    with mock_tool_failure("get_weather", "station offline") as handle:
        result = run(orchestrator, "Weather?")

    assert result.success, result.traceback
    assert handle.calls == 1
    assert result.tools_called[0] == "ask_forecaster"
    # The nested failure has no step in the outer run's items, so it gets its own.
    nested = [s for s in result.trace.failed_tool_calls if s.name == "get_weather"]
    assert nested and nested[0].metadata["unreported"] is True
    assert result.gracefully_degraded is True


def _slow_agent():
    pytest.importorskip("agents")
    from agents import Agent, set_tracing_disabled

    async def slow(items: list[Any]) -> list[Any]:
        await asyncio.sleep(3)
        return [_message("too late")]

    set_tracing_disabled(True)
    return Agent(name="slow", instructions="-", model=_scripted(slow))


def test_openai_run_timeout_with_runner_in_worker_thread():
    result = run(_slow_agent(), "hi", timeout=0.3)
    assert result.success is False
    with pytest.raises(AgentTestTimeoutError):
        result.raise_for_error()
    assert result.latency_ms < 2000


def test_openai_arun_timeout_cancels_the_run():
    result = asyncio.run(arun(_slow_agent(), "hi", timeout=0.3))
    with pytest.raises(AgentTestTimeoutError):
        result.raise_for_error()
    assert result.latency_ms < 2000


def test_openai_crashed_run_still_records_the_mocked_failure():
    pytest.importorskip("agents")
    from agents import Agent, function_tool

    @function_tool(failure_error_function=None)
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    agent = Agent(name="w", instructions="-", tools=[get_weather], model=_scripted(_tool_then_answer("get_weather", "c1")))
    with mock_tool_failure("get_weather", "fatal"):
        result = run(agent, "q")

    assert result.success is False
    step = result.trace.failed_tool_calls[0]
    assert step.error == "MockedToolError: fatal"
    assert step.metadata["call_id"] == "c1"


def test_openai_max_turns_error_keeps_partial_trace():
    pytest.importorskip("agents")
    from agents import Agent

    agent = Agent(name="w", instructions="-", tools=[_weather_tool()],
                  model=_scripted(_tool_then_answer("get_weather", "c1")))
    result = run(agent, "q", max_turns=1)

    assert result.success is False
    assert "MaxTurnsExceeded" in result.error or "turns" in result.error.lower()
    # The SDK attaches what ran before the error; it lands on the trace.
    assert len(result.trace.llm_calls) == 1
    assert result.tools_called == ["get_weather"]


# =============================================================================
# CrewAI — a full crew with a scripted LLM
# =============================================================================

@pytest.fixture
def crewai_offline(monkeypatch):
    for var, value in {"CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
                       "CREWAI_TRACING_ENABLED": "false"}.items():
        monkeypatch.setenv(var, value)
    return pytest.importorskip("crewai")


def _crew():
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool as crew_tool

    @crew_tool("web_search")
    def web_search(query: str) -> str:
        """Search the web."""
        return f"real results for {query}"

    class ScriptedLLM(BaseLLM):
        """ReAct-format replies: call web_search once, then answer from the observation."""

        def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
            history = messages if isinstance(messages, list) else []
            for message in reversed(history):
                content = str(message.get("content", ""))
                if message.get("role") == "assistant" and "Observation:" in content:
                    observation = content.rsplit("Observation:", 1)[1].strip()
                    return f"Thought: I have it\nFinal Answer: Based on: {observation}"
            return 'Thought: search first\nAction: web_search\nAction Input: {"query": "ai"}'

    researcher = Agent(role="researcher", goal="Find news", backstory="Curious.", tools=[web_search],
                       llm=ScriptedLLM(model="scripted"))
    task = Task(description="Find AI news", expected_output="A summary", agent=researcher)
    return Crew(agents=[researcher], tasks=[task])


@pytest.mark.filterwarnings("ignore")
def test_crewai_full_crew_is_traced(crewai_offline):
    result = run(_crew())

    assert result.success, result.traceback
    assert result.output == "Based on: real results for ai"
    assert result.tools_called == ["web_search"]
    assert [s.name for s in result.trace.steps if s.kind == "node"] == ["researcher"]


@pytest.mark.filterwarnings("ignore")
def test_crewai_full_crew_response_mock(crewai_offline):
    with mock_tool_response("web_search", "canned news") as handle:
        result = run(_crew())

    assert handle.calls == 1
    assert result.output == "Based on: canned news"
    assert result.trace.tool_calls[0].metadata["mocked"] == "response"


@pytest.mark.filterwarnings("ignore")
def test_crewai_full_crew_failure_mock_is_handled_by_crewai(crewai_offline):
    with mock_tool_failure("web_search", "search down") as handle:
        result = run(_crew())

    # CrewAI retries a failing tool itself before giving the agent the error.
    assert handle.calls >= 1
    assert result.success, result.traceback
    assert all(s.error == "MockedToolError: search down" for s in result.trace.failed_tool_calls)
    assert result.gracefully_degraded is True


@pytest.mark.filterwarnings("ignore")
def test_crewai_arun_uses_native_akickoff(crewai_offline, monkeypatch):
    crew = _crew()
    from crewai import Crew

    def no_sync(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("arun must not use Crew.kickoff")

    monkeypatch.setattr(Crew, "kickoff", no_sync)
    result = asyncio.run(arun(crew))

    assert result.success, result.traceback
    assert result.output == "Based on: real results for ai"
    assert result.tools_called == ["web_search"]


# =============================================================================
# Anthropic — the real client over a mocked HTTP transport
# =============================================================================

def _anthropic_reply(content: list[dict[str, Any]], stop_reason: str, tokens: tuple[int, int]) -> dict[str, Any]:
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]}}


def _http():
    """The HTTP package the installed Anthropic SDK is built on."""
    try:
        import httpx2 as http
    except ImportError:
        import httpx as http
    return http


def _anthropic_handler(request: Any) -> Any:
    body = json.loads(request.content)
    last = body["messages"][-1]
    if last["role"] == "user" and isinstance(last["content"], list) and last["content"][0].get("type") == "tool_result":
        result = last["content"][0]
        text = f"Error noted: {result['content']}" if result.get("is_error") else f"It is {result['content']}."
        reply = _anthropic_reply([{"type": "text", "text": text}], "end_turn", (40, 9))
    else:
        reply = _anthropic_reply([{"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}],
                                 "tool_use", (25, 12))
    return _http().Response(200, json=reply)


@agenttest.tool
def get_weather(city: str) -> str:
    return f"sunny in {city}"


def _tool_loop(client: Any) -> Any:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Weather in Paris?"}]
    while True:
        reply = client.messages.create(model="claude-sonnet-5-5", max_tokens=256, messages=messages,
                                       tools=[{"name": "get_weather", "input_schema": {"type": "object"}}])
        if reply.stop_reason != "tool_use":
            return reply
        use = next(b for b in reply.content if b.type == "tool_use")
        try:
            content, is_error = get_weather(**use.input), False
        except Exception as exc:
            content, is_error = str(exc), True
        messages += [{"role": "assistant", "content": [b.model_dump() for b in reply.content]},
                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id,
                                                   "content": content, "is_error": is_error}]}]


def _anthropic_client(async_: bool = False) -> Any:
    anthropic = pytest.importorskip("anthropic")
    http = _http()
    transport = http.MockTransport(_anthropic_handler)
    if async_:
        return anthropic.AsyncAnthropic(api_key="test", http_client=http.AsyncClient(transport=transport))
    return anthropic.Anthropic(api_key="test", http_client=http.Client(transport=transport))


def test_anthropic_real_client_tool_loop_is_traced():
    client = _anthropic_client()
    result = run(lambda: _tool_loop(client), adapter="anthropic")

    assert result.success, result.traceback
    assert result.output == "It is sunny in Paris."
    assert [s.tool_requests for s in result.trace.llm_calls] == [["get_weather"], []]
    assert result.tools_called == ["get_weather"]
    assert result.tokens_used == 25 + 12 + 40 + 9
    assert result.trace.llm_calls[0].model == "claude-sonnet-5-5"
    assert result.trace.llm_calls[0].metadata["stop_reason"] == "tool_use"


def test_anthropic_real_client_failure_mock():
    client = _anthropic_client()
    with mock_tool_failure("get_weather", "API down") as handle:
        result = run(lambda: _tool_loop(client), adapter="anthropic")

    assert handle.calls == 1
    assert result.output == "Error noted: API down"
    assert result.gracefully_degraded is True


def test_anthropic_real_async_client_is_traced():
    client = _anthropic_client(async_=True)

    async def loop() -> Any:
        messages = [{"role": "user", "content": "Weather in Paris?"}]
        return await client.messages.create(model="claude-sonnet-5-5", max_tokens=64, messages=messages)

    result = asyncio.run(arun(loop, adapter="anthropic"))
    assert result.success, result.traceback
    assert len(result.trace.llm_calls) == 1
    assert result.trace.llm_calls[0].tool_requests == ["get_weather"]
    assert result.tokens_used == 37


def test_anthropic_patch_is_removed_after_run():
    pytest.importorskip("anthropic")
    from anthropic.resources.messages import AsyncMessages, Messages

    before = (Messages.create, AsyncMessages.create)
    run(lambda: None, adapter="anthropic")
    assert (Messages.create, AsyncMessages.create) == before
