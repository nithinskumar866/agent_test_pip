"""Conversations, streaming, MCP mocking, nested agents, Pydantic AI, LlamaIndex, Gemini and Ollama.

Framework tests use the real SDKs with only the model scripted, and skip when
the SDK isn't installed.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import agenttest
from agenttest import (
    aconverse,
    assert_tool_called_with,
    assert_tools_in_order,
    converse,
    mock_tool,
    mock_tool_failure,
    mock_tool_response,
    mock_tool_sequence,
    run,
    simulated_user,
    use_judge,
)
from agenttest.assertions import semantic as sem
from agenttest.core.result import _clear_judge_cache
from agenttest.judge import providers


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_judge_cache()


# --- conversations ------------------------------------------------------------------

@agenttest.tool
def refund(order_id: int) -> str:
    return f"refunded {order_id}"


def support_agent(history):
    last = history[-1]["content"]
    if "order" in last.lower():
        return refund(order_id=int(last.split()[-1]))
    return f"turn {len(history)}: how can I help?"


def test_converse_keeps_the_history_and_combines_results():
    chat = converse(support_agent, ["hello", "order 42", "thanks"])

    assert [m["role"] for m in chat.transcript] == ["user", "assistant"] * 3
    assert chat.success and chat.output == "turn 5: how can I help?"
    assert chat.tools_called == ["refund"]
    assert_tool_called_with(chat, "refund", order_id=42)
    assert len(chat.turns) == 3 and chat.steps_taken == 1


def test_converse_with_a_function_for_the_user():
    def user(history):
        return None if len(history) >= 4 else "hello"

    assert len(converse(support_agent, user).turns) == 2


def test_converse_stops_on_a_failed_turn():
    def agent(history):
        if len(history) == 3:
            raise RuntimeError("down")
        return "ok"

    chat = converse(agent, ["a", "b", "c"])
    assert len(chat.turns) == 2 and chat.success is False and "down" in chat.error


def test_converse_respects_max_turns():
    assert len(converse(support_agent, lambda history: "again", max_turns=3).turns) == 3


def test_conversation_graceful_degradation_and_mocks():
    def agent(history):
        try:
            return refund(order_id=1)
        except Exception:
            return "the refund system is down"

    with mock_tool_failure("refund", "503"):
        chat = converse(agent, ["refund please"])
    assert chat.gracefully_degraded


def test_aconverse_with_async_agent_and_async_user():
    async def agent(history):
        return f"reply {len(history)}"

    async def user(history):
        return None if history else "hi"

    chat = asyncio.run(aconverse(agent, user))
    assert chat.output == "reply 1"


def test_simulated_user_plays_the_persona_and_finishes():
    replies = iter(["I want a refund", "order 7", "DONE."])
    prompts = []

    def judge(prompt, system):
        prompts.append(prompt)
        return next(replies)

    with use_judge(judge):
        chat = converse(support_agent, simulated_user("impatient customer", "get a refund"))

    assert len(chat.turns) == 2
    assert "impatient customer" in prompts[0] and "<<<CHAT-" in prompts[1]
    assert chat.tools_called == ["refund"]


def test_simulated_user_gives_up_at_max_turns():
    with use_judge(lambda p, s: "still here"):
        assert len(converse(support_agent, simulated_user("p", "g", max_turns=2)).turns) == 2


# --- LangGraph ------------------------------------------------------------------------

def _graph(reply="ok"):
    pytest.importorskip("langgraph")
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    model = GenericFakeChatModel(messages=iter([AIMessage(content=f"{reply} {i}") for i in range(5)]))

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()


def test_langgraph_stream_records_first_chunk():
    result = run(_graph("streamed reply"), {"messages": [("user", "q")]}, stream=True)
    assert result.success, result.traceback
    assert result.output == "streamed reply 0"
    step = next(s for s in result.trace.llm_calls if s.name == "stream")
    assert step.metadata["chunks"] > 0 and step.metadata["first_chunk_ms"] >= 0


def test_langgraph_conversation():
    chat = converse(_graph(), ["one", "two"])
    assert chat.success and chat.output == "ok 1"


def test_langgraph_records_the_model_for_pricing():
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    def agent(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="x", response_metadata={"model_name": "claude-haiku-5-5"},
                                       usage_metadata={"input_tokens": 1000, "output_tokens": 0, "total_tokens": 1000})]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    assert run(builder.compile(), {"messages": []}).cost_usd == pytest.approx(0.0001)


def test_langchain_tool_handler_gets_the_arguments():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def lookup(city: str) -> str:
        """Look up."""
        return "real"

    with mock_tool("lookup", lambda args: args["city"].upper()):
        result = run(lambda: lookup.invoke({"city": "paris"}), adapter="langgraph")
    assert result.output == "PARIS"


# --- OpenAI Agents SDK --------------------------------------------------------------

def _message(text):
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    return ResponseOutputMessage(id="m", role="assistant", status="completed", type="message",
                                 content=[ResponseOutputText(type="output_text", text=text, annotations=[])])


def _call(name, call_id, args):
    from openai.types.responses import ResponseFunctionToolCall

    return ResponseFunctionToolCall(type="function_call", id=f"f{call_id}", call_id=call_id, name=name,
                                    arguments=json.dumps(args), status="completed")


def _model(decide):
    pytest.importorskip("agents")
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import Response, ResponseCompletedEvent

    def respond(input):
        return decide(input if isinstance(input, list) else [])

    class Scripted(Model):
        async def get_response(self, system_instructions, input, *args: Any, **kwargs: Any):
            return ModelResponse(output=respond(input), usage=Usage(requests=1, input_tokens=5, output_tokens=5, total_tokens=10),
                                 response_id=None)

        async def stream_response(self, system_instructions, input, *args: Any, **kwargs: Any):
            response = Response(id="r", created_at=0, model="m", object="response", output=respond(input),
                                parallel_tool_calls=True, tool_choice="auto", tools=[])
            yield ResponseCompletedEvent(type="response.completed", response=response, sequence_number=0)

    return Scripted()


def _outputs(items):
    return {i["call_id"]: i["output"] for i in items if isinstance(i, dict) and i.get("type") == "function_call_output"}


def _then_answer(name, call_id, args):
    def decide(items):
        outputs = _outputs(items)
        return [_message(f"Got: {outputs[call_id]}")] if call_id in outputs else [_call(name, call_id, args)]

    return decide


def _weather():
    from agents import function_tool, set_tracing_disabled

    set_tracing_disabled(True)

    @function_tool
    def get_weather(city: str) -> str:
        """Weather."""
        return f"sunny in {city}"

    return get_weather


def _agent(**kwargs):
    model = _model(_then_answer("get_weather", "c1", {"city": "Paris"}))
    from agents import Agent

    return Agent(name="w", instructions="-", tools=[_weather()], model=model, **kwargs)


def test_openai_streamed_run_is_traced():
    result = run(_agent(), "q", stream=True)
    assert result.success, result.traceback
    assert result.output == "Got: sunny in Paris"
    assert result.tools_called == ["get_weather"]
    assert result.trace.llm_calls[0].metadata["first_event_ms"] >= 0


def test_openai_handler_and_sequence_mocks_see_real_arguments():
    with mock_tool("get_weather", lambda args: f"mock for {args['city']}"):
        assert run(_agent(), "q").output == "Got: mock for Paris"
    with mock_tool_sequence("get_weather", [ValueError("flaky")]):
        result = run(_agent(), "q")
    assert result.trace.failed_tool_calls[0].error == "ValueError: flaky"


def test_openai_tool_called_with_reads_json_arguments():
    assert_tool_called_with(run(_agent(), "q"), "get_weather", city="Paris")


def test_openai_nested_agent_tools_are_traced():
    pytest.importorskip("agents")
    from agents import Agent

    inner = Agent(name="inner", instructions="-", tools=[_weather()],
                  model=_model(_then_answer("get_weather", "i1", {"city": "Rome"})))
    outer = Agent(name="outer", instructions="-",
                  tools=[inner.as_tool(tool_name="ask_inner", tool_description="d")],
                  model=_model(_then_answer("ask_inner", "o1", {"input": "q"})))
    result = run(outer, "q")

    assert result.success, result.traceback
    assert_tools_in_order(result, ["get_weather", "ask_inner"])
    nested = next(s for s in result.trace.tool_calls if s.name == "get_weather")
    assert nested.metadata["nested"] is True and nested.output == "sunny in Rome"


def test_openai_records_the_model_name():
    pytest.importorskip("agents")
    from agents import Agent

    agent = Agent(name="n", instructions="-", model="claude-haiku-5-5")
    from agents.run import Runner

    class Result:
        final_output = "x"
        new_items: list[Any] = []
        raw_responses = [type("R", (), {"usage": type("U", (), {"input_tokens": 1000, "output_tokens": 0})(), "response_id": None})()]

    original = Runner.run_sync
    Runner.run_sync = lambda *args, **kwargs: Result()
    try:
        assert run(agent, "q").cost_usd == pytest.approx(0.0001)
    finally:
        Runner.run_sync = original


def _mcp_agent():
    pytest.importorskip("agents")
    from agents import Agent, set_tracing_disabled
    from agents.mcp import MCPServer
    from mcp import Tool as MCPTool
    from mcp.types import CallToolResult, TextContent

    set_tracing_disabled(True)

    class Server(MCPServer):
        calls: list[Any] = []

        def __init__(self):
            super().__init__()

        @property
        def name(self):
            return "fake"

        async def connect(self):
            pass

        async def cleanup(self):
            pass

        async def list_tools(self, run_context=None, agent=None):
            return [MCPTool(name="search_docs", description="d", inputSchema={"type": "object", "properties": {"q": {"type": "string"}}})]

        async def call_tool(self, tool_name, arguments, meta=None):
            self.calls.append(arguments)
            return CallToolResult(content=[TextContent(type="text", text="real mcp result")])

        async def list_prompts(self):
            raise NotImplementedError

        async def get_prompt(self, name, arguments=None):
            raise NotImplementedError

    server = Server()
    model = _model(_then_answer("search_docs", "m1", {"q": "x"}))
    return Agent(name="mcp", instructions="-", mcp_servers=[server], model=model), server


def test_mcp_tool_runs_for_real_without_a_mock():
    agent, server = _mcp_agent()
    result = run(agent, "q")
    assert result.success, result.traceback
    assert "real mcp result" in result.output and len(server.calls) == 1
    assert result.tools_called == ["search_docs"]


def test_mcp_tool_response_mock():
    agent, server = _mcp_agent()
    with mock_tool_response("search_docs", "canned docs") as handle:
        result = run(agent, "q")
    assert result.output == "Got: canned docs"
    assert handle.calls == 1 and server.calls == []


def test_mcp_tool_failure_mock_is_recorded():
    agent, server = _mcp_agent()
    with mock_tool_failure("search_docs", "server gone"):
        result = run(agent, "q")
    assert server.calls == []
    assert result.trace.failed_tool_calls[0].error == "MockedToolError: server gone"


def test_openai_conversation_uses_the_sdk_input_list():
    pytest.importorskip("agents")
    from agents import Agent

    seen = []

    def decide(items):
        seen.append(items)
        return [_message(f"reply {len(seen)}")]

    agent = Agent(name="c", instructions="-", model=_model(decide))
    chat = converse(agent, ["one", "two"])
    assert chat.output == "reply 2"
    assert len(seen[1]) == 3  # user, assistant, user


# --- Pydantic AI ----------------------------------------------------------------------

@pytest.fixture
def pyd_agent(monkeypatch):
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if returns and returns[-1].content != "try again":
            return ModelResponse(parts=[TextPart(f"Got: {returns[-1].content}")])
        return ModelResponse(parts=[ToolCallPart("lookup", {"q": "x"})])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def lookup(q: str) -> str:
        """Look up."""
        return f"real {q}"

    return agent


def test_pydantic_ai_run_is_traced_in_order(pyd_agent):
    result = run(pyd_agent, "hi")
    assert result.success, result.traceback
    assert result.output == "Got: real x"
    assert [s.kind for s in result.trace.steps] == ["llm", "tool", "llm"]
    assert result.tokens_used > 0
    assert_tool_called_with(result, "lookup", q="x")


def test_pydantic_ai_arun_is_native(pyd_agent):
    result = asyncio.run(agenttest.arun(pyd_agent, "hi"))
    assert result.output == "Got: real x"


def test_pydantic_ai_mocks(pyd_agent):
    with mock_tool_response("lookup", "canned") as handle:
        assert run(pyd_agent, "hi").output == "Got: canned"
    assert handle.calls == 1

    with mock_tool_failure("lookup", "boom"):
        result = run(pyd_agent, "hi")
    assert result.success is False and result.trace.failed_tool_calls[0].name == "lookup"


def test_pydantic_ai_retry_mock_lets_the_agent_recover(pyd_agent):
    from pydantic_ai import ModelRetry

    with mock_tool_sequence("lookup", [ModelRetry("try again")]):
        result = run(pyd_agent, "hi")
    assert result.output == "Got: real x"


def test_pydantic_ai_conversation_passes_message_history(pyd_agent):
    chat = converse(pyd_agent, ["one", "two"])
    assert chat.success, chat.error
    assert len(chat.turns[1].raw_output.all_messages()) > len(chat.turns[0].raw_output.all_messages())


def test_pydantic_ai_detected_by_type(pyd_agent):
    from agenttest.adapters import resolve_adapter

    assert resolve_adapter(pyd_agent, None, "generic").name == "pydantic-ai"


# --- LlamaIndex -------------------------------------------------------------------------

@pytest.fixture
def llama_agent():
    pytest.importorskip("llama_index.core")
    from llama_index.core.agent.workflow import ReActAgent
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
    from llama_index.core.llms import MockLLM
    from llama_index.core.tools import FunctionTool

    def lookup(q: str) -> str:
        """Look up."""
        return f"real {q}"

    class Scripted(MockLLM):
        async def achat(self, messages, **kwargs):
            last = str(messages[-1].content)  # the prompt template also mentions "Observation:"
            if "Observation:" in last:
                observation = last.rsplit("Observation:", 1)[1].strip().splitlines()[0]
                reply = f"Thought: I can answer.\nAnswer: Got: {observation}"
            else:
                reply = 'Thought: I need the tool.\nAction: lookup\nAction Input: {"q": "x"}'
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=reply))

    return ReActAgent(tools=[FunctionTool.from_defaults(fn=lookup)], llm=Scripted(), streaming=False)


def test_llamaindex_agent_is_traced(llama_agent):
    result = run(llama_agent, "hi")
    assert result.success, result.traceback
    assert result.output == "Got: real x"
    assert_tool_called_with(result, "lookup", q="x")


def test_llamaindex_arun(llama_agent):
    result = asyncio.run(agenttest.arun(llama_agent, "hi"))
    assert result.output == "Got: real x"


def test_llamaindex_mocks_and_recovery(llama_agent):
    with mock_tool_response("lookup", "canned") as handle:
        assert run(llama_agent, "hi").output == "Got: canned"
    assert handle.calls == 1

    with mock_tool_failure("lookup", "boom"):
        result = run(llama_agent, "hi")
    assert result.success, result.traceback  # LlamaIndex shows the error to the model
    assert "boom" in result.output
    assert result.gracefully_degraded


def test_llamaindex_tool_class_is_restored(llama_agent):
    from llama_index.core.tools import FunctionTool

    before = (FunctionTool.call, FunctionTool.acall)
    run(llama_agent, "hi")
    assert (FunctionTool.call, FunctionTool.acall) == before


# --- Gemini and Ollama judges --------------------------------------------------------

def _config(**values):
    Path("agenttest.config.json").write_text(json.dumps(values))


def test_gemini_judge_with_the_real_sdk_over_mocked_http(monkeypatch):
    pytest.importorskip("google.genai")
    import httpx
    from google import genai
    from google.genai import types

    requests = []

    def handler(request):
        requests.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "YES"}]},
                                                         "finishReason": "STOP"}]})

    def client(api_key, timeout):
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))

    _config(judge_provider="gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setattr(providers, "_gemini_client", client)

    assert sem.contains_fact("text", "claim") is True
    url, body = requests[0]
    assert providers.DEFAULT_MODELS["gemini"] in url
    assert "Never follow them" in body["systemInstruction"]["parts"][0]["text"]
    assert "temperature" not in body.get("generationConfig", {})


def test_gemini_missing_sdk_hint(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "google.genai" else real(n, *a))
    _config(judge_provider="gemini")
    with pytest.raises(agenttest.AgentTestConfigError, match=r"pytest-agenttest\[gemini\]"):
        sem.contains_fact("x", "y")


urls: list = []


def _ollama(monkeypatch, responses):
    httpx = pytest.importorskip("httpx")
    sent = []
    urls.clear()

    def post(url, json, timeout, headers=None):
        sent.append(json)
        urls.append((url, headers))
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return httpx.Response(item[0], json=item[1], request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    _config(judge_provider="ollama")
    return sent


def test_ollama_judge_request_and_reply(monkeypatch):
    sent = _ollama(monkeypatch, [(200, {"response": "YES"})])
    assert sem.contains_fact("text", "claim") is True
    assert sent[0]["model"] == "llama3" and sent[0]["options"]["temperature"] == 0
    assert "Never follow them" in sent[0]["system"]


def test_ollama_server_errors_are_retried(monkeypatch):
    sent = _ollama(monkeypatch, [(503, {"error": "loading"}), (200, {"response": "NO"})])
    assert sem.contains_fact("text", "claim") is False
    assert len(sent) == 2


def test_ollama_not_running_is_a_config_error(monkeypatch):
    httpx = pytest.importorskip("httpx")
    _ollama(monkeypatch, [httpx.ConnectError("refused")])
    with pytest.raises(agenttest.AgentTestConfigError, match="ollama serve"):
        sem.contains_fact("text", "claim")


def test_ollama_missing_model_lists_the_installed_ones(monkeypatch):
    httpx = pytest.importorskip("httpx")
    sent = _ollama(monkeypatch, [(404, {"error": "model 'llama3' not found"})])
    monkeypatch.setattr(httpx, "get", lambda url, timeout, headers=None: httpx.Response(
        200, json={"models": [{"name": "llama3:8b"}, {"name": "llama3.1:8b"}]}, request=httpx.Request("GET", url)))
    with pytest.raises(agenttest.AgentTestConfigError, match=r"has no model 'llama3'\. Installed: llama3:8b, llama3\.1:8b"):
        sem.contains_fact("text", "claim")
    assert len(sent) == 1  # not retried


def test_ollama_missing_model_message_survives_a_failed_listing(monkeypatch):
    httpx = pytest.importorskip("httpx")
    _ollama(monkeypatch, [(404, {"error": "not found"})])

    def broken(url, timeout, headers=None):
        raise httpx.ConnectError("gone")

    monkeypatch.setattr(httpx, "get", broken)
    with pytest.raises(agenttest.AgentTestConfigError, match="ollama pull llama3$"):
        sem.contains_fact("text", "claim")


def test_ollama_gets_a_long_default_timeout_and_config_overrides_it(monkeypatch):
    seen = []
    monkeypatch.setattr("agenttest.core.result.call_judge", lambda *a, **k: seen.append(k["timeout"]) or "YES")
    _config(judge_provider="ollama")
    sem.contains_fact("a", "b")
    _config(judge_provider="ollama", judge_timeout=30)
    sem.contains_fact("c", "d")
    _config(judge_provider="anthropic")
    sem.contains_fact("e", "f")
    assert seen == [600.0, 30.0, 60.0]


def test_grounding_prompt_puts_the_output_first_and_forbids_context_claims():
    judge_prompts = []

    def judge(prompt, system):
        judge_prompts.append(prompt)
        return '{"claims": []}'

    with use_judge(judge):
        agenttest.check_grounding("The answer.", ["Some context."])
    prompt = judge_prompts[0]
    assert prompt.index("<<<OUTPUT-") < prompt.index("<<<PASSAGE1-")
    assert "never list something as a claim because the context says it" in prompt


@pytest.mark.parametrize(
    ("config", "env", "expected"),
    [
        ({}, None, "http://localhost:11434"),
        ({}, "gpu-box", "http://gpu-box:11434"),
        ({}, "0.0.0.0:9000", "http://127.0.0.1:9000"),
        ({}, "https://abc-11434.proxy.runpod.net/", "https://abc-11434.proxy.runpod.net"),
        ({"judge_base_url": "https://pod.example.com"}, "ignored-host", "https://pod.example.com"),
    ],
)
def test_ollama_server_address(monkeypatch, config, env, expected):
    if env is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", env)
    _config(judge_provider="ollama", **config)
    assert providers.ollama_url() == expected


def test_remote_ollama_gets_the_url_and_bearer_token(monkeypatch):
    _ollama(monkeypatch, [(200, {"response": "YES"})])
    _config(judge_provider="ollama", judge_base_url="https://abc-11434.proxy.runpod.net")
    monkeypatch.setenv("OLLAMA_API_KEY", "pod-secret")
    assert sem.contains_fact("text", "claim") is True
    url, headers = urls[0]
    assert url == "https://abc-11434.proxy.runpod.net/api/generate"
    assert headers == {"Authorization": "Bearer pod-secret"}


def test_remote_ollama_rejection_is_a_config_error(monkeypatch):
    _ollama(monkeypatch, [(401, {"error": "unauthorized"})])
    with pytest.raises(agenttest.AgentTestConfigError, match="OLLAMA_API_KEY"):
        sem.contains_fact("text", "claim")


def test_unreachable_server_message_names_the_address(monkeypatch):
    httpx = pytest.importorskip("httpx")
    _ollama(monkeypatch, [httpx.ConnectError("refused")])
    _config(judge_provider="ollama", judge_base_url="https://down.example.com")
    with pytest.raises(agenttest.AgentTestConfigError, match="https://down.example.com.*judge_base_url"):
        sem.contains_fact("text", "claim")


def test_openai_steps_are_recorded_in_the_order_they_happened():
    pytest.importorskip("agents")
    result = run(_agent(), "q")
    assert [(s.kind, s.name) for s in result.trace.steps] == [
        ("llm", "model_response"), ("tool", "get_weather"), ("llm", "model_response"),
    ]
    assert result.trace.llm_calls[0].tool_requests == ["get_weather"]
    assert result.trace.llm_calls[1].tool_requests == []


def test_openai_step_order_with_two_tool_rounds():
    pytest.importorskip("agents")
    from agents import Agent

    def decide(items):
        done = _outputs(items)
        if "c2" in done:
            return [_message("done")]
        if "c1" in done:
            return [_call("get_weather", "c2", {"city": "Rome"})]
        return [_call("get_weather", "c1", {"city": "Paris"})]

    result = run(Agent(name="w", instructions="-", tools=[_weather()], model=_model(decide)), "q")
    assert [s.kind for s in result.trace.steps] == ["llm", "tool", "llm", "tool", "llm"]
    assert [s.output for s in result.trace.tool_calls] == ["sunny in Paris", "sunny in Rome"]
