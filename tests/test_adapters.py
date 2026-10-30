"""Tests for adapter selection and the framework adapters.

Framework objects are faked with lookalike classes (matching module names and
attributes), so these tests run without LangGraph, CrewAI or the OpenAI
Agents SDK installed.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from agenttest import run
from agenttest.adapters import get_adapter, resolve_adapter
from agenttest.adapters.anthropic_sdk import AnthropicAdapter, _record_message
from agenttest.adapters.base import GenericAdapter, text_of
from agenttest.adapters.crewai import CrewAIAdapter
from agenttest.adapters.langgraph import LangGraphAdapter
from agenttest.adapters.openai_sdk import OpenAIAdapter
from agenttest.core.trace import tracing


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# --- fakes -------------------------------------------------------------------

def ai(content, tool_calls=(), usage=None):
    return SimpleNamespace(type="ai", content=content, tool_calls=list(tool_calls), usage_metadata=usage or {})


def tool_msg(name, content, status="success"):
    return SimpleNamespace(type="tool", name=name, content=content, status=status, tool_call_id=f"id-{name}")


class FakeCompiledGraph:
    def __init__(self, updates):
        self._updates = updates
        self.calls = []

    def stream(self, state, stream_mode="values", **kwargs):
        self.calls.append((state, stream_mode))
        yield from self._updates

    def invoke(self, state):  # pragma: no cover - the adapter must stream instead
        raise AssertionError("adapter should stream, not invoke")


FakeCompiledGraph.__module__ = "langgraph.graph.state"


class FakeOpenAIAgent:
    def __init__(self, name="assistant"):
        self.name = name
        self.instructions = "be helpful"


FakeOpenAIAgent.__module__ = "agents.agent"


class FakeCrew:
    def __init__(self, result):
        self._result = result
        self.received = None

    def kickoff(self, inputs=None):
        self.received = inputs
        return self._result


FakeCrew.__module__ = "crewai.crew"


# --- selection ---------------------------------------------------------------

def test_resolve_detects_framework_by_type():
    assert isinstance(resolve_adapter(FakeCompiledGraph([]), None, "generic"), LangGraphAdapter)
    assert isinstance(resolve_adapter(FakeOpenAIAgent(), None, "generic"), OpenAIAdapter)
    assert isinstance(resolve_adapter(FakeCrew(None), None, "generic"), CrewAIAdapter)


def test_resolve_falls_back_to_configured_framework():
    assert isinstance(resolve_adapter(lambda: "x", None, "anthropic"), AnthropicAdapter)
    assert isinstance(resolve_adapter(lambda: "x", None, "generic"), GenericAdapter)


def test_explicit_adapter_wins_over_detection():
    adapter = GenericAdapter()
    assert resolve_adapter(FakeCompiledGraph([]), adapter, "langgraph") is adapter
    assert isinstance(resolve_adapter(FakeCompiledGraph([]), "generic", "langgraph"), GenericAdapter)


def test_get_adapter_for_every_framework():
    for name in ("langgraph", "crewai", "openai", "anthropic", "generic"):
        assert get_adapter(name).name == name


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        ("plain", "plain"),
        ({"output": "from dict"}, "from dict"),
        ({"messages": [ai("first"), ai("last")]}, "last"),
        (SimpleNamespace(final_output="final"), "final"),
        (42, "42"),
    ],
)
def test_text_of(value, expected):
    assert text_of(value) == expected


# --- LangGraph ---------------------------------------------------------------

def test_langgraph_records_nodes_tools_and_tokens():
    graph = FakeCompiledGraph([
        {"agent": {"messages": [ai("", tool_calls=[{"name": "search", "args": {}}],
                                   usage={"input_tokens": 10, "output_tokens": 5})]}},
        {"tools": {"messages": [tool_msg("search", "results...")]}},
        {"agent": {"messages": [ai("Paris is the capital.", usage={"input_tokens": 20, "output_tokens": 7})]}},
    ])

    result = run(graph, {"messages": [("user", "capital of France?")]})

    assert result.success is True
    assert result.output == "Paris is the capital."
    assert graph.calls[0][1] == "updates"
    assert [s.name for s in result.trace.steps if s.kind == "node"] == ["agent", "tools", "agent"]
    assert result.tools_called == ["search"]
    assert result.trace.steps[0].tool_requests == ["search"]
    assert result.tokens_used == 42


def test_langgraph_failed_tool_enables_graceful_degradation():
    graph = FakeCompiledGraph([
        {"tools": {"messages": [tool_msg("web_search", "Error: 503", status="error")]}},
        {"agent": {"messages": [ai("I couldn't search, but here is what I know.")]}},
    ])

    result = run(graph, {"messages": []})

    assert result.trace.failed_tool_calls[0].name == "web_search"
    assert result.gracefully_degraded is True


def test_langgraph_does_not_mutate_graph():
    graph = FakeCompiledGraph([{"agent": {"messages": [ai("hi")]}}])
    before = dict(vars(graph))
    run(graph, {})
    run(graph, {})
    assert set(vars(graph)) == set(before)
    assert len(graph.calls) == 2


# --- Anthropic ---------------------------------------------------------------

def _anthropic_response(text, tools=(), input_tokens=12, output_tokens=8):
    blocks = [SimpleNamespace(type="text", text=text)] + [
        SimpleNamespace(type="tool_use", name=t) for t in tools
    ]
    return SimpleNamespace(
        content=blocks,
        model="claude-haiku-5-5",
        stop_reason="tool_use" if tools else "end_turn",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_anthropic_records_llm_step():
    with tracing() as trace:
        _record_message({"model": "m", "messages": []}, _anthropic_response("hi", tools=["calc"]), 3.0)
    step = trace.steps[0]
    assert step.kind == "llm"
    assert step.tokens == 20
    assert step.tool_requests == ["calc"]
    assert step.output == "hi"
    assert step.metadata["stop_reason"] == "tool_use"


def test_anthropic_patch_is_installed_only_during_run():
    anthropic_messages = pytest.importorskip("anthropic.resources.messages")
    original = anthropic_messages.Messages.create
    seen = {}

    def agent():
        seen["patched"] = anthropic_messages.Messages.create is not original
        return _anthropic_response("done")

    result = run(agent, adapter="anthropic")
    assert seen["patched"] is True
    assert anthropic_messages.Messages.create is original
    assert result.output == "done"


def test_anthropic_patched_create_records_calls():
    anthropic_messages = pytest.importorskip("anthropic.resources.messages")
    responses = iter([_anthropic_response("", tools=["lookup"]), _anthropic_response("Answer")])

    def fake_create(self, **kwargs):
        return next(responses)

    original = anthropic_messages.Messages.create
    anthropic_messages.Messages.create = fake_create
    try:
        def agent():
            client = object.__new__(anthropic_messages.Messages)
            client.create(model="m", messages=[])
            return client.create(model="m", messages=[])

        result = run(agent, adapter="anthropic")
    finally:
        anthropic_messages.Messages.create = original

    assert [s.kind for s in result.trace.steps] == ["llm", "llm"]
    assert result.trace.steps[0].tool_requests == ["lookup"]
    assert result.tokens_used == 40
    assert result.output == "Answer"


# --- OpenAI Agents SDK -------------------------------------------------------

def test_openai_agent_run_via_runner(monkeypatch):
    raw_call = SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}', call_id="c1")
    run_result = SimpleNamespace(
        final_output="It is sunny in Paris.",
        new_items=[
            SimpleNamespace(type="tool_call_item", raw_item=raw_call),
            SimpleNamespace(type="tool_call_output_item", raw_item={"call_id": "c1"}, output="sunny"),
            SimpleNamespace(type="message_output_item", raw_item=None),
        ],
        raw_responses=[
            SimpleNamespace(usage=SimpleNamespace(input_tokens=30, output_tokens=10), response_id="r1"),
            SimpleNamespace(usage=SimpleNamespace(input_tokens=50, output_tokens=12), response_id="r2"),
        ],
    )
    calls = []

    class Runner:
        @staticmethod
        def run_sync(agent, prompt, **kwargs):
            calls.append((agent, prompt))
            return run_result

    monkeypatch.setitem(sys.modules, "agents", types.SimpleNamespace(Runner=Runner))
    agent = FakeOpenAIAgent()

    result = run(agent, "Weather in Paris?")

    assert calls == [(agent, "Weather in Paris?")]
    assert result.output == "It is sunny in Paris."
    assert result.tools_called == ["get_weather"]
    assert result.trace.tool_calls[0].output == "sunny"
    assert result.tokens_used == 102


def test_openai_adapter_calls_plain_callables():
    assert run(lambda q: q[::-1], "abc", adapter="openai").output == "cba"


# --- CrewAI ------------------------------------------------------------------

def test_crewai_kickoff_and_task_outputs():
    crew_output = SimpleNamespace(
        raw="Final report",
        tasks_output=[
            SimpleNamespace(agent="Researcher", description="research", raw="notes"),
            SimpleNamespace(agent="Writer", description="write", raw="Final report"),
        ],
        token_usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40),
    )
    crew = FakeCrew(crew_output)

    result = run(crew, {"topic": "AI"})

    assert crew.received == {"topic": "AI"}
    assert result.output == "Final report"
    assert [s.name for s in result.trace.steps] == ["Researcher", "Writer"]
    assert result.tokens_used == 140
    assert result.steps_taken == 2
