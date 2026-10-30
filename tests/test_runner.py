"""Tests for the execution engine: agenttest.run / agenttest.arun."""

import asyncio
import json
import threading
import time
from contextlib import contextmanager

import pytest

import agenttest
from agenttest import AgentResult, AgentTestError, arun, run, tool
from agenttest.adapters.base import BaseAdapter, GenericAdapter
from agenttest.core.runner import run_agent_test


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test in an empty directory so no stray config is picked up."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@tool
def lookup(city: str) -> str:
    return f"{city} weather: sunny"


@tool(name="flaky_search")
def broken_search(query: str) -> str:
    raise ConnectionError("503 Service Unavailable")


def test_runs_plain_function():
    result = run(lambda q: f"Response to {q}", "hello")
    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.output == "Response to hello"
    assert result.error is None
    assert result.latency_ms >= 0


def test_runs_object_with_run_method():
    class Agent:
        def run(self, prompt):
            return {"output": prompt.upper()}

    assert run(Agent(), "hi").output == "HI"


def test_agent_exception_is_captured_with_traceback():
    def failing_agent():
        raise ValueError("Agent internal failure")

    result = run(failing_agent)
    assert result.success is False
    assert result.output == ""
    assert result.error == "ValueError: Agent internal failure"
    assert "failing_agent" in result.traceback
    with pytest.raises(ValueError, match="Agent internal failure"):
        result.raise_for_error()


def test_successful_result_raise_for_error_is_noop():
    run(lambda: "ok").raise_for_error()


def test_tool_calls_are_traced():
    def agent(city):
        return f"Report: {lookup(city)}"

    result = run(agent, "Paris")
    assert result.tools_called == ["lookup"]
    assert result.steps_taken == 1
    step = result.trace.tool_calls[0]
    assert step.input == {"city": "Paris"}
    assert step.output == "Paris weather: sunny"
    assert step.error is None


def test_graceful_degradation_detected():
    def agent(query):
        try:
            return broken_search(query)
        except ConnectionError:
            return "Search is unavailable right now; please try again later."

    result = run(agent, "news")
    assert result.success is True
    assert result.tools_called == ["flaky_search"]
    assert result.trace.failed_tool_calls[0].error == "ConnectionError: 503 Service Unavailable"
    assert result.gracefully_degraded is True


def test_crash_after_tool_failure_is_not_graceful():
    result = run(lambda q: broken_search(q), "news")
    assert result.success is False
    assert result.gracefully_degraded is False
    assert result.tools_called == ["flaky_search"]


def test_no_tool_failure_is_not_graceful():
    assert run(lambda: lookup("Rome")).gracefully_degraded is False


def test_tool_outside_run_is_transparent():
    assert lookup("Oslo") == "Oslo weather: sunny"
    assert agenttest.current_trace() is None


def test_each_run_gets_a_fresh_trace():
    first = run(lambda: lookup("A"))
    second = run(lambda: lookup("B"))
    assert first.tools_called == ["lookup"]
    assert second.tools_called == ["lookup"]
    assert first.trace is not second.trace


def test_timeout_reports_failure():
    result = run(lambda: time.sleep(1), timeout=0.1)
    assert result.success is False
    assert result.error.startswith("AgentTestTimeoutError")
    assert result.latency_ms < 1500


def test_timeout_run_still_traces_tools():
    def agent():
        return lookup("Lima")

    result = run(agent, timeout=5)
    assert result.success is True
    assert result.tools_called == ["lookup"]


def test_config_timeout_is_used(isolated_cwd):
    (isolated_cwd / "agenttest.config.json").write_text(json.dumps({"timeout": 0.1}), encoding="utf-8")
    result = run(lambda: time.sleep(1))
    assert result.success is False
    assert "timeout" in result.error


def test_explicit_timeout_overrides_config(isolated_cwd):
    (isolated_cwd / "agenttest.config.json").write_text(json.dumps({"timeout": 0.05}), encoding="utf-8")
    result = run(lambda: time.sleep(0.2) or "done", timeout=5)
    assert result.output == "done"


def test_async_agent_under_sync_run():
    async def agent(q):
        await asyncio.sleep(0)
        return f"async {q}"

    assert run(agent, "x").output == "async x"


def test_arun_with_async_agent_and_async_tool():
    @tool
    async def fetch(n: int) -> int:
        return n * 2

    async def agent(n):
        return f"got {await fetch(n)}"

    result = asyncio.run(arun(agent, 21))
    assert result.output == "got 42"
    assert result.tools_called == ["fetch"]


def test_arun_runs_sync_agent_off_the_loop():
    loop_thread = {}

    def agent():
        loop_thread["agent"] = threading.current_thread().name
        return lookup("Cairo")

    async def main():
        loop_thread["loop"] = threading.current_thread().name
        return await arun(agent)

    result = asyncio.run(main())
    assert result.tools_called == ["lookup"]
    assert loop_thread["agent"] != loop_thread["loop"]


def test_arun_timeout():
    async def agent():
        await asyncio.sleep(1)

    result = asyncio.run(arun(agent, timeout=0.1))
    assert result.success is False
    assert "AgentTestTimeoutError" in result.error


def test_concurrent_arun_traces_are_isolated():
    @tool
    async def step(label: str) -> str:
        await asyncio.sleep(0.01)
        return label

    async def agent(label):
        await step(label)
        await step(label)
        return label

    async def main():
        return await asyncio.gather(arun(agent, "a"), arun(agent, "b"))

    first, second = asyncio.run(main())
    assert [s.output for s in first.trace.tool_calls] == ["a", "a"]
    assert [s.output for s in second.trace.tool_calls] == ["b", "b"]


def test_sync_run_inside_event_loop_is_rejected():
    async def main():
        return run(lambda: "x")

    with pytest.raises(AgentTestError, match="arun"):
        asyncio.run(main())


def test_explicit_adapter_instance_is_used():
    events = []

    class RecordingAdapter(BaseAdapter):
        @contextmanager
        def instrument(self):
            events.append("install")
            yield
            events.append("remove")

        def extract_output(self, raw):
            return f"<{raw}>"

    result = run(lambda: "x", adapter=RecordingAdapter())
    assert result.output == "<x>"
    assert events == ["install", "remove"]


def test_adapter_by_name():
    assert run(lambda: "x", adapter="generic").output == "x"


def test_unknown_adapter_name_raises():
    with pytest.raises(agenttest.AgentTestConfigError):
        run(lambda: "x", adapter="autogen")


def test_backwards_compatible_entry_points():
    assert run_agent_test(lambda: "x").output == "x"
    assert GenericAdapter().run_with_trace(lambda q: q, "y").output == "y"
