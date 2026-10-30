"""Tests for tool mocking context managers and registry."""

from agenttest.mocks import (
    MOCK_REGISTRY,
    mock_tool_failure,
    mock_tool_latency,
    mock_tool_response,
)


def test_mock_tool_failure_adds_to_registry():
    assert "web_search" not in MOCK_REGISTRY
    with mock_tool_failure("web_search", "Custom failure"):
        assert "web_search" in MOCK_REGISTRY
        assert MOCK_REGISTRY["web_search"]["type"] == "failure"
        assert MOCK_REGISTRY["web_search"]["error"] == "Custom failure"


def test_mock_tool_failure_cleans_up_after_exit():
    assert "temp_tool" not in MOCK_REGISTRY
    with mock_tool_failure("temp_tool"):
        assert "temp_tool" in MOCK_REGISTRY
    assert "temp_tool" not in MOCK_REGISTRY


def test_mock_tool_response_adds_to_registry():
    with mock_tool_response("calculator", {"result": 42}):
        assert "calculator" in MOCK_REGISTRY
        assert MOCK_REGISTRY["calculator"]["type"] == "response"
        assert MOCK_REGISTRY["calculator"]["data"] == {"result": 42}
    assert "calculator" not in MOCK_REGISTRY


def test_mock_tool_latency_adds_to_registry():
    with mock_tool_latency("database", 500):
        assert "database" in MOCK_REGISTRY
        assert MOCK_REGISTRY["database"]["type"] == "latency"
        assert MOCK_REGISTRY["database"]["delay"] == 0.5
    assert "database" not in MOCK_REGISTRY


def test_nested_mocks_work_correctly():
    with mock_tool_failure("tool_a"):
        with mock_tool_response("tool_b", "ok"):
            assert "tool_a" in MOCK_REGISTRY
            assert "tool_b" in MOCK_REGISTRY
        assert "tool_b" not in MOCK_REGISTRY
        assert "tool_a" in MOCK_REGISTRY
    assert "tool_a" not in MOCK_REGISTRY


# --- Phase 2: the mock engine and @agenttest.tool wiring ---------------------

import asyncio
import threading
import time

import pytest

import agenttest
from agenttest import MockedToolError, run, tool


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@tool
def web_search(query: str) -> str:
    return f"real results for {query}"


@tool
async def async_search(query: str) -> str:
    return f"async results for {query}"


def test_nested_same_name_restores_outer_mock():
    with mock_tool_response("web_search", "outer"):
        with mock_tool_response("web_search", "inner"):
            assert web_search("q") == "inner"
        assert web_search("q") == "outer"
    assert web_search("q") == "real results for q"


def test_out_of_order_exit_removes_the_right_mock():
    outer = mock_tool_response("web_search", "outer")
    inner = mock_tool_response("web_search", "inner")
    outer.__enter__()
    inner.__enter__()
    outer.__exit__(None, None, None)
    assert web_search("q") == "inner"
    inner.__exit__(None, None, None)
    assert "web_search" not in MOCK_REGISTRY


def test_failure_mock_raises_without_calling_tool():
    calls = []

    @tool
    def side_effect():
        calls.append(1)

    with mock_tool_failure("side_effect", "boom") as handle:
        with pytest.raises(MockedToolError, match="boom"):
            side_effect()
    assert calls == []
    assert handle.calls == 1 and handle.called


def test_failure_mock_custom_exception_class_and_instance():
    with mock_tool_failure("web_search", "503", exception=ConnectionError):
        with pytest.raises(ConnectionError, match="503"):
            web_search("q")
    with mock_tool_failure("web_search", exception=TimeoutError("slow")):
        with pytest.raises(TimeoutError, match="slow"):
            web_search("q")


def test_response_mock_returns_data_verbatim():
    payload = {"results": [1, 2]}
    with mock_tool_response("web_search", payload):
        assert web_search("q") is payload


def test_latency_mock_delays_then_calls_real_tool():
    with mock_tool_latency("web_search", 50) as handle:
        start = time.perf_counter()
        assert web_search("q") == "real results for q"
        assert time.perf_counter() - start >= 0.045
    assert handle.calls == 1


def test_negative_latency_rejected():
    with pytest.raises(ValueError):
        with mock_tool_latency("web_search", -1):
            pass


def test_unmatched_mock_reports_zero_calls():
    with mock_tool_failure("web_serach") as handle:  # typo: never matches
        web_search("q")
    assert handle.calls == 0 and not handle.called


def test_async_tool_mocks():
    async def scenario():
        with mock_tool_response("async_search", "canned"):
            assert await async_search("q") == "canned"
        with mock_tool_failure("async_search", "down"):
            with pytest.raises(MockedToolError):
                await async_search("q")
        with mock_tool_latency("async_search", 20):
            assert await async_search("q") == "async results for q"

    asyncio.run(scenario())


def test_mocked_calls_are_traced_and_marked():
    def agent():
        try:
            web_search("news")
        except MockedToolError:
            return "Search is down; here is a cached summary."

    with mock_tool_failure("web_search", "503 Service Unavailable"):
        result = run(agent)

    step = result.trace.tool_calls[0]
    assert step.error == "MockedToolError: 503 Service Unavailable"
    assert step.metadata["mocked"] == "failure"
    assert result.gracefully_degraded is True


def test_mock_applies_inside_timeout_worker_thread():
    seen = {}

    def agent():
        seen["thread"] = threading.current_thread().name
        return web_search("q")

    with mock_tool_response("web_search", "mocked"):
        result = run(agent, timeout=5)

    assert result.output == "mocked"
    assert seen["thread"].startswith("agenttest")


def test_mock_applies_in_threads_that_do_not_inherit_context():
    out = {}

    def agent():
        worker = threading.Thread(target=lambda: out.setdefault("value", web_search("q")))
        worker.start()
        worker.join()
        return out["value"]

    with mock_tool_response("web_search", "mocked"):
        assert run(agent).output == "mocked"


def test_nested_interception_applies_mock_once():
    @tool(name="lookup")
    def inner(q):
        return "real"

    @tool(name="lookup")
    def outer(q):
        return inner(q)

    with mock_tool_latency("lookup", 1) as handle:
        result = run(lambda: outer("x"))
    assert handle.calls == 1
    assert result.tools_called == ["lookup"]


def test_public_exports():
    assert agenttest.MockHandle is not None
    assert agenttest.mock_tool_failure is mock_tool_failure
