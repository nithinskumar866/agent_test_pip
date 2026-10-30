"""Tests for the pytest plugin: summary, failure traces, reports, async tests.

Each test runs a small pytest session with ``pytester`` and the plugin loaded
explicitly (``-p agenttest.plugin``), as an installed agenttest would load it
through its ``pytest11`` entry point.
"""

import json
import os
from pathlib import Path

import pytest

from tests._plugin_args import plugin_args

pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def importable_agenttest(monkeypatch):
    # Sessions run in a subprocess (so they don't share the plugin's state
    # with this one); make sure it can import agenttest from this checkout.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))

AGENT_TESTS = '''
import agenttest
from agenttest import agent_test, mock_tool_failure


@agenttest.tool
def search(query):
    return "Paris is the capital of France."


def agent(question):
    try:
        return "Answer: " + search(question)
    except Exception:
        return "Search is down; answering from memory: Paris."


@agent_test
def test_passes():
    result = agenttest.run(agent, "capital?")
    assert result.success


@agent_test
def test_fails_with_trace():
    with mock_tool_failure("search", "503 Service Unavailable"):
        result = agenttest.run(agent, "capital?")
    assert "Answer" in result.output, "the agent did not answer from search"


def test_plain_test_with_agent_run():
    assert agenttest.run(agent, "q").success


def test_plain_test_without_agent_runs():
    assert True
'''


def run_session(pytester, source, *args):
    pytester.makepyfile(test_agents=source)
    return pytester.runpytest_subprocess(*plugin_args(), "-p", "no:cacheprovider", *args)


def test_summary_counts_agent_tests_and_runs(pytester):
    result = run_session(pytester, AGENT_TESTS)
    result.assert_outcomes(passed=3, failed=1)
    result.stdout.fnmatch_lines([
        "*agenttest summary*",
        "Agent tests: 1 passed, 1 failed, 0 skipped",
        "Agent runs: 3 (0 failed), 0 tokens, 3 tool calls (1 failed), slowest * ms",
    ])


def test_failing_test_shows_the_agent_trace(pytester):
    result = run_session(pytester, AGENT_TESTS)
    result.stdout.fnmatch_lines([
        "*agenttest runs*",
        "run 1 of 1: ok - * ms, 1 steps, 0 tokens, gracefully degraded",
        "*1. tool search*mocked failure*",
        "*error: MockedToolError: 503 Service Unavailable",
        "*output: 'Search is down; answering from memory: Paris.'",
    ])


def test_trace_can_be_switched_off(pytester):
    result = run_session(pytester, AGENT_TESTS, "--agenttest-no-trace")
    assert "agenttest runs" not in result.stdout.str()


def test_json_report(pytester):
    result = run_session(pytester, AGENT_TESTS, "--agenttest-report=report.json")
    result.stdout.fnmatch_lines(["Report written: report.json"])
    report = json.loads((pytester.path / "report.json").read_text(encoding="utf-8"))

    assert report["summary"] | {"agent_latency_ms": 0} == {
        "tests": 3, "passed": 2, "failed": 1, "skipped": 0, "agent_runs": 3, "failed_runs": 0,
        "tokens_used": 0, "tool_calls": 3, "failed_tool_calls": 1, "agent_latency_ms": 0, "flaky": 0,
    }
    tests = {t["nodeid"].split("::")[1]: t for t in report["tests"]}
    assert set(tests) == {"test_passes", "test_fails_with_trace", "test_plain_test_with_agent_run"}
    failed = tests["test_fails_with_trace"]
    assert failed["outcome"] == "failed" and failed["agent_marked"] is True
    assert "the agent did not answer from search" in failed["message"]
    run = failed["runs"][0]
    assert run["failed_tools"] == ["search"]
    assert run["steps"][0]["metadata"] == {"mocked": "failure"}
    assert tests["test_plain_test_with_agent_run"]["agent_marked"] is False


def test_html_report(pytester):
    run_session(pytester, AGENT_TESTS, "--agenttest-html=report.html")
    page = (pytester.path / "report.html").read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>")
    assert "test_fails_with_trace" in page
    assert "MockedToolError: 503 Service Unavailable" in page


def test_no_summary_without_agent_tests(pytester):
    result = run_session(pytester, "def test_x():\n    assert True\n")
    assert "agenttest summary" not in result.stdout.str()


def test_flaky_tests_are_listed_in_summary_and_report(pytester):
    source = '''
from agenttest import agent_test

calls = []


@agent_test(runs=3, min_pass_rate=0.6)
def test_flaky():
    calls.append(1)
    assert len(calls) != 2
'''
    result = run_session(pytester, source, "--agenttest-report=r.json")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["Flaky (passed, but not on every run/attempt): 1"])
    report = json.loads((pytester.path / "r.json").read_text(encoding="utf-8"))
    assert report["summary"]["flaky"] == 1
    assert "2/3 runs passed" in report["flaky"][0]["summary"]


# --- async tests ------------------------------------------------------------------

ASYNC_TESTS = '''
import asyncio

import pytest

import agenttest
from agenttest import agent_test


async def agent(question):
    await asyncio.sleep(0)
    return "async answer"


@pytest.fixture
def question():
    return "q?"


@agent_test
async def test_async_agent(question):
    result = await agenttest.arun(agent, question)
    assert result.output == "async answer"


@agent_test(runs=2)
async def test_async_with_runs():
    result = await agenttest.arun(agent, "q")
    assert result.success


@agent_test
async def test_async_failure():
    result = await agenttest.arun(agent, "q")
    assert result.output == "something else"
'''


def test_async_agent_tests_run_without_an_async_plugin(pytester):
    result = run_session(pytester, ASYNC_TESTS, "-p", "no:anyio")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["Agent tests: 2 passed, 1 failed, 0 skipped"])


def test_async_tests_marked_for_anyio_are_left_to_anyio(pytester):
    pytest.importorskip("anyio")
    source = '''
import pytest

from agenttest import agent_test

ran = []


@pytest.mark.anyio
@agent_test
async def test_anyio_runs_it():
    ran.append(1)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_ran_once():
    assert ran == [1]
'''
    result = run_session(pytester, source)
    result.assert_outcomes(passed=2)


def test_plain_async_tests_are_not_claimed(pytester):
    # Only agent tests are run by the plugin; other async tests keep pytest's behaviour.
    source = "async def test_plain():\n    pass\n"
    result = run_session(pytester, source, "-p", "no:anyio")
    assert "agenttest summary" not in result.stdout.str()
    assert result.parseoutcomes().get("passed", 0) == 0  # pytest itself refuses to run it
