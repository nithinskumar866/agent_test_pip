"""Pytest plugin entry point for agenttest.

Registered via the ``pytest11`` entry point in setup.cfg, so pytest loads it
automatically wherever agenttest is installed. Collection is left entirely to
pytest — agent tests are ordinary test functions carrying the ``agent`` marker.

The plugin:

* records every agent run made during each test (see :mod:`agenttest.reporting`)
  and attaches a readable trace of those runs to a failing test's report;
* runs ``async def`` agent tests itself when no async plugin claims them, so
  they work without pytest-asyncio;
* adds an agenttest section to the terminal summary;
* writes ``--agenttest-report=PATH`` (JSON) and ``--agenttest-html=PATH``;
* stops the session at ``--agenttest-max-tokens`` / ``--agenttest-max-cost``;
* sets the cassette mode from ``--agenttest-record``.

Works under pytest-xdist: each worker sends its runs to the controller as an
extra attribute on the test report, which xdist serialises with the report.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections import Counter
from collections.abc import Generator, Iterator
from typing import Any

import pytest

from agenttest import reporting
from agenttest.core.retry import FLAKY_REPORTS
from agenttest.replay import MODES

__all__ = [
    "pytest_addoption",
    "pytest_configure",
    "pytest_sessionstart",
    "pytest_runtest_protocol",
    "pytest_runtest_makereport",
    "pytest_pyfunc_call",
    "pytest_runtest_logreport",
    "pytest_sessionfinish",
    "pytest_terminal_summary",
]

#: Outcomes of ``agent``-marked tests in this session.
_outcomes: Counter[str] = Counter()
#: Every test that is agent-marked or made agent runs, by node id.
_tests: dict[str, reporting.TestRecord] = {}
_session_start: float = 0.0
_written: list[str] = []
_flaky: list[dict[str, Any]] = []
_session: Any = None


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("agenttest")
    group.addoption("--agenttest-report", metavar="PATH", default=None,
                    help="write a JSON report of agent tests and their runs to PATH")
    group.addoption("--agenttest-html", metavar="PATH", default=None,
                    help="write a self-contained HTML report of agent tests to PATH")
    group.addoption("--agenttest-no-trace", action="store_true", default=False,
                    help="do not attach agent run traces to failing tests")
    group.addoption("--agenttest-record", choices=MODES, default=None,
                    help="cassette mode for tests using @agent_test(cassette=...)")
    group.addoption("--agenttest-max-tokens", type=int, default=None, metavar="N",
                    help="stop the session once agent runs have used more than N tokens")
    group.addoption("--agenttest-max-cost", type=float, default=None, metavar="USD",
                    help="stop the session once agent runs have cost more than USD")


def pytest_configure(config: Any) -> None:
    global _session_start
    config.addinivalue_line("markers", "agent: an agenttest behavioural test (added by @agent_test)")
    _outcomes.clear()
    _tests.clear()
    _written.clear()
    _flaky.clear()
    FLAKY_REPORTS.clear()
    _session_start = time.perf_counter()
    mode = config.getoption("agenttest_record", None)
    if mode:
        os.environ["AGENTTEST_RECORD"] = mode


def pytest_sessionstart(session: Any) -> None:
    global _session
    _session = session


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: Any, nextitem: Any) -> Iterator[None]:
    reporting.begin_test()
    try:
        yield
    finally:
        reporting.end_test()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Generator[None, Any, None]:
    outcome = yield
    report = outcome.get_result()
    if report.when == "teardown":
        # Picked up in pytest_runtest_logreport: here, or on the xdist controller.
        report.agenttest = {
            "runs": [reporting.result_to_dict(run) for run in reporting.current_runs()],
            "flaky": [{"test": f.test_name, "summary": f.summary()} for f in FLAKY_REPORTS],
        }
        FLAKY_REPORTS.clear()
        return
    if report.when != "call" or not report.failed or item.config.getoption("agenttest_no_trace"):
        return
    runs = reporting.current_runs()
    if runs:
        text = "\n\n".join(
            reporting.describe_result(run, f"run {i} of {len(runs)}") for i, run in enumerate(runs, start=1)
        )
        report.sections.append(("agenttest runs", text))


def _claimed_by_async_plugin(item: Any) -> bool:
    if any(item.get_closest_marker(name) for name in ("asyncio", "anyio", "trio")):
        return True
    config = item.config
    if config.pluginmanager.hasplugin("asyncio"):
        try:
            return config.getini("asyncio_mode") == "auto"
        except ValueError:
            return False
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: Any) -> bool | None:
    """Run an ``async def`` agent test on a fresh event loop if nothing else will."""
    function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(function) or pyfuncitem.get_closest_marker("agent") is None:
        return None
    if _claimed_by_async_plugin(pyfuncitem):
        return None
    names = pyfuncitem._fixtureinfo.argnames
    asyncio.run(function(**{name: pyfuncitem.funcargs[name] for name in names}))
    return True


def pytest_runtest_logreport(report: Any) -> None:
    """Count each test once: its call outcome, or a setup failure/skip."""
    extra = getattr(report, "agenttest", None)
    if extra is not None:
        _collect_runs(report.nodeid, extra)
        return
    if not (report.when == "call" or (report.when == "setup" and not report.passed)):
        return
    agent_marked = "agent" in report.keywords
    if agent_marked:
        _outcomes[report.outcome] += 1
    record = _tests.get(report.nodeid)
    if record is None:
        record = _tests[report.nodeid] = reporting.TestRecord(report.nodeid)
    record.outcome = report.outcome
    record.duration_s = report.duration
    record.agent_marked = agent_marked
    if report.failed or report.skipped:
        crash = getattr(report.longrepr, "reprcrash", None)
        record.message = getattr(crash, "message", None) or str(report.longrepr)[:2000]


def _collect_runs(nodeid: str, extra: dict[str, Any]) -> None:
    _flaky.extend(extra["flaky"])
    if not extra["runs"]:
        return
    record = _tests.get(nodeid)
    if record is None:
        record = _tests[nodeid] = reporting.TestRecord(nodeid)
    record.runs = extra["runs"]
    _enforce_budget()


def _all_runs() -> list[dict[str, Any]]:
    return [run for test in _tests.values() for run in test.runs]


def _enforce_budget() -> None:
    if _session is None:
        return
    config = _session.config
    max_tokens, max_cost = config.getoption("agenttest_max_tokens", None), config.getoption("agenttest_max_cost", None)
    if max_tokens is None and max_cost is None:
        return
    runs = _all_runs()
    tokens, cost = sum(r["tokens_used"] for r in runs), reporting.total_cost(runs)
    if max_tokens is not None and tokens > max_tokens:
        _session.shouldstop = f"agenttest: used {tokens} tokens; --agenttest-max-tokens is {max_tokens}"
    elif max_cost is not None and cost is None:
        _session.shouldstop = "agenttest: --agenttest-max-cost is set, but a model has no known price"
    elif max_cost is not None and cost > max_cost:
        _session.shouldstop = f"agenttest: cost ${cost:.4f}; --agenttest-max-cost is ${max_cost:g}"


def _reported_tests() -> list[reporting.TestRecord]:
    return [t for t in _tests.values() if t.agent_marked or t.runs]


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    config = session.config
    if hasattr(config, "workerinput"):  # an xdist worker: the controller writes the reports
        return
    json_path = config.getoption("agenttest_report", None)
    html_path = config.getoption("agenttest_html", None)
    if not (json_path or html_path):
        return
    report = reporting.build_report(_reported_tests(), list(_flaky),
                                    duration_s=time.perf_counter() - _session_start)
    if json_path:
        reporting.write_json(report, json_path)
        _written.append(json_path)
    if html_path:
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(reporting.render_html(report))
        _written.append(html_path)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: Any, config: Any) -> None:
    """Add an agenttest section to pytest's summary when agent tests ran."""
    tests = _reported_tests()
    if not _outcomes and not tests:
        return
    runs = _all_runs()
    terminalreporter.write_sep("=", "agenttest summary")
    if _outcomes:
        terminalreporter.write_line(
            f"Agent tests: {_outcomes['passed']} passed, {_outcomes['failed']} failed, "
            f"{_outcomes['skipped']} skipped"
        )
    if runs:
        failed_tools = sum(len(r["failed_tools"]) for r in runs)
        cost = reporting.total_cost(runs)
        terminalreporter.write_line(
            f"Agent runs: {len(runs)} ({sum(not r['success'] for r in runs)} failed), "
            f"{sum(r['tokens_used'] for r in runs)} tokens, "
            f"{sum(r['tool_calls'] for r in runs)} tool calls ({failed_tools} failed), "
            f"slowest {max(r['latency_ms'] for r in runs):.0f} ms"
            + ("" if cost is None or not cost else f", ${cost:.4f}")
        )
    if _flaky:
        terminalreporter.write_line(f"Flaky (passed, but not on every run/attempt): {len(_flaky)}")
        for item in _flaky:
            terminalreporter.write_line(f"  {item['test']}: {item['summary']}")
    for path in _written:
        terminalreporter.write_line(f"Report written: {path}")
