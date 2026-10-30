"""Pytest plugin entry point for agenttest.

Registered via the ``pytest11`` entry point in setup.cfg, so pytest loads it
automatically wherever agenttest is installed. Collection is left entirely to
pytest — agent tests are ordinary test functions carrying the ``agent`` marker.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["pytest_configure", "pytest_runtest_logreport", "pytest_terminal_summary"]

#: Outcomes of ``agent``-marked tests in this session.
_outcomes: Counter[str] = Counter()


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "agent: an agenttest behavioural test (added by @agent_test)")
    _outcomes.clear()


def pytest_runtest_logreport(report: Any) -> None:
    """Count each ``agent``-marked test once: its call outcome, or a setup failure/skip."""
    if "agent" not in report.keywords:
        return
    if report.when == "call" or (report.when == "setup" and not report.passed):
        _outcomes[report.outcome] += 1


def pytest_terminal_summary(terminalreporter: Any, exitstatus: Any, config: Any) -> None:
    """Add an agenttest line to pytest's summary when agent tests ran."""
    if not _outcomes:
        return
    terminalreporter.write_sep("=", "agenttest summary")
    terminalreporter.write_line(
        f"Agent tests: {_outcomes['passed']} passed, {_outcomes['failed']} failed, "
        f"{_outcomes['skipped']} skipped"
    )
