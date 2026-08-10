"""Pytest plugin entry point for agenttest.

Registered via the ``pytest11`` entry point in setup.cfg, which means pytest
loads this module automatically for any project that has agenttest installed.
"""

from __future__ import annotations

from typing import Any

import pytest

__all__ = [
    "pytest_collect_file",
    "pytest_runtest_call",
    "pytest_terminal_summary",
]


def pytest_collect_file(parent: Any, file_path: Any) -> Any:
    """Tell pytest to collect our test files."""
    if getattr(file_path, "suffix", None) == ".py" and getattr(file_path, "name", "").startswith("test_"):
        return pytest.Module.from_parent(parent, path=file_path)
    return None


def pytest_runtest_call(item: Any) -> None:
    """Hook into each test call.

    If the test function has _is_agent_test=True, handle it specially if needed.
    """
    pass


def pytest_terminal_summary(terminalreporter: Any, exitstatus: Any, config: Any) -> None:
    """Add agenttest summary to pytest output."""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    if passed + failed > 0:
        terminalreporter.write_sep("=", "agenttest summary")
        terminalreporter.write_line(f"Agent tests: {passed} passed, {failed} failed")
