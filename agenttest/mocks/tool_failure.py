"""Context managers that inject tool failures, custom responses, and latency.

Provides mocking utilities for testing agent behavior under edge cases like tool errors,
mocked API payloads, or slow response times.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from agenttest.mocks._registry import MOCK_REGISTRY

__all__ = [
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
]


@contextmanager
def mock_tool_failure(tool_name: str, error_message: str | None = None) -> Generator[None, None, None]:
    """Context manager that makes a named tool raise an exception when called.

    Args:
        tool_name: Name of the tool to simulate failure for.
        error_message: Optional custom error message string.

    Yields:
        None
    """
    err = error_message or f"Tool '{tool_name}' failed (mocked)"
    MOCK_REGISTRY[tool_name] = {"type": "failure", "error": err}
    try:
        yield
    finally:
        MOCK_REGISTRY.pop(tool_name, None)


@contextmanager
def mock_tool_response(tool_name: str, returns: dict[str, Any] | str) -> Generator[None, None, None]:
    """Context manager that makes a named tool return custom mock data.

    Args:
        tool_name: Name of the tool to mock response for.
        returns: Data dictionary or string to return when tool is invoked.

    Yields:
        None
    """
    MOCK_REGISTRY[tool_name] = {"type": "response", "data": returns}
    try:
        yield
    finally:
        MOCK_REGISTRY.pop(tool_name, None)


@contextmanager
def mock_tool_latency(tool_name: str, delay_ms: int) -> Generator[None, None, None]:
    """Context manager that injects artificial delay/latency to a named tool call.

    Args:
        tool_name: Name of the tool to delay.
        delay_ms: Artificial delay in milliseconds.

    Yields:
        None
    """
    MOCK_REGISTRY[tool_name] = {"type": "latency", "delay": delay_ms / 1000.0}
    try:
        yield
    finally:
        MOCK_REGISTRY.pop(tool_name, None)
