"""Mocking utilities for agent tool failures, custom responses, and latency."""

from agenttest.mocks._registry import (
    MOCK_REGISTRY,
    MockedToolError,
    MockHandle,
    apply_mock,
    apply_mock_async,
)
from agenttest.mocks.tool_failure import (
    mock_tool_failure,
    mock_tool_latency,
    mock_tool_response,
)

__all__ = [
    "MOCK_REGISTRY",
    "MockHandle",
    "MockedToolError",
    "apply_mock",
    "apply_mock_async",
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
]
