"""Mocking utilities for agent tool failures, custom responses, and latency."""

from agenttest.mocks._registry import MOCK_REGISTRY
from agenttest.mocks.tool_failure import (
    mock_tool_failure,
    mock_tool_latency,
    mock_tool_response,
)

__all__ = [
    "MOCK_REGISTRY",
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
]
