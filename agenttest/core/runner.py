"""Runner module for executing agent tests programmatically or under pytest.

Provides utilities for running agent functions with trace capturing and returning
an AgentResult object directly.
"""

from __future__ import annotations

from typing import Any, Callable

from agenttest.adapters.base import BaseAdapter, GenericAdapter
from agenttest.core.result import AgentResult

__all__ = ["run_agent_test"]


def run_agent_test(
    agent_fn: Callable[..., Any],
    *args: Any,
    adapter: BaseAdapter | None = None,
    **kwargs: Any,
) -> AgentResult:
    """Execute an agent function wrapped in a trace runner and return an AgentResult.

    Args:
        agent_fn: The target agent function to execute.
        *args: Positional arguments to pass to the agent function.
        adapter: Optional custom adapter instance. Defaults to GenericAdapter.
        **kwargs: Keyword arguments to pass to the agent function.

    Returns:
        An AgentResult instance containing execution outcomes and metrics.
    """
    runner_adapter = adapter if adapter is not None else GenericAdapter()
    return runner_adapter.run_with_trace(agent_fn, *args, **kwargs)
