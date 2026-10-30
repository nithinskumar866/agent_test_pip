"""Framework adapters, and how the engine picks one for a given agent."""

from __future__ import annotations

import importlib
from typing import Any, Final

from agenttest.adapters.base import BaseAdapter, GenericAdapter
from agenttest.config import FRAMEWORKS, AgentTestConfigError

__all__ = ["BaseAdapter", "GenericAdapter", "get_adapter", "resolve_adapter"]

_ADAPTER_PATHS: Final[dict[str, str]] = {
    "langgraph": "agenttest.adapters.langgraph:LangGraphAdapter",
    "crewai": "agenttest.adapters.crewai:CrewAIAdapter",
    "openai": "agenttest.adapters.openai_sdk:OpenAIAdapter",
    "anthropic": "agenttest.adapters.anthropic_sdk:AnthropicAdapter",
    "generic": "agenttest.adapters.base:GenericAdapter",
}

#: Adapters that can recognise an agent by its type, in detection order.
_DETECTABLE: Final[tuple[str, ...]] = ("langgraph", "openai", "crewai")


def _adapter_class(framework: str) -> type[BaseAdapter]:
    if framework not in _ADAPTER_PATHS:
        raise AgentTestConfigError(
            f"Unknown framework {framework!r}; expected one of {', '.join(FRAMEWORKS)}"
        )
    module_path, class_name = _ADAPTER_PATHS[framework].split(":")
    return getattr(importlib.import_module(module_path), class_name)


def get_adapter(framework: str) -> BaseAdapter:
    """Instantiate the adapter for a framework name from ``agenttest.config.json``."""
    return _adapter_class(framework)()


def resolve_adapter(agent: Any, adapter: BaseAdapter | str | None, configured: str) -> BaseAdapter:
    """Choose the adapter for one run.

    Precedence: an explicit ``adapter`` argument, then the agent's own type
    (a compiled LangGraph graph is always a LangGraph agent), then the
    framework in the config file.
    """
    if isinstance(adapter, BaseAdapter):
        return adapter
    if isinstance(adapter, str):
        return get_adapter(adapter)
    for framework in _DETECTABLE:
        cls = _adapter_class(framework)
        if cls.can_handle(agent):
            return cls()
    return get_adapter(configured)
