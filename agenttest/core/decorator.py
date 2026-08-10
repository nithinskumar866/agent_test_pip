"""The ``@agent_test`` decorator that marks a function as an agenttest test.

Interprets test parameters, loads configuration, instantiates framework adapters,
and wraps test functions with trace capturing and agent result generation.
"""

from __future__ import annotations

import functools
import importlib
import json
from pathlib import Path
from typing import Any, Callable

__all__ = ["agent_test", "_load_config", "_get_adapter"]


def agent_test(
    _fn: Callable[..., Any] | None = None,
    *,
    timeout: int = 60,
    max_retries: int = 1,
    model_override: str | None = None,
) -> Callable[..., Any]:
    """Decorator that turns a function into an agenttest test.

    Can be used as `@agent_test` or `@agent_test(timeout=30)`.

    Args:
        _fn: Decorated target function when used without parentheses.
        timeout: Timeout in seconds for agent run.
        max_retries: Maximum number of execution retries.
        model_override: Model string override for LLM judging or execution.

    Returns:
        Decorated test wrapper callable.
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._is_agent_test = True
        fn._agenttest_config = {
            "timeout": timeout,
            "max_retries": max_retries,
            "model_override": model_override,
        }

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = _load_config()
            adapter = _get_adapter(config.get("framework", "generic"))
            # Run wrapped function
            return fn(*args, **kwargs)

        wrapper._is_agent_test = True
        wrapper._agenttest_config = fn._agenttest_config
        return wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator


def _load_config() -> dict[str, Any]:
    """Load agenttest configuration from agenttest.config.json in current directory."""
    config_path = Path.cwd() / "agenttest.config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "agenttest.config.json not found. Run 'agenttest init' first."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def _get_adapter(framework: str) -> Any:
    """Instantiate the corresponding framework adapter for the given framework name."""
    adapters = {
        "langgraph": "agenttest.adapters.langgraph.LangGraphAdapter",
        "crewai": "agenttest.adapters.crewai.CrewAIAdapter",
        "openai": "agenttest.adapters.openai_sdk.OpenAIAdapter",
        "anthropic": "agenttest.adapters.anthropic_sdk.AnthropicAdapter",
        "generic": "agenttest.adapters.base.GenericAdapter",
    }
    if framework not in adapters:
        framework = "generic"

    module_path, class_name = adapters[framework].rsplit(".", 1)
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls()
