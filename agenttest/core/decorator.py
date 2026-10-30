"""The ``@agent_test`` decorator that marks a function as an agenttest test.

The decorator does two things:

* marks the test with the ``agent`` pytest marker, so agent tests can be
  selected (``pytest -m agent``) and counted in the summary;
* makes its settings active while the test runs, so every
  :func:`agenttest.run` / :func:`agenttest.arun` call inside the test
  inherits them — no need to repeat ``timeout=`` on each run.

It never loads configuration at decoration or call time; a project without
``agenttest.config.json`` runs on defaults.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import pytest

from agenttest.core.runner import RunSettings, settings_scope

__all__ = ["agent_test"]


def agent_test(
    _fn: Callable[..., Any] | None = None,
    *,
    timeout: float | None = None,
    max_retries: int = 1,
    model_override: str | None = None,
) -> Callable[..., Any]:
    """Mark a function as an agent test.

    Usable bare (``@agent_test``) or with options (``@agent_test(timeout=30)``).
    Works on ``async def`` tests too (with an async pytest plugin such as
    pytest-asyncio or anyio).

    Args:
        timeout: Default timeout in seconds for agent runs inside the test.
            ``None`` falls back to the config file's ``timeout``.
        max_retries: Attempts allowed per test (applied by the Phase 2 retry engine).
        model_override: Judge model to use inside this test.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    settings = RunSettings(timeout=timeout, max_retries=max_retries, model_override=model_override)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with settings_scope(settings):
                    return await fn(*args, **kwargs)

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with settings_scope(settings):
                    return fn(*args, **kwargs)

            wrapper = sync_wrapper

        wrapper._is_agent_test = True  # type: ignore[attr-defined]
        wrapper._agenttest_config = {  # type: ignore[attr-defined]
            "timeout": timeout,
            "max_retries": max_retries,
            "model_override": model_override,
        }
        return pytest.mark.agent(wrapper)

    if _fn is not None:
        return decorator(_fn)
    return decorator
