"""The ``@agent_test`` decorator that marks a function as an agenttest test.

The decorator:

* marks the test with the ``agent`` pytest marker, so agent tests can be
  selected (``pytest -m agent``) and counted in the summary;
* makes its settings active while the test runs, so every
  :func:`agenttest.run` / :func:`agenttest.arun` call inside the test
  inherits them (e.g. ``timeout``);
* executes the test body under a retry / pass-rate policy
  (see :mod:`agenttest.core.retry`) when ``runs`` or ``max_retries`` ask for it.

Options left as ``None`` fall back to ``agenttest.config.json`` (read when the
test runs, not at import), then to the built-in defaults. No config file is needed.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import pytest

from agenttest.config import load_config
from agenttest.core.retry import RetryPolicy, aexecute, execute
from agenttest.core.runner import RunSettings, settings_scope

__all__ = ["agent_test"]


def agent_test(
    _fn: Callable[..., Any] | None = None,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    runs: int | None = None,
    min_pass_rate: float | None = None,
    model_override: str | None = None,
) -> Callable[..., Any]:
    """Mark a function as an agent test.

    Usable bare (``@agent_test``) or with options::

        @agent_test(runs=5, min_pass_rate=0.8, max_retries=1, timeout=30)
        def test_agent_answers():
            ...

    Works on ``async def`` tests too (with an async pytest plugin such as
    pytest-asyncio or anyio).

    Args:
        timeout: Default timeout in seconds for agent runs inside the test.
        max_retries: Extra attempts a failing run gets (0 = none).
        runs: How many times to execute the test body.
        min_pass_rate: Fraction of runs that must pass (0 < rate <= 1).
        model_override: Judge model to use inside this test.

    Note:
        Fixtures are created once per test, so every run and retry sees the
        same fixture values.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    # Validate explicit values eagerly, so mistakes fail at import, not mid-suite.
    RetryPolicy(
        runs=1 if runs is None else runs,
        max_retries=0 if max_retries is None else max_retries,
        min_pass_rate=1.0 if min_pass_rate is None else min_pass_rate,
    )

    def resolve() -> tuple[RunSettings, RetryPolicy]:
        config = load_config()
        policy = RetryPolicy(
            runs=config.runs if runs is None else runs,
            max_retries=config.max_retries if max_retries is None else max_retries,
            min_pass_rate=config.min_pass_rate if min_pass_rate is None else min_pass_rate,
        )
        settings = RunSettings(timeout=timeout, max_retries=policy.max_retries, model_override=model_override)
        return settings, policy

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        test_name = f"{fn.__module__}::{fn.__qualname__}"

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings, policy = resolve()
                with settings_scope(settings):
                    return await aexecute(test_name, policy, lambda: fn(*args, **kwargs))

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings, policy = resolve()
                with settings_scope(settings):
                    return execute(test_name, policy, lambda: fn(*args, **kwargs))

            wrapper = sync_wrapper

        wrapper._is_agent_test = True  # type: ignore[attr-defined]
        wrapper._agenttest_config = {  # type: ignore[attr-defined]
            "timeout": timeout,
            "max_retries": max_retries,
            "runs": runs,
            "min_pass_rate": min_pass_rate,
            "model_override": model_override,
        }
        # Attach the mark directly: pytest.mark.agent(fn) would treat a
        # function named "<lambda>" as a mark argument instead of decorating it.
        wrapper.pytestmark = [*getattr(wrapper, "pytestmark", []), pytest.mark.agent.mark]  # type: ignore[attr-defined]
        return wrapper

    if _fn is not None:
        return decorator(_fn)
    return decorator
