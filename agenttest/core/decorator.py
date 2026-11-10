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
from contextlib import nullcontext
from typing import Any

import pytest

from agenttest.config import AgentTestBudgetError, load_config
from agenttest.core.retry import RetryPolicy, aexecute, execute
from agenttest.core.runner import RunSettings, Usage, settings_scope
from agenttest.replay import cassette_for

__all__ = ["agent_test"]


def agent_test(
    _fn: Callable[..., Any] | None = None,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    runs: int | None = None,
    min_pass_rate: float | None = None,
    model_override: str | None = None,
    max_tokens: int | None = None,
    max_cost_usd: float | None = None,
    cassette: bool | str | None = None,
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
        max_tokens: Fail the test once its agent runs have used more tokens than this.
        max_cost_usd: Fail the test once its agent runs have cost more than this.
        cassette: Record model API calls and replay them on later runs: ``True``
            for a file next to the test, or a path relative to the test file.
            See :mod:`agenttest.replay`.

    Note:
        Fixtures are created once per test, so every run and retry sees the
        same fixture values.
    """
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive")
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

        def check_budget(usage: Usage) -> None:
            if max_tokens is not None and usage.tokens > max_tokens:
                raise AgentTestBudgetError(f"{test_name}: used {usage.tokens} tokens; the budget is {max_tokens}.")
            if max_cost_usd is not None and not usage.priced:
                raise AgentTestBudgetError(
                    f"{test_name}: has a cost budget, but a model has no known price. "
                    "Add it under 'pricing' in agenttest.config.json."
                )
            if max_cost_usd is not None and usage.cost_usd > max_cost_usd:
                raise AgentTestBudgetError(
                    f"{test_name}: cost ${usage.cost_usd:.6f}; the budget is ${max_cost_usd:g}."
                )

        def recording() -> Any:
            return cassette_for(fn, cassette) if cassette else nullcontext()

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings, policy = resolve()

                async def body() -> Any:
                    value = await fn(*args, **kwargs)
                    check_budget(settings.usage)
                    return value

                with recording(), settings_scope(settings):
                    return await aexecute(test_name, policy, body)

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                settings, policy = resolve()

                def body() -> Any:
                    value = fn(*args, **kwargs)
                    check_budget(settings.usage)
                    return value

                with recording(), settings_scope(settings):
                    return execute(test_name, policy, body)

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
