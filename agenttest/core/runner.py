"""The execution engine: run an agent once and describe what happened.

:func:`run` and :func:`arun` are the single entry point for executing an
agent under test. For every run the engine:

1. picks an adapter (explicit > detected from the agent's type > config),
2. opens a fresh :class:`~agenttest.core.trace.Trace`,
3. installs the adapter's instrumentation for this run only,
4. calls the agent — sync or async — under an optional timeout,
5. captures any exception with its traceback instead of losing it,
6. returns an :class:`~agenttest.core.result.AgentResult` built from the trace.

The agent's exception never escapes ``run``; inspect ``result.success`` /
``result.error``, or call ``result.raise_for_error()``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import inspect
import time
import traceback as tb
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from agenttest.adapters import BaseAdapter, resolve_adapter
from agenttest.config import AgentTestError, AgentTestTimeoutError, load_config
from agenttest.core.result import AgentResult
from agenttest.core.trace import Trace, tracing
from agenttest.reporting import record_result

__all__ = ["run", "arun", "run_agent_test", "RunSettings", "current_settings", "settings_scope"]


@dataclass(frozen=True)
class RunSettings:
    """Per-test overrides installed by ``@agent_test``."""

    timeout: float | None = None
    max_retries: int = 0
    model_override: str | None = None


_SETTINGS: contextvars.ContextVar[RunSettings | None] = contextvars.ContextVar(
    "agenttest_run_settings", default=None
)


def current_settings() -> RunSettings | None:
    """Settings of the enclosing ``@agent_test`` test, if any."""
    return _SETTINGS.get()


@contextmanager
def settings_scope(settings: RunSettings) -> Iterator[RunSettings]:
    """Make ``settings`` active for runs inside the block."""
    token = _SETTINGS.set(settings)
    try:
        yield settings
    finally:
        _SETTINGS.reset(token)


def _prepare(agent: Any, adapter: BaseAdapter | str | None, timeout: float | None) -> tuple[BaseAdapter, float | None]:
    config = load_config()
    settings = current_settings()
    if timeout is None and settings is not None:
        timeout = settings.timeout
    if timeout is None:
        timeout = config.timeout
    if timeout is not None and timeout <= 0:
        raise AgentTestError("timeout must be positive")
    return resolve_adapter(agent, adapter, config.framework), timeout


def _finish(*args: Any, **kwargs: Any) -> AgentResult:
    result = _build_result(*args, **kwargs)
    record_result(result)
    return result


def _build_result(
    adapter: BaseAdapter,
    trace: Trace,
    start: float,
    raw: Any = None,
    exc: BaseException | None = None,
) -> AgentResult:
    latency_ms = (time.perf_counter() - start) * 1000
    if exc is not None:
        return AgentResult.from_trace(
            trace,
            output="",
            latency_ms=latency_ms,
            exception=exc,
            traceback="".join(tb.format_exception(exc)),
        )
    try:
        output = adapter.extract_output(raw)
    except Exception as extract_exc:
        return _build_result(adapter, trace, start, exc=extract_exc)
    return AgentResult.from_trace(trace, output=output, latency_ms=latency_ms, raw_output=raw)


def _call_sync(adapter: BaseAdapter, agent: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Invoke the agent, driving it to completion if it is async."""
    with adapter.instrument():
        raw = adapter.invoke(agent, *args, **kwargs)
        if inspect.isawaitable(raw):
            raw = asyncio.run(_await(raw))
        return raw


async def _await(awaitable: Any) -> Any:
    return await awaitable


def _timeout_error(timeout: float) -> AgentTestTimeoutError:
    return AgentTestTimeoutError(f"agent run exceeded timeout of {timeout:g}s")


def run(
    agent: Any,
    *args: Any,
    adapter: BaseAdapter | str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> AgentResult:
    """Run ``agent`` once and return what happened.

    Args:
        agent: A callable, an object with ``run``/``invoke``, or a framework
            agent (compiled LangGraph graph, OpenAI ``Agent``, CrewAI ``Crew``).
            Async agents are supported.
        *args: Passed to the agent.
        adapter: Force an adapter — an instance or a framework name.
        timeout: Seconds before the run is abandoned and reported as failed.
            Defaults to the ``@agent_test`` timeout, then the config's.
        **kwargs: Passed to the agent.

    Raises:
        AgentTestError: Only for agenttest misuse or misconfiguration — never
            for the agent's own failures, which land on the result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise AgentTestError(
            "agenttest.run() was called inside a running event loop; use `await agenttest.arun(...)`."
        )

    chosen, timeout = _prepare(agent, adapter, timeout)
    with tracing() as trace:
        start = time.perf_counter()
        if timeout is None:
            try:
                raw = _call_sync(chosen, agent, args, kwargs)
            except Exception as exc:
                return _finish(chosen, trace, start, exc=exc)
            return _finish(chosen, trace, start, raw)

        # Run in a worker so a hung agent can be abandoned. The worker gets a
        # copy of this context, so it records into the same trace.
        context = contextvars.copy_context()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="agenttest")
        future = pool.submit(context.run, _call_sync, chosen, agent, args, kwargs)
        try:
            raw = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # Snapshot: the abandoned worker may still append steps.
            snapshot = Trace(steps=list(trace.steps))
            return _finish(chosen, snapshot, start, exc=_timeout_error(timeout))
        except Exception as exc:
            return _finish(chosen, trace, start, exc=exc)
        finally:
            # Python cannot kill a thread; an abandoned run finishes in the background.
            pool.shutdown(wait=False, cancel_futures=True)
        return _finish(chosen, trace, start, raw)


async def arun(
    agent: Any,
    *args: Any,
    adapter: BaseAdapter | str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> AgentResult:
    """Async twin of :func:`run`, for async tests and async agents.

    Framework agents run through their native async API where they have one
    (``astream``, ``Runner.run``, ``kickoff_async``); a synchronous agent is
    run in a worker thread so it does not block the loop. On timeout a native
    async run is cancelled; a thread cannot be, and finishes in the background.
    """
    chosen, timeout = _prepare(agent, adapter, timeout)
    with tracing() as trace:
        start = time.perf_counter()

        async def call() -> Any:
            with chosen.instrument():
                return await chosen.ainvoke(agent, *args, **kwargs)

        try:
            raw = await asyncio.wait_for(call(), timeout=timeout)
        except asyncio.TimeoutError:
            return _finish(chosen, trace, start, exc=_timeout_error(timeout or 0))
        except Exception as exc:
            return _finish(chosen, trace, start, exc=exc)
        return _finish(chosen, trace, start, raw)


#: Backwards-compatible name from v0.1.
run_agent_test = run
