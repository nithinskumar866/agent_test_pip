"""Retries and pass-rate evaluation for non-deterministic agent tests.

An agent test can be executed several times and judged statistically:

* ``runs`` — how many independent trials of the test body to execute;
* ``max_retries`` — extra attempts a failing trial gets before it counts as
  failed (a trial passes if any of its ``1 + max_retries`` attempts passes);
* ``min_pass_rate`` — the fraction of trials that must pass.

Execution stops as soon as the verdict is certain, so a hopeless or an
already-passing test does not burn more agent calls than needed.

Only ordinary exceptions are retried. Configuration errors are raised
immediately, and pytest outcomes (``pytest.skip``, ``pytest.fail``,
``pytest.xfail``) propagate untouched — they are not ``Exception`` subclasses.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agenttest.config import AgentTestConfigError, AgentTestFlakyWarning, AgentTestPassRateError

__all__ = ["RetryPolicy", "TrialOutcome", "PassRateReport", "execute", "aexecute", "FLAKY_REPORTS"]

#: Exceptions that indicate misuse, not flakiness: never retried.
_NON_RETRYABLE: tuple[type[BaseException], ...] = (AgentTestConfigError,)


@dataclass(frozen=True)
class RetryPolicy:
    runs: int = 1
    max_retries: int = 0
    min_pass_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError("runs must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be 0 or more")
        if not 0.0 < self.min_pass_rate <= 1.0:
            raise ValueError("min_pass_rate must be in (0, 1]")

    @property
    def required_passes(self) -> int:
        # Tolerance keeps 0.8 * 5 at 4 despite floating-point error.
        return max(1, math.ceil(self.runs * self.min_pass_rate - 1e-9))

    @property
    def is_single_attempt(self) -> bool:
        return self.runs == 1 and self.max_retries == 0


@dataclass
class TrialOutcome:
    passed: bool
    attempts: int
    errors: list[BaseException] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class PassRateReport:
    test_name: str
    policy: RetryPolicy
    trials: list[TrialOutcome] = field(default_factory=list)

    @property
    def passes(self) -> int:
        return sum(t.passed for t in self.trials)

    @property
    def retries_used(self) -> int:
        return sum(t.attempts - 1 for t in self.trials)

    @property
    def last_error(self) -> BaseException | None:
        for trial in reversed(self.trials):
            if trial.errors:
                return trial.errors[-1]
        return None

    @property
    def passed(self) -> bool:
        return self.passes >= self.policy.required_passes

    @property
    def flaky(self) -> bool:
        return self.passed and (self.retries_used > 0 or self.passes < len(self.trials))

    def decided(self) -> bool:
        """Whether the remaining trials can no longer change the verdict."""
        remaining = self.policy.runs - len(self.trials)
        return self.passed or self.passes + remaining < self.policy.required_passes

    def summary(self) -> str:
        policy = self.policy
        rate = self.passes / len(self.trials) if self.trials else 0.0
        text = (
            f"{self.passes}/{len(self.trials)} runs passed ({rate:.0%}); "
            f"required {policy.required_passes}/{policy.runs} (min_pass_rate={policy.min_pass_rate:g})"
        )
        if len(self.trials) < policy.runs:
            text += f"; stopped early after {len(self.trials)} run{'s' if len(self.trials) != 1 else ''}"
        if self.retries_used:
            text += f"; {self.retries_used} retr{'y' if self.retries_used == 1 else 'ies'} used"
        return text

    def failure_details(self) -> str:
        lines = []
        for index, trial in enumerate(self.trials, start=1):
            if trial.passed:
                lines.append(f"  run {index}: passed" + (f" (attempt {trial.attempts})" if trial.attempts > 1 else ""))
                continue
            first_line = str(trial.errors[-1]).strip().splitlines()[0] if trial.errors else "failed"
            lines.append(f"  run {index}: {type(trial.errors[-1]).__name__}: {first_line}")
        return "\n".join(lines)


#: Flaky passes seen this session, for the pytest terminal summary.
FLAKY_REPORTS: list[PassRateReport] = []


def _conclude(report: PassRateReport, result: Any) -> Any:
    if report.passed:
        if report.flaky:
            FLAKY_REPORTS.append(report)
            warnings.warn(AgentTestFlakyWarning(f"{report.test_name}: {report.summary()}"), stacklevel=3)
        return result

    last = report.last_error
    if report.policy.runs == 1 and last is not None:
        # One trial with retries: surface the real failure, annotated.
        note = f"agenttest: failed on all {report.trials[0].attempts} attempts"
        if hasattr(last, "add_note"):
            last.add_note(note)
        raise last
    raise AgentTestPassRateError(
        f"{report.test_name}: {report.summary()}\n{report.failure_details()}"
    ) from last


def execute(test_name: str, policy: RetryPolicy, call: Callable[[], Any]) -> Any:
    """Run ``call`` under ``policy``; return its last passing result or raise."""
    if policy.is_single_attempt:
        return call()
    report = PassRateReport(test_name, policy)
    result: Any = None
    while not report.decided():
        trial = TrialOutcome(passed=False, attempts=0)
        start = time.perf_counter()
        for _ in range(policy.max_retries + 1):
            trial.attempts += 1
            try:
                result = call()
            except _NON_RETRYABLE:
                raise
            except Exception as exc:
                trial.errors.append(exc)
                continue
            trial.passed = True
            break
        trial.duration_ms = (time.perf_counter() - start) * 1000
        report.trials.append(trial)
    return _conclude(report, result)


async def aexecute(test_name: str, policy: RetryPolicy, call: Callable[[], Awaitable[Any]]) -> Any:
    """Async twin of :func:`execute`."""
    if policy.is_single_attempt:
        return await call()
    report = PassRateReport(test_name, policy)
    result: Any = None
    while not report.decided():
        trial = TrialOutcome(passed=False, attempts=0)
        start = time.perf_counter()
        for _ in range(policy.max_retries + 1):
            trial.attempts += 1
            try:
                result = await call()
            except _NON_RETRYABLE:
                raise
            except Exception as exc:
                trial.errors.append(exc)
                continue
            trial.passed = True
            break
        trial.duration_ms = (time.perf_counter() - start) * 1000
        report.trials.append(trial)
    return _conclude(report, result)
