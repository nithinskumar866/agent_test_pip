"""Tests for retries and pass-rate evaluation (@agent_test runs/max_retries/min_pass_rate)."""

import asyncio
import json
import warnings

import pytest

from agenttest import AgentTestConfigError, AgentTestFlakyWarning, AgentTestPassRateError, agent_test
from agenttest.core.retry import FLAKY_REPORTS, RetryPolicy


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FLAKY_REPORTS.clear()
    yield tmp_path
    FLAKY_REPORTS.clear()


def scripted(outcomes):
    """A test body that passes/fails following ``outcomes`` (True = pass)."""
    calls = {"n": 0}
    plan = iter(outcomes)

    def body():
        calls["n"] += 1
        if not next(plan):
            raise AssertionError(f"failure #{calls['n']}")
        return "ok"

    return body, calls


def test_default_runs_once_and_preserves_original_error():
    body, calls = scripted([False])
    with pytest.raises(AssertionError, match="failure #1") as exc_info:
        agent_test(body)()
    assert calls["n"] == 1
    assert not getattr(exc_info.value, "__notes__", None)


def test_retry_until_pass_is_flaky():
    body, calls = scripted([False, False, True])
    with pytest.warns(AgentTestFlakyWarning, match="2 retries used"):
        assert agent_test(max_retries=2)(body)() == "ok"
    assert calls["n"] == 3
    assert len(FLAKY_REPORTS) == 1


def test_retries_exhausted_reraises_last_error_with_note():
    body, calls = scripted([False, False, False])
    with pytest.raises(AssertionError, match="failure #3") as exc_info:
        agent_test(max_retries=2)(body)()
    assert calls["n"] == 3
    if hasattr(exc_info.value, "add_note"):  # exception notes need Python 3.11+
        assert "agenttest: failed on all 3 attempts" in exc_info.value.__notes__


def test_pass_rate_met_with_some_failures_warns():
    body, calls = scripted([True, False, True, True, True])
    with pytest.warns(AgentTestFlakyWarning, match="4/5 runs passed"):
        agent_test(runs=5, min_pass_rate=0.8)(body)()
    assert calls["n"] == 5


def test_pass_rate_stops_early_once_met():
    body, calls = scripted([True, True, True, True, True])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent_test(runs=5, min_pass_rate=0.6)(body)()
    assert calls["n"] == 3  # 3 of 5 needed; runs 4-5 cannot change the verdict


def test_pass_rate_stops_early_once_impossible():
    body, calls = scripted([False, False, True, True, True])
    with pytest.raises(AgentTestPassRateError) as exc_info:
        agent_test(runs=5, min_pass_rate=0.8)(body)()
    assert calls["n"] == 2
    message = str(exc_info.value)
    assert "0/2 runs passed (0%); required 4/5" in message
    assert "stopped early after 2 runs" in message
    assert "run 1: AssertionError: failure #1" in message
    assert isinstance(exc_info.value, AssertionError)
    assert isinstance(exc_info.value.__cause__, AssertionError)


def test_retries_apply_within_each_run():
    # run 1: fail then pass (retry), run 2: pass, run 3: fail twice
    body, calls = scripted([False, True, True, False, False])
    with pytest.raises(AgentTestPassRateError, match="2/3 runs passed"):
        agent_test(runs=3, max_retries=1, min_pass_rate=1.0)(body)()
    assert calls["n"] == 5


def test_all_runs_pass_is_not_flaky():
    body, calls = scripted([True] * 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        agent_test(runs=3)(body)()
    assert calls["n"] == 3
    assert FLAKY_REPORTS == []


def test_config_errors_are_not_retried():
    calls = {"n": 0}

    def body():
        calls["n"] += 1
        raise AgentTestConfigError("bad setup")

    with pytest.raises(AgentTestConfigError):
        agent_test(max_retries=3)(body)()
    assert calls["n"] == 1


def test_pytest_outcomes_are_not_retried():
    calls = {"n": 0}

    def body():
        calls["n"] += 1
        pytest.skip("not today")

    with pytest.raises(pytest.skip.Exception):
        agent_test(max_retries=3)(body)()
    assert calls["n"] == 1


def test_any_exception_counts_as_a_failed_attempt():
    outcomes = iter([ValueError("transient"), None])

    def body():
        exc = next(outcomes)
        if exc:
            raise exc

    with pytest.warns(AgentTestFlakyWarning):
        agent_test(max_retries=1)(body)()


def test_config_file_supplies_defaults(isolated):
    (isolated / "agenttest.config.json").write_text(
        json.dumps({"runs": 4, "min_pass_rate": 0.5, "max_retries": 0}), encoding="utf-8"
    )
    body, calls = scripted([False, True, False, True])
    with pytest.warns(AgentTestFlakyWarning, match="2/4 runs passed"):
        agent_test(body)()
    assert calls["n"] == 4


def test_decorator_overrides_config(isolated):
    (isolated / "agenttest.config.json").write_text(json.dumps({"runs": 10}), encoding="utf-8")
    body, calls = scripted([True])
    agent_test(runs=1)(body)()
    assert calls["n"] == 1


def test_invalid_config_values_rejected(isolated):
    (isolated / "agenttest.config.json").write_text(json.dumps({"min_pass_rate": 0}), encoding="utf-8")
    with pytest.raises(AgentTestConfigError, match="min_pass_rate"):
        agent_test(lambda: None)()


def test_async_test_retries():
    attempts = {"n": 0}

    @agent_test(runs=3, min_pass_rate=0.6)
    async def body():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise AssertionError("first run fails")

    with pytest.warns(AgentTestFlakyWarning):
        asyncio.run(body())
    assert attempts["n"] == 3


def test_test_arguments_are_passed_to_every_attempt():
    seen = []

    @agent_test(runs=2)
    def body(value, *, flag):
        seen.append((value, flag))

    body(1, flag=True)
    assert seen == [(1, True), (1, True)]


@pytest.mark.parametrize(
    ("runs", "rate", "required"),
    [(1, 1.0, 1), (5, 0.8, 4), (5, 0.6, 3), (10, 0.95, 10), (3, 0.01, 1), (10, 0.7, 7)],
)
def test_required_passes(runs, rate, required):
    assert RetryPolicy(runs=runs, min_pass_rate=rate).required_passes == required
