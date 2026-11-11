"""Tests for scripted mocks, tool-call checks, cost, budgets, redaction and the disk cache."""

import json
from pathlib import Path

import pytest

import agenttest
from agenttest import (
    REAL,
    AgentTestBudgetError,
    MockedToolError,
    agent_test,
    assert_cost_under,
    assert_no_tool_errors,
    assert_tool_call_count,
    assert_tool_called_with,
    assert_tools_in_order,
    mock_tool,
    mock_tool_failure,
    mock_tool_sequence,
)
from agenttest.assertions import semantic as sem
from agenttest.assertions.structural import tool_called_with, tools_in_order
from agenttest.core.result import AgentResult, _clear_judge_cache
from agenttest.core.trace import Step, Trace, record_step
from agenttest.mocks._registry import tool_args
from agenttest.pricing import price_for, trace_cost
from agenttest.redaction import REDACTED, redact, redact_value


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_judge_cache()
    yield
    _clear_judge_cache()


@agenttest.tool
def search(query: str, limit: int = 5) -> str:
    return f"real:{query}"


@agenttest.tool
async def asearch(query: str) -> str:
    return f"real:{query}"


# --- scripted mocks ---------------------------------------------------------------

def _attempts(n):
    outcomes = []
    for i in range(n):
        try:
            outcomes.append(search(f"q{i}"))
        except Exception as exc:
            outcomes.append(type(exc).__name__)
    return outcomes


def test_failure_times_fails_only_the_first_calls():
    with mock_tool_failure("search", "down", times=2) as handle:
        assert _attempts(4) == ["MockedToolError", "MockedToolError", "real:q2", "real:q3"]
    assert handle.calls == 2


def test_failure_times_validation():
    with pytest.raises(ValueError):
        with mock_tool_failure("search", times=-1):
            pass


def test_sequence_scripts_each_call_then_runs_the_real_tool():
    with mock_tool_sequence("search", [TimeoutError("slow"), "canned", REAL, ValueError]) as handle:
        assert _attempts(5) == ["TimeoutError", "canned", "real:q2", "ValueError", "real:q4"]
    assert handle.calls == 5


def test_sequence_then_controls_later_calls():
    with mock_tool_sequence("search", ["once"], then=MockedToolError("exhausted")):
        assert _attempts(2) == ["once", "MockedToolError"]


def test_handler_answers_from_the_arguments():
    def handler(args):
        if args["query"] == "boom":
            raise ConnectionError("no route")
        return REAL if args["query"] == "pass" else args["query"].upper()

    with mock_tool("search", handler) as handle:
        assert search("hello") == "HELLO"
        assert search("pass") == "real:pass"
        with pytest.raises(ConnectionError):
            search("boom")
    assert handle.calls == 3


def test_handler_must_be_callable():
    with pytest.raises(TypeError):
        with mock_tool("search", "nope"):
            pass


def test_async_tool_sequence():
    import asyncio

    async def go():
        out = [await asearch("a")]
        try:
            await asearch("b")
        except TimeoutError:
            out.append("timeout")
        out.append(await asearch("c"))
        return out

    with mock_tool_sequence("asearch", ["canned", TimeoutError(), REAL]):
        assert asyncio.run(go()) == ["canned", "timeout", "real:c"]


def test_sequence_failure_is_recorded_as_a_failed_step():
    def agent():
        try:
            search("a")
        except TimeoutError:
            return search("retry")

    with mock_tool_sequence("search", [TimeoutError("slow"), "ok"]):
        result = agenttest.run(agent)
    assert result.output == "ok"
    assert [s.failed for s in result.trace.tool_calls] == [True, False]
    assert result.gracefully_degraded


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"city": "Paris"}, {"city": "Paris"}),
        ('{"city": "Paris"}', {"city": "Paris"}),
        ({"name": "t", "args": {"city": "Paris"}, "id": "1", "type": "tool_call"}, {"city": "Paris"}),
        ("plain", {"input": "plain"}),
        ("[1]", {"input": [1]}),
        (None, {}),
        (["a"], {"input": ["a"]}),
    ],
)
def test_tool_args_normalises_every_framework_shape(raw, expected):
    assert tool_args(raw) == expected


# --- tool-call checks --------------------------------------------------------------

def _traced():
    def agent():
        search("alpha", limit=2)
        search("beta")
        try:
            with mock_tool_failure("asearch_sync"):
                raise_tool()
        except Exception:
            pass
        return "done"

    return agenttest.run(agent)


@agenttest.tool
def raise_tool():
    raise RuntimeError("broken")


def test_called_with_matches_a_subset_of_arguments():
    result = _traced()
    assert assert_tool_called_with(result, "search", query="alpha") is True
    assert assert_tool_called_with(result, "search", query="alpha", limit=2) is True
    assert tool_called_with(result, "search", query="gamma") is False


def test_called_with_failure_shows_the_actual_arguments():
    with pytest.raises(AssertionError, match="its calls had:.*alpha"):
        assert_tool_called_with(_traced(), "search", query="gamma")
    with pytest.raises(AssertionError, match="to be called"):
        assert_tool_called_with(_traced(), "missing", a=1)


def test_called_with_needs_a_result():
    with pytest.raises(TypeError, match="needs an AgentResult"):
        assert_tool_called_with(["search"], "search", q=1)


def test_call_count():
    result = _traced()
    assert_tool_call_count(result, "search", 2)
    with pytest.raises(AssertionError, match="Expected 3 call"):
        assert_tool_call_count(result, "search", 3)


def test_tools_in_order_allows_gaps_but_not_reordering():
    result = _traced()
    assert_tools_in_order(result, ["search", "raise_tool"])
    assert tools_in_order(["a", "b", "c"], ["a", "c"]) is True
    assert tools_in_order(["a", "b"], ["b", "a"]) is False
    with pytest.raises(AssertionError, match="actual order"):
        assert_tools_in_order(result, ["raise_tool", "search"])


def test_no_tool_errors():
    with pytest.raises(AssertionError, match="raise_tool: RuntimeError: broken"):
        assert_no_tool_errors(_traced())
    assert assert_no_tool_errors(agenttest.run(lambda: search("x"))) is True


# --- cost and budgets ---------------------------------------------------------------

def _trace(model="claude-haiku-5-5", input_tokens=1000, output_tokens=2000):
    return Trace(steps=[Step(kind="llm", name="m", model=model, input_tokens=input_tokens, output_tokens=output_tokens)])


def test_cost_from_the_builtin_table():
    assert trace_cost(_trace()) == pytest.approx((1000 * 0.1 + 2000 * 0.5) / 1e6)


def test_dated_model_ids_use_the_base_price():
    assert price_for("claude-haiku-4-5-20251001") == price_for("claude-haiku-4-5")


def test_unknown_model_is_none_not_zero():
    assert trace_cost(_trace("mystery")) is None
    assert trace_cost(Trace()) == 0.0


def test_custom_pricing_from_config():
    Path("agenttest.config.json").write_text(json.dumps({"pricing": {"mystery": {"input": 1, "output": 2}}}))
    assert trace_cost(_trace("mystery")) == pytest.approx((1000 + 4000) / 1e6)


def _spending_agent(tokens=100, model="claude-haiku-5-5"):
    def agent():
        record_step("llm", "m", model=model, input_tokens=tokens, output_tokens=0)
        return "ok"

    return agent


def test_result_reports_cost_and_assert_cost_under():
    result = agenttest.run(_spending_agent(1_000_000))
    assert result.cost_usd == pytest.approx(0.1)
    assert_cost_under(result, 0.2)
    with pytest.raises(AssertionError, match="cost \\$0.1"):
        assert_cost_under(result, 0.05)


def test_assert_cost_under_with_unknown_price_explains():
    with pytest.raises(AssertionError, match="no price"):
        assert_cost_under(agenttest.run(_spending_agent(model="mystery")), 1.0)


def test_token_budget_fails_the_test_and_is_not_retried():
    calls = []

    @agent_test(max_tokens=50, max_retries=3, runs=2, min_pass_rate=0.5)
    def body():
        calls.append(1)
        agenttest.run(_spending_agent(100))

    with pytest.raises(AgentTestBudgetError, match="used 100 tokens; the budget is 50"):
        body()
    assert calls == [1]


def test_budget_within_limits_passes_and_accumulates_across_runs():
    @agent_test(max_tokens=350, runs=3)
    def body():
        agenttest.run(_spending_agent(100))

    body()


def test_cost_budget():
    @agent_test(max_cost_usd=0.01)
    def body():
        agenttest.run(_spending_agent(1_000_000))

    with pytest.raises(AgentTestBudgetError, match="the budget is \\$0.01"):
        body()


def test_cost_budget_with_unknown_price_fails_loudly():
    @agent_test(max_cost_usd=1.0)
    def body():
        agenttest.run(_spending_agent(model="mystery"))

    with pytest.raises(AgentTestBudgetError, match="no known price"):
        body()


def test_budget_errors_are_assertion_errors():
    assert issubclass(AgentTestBudgetError, AssertionError)
    with pytest.raises(ValueError):
        agent_test(max_tokens=0)
    with pytest.raises(ValueError):
        agent_test(max_cost_usd=-1)


# --- redaction ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-abcdefghijklmnop1234",
        "sk-proj-abcdefghijklmnopqrstuv",
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_abcdefghijklmnopqrstuvwxyz1234",
        "Bearer abcdefghijklmnop1234567890",
    ],
)
def test_common_secret_formats_are_hidden(secret):
    assert redact(f"before {secret} after") == f"before {REDACTED} after"


def test_values_of_secret_environment_variables_are_hidden(monkeypatch):
    monkeypatch.setenv("MY_SERVICE_TOKEN", "plainlookingvalue99")
    monkeypatch.setenv("NORMAL_SETTING", "ordinaryvalue123456")
    text = redact("a plainlookingvalue99 b ordinaryvalue123456")
    assert "plainlookingvalue99" not in text and "ordinaryvalue123456" in text


def test_configured_patterns_are_hidden():
    Path("agenttest.config.json").write_text(json.dumps({"redact_patterns": [r"EMP-\d{4}"]}))
    assert redact("id EMP-1234") == f"id {REDACTED}"


def test_redact_value_is_recursive_and_leaves_other_types():
    secret = "sk-abcdefghijklmnopqrstuv"
    assert redact_value({"a": [secret, 1], "b": {"c": secret}, "d": None}) == {
        "a": [REDACTED, 1], "b": {"c": REDACTED}, "d": None,
    }


def test_reports_and_failure_traces_hide_secrets():
    from agenttest import reporting

    secret = "sk-abcdefghijklmnopqrstuvwxyz"
    trace = Trace(steps=[Step(kind="tool", name="t", input={"key": secret}, output=secret)])
    result = AgentResult.from_trace(trace, output=f"leaked {secret}", latency_ms=1.0)

    assert secret not in json.dumps(reporting.result_to_dict(result))
    assert secret not in reporting.describe_result(result)


# --- persistent judge cache --------------------------------------------------------------

def _config(**extra):
    Path("agenttest.config.json").write_text(json.dumps({"judge_provider": "anthropic", **extra}))


def _counting_provider(monkeypatch, reply="YES"):
    calls = []
    monkeypatch.setattr("agenttest.core.result.call_judge", lambda *a, **k: calls.append(1) or reply)
    return calls


def test_disk_cache_survives_a_new_session(monkeypatch):
    _config(judge_cache=True)
    calls = _counting_provider(monkeypatch)
    assert sem.contains_fact("x", "y") is True
    _clear_judge_cache()  # a new session starts with an empty memory cache
    assert sem.contains_fact("x", "y") is True
    assert len(calls) == 1
    assert Path(".agenttest_cache/.gitignore").read_text() == "*\n"


def test_disk_cache_is_off_by_default(monkeypatch):
    _config()
    calls = _counting_provider(monkeypatch)
    sem.contains_fact("x", "y")
    _clear_judge_cache()
    sem.contains_fact("x", "y")
    assert len(calls) == 2
    assert not Path(".agenttest_cache").exists()


def test_malformed_replies_never_reach_the_disk_cache(monkeypatch):
    _config(judge_cache=True)
    calls = _counting_provider(monkeypatch, "garbage")
    for _ in range(2):
        with pytest.raises(agenttest.AgentTestJudgeError):
            sem.contains_fact("x", "y")
        _clear_judge_cache()
    assert len(calls) == 4
