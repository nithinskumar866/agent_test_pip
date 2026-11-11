"""Tests for xdist support, session budgets, markdown, regression comparison and datasets."""

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

import agenttest
from agenttest import AgentTestConfigError, Case, load_cases, reporting
from agenttest.cli.init import main
from tests._plugin_args import plugin_args

pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def importable_agenttest(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))


SUITE = """
import agenttest
from agenttest import agent_test
from agenttest.core.trace import record_step

calls = []


def agent():
    record_step("llm", "m", model="claude-haiku-5-5", input_tokens=1000, output_tokens=0)
    return "ok"


@agent_test
def test_a():
    assert agenttest.run(agent).success


@agent_test
def test_b():
    assert agenttest.run(agent).success


@agent_test
def test_c():
    assert agenttest.run(agent).output == "nope"


@agent_test
def test_d():
    assert agenttest.run(agent).success


@agent_test(runs=2, min_pass_rate=0.5)
def test_flaky():
    calls.append(1)
    assert len(calls) != 1
"""


def session(pytester, source, *args):
    pytester.makepyfile(test_suite=source)
    return pytester.runpytest_subprocess(*plugin_args(), "-p", "no:cacheprovider", *args)


def test_xdist_runs_reach_the_controller(pytester):
    pytest.importorskip("xdist")
    result = session(pytester, SUITE, "-n", "2", "--agenttest-report=r.json", "--agenttest-html=r.html")

    result.assert_outcomes(passed=4, failed=1)
    result.stdout.fnmatch_lines(["Agent tests: 4 passed, 1 failed, 0 skipped", "Agent runs: * 4000 tokens*", "*Report written: r.json"])
    report = json.loads((pytester.path / "r.json").read_text(encoding="utf-8"))
    assert report["summary"]["agent_runs"] == 4
    assert report["summary"]["tokens_used"] == 4000
    assert report["summary"]["flaky"] == 1
    assert (pytester.path / "r.html").exists()


XFAIL_SUITE = """
import pytest
import agenttest
from agenttest import agent_test


@pytest.mark.xfail(strict=True, reason="FINDING: the agent crashes on empty input")
@agent_test
def test_known_problem():
    assert agenttest.run(lambda: "").output == "answer"


@pytest.mark.xfail(reason="sometimes fails")
@agent_test
def test_fixed_problem():
    assert agenttest.run(lambda: "answer").output == "answer"


@pytest.mark.skip(reason="not today")
@agent_test
def test_skipped():
    pass
"""


def test_expected_failures_are_not_reported_as_skipped(pytester):
    """pytest reports a failed xfail test as "skipped"; a recorded finding must not read as "not run"."""
    result = session(pytester, XFAIL_SUITE, "--agenttest-report=r.json", "--agenttest-html=r.html")

    result.stdout.fnmatch_lines(["Agent tests: 0 passed, 0 failed, 1 skipped, 1 xfailed, 1 xpassed"])
    report = json.loads((pytester.path / "r.json").read_text(encoding="utf-8"))
    assert {k: report["summary"][k] for k in ("skipped", "xfailed", "xpassed")} == {
        "skipped": 1, "xfailed": 1, "xpassed": 1}
    outcomes = {t["nodeid"].split("::")[1]: (t["outcome"], t["message"]) for t in report["tests"]}
    assert outcomes["test_known_problem"] == ("xfailed", "FINDING: the agent crashes on empty input")
    assert outcomes["test_fixed_problem"][0] == "xpassed"
    html = (pytester.path / "r.html").read_text(encoding="utf-8")
    assert '<td class="xfailed">xfailed</td>' in html and "expected failures" in html


def test_xdist_failing_test_still_shows_the_trace(pytester):
    pytest.importorskip("xdist")
    result = session(pytester, SUITE, "-n", "2")
    result.stdout.fnmatch_lines(["*agenttest runs*", "run 1 of 1: ok*"])


def test_session_token_budget_stops_the_run(pytester):
    result = session(pytester, SUITE, "--agenttest-max-tokens=1500")
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*used 2000 tokens; --agenttest-max-tokens is 1500*"])
    assert result.parseoutcomes().get("passed", 0) == 2


def test_session_cost_budget_stops_the_run(pytester):
    result = session(pytester, SUITE, "--agenttest-max-cost=0.00015")
    result.stdout.fnmatch_lines(["*--agenttest-max-cost is $0.00015*"])
    assert result.parseoutcomes().get("passed", 0) == 2


def test_record_option_sets_the_mode(pytester):
    source = """
import os
def test_mode():
    assert os.environ["AGENTTEST_RECORD"] == "none"
"""
    session(pytester, source, "--agenttest-record=none").assert_outcomes(passed=1)


def test_summary_shows_cost(pytester):
    session(pytester, SUITE, "-k", "test_a").stdout.fnmatch_lines(["Agent runs: 1 (0 failed)*$0.0001"])


# --- markdown and comparison ------------------------------------------------------

def _run(tokens=10, cost=0.5):
    return {"success": True, "output": "o", "error": None, "latency_ms": 1.0, "steps_taken": 1, "tools_called": [],
            "failed_tools": [], "tokens_used": tokens, "gracefully_degraded": False, "steps": [], "cost_usd": cost,
            "tool_calls": 0}


def _report(outcomes, tokens=10, cost=0.5):
    tests = [reporting.TestRecord(name, outcome=outcome, agent_marked=True,
                                  message="assert 1 | 2 @someone <b>" if outcome == "failed" else None,
                                  runs=[_run(tokens, cost)])
             for name, outcome in outcomes.items()]
    return reporting.build_report(tests, [], duration_s=1.0)


def test_markdown_summary_is_safe_for_pull_requests():
    text = reporting.render_markdown(_report({"t::a": "passed", "t::b": "failed"}))
    assert "agenttest: 1 passed, 1 failed" in text and "<!-- agenttest-report -->" in text
    assert "`t::b`" in text
    assert "assert 1 \\| 2" in text  # can't break the table
    assert "@someone" not in text  # can't notify anyone
    assert "<b>" not in text


def test_compare_finds_regressions_and_fixes():
    old = _report({"a": "passed", "b": "failed", "c": "passed", "gone": "passed"})
    new = _report({"a": "failed", "b": "passed", "c": "passed", "fresh": "passed"})
    result = reporting.compare_reports(old, new)
    assert result["newly_failing"] == ["a"] and result["newly_passing"] == ["b"]
    assert result["added"] == ["fresh"] and result["removed"] == ["gone"]
    assert result["regressions"] == ["1 test(s) newly failing"]


def test_compare_thresholds():
    old, new = _report({"a": "passed"}, 100, 1.0), _report({"a": "passed"}, 150, 3.0)
    assert reporting.compare_reports(old, new)["regressions"] == []
    assert reporting.compare_reports(old, new, max_token_increase=60)["regressions"] == []
    reasons = reporting.compare_reports(old, new, max_token_increase=20, max_cost_increase=100)["regressions"]
    assert reasons == ["tokens grew 50.0% (allowed 20%)", "cost grew 200.0% (allowed 100%)"]


def test_compare_handles_unknown_cost_and_zero_baseline():
    result = reporting.compare_reports(_report({"a": "passed"}, 0, None), _report({"a": "passed"}, 5, 0.1))
    assert result["tokens"]["change_pct"] is None and result["cost_usd"]["change_pct"] is None
    assert "n/a" in reporting.render_comparison(result)


def test_compare_command_exit_codes(tmp_path):
    a, b, md = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "out.md"
    a.write_text(json.dumps(_report({"x": "passed"})))
    b.write_text(json.dumps(_report({"x": "failed"})))

    result = CliRunner().invoke(main, ["compare", str(a), str(b), "--markdown", str(md)])
    assert result.exit_code == 1 and "Regressions" in result.output
    assert "Compared with baseline" in md.read_text(encoding="utf-8")

    assert CliRunner().invoke(main, ["compare", str(a), str(a)]).exit_code == 0


def test_compare_rejects_other_json(tmp_path):
    a = tmp_path / "a.json"
    a.write_text("{}")
    assert "is not an agenttest report" in CliRunner().invoke(main, ["compare", str(a), str(a)]).output


def test_report_command_writes_markdown(tmp_path):
    path, md = tmp_path / "r.json", tmp_path / "r.md"
    path.write_text(json.dumps(_report({"x": "passed"})))
    assert CliRunner().invoke(main, ["report", str(path), "--markdown", str(md)]).exit_code == 0
    assert "1 passed" in md.read_text(encoding="utf-8")


# --- datasets -----------------------------------------------------------------------

@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("c.jsonl", '{"id": "a", "q": "x"}\n\n{"id": "b", "q": "y"}\n'),
        ("c.json", '[{"id": "a", "q": "x"}, {"id": "b", "q": "y"}]'),
        ("c.csv", "id,q\na,x\nb,y\n"),
    ],
)
def test_load_cases_formats(tmp_path, name, content):
    (tmp_path / name).write_text(content)
    rows = load_cases(tmp_path / name)
    assert [r.q for r in rows] == ["x", "y"] and rows[0]["id"] == "a"


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("c.txt", "x", "unsupported dataset format"),
        ("c.json", "{}", "expected a JSON list"),
        ("c.json", "not json", "invalid JSON"),
        ("c.jsonl", '{"a": 1}\nbad\n', ":2: invalid JSON"),
        ("c.json", "[1]", "not an object"),
    ],
)
def test_load_cases_reports_bad_files(tmp_path, name, content, message):
    (tmp_path / name).write_text(content)
    with pytest.raises(AgentTestConfigError, match=message):
        load_cases(tmp_path / name)


def test_load_cases_missing_file(tmp_path):
    with pytest.raises(AgentTestConfigError, match="Dataset not found"):
        load_cases(tmp_path / "none.json")


def test_case_attribute_error_lists_fields():
    with pytest.raises(AttributeError, match="has: q"):
        _ = Case(q=1).missing


def test_cases_decorator_makes_one_named_test_per_row(pytester):
    (pytester.path / "data.jsonl").write_text('{"id": "first", "n": 1}\n{"id": "second", "n": 2}\n{"n": 3}\n')
    source = """
import agenttest
from agenttest import agent_test

@agenttest.cases("data.jsonl")
@agent_test
def test_row(case):
    assert case.n < 3
"""
    result = session(pytester, source, "-v")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["*test_row?first?*PASSED*", "*test_row?case3?*FAILED*"])
    assert agenttest.cases([{"id": "x"}]) is not None
