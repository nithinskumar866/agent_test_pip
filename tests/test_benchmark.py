"""Tests for the judge benchmark, using scripted judges that are right, wrong or fooled."""

import json
import re

import pytest
from click.testing import CliRunner

import agenttest
from agenttest.cli.init import main
from agenttest.judge.benchmark import KINDS, build_cases, run_benchmark


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


CASES = {case.output: case for case in build_cases()}


def _output_in(prompt):
    return re.search(r"<<<OUTPUT-(\w+)\n(.*)\nOUTPUT-\1>>>", prompt, re.DOTALL).group(2)


def _reply(claims, injection=False):
    return json.dumps({"claims": claims, "injection_detected": injection})


def perfect(prompt, system):
    case = CASES.get(_output_in(prompt))
    if case is None:  # a second look at one claim: the oracle knows only invented claims get one
        return _reply([{"claim": "x", "status": "unsupported"}])
    quote = case.context[0]
    good = {"claim": "c", "status": "supported", "evidence": [{"passage": 1, "quote": quote}]}
    if "<<<PASSAGE2-" in prompt:  # "grounded" and "disputed" share an answer; only the context differs
        return _reply([{"claim": "c", "status": "disputed", "evidence": [
            {"passage": 1, "quote": case.context[0]}, {"passage": 2, "quote": case.context[1]}]}])
    if case.kind in ("grounded", "disputed"):
        return _reply([good])
    if case.kind == "hallucinated":
        return _reply([good, {"claim": "x", "status": "unsupported"}])
    return _reply([{"claim": "c", "status": "contradicted", "evidence": [{"passage": 1, "quote": quote}]}],
                  injection=case.kind == "injection")


def approves_everything_without_evidence(prompt, system):
    return _reply([{"claim": "c", "status": "supported", "evidence": [{"passage": 1, "quote": "not in the context"}]}])


def approves_everything_quoting_the_context(prompt, system):
    case = CASES[_output_in(prompt)]
    return _reply([{"claim": "c", "status": "supported", "evidence": [{"passage": 1, "quote": case.context[0]}]}])


def test_dataset_is_100_balanced_unique_cases():
    cases = build_cases()
    assert len(cases) == 100
    assert {k: sum(c.kind == k for c in cases) for k in KINDS} == dict.fromkeys(KINDS, 20)
    assert len({c.id for c in cases}) == 100
    assert sum(c.should_pass for c in cases) == 20


def test_a_perfect_judge_scores_100_percent():
    with agenttest.use_judge(perfect):
        result = run_benchmark()
    assert result["accuracy"] == 1.0
    assert result["false_reject_rate"] == 0.0 and result["false_accept_rate"] == 0.0
    assert result["mistakes"] == []


def test_evidence_verification_catches_a_judge_that_invents_quotes():
    with agenttest.use_judge(approves_everything_without_evidence):
        result = run_benchmark(limit=2)
    assert result["per_kind"]["grounded"] == 0.0  # its "approval" had no real evidence
    assert result["false_accept_rate"] == 0.0


def test_a_judge_fooled_with_real_quotes_is_measured():
    with agenttest.use_judge(approves_everything_quoting_the_context):
        result = run_benchmark(limit=2)
    assert result["per_kind"]["grounded"] == 1.0
    assert result["per_kind"]["contradicted"] == 0.0 and result["false_accept_rate"] == 1.0
    assert result["accuracy"] == pytest.approx(0.2)
    assert len(result["mistakes"]) == 8


def test_judge_errors_are_counted_not_scored():
    with agenttest.use_judge(lambda p, s: "not json"):
        result = run_benchmark(limit=1)
    assert result["answered"] == 0 and len(result["judge_errors"]) == 5 and result["accuracy"] == 0.0


def test_progress_callback():
    seen = []
    with agenttest.use_judge(perfect):
        run_benchmark(limit=1, on_progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (5, 5)


def test_benchmark_command_reports_and_gates(tmp_path):
    out = tmp_path / "b.json"
    with agenttest.use_judge(perfect):
        result = CliRunner().invoke(main, ["benchmark", "--limit", "1", "--json", str(out), "--min-accuracy", "0.9"])
    assert result.exit_code == 0, result.output
    assert "Accuracy: 100%" in result.output
    assert json.loads(out.read_text())["accuracy"] == 1.0

    with agenttest.use_judge(approves_everything_quoting_the_context):
        result = CliRunner().invoke(main, ["benchmark", "--limit", "1", "--min-accuracy", "0.9"])
    assert result.exit_code == 1 and "Below the required accuracy" in result.output


def test_benchmark_command_with_the_judge_disabled():
    from pathlib import Path

    Path("agenttest.config.json").write_text('{"judge_provider": "none"}')
    result = CliRunner().invoke(main, ["benchmark", "--limit", "1"])
    assert result.exit_code != 0 and "disabled" in result.output


def test_benchmark_stops_at_the_first_rejected_key(monkeypatch, tmp_path):
    anthropic = pytest.importorskip("anthropic")
    http = pytest.importorskip("httpx2")
    from agenttest.judge import providers

    calls = []

    def handler(request):
        calls.append(1)
        return http.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad"}})

    (tmp_path / "agenttest.config.json").write_text('{"judge_provider": "anthropic"}')
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(providers, "_anthropic_client", lambda api_key, timeout: anthropic.Anthropic(
        api_key=api_key, max_retries=0, http_client=http.Client(transport=http.MockTransport(handler))))

    result = CliRunner().invoke(main, ["benchmark", "--limit", "4"])
    assert result.exit_code != 0 and "Stopped the benchmark" in result.output
    assert len(calls) == 1
