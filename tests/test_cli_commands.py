"""Tests for the CLI beyond `init` detection: --version, --force, --example, doctor, report."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import agenttest
from agenttest.cli.init import main
from tests._plugin_args import plugin_args

REPO = Path(__file__).resolve().parents[1]


def run_pytest(*paths, cwd=None):
    """Run pytest with the agenttest plugin in a separate process.

    A separate process keeps the plugin's per-session state apart from the
    session running this test.
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    return subprocess.run(
        [sys.executable, "-m", "pytest", *map(str, paths), "-q", *plugin_args(), "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=cwd, env=env, timeout=300,
    )


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class _Result:
    """CliRunner result with the table's line wrapping undone."""

    def __init__(self, result):
        self.exit_code = result.exit_code
        self.output = " ".join(result.output.split())


def invoke(*args):
    return _Result(CliRunner().invoke(main, list(args)))


def test_version_option():
    result = invoke("--version")
    assert result.exit_code == 0
    assert result.output.strip() == f"agenttest {agenttest.__version__}"


def test_init_refuses_to_overwrite_without_force():
    Path("agenttest.config.json").write_text('{"framework": "crewai"}', encoding="utf-8")

    result = invoke("init", "--framework", "generic", "--judge", "none")
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert json.loads(Path("agenttest.config.json").read_text())["framework"] == "crewai"

    result = invoke("init", "--framework", "generic", "--judge", "none", "--force")
    assert result.exit_code == 0
    assert json.loads(Path("agenttest.config.json").read_text())["framework"] == "generic"


def test_init_example_is_written_once_and_passes():
    assert invoke("init", "--framework", "generic", "--judge", "none", "--example").exit_code == 0
    example = Path("tests/test_agent_example.py")
    assert example.exists()

    assert run_pytest(example).returncode == 0

    example.write_text(example.read_text() + "\n# edited\n", encoding="utf-8")
    invoke("init", "--framework", "generic", "--judge", "none", "--example", "--force")
    assert example.read_text().endswith("# edited\n")  # an existing example is never overwritten


def test_doctor_reports_a_ready_project():
    Path("agenttest.config.json").write_text(json.dumps({"framework": "generic", "judge_provider": "none"}))
    result = invoke("doctor")
    assert result.exit_code == 0, result.output
    assert "Ready to run agent tests" in result.output
    assert "disabled" in result.output


def test_doctor_without_config_file_warns_but_passes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    result = invoke("doctor")
    assert "no config file; defaults apply" in result.output


def test_doctor_fails_on_missing_judge_key_and_never_prints_secrets(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    Path("agenttest.config.json").write_text(json.dumps({"judge_provider": "anthropic"}))
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY is not set" in result.output

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    result = invoke("doctor")
    assert "sk-secret-value" not in result.output


def test_doctor_fails_on_invalid_config():
    Path("agenttest.config.json").write_text('{"framework": "nope"}')
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "config" in result.output


def test_doctor_flags_a_framework_whose_package_is_missing(monkeypatch):
    from agenttest.cli import doctor

    Path("agenttest.config.json").write_text(json.dumps({"framework": "crewai", "judge_provider": "none"}))
    monkeypatch.setattr(doctor, "_installed", lambda module: module != "crewai")
    result = invoke("doctor")
    assert result.exit_code == 1
    assert "package 'crewai' is not installed" in result.output


def test_doctor_ping_judge_uses_the_judge():
    Path("agenttest.config.json").write_text(json.dumps({"judge_provider": "ollama"}))
    with agenttest.use_judge(lambda prompt, system: "OK"):
        result = invoke("doctor", "--ping-judge")
    assert "judge ping" in result.output and "'OK'" in result.output


def _report():
    run = {"success": True, "output": "hi", "error": None, "latency_ms": 2, "steps_taken": 0,
           "tools_called": ["search"], "failed_tools": [], "tokens_used": 30, "gracefully_degraded": False,
           "steps": []}
    return {
        "agenttest_version": "x", "created": "now", "duration_s": 1.0,
        "summary": {"tests": 2, "passed": 1, "failed": 1, "skipped": 0, "agent_runs": 1, "failed_runs": 0,
                    "tokens_used": 30, "tool_calls": 1, "failed_tool_calls": 0, "agent_latency_ms": 2, "flaky": 0},
        "tests": [
            {"nodeid": "t.py::test_ok", "outcome": "passed", "duration_s": 0.1, "agent_marked": True,
             "message": None, "runs": [run]},
            {"nodeid": "t.py::test_bad", "outcome": "failed", "duration_s": 0.1, "agent_marked": True,
             "message": "boom", "runs": []},
        ],
        "flaky": [],
    }


def test_report_command_summarises_and_renders_html(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_report()))
    html = tmp_path / "r.html"

    result = invoke("report", str(path), "--html", str(html))
    assert result.exit_code == 0, result.output
    assert "2 tests" in result.output and "1 failed" in result.output
    assert html.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    result = invoke("report", str(path), "--failed")
    assert "test_bad" in result.output and "test_ok" not in result.output


def test_report_command_rejects_other_json(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"hello": 1}')
    result = invoke("report", str(path))
    assert result.exit_code != 0
    assert "is not an agenttest report" in result.output
