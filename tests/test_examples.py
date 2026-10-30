"""The examples in examples/ are documentation: keep them passing."""

from pathlib import Path

import pytest

from tests.test_cli_commands import run_pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("test_*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_passes(example, tmp_path):
    result = run_pytest(example, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agenttest summary" in result.stdout


def test_examples_exist():
    assert {p.name for p in EXAMPLES} >= {"test_quickstart.py", "test_async_agent.py"}
