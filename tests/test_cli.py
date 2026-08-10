"""Tests for agenttest CLI init command using click.testing.CliRunner."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from agenttest.cli.init import _detect_framework, _detect_judge_provider, init


def test_cli_init_with_explicit_framework(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(init, ["--framework", "langgraph"])
        assert result.exit_code == 0
        assert "langgraph" in result.output

        config_file = Path("agenttest.config.json")
        assert config_file.exists()
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
        assert config_data["framework"] == "langgraph"


def test_cli_init_with_explicit_judge(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(init, ["--framework", "openai", "--judge", "openai"])
        assert result.exit_code == 0
        assert "openai" in result.output

        config_file = Path("agenttest.config.json")
        assert config_file.exists()
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
        assert config_data["judge_provider"] == "openai"
        assert config_data["judge_model"] == "gpt-4o-mini"


def test_cli_init_detects_anthropic_judge(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            with patch("importlib.import_module", return_value=True):
                detected = _detect_judge_provider()
                assert detected == "anthropic"


def test_framework_detection_from_requirements(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        req_file = Path("requirements.txt")
        req_file.write_text("crewai>=0.1.0\npytest>=7.0", encoding="utf-8")

        detected = _detect_framework()
        assert detected == "crewai"

        result = runner.invoke(init, [])
        assert result.exit_code == 0
        assert "crewai" in result.output


def test_config_json_written_correctly(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(init, ["--framework", "generic", "--judge", "ollama"])
        assert result.exit_code == 0
        config_file = Path("agenttest.config.json")
        assert config_file.exists()
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
        assert config_data["framework"] == "generic"
        assert config_data["judge_provider"] == "ollama"
        assert config_data["judge_model"] == "llama3"


def test_config_contains_judge_provider(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(init, ["--framework", "generic", "--judge", "none"])
        config_file = Path("agenttest.config.json")
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
        assert "judge_provider" in config_data
        assert config_data["judge_provider"] == "none"
        assert config_data["judge_model"] == "none"
