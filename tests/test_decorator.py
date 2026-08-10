"""Tests for @agent_test decorator and config loading."""

from pathlib import Path
from unittest.mock import patch

import pytest

from agenttest.config import AgentTestConfigError
from agenttest.core.decorator import _load_config, agent_test


def test_agent_test_decorator_marks_function():
    @agent_test
    def dummy_test():
        return 42

    assert getattr(dummy_test, "_is_agent_test", False) is True
    assert dummy_test._agenttest_config["timeout"] == 60


def test_agent_test_decorator_with_params():
    @agent_test(timeout=120, max_retries=3, model_override="claude-3-opus")
    def custom_test():
        return "ok"

    assert getattr(custom_test, "_is_agent_test", False) is True
    assert custom_test._agenttest_config["timeout"] == 120
    assert custom_test._agenttest_config["max_retries"] == 3
    assert custom_test._agenttest_config["model_override"] == "claude-3-opus"


def test_missing_config_raises_file_not_found():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises((AgentTestConfigError, FileNotFoundError)) as exc_info:
            _load_config()
        assert "agenttest.config.json not found" in str(exc_info.value)
