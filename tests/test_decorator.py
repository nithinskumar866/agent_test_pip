"""Tests for the @agent_test decorator and configuration loading."""

import asyncio
import json

import pytest

from agenttest import agent_test, run
from agenttest.config import AgentTestConfigError, _load_config, load_config
from agenttest.core.runner import current_settings


def test_agent_test_decorator_marks_function():
    @agent_test
    def dummy_test():
        return 42

    assert getattr(dummy_test, "_is_agent_test", False) is True
    assert dummy_test._agenttest_config["timeout"] is None
    assert any(mark.name == "agent" for mark in dummy_test.pytestmark)


def test_agent_test_decorator_with_params():
    @agent_test(timeout=120, max_retries=3, model_override="judge-x")
    def custom_test():
        return "ok"

    assert custom_test._agenttest_config == {"timeout": 120, "max_retries": 3, "model_override": "judge-x"}
    assert custom_test() == "ok"


def test_decorated_test_runs_without_config_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @agent_test
    def needs_no_config():
        return run(lambda q: f"echo {q}", "hi")

    result = needs_no_config()
    assert result.success is True
    assert result.output == "echo hi"


def test_settings_active_only_inside_test():
    seen = {}

    @agent_test(timeout=5)
    def check():
        seen["settings"] = current_settings()

    assert current_settings() is None
    check()
    assert seen["settings"].timeout == 5
    assert current_settings() is None


def test_decorator_timeout_applies_to_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import time

    @agent_test(timeout=0.2)
    def slow_test():
        return run(lambda: time.sleep(1))

    result = slow_test()
    assert result.success is False
    assert "AgentTestTimeoutError" in result.error


def test_async_test_is_supported():
    @agent_test(timeout=3)
    async def async_test():
        return current_settings().timeout

    assert asyncio.run(async_test()) == 3


@pytest.mark.parametrize("kwargs", [{"timeout": 0}, {"timeout": -1}, {"max_retries": 0}])
def test_invalid_decorator_arguments(kwargs):
    with pytest.raises(ValueError):
        agent_test(**kwargs)


def test_missing_config_file_uses_defaults(tmp_path):
    config = load_config(tmp_path / "agenttest.config.json")
    assert config.framework == "generic"
    assert config.timeout is None


def test_raw_loader_still_reports_missing_file(tmp_path):
    with pytest.raises(AgentTestConfigError, match="agenttest.config.json not found"):
        _load_config(tmp_path / "agenttest.config.json")


def test_invalid_config_is_rejected(tmp_path):
    path = tmp_path / "agenttest.config.json"
    path.write_text(json.dumps({"framework": "autogen"}), encoding="utf-8")
    with pytest.raises(AgentTestConfigError, match="unknown framework"):
        load_config(path)


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "agenttest.config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AgentTestConfigError, match="Failed to parse"):
        load_config(path)
