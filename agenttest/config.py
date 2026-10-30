"""Configuration loading and the agenttest exception hierarchy.

Owns reading/validating ``agenttest.config.json`` and the errors
raised across the package. Every agenttest error derives from
:class:`AgentTestError`, so callers can catch the whole family with one
``except`` clause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

__all__ = [
    "AgentTestError",
    "AgentTestConfigError",
    "AgentTestJudgeError",
    "AgentTestTimeoutError",
    "AgentTestConfig",
    "FRAMEWORKS",
    "JUDGE_PROVIDERS",
    "CONFIG_FILENAME",
    "load_config",
    "_load_config",
]

CONFIG_FILENAME: Final[str] = "agenttest.config.json"
FRAMEWORKS: Final[tuple[str, ...]] = ("langgraph", "crewai", "openai", "anthropic", "generic")
JUDGE_PROVIDERS: Final[tuple[str, ...]] = ("anthropic", "openai", "gemini", "ollama", "none")


class AgentTestError(Exception):
    """Base class for every error raised by agenttest."""


class AgentTestConfigError(AgentTestError):
    """Raised when agenttest is misconfigured.

    Covers a malformed ``agenttest.config.json``, an unknown framework or
    judge provider, and a missing API key for the LLM judge.
    """


class AgentTestJudgeError(AgentTestError):
    """Raised when the LLM judge fails or returns an unusable answer."""


class AgentTestTimeoutError(AgentTestError):
    """Raised (and recorded on the result) when an agent run exceeds its timeout."""


class AgentTestConfig(BaseModel):
    """Validated contents of ``agenttest.config.json``.

    Every field has a default, so a project without a config file still runs:
    the generic adapter, no timeout, and the Anthropic judge.
    """

    model_config = ConfigDict(extra="allow")

    framework: str = "generic"
    judge_provider: str = "anthropic"
    judge_model: str | None = None
    timeout: float | None = None
    max_retries: int = 1

    @field_validator("framework")
    @classmethod
    def _known_framework(cls, value: str) -> str:
        if value not in FRAMEWORKS:
            raise ValueError(f"unknown framework {value!r}; expected one of {', '.join(FRAMEWORKS)}")
        return value

    @field_validator("judge_provider")
    @classmethod
    def _known_judge(cls, value: str) -> str:
        if value not in JUDGE_PROVIDERS:
            raise ValueError(
                f"unknown judge_provider {value!r}; expected one of {', '.join(JUDGE_PROVIDERS)}"
            )
        return value

    @field_validator("timeout")
    @classmethod
    def _positive_timeout(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("timeout must be positive (or null for no timeout)")
        return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AgentTestConfigError(f"Failed to parse configuration file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentTestConfigError(f"Invalid configuration in {path}: expected JSON object.")
    return data


def load_config(config_path: Path | str | None = None) -> AgentTestConfig:
    """Load and validate agenttest configuration.

    A missing file is not an error: the defaults on :class:`AgentTestConfig`
    apply. A file that exists but is malformed or invalid raises.

    Args:
        config_path: Optional explicit path. Defaults to
            ``agenttest.config.json`` in the current working directory.

    Raises:
        AgentTestConfigError: If the file exists but is not valid JSON or fails validation.
    """
    path = Path(config_path) if config_path else Path.cwd() / CONFIG_FILENAME
    if not path.exists():
        return AgentTestConfig()
    try:
        return AgentTestConfig.model_validate(_read_json(path))
    except ValidationError as exc:
        raise AgentTestConfigError(f"Invalid configuration in {path}: {exc}") from exc


def _load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the raw config dict, raising if the file is missing.

    Kept for the judge, which distinguishes "no config" from "configured".
    New code should use :func:`load_config`.

    Raises:
        AgentTestConfigError: If the config file does not exist or contains invalid JSON.
    """
    path = Path(config_path) if config_path else Path.cwd() / CONFIG_FILENAME
    if not path.exists():
        raise AgentTestConfigError(
            f"{CONFIG_FILENAME} not found at {path}. Run 'agenttest init' first."
        )
    return _read_json(path)
