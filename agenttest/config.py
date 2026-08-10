"""Configuration loading and the agenttest exception hierarchy.

Owns reading/validating ``agenttest.config.json`` and the errors
raised across the package. Every agenttest error derives from
:class:`AgentTestError`, so callers can catch the whole family with one
``except`` clause.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "AgentTestError",
    "AgentTestConfigError",
    "AgentTestJudgeError",
    "_load_config",
]


class AgentTestError(Exception):
    """Base class for every error raised by agenttest."""


class AgentTestConfigError(AgentTestError):
    """Raised when agenttest is misconfigured.

    Covers a missing or malformed ``agenttest.config.json``, an unknown
    framework name, and a missing ``ANTHROPIC_API_KEY`` for the LLM judge.
    """


class AgentTestJudgeError(AgentTestError):
    """Raised when the LLM judge fails or returns an unusable answer.

    Covers transport failures talking to the Anthropic API and responses that
    cannot be parsed into the expected shape (a YES/NO verdict or a float).
    """


def _load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load agenttest configuration from agenttest.config.json.

    Args:
        config_path: Optional explicit path to config file. Defaults to
            ``agenttest.config.json`` in the current working directory.

    Returns:
        Dict containing configuration key-value pairs.

    Raises:
        AgentTestConfigError: If the config file does not exist or contains invalid JSON.
    """
    path = Path(config_path) if config_path else Path.cwd() / "agenttest.config.json"
    if not path.exists():
        raise AgentTestConfigError(
            f"agenttest.config.json not found at {path}. Run 'agenttest init' first."
        )

    try:
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise AgentTestConfigError(
                f"Invalid configuration in {path}: expected JSON object."
            )
        return data
    except json.JSONDecodeError as exc:
        raise AgentTestConfigError(
            f"Failed to parse configuration file {path}: {exc}"
        ) from exc
