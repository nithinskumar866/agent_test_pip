"""Hiding secrets in everything agenttest writes down.

Reports and cassettes end up as CI artifacts and in git, so anything that
looks like a credential is replaced before it is written:

* common API-key and token formats (built in);
* the value of any environment variable whose name suggests a secret;
* patterns you add under ``redact_patterns`` in ``agenttest.config.json``.

This is a safety net, not a guarantee: a secret in an unrecognised format, or
personal data, needs a ``redact_patterns`` entry.
"""

from __future__ import annotations

import os
import re
from typing import Any, Final

from agenttest.config import AgentTestConfigError, load_config

__all__ = ["REDACTED", "redact", "redact_value"]

REDACTED: Final[str] = "[REDACTED]"

_BUILTIN: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p)
    for p in (
        r"sk-[A-Za-z0-9_-]{16,}",  # OpenAI, Anthropic
        r"AIza[0-9A-Za-z_-]{30,}",  # Google
        r"AKIA[0-9A-Z]{16}",  # AWS access key id
        r"gh[pousr]_[A-Za-z0-9]{20,}",  # GitHub
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"xox[abprs]-[A-Za-z0-9-]{10,}",  # Slack
        r"(?i)bearer\s+[A-Za-z0-9._~+/-]{16,}=*",
        r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",  # JWT
    )
)
_SECRET_NAME: Final[re.Pattern[str]] = re.compile(r"(?i)(api_?key|secret|token|password|passwd|credential)")
#: Shorter values are too likely to match ordinary text.
_MIN_ENV_SECRET: Final[int] = 12


def _configured() -> list[re.Pattern[str]]:
    try:
        return [re.compile(p) for p in load_config().redact_patterns]
    except (AgentTestConfigError, re.error):
        return []


def _env_secrets() -> list[str]:
    values = {v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= _MIN_ENV_SECRET}
    return sorted(values, key=len, reverse=True)


def redact(text: str) -> str:
    """``text`` with anything that looks like a secret replaced."""
    for secret in _env_secrets():
        text = text.replace(secret, REDACTED)
    for pattern in (*_BUILTIN, *_configured()):
        text = pattern.sub(REDACTED, text)
    return text


def redact_value(value: Any) -> Any:
    """A copy of a JSON-like value with every string redacted."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value
