"""Calling the judge model: one function per provider, plus retries.

Every provider is called with the shared judge system prompt, at
temperature 0 where the provider still accepts sampling parameters (OpenAI,
Ollama). Current Anthropic and Gemini models reject or ignore ``temperature``,
so for those, consistency comes from the strict reply formats and the
deterministic evidence checks instead. :func:`call_judge` adds the
reliability layer:

* transient failures — timeouts, connection errors, rate limits (429),
  overload and server errors (5xx) — are retried with exponential backoff;
* any other API failure, or retries running out, raises
  :class:`~agenttest.AgentTestJudgeError` naming the provider and model;
* an empty reply is an error, never an implicit verdict;
* configuration problems (missing package or API key) raise
  :class:`~agenttest.AgentTestConfigError` immediately and are not retried.

The SDK clients' own retries are switched off so attempts are not multiplied.
"""

from __future__ import annotations

import importlib.util
import os
import time
from collections.abc import Callable
from typing import Any, Final

from agenttest.config import AgentTestConfigError, AgentTestJudgeError

__all__ = ["PROVIDERS", "DEFAULT_MODELS", "DEFAULT_TIMEOUTS", "call_judge", "is_transient"]

DEFAULT_MODELS: Final[dict[str, str]] = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.6-flash",
    "ollama": "llama3",
}

#: Current Claude models think before answering, and thinking counts against
#: ``max_tokens``; this floor keeps a short verdict from being cut off.
_ANTHROPIC_MIN_TOKENS: Final[int] = 4096

#: Gemini thinking also counts against the output limit.
_GEMINI_MIN_TOKENS: Final[int] = 4096

#: Seconds per call when ``judge_timeout`` isn't set. A local model on a CPU can
#: take minutes to write a grounding report; hosted APIs answer in seconds.
DEFAULT_TIMEOUTS: Final[dict[str, float]] = {"ollama": 600.0}

#: Seconds before the first retry; doubles per attempt, capped at ``_MAX_DELAY``.
_BASE_DELAY: Final[float] = 0.5
_MAX_DELAY: Final[float] = 8.0
_TRANSIENT_NAMES: Final[tuple[str, ...]] = ("timeout", "connection", "ratelimit", "overloaded", "unavailable")


def is_transient(exc: BaseException) -> bool:
    """Whether a failed judge call is worth retrying."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    return any(word in cls.__name__.lower() for cls in type(exc).__mro__ for word in _TRANSIENT_NAMES)


def _require_package(module: str, extra: str) -> None:
    """Report a missing SDK before a missing API key: installing comes first."""
    if importlib.util.find_spec(module) is None:
        raise AgentTestConfigError(f'{module} package not installed. Run: pip install "pytest-agenttest[{extra}]"')


def _require_key(env_var: str) -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise AgentTestConfigError(
            f"{env_var} environment variable not set. Set it with: export {env_var}=your_key"
        )
    return api_key


# --- providers -----------------------------------------------------------------
# Each takes (prompt, system, model, max_tokens, timeout) and returns the reply text.

def _anthropic_client(api_key: str, timeout: float) -> Any:
    try:
        import anthropic
    except ImportError:
        raise AgentTestConfigError(
            'anthropic package not installed. Run: pip install "pytest-agenttest[anthropic]"'
        ) from None
    return anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)


def _call_anthropic(prompt: str, system: str, model: str, max_tokens: int, timeout: float) -> str:
    _require_package("anthropic", "anthropic")
    client = _anthropic_client(_require_key("ANTHROPIC_API_KEY"), timeout)
    response = client.messages.create(
        model=model,
        max_tokens=max(max_tokens, _ANTHROPIC_MIN_TOKENS),
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        raise AgentTestJudgeError(f"Judge {model} refused to evaluate (category: {category}).")
    return "".join(getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text")


def _openai_client(api_key: str, timeout: float) -> Any:
    try:
        import openai
    except ImportError:
        raise AgentTestConfigError(
            'openai package not installed. Run: pip install "pytest-agenttest[openai]"'
        ) from None
    return openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=0)


def _call_openai(prompt: str, system: str, model: str, max_tokens: int, timeout: float) -> str:
    _require_package("openai", "openai")
    client = _openai_client(_require_key("OPENAI_API_KEY"), timeout)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _gemini_client(api_key: str, timeout: float) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))


def _call_gemini(prompt: str, system: str, model: str, max_tokens: int, timeout: float) -> str:
    _require_package("google.genai", "gemini")
    from google.genai import types

    client = _gemini_client(_require_key("GOOGLE_API_KEY"), timeout)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max(max_tokens, _GEMINI_MIN_TOKENS),
        ),
    )
    return response.text or ""


def _call_ollama(prompt: str, system: str, model: str, max_tokens: int, timeout: float) -> str:
    try:
        import httpx
    except ImportError:
        raise AgentTestConfigError("httpx not installed. Run: pip install \"pytest-agenttest[ollama]\"") from None
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
            timeout=timeout,
        )
    except httpx.ConnectError:
        raise AgentTestConfigError(
            "Cannot connect to Ollama at localhost:11434. Make sure Ollama is running: ollama serve"
        ) from None
    if response.status_code == 404:
        raise AgentTestConfigError(_missing_ollama_model(httpx, model))
    response.raise_for_status()
    return response.json()["response"]


def _missing_ollama_model(httpx: Any, model: str) -> str:
    """Explain a missing model, listing the ones that are installed."""
    message = f"Ollama has no model {model!r}."
    try:
        installed = [m["name"] for m in httpx.get("http://localhost:11434/api/tags", timeout=5).json()["models"]]
    except Exception:  # the hint is a bonus; never hide the real problem behind it
        installed = []
    if installed:
        message += f" Installed: {', '.join(installed)}."
    return message + f' Set "judge_model" in agenttest.config.json to one of them, or run: ollama pull {model}'


PROVIDERS: Final[dict[str, Callable[[str, str, str, int, float], str]]] = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
    "ollama": _call_ollama,
}


def call_judge(
    provider: str,
    model: str | None,
    prompt: str,
    *,
    system: str,
    max_tokens: int,
    max_retries: int = 2,
    timeout: float = 60.0,
) -> str:
    """Ask the judge model and return its non-empty reply, retrying transient failures."""
    call = PROVIDERS[provider]
    model = model or DEFAULT_MODELS[provider]
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            reply = call(prompt, system, model, max_tokens, timeout)
        except (AgentTestConfigError, AgentTestJudgeError):
            raise
        except Exception as exc:
            where = f"{provider}/{model}"
            if not is_transient(exc):
                raise AgentTestJudgeError(f"Judge call to {where} failed: {type(exc).__name__}: {exc}") from exc
            if attempt == attempts:
                raise AgentTestJudgeError(
                    f"Judge call to {where} failed after {attempts} attempt{'s' if attempts > 1 else ''}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(min(_BASE_DELAY * 2 ** (attempt - 1), _MAX_DELAY))
            continue
        if not isinstance(reply, str) or not reply.strip():
            raise AgentTestJudgeError(f"Judge {provider}/{model} returned an empty reply.")
        return reply.strip()
    raise AssertionError("unreachable")  # pragma: no cover
