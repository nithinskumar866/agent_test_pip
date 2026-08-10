"""``AgentResult`` — the outcome of one agent run, with assertions attached.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from typing import Any, Final

from pydantic import BaseModel, Field, PrivateAttr

from agenttest.config import AgentTestConfigError, AgentTestError, AgentTestJudgeError, _load_config

__all__ = ["AgentResult", "DEFAULT_JUDGE_MODEL", "_judge_llm", "_clear_judge_cache"]

DEFAULT_JUDGE_MODEL: Final[str] = "claude-haiku-4-5-20251001"

_JUDGE_CACHE: dict[str, str] = {}
_CACHE_LOCK: Final[threading.Lock] = threading.Lock()
_JUDGE_CACHE_LOCK: Final[threading.Lock] = _CACHE_LOCK

_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def _judge_llm(prompt: str) -> str:
    try:
        config = _load_config()
    except AgentTestConfigError:
        config = {}

    provider = config.get("judge_provider", "anthropic")
    model = config.get("judge_model", DEFAULT_JUDGE_MODEL)

    # Thread-safe cache
    cache_key = hashlib.sha256(
        f"{provider}:{model}:{prompt}".encode()
    ).hexdigest()

    with _CACHE_LOCK:
        if cache_key in _JUDGE_CACHE:
            return _JUDGE_CACHE[cache_key]

    if provider == "anthropic":
        result = _judge_anthropic(prompt, model)
    elif provider == "openai":
        result = _judge_openai(prompt, model)
    elif provider == "gemini":
        result = _judge_gemini(prompt, model)
    elif provider == "ollama":
        result = _judge_ollama(prompt, model)
    elif provider == "none":
        raise AgentTestConfigError(
            "LLM judge is disabled (judge_provider=none). "
            "Use structural assertions like word_count_between() "
            "instead, or set a judge provider in agenttest init."
        )
    else:
        raise AgentTestConfigError(
            f"Unknown judge_provider: '{provider}'. "
            "Valid options: anthropic, openai, gemini, ollama, none"
        )

    with _CACHE_LOCK:
        _JUDGE_CACHE[cache_key] = result
    return result


def _judge_anthropic(prompt: str, model: str) -> str:
    import os
    try:
        import anthropic
    except ImportError:
        raise AgentTestConfigError(
            "anthropic package not installed. Run: pip install anthropic"
        )
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise AgentTestConfigError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Set it with: export ANTHROPIC_API_KEY=your_key"
        )
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or DEFAULT_JUDGE_MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def _judge_openai(prompt: str, model: str) -> str:
    import os
    try:
        import openai
    except ImportError:
        raise AgentTestConfigError(
            "openai package not installed. Run: pip install openai"
        )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise AgentTestConfigError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY=your_key"
        )
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def _judge_gemini(prompt: str, model: str) -> str:
    import os
    try:
        import google.generativeai as genai
    except ImportError:
        raise AgentTestConfigError(
            "google-generativeai not installed. "
            "Run: pip install google-generativeai"
        )
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise AgentTestConfigError(
            "GOOGLE_API_KEY environment variable not set. "
            "Set it with: export GOOGLE_API_KEY=your_key"
        )
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model or "gemini-1.5-flash")
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


def _judge_ollama(prompt: str, model: str) -> str:
    try:
        import httpx
    except ImportError:
        raise AgentTestConfigError(
            "httpx not installed. Run: pip install httpx"
        )
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model or "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except httpx.ConnectError:
        raise AgentTestConfigError(
            "Cannot connect to Ollama at localhost:11434. "
            "Make sure Ollama is running: ollama serve"
        )
    except Exception as exc:
        raise AgentTestConfigError(
            f"Ollama request failed: {exc}"
        ) from exc


def _clear_judge_cache() -> None:
    """Empty the judge cache. Intended for agenttest's own test suite."""
    with _CACHE_LOCK:
        _JUDGE_CACHE.clear()


class AgentResult(BaseModel):
    """The outcome of one agent run, with assertions attached."""

    success: bool
    output: str
    steps_taken: int
    tools_called: list[str]
    tokens_used: int
    latency_ms: float
    gracefully_degraded: bool
    raw_trace: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    _hallucination_cache: float | None = PrivateAttr(default=None)

    def contains_fact(self, claim: str) -> bool:
        """Check whether the output asserts a factual claim using an LLM judge."""
        prompt = (
            f"Does the following text contain the factual claim: "
            f"'{claim}'?\n\nText: {self.output}\n\n"
            f"Answer only YES or NO."
        )
        answer = _judge_llm(prompt)
        return answer.strip().upper().startswith("YES")

    @property
    def hallucination_score(self) -> float:
        """Rate hallucination likelihood from 0.0 to 1.0."""
        if self._hallucination_cache is not None:
            return self._hallucination_cache

        prompt = (
            f"Rate how likely this text contains hallucinations "
            f"or factual inaccuracies on a scale from 0.0 (completely "
            f"accurate) to 1.0 (severe hallucinations).\n\n"
            f"Text: {self.output}\n\n"
            f"Return only a single float number between 0.0 and 1.0. "
            f"Nothing else."
        )
        answer = _judge_llm(prompt)
        match = _FLOAT_PATTERN.search(answer)
        if match is None:
            score = 0.5
        else:
            score = min(1.0, max(0.0, float(match.group())))

        self._hallucination_cache = score
        return score

    def tone_matches(self, tone: str) -> bool:
        """Check whether the output is written in a requested tone."""
        prompt = (
            f"Does the following text match a '{tone}' tone?\n\n"
            f"Text: {self.output}\n\n"
            f"Answer only YES or NO."
        )
        answer = _judge_llm(prompt)
        return answer.strip().upper().startswith("YES")

    def word_count_between(self, min_count: int, max_count: int) -> bool:
        """Check the output's word count against an inclusive range."""
        count = len(self.output.split())
        return min_count <= count <= max_count
