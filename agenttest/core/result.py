"""``AgentResult`` — the outcome of one agent run, with assertions attached.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Final

from pydantic import BaseModel, Field, PrivateAttr

from agenttest.config import AgentTestConfigError, AgentTestError, AgentTestJudgeError, _load_config
from agenttest.core.trace import Trace
from agenttest.judge.protocol import JUDGE_SYSTEM_PROMPT, current_judge
from agenttest.judge.providers import DEFAULT_MODELS, PROVIDERS, call_judge

__all__ = ["AgentResult", "DEFAULT_JUDGE_MODEL", "_judge_llm", "_clear_judge_cache", "_discard_cached"]

DEFAULT_JUDGE_MODEL: Final[str] = DEFAULT_MODELS["anthropic"]

#: Default reply budget; verdicts are short, grounding reports ask for more.
_DEFAULT_MAX_TOKENS: Final[int] = 256

_JUDGE_CACHE: dict[str, str] = {}
_CACHE_LOCK: Final[threading.Lock] = threading.Lock()
_JUDGE_CACHE_LOCK: Final[threading.Lock] = _CACHE_LOCK


def _judge_settings() -> dict[str, Any]:
    try:
        return _load_config()
    except AgentTestConfigError:
        return {}


def _cache_key(prompt: str, max_tokens: int) -> str:
    config = _judge_settings()
    provider = config.get("judge_provider", "anthropic")
    model = config.get("judge_model") or DEFAULT_MODELS.get(provider, "")
    return hashlib.sha256(f"{provider}:{model}:{max_tokens}:{JUDGE_SYSTEM_PROMPT}:{prompt}".encode()).hexdigest()


def _judge_llm(prompt: str, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> str:
    """Send ``prompt`` to the judge and return its reply.

    The single gateway to the judge: a judge installed with
    :func:`agenttest.use_judge` takes precedence; otherwise the provider from
    ``agenttest.config.json`` is called (see :mod:`agenttest.judge.providers`
    for retries and error handling). Provider replies are cached per session.
    """
    custom = current_judge()
    if custom is not None:
        reply = custom(prompt, JUDGE_SYSTEM_PROMPT)
        if not isinstance(reply, str) or not reply.strip():
            raise AgentTestJudgeError("Custom judge returned an empty reply.")
        return reply.strip()

    config = _judge_settings()
    provider = config.get("judge_provider", "anthropic")
    if provider == "none":
        raise AgentTestConfigError(
            "LLM judge is disabled (judge_provider=none). "
            "Use structural assertions like word_count_between() "
            "instead, or set a judge provider in agenttest init."
        )
    if provider not in PROVIDERS:
        raise AgentTestConfigError(
            f"Unknown judge_provider: '{provider}'. "
            "Valid options: anthropic, openai, gemini, ollama, none"
        )

    key = _cache_key(prompt, max_tokens)
    with _CACHE_LOCK:
        if key in _JUDGE_CACHE:
            return _JUDGE_CACHE[key]

    reply = call_judge(
        provider,
        config.get("judge_model"),
        prompt,
        system=JUDGE_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        max_retries=int(config.get("judge_max_retries", 2)),
        timeout=float(config.get("judge_timeout", 60.0)),
    )
    with _CACHE_LOCK:
        _JUDGE_CACHE[key] = reply
    return reply


def _discard_cached(prompt: str, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
    """Forget a cached reply (used when it turned out to be malformed)."""
    with _CACHE_LOCK:
        _JUDGE_CACHE.pop(_cache_key(prompt, max_tokens), None)


def _clear_judge_cache() -> None:
    """Empty the judge cache. Intended for agenttest's own test suite."""
    with _CACHE_LOCK:
        _JUDGE_CACHE.clear()


class AgentResult(BaseModel):
    """The outcome of one agent run, with assertions attached.

    Built by the engine via :meth:`from_trace`; the summary fields
    (``steps_taken``, ``tools_called``, ``tokens_used``,
    ``gracefully_degraded``) are derived from ``trace``.
    """

    success: bool
    output: str
    steps_taken: int
    tools_called: list[str]
    tokens_used: int
    latency_ms: float
    gracefully_degraded: bool
    raw_trace: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    trace: Trace = Field(default_factory=Trace)
    #: The agent's untouched return value (not serialised).
    raw_output: Any = Field(default=None, exclude=True, repr=False)

    _hallucination_cache: float | None = PrivateAttr(default=None)
    _exception: BaseException | None = PrivateAttr(default=None)

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        output: str,
        latency_ms: float,
        raw_output: Any = None,
        exception: BaseException | None = None,
        traceback: str | None = None,
    ) -> AgentResult:
        """Summarise a finished run.

        ``gracefully_degraded`` means at least one tool failed during the run,
        yet the agent still completed without raising and produced output.
        """
        success = exception is None
        result = cls(
            success=success,
            output=output,
            steps_taken=len(trace.steps),
            tools_called=trace.tools_called,
            tokens_used=trace.tokens_used,
            latency_ms=latency_ms,
            gracefully_degraded=success and bool(output.strip()) and bool(trace.failed_tool_calls),
            raw_trace=trace.model_dump(),
            error=f"{type(exception).__name__}: {exception}" if exception is not None else None,
            traceback=traceback,
            trace=trace,
            raw_output=raw_output,
        )
        result._exception = exception
        return result

    def raise_for_error(self) -> None:
        """Re-raise the exception that ended the run, if there was one."""
        if self._exception is not None:
            raise self._exception
        if self.error is not None:
            raise AgentTestError(self.error)

    # Judge-backed checks delegate to agenttest.assertions.semantic so each
    # prompt and its parsing live in one place (imported lazily: semantic
    # imports this module).

    def contains_fact(self, claim: str) -> bool:
        """Check whether the output asserts a factual claim using an LLM judge."""
        from agenttest.assertions.semantic import contains_fact

        return contains_fact(self.output, claim)

    @property
    def hallucination_score(self) -> float:
        """Judge-rated hallucination likelihood from 0.0 to 1.0 (cached per result)."""
        if self._hallucination_cache is None:
            from agenttest.assertions.semantic import hallucination_score

            self._hallucination_cache = hallucination_score(self.output)
        return self._hallucination_cache

    def check_grounding(self, context: Any = None) -> Any:
        """Check the output's claims against ``context`` (default: this run's tool outputs).

        Returns a :class:`~agenttest.judge.GroundingReport`.
        """
        from agenttest.assertions.semantic import check_grounding

        return check_grounding(self, context)

    def tone_matches(self, tone: str) -> bool:
        """Check whether the output is written in a requested tone."""
        from agenttest.assertions.semantic import tone_matches

        return tone_matches(self.output, tone)

    def word_count_between(self, min_count: int, max_count: int) -> bool:
        """Check the output's word count against an inclusive range."""
        count = len(self.output.split())
        return min_count <= count <= max_count
