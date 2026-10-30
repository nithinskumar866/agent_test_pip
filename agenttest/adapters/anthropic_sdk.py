"""Anthropic SDK adapter — traces hand-written tool loops on the raw SDK.

During a run, ``anthropic.resources.messages.Messages.create`` (and its async
twin) are patched so every Messages API call is recorded as an ``llm`` step
with its token usage and the tools the model requested. The patch is removed
when the run ends. Tools the loop executes are traced with
:func:`agenttest.tool`.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, ClassVar
from unittest import mock

from agenttest.adapters.base import BaseAdapter, text_of
from agenttest.core.trace import record_step

__all__ = ["AnthropicAdapter"]


def _record_message(request: dict[str, Any], response: Any, duration_ms: float) -> None:
    usage = getattr(response, "usage", None)
    blocks = getattr(response, "content", None) or []
    record_step(
        "llm",
        "messages.create",
        input=request.get("messages"),
        output=" ".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text"),
        model=getattr(response, "model", None) or request.get("model"),
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        tool_requests=[getattr(b, "name", "unknown") for b in blocks if getattr(b, "type", None) == "tool_use"],
        duration_ms=duration_ms,
        metadata={"stop_reason": getattr(response, "stop_reason", None)},
    )


def _traced_sync(original: Any) -> Any:
    @functools.wraps(original)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response = original(self, *args, **kwargs)
        _record_message(kwargs, response, (time.perf_counter() - start) * 1000)
        return response

    return create


def _traced_async(original: Any) -> Any:
    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        response = await original(self, *args, **kwargs)
        _record_message(kwargs, response, (time.perf_counter() - start) * 1000)
        return response

    return create


class AnthropicAdapter(BaseAdapter):
    """Adapter for agents built directly on the Anthropic SDK."""

    name: ClassVar[str] = "anthropic"

    @contextmanager
    def instrument(self) -> Iterator[None]:
        try:
            from anthropic.resources.messages import AsyncMessages, Messages
        except ImportError:
            yield
            return
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(Messages, "create", _traced_sync(Messages.create)))
            stack.enter_context(mock.patch.object(AsyncMessages, "create", _traced_async(AsyncMessages.create)))
            yield

    def extract_output(self, raw: Any) -> str:
        blocks = getattr(raw, "content", None)
        if isinstance(blocks, list):
            return " ".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text")
        return text_of(raw)
