"""Estimating what an agent run cost, from its token counts.

Prices are USD per million tokens. The built-in table covers Claude models;
add any other model under ``pricing`` in ``agenttest.config.json``. A run that
used a model with no known price reports ``cost_usd = None`` rather than a
number that is silently too low.

These are estimates: prompt caching, batch discounts and long-context pricing
are not taken into account.
"""

from __future__ import annotations

from typing import Any, Final

from agenttest.config import AgentTestConfigError, load_config

__all__ = ["DEFAULT_PRICING", "price_for", "trace_cost"]

DEFAULT_PRICING: Final[dict[str, dict[str, float]]] = {
    "claude-fable-5-1": {"input": 10.0, "output": 50.0},
    "claude-fable-5": {"input": 10.0, "output": 50.0},
    "claude-opus-5-5": {"input": 4.0, "output": 20.0},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-5-5": {"input": 0.1, "output": 0.5},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


def _configured() -> dict[str, dict[str, float]]:
    try:
        return load_config().pricing
    except AgentTestConfigError:
        return {}


def price_for(model: str | None, table: dict[str, dict[str, float]] | None = None) -> dict[str, float] | None:
    """The price entry for ``model``, matching dated ids by their base name."""
    if not model:
        return None
    prices = {**DEFAULT_PRICING, **(_configured() if table is None else table)}
    if model in prices:
        return prices[model]
    # "claude-haiku-4-5-20251001" -> the longest known id it starts with.
    matches = [known for known in prices if model.startswith(f"{known}-")]
    return prices[max(matches, key=len)] if matches else None


def trace_cost(trace: Any) -> float | None:
    """Estimated USD cost of a trace; ``None`` if any step's model has no known price."""
    charged = [step for step in trace.steps if step.tokens]
    if not charged:
        return 0.0
    table = _configured()
    total = 0.0
    for step in charged:
        price = price_for(step.model, table)
        if price is None:
            return None
        total += (step.input_tokens * price["input"] + step.output_tokens * price["output"]) / 1_000_000
    return round(total, 6)
