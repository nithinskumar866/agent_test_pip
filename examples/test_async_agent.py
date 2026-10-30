"""Async agents and async tests.

`async def` agent tests run as-is: the agenttest plugin runs them on an event
loop when no async plugin (pytest-asyncio, anyio) claims them.

    pytest examples/test_async_agent.py
"""

import asyncio

import agenttest
from agenttest import agent_test, assert_tool_was_called, mock_tool_latency


@agenttest.tool
async def fetch_price(ticker: str) -> float:
    await asyncio.sleep(0.01)  # stands in for a network call
    return 187.5


async def price_agent(ticker: str) -> str:
    price = await fetch_price(ticker)
    return f"{ticker} trades at ${price:.2f}."


@agent_test
async def test_async_agent_reports_the_price():
    result = await agenttest.arun(price_agent, "ACME")

    assert result.output == "ACME trades at $187.50."
    assert_tool_was_called(result, "fetch_price")


@agent_test(timeout=0.2)
async def test_slow_tool_hits_the_timeout():
    with mock_tool_latency("fetch_price", delay_ms=1000):
        result = await agenttest.arun(price_agent, "ACME")

    assert result.success is False
    assert "timeout" in result.error
