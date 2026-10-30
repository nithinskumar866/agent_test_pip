# Getting started

## Install

```bash
pip install agenttest
```

agenttest needs Python 3.10+. It is a pytest plugin: once installed, `pytest` loads it automatically.

## Set up a project

```bash
agenttest init --example
agenttest doctor
```

`agenttest init` detects your agent framework from your dependency files and picks a judge provider
from the API keys in your environment, then writes `agenttest.config.json`. Override either with
`--framework` and `--judge`. It never overwrites an existing config unless you pass `--force`.

`--example` also writes `tests/test_agent_example.py`, a small suite that runs without any API key.

`agenttest doctor` checks that everything is in place: the config is valid, the framework's package
is installed, and the judge's package and API key are present. It exits non-zero if something is
missing, so it works as a CI pre-flight step. Add `--ping-judge` to make one real judge call.

## Your first test

```python
import agenttest
from agenttest import agent_test, assert_tool_was_called


@agenttest.tool                     # traced and mockable
def search(query: str) -> str:
    return "Paris is the capital of France."


def my_agent(question: str) -> str:
    return f"According to my search: {search(question)}"


@agent_test
def test_agent_uses_search():
    result = agenttest.run(my_agent, "What is the capital of France?")
    assert result.success, result.traceback
    assert_tool_was_called(result, "search")
```

```bash
pytest
```

The output ends with an agenttest summary:

```
============================= agenttest summary ==============================
Agent tests: 1 passed, 0 failed, 0 skipped
Agent runs: 1 (0 failed), 0 tokens, 1 tool calls (0 failed), slowest 1 ms
```

When an agent test fails, pytest shows a trace of every agent run the test made, step by step.
See [Reports and CI](reports-and-ci.md).

## Next

- [Running agents](running-agents.md): `run`, `arun`, `AgentResult`, timeouts, retries, async tests.
- [Frameworks](frameworks.md): LangGraph, CrewAI, the OpenAI Agents SDK and the Anthropic SDK.
- [Assertions](assertions.md) and [grounded hallucination checks](judge.md).
- [Tool mocking](mocking.md).
- Runnable examples: [`examples/`](../examples).
