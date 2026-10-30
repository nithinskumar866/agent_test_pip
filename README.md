# agenttest

[![CI](https://github.com/nithinskumar866/agent_test_pip/actions/workflows/ci.yml/badge.svg)](https://github.com/nithinskumar866/agent_test_pip/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

> **pytest for AI agents.** Catch silent agent failures before they reach production.

AI agents fail in ways unit tests miss: hallucinations, tool loops, broken JSON, unhandled tool
outages, answers that change from run to run. `agenttest` is a pytest plugin for testing agent
*behaviour*. It runs your agent, traces every model and tool step, simulates tool failures, checks
answers against the context the agent was given, and judges flaky behaviour statistically.

## Install

```bash
pip install "pytest-agenttest[anthropic]"   # or [openai], [gemini], [ollama]: the judge provider you use
agenttest init --example     # write agenttest.config.json and a runnable example test
agenttest doctor             # check the config, framework packages and judge API key
```

The package is published as `pytest-agenttest`; you `import agenttest` and run the `agenttest`
command. Add `[langgraph]`, `[crewai]` or `[openai-agents]` for those frameworks, or `[all]`.

## Quick start

```python
import agenttest
from agenttest import agent_test, assert_grounded, assert_tool_was_called, mock_tool_failure


@agent_test(timeout=30)
def test_agent_answers_from_its_tools():
    result = agenttest.run(my_agent, "What is the capital of France?")

    assert result.success, result.traceback
    assert_tool_was_called(result, "web_search")
    assert_grounded(result)          # every claim is backed by what the tools returned


@agent_test
def test_agent_survives_a_search_outage():
    with mock_tool_failure("web_search", "503 Service Unavailable"):
        result = agenttest.run(my_agent, "What is the capital of France?")

    assert result.gracefully_degraded   # the tool failed, yet the agent still answered


@agent_test(runs=5, min_pass_rate=0.8)
async def test_async_agent_is_mostly_right():
    result = await agenttest.arun(my_async_agent, "What is 6 x 7?")
    assert "42" in result.output
```

```bash
pytest --agenttest-html=agenttest-report.html
```

## What you get

- **One way to run any agent.** `agenttest.run` / `arun` accept a function, a compiled LangGraph graph,
  an OpenAI Agents SDK `Agent`, a CrewAI `Crew` or an Anthropic SDK tool loop. They return an
  `AgentResult` with the output, a step-by-step trace, tools, tokens and latency. The agent's
  exceptions never escape; timeouts are built in.
- **Tool mocking across frameworks.** `mock_tool_failure`, `mock_tool_response` and `mock_tool_latency`
  match tools by name in every supported framework. A mocked failure is handled exactly like a real one.
- **Grounded hallucination checks.** `assert_grounded` checks each claim against the documents or tool
  results the agent had. The judge must quote its evidence, and every quote is verified in code.
- **A judge you can rely on.** Defences against prompt injection from agent output, a repair retry for
  malformed replies, backoff on API failures, and `use_judge` for scripted judges in your own tests.
- **Statistics for non-deterministic agents.** `@agent_test(runs=5, min_pass_rate=0.8, max_retries=1)`
  stops as soon as the verdict is certain, and reports flaky tests.
- **Useful failures and reports.** A failing test shows a trace of every agent run it made. Add
  `--agenttest-report` (JSON) or `--agenttest-html` for CI artifacts.
- **Assertions that explain themselves.** Every `assert_*` raises `AssertionError` with the expected
  value, the actual value and an excerpt of the output.

## Frameworks

| Framework | Trace capture | Tool mocking | Native `arun` |
| :--- | :---: | :---: | :---: |
| LangGraph | nodes + tools | LangChain tools | `astream` |
| CrewAI | tasks + tools | CrewAI tools | `akickoff` |
| OpenAI Agents SDK | model responses + tools | function tools, incl. handoffs and `as_tool` agents | `Runner.run` |
| Anthropic SDK | `messages.create` calls | `@agenttest.tool` | async client |
| Plain Python | `@agenttest.tool` | `@agenttest.tool` | `async def` agents |

Each integration is tested end to end against the real SDK, with only the model scripted. Details
and limitations are in [docs/frameworks.md](docs/frameworks.md).

## Documentation

- [Getting started](docs/getting-started.md)
- [Running agents](docs/running-agents.md): `run`, `arun`, `AgentResult`, timeouts, retries, async tests, custom adapters
- [Frameworks](docs/frameworks.md)
- [Assertions](docs/assertions.md)
- [The judge and grounded hallucination checks](docs/judge.md), including known limitations
- [Tool mocking](docs/mocking.md)
- [Reports, the CLI and CI](docs/reports-and-ci.md), including the configuration reference
- Runnable [examples](examples/)
- [Releasing](docs/releasing.md) and [security](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Contributing

Contributions are welcome. Open an issue or a pull request on GitHub.

```bash
git clone https://github.com/nithinskumar866/agent_test_pip.git
cd agent_test_pip
pip install -e ".[dev]"          # add ,all to also run the framework integration tests
ruff check . && mypy && pytest
```

See [docs/releasing.md](docs/releasing.md) for how releases are built and published, and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

MIT License. Copyright (c) 2026.
