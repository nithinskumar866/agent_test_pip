# agenttest documentation

- [See it in action](demo.md)
- [Getting started](getting-started.md): install, `agenttest init`, `agenttest doctor`, a first test.
- [Running agents](running-agents.md): `run`, `arun`, `AgentResult`, timeouts, `@agent_test`, retries, async tests, custom adapters.
- [Frameworks](frameworks.md): LangGraph, CrewAI, the OpenAI Agents SDK, Pydantic AI, LlamaIndex and the Anthropic SDK.
- [Assertions](assertions.md): structural and judge-backed checks.
- [Tool mocking](mocking.md): failures, canned responses, scripted sequences and latency.
- [Conversations](conversations.md): multi-turn tests and simulated users.
- [Record and replay](replay.md): fast, free, repeatable tests.
- [Datasets](datasets.md): one test over many cases.
- [The judge](judge.md): grounded hallucination checks, providers, accuracy, limitations.
- [Cost, budgets and secrets](cost.md).
- [Reports, the CLI and CI](reports-and-ci.md): failure traces, reports, regressions, parallel runs, configuration.
- [Releasing](releasing.md): packaging and publishing to PyPI.

Runnable examples are in the [`examples/`](https://github.com/nithinskumar866/agent_test_pip/tree/main/examples) folder. The test suite runs them, so they stay correct.
