# agenttest documentation

1. [Getting started](getting-started.md): install, `agenttest init`, `agenttest doctor`, a first test.
2. [Running agents](running-agents.md): `run`, `arun`, `AgentResult`, timeouts, `@agent_test`, retries, async tests, custom adapters.
3. [Frameworks](frameworks.md): LangGraph, CrewAI, the OpenAI Agents SDK and the Anthropic SDK.
4. [Assertions](assertions.md): structural and judge-backed checks.
5. [The judge and grounded hallucination checks](judge.md): providers, reliability, limitations.
6. [Tool mocking](mocking.md): failures, canned responses and latency.
7. [Reports, the CLI and CI](reports-and-ci.md): the summary, failure traces, JSON/HTML reports, configuration.

Runnable examples are in [`examples/`](../examples). The test suite runs them, so they stay correct.
