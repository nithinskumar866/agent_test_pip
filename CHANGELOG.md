# Changelog

## 0.2.0 (unreleased): the rework

### Packaging and release
- Published as **`pytest-agenttest`** (the PyPI name `agenttest` belongs to another project). The import
  name and the `agenttest` command are unchanged.
- Packaging moved to `pyproject.toml` alone; `setup.cfg` is gone. Added a `LICENSE` file and a
  `py.typed` marker.
- The core install needs only pytest, pydantic, click and rich. Judge providers and frameworks are
  extras (`[anthropic]`, `[openai]`, `[gemini]`, `[ollama]`, `[langgraph]`, `[crewai]`,
  `[openai-agents]`, `[all]`, `[dev]`), and a missing one is reported with the `pip install` command.
- CI: ruff and mypy, tests on Python 3.10-3.13 (Linux, plus Windows and macOS), real-SDK integration
  tests, a wheel-install check that runs the examples, and `pip-audit`. Dependabot is configured.
- Releases are published with PyPI trusted publishing from a tag, after a TestPyPI rehearsal; the
  previous workflow's stored API token is no longer used. See `docs/releasing.md` and `SECURITY.md`.

### Developer experience
- pytest plugin: a failing agent test shows a trace of every agent run it made (`--agenttest-no-trace`
  turns it off); the summary adds run, token and tool-call totals.
- `--agenttest-report=PATH` (JSON) and `--agenttest-html=PATH` (self-contained, escaped HTML) reports.
- `async def` agent tests run without pytest-asyncio; tests claimed by pytest-asyncio or anyio are
  left to them.
- CLI: `agenttest doctor`, `agenttest report`, `agenttest --version`, and `init --example` / `--force`.
  `init` no longer overwrites an existing config silently. CLI output no longer crashes on legacy
  Windows consoles.
- Documentation in `docs/`, and runnable examples in `examples/` that the test suite executes.

### AI evaluation
- `assert_grounded` / `check_grounding`: context-grounded hallucination checks with evidence verified
  in code; `hallucination_score(..., context=...)`.
- Judge hardening: prompt-injection fencing, one repair retry for malformed replies, backoff on
  transient API failures (`judge_max_retries`, `judge_timeout`), `AgentTestJudgeError` for other
  failures, and `use_judge` for scripted judges.
- The Anthropic judge no longer sends `temperature` (the current API rejects it). The default judge
  model is now `claude-haiku-5-5`.

### Framework integrations
- Native async runs: LangGraph `astream`, OpenAI `Runner.run`, CrewAI `akickoff`.
- LangGraph traces tools called inside nodes, and mocked failures that crash the graph.
- OpenAI mocks reach agents behind `handoff()` and `Agent.as_tool()`.
- End-to-end tests against the real SDKs with scripted models.

### Testing engine
- Tool mocks work across `@agenttest.tool`, LangChain/LangGraph, OpenAI Agents SDK and CrewAI tools.
  Each mock yields a handle that counts calls.
- Every `assert_*` raises `AssertionError` with details. Boolean versions are in `agenttest.assertions`.
- `@agent_test(runs=..., min_pass_rate=..., max_retries=...)` with early stopping and flaky reporting.
  `max_retries` now means extra attempts (default 0).

### Core
- `agenttest.run` / `arun` execution engine with timeouts and captured exceptions; `AgentResult`
  with a step-by-step `Trace`; the `@agenttest.tool` decorator; adapter auto-detection.
