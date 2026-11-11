# Changelog

## 0.3.0 (unreleased)

### Added
- **Record and replay**: `@agent_test(cassette=True)` and `agenttest.cassette(...)`, at the HTTP layer, with
  `--agenttest-record`.
- **Scripted mocks**: `mock_tool_failure(times=)`, `mock_tool_sequence`, `mock_tool(handler)`, `REAL`.
- **Tool checks**: `assert_tool_called_with`, `assert_tool_call_count`, `assert_tools_in_order`, `assert_no_tool_errors`.
- **Cost and budgets**: `result.cost_usd`, `assert_cost_under`, `@agent_test(max_tokens=, max_cost_usd=)`,
  `--agenttest-max-tokens`, `--agenttest-max-cost`, and `pricing` in the config.
- **Conversations**: `converse`, `aconverse`, `simulated_user`.
- **Datasets**: `agenttest.cases`, `load_cases`.
- **Regression tracking**: `agenttest compare`, `agenttest report --markdown`, and a GitHub Action for pull requests.
- **Judge accuracy**: `agenttest benchmark`, 100 labelled cases.
- **Invented numbers caught in code**: `assert_grounded` first checks that every number in the output appears
  in the context, with no judge call; also `assert_numbers_grounded` and `invented_numbers_in`.
- **A second look at doubtful claims**: each claim judged unsupported is re-asked on its own, and only upgraded
  with verified quotes that contain all of its numbers (`judge_recheck`, on by default).
- **Test coverage gate** in CI (90%), and a CI job that runs the GitHub Action itself.
- **Safety checks**: `assert_no_pii_leak` / `find_pii` (emails, phones, Luhn-valid cards, IPs, API keys);
  `assert_resists_injection` with 12 built-in attacks, direct or planted in a tool's response, detected in code
  with a canary; `assert_refuses` (judge-backed).
- **Frameworks**: Pydantic AI and LlamaIndex adapters; MCP tool mocking and nested `as_tool` tracing for the OpenAI
  Agents SDK; `stream=True` for OpenAI and LangGraph.
- **Secret redaction** in reports, failure traces and cassettes.
- `pytest-xdist` support, a persistent judge cache (`judge_cache`), nightly CI against the newest frameworks, and a docs site.

### Changed
- The Gemini judge uses `google-genai` (the old SDK is deprecated) and defaults to `gemini-3.6-flash`.
- The JSON report gains `cost_usd` and per-run `tool_calls`.
- **`assert_grounded` now fails when the output has a number that is nowhere in the context**, without
  asking the judge. Pass `check_numbers=False` to get the old behaviour.
- The grounding judge no longer counts a claim that only joins already-supported claims as unsupported, and is
  told to judge meaning rather than wording ("March 6th" = "6 March").
- A judge connection dropped mid-reply (httpx `ReadError`, `RemoteProtocolError`) is retried like a timeout.
- The OpenAI Agents SDK trace records model responses and tool calls in the order they happened.

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
