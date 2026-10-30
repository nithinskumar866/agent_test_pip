# agenttest

![PyPI version](https://badge.fury.io/py/agenttest.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

> **pytest for AI agents.** Catch silent agent failures before they reach production.

---

## ⚡ The Problem

AI agents fail in non-traditional ways: hallucinations, infinite tool loops, persona drift, broken JSON outputs, and unhandled tool outages. Standard unit tests fail to catch these behavioral issues, and building custom test harnesses takes weeks.

`agenttest` brings automated, production-grade behavioral testing to AI agent pipelines. It integrates seamlessly into standard `pytest` workflows with zero hassle.

---

## 📦 Install

```bash
pip install agenttest
agenttest init
```

Running `agenttest init` auto-detects your agent framework and configures your `agenttest.config.json`.

---

## 🚀 Quick Start

```python
import agenttest
from agenttest import agent_test

@agent_test(timeout=30)
def test_agent_is_factual_and_efficient():
    result = agenttest.run(my_agent, "What is the capital of France?")
    assert result.success, result.traceback
    assert result.contains_fact("Paris")
    assert result.steps_taken <= 3
```

`agenttest.run(agent, *args)` runs any agent — a function, an object with
`run()`/`invoke()`, a compiled LangGraph graph, an OpenAI Agents SDK `Agent` or a
CrewAI `Crew` — and returns an `AgentResult` with the output, a step-by-step
`trace`, tools called, tokens and latency. Agent exceptions never escape: check
`result.success` / `result.error`, or call `result.raise_for_error()`.
Use `await agenttest.arun(...)` in async tests: it drives each framework through its
native async API, so a timeout cancels the run instead of abandoning a thread.

For framework-less agents, decorate tools with `@agenttest.tool` so their calls
are traced. A run where a tool failed but the agent still answered is reported
as `result.gracefully_degraded`:

```python
@agenttest.tool
def web_search(query: str) -> str: ...

def test_handles_tool_failure_gracefully():
    result = agenttest.run(my_agent, "Search for recent tech news")
    assert result.gracefully_degraded
```

Run directly with `pytest`:

```bash
pytest
```

---

## 🛠️ Supported Frameworks

| Framework | Pass to `run` | Trace Capture | Tool Mocking | Token Tracking | Native `arun` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LangGraph** | compiled graph | nodes + tools | LangChain tools | per node (`usage_metadata`) | `astream` |
| **CrewAI** | `Crew` | tasks + tools | CrewAI tools | crew total, on the last step | `akickoff` |
| **OpenAI Agents SDK** | `Agent` | model responses + tools | function tools, incl. handoff and `as_tool` agents | per response | `Runner.run` |
| **Anthropic SDK** | your tool loop (`adapter="anthropic"`) | `messages.create` calls | `@agenttest.tool` | per call | async client |
| **Generic Python Agent** | any callable | `@agenttest.tool` | `@agenttest.tool` | — | `async def` agents |

Each framework is tested end to end against its real SDK, with only the model scripted
(LangGraph 1.x, OpenAI Agents SDK 0.24, CrewAI 1.15, Anthropic SDK 2.3).

Notes:

- **LangGraph** — a tool called directly inside a node is traced too, and so is a mocked
  failure that crashes the graph (`metadata["unreported"]`).
- **CrewAI** retries a failing tool on its own before the agent sees the error, so a
  `mock_tool_failure` handle usually counts several calls.
- **OpenAI** — tools the SDK resolves at run time (MCP servers) cannot be mocked. A crashed
  run keeps whatever the SDK attached to the exception.

---

## 🧠 Bring Your Own Judge Model

`agenttest` supports multi-provider LLM-as-a-judge routing. Select your preferred judge model during initialization or in `agenttest.config.json`.

| Provider | Command | Env Var | Default Model | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Anthropic** | `agenttest init --judge anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | Low |
| **OpenAI** | `agenttest init --judge openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | Low |
| **Gemini** | `agenttest init --judge gemini` | `GOOGLE_API_KEY` | `gemini-1.5-flash` | Very Low |
| **Ollama** | `agenttest init --judge ollama` | *(Local instance)* | `llama3` | **Free** |
| **None** | `agenttest init --judge none` | *(None needed)* | `none` | **Free** |

---

## 🔍 Structural Assertions (No API Key Needed)

Pure Python assertions that evaluate speed, output structure, token count, and tool usage without touching the network. Each one raises `AssertionError` with the expected value, the actual value and an excerpt of the output; each accepts an `AgentResult` or the raw value. Bool versions without the `assert_` prefix live in `agenttest.assertions`.

1. `assert_word_count_between(output, min_count, max_count)` — Verify word counts within inclusive limits.
2. `assert_steps_at_most(steps_taken, max_steps)` — Cap maximum execution steps.
3. `assert_tool_was_called(tools_called, tool_name)` — Ensure a specific tool was invoked.
4. `assert_tool_was_not_called(tools_called, tool_name)` — Verify forbidden tools were avoided.
5. `assert_latency_under(latency_ms, threshold_ms)` — Enforce strict response latency targets.
6. `assert_tokens_under(tokens_used, max_tokens)` — Enforce token consumption budgets.
7. `assert_output_contains(output, substring)` — Simple case-insensitive substring search.
8. `assert_output_matches_regex(output, pattern)` — Regex pattern matching.
9. `assert_output_is_valid_json(output)` — Validate JSON output syntax.
10. `assert_gracefully_degraded(result)` — Ensure agent handled tool errors gracefully.

---

## 🤖 Semantic Assertions (Judge Model Required)

LLM-judged assertions powered by your configured judge provider. They raise `AssertionError` including the judge's answer.

1. `assert_grounded(result, context)` — Every factual claim is backed by the context (see below).
2. `assert_contains_fact(output, "Paris")` — Fact verification.
3. `assert_hallucination_score(output, max_score=0.1, context=...)` — Hallucination score (0.0 to 1.0) at most `max_score`; grounded when `context` is given.
4. `assert_tone_matches(output, "professional")` — Tone and style checking.
5. `assert_answers_question(output, "What is Python?")` — Question coverage evaluation.
6. `assert_language_matches(output, "Spanish")` — Target language detection.

### Grounded hallucination checks

"Is this hallucinated?" has no reliable answer in the abstract. A grounded check asks a checkable
question instead: is every claim backed by the documents or tool results the agent was given?

```python
from agenttest import assert_grounded, check_grounding

# Context defaults to the run's successful tool outputs
assert_grounded(result)

# Or pass the retrieved documents explicitly
report = assert_grounded(result, context=[doc1, doc2], max_unsupported=0.0)
report.score          # fraction of claims contradicted or unsupported
report.contradicted   # [ClaimVerdict(claim=..., evidence=..., reason=...)]
```

Each claim is labelled `supported`, `contradicted`, `unsupported` or `disputed` (the context
disagrees with itself). The assertion fails on any contradicted claim, on more unsupported claims
than `max_unsupported` allows, on disputed claims (unless `allow_disputed=True`), and when the
output tries to instruct the evaluator.

**The judge must quote its evidence, and every quote is checked in code.** It has to appear in the
passage it cites, ignoring case, whitespace and quote style. A label whose evidence doesn't check
out is downgraded to `unsupported`. So a judge that was misled can't certify a claim with a quote
that only exists in the agent's own output.

### Judge reliability

- **Prompt injection:** the agent's output (and every context passage) is enclosed in markers derived
  from a hash of that text, so the text can't contain its own closing marker. The system prompt tells
  the judge that the enclosed text is data and that instructions inside it must not be followed.
- **Malformed replies** get one retry with the parse error attached, then raise `AgentTestJudgeError`.
  Malformed replies are never cached.
- **API failures:** timeouts, connection errors, rate limits and 5xx errors are retried with
  exponential backoff (`judge_max_retries`, default 2; `judge_timeout`, default 60s). Other API
  errors, empty replies and refusals raise `AgentTestJudgeError`, which pytest reports as an
  *error*, not a test failure.
- **Determinism:** OpenAI, Gemini and Ollama judges run at temperature 0. The Anthropic API no
  longer accepts sampling parameters, so Claude judges rely on the strict reply format and the
  evidence checks instead.
- **Scripted judges for your own tests:** `with agenttest.use_judge(lambda prompt, system: "YES"):`

### Known limitations

- A judge can still **miss a claim entirely**, or return no claims at all. Only the evidence for
  claims it *does* list is verified, so `assert_grounded` is only as complete as the judge's
  claim extraction.
- Evidence quotes are matched exactly (after normalisation). A paraphrased quote is rejected, which
  can mark a true claim `unsupported`; very short quotes match trivially.
- The injection defences reduce risk, but they can't guarantee that a model never follows injected
  text. `injection_detected` is the judge's own report.
- Context isn't chunked or truncated: very large contexts must fit in the judge model's window.
- Without `context`, `hallucination_score` is the judge's opinion, not a verification.
- The scripted tests cover parsing, verification and failure handling. To check that a real judge
  labels claims correctly, run the live tests:
  `AGENTTEST_LIVE_JUDGE=anthropic ANTHROPIC_API_KEY=... pytest tests/test_judge_live.py`

---

## 🧪 Tool Mocking

Simulate tool failures, canned responses and network delays. Mocks match tools by name and apply to
`@agenttest.tool` functions, LangChain/LangGraph tools, OpenAI Agents SDK function tools and CrewAI tools:

```python
import agenttest
from agenttest import mock_tool_failure, mock_tool_response, mock_tool_latency

# Make the tool fail (the framework handles it exactly like a real failure)
with mock_tool_failure("web_search", "503 Service Unavailable", exception=ConnectionError) as handle:
    result = agenttest.run(agent, "Fetch latest weather")
assert handle.calls == 1          # catches mocks whose name never matched
assert result.gracefully_degraded

# Return canned data instead of running the tool
with mock_tool_response("calculator", {"result": 42}):
    agenttest.run(agent, "Calculate 6 * 7")

# Delay the real tool (milliseconds)
with mock_tool_latency("database_query", delay_ms=500):
    agenttest.run(agent, "Fetch user record")
```

---

## 🔁 Retries and Pass Rates

Agents are non-deterministic. Run a test several times and require a pass rate, optionally retrying failed runs:

```python
@agent_test(runs=5, min_pass_rate=0.8, max_retries=1)
def test_agent_answers_correctly():
    result = agenttest.run(my_agent, "What is the capital of France?")
    assert_output_contains(result, "Paris")
```

Execution stops as soon as the verdict is certain. Tests that pass only after retries or with some failed
runs emit `AgentTestFlakyWarning` and are listed in the agenttest summary. `runs`, `min_pass_rate` and
`max_retries` can also be set project-wide in `agenttest.config.json`.

---

## ⚙️ CI/CD Integration

Add this step to `.github/workflows/test.yml` to run automated agent testing in GitHub Actions:

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install agenttest pytest
      - name: Run agenttest suite
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          pytest
```

---

## 🤝 Contributing

Contributions are very welcome! Please feel free to open an issue or submit a pull request on GitHub.

1. Fork the repo and create your feature branch (`git checkout -b feature/my-feature`).
2. Install development dependencies (`pip install -e .[dev]`).
3. Run test suite (`pytest tests/ -v`).
4. Push to branch and open a Pull Request.

---

## 📜 License

MIT License. Copyright (c) 2026.
