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
Use `await agenttest.arun(...)` in async tests.

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

| Framework | Status | Trace Capture | Tool Mocking | Token Tracking |
| :--- | :---: | :---: | :---: | :---: |
| **LangGraph** | ✅ | ✅ | ✅ | ⚠️ (0) |
| **CrewAI** | ✅ | ✅ | ✅ | ⚠️ (0) |
| **OpenAI Agents SDK** | ✅ | ✅ | ✅ | ✅ |
| **Anthropic SDK** | ✅ | ✅ | ✅ | ✅ |
| **Generic Python Agent** | ✅ | ✅ | ✅ | ⚠️ (0) |

---

## 🧠 Bring Your Own Judge Model

`agenttest` supports multi-provider LLM-as-a-judge routing. Select your preferred judge model during initialization or in `agenttest.config.json`.

| Provider | Command | Env Var | Default Model | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **Anthropic** | `agenttest init --judge anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | Low |
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

LLM-judged assertions powered by your configured judge provider. They raise `AssertionError` including the judge's answer; a judge reply that cannot be parsed raises `AgentTestJudgeError` instead of being guessed at.

1. `assert_contains_fact(output, "Paris")` — Fact verification.
2. `assert_hallucination_score(output, max_score=0.1)` — Hallucination likelihood rating (0.0 to 1.0) at most `max_score`.
3. `assert_tone_matches(output, "professional")` — Tone and style checking.
4. `assert_answers_question(output, "What is Python?")` — Question coverage evaluation.
5. `assert_language_matches(output, "Spanish")` — Target language detection.

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
