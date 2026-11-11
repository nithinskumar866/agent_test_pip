# Reports, the CLI and CI

## In the pytest output

When agent tests run, pytest's output ends with an agenttest summary:

```
============================= agenttest summary ==============================
Agent tests: 11 passed, 1 failed, 0 skipped
Agent runs: 14 (1 failed), 5230 tokens, 22 tool calls (3 failed), slowest 4210 ms
Flaky (passed, but not on every run/attempt): 1
  tests/test_support.py::test_refund_policy: 4/5 runs passed (80%); required 4/5 (min_pass_rate=0.8)
```

When an agent test fails, the failure output includes an **agenttest runs** section: every agent run
the test made, step by step, with tool results, errors, mocks and tokens.

```
------------------------------- agenttest runs --------------------------------
run 1 of 1: ok - 812 ms, 3 steps, 96 tokens, gracefully degraded
   1. node agent  (tokens 40+12; requested web_search)
   2. tool web_search  (mocked failure)
        error: MockedToolError: 503 Service Unavailable
   3. node agent  (tokens 34+10)
  output: 'Search is down; answering from memory: ...'
```

Turn it off with `--agenttest-no-trace`. Run only agent tests with `pytest -m agent`.

## Report files

```bash
pytest --agenttest-report=agenttest-report.json --agenttest-html=agenttest-report.html
```

- **JSON** (`--agenttest-report`): totals, and for each agent test (and any other test that ran an
  agent) its outcome, failure message and every run with its steps. Framework objects in the trace are
  converted to text, and long values are shortened.
- **HTML** (`--agenttest-html`): a single self-contained page with the same content. Every value is
  HTML-escaped, so agent output can't inject markup.

## CLI

| Command | What it does |
| :--- | :--- |
| `agenttest init [--framework X] [--judge Y] [--example] [--force]` | Write `agenttest.config.json` (and an example test). |
| `agenttest doctor [--ping-judge]` | Check config, framework packages, judge package and key. Exits 1 on a problem. |
| `agenttest report PATH [--failed] [--html OUT]` | Summarise a JSON report, or turn it into HTML. |
| `agenttest report PATH --markdown OUT` | Write a markdown summary. |
| `agenttest compare BASE CURRENT [--max-token-increase N] [--max-cost-increase N]` | Compare two reports. Exits 1 on a regression. |
| `agenttest benchmark [--limit N] [--min-accuracy X]` | Measure your [judge's accuracy](judge.md). |
| `agenttest --version` | Print the version. |

`doctor` never prints API keys, only whether they are set.

## Catching regressions

Keep a report from a good run and compare against it:

```bash
agenttest compare baseline.json current.json --max-token-increase 20 --max-cost-increase 20
```

A test that passed and now fails is always a regression. Token and cost growth count only if you set a limit.

## Parallel runs

Summaries and reports work under pytest-xdist: `pytest -n 4 --agenttest-report=r.json`.

## Comment on pull requests

This repository is also a GitHub Action that posts the results on the pull request, and updates the same
comment on each push:

```yaml
permissions:
  contents: read
  pull-requests: write

steps:
  # ... install, then run pytest --agenttest-report=agenttest-report.json ...
  - uses: nithinskumar866/agent_test_pip@main
    if: always()
    with:
      report: agenttest-report.json
      baseline: baseline/agenttest-report.json   # optional
      max-token-increase: "20"                   # optional
```

## GitHub Actions

```yaml
name: Agent tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install "pytest-agenttest[anthropic]" -r requirements.txt
      - run: agenttest doctor
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - run: pytest --agenttest-report=agenttest-report.json --agenttest-html=agenttest-report.html
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: agenttest-report
          path: agenttest-report.*
```

Judge calls cost money and can be rate-limited. Keep semantic assertions to the tests that need them,
and use `use_judge` in tests that only exercise your own logic.

## Configuration reference

`agenttest.config.json` (every key is optional):

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `framework` | `"generic"` | Adapter used when the agent's type doesn't identify it. |
| `timeout` | none | Seconds before a run is abandoned. |
| `runs` | `1` | Times each `@agent_test` body runs. |
| `min_pass_rate` | `1.0` | Fraction of runs that must pass. |
| `max_retries` | `0` | Extra attempts per failing run. |
| `judge_provider` | `"anthropic"` | See [the judge](judge.md). |
| `judge_model` | provider default | |
| `judge_max_retries` | `2` | Retries for transient judge API failures. |
| `judge_timeout` | `60` (`600` for Ollama) | Seconds per judge call. |
| `judge_cache` | `false` | Keep judge replies on disk between sessions. |
| `pricing` | `{}` | Extra model prices. See [cost](cost.md). |
| `redact_patterns` | `[]` | Extra patterns to hide. See [cost](cost.md#hiding-secrets). |

An invalid file (bad JSON, an unknown framework, a negative timeout) raises `AgentTestConfigError` when
it is read. `agenttest doctor` reports it up front.
