# Cost, budgets and secrets

## Cost

Every run reports an estimated cost:

```python
result = agenttest.run(my_agent, "question")
print(result.cost_usd)
assert_cost_under(result, 0.01)
```

Claude models are priced out of the box. Add any other model to `agenttest.config.json`, in USD per
million tokens:

```json
{"pricing": {"gpt-5.1-mini": {"input": 0.25, "output": 2.0}}}
```

If a model's price isn't known, `cost_usd` is `None` rather than a number that's silently too low. The
figure is an estimate: prompt caching and batch discounts aren't included. Frameworks that don't report
the model or tokens (CrewAI's model, LlamaIndex's tokens) can't be priced.

## Budgets

Stop a test that spends too much:

```python
@agent_test(max_tokens=20_000, max_cost_usd=0.05)
def test_agent(): ...
```

The budget covers all the test's runs and retries. Going over it fails the test with
`AgentTestBudgetError`, and it isn't retried.

Stop a whole session:

```bash
pytest --agenttest-max-tokens=500000 --agenttest-max-cost=2.00
```

Both count your agent's model usage, not the judge's. Under pytest-xdist, the limit is checked as results
reach the controller, so tests already running in other workers will still finish.

## Cheaper test suites

- [Record and replay](replay.md) model calls.
- Set `"judge_cache": true` to keep judge verdicts in `.agenttest_cache/` between sessions (`agenttest init` turns it on).
  A changed prompt, output or judge model is a new entry.
- Use `use_judge` in tests that only exercise your own logic.

## Hiding secrets

Reports, failure traces and cassettes hide anything that looks like a credential:

- common formats (OpenAI, Anthropic, Google, AWS, GitHub and Slack keys, bearer tokens, JWTs);
- the value of any environment variable with `KEY`, `SECRET`, `TOKEN` or `PASSWORD` in its name;
- your own patterns:

```json
{"redact_patterns": ["EMP-\\d{4}", "\\b\\d{3}-\\d{2}-\\d{4}\\b"]}
```

This is a safety net. A secret in an unusual format, or personal data, needs a pattern of its own.
