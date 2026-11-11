# Record and replay

Tests that call a real model are slow, cost money and give different answers each time. A
**cassette** records the model API calls once, then replays them, so the test runs in milliseconds,
for free, and the same way every time.

```python
@agent_test(cassette=True)
def test_agent_finds_the_capital():
    result = agenttest.run(my_agent, "What is the capital of France?")
    assert_output_contains(result, "Paris")
```

The first run calls the API and saves `cassettes/<test file>/<test name>.json` next to the test.
Commit that file. Later runs, including CI, replay it.

Replay works at the HTTP layer, underneath the Anthropic, OpenAI and Gemini SDKs and the frameworks
built on them, so it needs no framework-specific setup. Tools still run, so mock any that touch the
outside world. Use `with agenttest.cassette("path.json"):` around any block of code too.

## Record modes

| Mode | Behaviour |
| :--- | :--- |
| `once` | Replay if the cassette exists, otherwise record. The default on your machine. |
| `none` | Replay only, and never touch the network. The default when `CI` is set. |
| `new` | Replay what matches, and record calls that don't. |
| `all` | Always call the API, and rewrite the cassette. |

Set the mode with `pytest --agenttest-record=all`, the `AGENTTEST_RECORD` environment variable, or
`cassette("path.json", mode="new")`.

## When the prompt changes

A request matches on its method, URL and body. If you change a prompt or a tool definition, the
request no longer matches and the test stops with `CassetteMissError`, rather than replaying an answer
to a different question. Re-record with `--agenttest-record=all`.

If a field changes on every run, such as a request id, ignore it:
`cassette("path.json", ignore_body_keys=["request_id"])`.

## What gets intercepted

Only known model providers: Anthropic, OpenAI (including Azure), Gemini, Mistral, Groq, Cohere,
Together, OpenRouter and a local Ollama. Telemetry and other traffic pass straight through. Add a
private gateway with `cassette("path.json", hosts=["llm.internal"])`. `@agent_test(cassette=...)` uses
the default hosts; use the context manager for extra options.

## Good to know

- Request headers, which hold API keys, are never stored. Bodies go through [redaction](cost.md#hiding-secrets),
  but read a cassette before you commit it if your prompts contain personal data.
- A cassette is only saved if the block finishes without an exception, so a failed recording is never replayed.
  Rate-limit and server errors are not recorded.
- Repeated identical requests replay in the order they were recorded, then start again from the first.
- With `runs=5`, the recording pass makes five real calls, so replay reproduces any variation between them.
- Replayed runs have almost no latency, so don't combine them with latency assertions.
- SDKs that use another HTTP client, such as `aiohttp`, aren't intercepted. In `none` mode such a call
  reaches the network instead of failing.
