# Running agents

## `agenttest.run`

```python
result = agenttest.run(agent, *args, adapter=None, timeout=None, **kwargs)
```

`agent` can be a function, an object with a `run()` or `invoke()` method, or a framework agent: a
compiled LangGraph graph, an OpenAI Agents SDK `Agent`, or a CrewAI `Crew`. Framework agents are
detected by type. Force an adapter with `adapter="langgraph"` (or `"crewai"`, `"openai"`,
`"anthropic"`, `"generic"`) or by passing an adapter instance. `*args` and `**kwargs` go to the agent.

**The agent's exceptions never escape `run`.** They are recorded on the result:

```python
if not result.success:
    print(result.error)       # "TimeoutError: upstream timed out"
    print(result.traceback)
result.raise_for_error()      # re-raise it, if you prefer
```

`run` only raises for agenttest misuse, such as an invalid config or calling it inside a running
event loop (use `arun` there).

## `AgentResult`

| Field | Meaning |
| :--- | :--- |
| `success` | The run finished without raising. |
| `output` | The agent's final text. |
| `trace` | Every recorded step: `trace.steps`, `trace.tool_calls`, `trace.llm_calls`, `trace.failed_tool_calls`. |
| `tools_called` | Names of the tools called, in order. |
| `steps_taken` | Number of recorded steps. |
| `tokens_used` | Input + output tokens, where the framework reports them. |
| `latency_ms` | Wall-clock duration of the run. |
| `gracefully_degraded` | A tool failed, yet the agent finished and produced output. |
| `error`, `traceback` | Set when `success` is false. |
| `raw_output` | The agent's untouched return value (not serialised). |

Each `Step` has `kind` (`"llm"`, `"tool"` or `"node"`), `name`, `input`, `output`, `error`,
`duration_ms`, token counts, `tool_requests` (tools a model asked for) and `metadata` (for example
`{"mocked": "failure"}`).

Judge-backed helpers are also available on the result: `result.contains_fact(...)`,
`result.tone_matches(...)`, `result.hallucination_score` and `result.check_grounding(context)`.

## Timeouts

```python
result = agenttest.run(agent, "question", timeout=30)
```

When the timeout passes, the result has `success=False` with an `AgentTestTimeoutError`, and its trace
holds the steps recorded so far. A synchronous run can't be killed, so it finishes in a background
thread; with `arun`, a framework's native async run is cancelled.

The default comes from `@agent_test(timeout=...)`, then from `timeout` in `agenttest.config.json`.

## `@agent_test`

```python
@agent_test(timeout=30, runs=5, min_pass_rate=0.8, max_retries=1)
def test_agent_answers():
    ...
```

The decorator marks the test with the `agent` marker (`pytest -m agent` selects agent tests), makes
`timeout` the default for runs inside the test, and runs the body under a retry and pass-rate policy:

- `runs`: how many times to execute the test body.
- `max_retries`: extra attempts a failing run gets before it counts as failed.
- `min_pass_rate`: the fraction of runs that must pass.

Execution stops as soon as the verdict is certain. A test that passes only after retries, or with some
failed runs, is reported as **flaky**: it emits `AgentTestFlakyWarning` and is listed in the summary
and the reports. When it fails, the error lists the reason for each run.

Options you leave unset fall back to `agenttest.config.json`, then to the defaults. Fixtures are
created once per test, so every run sees the same fixture values.

## Async agents and async tests

```python
@agent_test
async def test_async_agent():
    result = await agenttest.arun(my_async_agent, "question")
    assert result.success
```

`async def` agent tests work without extra plugins: when neither pytest-asyncio (auto mode, or an
`asyncio` marker) nor anyio (an `anyio` marker) claims an agent test, agenttest runs it on a fresh
event loop.

`arun` runs framework agents through their native async API: LangGraph `astream`, the OpenAI
`Runner.run`, CrewAI `akickoff`. A plain `async def` agent runs on the event loop, and a synchronous
agent runs in a worker thread.

## Tracing plain Python agents

Frameworks are traced automatically. For your own code, decorate tools:

```python
@agenttest.tool
def lookup(order_id: str) -> dict: ...

@agenttest.tool(name="search_web")       # custom name
async def search(query: str) -> str: ...
```

Calls are recorded as tool steps (with input, output, error and timing) only during an agenttest run,
and the decorated function works normally outside of one. Tool mocks match these names.

## Custom adapters

To support another framework, subclass `BaseAdapter` and pass an instance as `adapter=`:

```python
from contextlib import contextmanager
from agenttest import BaseAdapter
from agenttest.core.trace import record_step

class MyFrameworkAdapter(BaseAdapter):
    name = "myframework"

    @contextmanager
    def instrument(self):          # install hooks for one run; remove them on exit
        yield

    def invoke(self, agent, *args, **kwargs):
        result = agent.execute(*args, **kwargs)
        for event in result.events:
            record_step("tool", event.tool, input=event.args, output=event.result)
        return result

    async def ainvoke(self, agent, *args, **kwargs):   # optional native async path
        return await agent.aexecute(*args, **kwargs)

    def extract_output(self, raw):
        return raw.final_text

result = agenttest.run(agent, "question", adapter=MyFrameworkAdapter())
```
