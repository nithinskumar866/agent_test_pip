# Frameworks

| Framework | Pass to `run` | Trace capture | Tool mocking | Token tracking | Native `arun` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LangGraph** | compiled graph | nodes + tools | LangChain tools | per node (`usage_metadata`) | `astream` |
| **CrewAI** | `Crew` | tasks + tools | CrewAI tools | crew total, on the last step | `akickoff` |
| **OpenAI Agents SDK** | `Agent` | model responses + tools | function tools, incl. handoff and `as_tool` agents | per response | `Runner.run` |
| **Pydantic AI** | `Agent` | model responses + tools | function tools | per response | `Agent.run` |
| **LlamaIndex** | workflow agent | tools | `FunctionTool` | not reported | `agent.run` |
| **Anthropic SDK** | your tool loop (`adapter="anthropic"`) | `messages.create` calls | `@agenttest.tool` | per call | async client |
| **Plain Python** | any callable | `@agenttest.tool` | `@agenttest.tool` | - | `async def` agents |

Each integration is tested end to end against its real SDK, with only the model scripted: LangGraph
1.2, CrewAI 1.15, Anthropic SDK 2.3, Pydantic AI 2.66, LlamaIndex 0.14, and the OpenAI Agents SDK at
both 0.20 and 0.24. A nightly CI job tests the newest release of each.

The newest OpenAI Agents SDK needs `openai` 3, while installing `pytest-agenttest[all]` currently
resolves to `openai` 2 and OpenAI Agents SDK 0.20. To get the newest version, install
`pytest-agenttest[openai-agents]` without the `crewai` extra.

## LangGraph

```python
graph = builder.compile()
result = agenttest.run(graph, {"messages": [("user", "latest news?")]})
```

The graph runs with `stream(stream_mode="updates")`, and each update becomes a node step. Tool steps
come from the `ToolMessage`s that tool nodes add. A LangChain tool called directly inside a node is
traced too, as is a mocked failure that crashes the graph (`metadata["unreported"]`).

Whether a mocked tool failure crashes the graph or becomes an error `ToolMessage` depends on the graph
(`ToolNode(handle_tool_errors=...)`), exactly as with a real failure.

## CrewAI

```python
result = agenttest.run(crew, {"topic": "AI agents"})     # passed as kickoff(inputs=...)
```

Each task output becomes a node step named after its agent's role. CrewAI only reports crew-wide token
usage, which is attached to the last step. CrewAI retries a failing tool on its own before the agent
sees the error, so a `mock_tool_failure` handle usually counts several calls.

## OpenAI Agents SDK

```python
agent = Agent(name="support", instructions="...", tools=[lookup_order])
result = agenttest.run(agent, "Where is order 42?")      # Runner.run_sync / Runner.run
```

Mocks reach the agent's function tools, the tools of agents it hands off to (listed directly or through
`handoff()`), and the tools of agents exposed with `Agent.as_tool()`. A mocked failure goes through the
tool's own failure policy, so the model sees what it would see for a real error.

Tools on MCP servers can be mocked by name too, and tools inside an `as_tool` agent appear on the trace with
`metadata["nested"]`. When a run crashes, the trace keeps whatever the SDK attached to the exception.

## Streaming

Pass `stream=True` to run through the framework's streaming path and measure time to first output:

```python
result = agenttest.run(agent, "question", stream=True)
result.trace.llm_calls[0].metadata["first_event_ms"]       # OpenAI Agents SDK

result = agenttest.run(graph, {"messages": [...]}, stream=True)
step = next(s for s in result.trace.llm_calls if s.name == "stream")
step.metadata["first_chunk_ms"], step.metadata["chunks"]   # LangGraph
```

## Pydantic AI

```python
result = agenttest.run(agent, "question")
```

Each model response becomes an LLM step, with its tokens and model name, followed by the tools it asked for.

## LlamaIndex

```python
result = agenttest.run(react_agent, "question")     # passed as user_msg=
```

LlamaIndex doesn't report token usage on the result, so `tokens_used` is 0.

## Anthropic SDK

Agents written directly on the Anthropic SDK are your own tool loops, so pass the function that runs
the loop:

```python
@agenttest.tool
def get_weather(city: str) -> str: ...

result = agenttest.run(lambda: my_tool_loop(client), adapter="anthropic")
```

During the run, every `messages.create` call (sync or async) is recorded as an LLM step with its
tokens, model, stop reason and the tools the model requested. Tool calls are traced and mocked through
`@agenttest.tool`.

## Not supported yet

- A user's own call to `Runner.run_streamed` or `graph.stream` inside a plain function isn't traced; use `stream=True`.
- No test runs against real OpenAI, Anthropic, CrewAI, Pydantic AI or LlamaIndex models. The integration tests
  use the real SDK code paths with scripted models.
