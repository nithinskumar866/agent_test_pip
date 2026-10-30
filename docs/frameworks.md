# Frameworks

| Framework | Pass to `run` | Trace capture | Tool mocking | Token tracking | Native `arun` |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LangGraph** | compiled graph | nodes + tools | LangChain tools | per node (`usage_metadata`) | `astream` |
| **CrewAI** | `Crew` | tasks + tools | CrewAI tools | crew total, on the last step | `akickoff` |
| **OpenAI Agents SDK** | `Agent` | model responses + tools | function tools, incl. handoff and `as_tool` agents | per response | `Runner.run` |
| **Anthropic SDK** | your tool loop (`adapter="anthropic"`) | `messages.create` calls | `@agenttest.tool` | per call | async client |
| **Plain Python** | any callable | `@agenttest.tool` | `@agenttest.tool` | - | `async def` agents |

Each integration is tested end to end against its real SDK, with only the model scripted: LangGraph
1.x, OpenAI Agents SDK 0.24, CrewAI 1.15 and Anthropic SDK 2.3.

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

Tools the SDK resolves only at run time (MCP servers) can't be mocked. When a run crashes, the trace
keeps whatever the SDK attached to the exception.

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

- Streaming APIs (`Runner.run_streamed`, LangGraph `stream_mode="messages"`) aren't traced.
- Tools inside a nested OpenAI `as_tool` agent are traced only when a mocked failure hits them.
- No live-API test runs against real OpenAI, Anthropic or CrewAI models. The integration tests use the
  real SDK code paths with scripted models.
