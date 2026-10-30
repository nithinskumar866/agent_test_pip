# Tool mocking

Mocks simulate tool failures, canned responses and slow tools. They match tools **by name** and apply
to `@agenttest.tool` functions, LangChain/LangGraph tools, OpenAI Agents SDK function tools and CrewAI
tools.

```python
from agenttest import mock_tool_failure, mock_tool_latency, mock_tool_response

# Make the tool fail. The framework handles it exactly like a real failure.
with mock_tool_failure("web_search", "503 Service Unavailable") as handle:
    result = agenttest.run(agent, "latest news?")
assert handle.calls == 1           # catches a mock whose name never matched a tool
assert result.gracefully_degraded

# Raise a specific exception type
with mock_tool_failure("web_search", exception=ConnectionError("reset")):
    ...

# Return canned data instead of running the tool
with mock_tool_response("calculator", {"result": 42}):
    ...

# Run the real tool, but slower (milliseconds)
with mock_tool_latency("database_query", delay_ms=500):
    ...
```

Each context manager yields a `MockHandle`. `handle.calls` counts how many calls it intercepted.
Mocked calls are recorded on the trace with `metadata["mocked"]` set to `"failure"`, `"response"` or
`"latency"`.

Nested mocks of the same tool stack: the innermost one applies, and leaving it restores the outer one.
A call that passes through several layers (for example a CrewAI tool wrapping a decorated function) is
mocked and recorded once.

## How failures behave per framework

| Framework | A mocked failure... |
| :--- | :--- |
| Plain Python | raises `MockedToolError` (or your exception) from the tool call. |
| LangGraph | becomes an error `ToolMessage` or crashes the graph, depending on `ToolNode(handle_tool_errors=...)`. |
| OpenAI Agents SDK | goes through the tool's failure policy: the default message, your `failure_error_function`, or re-raising. |
| CrewAI | is retried by CrewAI before the agent sees the error, so `handle.calls` is usually above 1. |
