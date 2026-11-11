# Multi-turn conversations

`agenttest.converse` holds a whole conversation with an agent. The result works with the same
assertions as a single run.

```python
chat = agenttest.converse(agent, ["Hi, I need a refund", "Order 42", "Thanks"])

assert chat.success, chat.error
assert_tool_called_with(chat, "refund", order_id=42)
assert "refund" in chat.output     # the agent's last reply
print(chat.transcript)             # every message
print(chat.turns[0].tools_called)  # each turn is an AgentResult
```

The conversation stops when the messages run out, when a turn fails, or at `max_turns`. In async
tests, use `await agenttest.aconverse(...)`.

## How the history reaches the agent

| Agent | Each turn receives |
| :--- | :--- |
| Plain function | The whole history: a list of `{"role": ..., "content": ...}` |
| LangGraph | `{"messages": [(role, content), ...]}` |
| OpenAI Agents SDK | The SDK's own input list from the previous turn, plus the new message |
| Pydantic AI | The new message, with `message_history=` |
| LlamaIndex | The new message, with `chat_history=` |
| CrewAI | `inputs={"message": ..., "history": ...}`: use `{message}` and `{history}` in your task text |

For anything else, subclass `BaseAdapter` and override `turn_arguments`.

## A user that reacts

Pass a function instead of a list. It receives the history and returns the next message, or `None`
to stop.

```python
def user(history):
    return "Order 42" if "order number" in history[-1]["content"] else None
```

## A simulated user

`simulated_user` has the judge model play a person:

```python
user = agenttest.simulated_user("an impatient customer", "get a refund for order 42", max_turns=6)
chat = agenttest.converse(agent, user)
```

Its messages vary from run to run, and every message is a judge call. Wrap the test in a
[cassette](replay.md) to make it repeatable.
