"""agenttest quickstart: a plain-Python agent, tested without any API key.

Run it with:

    pytest examples/test_quickstart.py

The agent below answers questions with a search tool. The tests show the main
features: running an agent, structural assertions, tool mocking, retries, and
a grounded hallucination check (with a scripted judge, so no key is needed).
"""

import agenttest
from agenttest import (
    agent_test,
    assert_grounded,
    assert_latency_under,
    assert_output_contains,
    assert_tool_was_called,
    mock_tool_failure,
    mock_tool_response,
    use_judge,
)

# --- the agent under test -------------------------------------------------------


@agenttest.tool  # traced, and mockable by name
def search(query: str) -> str:
    return "The Eiffel Tower is 330 metres tall and was completed in 1889."


def tower_agent(question: str) -> str:
    try:
        facts = search(question)
    except Exception:
        return "My search tool is unavailable, so I can't answer that reliably right now."
    return f"Here is what I found: {facts}"


# --- tests ----------------------------------------------------------------------


@agent_test(timeout=10)
def test_agent_answers_using_search():
    result = agenttest.run(tower_agent, "How tall is the Eiffel Tower?")

    assert result.success, result.traceback
    assert_tool_was_called(result, "search")
    assert_output_contains(result, "330 metres")
    assert_latency_under(result, 1000)


@agent_test
def test_agent_degrades_gracefully_when_search_is_down():
    with mock_tool_failure("search", "503 Service Unavailable") as handle:
        result = agenttest.run(tower_agent, "How tall is the Eiffel Tower?")

    assert handle.calls == 1
    assert result.gracefully_degraded  # a tool failed, yet the agent still answered


@agent_test
def test_agent_passes_on_whatever_search_returns():
    with mock_tool_response("search", "No results."):
        result = agenttest.run(tower_agent, "Who built the tower?")

    assert_output_contains(result, "No results.")


@agent_test(runs=3, min_pass_rate=1.0)
def test_agent_is_consistent():
    result = agenttest.run(tower_agent, "How tall is the Eiffel Tower?")
    assert_output_contains(result, "330")


# A scripted judge reply, in the format the grounding check asks for.
# Delete `use_judge(...)` to ask your configured judge model instead.
GROUNDED = (
    '{"claims": ['
    '{"claim": "The tower is 330 metres tall", "status": "supported",'
    ' "evidence": [{"passage": 1, "quote": "330 metres tall"}]},'
    '{"claim": "It was completed in 1889", "status": "supported",'
    ' "evidence": [{"passage": 1, "quote": "completed in 1889"}]}'
    '], "injection_detected": false}'
)


@agent_test
def test_answer_is_grounded_in_what_search_returned():
    result = agenttest.run(tower_agent, "Tell me about the Eiffel Tower.")

    with use_judge(lambda prompt, system: GROUNDED):
        report = assert_grounded(result)  # context = the run's tool outputs

    assert report.score == 0.0
