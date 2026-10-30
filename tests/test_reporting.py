"""Tests for run recording and the text/JSON/HTML renderers."""

import json
import threading

import agenttest
from agenttest import reporting
from agenttest.core.result import AgentResult
from agenttest.core.trace import Step, Trace


def _result(**overrides):
    trace = Trace(steps=[
        Step(kind="llm", name="model", input_tokens=10, output_tokens=4, tool_requests=["search"]),
        Step(kind="tool", name="search", output="results " * 40),
        Step(kind="tool", name="db", error="TimeoutError: slow", metadata={"mocked": "failure"}),
    ])
    fields = dict(output="final answer", latency_ms=12.5)
    fields.update(overrides)
    return AgentResult.from_trace(trace, **fields)


def test_runs_are_recorded_only_while_a_test_is_active():
    # If this suite itself runs with the plugin, a recording is already active: pause it.
    reporting.end_test()
    try:
        agenttest.run(lambda: "outside")
        assert reporting.current_runs() == []

        reporting.begin_test()
        agenttest.run(lambda: "inside")
        thread = threading.Thread(target=lambda: agenttest.run(lambda: "from a thread"))
        thread.start()
        thread.join()
        runs = reporting.end_test()

        assert [r.output for r in runs] == ["inside", "from a thread"]
        assert reporting.current_runs() == []
    finally:
        reporting.begin_test()


def test_describe_result_lists_steps_errors_and_mocks():
    text = reporting.describe_result(_result(), "run 1 of 1")
    lines = text.splitlines()
    assert lines[0] == "run 1 of 1: ok - 12 ms, 3 steps, 14 tokens, gracefully degraded"
    assert "1. llm  model  (tokens 10+4; requested search)" in text
    assert "returned: results results" in text and "..." in text  # long output shortened
    assert "3. tool db  (mocked failure)" in text
    assert "error: TimeoutError: slow" in text
    assert lines[-1] == "  output: 'final answer'"
    assert text.isascii()


def test_describe_failed_run():
    failed = AgentResult.from_trace(Trace(), output="", latency_ms=1.0, exception=RuntimeError("down"))
    text = reporting.describe_result(failed)
    assert "FAILED (RuntimeError: down)" in text and "output: (empty)" in text


def test_result_to_dict_is_json_safe_for_framework_objects():
    class Message:
        content = "hello"

    class Opaque:
        def __repr__(self):
            return "<Opaque>"

    trace = Trace(steps=[Step(kind="node", name="agent",
                              output={"messages": [Message()], "obj": Opaque(), "long": "x" * 5000})])
    data = reporting.result_to_dict(AgentResult.from_trace(trace, output="o", latency_ms=1.0))
    json.dumps(data)  # must not raise
    output = data["steps"][0]["output"]
    assert output["messages"] == [{"type": "Message", "content": "hello"}]
    assert output["obj"] == "<Opaque>"
    assert len(output["long"]) <= 2001


def test_build_report_and_html_escape_untrusted_text():
    hostile = _result(output="<script>alert(1)</script>")
    tests = [reporting.TestRecord("t.py::test_<b>x</b>", outcome="failed", agent_marked=True,
                                  message="<img src=x onerror=alert(1)>", runs=[hostile])]
    report = reporting.build_report(tests, [], duration_s=1.5)

    assert report["summary"]["failed"] == 1 and report["summary"]["tokens_used"] == 14
    assert report["summary"]["failed_tool_calls"] == 1
    page = reporting.render_html(report)
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "<img src=x" not in page
    assert "test_&lt;b&gt;x&lt;/b&gt;" in page
