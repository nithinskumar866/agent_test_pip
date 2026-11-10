"""Tests for cassettes: real SDK clients over a mocked HTTP transport, then replay."""

import asyncio
import json
from pathlib import Path

import pytest

import agenttest
from agenttest import AgentTestConfigError, CassetteMissError, agent_test, cassette
from agenttest.replay import record_mode


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("AGENTTEST_RECORD", raising=False)


def _server():
    """A scripted Anthropic server; counts calls so tests can prove replay is offline."""
    anthropic = pytest.importorskip("anthropic")
    http = pytest.importorskip("httpx2")
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        text = f"answer {len(calls)} to {body['messages'][-1]['content']}"
        return http.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "stop_reason": "end_turn",
            "stop_sequence": None, "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        })

    transport = http.MockTransport(handler)
    client = anthropic.Anthropic(api_key="sk-test-secret-value-123456", max_retries=0,
                                 http_client=http.Client(transport=transport))
    async_client = anthropic.AsyncAnthropic(api_key="sk-test-secret-value-123456", max_retries=0,
                                            http_client=http.AsyncClient(transport=transport))
    return client, async_client, calls


def ask(client, question="hello"):
    reply = client.messages.create(model="claude-haiku-5-5", max_tokens=16,
                                   messages=[{"role": "user", "content": question}])
    return reply.content[0].text


def test_once_records_then_replays_without_the_network():
    client, _, calls = _server()
    with cassette("c.json") as first:
        assert ask(client) == "answer 1 to hello"
    assert first.recorded == 1 and len(calls) == 1

    with cassette("c.json") as second:
        assert ask(client) == "answer 1 to hello"
    assert second.replayed == 1 and len(calls) == 1


def test_changed_prompt_is_a_miss_not_a_stale_answer():
    client, _, _ = _server()
    with cassette("c.json"):
        ask(client, "hello")
    with pytest.raises(CassetteMissError, match="has no recording of this request.*--agenttest-record=all"):
        with cassette("c.json"):
            ask(client, "something else")


def test_miss_is_not_swallowed_or_retried_by_the_sdk():
    client, _, calls = _server()
    with pytest.raises(CassetteMissError):  # not wrapped in anthropic.APIConnectionError
        with cassette("c.json", mode="none"):
            ask(client)
    assert calls == []


def test_mode_none_requires_the_file():
    client, _, _ = _server()
    with pytest.raises(CassetteMissError, match="does not exist"):
        with cassette("missing.json", "none"):
            ask(client)


def test_mode_new_adds_only_new_calls():
    client, _, calls = _server()
    with cassette("c.json"):
        ask(client, "a")
    with cassette("c.json", "new") as session:
        assert ask(client, "a") == "answer 1 to a"
        assert ask(client, "b") == "answer 2 to b"
    assert (session.replayed, session.recorded) == (1, 1)
    assert len(json.loads(Path("c.json").read_text(encoding="utf-8"))["interactions"]) == 2
    assert len(calls) == 2


def test_mode_all_overwrites():
    client, _, calls = _server()
    with cassette("c.json"):
        ask(client)
    with cassette("c.json", "all"):
        assert ask(client) == "answer 2 to hello"
    with cassette("c.json"):
        assert ask(client) == "answer 2 to hello"
    assert len(calls) == 2


def test_repeated_requests_replay_in_order_then_cycle():
    client, _, _ = _server()
    with cassette("c.json"):
        assert ask(client) == "answer 1 to hello"
        assert ask(client) == "answer 2 to hello"
    with cassette("c.json"):
        assert [ask(client) for _ in range(3)] == ["answer 1 to hello", "answer 2 to hello", "answer 1 to hello"]


def test_async_client_records_and_replays():
    _, client, calls = _server()

    async def go():
        reply = await client.messages.create(model="claude-haiku-5-5", max_tokens=16,
                                             messages=[{"role": "user", "content": "hi"}])
        return reply.content[0].text

    with cassette("c.json"):
        assert asyncio.run(go()) == "answer 1 to hi"
    with cassette("c.json"):
        assert asyncio.run(go()) == "answer 1 to hi"
    assert len(calls) == 1


def test_streaming_response_replays():
    anthropic = pytest.importorskip("anthropic")
    http = pytest.importorskip("httpx2")
    events = "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n"
        for name, data in [
            ("message_start", {"type": "message_start", "message": {
                "id": "m", "type": "message", "role": "assistant", "model": "x", "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "streamed"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 1}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    calls = []

    def handler(request):
        calls.append(1)
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=events.encode())

    client = anthropic.Anthropic(api_key="k", max_retries=0, http_client=http.Client(transport=http.MockTransport(handler)))

    def go():
        with client.messages.stream(model="x", max_tokens=8, messages=[{"role": "user", "content": "q"}]) as stream:
            return "".join(stream.text_stream)

    with cassette("c.json"):
        assert go() == "streamed"
    with cassette("c.json"):
        assert go() == "streamed"
    assert len(calls) == 1


def test_secrets_never_reach_the_file():
    client, _, _ = _server()
    with cassette("c.json"):
        ask(client, "my key is sk-abcdefghijklmnopqrstuvwx")
    text = Path("c.json").read_text(encoding="utf-8")
    assert "sk-test-secret-value" not in text
    assert "sk-abcdefghijklmnopqrstuvwx" not in text
    assert "x-api-key" not in text.lower()


def test_failed_block_saves_nothing():
    client, _, _ = _server()
    with pytest.raises(RuntimeError):
        with cassette("c.json"):
            ask(client)
            raise RuntimeError("test failed")
    assert not Path("c.json").exists()


def test_server_errors_are_not_recorded():
    anthropic = pytest.importorskip("anthropic")
    http = pytest.importorskip("httpx2")
    transport = http.MockTransport(lambda request: http.Response(529, json={"type": "error", "error": {"type": "x", "message": "m"}}))
    client = anthropic.Anthropic(api_key="k", max_retries=0, http_client=http.Client(transport=transport))
    with cassette("c.json") as session:
        with pytest.raises(anthropic.APIStatusError):
            ask(client)
    assert session.recorded == 0 and not Path("c.json").exists()


def test_other_hosts_are_left_alone():
    http = pytest.importorskip("httpx2")
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return http.Response(200, json={"ok": True})

    client = http.Client(transport=http.MockTransport(handler))
    with cassette("c.json", "none"):
        assert client.get("https://telemetry.example.com/ping").json() == {"ok": True}
    assert len(seen) == 1


def test_extra_hosts_are_intercepted():
    http = pytest.importorskip("httpx2")
    client = http.Client(transport=http.MockTransport(lambda r: http.Response(200, json={"n": 1})))
    with cassette("c.json", hosts=["llm.internal"]) as session:
        client.post("https://llm.internal/v1", json={"a": 1})
    assert session.recorded == 1


def test_ignored_body_keys_do_not_affect_matching():
    http = pytest.importorskip("httpx2")
    client = http.Client(transport=http.MockTransport(lambda r: http.Response(200, json={"n": 1})))
    url = "https://api.openai.com/v1/x"
    with cassette("c.json", ignore_body_keys=["request_id"]):
        client.post(url, json={"a": 1, "request_id": "first"})
    with cassette("c.json", "none", ignore_body_keys=["request_id"]) as session:
        client.post(url, json={"request_id": "second", "a": 1})
    assert session.replayed == 1


def test_query_credentials_do_not_affect_matching_or_get_stored():
    http = pytest.importorskip("httpx2")
    client = http.Client(transport=http.MockTransport(lambda r: http.Response(200, json={})))
    host = "https://generativelanguage.googleapis.com/v1/m"
    with cassette("c.json"):
        client.post(f"{host}?key=first-secret", json={"a": 1})
    with cassette("c.json", "none"):
        client.post(f"{host}?key=second-secret", json={"a": 1})
    assert "first-secret" not in Path("c.json").read_text(encoding="utf-8")


def test_nesting_is_rejected():
    with cassette("a.json"):
        with pytest.raises(AgentTestConfigError, match="can't be nested"):
            with cassette("b.json"):
                pass


def test_corrupt_cassette_is_a_clear_error():
    Path("c.json").write_text("not json")
    with pytest.raises(AgentTestConfigError, match="Cannot read cassette"):
        with cassette("c.json"):
            pass


def test_send_is_restored_afterwards():
    http = pytest.importorskip("httpx2")
    before = http.Client.send
    with cassette("c.json"):
        assert http.Client.send is not before
    assert http.Client.send is before


def test_record_mode_resolution(monkeypatch):
    assert record_mode() == "once"
    monkeypatch.setenv("CI", "true")
    assert record_mode() == "none"
    monkeypatch.setenv("AGENTTEST_RECORD", "new")
    assert record_mode() == "new"
    assert record_mode("all") == "all"
    with pytest.raises(AgentTestConfigError, match="Unknown record mode"):
        record_mode("sometimes")


def test_openai_client_records_and_replays():
    openai = pytest.importorskip("openai")
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        })

    client = openai.OpenAI(api_key="k", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    def go():
        return client.chat.completions.create(model="gpt", messages=[{"role": "user", "content": "q"}]).choices[0].message.content

    with cassette("c.json"):
        assert go() == "hi"
    with cassette("c.json"):
        assert go() == "hi"
    assert len(calls) == 1


def test_decorator_records_next_to_the_test_and_covers_retries(tmp_path):
    client, _, calls = _server()
    target = tmp_path / "suite" / "test_x.py"
    target.parent.mkdir()
    target.write_text("")

    def body():
        result = agenttest.run(lambda: ask(client))
        assert result.output.startswith("answer")

    body.__qualname__ = "test_thing"
    import inspect

    original = inspect.getfile
    inspect.getfile = lambda obj: str(target) if obj is body else original(obj)
    try:
        wrapped = agent_test(cassette=True, runs=2)(body)
        wrapped()
        wrapped()
    finally:
        inspect.getfile = original
    assert (tmp_path / "suite" / "cassettes" / "test_x" / "test_thing.json").exists()
    assert len(calls) == 2  # both runs of the first pass were recorded; the second pass replayed
