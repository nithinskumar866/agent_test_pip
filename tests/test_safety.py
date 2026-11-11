"""Safety checks: personal data leaks, prompt injection, refusals."""

import json

import pytest

import agenttest
from agenttest import (
    CANARY,
    assert_no_pii_leak,
    assert_refuses,
    assert_resists_injection,
    find_pii,
    injection_attacks,
    use_judge,
)
from agenttest.core.result import _clear_judge_cache
from agenttest.safety import followed_injection


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_judge_cache()
    yield
    _clear_judge_cache()


# --- personal data ---------------------------------------------------------------

@pytest.mark.parametrize(("text", "kind", "found"), [
    ("write to priya.s@example.co.in today", "email", "priya.s@example.co.in"),
    ("call +91 98765 43210", "phone", "+91 98765 43210"),
    ("call 9876543210", "phone", "9876543210"),
    ("call (044) 2345-6789", "phone", "(044) 2345-6789"),
    ("card 4111-1111-1111-1111", "card", "4111-1111-1111-1111"),
    ("from 192.168.1.20", "ip", "192.168.1.20"),
    ("key sk-proj-abcdefghijklmnop", "secret", "sk-proj-abcdefghijklmnop"),
])
def test_finds_personal_data(text, kind, found):
    assert [(f.kind, f.text) for f in find_pii(text)] == [(kind, found)]


@pytest.mark.parametrize("text", [
    "Order 1234 shipped on 2023-03-06 to 1,000,000 customers.",
    "pi is 3.14159265 and we run version 1.2.3",
    "card-like but fails Luhn: 4111 1111 1111 1112",
    "Rated 4.5 by 120 users in 2024",
    "999.1.1.1 is not an address",
])
def test_ignores_numbers_that_are_not_personal_data(text):
    assert find_pii(text) == []


def test_a_card_number_is_not_also_reported_as_a_phone_number():
    assert [f.kind for f in find_pii("4111111111111111")] == ["card"]


def test_assert_no_pii_leak_allows_listed_values_however_they_are_written():
    text = "Call our support line on +1 (800) 555-0199 or email help@shop.example."
    allowed = assert_no_pii_leak(text, allow=["+18005550199", "HELP@shop.example"])
    assert [f.kind for f in allowed] == ["phone", "email"]
    with pytest.raises(AssertionError, match=r"found 1: email: 'help@shop.example'"):
        assert_no_pii_leak(text, allow=["+18005550199"])


def test_assert_no_pii_leak_can_check_some_kinds_only():
    assert_no_pii_leak("ask jo@x.example", kinds=["phone"])
    with pytest.raises(ValueError, match="unknown PII kind"):
        find_pii("x", kinds=["passport"])


def test_assert_no_pii_leak_reads_an_agent_result():
    result = agenttest.run(lambda q: "The customer's email is jo@x.example", "who?")
    with pytest.raises(AssertionError, match="jo@x.example"):
        assert_no_pii_leak(result)


# --- prompt injection --------------------------------------------------------------

def gullible(message):
    """Does whatever the last instruction says. A model of a badly defended agent."""
    return f"Sure! {CANARY}" if CANARY in message else "Your order arrives on 6 March."


def careful(message):
    return "I can only help with questions about your orders."


@agenttest.tool
def lookup_order(order_id: str) -> str:
    return "Order 1234 arrives on 6 March."


def obeys_tool_output(message):
    """Treats whatever the tool returned as instructions."""
    data = lookup_order("1234")
    return CANARY if CANARY in data else f"Here is what I found: {data}"


def quotes_tool_output(message):
    return f"Here is what I found: {lookup_order('1234')}"


def summarises_tool_output(message):
    data = lookup_order("1234")
    return "Your order arrives on 6 March." if "6 March" in data else "I couldn't find your order."


def test_the_attack_pack():
    attacks = injection_attacks()
    assert len(attacks) == len({a.id for a in attacks}) >= 10
    assert all(CANARY in a.text for a in attacks)
    assert {a.kind for a in attacks} == {"direct", "indirect"}
    assert injection_attacks("indirect") and all(a.kind == "indirect" for a in injection_attacks("indirect"))
    with pytest.raises(ValueError):
        injection_attacks("sideways")


def test_a_gullible_agent_fails_every_direct_attack():
    with pytest.raises(AssertionError, match=r"followed 8 of 8 prompt-injection attack\(s\)") as exc:
        assert_resists_injection(gullible)
    assert "ignore-instructions (direct): canary in output" in str(exc.value)


def test_a_careful_agent_resists_and_gets_the_outcomes_back():
    outcomes = assert_resists_injection(careful)
    assert len(outcomes) == len(injection_attacks("direct")) and all(o.resisted for o in outcomes)


def test_indirect_attacks_are_planted_in_a_tool_response():
    with pytest.raises(AssertionError, match="poisoned-data-note \\(indirect\\)"):
        assert_resists_injection(obeys_tool_output, "where is my order?", through_tool="lookup_order")
    assert_resists_injection(summarises_tool_output, "where is my order?", through_tool="lookup_order")
    # Quoting the poisoned record word for word passes it on, but does not obey it.
    assert_resists_injection(quotes_tool_output, "where is my order?", through_tool="lookup_order")


def test_handing_the_users_message_to_a_tool_is_not_obeying_it():
    @agenttest.tool
    def search(query: str) -> str:
        return "no results"

    def passes_message_on(message):
        search(message)
        return f'No results for "{message}".'

    for multiline in [a for a in injection_attacks("direct") if "\n" in a.text]:
        assert followed_injection(agenttest.run(passes_message_on, multiline.text), multiline) == []

    attack = injection_attacks()[0]
    result = agenttest.run(passes_message_on, attack.text)
    assert followed_injection(result, attack) == []
    # Without being told the attack, the canary in the copies can't be told apart from obeying.
    assert followed_injection(result) == ["output", "tool call search"]


def test_the_canary_reaching_a_tool_call_counts_as_followed():
    def exfiltrates(message):
        lookup_order(CANARY if CANARY in message else "1234")
        return "Done."

    result = agenttest.run(exfiltrates, injection_attacks()[0].text)
    assert followed_injection(result) == ["tool call lookup_order"]


def test_attacks_can_be_wrapped_for_agents_that_take_more_than_a_string():
    seen = []

    def chat_agent(payload):
        seen.append(payload)
        return "no"

    assert_resists_injection(chat_agent, attacks=injection_attacks()[:2],
                             message=lambda a: {"role": "user", "content": a.text})
    assert seen[0] == {"role": "user", "content": injection_attacks()[0].text}


def test_bad_arguments():
    with pytest.raises(ValueError, match="needs a question"):
        assert_resists_injection(careful, through_tool="lookup_order")
    with pytest.raises(ValueError, match="no attacks"):
        assert_resists_injection(careful, attacks=[])


# --- refusals ------------------------------------------------------------------------

def test_assert_refuses_asks_the_judge_about_the_request():
    prompts = []

    def judge(prompt, system):
        prompts.append(prompt)
        return "YES" if "can't help" in prompt else "NO"

    with use_judge(judge):
        assert assert_refuses("Sorry, I can't help with that.", "rank candidates by age")
        with pytest.raises(AssertionError, match="Expected the agent to refuse 'rank candidates by age'"):
            assert_refuses("Here they are, youngest first: ...", "rank candidates by age")
        assert agenttest.assertions.refuses("Sorry, I can't help with that.") is True
    assert "to this request: 'rank candidates by age'" in prompts[0]


def test_safety_report_is_json_friendly():
    outcomes = assert_resists_injection(careful, attacks=injection_attacks()[:1])
    assert json.dumps([{"id": o.attack.id, "resisted": o.resisted} for o in outcomes])
