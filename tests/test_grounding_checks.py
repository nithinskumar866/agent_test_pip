"""The code-only number check and the judge's second look at unsupported claims."""

import json
from pathlib import Path

import pytest

from agenttest import assert_grounded, assert_numbers_grounded, check_grounding, invented_numbers_in, use_judge
from agenttest.core.result import _clear_judge_cache
from agenttest.judge.grounding import MAX_RECHECKS
from agenttest.judge.numbers import invented_numbers, numbers_in

ORDER = ["Order 1234 shipped on 3 March and will arrive on 6 March."]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _clear_judge_cache()
    yield
    _clear_judge_cache()


def reply(*claims):
    return json.dumps({"claims": list(claims), "injection_detected": False})


def claim(text, status, quote=None):
    return {"claim": text, "status": status, "reason": "r",
            "evidence": [{"passage": 1, "quote": quote}] if quote else []}


class Judge:
    """Answers the full check with ``first``, and each second look from ``second``."""

    def __init__(self, first, second=None):
        self.first, self.second, self.prompts = first, second or {}, []

    def __call__(self, prompt, system):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return self.first
        for text, answer in self.second.items():
            if f"\n{text}\n" in prompt:
                return answer
        raise AssertionError("unexpected second look")


# --- numbers -------------------------------------------------------------------

@pytest.mark.parametrize(("text", "expected"), [
    ("Order 1,234 costs $3.50 and ships on the 3rd", ["1234", "3.5", "3"]),
    ("2023-03-06", ["2023", "3", "6"]),
    ("Tracking code GLK1321.", ["1321"]),
    ("1. Check the order\n2) Call us\nWe open at 9.", ["9"]),
    ("no numbers here", []),
])
def test_numbers_are_compared_by_value(text, expected):
    assert numbers_in(text) == expected


def test_invented_numbers_ignores_rewording_but_catches_new_numbers():
    assert invented_numbers("Your order 1234 arrives on March 6th.", ORDER) == []
    assert invented_numbers("Tracking number 99812, 20% off, arriving 2023-03-06.", ORDER) == ["99812", "20", "2023"]
    assert invented_numbers("We reply within 24 hours.", ORDER, ignore=[24]) == []


def test_assert_numbers_grounded_uses_the_run_tool_outputs_by_default():
    assert assert_numbers_grounded("Order 1234 arrives on 6 March.", ORDER) == ["1234", "6"]
    with pytest.raises(AssertionError, match="99812"):
        assert_numbers_grounded("Your tracking number is 99812.", ORDER)
    assert invented_numbers_in("Arrives on 7 March.", ORDER) == ["7"]


def test_assert_grounded_fails_on_an_invented_number_without_asking_the_judge():
    judge = Judge(reply())
    with use_judge(judge), pytest.raises(AssertionError, match=r"99812 \(checked in code; the judge was not asked\)"):
        assert_grounded("Order 1234 arrives on 6 March. Tracking number: 99812.", ORDER)
    assert judge.prompts == []


def test_number_check_is_part_of_the_report():
    with use_judge(Judge(reply(claim("x", "supported", "Order 1234")))):
        report = check_grounding("Order 1234, tracking 777.", ORDER, recheck=False)
    assert report.invented_numbers == ["777"]
    assert "[invented numbers] 777" in report.describe()


# --- second look ----------------------------------------------------------------

SHIPPED = "your order 1234 was shipped on March 3rd"


def test_second_look_confirms_a_claim_the_judge_missed():
    judge = Judge(reply(claim(SHIPPED, "unsupported")),
                  {SHIPPED: reply(claim(SHIPPED, "supported", "Order 1234 shipped on 3 March"))})
    with use_judge(judge):
        report = assert_grounded("Your order 1234 was shipped on March 3rd.", ORDER)
    verdict = report.claims[0]
    assert verdict.status == "supported" and verdict.rechecked
    assert verdict.evidence[0].quote == "Order 1234 shipped on 3 March"
    assert "confirmed on a second look" in report.describe()
    assert len(judge.prompts) == 2


@pytest.mark.parametrize("second", [
    reply(claim(SHIPPED, "unsupported")),                                   # judge still unsure
    reply(claim(SHIPPED, "supported", "Order 1234 shipped on 4 March")),   # quote not in the context
    reply(claim(SHIPPED, "supported", "Order 1234 shipped")),              # quote lacks the claim's "3"
    reply(),                                                                # no claims at all
])
def test_second_look_upgrades_only_with_verified_quotes_holding_every_number(second):
    with use_judge(Judge(reply(claim(SHIPPED, "unsupported")), {SHIPPED: second})):
        report = check_grounding("Your order 1234 was shipped on March 3rd.", ORDER)
    assert report.claims[0].status == "unsupported" and not report.claims[0].rechecked


def test_second_look_can_be_switched_off_in_the_config():
    Path("agenttest.config.json").write_text('{"judge_recheck": false}')
    judge = Judge(reply(claim(SHIPPED, "unsupported")))
    with use_judge(judge):
        report = check_grounding("Your order 1234 was shipped on March 3rd.", ORDER)
    assert report.claims[0].status == "unsupported" and len(judge.prompts) == 1


def test_second_looks_are_capped():
    claims = [claim(f"claim number {i}", "unsupported") for i in range(MAX_RECHECKS + 3)]
    second = {f"claim number {i}": reply(claim("c", "unsupported")) for i in range(MAX_RECHECKS + 3)}
    judge = Judge(reply(*claims), second)
    with use_judge(judge):
        check_grounding("output", ORDER)
    assert len(judge.prompts) == 1 + MAX_RECHECKS
