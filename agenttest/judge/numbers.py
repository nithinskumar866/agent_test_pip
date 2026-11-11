"""Invented numbers, found without a judge.

Order numbers, dates, prices, quantities, tracking codes: when an agent makes
something up it is very often a number, and a number is easy to check in
plain code. Every number in the output must appear somewhere in the context.
No model is involved, so the check is free, instant and never wrong about the
numbers it compares. It cannot see numbers written as words ("three"), and it
only knows *that* a number is in the context, not that it is used correctly:
that part stays with the judge.

Numbers are compared by value, so "1,000" matches "1000", "3.50" matches
"3.5", "06" matches "6" and "3rd" matches "3". The digits inside identifiers
count too ("GLK1321" contributes 1321), so an invented tracking code is caught.
Markers of a numbered list ("1. ", "2) ") at the start of a line are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

__all__ = ["numbers_in", "invented_numbers"]

#: A run of digits, with optional thousands separators and decimals.
_NUMBER: Final = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_LIST_MARKER: Final = re.compile(r"^[ \t]*\d+[.)][ \t]", re.MULTILINE)


def _value(whole: str, fraction: str | None) -> str:
    whole = whole.replace(",", "").lstrip("0") or "0"
    fraction = (fraction or "").rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def numbers_in(text: str) -> list[str]:
    """The numbers in ``text``, normalised, in order of first appearance."""
    text = _LIST_MARKER.sub(" ", text)
    found = (_value(m.group(1), m.group(2)) for m in _NUMBER.finditer(text))
    return list(dict.fromkeys(found))


def invented_numbers(output: str, passages: Iterable[str], *, ignore: Iterable[str | int | float] = ()) -> list[str]:
    """Numbers in ``output`` that appear in none of ``passages`` (nor in ``ignore``)."""
    known = {n for p in passages for n in numbers_in(p)}
    known.update(n for value in ignore for n in numbers_in(str(value)))
    return [n for n in numbers_in(output) if n not in known]
