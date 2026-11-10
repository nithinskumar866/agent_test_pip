"""Test cases kept in a data file, so one test can cover many questions.

::

    # cases.jsonl
    {"id": "capital", "question": "Capital of France?", "expected": "Paris"}

    @agenttest.cases("cases.jsonl")
    @agent_test
    def test_answers(case):
        result = agenttest.run(my_agent, case.question)
        assert_output_contains(result, case.expected)

Each row becomes its own pytest test, named after its ``id`` field.
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from agenttest.config import AgentTestConfigError

__all__ = ["Case", "load_cases", "cases"]


class Case(dict):  # type: ignore[type-arg]
    """One row of a dataset. Fields work as keys or attributes: ``case["q"]`` or ``case.q``."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"This case has no field {name!r}; it has: {', '.join(self)}") from None


def _read(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as exc:
                    raise AgentTestConfigError(f"{path}:{number}: invalid JSON: {exc}") from exc
        return rows
    if suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise AgentTestConfigError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise AgentTestConfigError(f"{path}: expected a JSON list of cases")
        return data
    if suffix == ".csv":
        return list(csv.DictReader(text.splitlines()))
    raise AgentTestConfigError(f"{path}: unsupported dataset format; use .jsonl, .json or .csv")


def load_cases(source: str | Path, *, base: Path | None = None) -> list[Case]:
    """Read a ``.jsonl``, ``.json`` or ``.csv`` dataset into a list of :class:`Case`.

    A relative path is resolved against ``base`` (the current directory by default).
    """
    path = Path(source)
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    if not path.exists():
        raise AgentTestConfigError(f"Dataset not found: {path}")
    rows = _read(path)
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AgentTestConfigError(f"{path}: case {index} is not an object")
    return [Case(row) for row in rows]


def cases(
    source: str | Path | Iterable[dict[str, Any]], *, name: str = "case", id_field: str = "id"
) -> Callable[..., Any]:
    """Decorator: run the test once per case, passing the case as the ``case`` argument.

    A relative path is resolved against the file containing the test.

    Args:
        source: A dataset path, or the cases themselves.
        name: The test argument that receives each case.
        id_field: The field used to name each test.
    """
    if isinstance(source, (str, Path)):
        rows = load_cases(source, base=Path(sys._getframe(1).f_code.co_filename).resolve().parent)
    else:
        rows = [Case(row) for row in source]
    ids = [str(row.get(id_field, f"case{index}")) for index, row in enumerate(rows, start=1)]
    return pytest.mark.parametrize(name, rows, ids=ids)
