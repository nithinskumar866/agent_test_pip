"""Guards on the packaging metadata in pyproject.toml."""

import re
from pathlib import Path

import pytest

import agenttest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_distribution_name_and_entry_points():
    assert PROJECT["name"] == "pytest-agenttest"
    assert PROJECT["scripts"] == {"agenttest": "agenttest.cli.init:main"}
    assert PROJECT["entry-points"]["pytest11"] == {"agenttest": "agenttest.plugin"}


def test_version_is_single_sourced_from_the_package():
    assert PROJECT["dynamic"] == ["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+([ab]\d+|rc\d+)?", agenttest.__version__)


def test_core_dependencies_stay_minimal():
    names = {re.split(r"[<>=!~\[ ]", dep, maxsplit=1)[0] for dep in PROJECT["dependencies"]}
    assert names == {"pytest", "pydantic", "click", "rich"}


def test_every_install_hint_names_a_real_extra():
    extras = set(PROJECT["optional-dependencies"])
    hinted = set()
    for path in (ROOT / "agenttest").rglob("*.py"):
        hinted |= set(re.findall(r"pytest-agenttest\[([a-z-]+)\]", path.read_text(encoding="utf-8")))
    assert hinted and hinted <= extras


def test_license_and_typed_marker_ship():
    assert (ROOT / "LICENSE").exists()
    assert (ROOT / "agenttest" / "py.typed").exists()
