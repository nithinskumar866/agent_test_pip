"""``agenttest doctor`` — check that a project is ready to run agent tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from agenttest.cli._console import symbols
from agenttest.config import CONFIG_FILENAME, AgentTestConfigError, load_config

__all__ = ["doctor", "run_checks", "Check"]

_FRAMEWORK_PACKAGES = {
    "langgraph": "langgraph",
    "crewai": "crewai",
    "openai": "agents",
    "anthropic": "anthropic",
}
_JUDGE_REQUIREMENTS = {
    "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
    "openai": ("openai", "OPENAI_API_KEY"),
    "gemini": ("google.generativeai", "GOOGLE_API_KEY"),
    "ollama": ("httpx", None),
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def run_checks(directory: Path | None = None) -> list[Check]:
    """Inspect the environment and ``agenttest.config.json`` in ``directory``."""
    from agenttest import __version__

    directory = directory or Path.cwd()
    checks = [Check("agenttest", "ok", __version__)]
    version = sys.version_info
    checks.append(Check("python", "ok" if version >= (3, 10) else "fail",
                        f"{version.major}.{version.minor}.{version.micro}"
                        + ("" if version >= (3, 10) else " (agenttest needs 3.10+)")))

    path = directory / CONFIG_FILENAME
    try:
        config = load_config(path)
    except AgentTestConfigError as exc:
        checks.append(Check("config", "fail", str(exc).splitlines()[0]))
        return checks
    checks.append(Check("config", "ok" if path.exists() else "warn",
                        str(path.name) if path.exists() else "no config file; defaults apply (run `agenttest init`)"))

    framework = config.framework
    package = _FRAMEWORK_PACKAGES.get(framework)
    if package is None:
        checks.append(Check("framework", "ok", f"{framework} (plain Python agents)"))
    else:
        found = _installed(package)
        checks.append(Check("framework", "ok" if found else "fail",
                            f"{framework}" + ("" if found else f" - package '{package}' is not installed")))

    installed = [name for name, pkg in _FRAMEWORK_PACKAGES.items() if _installed(pkg)]
    checks.append(Check("installed SDKs", "ok", ", ".join(installed) or "none"))

    provider = config.judge_provider
    if provider == "none":
        checks.append(Check("judge", "warn", "disabled - semantic assertions will raise; structural ones work"))
    else:
        module, env_var = _JUDGE_REQUIREMENTS[provider]
        problems = []
        if not _installed(module):
            problems.append(f"package '{module}' is not installed")
        if env_var and not os.getenv(env_var):
            problems.append(f"{env_var} is not set")
        model = config.judge_model or "provider default"
        checks.append(Check("judge", "fail" if problems else "ok",
                            f"{provider} / {model}" + (f" - {'; '.join(problems)}" if problems else "")))
    return checks


_COLOURS = {"ok": "green", "warn": "yellow", "fail": "red"}


@click.command()
@click.option("--ping-judge", is_flag=True, help="Also send one tiny request to the judge (costs a call).")
def doctor(ping_judge: bool) -> None:
    """Check config, judge credentials and installed frameworks."""
    console = Console()
    checks = run_checks()
    if ping_judge and not any(c.name == "judge" and c.status == "fail" for c in checks):
        from agenttest.core.result import _judge_llm

        try:
            reply = _judge_llm("Reply with the single word OK.")
            checks.append(Check("judge ping", "ok", f"replied {reply[:40]!r}"))
        except Exception as exc:  # report every failure; this is a diagnostic
            checks.append(Check("judge ping", "fail", f"{type(exc).__name__}: {exc}"))

    marks = symbols()
    table = Table(show_header=False, box=None, padding=(0, 1))
    for check in checks:
        colour = _COLOURS[check.status]
        table.add_row(f"[{colour}]{marks[check.status]}[/{colour}]", check.name, check.detail)
    console.print(table)
    if any(c.status == "fail" for c in checks):
        console.print("\n[red]Some checks failed.[/red]")
        raise SystemExit(1)
    console.print("\n[green]Ready to run agent tests.[/green]")
