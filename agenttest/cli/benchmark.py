"""``agenttest benchmark`` - measure the configured judge against labelled cases."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from agenttest.config import AgentTestConfigError
from agenttest.judge.benchmark import KINDS, run_benchmark

__all__ = ["benchmark"]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


@click.command()
@click.option("--limit", type=click.IntRange(min=1, max=20), help="Use only this many facts (five cases each).")
@click.option("--json", "json_path", type=click.Path(dir_okay=False, path_type=Path), help="Write full results here.")
@click.option("--min-accuracy", type=click.FloatRange(0, 1), help="Exit 1 if accuracy is below this (0 to 1).")
def benchmark(limit: int | None, json_path: Path | None, min_accuracy: float | None) -> None:
    """Score the configured judge on labelled cases. This makes real judge calls, which cost money."""
    console = Console(highlight=False)
    try:
        result = run_benchmark(limit)
    except AgentTestConfigError as exc:
        raise click.ClickException(str(exc)) from None

    answered, total = result["answered"], result["cases"]
    console.print(f"Accuracy: [bold]{_pct(result['accuracy'])}[/bold] on {answered} of {total} cases")
    for kind in KINDS:
        console.print(f"  {kind:<13} {_pct(result['per_kind'][kind])}")
    console.print(f"False rejects (good answers failed): {_pct(result['false_reject_rate'])}")
    console.print(f"False accepts (bad answers passed):  {_pct(result['false_accept_rate'])}")
    if result["judge_errors"]:
        console.print(f"[yellow]{len(result['judge_errors'])} case(s) had judge errors and were not scored.[/yellow]")
    if json_path:
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if min_accuracy is not None and result["accuracy"] < min_accuracy:
        console.print(f"[red]Below the required accuracy of {_pct(min_accuracy)}.[/red]")
        raise SystemExit(1)
