"""``agenttest report`` — summarise a saved ``--agenttest-report`` JSON file."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from agenttest.reporting import render_html

__all__ = ["report"]


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--html", "html_path", type=click.Path(dir_okay=False, path_type=Path),
              help="Also render the report as a self-contained HTML page.")
@click.option("--failed", is_flag=True, help="List only failed tests.")
def report(path: Path, html_path: Path | None, failed: bool) -> None:
    """Show a JSON report written by `pytest --agenttest-report=PATH`."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary, tests = data["summary"], data["tests"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise click.ClickException(f"{path} is not an agenttest report: {exc}") from None

    console = Console()
    console.print(
        f"[bold]{summary['tests']} tests[/bold]: [green]{summary['passed']} passed[/green], "
        f"[red]{summary['failed']} failed[/red], {summary['skipped']} skipped · "
        f"{summary['agent_runs']} agent runs · {summary['tokens_used']} tokens · "
        f"{summary['failed_tool_calls']} failed tool calls · {summary['flaky']} flaky"
    )
    table = Table("outcome", "test", "runs", "tokens", "tools")
    for test in tests:
        if failed and test["outcome"] != "failed":
            continue
        colour = {"passed": "green", "failed": "red"}.get(test["outcome"], "yellow")
        tools = sorted({t for run in test["runs"] for t in run["tools_called"]})
        table.add_row(f"[{colour}]{test['outcome']}[/{colour}]", test["nodeid"], str(len(test["runs"])),
                      str(sum(run["tokens_used"] for run in test["runs"])), ", ".join(tools) or "-")
    console.print(table)
    for item in data.get("flaky", []):
        console.print(f"[yellow]flaky[/yellow] {item['test']}: {item['summary']}")
    if html_path:
        html_path.write_text(render_html(data), encoding="utf-8")
        console.print(f"HTML report written: {html_path}")
