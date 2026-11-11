"""``agenttest report`` — summarise a saved ``--agenttest-report`` JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from agenttest.reporting import compare_reports, render_comparison, render_html, render_markdown

__all__ = ["report", "compare"]


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--html", "html_path", type=click.Path(dir_okay=False, path_type=Path),
              help="Also render the report as a self-contained HTML page.")
@click.option("--failed", is_flag=True, help="List only failed tests.")
@click.option("--markdown", "markdown_path", type=click.Path(dir_okay=False, path_type=Path),
              help="Also write a markdown summary, for a pull request comment.")
def report(path: Path, html_path: Path | None, failed: bool, markdown_path: Path | None) -> None:
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
    if markdown_path:
        markdown_path.write_text(render_markdown(data), encoding="utf-8")
        console.print(f"Markdown written: {markdown_path}")


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["summary"], data["tests"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise click.ClickException(f"{path} is not an agenttest report: {exc}") from None
    return data


@click.command()
@click.argument("baseline", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("current", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-token-increase", type=float, help="Percent growth in tokens that counts as a regression.")
@click.option("--max-cost-increase", type=float, help="Percent growth in cost that counts as a regression.")
@click.option("--markdown", "markdown_path", type=click.Path(dir_okay=False, path_type=Path),
              help="Write the report and this comparison as markdown.")
def compare(
    baseline: Path,
    current: Path,
    max_token_increase: float | None,
    max_cost_increase: float | None,
    markdown_path: Path | None,
) -> None:
    """Compare two JSON reports. Exits 1 if anything regressed."""
    old, new = _load(baseline), _load(current)
    result = compare_reports(old, new, max_token_increase=max_token_increase, max_cost_increase=max_cost_increase)
    Console(highlight=False, markup=False).print(render_comparison(result))
    if markdown_path:
        markdown_path.write_text(render_markdown(new, result), encoding="utf-8")
    if result["regressions"]:
        raise SystemExit(1)
