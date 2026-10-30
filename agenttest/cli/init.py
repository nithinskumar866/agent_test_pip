"""``agenttest init`` — framework and judge provider detection and config generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from agenttest import __version__
from agenttest.cli._console import make_safe, symbols

console = Console()

__all__ = ["main", "init", "_detect_framework", "_detect_judge_provider"]


@click.group()
@click.version_option(version=__version__, message="agenttest %(version)s")
def main() -> None:
    """agenttest - behavioural testing for AI agents."""
    make_safe()


@main.command()
@click.option(
    "--framework",
    default=None,
    help="Force framework: langgraph/crewai/openai/anthropic/generic",
)
@click.option(
    "--judge",
    default=None,
    help="Judge provider: anthropic/openai/gemini/ollama/none",
)
@click.option("--force", is_flag=True, help="Overwrite an existing agenttest.config.json.")
@click.option("--example", is_flag=True, help="Also write tests/test_agent_example.py, a runnable starting point.")
def init(framework: str | None, judge: str | None, force: bool, example: bool) -> None:
    """Initialize agenttest in the current project."""
    console.print(
        Panel(
            f"[bold cyan]agenttest[/bold cyan] {symbols()['dash']} AI agent testing framework",
            subtitle=f"v{__version__} initialization",
            border_style="cyan",
        )
    )

    config_path = Path.cwd() / "agenttest.config.json"
    if config_path.exists() and not force:
        console.print("[yellow]agenttest.config.json already exists.[/yellow] Use --force to overwrite it.")
        raise SystemExit(1)

    # Framework detection
    detected_framework = framework or _detect_framework()
    if not detected_framework:
        console.print(
            "[yellow]Could not auto-detect framework.[/yellow]\n"
            "Use --framework to specify one of: "
            "langgraph, crewai, openai, anthropic, generic"
        )
        raise click.Abort()

    # Judge provider detection
    detected_judge = judge or _detect_judge_provider()

    # Judge model defaults per provider
    judge_model_defaults = {
        "anthropic": "claude-haiku-5-5",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "ollama": "llama3",
        "none": "none",
    }
    judge_model = judge_model_defaults.get(detected_judge, "none")

    config = {
        "framework": detected_framework,
        "judge_provider": detected_judge,
        "judge_model": judge_model,
        "timeout": 60,
        "max_retries": 0,
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    example_path = _write_example() if example else None

    # Rich output table
    from rich.table import Table

    tick = symbols()["ok"]
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="green")
    table.add_column(style="bold white")
    table.add_row(f"{tick} Framework detected", detected_framework)
    table.add_row(f"{tick} Judge provider", detected_judge)
    table.add_row(f"{tick} Judge model", judge_model)
    table.add_row(f"{tick} Config written", "agenttest.config.json")
    if example_path is not None:
        table.add_row(f"{tick} Example test", str(example_path))
    console.print(table)

    # Show next steps based on judge provider
    env_var_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "ollama": None,
        "none": None,
    }
    env_var = env_var_map.get(detected_judge)

    console.print("\n[bold]Next steps:[/bold]")
    if env_var:
        console.print(
            f"  1. Set your judge API key:\n"
            f"     [cyan]export {env_var}=your_key_here[/cyan]"
        )
        console.print("  2. Write tests using [bold]@agent_test[/bold]")
        console.print("  3. Run [bold cyan]pytest[/bold cyan]")
    elif detected_judge == "ollama":
        console.print(
            "  1. Make sure Ollama is running:\n     [cyan]ollama serve[/cyan]"
        )
        console.print("  2. Pull a model:\n     [cyan]ollama pull llama3[/cyan]")
        console.print("  3. Write tests using [bold]@agent_test[/bold]")
        console.print("  4. Run [bold cyan]pytest[/bold cyan]")
    else:
        console.print("  Judge disabled - use structural assertions only")
        console.print("  Write tests using [bold]@agent_test[/bold]")
        console.print("  Run [bold cyan]pytest[/bold cyan]")


_EXAMPLE = '''"""A starting point for agent tests (written by `agenttest init --example`).

Runs without any API key: the agent is plain Python, and the one judge-backed
check uses a scripted judge. Replace `answer_question` with your own agent.
"""

import agenttest
from agenttest import agent_test, assert_tool_was_called, mock_tool_failure


@agenttest.tool
def search(query: str) -> str:
    return "Paris is the capital of France."


def answer_question(question: str) -> str:
    try:
        return f"According to my search: {search(question)}"
    except Exception:
        return "Search is unavailable right now, so I cannot answer reliably."


@agent_test
def test_agent_uses_search():
    result = agenttest.run(answer_question, "What is the capital of France?")
    assert result.success, result.error
    assert_tool_was_called(result, "search")


@agent_test
def test_agent_degrades_gracefully_when_search_fails():
    with mock_tool_failure("search", "503 Service Unavailable"):
        result = agenttest.run(answer_question, "What is the capital of France?")
    assert result.gracefully_degraded


@agent_test
def test_answer_is_grounded_in_the_search_result():
    result = agenttest.run(answer_question, "What is the capital of France?")
    scripted = ('{"claims": [{"claim": "Paris is the capital of France", "status": "supported", '
                '"evidence": [{"passage": 1, "quote": "Paris is the capital of France"}]}]}')
    # Remove use_judge to check with your configured judge model instead.
    with agenttest.use_judge(lambda prompt, system: scripted):
        agenttest.assert_grounded(result)  # context defaults to the run's tool outputs
'''


def _write_example() -> Path | None:
    path = Path.cwd() / "tests" / "test_agent_example.py"
    if path.exists():
        console.print(f"[yellow]{path} already exists; not overwritten.[/yellow]")
        return None
    path.parent.mkdir(exist_ok=True)
    path.write_text(_EXAMPLE, encoding="utf-8")
    return path.relative_to(Path.cwd())


def _detect_framework() -> str:
    """Scan requirements.txt, pyproject.toml, setup.cfg, setup.py for known packages."""
    dep_files = [
        Path.cwd() / "requirements.txt",
        Path.cwd() / "pyproject.toml",
        Path.cwd() / "setup.cfg",
        Path.cwd() / "setup.py",
    ]

    all_deps = ""
    for f in dep_files:
        if f.exists():
            try:
                all_deps += f.read_text(encoding="utf-8").lower() + "\n"
            except Exception:  # noqa: S110 - an unreadable file just doesn't count
                pass

    if "langgraph" in all_deps:
        return "langgraph"
    elif "crewai" in all_deps:
        return "crewai"
    elif "openai-agents" in all_deps or "openai_agents" in all_deps or "openai" in all_deps:
        return "openai"
    elif "anthropic" in all_deps:
        return "anthropic"
    return "generic"


def _detect_judge_provider() -> str:
    """
    Detect which LLM provider is available based on installed
    packages and environment variables.
    Priority: anthropic > openai > gemini > ollama > none
    """
    import importlib
    import os

    checks = [
        ("anthropic", "anthropic", "ANTHROPIC_API_KEY"),
        ("openai", "openai", "OPENAI_API_KEY"),
        ("gemini", "google.generativeai", "GOOGLE_API_KEY"),
    ]

    for provider, package, env_var in checks:
        try:
            importlib.import_module(package)
            if os.getenv(env_var):
                return provider
        except ImportError:
            continue

    # Check if ollama is running
    try:
        import httpx

        httpx.get("http://localhost:11434", timeout=2.0)
        return "ollama"
    except Exception:  # noqa: S110 - no local Ollama server
        pass

    return "none"


def _register() -> None:
    from agenttest.cli.doctor import doctor
    from agenttest.cli.report import report

    main.add_command(doctor)
    main.add_command(report)


_register()

if __name__ == "__main__":
    main()
