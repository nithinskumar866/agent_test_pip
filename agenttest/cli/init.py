"""``agenttest init`` — framework and judge provider detection and config generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

__all__ = ["main", "init", "_detect_framework", "_detect_judge_provider"]


@click.group()
def main() -> None:
    """agenttest CLI tools."""
    pass


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
def init(framework: str | None, judge: str | None) -> None:
    """Initialize agenttest in the current project."""
    console.print(
        Panel(
            "[bold cyan]agenttest[/bold cyan] — AI agent testing framework",
            subtitle="v0.1.0 initialization",
            border_style="cyan",
        )
    )

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

    config_path = Path.cwd() / "agenttest.config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Rich output table
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="green")
    table.add_column(style="bold white")
    table.add_row("✓ Framework detected", detected_framework)
    table.add_row("✓ Judge provider", detected_judge)
    table.add_row("✓ Judge model", judge_model)
    table.add_row("✓ Config written", "agenttest.config.json")
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
        console.print("  Judge disabled — use structural assertions only")
        console.print("  Write tests using [bold]@agent_test[/bold]")
        console.print("  Run [bold cyan]pytest[/bold cyan]")


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
            except Exception:
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
    except Exception:
        pass

    return "none"


if __name__ == "__main__":
    main()
