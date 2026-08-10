# agenttest — Full Build Prompt for Claude

Paste this entire prompt into Claude Code or Claude to build the library step by step.

---

## THE PROMPT

You are a senior Python open-source library developer with deep experience
publishing production pip packages. We are building a library called
**agenttest** — a pytest-style behavioral testing framework for AI agent
pipelines. This must work exactly like pytest, httpx, or pydantic — a
real, publishable, pip-installable library. Not a script. Not a demo.

---

## WHAT THIS LIBRARY DOES

When a developer installs agenttest and runs `agenttest init` inside their
project folder, the library:

1. Detects which AI agent framework they use (LangGraph, CrewAI, OpenAI
   Agents SDK, raw Anthropic SDK)
2. Auto-configures itself by writing agenttest.config.json
3. From that point, they can write agent tests using @agent_test that
   pytest collects and runs automatically — zero extra config needed

The test style looks exactly like pytest:

```python
from agenttest import agent_test, mock_tool_failure

@agent_test
def test_agent_is_factual():
    result = my_agent.run("What is the capital of France?")
    assert result.contains_fact("Paris")
    assert result.hallucination_score < 0.1
    assert result.steps_taken <= 3

@agent_test
def test_handles_tool_failure():
    with mock_tool_failure("web_search"):
        result = my_agent.run("Search for AI news")
    assert result.gracefully_degraded == True
```

---

## STEP 1 — PROJECT SCAFFOLD

Create this exact folder structure:

```
agenttest/
├── agenttest/
│   ├── __init__.py
│   ├── plugin.py
│   ├── config.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── decorator.py
│   │   ├── result.py
│   │   └── runner.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── langgraph.py
│   │   ├── crewai.py
│   │   ├── openai_sdk.py
│   │   └── anthropic_sdk.py
│   ├── mocks/
│   │   ├── __init__.py
│   │   ├── tool_failure.py
│   │   └── tool_response.py
│   ├── assertions/
│   │   ├── __init__.py
│   │   ├── semantic.py
│   │   └── structural.py
│   └── cli/
│       ├── __init__.py
│       └── init.py
├── tests/
│   ├── test_decorator.py
│   ├── test_result.py
│   └── test_adapters.py
├── setup.cfg
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
        └── publish.yml
```

Write setup.cfg with these EXACT entry points — these are what make it
a real pip library:

```ini
[metadata]
name = agenttest
version = 0.1.0
description = Pytest-style behavioral testing for AI agent pipelines
long_description = file: README.md
long_description_content_type = text/markdown
author = Your Name
license = MIT
classifiers =
    Programming Language :: Python :: 3.10
    Framework :: Pytest
    Topic :: Software Development :: Testing

[options]
packages = find:
python_requires = >=3.10
install_requires =
    pytest>=7.0
    pydantic>=2.0
    httpx>=0.24
    anthropic>=0.25
    click>=8.0
    rich>=13.0

[options.entry_points]
console_scripts =
    agenttest = agenttest.cli.init:main
pytest11 =
    agenttest = agenttest.plugin
```

Write pyproject.toml:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"
```

---

## STEP 2 — AgentResult CLASS

File: agenttest/core/result.py

Build a pydantic v2 BaseModel called AgentResult with these fields:

```python
success: bool
output: str
steps_taken: int
tools_called: list[str]
tokens_used: int
latency_ms: float
gracefully_degraded: bool
raw_trace: dict
error: str | None = None
```

Add these methods to AgentResult:

```python
def contains_fact(self, claim: str) -> bool:
    # Use LLM judge — call _judge_llm(prompt) internally
    # Prompt: "Does this text contain the factual claim '{claim}'?
    #          Text: {self.output}. Answer only YES or NO."
    # Return True if response is YES

def hallucination_score(self) -> float:
    # Property decorated with @property
    # Use LLM judge to score 0.0 to 1.0
    # Prompt: "Rate how likely this text contains hallucinations
    #          on a scale of 0.0 (no hallucinations) to 1.0 (severe).
    #          Text: {self.output}. Return only a float number."
    # Cache result in self._hallucination_cache

def tone_matches(self, tone: str) -> bool:
    # LLM judge: does output match requested tone?

def word_count_between(self, min_count: int, max_count: int) -> bool:
    # Pure Python — no LLM needed
    count = len(self.output.split())
    return min_count <= count <= max_count
```

Write _judge_llm(prompt: str) -> str as a private module-level function:
- Uses anthropic.Anthropic() client
- Reads api key from env ANTHROPIC_API_KEY
- Model: claude-haiku-4-5-20251001
- max_tokens: 50 (we only need short answers)
- Cache results in a module-level dict keyed by hash(prompt)
- If ANTHROPIC_API_KEY not set, raise AgentTestConfigError with helpful message

---

## STEP 3 — BASE ADAPTER

File: agenttest/adapters/base.py

```python
from abc import ABC, abstractmethod
from agenttest.core.result import AgentResult

class BaseAdapter(ABC):

    @abstractmethod
    def wrap(self, agent_fn: callable) -> callable:
        """Wrap an agent function to capture its execution trace"""
        pass

    @abstractmethod
    def capture_trace(self) -> dict:
        """Return raw execution trace after agent runs"""
        pass

    @abstractmethod
    def build_result(self, trace: dict, latency_ms: float) -> AgentResult:
        """Convert raw trace into AgentResult"""
        pass

    def run_with_trace(self, agent_fn: callable, *args, **kwargs) -> AgentResult:
        """Template method — wraps run + trace + build in one call"""
        import time
        start = time.time()
        try:
            agent_fn(*args, **kwargs)
            trace = self.capture_trace()
            latency = (time.time() - start) * 1000
            return self.build_result(trace, latency)
        except Exception as e:
            return AgentResult(
                success=False,
                output="",
                steps_taken=0,
                tools_called=[],
                tokens_used=0,
                latency_ms=(time.time() - start) * 1000,
                gracefully_degraded=False,
                raw_trace={},
                error=str(e)
            )
```

---

## STEP 4 — LANGGRAPH ADAPTER

File: agenttest/adapters/langgraph.py

Implement by monkey-patching LangGraph's execution:

```python
import functools
from agenttest.adapters.base import BaseAdapter
from agenttest.core.result import AgentResult

class LangGraphAdapter(BaseAdapter):

    def __init__(self):
        self._trace = []
        self._original_node_fn = None

    def wrap(self, graph):
        # Monkey-patch the graph's _call method
        # Before each node executes, append to self._trace:
        # {"node": node_name, "input": state, "output": result}
        # Use functools.wraps to preserve the original function signature
        original_invoke = graph.invoke

        @functools.wraps(original_invoke)
        def patched_invoke(state, config=None):
            # Hook into LangGraph's node execution by patching
            # graph.nodes dictionary — for each node, wrap its
            # runnable with a trace-capturing wrapper
            for node_name, node_runnable in graph.nodes.items():
                original_runnable = node_runnable
                def make_traced(name, fn):
                    def traced_fn(s, c=None):
                        result = fn(s, c) if c else fn(s)
                        self._trace.append({
                            "node": name,
                            "input": dict(s),
                            "output": dict(result) if result else {}
                        })
                        return result
                    return traced_fn
                graph.nodes[node_name] = make_traced(node_name, original_runnable)
            return original_invoke(state, config)

        graph.invoke = patched_invoke
        return graph

    def capture_trace(self) -> dict:
        return {"steps": self._trace}

    def build_result(self, trace: dict, latency_ms: float) -> AgentResult:
        steps = trace.get("steps", [])
        # Extract tool calls from trace — look for nodes named
        # "tools" or containing tool_calls in output
        tools_called = []
        for step in steps:
            if "tool_calls" in step.get("output", {}):
                tools_called.extend([
                    t["name"] for t in step["output"]["tool_calls"]
                ])
        # Get final output from last step
        final_output = ""
        if steps:
            last = steps[-1].get("output", {})
            if "messages" in last and last["messages"]:
                final_output = last["messages"][-1].content

        return AgentResult(
            success=True,
            output=final_output,
            steps_taken=len(steps),
            tools_called=tools_called,
            tokens_used=0,  # LangGraph doesn't expose this directly
            latency_ms=latency_ms,
            gracefully_degraded=False,
            raw_trace=trace
        )
```

---

## STEP 5 — @agent_test DECORATOR

File: agenttest/core/decorator.py

This is the most important file. It must:
1. Read agenttest.config.json to know which adapter to use
2. Import the right adapter
3. Wrap the test function so it gets an AgentResult injected
4. Work with pytest's collection system via the pytest plugin

```python
import functools
import json
from pathlib import Path
from typing import Callable

def agent_test(_fn=None, *, timeout: int = 60, max_retries: int = 1,
               model_override: str | None = None):
    """
    Decorator that turns a function into an agenttest test.
    Can be used as @agent_test or @agent_test(timeout=30)
    """
    def decorator(fn: Callable) -> Callable:
        # Mark this function so pytest plugin can find it
        fn._is_agent_test = True
        fn._agenttest_config = {
            "timeout": timeout,
            "max_retries": max_retries,
            "model_override": model_override
        }

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            config = _load_config()
            adapter = _get_adapter(config["framework"])
            # The test function receives the adapter as context
            # It should call agent.run() inside — adapter intercepts it
            return fn(*args, **kwargs)

        return wrapper

    # Handle both @agent_test and @agent_test() usage
    if _fn is not None:
        return decorator(_fn)
    return decorator


def _load_config() -> dict:
    config_path = Path.cwd() / "agenttest.config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "agenttest.config.json not found. Run 'agenttest init' first."
        )
    return json.loads(config_path.read_text())


def _get_adapter(framework: str):
    adapters = {
        "langgraph": "agenttest.adapters.langgraph.LangGraphAdapter",
        "crewai": "agenttest.adapters.crewai.CrewAIAdapter",
        "openai": "agenttest.adapters.openai_sdk.OpenAIAdapter",
        "anthropic": "agenttest.adapters.anthropic_sdk.AnthropicAdapter",
        "generic": "agenttest.adapters.base.GenericAdapter",
    }
    import importlib
    module_path, class_name = adapters[framework].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()
```

---

## STEP 6 — CLI INIT COMMAND

File: agenttest/cli/init.py

Use Click to build the CLI:

```python
import click
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.group()
def main():
    pass

@main.command()
@click.option("--framework", default=None,
              help="Force a specific framework (langgraph/crewai/openai/anthropic)")
def init(framework):
    """Initialize agenttest in the current project."""
    console.print(Panel("[bold]agenttest init[/bold]", subtitle="Detecting your project"))

    detected = framework or _detect_framework()

    if not detected:
        console.print("[red]Could not detect framework. Use --framework to specify.[/red]")
        raise click.Abort()

    config = {
        "framework": detected,
        "judge_model": "claude-haiku-4-5-20251001",
        "timeout": 60,
        "max_retries": 1
    }

    config_path = Path.cwd() / "agenttest.config.json"
    config_path.write_text(json.dumps(config, indent=2))

    console.print(f"[green]✓[/green] Detected framework: [bold]{detected}[/bold]")
    console.print(f"[green]✓[/green] Written agenttest.config.json")
    console.print(f"\nNow write tests using [bold]@agent_test[/bold] and run [bold]pytest[/bold]")


def _detect_framework() -> str | None:
    """
    Scan requirements.txt, pyproject.toml, setup.cfg for known packages.
    Returns the framework name or None if not detected.
    """
    dep_files = [
        Path.cwd() / "requirements.txt",
        Path.cwd() / "pyproject.toml",
        Path.cwd() / "setup.cfg",
        Path.cwd() / "setup.py",
    ]

    all_deps = ""
    for f in dep_files:
        if f.exists():
            all_deps += f.read_text().lower()

    if "langgraph" in all_deps:
        return "langgraph"
    elif "crewai" in all_deps:
        return "crewai"
    elif "openai-agents" in all_deps or "openai_agents" in all_deps:
        return "openai"
    elif "anthropic" in all_deps:
        return "anthropic"
    return None
```

---

## STEP 7 — TOOL MOCKING

File: agenttest/mocks/tool_failure.py

```python
from contextlib import contextmanager
from unittest.mock import patch

@contextmanager
def mock_tool_failure(tool_name: str):
    """
    Context manager that makes a named tool raise an exception when called.
    Works by patching the tool at the framework level.
    Usage:
        with mock_tool_failure("web_search"):
            result = my_agent.run("task")
    """
    # We patch at a generic level — the adapter layer intercepts
    # tool calls and checks if the tool name is in the mock registry
    from agenttest.mocks._registry import MOCK_REGISTRY
    MOCK_REGISTRY[tool_name] = {"type": "failure", "error": f"Tool '{tool_name}' failed (mocked)"}
    try:
        yield
    finally:
        MOCK_REGISTRY.pop(tool_name, None)


@contextmanager
def mock_tool_response(tool_name: str, returns: dict):
    """
    Context manager that makes a named tool return custom data.
    """
    from agenttest.mocks._registry import MOCK_REGISTRY
    MOCK_REGISTRY[tool_name] = {"type": "response", "data": returns}
    try:
        yield
    finally:
        MOCK_REGISTRY.pop(tool_name, None)


@contextmanager
def mock_tool_latency(tool_name: str, delay_ms: int):
    """
    Context manager that adds artificial latency to a tool.
    """
    import time
    from agenttest.mocks._registry import MOCK_REGISTRY
    MOCK_REGISTRY[tool_name] = {"type": "latency", "delay": delay_ms / 1000}
    try:
        yield
    finally:
        MOCK_REGISTRY.pop(tool_name, None)
```

File: agenttest/mocks/_registry.py

```python
# Module-level dict that mocks write to and adapters check
MOCK_REGISTRY: dict = {}
```

---

## STEP 8 — PYTEST PLUGIN

File: agenttest/plugin.py

This file is what makes agenttest a first-class pytest plugin:

```python
import pytest
from agenttest.core.result import AgentResult

def pytest_collect_file(parent, file_path):
    """Tell pytest to collect our test files"""
    if file_path.suffix == ".py" and file_path.name.startswith("test_"):
        return pytest.Module.from_parent(parent, path=file_path)


def pytest_runtest_call(item):
    """
    Hook into each test call.
    If the test function has _is_agent_test=True, we know it's
    an agenttest test and handle it specially.
    """
    pass  # The @agent_test decorator handles wrapping — pytest runs it normally


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add agenttest summary to pytest output"""
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    if passed + failed > 0:
        terminalreporter.write_sep("=", "agenttest summary")
        terminalreporter.write_line(f"Agent tests: {passed} passed, {failed} failed")
```

---

## STEP 9 — __init__.py EXPORTS

File: agenttest/__init__.py

Export only the public API:

```python
from agenttest.core.decorator import agent_test
from agenttest.mocks.tool_failure import (
    mock_tool_failure,
    mock_tool_response,
    mock_tool_latency
)
from agenttest.core.result import AgentResult

__version__ = "0.1.0"
__all__ = [
    "agent_test",
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
    "AgentResult",
]
```

---

## STEP 10 — README.md

Write a professional README with:

1. One-line description
2. Install section:
   ```
   pip install agenttest
   agenttest init
   ```
3. Quick start with a 10-line example showing @agent_test
4. Supported frameworks table (LangGraph, CrewAI, OpenAI SDK, Anthropic SDK)
5. Available assertions table
6. Available mocks table
7. CI/CD section showing how to add to GitHub Actions
8. Contributing section

---

## STEP 11 — GITHUB ACTIONS PUBLISH WORKFLOW

File: .github/workflows/publish.yml

Write a workflow that:
- Triggers on push to main with a version tag (v*)
- Runs tests with pytest
- Builds the package with python -m build
- Publishes to PyPI using TWINE_API_KEY secret

---

## STEP 12 — SELF-TESTS

Write tests in tests/ that test agenttest itself without needing a real agent:

File: tests/test_result.py
- Test AgentResult construction
- Test word_count_between with mock output
- Test that hallucination_score raises proper error when no API key

File: tests/test_decorator.py
- Test that @agent_test marks the function with _is_agent_test = True
- Test that missing config file raises FileNotFoundError with helpful message

File: tests/test_cli.py
- Use click.testing.CliRunner to test agenttest init
- Test framework detection from a mock requirements.txt

---

## REQUIREMENTS

- Python 3.10+ with full type hints everywhere
- Pydantic v2 for AgentResult
- Click for CLI
- Rich for terminal output (colors, panels)
- Pytest 7+ compatibility
- All files must have module-level docstrings
- Every public function must have a docstring with Args and Returns
- Zero global state except MOCK_REGISTRY
- Thread-safe where possible

---

## START NOW

Build Step 1 first (complete scaffold + setup.cfg + pyproject.toml).
After I confirm, move to Step 2. Do not skip ahead.
Show me the complete file contents for every file you create.
