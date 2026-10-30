"""Recording agent runs per test, and rendering them as text, JSON and HTML.

While pytest runs a test, the plugin marks it as the *current test*; every
:func:`agenttest.run` / :func:`agenttest.arun` made meanwhile is recorded
against it (from any thread). The plugin then uses the records to:

* attach a readable trace of every run to a failing test's report,
* add totals to the terminal summary,
* write ``--agenttest-report`` (JSON) and ``--agenttest-html`` reports.

Outside pytest nothing is recorded.
"""

from __future__ import annotations

import html
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from agenttest.core.result import AgentResult

__all__ = [
    "TestRecord",
    "begin_test",
    "end_test",
    "record_result",
    "describe_result",
    "result_to_dict",
    "build_report",
    "render_html",
]

#: Longest string kept per field in reports; agent state can be huge.
_MAX_TEXT: Final[int] = 2000


# --- recording -----------------------------------------------------------------------

_LOCK: Final[threading.Lock] = threading.Lock()
# A global, not a ContextVar: agents often run in worker threads that do not
# inherit the test's context, and pytest runs one test at a time per process.
_current: list[AgentResult] | None = None


def begin_test() -> None:
    """Start recording runs for the test that is about to run."""
    global _current
    with _LOCK:
        _current = []


def end_test() -> list[AgentResult]:
    """Stop recording and return the runs made during the test."""
    global _current
    with _LOCK:
        runs, _current = _current or [], None
    return runs


def current_runs() -> list[AgentResult]:
    """Runs recorded so far for the current test (a copy)."""
    with _LOCK:
        return list(_current or [])


def record_result(result: AgentResult) -> None:
    """Called by the engine for every finished run; a no-op outside a test."""
    with _LOCK:
        if _current is not None:
            _current.append(result)


@dataclass
class TestRecord:
    """One test's outcome and the agent runs it made."""

    nodeid: str
    outcome: str = "passed"
    duration_s: float = 0.0
    agent_marked: bool = False
    message: str | None = None
    runs: list[AgentResult] = field(default_factory=list)


# --- text ----------------------------------------------------------------------------

def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_result(result: AgentResult, title: str = "agent run") -> str:
    """Multi-line, human-readable account of one run: status, totals, each step, output."""
    status = "ok" if result.success else f"FAILED ({result.error})"
    lines = [
        f"{title}: {status} - {result.latency_ms:.0f} ms, {result.steps_taken} steps, "
        f"{result.tokens_used} tokens"
        + (", gracefully degraded" if result.gracefully_degraded else "")
    ]
    for index, step in enumerate(result.trace.steps, start=1):
        line = f"  {index:>2}. {step.kind:<4} {step.name}"
        details = []
        if step.input_tokens or step.output_tokens:
            details.append(f"tokens {step.input_tokens}+{step.output_tokens}")
        if step.tool_requests:
            details.append("requested " + ", ".join(step.tool_requests))
        if step.metadata.get("mocked"):
            details.append(f"mocked {step.metadata['mocked']}")
        if details:
            line += f"  ({'; '.join(details)})"
        if step.error:
            line += f"\n        error: {_short(step.error)}"
        elif step.kind == "tool" and step.output is not None:
            line += f"\n        returned: {_short(step.output)}"
        lines.append(line)
    lines.append(f"  output: {_short(result.output, 300)!r}" if result.output else "  output: (empty)")
    return "\n".join(lines)


# --- JSON ----------------------------------------------------------------------------

def _jsonable(value: Any, depth: int = 0) -> Any:
    """A JSON-safe, size-bounded copy of ``value`` (framework objects become text)."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT] + "…"
    if depth >= 4:
        return _jsonable(repr(value), depth)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(value)[:100]]
    content = getattr(value, "content", None)  # chat messages
    if isinstance(content, str):
        return {"type": type(value).__name__, "content": _jsonable(content, depth + 1)}
    return _jsonable(repr(value), depth)


def result_to_dict(result: AgentResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "output": _jsonable(result.output),
        "error": result.error,
        "latency_ms": round(result.latency_ms, 2),
        "steps_taken": result.steps_taken,
        "tools_called": result.tools_called,
        "failed_tools": [s.name for s in result.trace.failed_tool_calls],
        "tokens_used": result.tokens_used,
        "gracefully_degraded": result.gracefully_degraded,
        "steps": [
            {
                "kind": s.kind,
                "name": s.name,
                "input": _jsonable(s.input),
                "output": _jsonable(s.output),
                "error": s.error,
                "duration_ms": round(s.duration_ms, 2),
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "tool_requests": s.tool_requests,
                "metadata": _jsonable(s.metadata),
            }
            for s in result.trace.steps
        ],
    }


def build_report(tests: list[TestRecord], flaky: list[Any], *, duration_s: float) -> dict[str, Any]:
    """The ``--agenttest-report`` document."""
    from agenttest import __version__

    runs = [run for test in tests for run in test.runs]
    outcomes: dict[str, int] = {}
    for test in tests:
        outcomes[test.outcome] = outcomes.get(test.outcome, 0) + 1
    return {
        "agenttest_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 3),
        "summary": {
            "tests": len(tests),
            **{k: outcomes.get(k, 0) for k in ("passed", "failed", "skipped")},
            "agent_runs": len(runs),
            "failed_runs": sum(not r.success for r in runs),
            "tokens_used": sum(r.tokens_used for r in runs),
            "tool_calls": sum(len(r.trace.tool_calls) for r in runs),
            "failed_tool_calls": sum(len(r.trace.failed_tool_calls) for r in runs),
            "agent_latency_ms": round(sum(r.latency_ms for r in runs), 2),
            "flaky": len(flaky),
        },
        "tests": [
            {
                "nodeid": t.nodeid,
                "outcome": t.outcome,
                "duration_s": round(t.duration_s, 4),
                "agent_marked": t.agent_marked,
                "message": _jsonable(t.message),
                "runs": [result_to_dict(r) for r in t.runs],
            }
            for t in tests
        ],
        "flaky": [{"test": f.test_name, "summary": f.summary()} for f in flaky],
    }


# --- HTML ----------------------------------------------------------------------------

_CSS: Final[str] = """
body{font:14px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1f2328}
h1{font-size:1.5rem;margin:0 0 .25rem} .muted{color:#656d76}
.tiles{display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0}
.tile{border:1px solid #d0d7de;border-radius:6px;padding:.5rem .9rem;min-width:7rem}
.tile b{display:block;font-size:1.3rem}
table{border-collapse:collapse;width:100%} th,td{text-align:left;padding:.35rem .5rem;border-bottom:1px solid #d0d7de;vertical-align:top}
.passed{color:#1a7f37} .failed{color:#cf222e} .skipped{color:#9a6700}
details{margin:.25rem 0} summary{cursor:pointer}
pre{background:#f6f8fa;padding:.6rem;border-radius:6px;overflow:auto;white-space:pre-wrap;margin:.4rem 0}
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_html(report: dict[str, Any]) -> str:
    """A self-contained HTML page for a report built by :func:`build_report`."""
    s = report["summary"]
    tiles = "".join(
        f'<div class="tile"><b>{_esc(value)}</b>{_esc(label)}</div>'
        for label, value in [
            ("tests", s["tests"]), ("passed", s["passed"]), ("failed", s["failed"]), ("skipped", s["skipped"]),
            ("agent runs", s["agent_runs"]), ("tokens", s["tokens_used"]),
            ("failed tool calls", s["failed_tool_calls"]), ("flaky", s["flaky"]),
        ]
    )
    rows = []
    for test in report["tests"]:
        runs_html = []
        for index, run in enumerate(test["runs"], start=1):
            steps = "\n".join(
                f"{n:>2}. {st['kind']:<4} {st['name']}"
                + (f"  ✗ {st['error']}" if st["error"] else "")
                for n, st in enumerate(run["steps"], start=1)
            ) or "(no steps recorded)"
            status = "ok" if run["success"] else f"FAILED: {run['error']}"
            runs_html.append(
                f"<details><summary>run {index}: {_esc(status)} · {run['latency_ms']:.0f} ms · "
                f"{run['tokens_used']} tokens · tools: {_esc(', '.join(run['tools_called']) or '—')}</summary>"
                f"<pre>{_esc(steps)}</pre><pre>output: {_esc(run['output'])}</pre></details>"
            )
        message = f"<pre>{_esc(test['message'])}</pre>" if test["message"] else ""
        rows.append(
            f'<tr><td class="{_esc(test["outcome"])}">{_esc(test["outcome"])}</td>'
            f"<td><code>{_esc(test['nodeid'])}</code>{message}{''.join(runs_html)}</td>"
            f"<td>{test['duration_s']:.2f}s</td><td>{len(test['runs'])}</td></tr>"
        )
    flaky = "".join(f"<li><code>{_esc(f['test'])}</code>: {_esc(f['summary'])}</li>" for f in report["flaky"])
    flaky_html = f"<h2>Flaky tests</h2><ul>{flaky}</ul>" if flaky else ""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>agenttest report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>agenttest report</h1>"
        f'<div class="muted">{_esc(report["created"])} · agenttest {_esc(report["agenttest_version"])} · '
        f"{report['duration_s']:.1f}s</div>"
        f'<div class="tiles">{tiles}</div>'
        "<table><thead><tr><th>outcome</th><th>test</th><th>time</th><th>runs</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{flaky_html}</body></html>"
    )


def write_json(report: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
