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

from agenttest.redaction import redact

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
    #: ``result_to_dict`` output (results are also accepted, and converted when reported).
    runs: list[Any] = field(default_factory=list)


# --- text ----------------------------------------------------------------------------

def _short(value: Any, limit: int = 120) -> str:
    text = redact(value if isinstance(value, str) else repr(value))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_result(result: AgentResult, title: str = "agent run") -> str:
    """Multi-line, human-readable account of one run: status, totals, each step, output."""
    status = "ok" if result.success else f"FAILED ({redact(result.error or '')})"
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
        value = redact(value)
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
        "cost_usd": result.cost_usd,
        "tool_calls": len(result.trace.tool_calls),
        "success": result.success,
        "output": _jsonable(result.output),
        "error": _jsonable(result.error),
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
                "error": _jsonable(s.error),
                "duration_ms": round(s.duration_ms, 2),
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "tool_requests": s.tool_requests,
                "metadata": _jsonable(s.metadata),
            }
            for s in result.trace.steps
        ],
    }


def _as_dict(run: Any) -> dict[str, Any]:
    return run if isinstance(run, dict) else result_to_dict(run)


def total_cost(runs: list[dict[str, Any]]) -> float | None:
    """Sum of the runs' costs; ``None`` if any of them is unknown."""
    costs = [r.get("cost_usd") for r in runs]
    if any(c is None for c in costs):
        return None
    return round(sum(c for c in costs if c is not None), 6)


def build_report(tests: list[TestRecord], flaky: list[Any], *, duration_s: float) -> dict[str, Any]:
    """The ``--agenttest-report`` document."""
    from agenttest import __version__

    rendered = [
        {
            "nodeid": t.nodeid,
            "outcome": t.outcome,
            "duration_s": round(t.duration_s, 4),
            "agent_marked": t.agent_marked,
            "message": _jsonable(t.message),
            "runs": [_as_dict(r) for r in t.runs],
        }
        for t in tests
    ]
    runs = [run for test in rendered for run in test["runs"]]
    outcomes: dict[str, int] = {}
    for test in tests:
        outcomes[test.outcome] = outcomes.get(test.outcome, 0) + 1
    return {
        "agenttest_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration_s, 3),
        "summary": {
            "tests": len(tests),
            **{k: outcomes.get(k, 0) for k in ("passed", "failed", "skipped", "xfailed", "xpassed")},
            "agent_runs": len(runs),
            "failed_runs": sum(not r["success"] for r in runs),
            "tokens_used": sum(r["tokens_used"] for r in runs),
            "cost_usd": total_cost(runs),
            "tool_calls": sum(r["tool_calls"] for r in runs),
            "failed_tool_calls": sum(len(r["failed_tools"]) for r in runs),
            "agent_latency_ms": round(sum(r["latency_ms"] for r in runs), 2),
            "flaky": len(flaky),
        },
        "tests": rendered,
        "flaky": [f if isinstance(f, dict) else {"test": f.test_name, "summary": f.summary()} for f in flaky],
    }


# --- markdown and comparison -----------------------------------------------------

def _cost_text(cost: float | None) -> str:
    return "unknown" if cost is None else f"${cost:.4f}"


def _cell(text: Any) -> str:
    """Safe inside a markdown table, and unable to inject mentions or markup."""
    return " ".join(str(text).split()).replace("|", "\\|").replace("@", "@\u200b").replace("<", "&lt;")


def render_markdown(report: dict[str, Any], comparison: dict[str, Any] | None = None) -> str:
    """A report as markdown, for a pull request comment or job summary."""
    s = report["summary"]
    icon = "\u274c" if s["failed"] else "\u2705"
    lines = [
        "<!-- agenttest-report -->",
        f"### {icon} agenttest: {s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped",
        "",
        f"{s['agent_runs']} agent runs \u00b7 {s['tokens_used']} tokens \u00b7 cost {_cost_text(s.get('cost_usd'))} "
        f"\u00b7 {s['failed_tool_calls']} failed tool calls \u00b7 {s['flaky']} flaky",
    ]
    failed = [t for t in report["tests"] if t["outcome"] == "failed"]
    if failed:
        lines += ["", "| Failed test | Why |", "| :--- | :--- |"]
        lines += [f"| `{_cell(t['nodeid'])}` | {_cell(t['message'] or '')[:300]} |" for t in failed[:30]]
        if len(failed) > 30:
            lines.append(f"\n...and {len(failed) - 30} more.")
    if report["flaky"]:
        lines += ["", "**Flaky:**"] + [f"- `{_cell(f['test'])}`: {_cell(f['summary'])}" for f in report["flaky"][:30]]
    if comparison is not None:
        lines += ["", render_comparison(comparison)]
    return "\n".join(lines) + "\n"


def _change(before: float, after: float) -> float | None:
    return None if before == 0 else round((after - before) / before * 100, 1)


def compare_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    max_token_increase: float | None = None,
    max_cost_increase: float | None = None,
) -> dict[str, Any]:
    """What changed between two reports, and whether that counts as a regression.

    Args:
        max_token_increase: Percent growth in total tokens treated as a regression.
        max_cost_increase: Percent growth in total cost treated as a regression.
    """
    old = {t["nodeid"]: t for t in baseline["tests"]}
    new = {t["nodeid"]: t for t in current["tests"]}
    shared = sorted(old.keys() & new.keys())
    newly_failing = [n for n in shared if old[n]["outcome"] == "passed" and new[n]["outcome"] == "failed"]
    newly_passing = [n for n in shared if old[n]["outcome"] == "failed" and new[n]["outcome"] == "passed"]
    a, b = baseline["summary"], current["summary"]
    tokens_pct = _change(a["tokens_used"], b["tokens_used"])
    cost_a, cost_b = a.get("cost_usd"), b.get("cost_usd")
    cost_pct = _change(cost_a, cost_b) if cost_a is not None and cost_b is not None else None
    reasons = [f"{len(newly_failing)} test(s) newly failing"] if newly_failing else []
    if max_token_increase is not None and tokens_pct is not None and tokens_pct > max_token_increase:
        reasons.append(f"tokens grew {tokens_pct}% (allowed {max_token_increase:g}%)")
    if max_cost_increase is not None and cost_pct is not None and cost_pct > max_cost_increase:
        reasons.append(f"cost grew {cost_pct}% (allowed {max_cost_increase:g}%)")
    return {
        "newly_failing": newly_failing,
        "newly_passing": newly_passing,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
        "tokens": {"before": a["tokens_used"], "after": b["tokens_used"], "change_pct": tokens_pct},
        "cost_usd": {"before": cost_a, "after": cost_b, "change_pct": cost_pct},
        "regressions": reasons,
    }


def render_comparison(comparison: dict[str, Any]) -> str:
    """A comparison as markdown (also readable as plain text)."""
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+g}%"

    tokens, cost = comparison["tokens"], comparison["cost_usd"]
    reasons = comparison["regressions"]
    verdict = "**Regressions:** " + "; ".join(reasons) if reasons else "No regressions."
    lines = [
        "#### Compared with baseline",
        "",
        verdict,
        "",
        f"- Tokens: {tokens['before']} -> {tokens['after']} ({pct(tokens['change_pct'])})",
        f"- Cost: {_cost_text(cost['before'])} -> {_cost_text(cost['after'])} ({pct(cost['change_pct'])})",
    ]
    for label, key in [("Newly failing", "newly_failing"), ("Newly passing", "newly_passing"),
                       ("Added", "added"), ("Removed", "removed")]:
        if comparison[key]:
            names = ", ".join(f"`{_cell(n)}`" for n in comparison[key][:20])
            more = f" and {len(comparison[key]) - 20} more" if len(comparison[key]) > 20 else ""
            lines.append(f"- {label} ({len(comparison[key])}): {names}{more}")
    return "\n".join(lines)


# --- HTML ----------------------------------------------------------------------------

_CSS: Final[str] = """
body{font:14px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1f2328}
h1{font-size:1.5rem;margin:0 0 .25rem} .muted{color:#656d76}
.tiles{display:flex;gap:.75rem;flex-wrap:wrap;margin:1rem 0}
.tile{border:1px solid #d0d7de;border-radius:6px;padding:.5rem .9rem;min-width:7rem}
.tile b{display:block;font-size:1.3rem}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:.35rem .5rem;border-bottom:1px solid #d0d7de;vertical-align:top}
.passed{color:#1a7f37} .failed{color:#cf222e} .skipped{color:#9a6700} .xfailed{color:#8250df} .xpassed{color:#0969da}
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
            ("agent runs", s["agent_runs"]), ("tokens", s["tokens_used"]), ("cost", _cost_text(s.get("cost_usd"))),
            ("failed tool calls", s["failed_tool_calls"]), ("flaky", s["flaky"]),
            # Reports written before 0.3.0 have no xfail counts.
            *[(label, s[key]) for key, label in (("xfailed", "expected failures"), ("xpassed", "unexpected passes"))
              if s.get(key)],
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
