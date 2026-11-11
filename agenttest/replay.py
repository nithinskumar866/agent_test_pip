"""Record model API calls once, then replay them: fast, free, repeatable tests.

A *cassette* is a JSON file holding the HTTP exchanges with a model provider.
It works at the HTTP layer (``httpx`` and ``httpx2``), underneath the Anthropic,
OpenAI and Gemini SDKs and the frameworks built on them, so no framework needs
its own support::

    @agent_test(cassette=True)
    def test_agent():
        result = agenttest.run(agent, "What is the capital of France?")
        ...

Modes (``--agenttest-record=MODE``, ``AGENTTEST_RECORD``, or ``mode=``):

* ``once`` (default): replay if the cassette exists, otherwise record it.
* ``new``: replay what matches, and record calls that don't.
* ``all``: always call the API and rewrite the cassette.
* ``none`` (default when ``CI`` is set): replay only, and never touch the network.

A request matches on method, URL and body, so **changing a prompt is a cassette
miss**: the test stops with :class:`CassetteMissError` instead of quietly
replaying an answer to a different question. Only known model-provider hosts are
intercepted (add others with ``hosts=``), so telemetry is left alone.

Secrets: request headers are never stored, and bodies pass through
:mod:`agenttest.redaction`. A cassette is saved only when the block finishes
without an exception, so a failed recording never gets replayed.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import re
import threading
from collections.abc import Callable, Iterable
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Final
from unittest import mock
from urllib.parse import parse_qsl, urlencode

from agenttest.config import AgentTestConfigError
from agenttest.redaction import redact, redact_value

__all__ = ["Cassette", "CassetteMissError", "cassette", "cassette_for", "record_mode", "MODES"]

MODES: Final[tuple[str, ...]] = ("once", "new", "all", "none")
_FORMAT: Final[int] = 1
_HOSTS: Final[tuple[str, ...]] = (
    "api.anthropic.com",
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "api.mistral.ai",
    "api.groq.com",
    "api.cohere.com",
    "api.together.xyz",
    "openrouter.ai",
    "localhost:11434",
    "127.0.0.1:11434",
)
_SECRET_QUERY: Final[frozenset[str]] = frozenset({"key", "api_key", "apikey", "token"})


class CassetteMissError(BaseException):
    """A request has no recording.

    Deliberately not an ``Exception``: SDKs catch ``Exception`` around HTTP
    calls and would retry, then report a vague connection error instead.
    """


def record_mode(mode: str | None = None) -> str:
    """The mode to use: explicit, then ``AGENTTEST_RECORD``, then ``CI``-aware default."""
    chosen = mode or os.environ.get("AGENTTEST_RECORD") or ("none" if os.environ.get("CI") else "once")
    if chosen not in MODES:
        raise AgentTestConfigError(f"Unknown record mode {chosen!r}; expected one of {', '.join(MODES)}")
    return chosen


_ACTIVE: Cassette | None = None
_PATCH_LOCK: Final[threading.Lock] = threading.Lock()


def _http_modules() -> list[Any]:
    modules = []
    for name in ("httpx", "httpx2"):
        try:
            modules.append(__import__(name))
        except ImportError:
            pass
    return modules


def _canonical_url(url: Any) -> str:
    """Scheme-less URL with sorted query parameters and no credentials."""
    query = sorted((k, v) for k, v in parse_qsl(str(url.query, "ascii")) if k.lower() not in _SECRET_QUERY)
    return f"{url.host}{f':{url.port}' if url.port else ''}{url.path}{'?' + urlencode(query) if query else ''}"


def _parse_json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except ValueError:
        return None


class Cassette:
    """One recording of model API traffic. Use as a context manager."""

    def __init__(
        self,
        path: str | Path,
        mode: str | None = None,
        *,
        hosts: Iterable[str] = (),
        ignore_body_keys: Iterable[str] = (),
    ) -> None:
        """
        Args:
            path: The cassette file.
            mode: ``once``, ``new``, ``all`` or ``none``.
            hosts: Extra hosts to intercept, e.g. a private gateway.
            ignore_body_keys: Top-level request-body keys that vary between
                runs and shouldn't affect matching.
        """
        self.path = Path(path)
        self.mode = record_mode(mode)
        self.hosts = (*_HOSTS, *hosts)
        self.ignore_body_keys = frozenset(ignore_body_keys)
        self.interactions: list[dict[str, Any]] = []
        self.replayed = 0
        self.recorded = 0
        self._existed = False
        self._loaded = 0  # only what was on disk can be replayed, not this session's recordings
        self._dirty = False
        self._positions: dict[str, int] = {}
        self._lock = threading.Lock()
        self._patches: list[Any] = []

    # --- lifecycle ------------------------------------------------------------

    def __enter__(self) -> Cassette:
        global _ACTIVE
        self._load()
        with _PATCH_LOCK:
            if _ACTIVE is not None:
                raise AgentTestConfigError("Cassettes can't be nested: another cassette is already active.")
            _ACTIVE = self
        for module in _http_modules():
            for patcher in (
                mock.patch.object(module.Client, "send", _sync_send(module, module.Client.send)),
                mock.patch.object(module.AsyncClient, "send", _async_send(module, module.AsyncClient.send)),
            ):
                patcher.start()
                self._patches.append(patcher)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        global _ACTIVE
        for patcher in reversed(self._patches):
            patcher.stop()
        self._patches.clear()
        with _PATCH_LOCK:
            _ACTIVE = None
        if exc_type is None and self._dirty:
            self._save()

    def _load(self) -> None:
        self._existed = self.path.exists()
        if not self._existed or self.mode == "all":
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["version"] != _FORMAT:
                raise ValueError(f"unsupported cassette version {data['version']}")
            self.interactions = data["interactions"]
            self._loaded = len(self.interactions)
        except (ValueError, KeyError, TypeError) as exc:
            raise AgentTestConfigError(f"Cannot read cassette {self.path}: {exc}") from exc

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"version": _FORMAT, "interactions": self.interactions}, indent=1, ensure_ascii=False)
        self.path.write_text(text + "\n", encoding="utf-8")

    # --- matching -------------------------------------------------------------

    def intercepts(self, request: Any) -> bool:
        if request.url.path.startswith("/api/generate"):  # an Ollama judge, wherever it runs
            return True
        netloc = request.url.host + (f":{request.url.port}" if request.url.port else "")
        return any(netloc == h or request.url.host == h or request.url.host.endswith(f".{h}") for h in self.hosts) or (
            request.url.host.endswith(".openai.azure.com")
        )

    def _key(self, request: Any, body: bytes) -> str:
        parsed = _parse_json(body)
        if isinstance(parsed, dict):
            parsed = {k: v for k, v in parsed.items() if k not in self.ignore_body_keys}
            canonical = json.dumps(parsed, sort_keys=True, ensure_ascii=False).encode()
        else:
            canonical = body
        material = request.method.encode() + b" " + _canonical_url(request.url).encode() + b"\n" + canonical
        return hashlib.sha256(material).hexdigest()

    def _find(self, key: str) -> dict[str, Any] | None:
        """The next recording for ``key``; repeated requests cycle through them in order."""
        matches = [i for i in self.interactions[: self._loaded] if i["key"] == key]
        if not matches:
            return None
        with self._lock:
            position = self._positions.get(key, 0)
            self._positions[key] = position + 1
            self.replayed += 1
        return matches[position % len(matches)]

    def _miss(self, request: Any) -> CassetteMissError:
        what = f"{request.method} {_canonical_url(request.url)}"
        if not self._existed and self.mode == "none":
            why = f"cassette {self.path} does not exist and the record mode is 'none'"
        else:
            why = f"cassette {self.path} has no recording of this request (mode '{self.mode}')"
        return CassetteMissError(
            f"{what}: {why}. If the prompt or tools changed on purpose, re-record with "
            "--agenttest-record=all, or use --agenttest-record=new to add only the new calls."
        )

    def lookup(self, request: Any, body: bytes) -> tuple[str, dict[str, Any] | None]:
        """``(key, recording)``; raises on a miss when this mode can't go live."""
        key = self._key(request, body)
        if self.mode == "all":
            return key, None
        found = self._find(key)
        if found is None and (self.mode == "none" or (self.mode == "once" and self._existed)):
            raise self._miss(request)
        return key, found

    def store(self, key: str, request: Any, body: bytes, status: int, headers: Any, content: bytes) -> None:
        if status == 429 or status >= 500:
            return  # transient failures would otherwise be replayed forever
        response: dict[str, Any] = {"status": status, "content_type": headers.get("content-type", "application/json")}
        try:
            response["body"] = redact(content.decode("utf-8"))
        except UnicodeDecodeError:
            response["body_base64"] = base64.b64encode(content).decode("ascii")
        parsed = _parse_json(body)
        with self._lock:
            self.interactions.append({
                "key": key,
                "request": {
                    "method": request.method,
                    "url": redact(_canonical_url(request.url)),
                    "body": redact_value(parsed) if parsed is not None else None,
                },
                "response": response,
            })
            self.recorded += 1
            self._dirty = True


def _build_response(module: Any, request: Any, status: int, content_type: str, content: bytes) -> Any:
    response = module.Response(status, headers={"content-type": content_type}, content=content, request=request)
    response.elapsed = timedelta(0)
    return response


def _replayed(module: Any, request: Any, found: dict[str, Any]) -> Any:
    stored = found["response"]
    content = (
        base64.b64decode(stored["body_base64"]) if "body_base64" in stored else stored["body"].encode("utf-8")
    )
    return _build_response(module, request, stored["status"], stored["content_type"], content)


def _sync_send(module: Any, original: Callable[..., Any]) -> Callable[..., Any]:
    def send(client: Any, request: Any, **kwargs: Any) -> Any:
        active = _ACTIVE
        if active is None or not active.intercepts(request):
            return original(client, request, **kwargs)
        body = request.read()
        key, found = active.lookup(request, body)
        if found is not None:
            return _replayed(module, request, found)
        response = original(client, request, **kwargs)
        content = response.read()
        active.store(key, request, body, response.status_code, response.headers, content)
        return _build_response(
            module, request, response.status_code, response.headers.get("content-type", "application/json"), content
        )

    return send


def _async_send(module: Any, original: Callable[..., Any]) -> Callable[..., Any]:
    async def send(client: Any, request: Any, **kwargs: Any) -> Any:
        active = _ACTIVE
        if active is None or not active.intercepts(request):
            return await original(client, request, **kwargs)
        body = await request.aread()
        key, found = active.lookup(request, body)
        if found is not None:
            return _replayed(module, request, found)
        response = await original(client, request, **kwargs)
        content = await response.aread()
        active.store(key, request, body, response.status_code, response.headers, content)
        return _build_response(
            module, request, response.status_code, response.headers.get("content-type", "application/json"), content
        )

    return send


def cassette(path: str | Path, mode: str | None = None, **options: Any) -> Cassette:
    """Open a cassette: ``with agenttest.cassette("cassettes/news.json"): ...``."""
    return Cassette(path, mode, **options)


def cassette_for(function: Callable[..., Any], value: bool | str | Path) -> Cassette:
    """The cassette for ``@agent_test(cassette=value)``.

    ``True`` stores it at ``cassettes/<module>/<test>.json`` next to the test
    file; a string is a path relative to the test file.
    """
    base = Path(inspect.getfile(function)).resolve().parent
    if value is True:
        stem = Path(inspect.getfile(function)).stem
        name = re.sub(r"[^\w.-]+", "_", function.__qualname__)
        return Cassette(base / "cassettes" / stem / f"{name}.json")
    return Cassette(base / Path(str(value)))
