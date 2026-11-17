"""The HTTP library an SDK is built on, for tests that hand it a mock transport.

SDKs are moving from ``httpx`` to its successor ``httpx2`` (openai 3 and
google-genai 3 depend on httpx2 only, so plain httpx may not be installed).
A mock client has to be built with the library the SDK itself uses, so look
at what the installed SDK declares rather than guessing.
"""

from __future__ import annotations

import importlib.metadata
import re
from types import ModuleType

import pytest


def sdk_http(distribution: str) -> ModuleType:
    """``httpx2`` if the installed ``distribution`` depends on it, else ``httpx``; skip if missing."""
    try:
        requirements = importlib.metadata.requires(distribution) or []
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{distribution} is not installed")
    names = {re.match(r"[A-Za-z0-9_.-]+", r).group().lower() for r in requirements if not r.startswith(" ")}
    return pytest.importorskip("httpx2" if "httpx2" in names else "httpx")
