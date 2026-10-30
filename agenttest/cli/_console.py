"""Console output that works on every terminal, including legacy Windows code pages."""

from __future__ import annotations

import sys
from typing import Final

__all__ = ["make_safe", "symbols"]

_UNICODE: Final[dict[str, str]] = {"ok": "✓", "warn": "!", "fail": "✗", "dash": "—"}
_ASCII: Final[dict[str, str]] = {"ok": "+", "warn": "!", "fail": "x", "dash": "-"}


def make_safe() -> None:
    """Never crash on a character the terminal cannot encode: replace it instead."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):  # detached or closed streams
                pass


def symbols() -> dict[str, str]:
    """Status symbols the current stdout can display."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "".join(_UNICODE.values()).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return _ASCII
    return _UNICODE
