"""Registry for mock tool behaviors used by agenttest context managers.

This dictionary holds active mock configurations keyed by tool name.
Adapters check this registry during agent execution to intercept tool calls.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MOCK_REGISTRY"]

#: Global registry mapping tool name to mock configuration dict.
#: Example: {"web_search": {"type": "failure", "error": "Tool failed"}}
MOCK_REGISTRY: dict[str, dict[str, Any]] = {}
