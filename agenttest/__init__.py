"""agenttest — pytest-style behavioral testing framework for AI agent pipelines.
"""

from agenttest.assertions.semantic import (
    assert_answers_question,
    assert_contains_fact,
    assert_hallucination_score,
    assert_language_matches,
    assert_tone_matches,
)
from agenttest.assertions.structural import (
    assert_gracefully_degraded,
    assert_latency_under,
    assert_output_contains,
    assert_output_is_valid_json,
    assert_output_matches_regex,
    assert_steps_at_most,
    assert_tokens_under,
    assert_tool_was_called,
    assert_tool_was_not_called,
    assert_word_count_between,
)
from agenttest.adapters.base import BaseAdapter
from agenttest.config import (
    AgentTestConfigError,
    AgentTestError,
    AgentTestJudgeError,
    AgentTestTimeoutError,
)
from agenttest.core.decorator import agent_test
from agenttest.core.result import AgentResult
from agenttest.core.runner import arun, run
from agenttest.core.trace import Step, Trace, current_trace, tool
from agenttest.mocks.tool_failure import (
    mock_tool_failure,
    mock_tool_latency,
    mock_tool_response,
)

__version__ = "0.1.0"
__all__ = [
    "run",
    "arun",
    "agent_test",
    "tool",
    "AgentResult",
    "Trace",
    "Step",
    "current_trace",
    "BaseAdapter",
    "AgentTestError",
    "AgentTestConfigError",
    "AgentTestJudgeError",
    "AgentTestTimeoutError",
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
    "assert_contains_fact",
    "assert_hallucination_score",
    "assert_tone_matches",
    "assert_answers_question",
    "assert_language_matches",
    "assert_word_count_between",
    "assert_steps_at_most",
    "assert_tool_was_called",
    "assert_tool_was_not_called",
    "assert_latency_under",
    "assert_tokens_under",
    "assert_output_contains",
    "assert_output_matches_regex",
    "assert_output_is_valid_json",
    "assert_gracefully_degraded",
]
