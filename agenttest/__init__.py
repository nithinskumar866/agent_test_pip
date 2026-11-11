"""agenttest — pytest-style behavioral testing framework for AI agent pipelines.
"""

from agenttest.adapters.base import BaseAdapter
from agenttest.assertions.semantic import (
    assert_answers_question,
    assert_contains_fact,
    assert_grounded,
    assert_hallucination_score,
    assert_language_matches,
    assert_numbers_grounded,
    assert_tone_matches,
    check_grounding,
    invented_numbers_in,
)
from agenttest.assertions.structural import (
    assert_cost_under,
    assert_gracefully_degraded,
    assert_latency_under,
    assert_no_tool_errors,
    assert_output_contains,
    assert_output_is_valid_json,
    assert_output_matches_regex,
    assert_steps_at_most,
    assert_tokens_under,
    assert_tool_call_count,
    assert_tool_called_with,
    assert_tool_was_called,
    assert_tool_was_not_called,
    assert_tools_in_order,
    assert_word_count_between,
)
from agenttest.config import (
    AgentTestBudgetError,
    AgentTestConfigError,
    AgentTestError,
    AgentTestFlakyWarning,
    AgentTestJudgeError,
    AgentTestPassRateError,
    AgentTestTimeoutError,
)
from agenttest.core.conversation import Conversation, aconverse, converse, simulated_user
from agenttest.core.decorator import agent_test
from agenttest.core.result import AgentResult
from agenttest.core.runner import arun, run
from agenttest.core.trace import Step, Trace, current_trace, tool
from agenttest.datasets import Case, cases, load_cases
from agenttest.judge import ClaimVerdict, GroundingReport, use_judge
from agenttest.mocks import (
    REAL,
    MockedToolError,
    MockHandle,
    mock_tool,
    mock_tool_failure,
    mock_tool_latency,
    mock_tool_response,
    mock_tool_sequence,
)
from agenttest.replay import Cassette, CassetteMissError, cassette

__version__ = "0.3.0"
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
    "AgentTestPassRateError",
    "AgentTestFlakyWarning",
    "MockHandle",
    "MockedToolError",
    "mock_tool_failure",
    "mock_tool_response",
    "mock_tool_latency",
    "Conversation",
    "converse",
    "aconverse",
    "simulated_user",
    "Case",
    "cases",
    "load_cases",
    "mock_tool_sequence",
    "mock_tool",
    "REAL",
    "cassette",
    "Cassette",
    "CassetteMissError",
    "AgentTestBudgetError",
    "assert_tool_called_with",
    "assert_tool_call_count",
    "assert_tools_in_order",
    "assert_no_tool_errors",
    "assert_cost_under",
    "use_judge",
    "GroundingReport",
    "ClaimVerdict",
    "check_grounding",
    "assert_grounded",
    "assert_numbers_grounded",
    "invented_numbers_in",
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
