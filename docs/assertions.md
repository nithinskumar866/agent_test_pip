# Assertions

Every `assert_*` function raises `AssertionError` with the expected value, the actual value and an
excerpt of the output, and returns `True` when it passes. Each accepts an `AgentResult` or the raw
value. Boolean versions without the `assert_` prefix (`tool_was_called`, `contains_fact`, ...) live in
`agenttest.assertions`.

## Structural (no API key needed)

| Assertion | Checks |
| :--- | :--- |
| `assert_word_count_between(result, min_count, max_count)` | Word count within inclusive limits. |
| `assert_steps_at_most(result, max_steps)` | At most `max_steps` recorded steps. |
| `assert_tool_was_called(result, name)` | The tool was called. |
| `assert_tool_was_not_called(result, name)` | The tool was not called. |
| `assert_latency_under(result, threshold_ms)` | The run took less than `threshold_ms`. |
| `assert_tokens_under(result, max_tokens)` | Token budget respected. |
| `assert_output_contains(result, substring)` | Case-insensitive substring. |
| `assert_output_matches_regex(result, pattern)` | Regex search. |
| `assert_output_is_valid_json(result)` | The output parses as JSON. |
| `assert_gracefully_degraded(result)` | A tool failed, yet the agent still answered. If not, the error says whether it crashed, no tool failed, or there was no output. |

## Semantic (judge model required)

| Assertion | Checks |
| :--- | :--- |
| `assert_grounded(result, context=None)` | Every factual claim is backed by the context. See [the judge](judge.md). |
| `assert_hallucination_score(result, max_score=0.1, context=None)` | Hallucination score at most `max_score`; grounded when `context` is given. |
| `assert_contains_fact(result, claim)` | The output asserts the claim. |
| `assert_answers_question(result, question)` | The output answers the question. |
| `assert_tone_matches(result, tone)` | The output's tone. |
| `assert_language_matches(result, language)` | The output's language. |
| `assert_eval_prompt(result, criteria)` | Free-form criteria. |

If the judge itself fails (an API outage, or a reply it can't parse after one retry), these raise
`AgentTestJudgeError`. pytest reports that as an *error*, not as a failed assertion.
