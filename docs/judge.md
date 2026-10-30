# The judge and grounded hallucination checks

## Choosing a judge

| Provider | `agenttest init --judge ...` | Env var | Default model |
| :--- | :--- | :--- | :--- |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Gemini | `gemini` | `GOOGLE_API_KEY` | `gemini-1.5-flash` |
| Ollama | `ollama` | (local server) | `llama3` |
| None | `none` | - | semantic assertions raise; structural ones work |

Set `judge_model` in `agenttest.config.json` to use another model.

## Grounded hallucination checks

"Is this hallucinated?" has no reliable answer in the abstract. A grounded check asks a checkable
question instead: is every claim backed by the documents or tool results the agent was given?

```python
from agenttest import assert_grounded, check_grounding

assert_grounded(result)                                   # context = the run's tool outputs
report = assert_grounded(result, context=[doc1, doc2], max_unsupported=0.0)

report.score          # fraction of claims contradicted or unsupported
report.contradicted   # [ClaimVerdict(claim=..., evidence=(...), reason=...)]
print(report.describe())
```

One judge call splits the output into claims and labels each one:

- `supported`: a passage states it.
- `contradicted`: a passage states something incompatible with it.
- `unsupported`: the context says nothing either way.
- `disputed`: the passages disagree with each other.

`assert_grounded` fails on any contradicted claim, on more unsupported claims than `max_unsupported`
allows, on disputed claims (unless `allow_disputed=True`), and when the judge reports that the output
tries to instruct it. It returns the `GroundingReport` for further checks. `check_grounding` returns
the report without asserting, and `hallucination_score(result, context=...)` returns just the score.

**Evidence is verified in code.** The judge has to quote its evidence, and each quote must appear in
the passage it cites (ignoring case, whitespace and quote style). A label whose evidence doesn't check
out is downgraded to `unsupported`, and the rejected quote is kept on the claim for diagnosis. So a
judge that was misled, for instance by text planted in the agent's output, can't certify a claim
without real evidence from the context.

## Reliability

- **Prompt injection.** The agent's output and every context passage are enclosed in markers derived
  from a hash of that text, so the text can't contain its own closing marker. The system prompt tells
  the judge that enclosed text is data and that instructions inside it must not be followed.
- **Malformed replies** get one retry with the parse error attached, then raise `AgentTestJudgeError`.
  They are never cached.
- **API failures.** Timeouts, connection errors, rate limits (429) and server errors (5xx) are retried
  with exponential backoff. Other API errors, empty replies and refusals raise `AgentTestJudgeError`.
- **Determinism.** OpenAI, Gemini and Ollama judges run at temperature 0. The Anthropic API no longer
  accepts sampling parameters, so Claude judges rely on the strict reply formats and the evidence checks.
- **Caching.** Successful judge replies are cached for the session, keyed by provider, model and prompt.

Configure in `agenttest.config.json`:

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `judge_provider` | `"anthropic"` | `anthropic`, `openai`, `gemini`, `ollama` or `none`. |
| `judge_model` | provider default | Model id. |
| `judge_max_retries` | `2` | Retries for transient API failures. |
| `judge_timeout` | `60` | Seconds per judge API call. |

## Testing your tests: scripted judges

```python
with agenttest.use_judge(lambda prompt, system: "YES"):
    assert_contains_fact(result, "Paris")
```

`use_judge` routes every judge call in the block to your function, which takes `(prompt, system)` and
returns the reply text. Replies from a custom judge aren't cached. Use it for deterministic tests, or
to plug in a provider agenttest doesn't ship.

## Known limitations

- A judge can still **miss a claim**, or return no claims at all. Only the evidence for claims it lists
  is verified, so `assert_grounded` is only as complete as the judge's claim extraction.
- Quotes are matched exactly after normalisation. A paraphrased quote is rejected, which can mark a true
  claim `unsupported`, and very short quotes match too easily.
- The injection defences reduce the risk, but can't guarantee a model never follows injected text.
  `injection_detected` is the judge's own report.
- Context isn't chunked or truncated, so very large contexts must fit in the judge model's window.
- Without `context`, `hallucination_score` is the judge's opinion, not a verification.
- To check how a real judge behaves, run the opt-in live tests (they cost API credits):
  `AGENTTEST_LIVE_JUDGE=anthropic ANTHROPIC_API_KEY=... pytest tests/test_judge_live.py`
