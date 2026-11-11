# The judge and grounded hallucination checks

## Choosing a judge

| Provider | `agenttest init --judge ...` | Env var | Default model |
| :--- | :--- | :--- | :--- |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Gemini | `gemini` | `GOOGLE_API_KEY` | `gemini-3.6-flash` |
| Ollama | `ollama` | (local server) | `llama3` |
| None | `none` | - | semantic assertions raise; structural ones work |

Set `judge_model` in `agenttest.config.json` to use another model.

### Ollama on another machine (RunPod, a GPU server)

```json
{"judge_provider": "ollama", "judge_model": "llama3.1:8b", "judge_base_url": "https://<pod-id>-11434.proxy.runpod.net"}
```

`OLLAMA_HOST` works too. If the server sits behind an authenticating proxy, set `OLLAMA_API_KEY` and it's
sent as a bearer token. Ollama judges get a 600-second timeout by default, because CPU inference is slow.

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

**Invented numbers are caught in code first.** Order numbers, dates, prices, tracking codes: when an agent
makes something up, it is very often a number. Before the judge is asked, `assert_grounded` checks that
every number in the output appears somewhere in the context. That needs no model, so it is free, instant,
and never wrong about the numbers it compares. Numbers are compared by value ("1,000" = "1000", "3rd" = "3",
"06" = "6"), and digits inside identifiers count ("GLK1321" contributes 1321). It can't see numbers written
as words, and it only knows a number *appears* in the context, not that it is used correctly; that part
stays with the judge. Turn it off with `check_numbers=False`, or use it on its own:

```python
from agenttest import assert_numbers_grounded, invented_numbers_in

assert_numbers_grounded(result)                    # no judge needed
assert_numbers_grounded(result, ignore=[24])       # e.g. "we reply within 24 hours" from the prompt
invented_numbers_in(result)                        # ["99812"]
```

**A second look at doubtful claims.** A judge reading a whole answer at once sometimes misses support it
would find for a single claim; small local models do this most. So each claim judged `unsupported` (up to
five per check) is asked about again on its own. It only becomes `supported` if every part of the second
answer is supported by quotes that are verified in code *and* those quotes contain every number in the
claim, so an invented number can't be waved through. Each second look is one extra judge call, made only
for claims that would otherwise fail. Turn it off with `"judge_recheck": false`. The report marks claims
confirmed this way: `(confirmed on a second look)`.

A claim that only joins claims already supported ("A and B" next to "A" and "B") is not counted as
unsupported either: each part must match a verified claim exactly, so a changed or added fact inside
the combination still fails.

## Reliability

- **Prompt injection.** The agent's output and every context passage are enclosed in markers derived
  from a hash of that text, so the text can't contain its own closing marker. The system prompt tells
  the judge that enclosed text is data and that instructions inside it must not be followed.
- **Malformed replies** get one retry with the parse error attached, then raise `AgentTestJudgeError`.
  They are never cached.
- **API failures.** Timeouts, connection errors (including a connection a proxy dropped mid-reply), rate
  limits (429) and server errors (5xx) are retried
  with exponential backoff. Other API errors, empty replies and refusals raise `AgentTestJudgeError`.
- **Determinism.** OpenAI and Ollama judges run at temperature 0. Current Anthropic and Gemini models don't
  accept sampling parameters, so those judges rely on the strict reply formats and the evidence checks.
- **Caching.** Successful judge replies are cached for the session, keyed by provider, model and prompt.
  Set `"judge_cache": true` to keep them on disk between sessions.

Configure in `agenttest.config.json`:

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `judge_provider` | `"anthropic"` | `anthropic`, `openai`, `gemini`, `ollama` or `none`. |
| `judge_model` | provider default | Model id. |
| `judge_max_retries` | `2` | Retries for transient API failures. |
| `judge_timeout` | `60` (`600` for Ollama) | Seconds per judge call. |
| `judge_recheck` | `true` | Give each claim judged unsupported a second look on its own. |

## How accurate is your judge?

Measure it instead of assuming. `agenttest benchmark` runs 100 labelled cases through your configured judge,
using invented facts so that it can't rely on its own knowledge:

```bash
agenttest benchmark --json benchmark.json --min-accuracy 0.9
```

It reports accuracy for correct answers, hallucinations, contradictions, self-contradicting sources and
prompt injection, plus how often good answers were rejected and bad ones accepted. It makes real judge calls
(about 100), so it costs money; `--limit 4` runs a 20-case sample.

The benchmark scores the judge alone (including its second look). The code-only number check is left
out on purpose: most contradicted cases turn on a changed number, so it would settle them without the
judge and hide how good the judge really is.

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
