# Datasets

Keep test cases in a file, and run one test over all of them.

```python
@agenttest.cases("cases.jsonl")
@agent_test(runs=3, min_pass_rate=0.67)
def test_answers(case):
    result = agenttest.run(my_agent, case.question)
    assert_output_contains(result, case.expected)
```

```json
{"id": "capital", "question": "Capital of France?", "expected": "Paris"}
{"id": "maths", "question": "What is 6 x 7?", "expected": "42"}
```

Each row becomes its own pytest test, named after its `id` field, so failures say which case broke and
`pytest -k capital` runs just one. `.jsonl`, `.json` (a list) and `.csv` all work. A relative path is
resolved against the test file.

Fields work as attributes or keys: `case.question` or `case["question"]`. `agenttest.load_cases(path)`
returns the rows if you'd rather handle them yourself, and `agenttest.cases([{...}, {...}])` accepts a
list directly.
