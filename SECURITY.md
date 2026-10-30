# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/nithinskumar866/agent_test_pip/security/advisories/new),
not in a public issue. Include the version, a description, and steps to reproduce. You should hear back
within a week.

## Supported versions

Security fixes go into the latest release.

## What agenttest does with your data

- **API keys** are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `GOOGLE_API_KEY`) and passed only to the matching provider SDK. agenttest never writes them to disk,
  reports or logs, and `agenttest doctor` only says whether a key is set.
- **The judge** receives the agent's output and, for grounded checks, the context passages you pass
  (or the run's tool outputs). Don't put data in tests that your judge provider must not see. Use
  `judge_provider: "ollama"` to keep judging local, or `"none"` to disable it.
- **Reports** (`--agenttest-report`, `--agenttest-html`) contain agent outputs, tool inputs and tool
  outputs, shortened. Treat them like test logs: if your tools handle secrets or personal data, check
  the reports before publishing them as CI artifacts. The HTML report escapes all content, so agent
  output can't inject script into the page.
- **Untrusted agent output** given to the judge is enclosed as data under a system prompt that forbids
  following instructions inside it, and grounded verdicts must be backed by quotes verified in code.
  This reduces, but can't eliminate, the risk of prompt injection steering a judge. See
  [docs/judge.md](docs/judge.md).
- **Configuration** is read from `agenttest.config.json` as plain JSON. No code is executed from it.

## Supply chain

Releases are built in GitHub Actions and published to PyPI with trusted publishing (OIDC), so no
long-lived upload token exists. CI runs `pip-audit` on the dependencies, and Dependabot proposes updates.
