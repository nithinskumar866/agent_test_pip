# Releasing

agenttest is published to PyPI as **`pytest-agenttest`** (the name `agenttest` belongs to an unrelated
project). The import name and the CLI stay `agenttest`.

## One-time setup: trusted publishing

Releases are uploaded by `.github/workflows/release.yml` using PyPI trusted publishing (OIDC), so no API
token is stored anywhere.

1. On PyPI, open **Your account → Publishing → Add a new pending publisher** and enter:
   - PyPI project name: `pytest-agenttest`
   - Owner: `nithinskumar866`
   - Repository: `agent_test_pip`
   - Workflow: `release.yml`
   - Environment: `pypi`
2. Do the same on [TestPyPI](https://test.pypi.org) with environment `testpypi`.
3. In the GitHub repository, open **Settings → Environments** and create `pypi` and `testpypi`. For
   `pypi`, add yourself as a required reviewer, so every release waits for your approval.

## Cutting a release

1. Make sure CI is green on `main`.
2. Update `__version__` in `agenttest/__init__.py` and move the CHANGELOG's "Unreleased" entries under
   the new version.
3. Rehearse on TestPyPI: **Actions → Release → Run workflow → testpypi**, then
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pytest-agenttest`.
4. Tag and push:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

   The workflow runs the tests, checks that the tag matches `__version__`, builds and checks the
   distributions, and publishes to PyPI once the `pypi` environment is approved.

A published version can't be replaced on PyPI, only yanked, so always rehearse on TestPyPI first.

## Building locally

```bash
pip install -e ".[dev]"
ruff check . && mypy && pytest
python -m build && twine check --strict dist/*
```
