# Newton Calibration Development Guidelines

Read and follow the canonical [source code and public API guidelines](CODING_GUIDELINES.rst) before changing or reviewing code. This file contains agent-specific workflow instructions; the linked guide is authoritative for coding and API design.
For reviews, also read and apply the [review guidelines](REVIEW_GUIDELINES.rst).

- Create a feature branch on your fork before committing—never commit directly to `main`. Give the pull request a concise, descriptive title.
- Use imperative mood in commit messages ("Fix X", not "Fixed X"), with a roughly 50-character subject and a body wrapped at 72 characters that explains what and why.
- Verify regression tests fail without the fix before committing.

Run `uvx pre-commit run -a` to lint and format before committing. Use `uv` for all commands; fall back to `venv` or `conda` only if `uv` is unavailable.

```bash
# Install with the optional extras
uv sync --extra cma --extra cable --extra dev
uv run task-tuning demo --case oscillator --optimizer cma_es
```

## Tests

```bash
uv run --extra cma --extra dev -m pytest -q
uv run --extra cma --extra dev -m pytest -q tests/test_workflow.py      # one file
python -m examples.cable_calibration.verify                             # archived cable scoring, CPU
```

Tests that simulate with Newton need a CUDA device and a Newton checkout on `PYTHONPATH`; they skip otherwise.

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Changelog fragments (`changelog/README.md`) start with the first release; no fragment is needed until then.
