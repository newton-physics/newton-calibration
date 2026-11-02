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

The default Ruff checks use the shared Newton profile in `ruff-newton.toml`.

## Tests

```bash
uv run --extra cma --extra dev -m pytest -q
uv run --extra cma --extra dev -m pytest -q tests/test_workflow.py      # one file
python -m examples.cable_calibration.verify                             # archived cable scoring, CPU
```

The CPU suite uses synthetic cases and archived cable evidence. Run new Newton simulations separately with the pinned solver and record the hardware and scope of validation.

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- Changelog fragments (`changelog/README.md`) start with the first release; no fragment is needed until then.

## Portable, project-owned content

- Do not commit developer-machine paths, usernames, private environment names,
  or machine-specific setup assumptions. Use repository-relative paths,
  user-supplied paths, or clearly labeled generic examples.
- Do not reference personal repositories, personal forks, or personal hosted sites
  in documentation, code, configuration, reports, or provenance metadata.
  Use project-owned upstream repositories or bundled, repository-relative artifacts.
- Do not name individual contributors or coordination contacts in maintained prose,
  comments, examples, or report labels. Describe roles and technical responsibilities.
- Keep required copyright and third-party license notices intact. Scientific names
  and standard technical terminology are not contributor references.
- Retain reproducibility information: dependency versions, device requirements,
  source revisions, patch identities, units, and experimental conditions.
  Preserve retained original-evidence and historical-run hashes; they identify source
  records, not current exports. Use Git to track bundled-file revisions.

## Documentation scope

- Repository documentation describes current behavior, installation, usage,
  reproducibility, and limitations.
- Put one-time implementation plans, coordination notes, migration proposals,
  and dated validation summaries in the PR description.
- Link bundled reports and source evidence with relative paths. Do not imply
  that a hosted site or publication workflow exists unless it is configured.
- Check Markdown and HTML links, retained provenance records, and all tracked text
  for personal or machine-specific references before submitting changes.

## Focused changes and validation

- Follow CONTRIBUTING.md and preserve the existing public package and CLI.
- Keep the common Tuning lifecycle; extend members for case-specific behavior.
- Preserve measured data, numerical settings, archived scores, and solver behavior
  during documentation cleanup. Never reinterpret a historical rejection as acceptance.
- Run pre-commit and checks relevant to the change. For report/source-link edits,
  verify local links, retained provenance records, and archived scores. A documentation-only change
  does not require rerunning a calibration search.
