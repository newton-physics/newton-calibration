# Contributing to Newton Calibration

The framework uses the `task_tuning` package and `task-tuning` CLI. Keep pull requests
focused and describe the behavior changed, its validation and any remaining limitations.

## Development

Use Python 3.11 or newer. The tested local cable runtime uses Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cma,cable,dev]"
pre-commit install
pre-commit run --all-files
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m examples.cable_calibration.verify
python -m build
```

Core tests and the archived cable verifier require no Newton installation or GPU.
For CPU-only development, install Torch from its CPU wheel index before installing
the project. The CI workflow uses this setup.

Ruff owns formatting and import ordering; the existing package uses a 100-column limit.
Use descriptive errors, explicit units and real behavioral tests. Do not bypass
`Tuning.run()` with a second calibration loop. Keep expensive Newton simulations separate
from the CPU checks and document the exact solver source, hardware and qualification scope.
Do not change historical acceptance statuses when reproducing old scores.

## Seed-specific conventions

Newton's main repository uses unittest. This first seed retains the existing pytest
suite and its real simulation/acceptance checks; the target repository's test convention
can be agreed with maintainers separately. This is an explicit integration difference,
not a claim that the seed already satisfies every main-repository convention.

The adapters in `examples/cable_calibration/backends/` retain historical source
contents, with byte-identical stage copies shared. Original hashes are preserved;
`archive/provenance.json` maps relocated files through `export_path`. These adapters
are excluded from Ruff. Keep new maintained adapters outside this directory. If an existing snapshot
must change, explain the change, update provenance, and rerun the relevant behavior checks.
The two large HTML reports are retained as evidence and are the only large-file hook exceptions.

GitHub Actions are pinned by commit SHA. CPU CI runs tests, archive verification, both
synthetic demos, package construction and a standalone wheel smoke test. This does not
qualify a Newton/CUDA environment or rerun the cable optimization.

## Licensing and integration

Existing runtime dependencies retain their own licenses: NumPy uses BSD-3-Clause
with bundled components under additional permissive licenses, PyTorch/CMA-ES use
BSD-3-Clause, SciPy uses BSD licensing, and optional Pyglet uses BSD licensing.
Consult the installed distributions for their complete notices; these dependencies
are not relicensed by this repository.

Original contributions are Apache-2.0; retain third-party notices and identify new
dependency licenses. See [NOTICE](NOTICE) for the original contribution and Newton patch.

Keep NumPy and Torch as the existing core runtime; CMA-ES, cable preprocessing and GL
playback remain optional extras. Newton is installed explicitly for the cable example.
Do not silently replace the pinned patched solver with a released package.

Follow [AGENTS.md](AGENTS.md) for portable content and documentation scope.
See the [cable runtime requirements](examples/cable_calibration/README.md#5-construct-or-rerun-a-stage-with-newton)
for the pinned solver dependency. Discuss interface changes using a runnable case and
structured evaluation results before changing the public package.
