# Repository preparation checks

Checked on 2026-10-14 with Python 3.12 and Torch 2.9.1+cpu.

- Tests: 31 passed, 1 skipped because Torch CUDA was unavailable.
- Ruff check and format check: passed for core, tests, and the new case factory/verifier. Historical backend snapshots remain byte-preserved.
- Both documented synthetic CLI demos completed with accepted results: sliding friction 0.35; oscillator stiffness 5.00463360294 and damping 0.69668629958.
- CPU archive verification reproduced all 14 cable traces and the 76.0088% B* →C objective reduction. A regression test confirms that a changed prediction is rejected.
- All four cable factories constructed real Tuning instances in fresh processes without simulation or workspace creation.
- Applying the SLS patch to the pinned base reproduced all three recorded solver-file hashes.
- All five skills passed quick_validate.py; local Markdown/HTML links resolved.
- Editable source usage and a standalone core wheel build succeeded.
- Repository text was checked for developer-machine paths and common credential patterns; no matches were found.
- English documentation was checked throughout; cable_calibration_zh.html is the intentional language exception.

No new Newton calibration search was run during repository preparation. Historical cable outcomes and numerical limitations remain unchanged. These packaging checks do not provide new physical validation, full raw-data preprocessing reproducibility, or statistical confidence intervals.

See the [user guide](user-guide.md) for commands and the [cable example](../examples/cable_calibration/README.md) for simulation dependencies and scope.
