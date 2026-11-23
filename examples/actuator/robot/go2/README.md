<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Go2 robot actuator sysid study

This example identifies 24 effective position-actuator gains (Kp/Kd for 12 joints)
using Newton's CUDA `SolverMuJoCo` backend and the existing calibration CMA-ES,
qualification, reload and acceptance lifecycle. It is a step toward robot system
identification. It does not integrate the official MuJoCo sysid optimizer.
See [the robot adapter contract](../README.md) for supported MJCF/URDF models and
[the experiment report](REPORT.md) for measured outcomes and limitations.

## Prepare the environment and public inputs

Run from this repository checkout; the examples are not installed in the wheel.
Use an isolated environment on a CUDA-compatible host:

```bash
uv sync --python 3.13 --extra robot --extra cma --extra dev
uv run --no-sync -m examples.actuator.robot.go2.prepare --output runs/go2-input
```

The optional dependencies pin Newton, Warp, MuJoCo, mujoco-warp, trimesh and Pillow.
The historical study also used Python 3.13.5, NumPy 2.5.4, SciPy 1.18.1 and cma 4.5.0;
see [constraints-study.txt](constraints-study.txt) when reproducing that environment:

```bash
uv venv --python 3.13.5 .venv-study
uv pip install --python .venv-study/bin/python -e '.[robot,cma]' \
  --constraint examples/actuator/robot/go2/constraints-study.txt
.venv-study/bin/python -m examples.actuator.robot.go2.prepare --output runs/go2-input-pinned
```

`sources.json` pins both public repositories to immutable revisions and records
size/SHA-256 for each downloaded file. The downloader retains upstream licenses;
no measurements or Go2 meshes are bundled here. A modified cache or upstream file
fails verification. The `.pt` reader checks the complete archive hash and reads only
the known numeric storages of these exact archives, without loading pickle or
requiring PyTorch. It is intentionally not a loader for arbitrary `.pt` datasets.

Preparation outputs:

- `recordings.json`: excitation 1/2 for training and the complete excitation 3 for
  validation. Each capture is one acquisition group. All 4,800 commands are kept;
  training observations use every fourth sample plus the final sample (1,201).
  Validation uses all 4,800 samples. Position/velocity units are rad and rad/s.
- `derived/`: fixed-base, stateless native position-PD model, with passive viscous
  damping removed so fitted Kd represents total effective viscous response. Mass,
  inertia, rotor armature, Coulomb friction and joint limits remain fixed. See
  [recipe.json](recipe.json). Contacts are disabled by the robot adapter.
- `imported/`: frozen, hashed model resources and reconstructable `asset.json`.
- `config.json`: explicit baselines, bounds, controller timing, observation scales,
  physics refinement and physical acceptance gates.

The timestamps are reconstructed from the nominal **200 Hz** sample rate. Actual
command, sensor and transport timing are unavailable. The level, suspended mount
is assumed. Results identify equivalent gains under this model and timing policy;
they do not establish manufacturer firmware gains or hardware safety limits.

## Run without an agent or a persistent connection

A single comparison uses these commands:

```bash
uv run --no-sync -m examples.actuator.robot.go2.study \
  --prepared runs/go2-input --output runs/go2-4096 \
  --batch-sizes 64 512 --sigmas 0.02 --seeds 42 --candidates 4096 \
  --device cuda:0 --worlds 1024 --max-seconds 10800
```

The matrix matching the overnight search budgets is:

```bash
nohup uv run --no-sync -m examples.actuator.robot.go2.study \
  --prepared runs/go2-input --output runs/go2-matrix \
  --batch-sizes 64 128 256 512 --sigmas 0.01 0.02 0.05 --seeds 42 \
  --candidates 2048 --worlds 1024 --max-seconds 10800 \
  > runs/go2-matrix.log 2>&1 < /dev/null &
```

Use `.venv-study/bin/python` instead of `uv run --no-sync` if using the constrained
environment above. For seed checks, use `--batch-sizes 64 --sigmas 0.02 --seeds 7 42 123`.
`worlds` is simulator capacity; `batch-sizes` are CMA-ES populations. They are
independent. Candidate budgets must be divisible by each population: 4,096 means
64 generations at population 64, or 8 generations at population 512. Sigma is the
initial scale in normalized parameter coordinates; covariance and step size adapt
throughout search. Every job starts from Kp=25, Kd=1.5, without a warm start.

The program runs one independent child process per configuration. It calls the
unchanged `Tuning.run()` and keeps the full validation record available to the
final acceptance stage, never to CMA-ES scoring. There is no early stopping target
or patience limit; candidate/generation/time budgets still apply. Each accepted
release is replayed in another process, requiring per-capture objective agreement
within 1e-6 as well as the physical gates. Prefix comparisons across separate studies
do not block validation. This differs from the historical training-only screening
and separate shortlisted validation described in the report.

The study records the grid and input hashes before launching workers, checks those
inputs between jobs, and appends each terminal outcome to `summary.json`. Each job
contains configuration, logs, a core run directory, convergence summary, saved-trace
response/residual HTML, and an accepted release plus replay evidence when applicable.
The core run records resolved backend versions, resource hashes and adapter hashes.
Downloads are the only network-dependent step. `nohup` permits SSH disconnection;
the server itself must remain running. Existing study directories are never
replaced. An interrupted job is not resumed from a CMA-ES checkpoint: choose a new
output directory and rerun the desired configurations.

`--max-seconds` bounds a core search; the parent watchdog adds 1,200 seconds for
finalization and bounds accepted-release replay separately at 1,200 seconds.
Timeouts/nonzero worker exits are recorded and later jobs continue. A failed
infrastructure or replay check makes the study exit nonzero. Calibration rejections
are valid experimental outcomes and do not by themselves make the supervisor fail;
always inspect `acceptance` in the summary. A time-budget stop can produce fewer
than the requested candidates, also recorded in the summary.

Full trajectories and checkpoints remain local. They are not automatically deleted
or uploaded. The historical 14-job screening produced approximately 224 GiB before
compression; plan storage accordingly. Do not commit output directories. CPU mode
is available for diagnostics with `--device cpu --worlds 1`; it is not a practical
replacement for the full Go2 GPU study.

## Verification

```bash
uv run --no-sync -m newton_calibration.tests -k go2 -v
uv run --no-sync -m newton_calibration.tests -k robot -v
ROBOT_GPU_TESTS=1 uv run --no-sync -m newton_calibration.tests -k robot_gpu -v
```

The Go2-specific tests exercise hash mismatch rejection, numeric-storage loading,
model derivation, fixed physics across search settings, full-generation budgets,
worker failure and watchdog timeout. Robot tests cover parameter readback,
control timing, CPU/GPU agreement, reset isolation, held-out independence,
physical gates and relocatable reconstruction. Synthetic recovery, backend
integration and this public-data exploration are separate kinds of evidence.
