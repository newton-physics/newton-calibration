<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Box friction calibration with Newton XPBD

Estimate a box–plane friction coefficient from analytical position trajectories using Newton's `SolverXPBD`. The quickstart explains model construction, evidence, member assembly, and interpretation of the output. [Quickstart](../../docs/quickstart.md)

## Install and run

From the repository root, use Python 3.11 or newer.

```bash
python -m pip install -e ".[newton]"
python -m examples.box_friction_calibration.run --workspace workspace/newton-sliding
```

The example runs on the CPU with one worker by default. Add `--capacity 4` to run up to four episodes concurrently. The first run compiles Warp kernels. The command prints acceptance, the stopping reason, fitted parameters, the run directory, and the released asset path. Exit code `0` means acceptance. A completed rejected or inconclusive run returns `1`.

The checked environment uses Linux x86_64, Python 3.12, Newton `1.9.0.dev0` at revision `c53345412d401232beff0bc71dcb39ce554fdf96`, and Warp `1.17.0`. The `newton` extra records these versions in `pyproject.toml` so the installation command selects the tested runtime. Warp 1.17 is supported by this Newton revision and supports NVIDIA R545–R579 drivers, while Warp 1.18 requires R580 or newer for GPU execution. [Example dependencies](../../pyproject.toml) · [Newton source revision](https://github.com/newton-physics/newton/commit/c53345412d401232beff0bc71dcb39ce554fdf96) · [Newton installation requirements](https://github.com/newton-physics/newton/blob/c53345412d401232beff0bc71dcb39ce554fdf96/README.md#requirements)

## Model and synthetic evidence

The scene contains a 0.2 × 0.2 × 0.1 m box of density 1000 kg/m³ on a horizontal ground plane. Z points upward, gravity is 9.81 m/s² downward, and the box starts level with a prescribed X velocity. Position observations use the body's center X coordinate in meters.

The adapter assigns the candidate coefficient to `ShapeConfig.mu` on both the box and the ground. The pinned XPBD solver averages the two contacting shape coefficients. Assigning them the same value gives one pair coefficient to estimate. Torsional and rolling friction coefficients are zero. The adapter reads friction and gravity back from `Model.shape_material_mu` and `Model.gravity`. [Adapter](backend.py) · [Pinned contact implementation](https://github.com/newton-physics/newton/blob/c53345412d401232beff0bc71dcb39ce554fdf96/newton/_src/solvers/xpbd/kernels.py#L2630)

`run.py` generates each reference from constant deceleration `a = 0.35 * 9.81`. With `moving_time = min(time, initial_speed / a)`, the position is `initial_speed * moving_time - 0.5 * a * moving_time**2`. It samples 41 times from 0 to 1 s. Training uses initial speeds 1.2 and 2.1 m/s. Validation uses 1.7 m/s. These noiseless synthetic trajectories test parameter recovery and a held-out initial condition under the stated model assumptions. Physical calibration requires measured evidence. [Evidence generation and configuration](run.py)

## Parameter search and numerical checks

The friction baseline is `0.15`, and a 41-level grid covers `[0.05, 0.65]`. Each candidate runs with a 1/480 s timestep, eight XPBD iterations, and float32 state. Numerical checks independently halve the timestep and double the iterations, at the initial qualification points and again at the selected candidate. The parameter readback relative tolerance is `1e-6` to accommodate float32 storage.

The evidence scale is 0.03 m. The validator requires per-experiment RMSE at most 3 mm and maximum error at most 5 mm. The numerical tolerance corresponds to a maximum position change of 1.5 mm across each refinement. Its feature-space expression accounts for the two equally weighted training experiments and 41 samples in each. The search target is normalized mean squared error `0.005`.

The expected result is `accepted`, `target_reached`, and friction `0.35`. Inspect `run_state.json` for the actual qualification, identifiability, training retest, validation, and numerical retest reports. The run also saves observations, predictions, the assembled configuration, and the exported asset. [Result files](../../docs/api-reference.md#result-files)

The quickstart also explains how to schedule independent CPU environments with `Rollout.capacity` and how to implement a Newton `BatchEnvironment` for multiple candidate worlds on a GPU. It covers parameter binding, per-world ground friction, output ordering, chunk sizes, and verification of the switch. [Replace single env rollout with multi env rollout](../../docs/quickstart.md#replace-single-env-rollout-with-multi-env-rollout)

After installation, inspect a run with its printed directory name.

```bash
newton-calibration inspect workspace/newton-sliding/<run-id>
```
