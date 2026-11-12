<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Actuator calibration example

[Home](../../README.md) · [User guide](../../docs/user-guide.md)

This example identifies actuator and drive parameters of a fixed-base robot arm by replaying
recorded joint position commands in Newton and comparing simulated with measured joint
positions. It uses the same members and the same `Tuning.run()` lifecycle as the synthetic
demos and the cable example.

## 1. What is calibrated

For every configured joint, five quantities are fitted independently:

| Parameter | Newton model array | Unit (revolute joint) |
|---|---|---|
| `<joint>.stiffness` | `joint_target_ke` | N m/rad |
| `<joint>.damping` | `joint_target_kd` | N m s/rad |
| `<joint>.dry_friction` | `joint_friction` | N m |
| `<joint>.viscous_friction` | `joint_damping` | N m s/rad |
| `<joint>.armature` | `joint_armature` | kg m² |

Joint groups in the configuration share bounds, baselines and transforms; they do not tie
joints together. A bound with equal limits fixes that quantity. Effort and velocity
limits and the command delay (in control steps) are fixed parameters, so every world
reads them back from the Newton model before it runs.

## 2. How the case maps to the members

| Member | This example |
|---|---|
| Evidence | One Experiment per recording. `reference` is the measured joint position after each control interval, `inputs` hold the commanded positions and the measured initial state. The first `buffer_time` is masked. Weights equal scored sample counts. Each recording file is its own acquisition group. |
| Parameters | 30 parameters for SO-101 (6 joints × 5); gains use a log transform. |
| Template | Physics frequency, MuJoCo solver, integrator and iterations; control frequency (equal to the data rate); fixed clamps and delays; refinements to 2× physics frequency and 2× solver iterations. |
| Asset | `ArticulationAsset`: the source USD with baseline values applied. `export` writes a sparse `.usda` layer over the source; `reload` reads that layer through Newton's USD importer without overrides; `release` relocates the layer into `release/asset.usda` with a relative sublayer path. |
| Rollout | `Rollout(worlds=world_count)` on the batched path: `ArticulationAsset.create_batch_environment` reuses one replicated Newton model per Template, with one candidate per world; a smaller final chunk repeats its last candidate in the unused worlds. Commands are uploaded once per recording; joint and body positions are recorded on the device. |
| Evaluator | Standard squared error with 1 rad scales, i.e. pooled position MSE in rad² over all scored samples and joints. |
| Optimizer / Decision | CMA-ES with one generation per population; generation, time and stagnation budgets from the configuration. |
| Validator | Per-recording MSE and maximum error thresholds on the training retest and on the held-out recordings. |
| Play | Body positions of the baseline and calibrated replays as a USDA point animation. |

Numerical qualification uses a pointwise tolerance in radians,
`pointwise_precision_tolerance_rad`, converted to Evaluator feature units as
`tolerance / sqrt(N · J)` for N scored training samples and J joints.

## 3. Data and asset

[configs/so101.json](configs/so101.json) uses the `nvidia/Anchor-Lab` Hugging Face dataset
at a pinned revision: six of the dataset's `train` recordings for search and the eight
`heldout` recordings for validation. Every file is checked against its declared sha256.
Recordings are long-form Parquet (`time_ns`, `experiment`, `field`, `value`) and are
resampled by linear interpolation onto a uniform 0.02 s grid spanning the overlap of all
recorded fields. The example uses each joint's `command_q` and `actual_q`, in radians.

Each fitted joint must move in the training data. The table gives the command range
(rad) and measured velocity RMS (rad/s) per joint after resampling.

| Training recording | Duration | Rotation | Pitch | Elbow | Wrist_Pitch | Wrist_Roll | Jaw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random-waypoints` | 41 s | 1.14 / 0.26 | 0.49 / 0.14 | 0.56 / 0.16 | 0.74 / 0.25 | 0.94 / 0.34 | 0 / 0 |
| `staircase-motion` | 41 s | 1.60 / 0.14 | 0.25 / 0.04 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| `wave-motion` | 34 s | 0 / 0 | 0.50 / 0.07 | 0.30 / 0.05 | 0 / 0 | 0.70 / 0.54 | 0 / 0 |
| `gripper-cycles` | 71 s | 0.30 / 0.03 | 0.50 / 0.05 | 0.20 / 0.02 | 0 / 0 | 0 / 0 | 0.80 / 0.49 |
| `multisine` | 120 s | 0.24 / 0.07 | 0.24 / 0.14 | 0.24 / 0.17 | 0.24 / 0.24 | 0.24 / 0.30 | 0.24 / 0.36 |
| `chirp-sweep` | 240 s | 0.30 / 0.39 | 0.30 / 0.45 | 0.30 / 0.39 | 0.30 / 0.39 | 0.30 / 0.39 | 0.30 / 0.39 |

Without the last three recordings the Jaw is never commanded during training, and its
parameters cannot be identified.

The robot is the dataset's `robot_assets/so101_camera_mass_23g_at_camera.usd`, an SO-101
with a 23 g camera payload. It is downloaded from the same revision and checked against
`asset.usd_sha256`; `--usd` substitutes a local file with the same hash.

## 4. Install

The CPU tests for configuration and data handling need the `actuator` extra:

```bash
python -m pip install -e ".[cma,actuator,dev]"
```

Simulation additionally needs the repository's pinned Newton revision with OpenUSD
import support and the MuJoCo solver. The unversioned `newton[sim]` resolves to the
pinned revision and adds the MuJoCo and MuJoCo Warp versions that revision requires:

```bash
python -m pip install -e ".[cma,newton,actuator]" "newton[sim]"
```

## 5. Run

```bash
python -m examples.actuator_calibration.run --workspace workspace/actuator
```

The first run downloads the pinned recordings and robot USD into the Hugging Face cache;
`--data-root` reads them from a local directory with the same relative paths instead. `--device` selects
the Warp device. The command exits with 0 only for an accepted result. Outputs follow the
[standard run layout](../../docs/user-guide.md#7-read-the-artifacts), with
`candidate_asset.usda` and, if accepted, `release/asset.usda`.

With 30 free parameters the identifiability check can find a rank-deficient local
response and stop the run as `inconclusive`. Reduce or fix parameters, or set
`analysis.allow_rank_reduction`, as an explicit configuration decision.

## 6. Limitations

- No calibration result is archived for this example, and the acceptance thresholds in
  the configuration are declared before any run, not derived from one.
- The Newton population and the USD export/reload round trip are not exercised by
  CPU CI; they require Newton and a simulation device.
- Only scalar (one-coordinate, one-DOF) joints and position-target PD drives are supported.
- Replay starts from the measured joint positions with zero velocity.
- Held-out recordings come from the same dataset revision and robot; their acquisition
  sessions are not documented beyond the file names, so each file is treated as its own
  acquisition group.
