<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Single-joint actuator calibration

Fit equivalent position-PD gains to independently generated or offline recorded
motion. This checkout-only recipe subclasses the public calibration members.
`TuningActuator` assembles them and inherits `Tuning.run()` unchanged.

## Run

From the repository root, install the pinned Newton and optional CMA-ES dependencies:

```bash
uv sync --extra newton --extra cma --extra dev
uv run --no-sync -m examples.actuator.run --workspace workspace/actuator
```

The default CPU demo fits two one-second analytical captures and validates on a
third excitation. It returns exit code 0 only for `accepted`. Inspect
`result.json`, `run_state.json` and `actuator-report.html` in the printed run
directory. `--optimizer grid` selects a 21 × 21 debugging grid; its resolution
does not guarantee the parameter accuracy of the recovery test.

To run the same system-identification experiment through Newton's
`SolverMuJoCo`:

```bash
uv sync --extra mujoco --extra cma --extra dev
uv run --no-sync -m examples.actuator.run --solver SolverMuJoCo \
  --workspace workspace/actuator-mujoco
```

The `mujoco` extra includes the pinned Newton and Warp dependencies, plus
MuJoCo 3.12.0 and MuJoCo Warp 3.12.0 (both Apache-2.0). Newton requires both
packages even for its MuJoCo-C CPU path. GPU execution is not supported by this
recipe yet. The default solver remains `SolverFeatherstone`.

Dependencies are Newton 1.6.0.dev0 at
`d37f4d3d341ccce1e06a1dff21e9a054759b4855` and Warp 1.17.0, as declared by
the repository's `newton` extra. Newton state and control arrays use float32;
MuJoCo-C internally uses double precision, with state synchronized from Newton
on every physics step. Scoring uses NumPy float64. The recipe requires no CUDA GPU. Imported backend versions,
Git installation provenance, adapter hashes and the resolved protocol are recorded
under `resolved.json`.

## Physical and execution contract

The fixed-base rotor has one revolute joint, known inertia and passive viscous
damping, no contacts, and zero gravity. At each controller tick:

```text
torque = clip(kp * (position_target - q) - kd * qd, -torque_limit, torque_limit)
inertia * qdd = torque - damping * qd
```

Torque is held until the next tick. Native joint drive gains, armature, friction,
joint-limit forces and solver angular damping are disabled. Passive damping is
applied by Newton once. The controller uses simulated state, never recorded state
as feedback. Each episode creates a fresh model and starts from the explicitly
recorded q/qd at time zero.

For `SolverMuJoCo`, the frozen numerical protocol explicitly selects
`use_mujoco_cpu=True`, `integrator="implicitfast"`, `disable_contacts=True`
and `update_data_interval=1`. Its compiled model has no native actuators:
the held torque enters through Newton's `Control.joint_f`. Inertia and
passive damping are read back from the active MuJoCo model, while kp/kd are
read from the sampled controller. MuJoCo simulation warnings fail the episode,
including cases where the engine resets divergent state to finite values.

The synthetic protocol uses inertia 0.01 kg·m², damping 0.02 N·m·s/rad, torque limit
2 N·m and a 2 ms control period. True gains are kp=2 N·m/rad and
kd=0.15 N·m·s/rad. The physics step is 0.25 ms and its declared refinement is
0.125 ms. Refinement preserves the controller period and input hold semantics.
Both physics steps must divide the controller period exactly.
The independent reference analytically integrates the viscously damped rotor under
constant torque between controller ticks; it does not call Newton.

The synthetic settings are also available in
[configs/synthetic.json](configs/synthetic.json). They are demo settings, not
defaults for hardware. Inertia and damping must be independently known: jointly
freeing damping and kd can confound their effects. The result represents equivalent
gains of this model, not necessarily the drive firmware's internal gains.

No lag, pure delay, integral control, Coulomb friction, backlash, sensor bias or
contact model is identified. Residuals and independent validation determine whether
this model is adequate for a particular recording.

## Offline recordings

```bash
uv run --no-sync -m examples.actuator.run \
  --recordings /path/to/recordings.json \
  --config /path/to/protocol.json \
  --workspace workspace/actuator-recorded
```

The protocol uses the fields in `configs/synthetic.json`, all explicitly supplied.
Set `"solver": "SolverMuJoCo"` in the protocol to select that backend.
Offline runs and replay use the recorded solver, so `--solver` is reserved for
the synthetic demo. Earlier configurations without `solver` retain Featherstone.
Choose physical values, bounds, observation scales and acceptance thresholds for
the actual setup. All protocol values and initial states use SI units.

The recording document has `schema: "actuator-recordings/1"` and a `records`
list. Each record has this structure (the short signal below illustrates the
format only and is not an identification experiment):

```json
{
  "id": "capture-01",
  "group": "acquisition-01",
  "split": "train",
  "initial_state": [0.0, 0.0],
  "command": {
    "times": [0.0, 0.002, 0.004],
    "values": [0.0, 0.02, 0.04],
    "unit": "rad"
  },
  "position": {
    "times": [0.0, 0.002, 0.004],
    "values": [0.0, 0.0, 0.000016],
    "unit": "rad"
  },
  "velocity": {
    "times": [0.0, 0.002, 0.004],
    "values": [0.0, 0.0, 0.016],
    "unit": "rad/s"
  }
}
```

Use `split: "validation"` for independent captures. IDs must be unique and
acquisition groups must not cross the split. Slicing one recording does not make
an independent holdout.

Command and observation times are seconds relative to the same zero. Every signal
must have finite values and strictly increasing nonnegative times. Commands must
start at zero and cover the last observation. At controller ticks they are sampled
by zero-order hold, including a command at that exact tick. Position and command
accept `rad` or `deg`; optional velocity accepts `rad/s` or `deg/s`. The
single joint is named `actuator`, with positive rotation about +Z.

Position times define scoring times. Measured velocity, if present on another
time axis, is linearly interpolated to those times without extrapolation. If absent,
only position is scored; velocity is never silently derived from position.
Timestamps, alignment and filtering should be established before fitting.

Scoring preserves the core normalized RMSE, maximum error and weighted residual
features, and adds worst-per-capture position/velocity RMSE and maximum errors in
SI units. Missing velocity contributes zero only to the optional velocity metrics;
the channel map records its absence. Validation checks every capture independently,
including the normalized core gates. Adjust observation scales together with the
physical thresholds; both sets of gates must pass.

## Outputs and replay

```bash
uv run --no-sync -m examples.actuator.run --report workspace/actuator/RUN_ID
uv run --no-sync -m examples.actuator.run --replay workspace/actuator/RUN_ID
```

Reporting reads saved baseline/candidate traces and does not construct a simulator.
The self-contained HTML contains command/response, before/after residuals and
validation scope. USDA output from the core Play member is a schematic rotor
indicator, not a complete robot asset.

Replay reconstructs `candidate_asset.json` and the saved template, simulates all
recorded experiments, repeats the per-capture validation gates, and writes
`actuator-replay.json`. It is diagnostic and
does not change the original acceptance or publish an asset.

An accepted release retains the model/controller configuration, fitted parameters,
template and validation manifest. The JSON format requires this adapter and the
recorded Newton dependencies to reconstruct physics. Controller protocol is runtime
configuration, not a new USD schema; the result is not advertised as a portable USD
actuator. The runtime constructs the simple rotor directly without external meshes.

`backend.py` owns solver construction, sampled control and native parameter
readback. `recording.py` owns data alignment and units; `evaluation.py` owns
physical error gates. `TuningActuator` composes these members and uses the same
search, identifiability, export and validation lifecycle for either solver.
This provides rollout-based sysid of the equivalent PD controller.
[Multi-joint robot calibration](robot/README.md) adds a fixed-base MJCF/URDF subset,
native position-actuator gain fitting and relocatable resource bundles through a
separate entry point. GPU batches and MuJoCo's separate `sysid` optimizer are not
implemented.

## Tests and scope

```bash
uv run --no-sync -m unittest newton_calibration.tests.test_actuator -v
ACTUATOR_RECOVERY_TESTS=1 ACTUATOR_TEST_WORKSPACE=workspace/actuator-checks \
  uv run --no-sync -m unittest newton_calibration.tests.test_actuator.TestActuatorRecovery -v
ACTUATOR_RECOVERY_TESTS=1 ACTUATOR_TEST_SOLVER=SolverMuJoCo \
  ACTUATOR_TEST_WORKSPACE=workspace/actuator-mujoco-checks \
  uv run --no-sync -m unittest newton_calibration.tests.test_actuator.TestActuatorRecovery -v
uv run --no-sync -m newton_calibration.tests
uv run --no-sync -m examples.cable_calibration.verify
```

Regular tests cover the input contract, independent analytical response, refinement,
held control and saturation, A/B/A isolation, readback, deadlines, invalid results,
missing validation, unexcited trajectories and replay in a fresh process. Newton
tests skip only when the optional backend is absent. Full searches are opt-in
because they execute many CPU rollouts; they require gains within 2% from three
initial guesses, plus accepted independent validation.
MuJoCo contract tests additionally check native-model readback, absence of
duplicate native drives, nonzero initial conditions, simulation warnings and
solver-preserving replay in a new process.

The synthetic position acceptance thresholds are RMSE ≤0.01 rad and maximum
error ≤0.03 rad; velocity thresholds are 0.05 and 0.15 rad/s. These verify this
implementation against synthetic evidence. No real actuator recording is bundled,
and synthetic acceptance does not establish real-hardware accuracy.
