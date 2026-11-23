<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Robot import and actuator calibration

This adapter imports fixed-base, open-tree MJCF or URDF robots into Newton's
`SolverMuJoCo` CPU or CUDA backend. It freezes source files and dependencies, resolves
named bodies, joints and native actuators, and exports a relocatable asset.
`TuningRobot(Tuning)` adds multi-joint position-PD calibration and native MJCF
position-actuator gain fitting through the existing calibration lifecycle.
The existing [single-joint calibration](../README.md) remains available.
For public measured recordings, see the [Go2 sysid study](go2/README.md),
including a pinned downloader, unattended training script and [experiment report](go2/REPORT.md).

## Import a model

Run from a repository checkout; examples are not part of the installed wheel.

```bash
uv sync --extra robot --extra cma --extra dev
uv run --no-sync -m examples.actuator.robot.run \
  --model examples/actuator/robot/fixtures/two_link.xml \
  --gravity 0 0 -9.81 --output workspace/robot-mjcf
uv run --no-sync -m examples.actuator.robot.run \
  --model examples/actuator/robot/fixtures/two_link.urdf \
  --package two_link=examples/actuator/robot/fixtures \
  --gravity 0 0 -9.81 --output workspace/robot-urdf
```

Choose a new output directory for every invocation. Gravity must be explicit in
m/s². `--package NAME=DIR` resolves URDF `package://` resources locally and can be
repeated for different packages. No ROS installation or network lookup is used.
Import warnings are errors, so an unsupported conversion cannot pass unnoticed.
The `robot` extra includes Pillow for OBJ texture coordinates and image textures;
the exported dependency manifest records its version alongside trimesh.

The output contains:

- `asset.json`: adapter type, empty parameter set, construction configuration,
  dependency versions/revision and resource hashes.
- `robot-<asset_id>/`: normalized model entry, mesh/texture files, and original
  XML files preserved byte-for-byte under hashed names. Internal references are
  relative to this directory.
- `import-report.json`: status `imported`, asset identity, source names mapped to
  Newton and MuJoCo indices, coordinate addresses and compiled physical values.
  It is an inspection report and makes no calibration acceptance claim.

Move `asset.json` together with its `robot-<asset_id>` directory. Rebuild without
the original model, package directories or candidate workspace:

```bash
uv run --no-sync -m examples.actuator.robot.run \
  --reload workspace/robot-mjcf/asset.json --output workspace/robot-reloaded
```

Reconstruction verifies every resource hash and rejects missing or extra files,
symlinks, and changed backend versions/revisions. Install the recorded dependency
versions to reproduce an export; no version migration is attempted. The JSON is
runtime configuration for this adapter, not a USD schema or a portable USD robot
asset. Its construction depends on the matching adapter code and Newton revision.

## Import contract

- Supports named bodies/links and named hinge (MJCF), revolute, continuous or
  fixed (URDF) joints. Body and joint names must be unique. Fixed joints are
  retained; floating bases, ball/prismatic joints, mimic joints, transmissions,
  closed chains, tendons, plugins and deformable structures are rejected.
- Preserves supplied inertia definitions and disables fixed-joint collapsing.
  Explicit gravity overrides source simulation options. Contacts are disabled;
  the solver uses `implicitfast` and updates Newton state every step. The source
  file's solver/timestep settings do not define a calibration protocol.
- MJCF native `motor`, `position`, `velocity` and `general` actuators must be named and
  reference scalar joints directly. Import uses `ctrl_direct=True`; inspection
  verifies the actual compiled transmission, control mapping, gear, gain/bias and
  limits. Native calibration links proportional gain and position bias and
  separately updates velocity bias; transmission gear and limits remain fixed.
- URDF defines no actuator gains. At the pinned Newton revision, its importer
  places URDF damping in `joint_target_kd` and generates position actuators. This
  adapter moves that damping into `joint_damping`, zeros target gains and disables
  the inferred drives. The frozen `urdf_policy` records this conversion. The
  resulting robot has passive damping and no invented actuator gains.
- A MuJoCo actuator name may be empty after Newton compilation. Bindings recover
  the source name through Newton's actual direct-control map and verify the
  target joint. No assumption equates actuator, joint, coordinate and data-column
  indices. `joint_values(values, names, report)` maps complete named rotary state
  vectors in rad or rad/s; partial, duplicated or nonfinite inputs fail.

Resource freezing supports nested MJCF includes, local paths, explicit package
maps, OBJ/STL meshes, and PNG/JPEG textures. OBJ material-library references are
rejected by default. Explicit `--obj-materials mjcf` uses MJCF materials and removes
only OBJ `mtllib`/`usemtl` directives from normalized copies. Original OBJ bytes,
normalization records and hashes are retained; geometry is unchanged. This option
is restricted to MJCF. Source LICENSE/NOTICE files are included in the hashed bundle.
Relative asset directories follow the pinned Newton importer's per-file
compiler resolution. Included XML is expanded into the frozen entry while original
bytes remain available for provenance. `strippath`, `fusestatic`, other mesh
formats and unsupported file dependencies fail explicitly. Convert those inputs
before import; no geometry conversion or silent dependency omission is performed.
Visuals may be retained, but this entry point does not provide a rendering viewer.

## Adapter boundaries and verification

`resources.py` owns dependency discovery and hashing. `asset.py` owns construction,
export/reload and release staging. `bindings.py` owns the actual compiled mappings.
These remain example-specific. The only core addition is
`Asset.stage_release(candidate_path, release_path)`, whose default keeps the
existing single-file copy behavior. File-backed subclasses also stage dependent
resources; failures prevent acceptance in the common `Tuning.run()` lifecycle.

```bash
uv run --no-sync -m newton_calibration.tests -k robot -v
uv run --no-sync -m newton_calibration.tests -k asset_release -v
```

The paired two-link fixtures include nontrivial inertias, passive damping, a mesh,
an MJCF include and a URDF package URI. Tests compare compiled masses and full
inertia tensors against known values, joint axes/limits and nonzero-state forced
motion across formats. They check reordered declarations/data columns, native
actuator names, invalid structures, altered bundles and dependency drift. A new
process rebuilds a moved release after source and candidate directories are
removed. These are import and SolverMuJoCo integration checks. Additional
calibration tests are described below. Bundled records are synthetic.

## Multi-joint calibration

Generate runnable inputs for the bundled two-link model:

```bash
uv run --no-sync -m examples.actuator.robot.reference --output workspace/two-link-evidence
uv run --no-sync -m examples.actuator.robot.run \
  --model examples/actuator/robot/fixtures/two_link.xml \
  --gravity 0 0 0 --output workspace/two-link-model
uv run --no-sync -m examples.actuator.robot.calibrate \
  --asset workspace/two-link-model/asset.json \
  --recordings workspace/two-link-evidence/recordings.json \
  --config workspace/two-link-evidence/config.json --workspace workspace/two-link-runs
```

`reference.py` generates two independent training excitations and one validation
excitation by integrating the coupled two-link Lagrange equations with DOP853.
It does not call Newton or MuJoCo to generate truth. Its fixture-specific bounds,
scales, thresholds and numerical protocol are returned by `synthetic_config()`.
The controller updates at 2 ms, independently of the 0.25 ms physics step and
0.125 ms refinement. These are synthetic test settings, not hardware defaults.

Offline input uses `schema: "robot-recordings/1"`, explicit
`provenance.kind: "synthetic" | "measured"`, and a `records` list. Each record has
`id`, `group`, `split`, and `initial_state.position` / `initial_state.velocity`
maps for **all** active joints at time zero, in SI units. `command`, `position`
and optional `velocity` are maps from names to `{times, values, unit}` signals.
Commands and positions accept rad/deg; velocity accepts rad/s or deg/s. Each signal
has its own strictly increasing time axis. The loader takes the union of
observation timestamps and masks absent samples; it does not interpolate
observations or derive velocity. Commands cover the entire experiment from zero
and use zero-order hold. Acquisition groups must not cross training/validation.
Every capture must observe position for all driven joints. Velocity is optional;
validation reports list observed channels and use null for unobserved velocity
errors rather than reporting them as zero.

The explicit configuration declares controller mode, control period, driven joints,
gain baselines, torque limits (equivalent PD), selected parameter bounds, per-joint
position/velocity scales and RMSE/maximum-error gates, both integration steps,
numerical tolerance and search budget. Unselected gains remain fixed. Multiple
parameters use CMA-ES; a single selected gain uses GridSearch. The first two core
normalized metrics and residual features are preserved. Physical gates check each
observed joint independently in every validation capture.

For multiple selected gains, `"optimizer": {"sigma": 0.02}` sets CMA-ES's
initial sampling scale in normalized parameter coordinates. The default is `0.2`;
CMA-ES subsequently adapts its step size and covariance. Sigma must be a finite
number strictly between 0 and 1. Unknown optimizer settings are rejected, and
optimizer settings cannot be specified for the single-gain GridSearch recipe.
The optional `optimizer.seed` is an integer in `[0, 2**32)` and defaults to 42.
Using the same seed reproduces the proposals under the same fixed environment. For an unrotated linear parameter, initial physical scale is approximately
`sigma * (upper_bound - lower_bound)` before boundary handling. Population size
remains in `execution.batch_size`; changing sigma does not change population or
simulation capacity. Optimizer settings are retained in the evidence protocol.

### GPU batches (experimental)

The optional `execution` configuration selects the backend and batch sizes:

```json
"execution": {"device": "cuda:0", "worlds": 128, "batch_size": 128}
```

`worlds` caps simultaneous candidate worlds per experiment; larger generations
are split into ordered chunks. `batch_size` is the optimizer population size.
They are independent: increasing the latter changes CMA-ES search behavior, not
just execution speed. The search budget must permit complete generations (for
example, `max_trials >= batch_size`) and leave enough rollout/time budget for
numerical qualification, identifiability, retesting and validation. Default CPU
execution remains serial with population 8. Selecting CUDA defaults both sizes
to 128; there is no automatic CPU fallback when CUDA is unavailable.

`BatchEnvironmentRobotGPU` replicates the imported robot into independent Newton
worlds and uses `SolverMuJoCo(use_mujoco_cpu=False, separate_worlds=True)`.
Native gain/bias coefficients have a world dimension. Equivalent-PD feedback and
held torques also run on CUDA. Each batch gets fresh states, controls and solver
buffers. Integration segments use CUDA graphs; observation/command times retain
the CPU protocol, including off-grid samples. Failed states are checked every
integration step and returned as failures in their own world slots. The CPU
model used for import/routing inspection does not simulate candidates.

The existing `Rollout` handles ordering, chunking, deadlines and readback checks;
`Tuning.run()` is unchanged. Scoring and CMA-ES remain on CPU. Traces are collected
on GPU and transferred once at the end of each batch, so trajectory storage grows
with worlds, samples and bodies. Benchmark 128, 256 and 512 on the actual model
before selecting a capacity. Use separate processes for comparable memory peaks:

```bash
uv run --no-sync -m examples.actuator.robot.benchmark \
  --asset /path/to/imported/asset.json --recordings /path/to/recordings.json \
  --config /path/to/config.json --worlds 128 --duration 1 --repeats 2 \
  --output workspace/gpu-benchmark-128.json
ROBOT_GPU_TESTS=1 uv run --no-sync -m newton_calibration.tests -k robot_gpu -v
```

The benchmark uses only a prefix of the first training recording, checks the
first/last candidate against CPU, discards one warmup batch, and reports build,
run and total time. Reported memory is the Warp pool's process high-water mark;
it excludes driver allocations and other applications. It performs no search or
hardware acceptance. A changed backend requires fresh numerical qualification;
CPU qualification and acceptance are not automatically inherited by GPU runs.
Exports retain the selected device and replay requires a compatible CUDA runtime.
Both backends retain the fixed-base, no-contact, stateless-PD scope described here.

`equivalent_pd` computes and holds clipped joint torque from simulated feedback.
Native actuation and drive gains are disabled in the runtime model to avoid double actuation;
source files and passive damping remain unchanged. Each candidate/experiment
constructs fresh state, control and solver objects.

For `native_position`, each drive specifies `kp`, `kd`, and its source `actuator`
name instead of `torque_limit`. Declare `command_space` explicitly:

- `joint_position`: commands use joint names and rad; native control is `gear*q_ref`.
- `actuator_control`: commands use actuator names and actuator-length coordinates;
  no gear conversion is applied to the recorded values.

All source actuators must be explicitly routed once. Only stateless joint position
actuators with fixed gain and affine bias are supported, including the equivalent
`general` PD form. Constant-force offsets or unmatched gain/position bias are
rejected before fitting. Native mode holds the
position command; MuJoCo evaluates feedback every physics step. It does not hold
torque like equivalent-PD mode. Native gains use actuator coordinates: with scalar
gear `g`, unconstrained joint stiffness is `g²*kp`. Control and actuator-force
limits are retained. Bridge tests compare with direct MuJoCo execution using signed
non-unit gears and exercise both limit mechanisms.

```bash
uv run --no-sync -m examples.actuator.robot.calibrate \
  --asset /path/to/imported/asset.json --recordings /path/to/recordings.json \
  --config /path/to/config.json --workspace workspace/robot-runs
uv run --no-sync -m examples.actuator.robot.calibrate --report workspace/robot-runs/RUN_ID
uv run --no-sync -m examples.actuator.robot.calibrate \
  --replay workspace/robot-runs/RUN_ID/release --output workspace/robot-replay.json
uv run --no-sync -m newton_calibration.tests -k robot
ROBOT_RECOVERY_TESTS=1 ROBOT_TEST_WORKSPACE=workspace/robot-recovery-checks \
  uv run --no-sync -m unittest newton_calibration.tests.test_robot_calibration.TestRobotRecovery -v
```

Accepted CLI runs deliver model resources, fitted parameters, template, manifest
and `resolved.json` containing evidence for independent replay. Reports read saved
trajectories and respect observation masks. Comparisons use the declared physics
step; refinement trajectories contribute to numerical checks, not the response curves.
Full recovery tests fit all four gains
from three initial guesses, require each relative error ≤2%, and require independent
validation acceptance. Regular tests cover physics refinement, reset/order isolation,
held controls, saturation, readback, independent time axes, failure gates, and a
moved release revalidated in a new process.

The synthetic search stops at normalized objective `1e-4`, allowing for the
declared integration step's numerical error. This stopping target does not replace
the gain-recovery or independent validation gates. Set `ROBOT_RECOVERY_START` to
`0`, `1` or `2` to run one recovery start separately; by default all three run.

Real-robot acceptance requires an actual robot model, measured command/response
captures, known controller semantics, physical configuration and independently
justified engineering thresholds. Those inputs must be supplied for the selected
hardware. The bundled synthetic evidence cannot establish real-hardware accuracy.

## Six-joint UR5e model integration

The optional UR5e check downloads the public MuJoCo Menagerie model at the commit
pinned in `ur5e.py`, preserving its BSD-3-Clause license and resource hashes:

```bash
uv run --no-sync -m examples.actuator.robot.ur5e --output workspace/ur5e-source
uv run --no-sync -m examples.actuator.robot.ur5e \
  --check workspace/ur5e-source --output workspace/ur5e-checks
uv run --no-sync -m examples.actuator.robot.calibrate \
  --replay workspace/ur5e-checks/delivery --output workspace/ur5e-replay.json
```

The check exercises all six native position actuators with three initial poses and
frequency combinations, compares with direct MuJoCo on the original source model,
and exports a standalone replay bundle. Source OBJ files reference material
libraries absent from upstream; the check explicitly selects MJCF material policy.
`integration-report.json` records `calibration_executed: false`: this verifies the
model and control bridge, without optimization or measured-hardware acceptance.
The generated configuration and records are explicitly synthetic; replace them
with the selected robot's measured captures and justified engineering thresholds
for a hardware calibration run.
