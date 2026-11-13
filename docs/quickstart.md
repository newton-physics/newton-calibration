<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Run your first calibration

[Calibration guide](calibration-guide.md) · [API reference](api-reference.md) · [Cable study](../examples/cable_calibration/README.md)

Fit contact friction for a box simulated by Newton’s `SolverXPBD`, read the result, then use the same interfaces with your own observations and model. The reference trajectories are analytical synthetic data with known initial speeds and friction.

## Install

From a repository checkout, use Python 3.11 or newer and install the `newton` extra. Its dependencies in `pyproject.toml` select the Newton revision and Warp version used to verify this example.

```bash
python -m pip install -e ".[newton]"
```

The example runs Newton on the CPU with float32 state. NumPy generates the analytical observations and scores predictions. Optional extras add CMA-ES (`cma`), GL playback (`gl`), or the SciPy utilities used by the cable adapter (`cable`). The first run compiles Warp kernels. [Example setup and scope](../examples/box_friction_calibration/README.md)

## Run the synthetic example

```bash
python -m examples.box_friction_calibration.run --workspace workspace/newton-sliding
```

`SlidingAsset` builds a box and a horizontal ground plane. `SlidingEnvironment` detects contacts with `CollisionPipeline`, advances the model with `SolverXPBD.step()`, and reads the box’s X position from `State.body_q`. `GridSearch` selects candidate friction coefficients for those simulations.

The box measures 0.2 × 0.2 × 0.1 m and has density 1000 kg/m³. It starts level on the plane with an initial velocity along X. The reference trajectories use friction `0.35`, with initial speeds 1.2 and 2.1 m/s for fitting and 1.7 m/s for validation. Both contacting shapes receive the same `mu`, which represents the pair’s friction coefficient under this XPBD configuration.

The command creates a fresh run directory, checks timestep and iteration refinements and local sensitivity, searches for friction, then checks the exported and reloaded candidate.

## Understand the result

The example should return `accepted` and recover friction close to `0.35`. Read the two status fields together.

| Output | What to check |
|---|---|
| `acceptance` | Whether the candidate passed the configured training, validation, and numerical checks |
| `termination_reason` | Why the search or run stopped, such as reaching its target or exhausting a budget |
| `parameters` | The selected parameter values and their meaning under this model and configuration |
| `asset_path` | The accepted asset file, when an accepted release was produced |
| `run_directory` | The observations, settings, scores, and reports recorded for this run |

For example, `target_reached` describes a search stopping condition. Inspect `acceptance` to establish whether the final checks also passed. An accepted synthetic result confirms this example's configured checks. Real prediction and task-transfer claims need the corresponding measured evidence. [Calibration goals](calibration-guide.md#choose-the-goal-and-the-evidence-it-needs)

CLI exit code `0` indicates acceptance. Exit code `1` also covers completed runs with rejected or inconclusive results, so inspect the saved status and reason when diagnosing a run.

## Inspect the saved run

Use the run directory printed by the command. Replace `<run-id>` with that directory's name.

```bash
newton-calibration inspect workspace/newton-sliding/<run-id>
```

Open `result.json` for the result, `run_state.json` for analysis and validation reports, and `resolved.json` for the frozen problem definition. Accepted runs include `release/asset.json`, `release/template.json`, and `release/manifest.json`. Keep them together when reusing a fitted model. [Result file reference](api-reference.md#result-files)

## Assemble the calibration yourself

Save the following as `box_friction_calibration.py` and run `python box_friction_calibration.py`. It shows Newton model construction, synthetic evidence, and all ten calibration members. The bundled command uses the same adapter and assembly. [Adapter source](../examples/box_friction_calibration/backend.py) · [Run source](../examples/box_friction_calibration/run.py)

The adapter consists of `SlidingAsset` and `SlidingEnvironment`. Each candidate and experiment gets a fresh model, solver, collision pipeline, and state pair. Parameter readback uses the model arrays consumed by the solver. Observation snapshots are copied so later CPU simulation steps preserve earlier samples.

```python
import time
from threading import Lock

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverXPBD

from newton_calibration import (
    Asset,
    Decision,
    Environment,
    Evaluator,
    Evidence,
    Experiment,
    GridSearch,
    Parameter,
    Parameters,
    Play,
    Rollout,
    Template,
    Trace,
    Tuning,
    Validator,
)

# Pipeline construction and collision detection both specialize kernels in a shared Warp module.
_collision_lock = Lock()


class SlidingEnvironment(Environment):
    """Simulate a box on a horizontal plane and observe its center position [m]."""

    def __init__(self, parameters, numeric_settings, initial_speed, config):
        with _collision_lock:
            self.dt = float(numeric_settings["dt"])
            half_x, half_y, half_z = config["half_extents"]
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -parameters["gravity"]))
            material = newton.ModelBuilder.ShapeConfig(
                density=config["density"],
                mu=parameters["friction"],
                mu_torsional=0.0,
                mu_rolling=0.0,
            )
            self.body = builder.add_body(xform=wp.transform((0.0, 0.0, half_z), wp.quat_identity()))
            self.box_shape = builder.add_shape_box(self.body, hx=half_x, hy=half_y, hz=half_z, cfg=material)
            self.ground_shape = builder.add_ground_plane(cfg=material)
            self.model = builder.finalize(device=numeric_settings["device"])
            self.solver = SolverXPBD(self.model, iterations=numeric_settings["iterations"])
            self.state_in, self.state_out = self.model.state(), self.model.state()
            velocities = self.state_in.body_qd.numpy().copy()
            velocities[self.body, 0] = initial_speed
            self.state_in.body_qd.assign(velocities)
            self.collision = newton.CollisionPipeline(self.model)
            self.contacts = self.collision.contacts()

    @property
    def applied_parameters(self):
        materials = self.model.shape_material_mu.numpy()
        if materials[self.box_shape] != materials[self.ground_shape]:
            raise ValueError("the box and ground must share the fitted friction coefficient")
        return {
            "friction": float(materials[self.box_shape]),
            "gravity": -float(self.model.gravity.numpy()[0, 2]),
        }

    def run(self, times, *, seed, deadline):
        current_time = 0.0
        positions = []
        for target in times:
            while current_time < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("sliding simulation deadline")
                step = min(self.dt, target - current_time)
                self.state_in.clear_forces()
                with _collision_lock:
                    self.collision.collide(self.state_in, self.contacts)
                self.solver.step(self.state_in, self.state_out, None, self.contacts, step)
                self.state_in, self.state_out = self.state_out, self.state_in
                current_time += step
            # CPU NumPy views share state memory that later steps overwrite.
            positions.append(self.state_in.body_q.numpy()[self.body, :3].copy())
        positions = np.asarray(positions, dtype=np.float64)[:, None, :]
        return Trace(times.copy(), positions[:, :, 0].copy(), positions)


class SlidingAsset(Asset):
    """Construct a fresh Newton model for each candidate and experiment."""

    def create_environment(self, template, parameter_values, experiment_settings):
        settings = template.numeric_settings
        if settings["solver"] != "SolverXPBD" or settings["dtype"] != "float32":
            raise ValueError("SlidingAsset requires SolverXPBD with float32 state")
        if not np.isfinite(settings["dt"]) or settings["dt"] <= 0:
            raise ValueError("the timestep must be positive and finite")
        if not isinstance(settings["iterations"], int) or settings["iterations"] < 1:
            raise ValueError("solver iterations must be a positive integer")
        return SlidingEnvironment(parameter_values, settings, experiment_settings["initial_speed"], self.config)


parameters = Parameters([Parameter("friction", bounds=(0.05, 0.65), baseline=0.15)])
asset = SlidingAsset(
    parameters.as_dict(parameters.baseline),
    config={"half_extents": [0.1, 0.1, 0.05], "density": 1000.0},
)
settings = {
    "slow": {"initial_speed": 1.2, "seed": 1},
    "fast": {"initial_speed": 2.1, "seed": 2},
    "holdout": {"initial_speed": 1.7, "seed": 3},
}
times = np.linspace(0.0, 1.0, 41)
gravity = 9.81
reference_deceleration = 0.35 * gravity
experiments = []
for experiment_id, setting in settings.items():
    speed = setting["initial_speed"]
    moving_time = np.minimum(times, speed / reference_deceleration)
    position = speed * moving_time - 0.5 * reference_deceleration * moving_time**2
    experiments.append(
        Experiment(
            id=experiment_id,
            group=experiment_id,
            reference=position[:, None],
            times=times,
            scale=0.03,
        )
    )

evidence = Evidence(
    train_data=experiments[:2],
    validation_data=experiments[2:],
    sources={"kind": "analytical synthetic sliding trajectories", "reference_friction": 0.35},
)
template = Template(
    numeric_settings={
        "solver": "SolverXPBD",
        "device": "cpu",
        "dt": 1.0 / 480.0,
        "iterations": 8,
        "dtype": "float32",
    },
    discretization={"rigid_bodies": 1, "collision_shapes": 2},
    fixed_parameters={"gravity": gravity},
    experiment_settings=settings,
    refinements=(
        {"numeric_settings": {"dt": 1.0 / 960.0}},
        {"numeric_settings": {"iterations": 16}},
    ),
    # Map a 1.5 mm position change to features for two equally weighted traces.
    precision_tolerance=0.0015 / (0.03 * np.sqrt(2 * len(times))),
    qualification_mode="refinement",
)
tuning = Tuning(
    evidence=evidence,
    asset=asset,
    evaluator=Evaluator(loss="squared_error", max_normalized_error=2.0),
    parameters=parameters,
    rollout=Rollout(capacity=1, parameter_rtol=1e-6),
    optimizer=GridSearch(levels=41, batch_size=8),
    decision=Decision(
        max_trials=41,
        max_rollouts=200,
        max_generations=6,
        max_seconds=180.0,
        target=0.005,
    ),
    validator=Validator(max_objective=0.01, max_error=1.0 / 6.0),
    template=template,
    play=Play(),
    workspace="workspace/newton-sliding-guide",
)
result = tuning.run()
print(result.acceptance)
print(result.termination_reason)
print(result.parameters)
print(result.run_directory)
print(result.asset_path)
if result.acceptance != "accepted":
    raise SystemExit(1)
```

The reference assumes constant Coulomb deceleration until stopping. Newton computes the box motion through rigid-body contact and friction. The timestep is 1/480 s with eight XPBD iterations. Numerical checks independently halve the timestep and double the iteration count, before fitting and at the final candidate.

`precision_tolerance` is expressed in evaluator feature units. With two equally weighted training traces, 41 samples per trace, and a 0.03 m scale, the expression above maps to a maximum position change of 1.5 mm. The validator permits 3 mm RMSE and 5 mm maximum error for each checked experiment. The stopping target is a normalized mean squared error of `0.005`. These choices belong to this synthetic example.

`Rollout` uses one worker and a readback relative tolerance of `1e-6` for Newton’s float32 parameter arrays. `Template` fixes gravity, initial speeds, and solver settings during parameter search. All numerical refinements retain the candidate’s physical parameters.

Each `Tuning` instance runs once. Construct fresh members and a fresh `Tuning` when changing the evidence, model, numerical settings, or search definition. Its `run()` method performs preparation and the subsequent stages. [Lifecycle reference](api-reference.md#lifecycle-and-analysis-scope)

## Replace single env rollout with multi env rollout

The default example uses single-environment execution with `Rollout(capacity=1)`. `GridSearch(batch_size=8)` supplies candidate parameters, and one worker executes each candidate and experiment in a fresh `SlidingEnvironment`. This optional section shows how to replace that execution setup with multiple environments, using worker threads or a simulator's batch implementation.

### Run independent CPU environments

Create the rollout below and pass it as `rollout=rollout` when constructing a fresh `Tuning` from the full assembly above.

```python
rollout = Rollout(capacity=4, parameter_rtol=1e-6)
```

Run the bundled example with `python -m examples.box_friction_calibration.run --capacity 4` to use the same configuration. Each worker builds its own model, solver, collision pipeline, and state. The adapter serializes scene construction and collision detection because Warp 1.17.0 specializes contact kernels in a shared module. The XPBD solver steps run independently across workers.

### Implement a Newton batch environment

For GPU execution across several candidates, import `BatchEnvironment` from `newton_calibration`, implement a subclass, and connect it through `Asset.create_batch_environment()`. The bundled sliding adapter supplies `create_environment()` for individual episodes. The batch path below describes the additional adapter implementation needed for Newton multi-world execution.

A batch environment receives several complete parameter dictionaries for one experiment and repeat. In this example, the worlds share the experiment's initial speed and observation times, while each world receives its candidate's friction coefficient. Implement the following operations in a `SlidingBatchEnvironment` subclass.

1. Build one Newton world per candidate using `ModelBuilder.begin_world()` and `end_world()`. Add both the box and its ground plane inside that world, and apply the candidate's friction to both shapes. Retain the mapping from candidate order to the final model's body, shape, and world indices. The per-world ground supplies the matching friction coefficient for each candidate.
2. Finalize the combined model on the device selected by `Template.numeric_settings`, such as `cuda:0`. Create a solver, collision pipeline, and state pair for the combined model, and initialize every box with the experiment's initial speed.
3. Expose `applied_parameters` as one mapping per candidate, reading friction and gravity from the corresponding model arrays.
4. In `run(times, *, seed, deadline)`, advance the combined state with the same timestep and XPBD iteration settings as the single-environment case. At each step, clear forces, detect contacts, call `SolverXPBD.step()`, and swap the state pair. This call advances the candidate worlds together.
5. Read each box's position at the requested times and return one `Trace` per candidate in input order. For this example, each trace has `values` shaped `[T,1]` and `positions` shaped `[T,1,3]`. Copy retained samples and convert GPU observations to NumPy arrays for the existing evaluator.
6. Check the deadline during simulation. Return a `SimulationError` in a failed candidate's slot when the failure can be isolated. An exception from the whole call affects the chunk. Release adapter-owned resources in `close()` when needed.

Newton's world identifiers control collision isolation. The adapter must also preserve the correspondence between parameter readback, simulated bodies, and returned traces. [Pinned Newton world construction](https://github.com/newton-physics/newton/blob/c53345412d401232beff0bc71dcb39ce554fdf96/newton/_src/sim/builder.py#L5689) · [Batch interface](api-reference.md#batchenvironment-rollout-and-episode-execute-candidates)

Once `SlidingBatchEnvironment` implements these operations with the constructor arguments shown below, add this asset subclass alongside the adapter. It inherits the single-environment construction method and the asset export/reload behavior from `SlidingAsset`.

```python
class BatchedSlidingAsset(SlidingAsset):
    def create_batch_environment(self, template, parameter_values, experiment_settings):
        return SlidingBatchEnvironment(
            parameter_values,
            template.numeric_settings,
            experiment_settings["initial_speed"],
            self.config,
        )
```

### Select the batch path in the assembly

Use the implemented `BatchedSlidingAsset` when constructing the asset, and set `device="cuda:0"` in the new template's `numeric_settings`. Construct the rollout and optimizer as follows, then pass them as `rollout=rollout` and `optimizer=optimizer` to the new `Tuning`.

```python
asset = BatchedSlidingAsset(
    parameters.as_dict(parameters.baseline),
    config={"half_extents": [0.1, 0.1, 0.05], "density": 1000.0},
)
rollout = Rollout(worlds=4, parameter_rtol=1e-6)
optimizer = GridSearch(levels=41, batch_size=8)
```

Overriding `create_batch_environment()` selects the batch execution path automatically. `worlds=4` limits each batch environment to four candidates. Eight candidates therefore use two chunks per experiment. Across the two training experiments, that produces four batch-environment calls and 16 episodes. The current rollout executes experiments, repeats, and chunks in sequence, while the simulator advances the worlds within each chunk together.

| Setting | Effect |
|---|---|
| `Rollout.capacity` | Worker count for the ordinary independent-environment path |
| `Rollout.worlds` | Maximum candidates in one batch environment, after the asset supplies the batch method |
| `Optimizer.batch_size` | Requested candidates per search batch, subject to available candidates and budget |
| `Template.numeric_settings["device"]` | Simulation device consumed by the Newton adapter |

The batch implementation must also handle smaller chunks, including one candidate during validation. Episode budgets count every candidate, experiment, and repeat. In this example, a search batch of eight candidates fits within the existing search and rollout budgets.

Before a full calibration, compare the same candidate parameters under single-environment and batch execution. Check parameter readback, trace order, observation times, residuals, and state isolation. Compare different chunk sizes and candidate orders, and repeat the numerical refinements on the selected device. Then run a fresh `Tuning` through export/reload and final validation. The CPU sliding result above establishes the single-environment reference for these checks. [Execution contracts](api-reference.md#running-candidates-together)

## Replace synthetic observations with measurements

Begin with the same sliding experiment to isolate the data integration step. Record time in seconds and position in meters for each acquisition, along with initial speed, gravity convention, frame, release state, and processing. Use separate acquisitions for fitting and validation.

For a CSV with exactly two columns named `time_s,position_m`, the observation-loading part becomes

```python
measured = np.loadtxt("data/slow.csv", delimiter=",", skiprows=1, ndmin=2)
slow = Experiment(
    id="slow",
    group="acquisition-slow",
    reference=measured[:, 1:2],
    times=measured[:, 0],
    scale=0.03,
)
```

Create the other experiments from their own recordings and put the fitting and held-out experiments into `Evidence`. Choose the scale from the observation definition and error tolerance for your study. Set `Template.experiment_settings[experiment.id]` to the measured initial speed and other conditions for each acquisition. Review the physical bounds, numeric settings, and acceptance thresholds, then construct a new `Tuning` and run it.

The sample times belong to the observations. The simulator's integration timestep belongs to `Template.numeric_settings`. The environment advances with that timestep and samples its prediction at the observation times. Preserve acquisition groups when processing or splitting recordings. [Measurement design](calibration-guide.md#stage-4-measurement-and-data-design)

## Connect a different simulator or object

The simulator enters this example through the `asset=asset` argument of `Tuning`. `SlidingAsset.create_environment()` constructs `SlidingEnvironment`, which owns the Newton model, solver, and state. To run the same sliding calibration with another simulator, implement a replacement environment, connect it through an asset, and supply that asset and its matching template to a fresh `Tuning`.

### Replace the Newton calls in the environment

Create a class derived from `Environment` for the selected simulator. Its constructor must build a fresh sliding scene, apply the candidate parameters, configure the numerical settings, and initialize the experiment. The following table identifies the calls in the existing adapter that need corresponding operations in the new backend. [Current sliding adapter](../examples/box_friction_calibration/backend.py)

| Location in the existing adapter | Current Newton operation | Operation to implement with the selected simulator |
|---|---|---|
| `SlidingEnvironment.__init__()` | `ModelBuilder`, `add_body()`, `add_shape_box()`, `add_ground_plane()` | Build the box and plane with the specified dimensions, density, coordinate frame, and initial pose |
| Material and gravity setup | `ShapeConfig(mu=...)` and `ModelBuilder(gravity=...)` | Apply the pair friction coefficient and gravity using the new engine's material and force conventions |
| Solver and state setup | `builder.finalize()`, `SolverXPBD(...)`, `model.state()`, `body_qd.assign()` | Create the simulation state, select the integrator or solver, consume the numerical settings, and set the measured initial speed |
| `applied_parameters` | Read `model.shape_material_mu` and `model.gravity` | Read the effective values consumed by the new simulator and report them under the calibration keys `friction` and `gravity` |
| `run()` stepping loop | `clear_forces()`, `collision.collide()`, `solver.step()`, state swap | Advance the new simulator from its initial state, checking the deadline during execution |
| `run()` observations | Read `state_in.body_q` and copy positions | Sample or interpolate the box center at every requested observation time, convert to meters in the evidence frame, and retain independent samples |
| `close()` | The current adapter uses the base implementation | Release any native simulation handles, processes, or other resources owned by the replacement environment |

For the sliding experiment, `parameter_values` contains both the candidate `friction` and fixed `gravity`. The experiment settings supply `initial_speed` and `seed`. Store the dimensions and density in `Asset.config`, so export/reload reconstructs the same scene. Check how the new simulator combines surface friction values, then map the calibration's pair coefficient into those fields and read it back in the same convention.

At the end of the replacement environment's `run()`, convert the collected center positions into the observation format below. Here, `sampled_positions` contains one XYZ position per requested time, expressed in meters in the evidence frame.

```python
positions = np.asarray(sampled_positions, dtype=np.float64)[:, None, :]
values = positions[:, :, 0].copy()
return Trace(times.copy(), values, positions)
```

The evaluator receives the X-position channel as `[T,1]`, and playback receives XYZ positions as `[T,1,3]`. The simulator-specific state stays inside the environment. [Environment and trace contract](api-reference.md#asset-environment-and-trace-connect-the-simulator)

### Connect the replacement asset to Tuning

The following wiring uses `ReplacementSlidingEnvironment` as the name of the environment you implement with the selected simulator's API. Its constructor accepts the complete physical parameters, numerical settings, experiment settings, and scene configuration. Define this asset class and use its instance for the `asset` variable in the full assembly above.

```python
from newton_calibration import Asset


class ReplacementSlidingAsset(Asset):
    def create_environment(self, template, parameter_values, experiment_settings):
        return ReplacementSlidingEnvironment(
            parameter_values,
            template.numeric_settings,
            experiment_settings,
            self.config,
        )


asset = ReplacementSlidingAsset(
    parameters.as_dict(parameters.baseline),
    config={"half_extents": [0.1, 0.1, 0.05], "density": 1000.0},
)
```

In that assembly, `Tuning(asset=asset, template=template, ...)` now routes simulations through the replacement environment. Define a new `Template` whose `numeric_settings` keys are consumed by that environment. In particular, replace the XPBD solver selection, iteration settings, and float32 declaration with settings supported by the selected backend. Set meaningful refinements for its timestep, iteration controls, or discretization. `Template` records and passes these dictionaries to the adapter, and the adapter constructs the actual solver.

For the same sliding task, use the reference X-position trajectories and physical parameter definitions to compare the backends. Configure `Rollout` readback tolerances for the new simulator's parameter precision and select execution capacity after checking its concurrency support. Construct fresh members and run the new `Tuning` through the existing evaluation, search, export/reload, and validation lifecycle.

### Check the integration and adapt the physical task

Run the baseline experiment first. Check parameter readback, observation times, channel order, units, and coordinate frames. Perturb friction and confirm that the intended material fields and trajectory change. Repeat runs with fresh environments to check state isolation, then export and reload a candidate and compare its response under the same conditions. Complete numerical qualification and held-out validation with the new backend. [Adapter checks](api-reference.md#contracts-to-check-during-integration)

For a cable, robot, or another object, also replace the scene, experiment inputs, and observation channels. Supply the corresponding `Evidence` and `Experiment` data, define its `Parameters` and `Template`, and choose scales, scoring rules, and validation thresholds for that task. The cable adapter demonstrates the mapping from continuum bending rigidity to discrete joint coefficients, together with measured static and dynamic observations. [Cable parameter registration](calibration-guide.md#cable-example-registering-measured-quantities-and-initial-assumptions)

## Learn from the cable experiment

The completed no-plug study follows a 17 cm free cable. Static shapes constrain bending and the clamp boundary. Release motion exposes a period-and-decay mismatch, leading to a rate-dependent SLS model. The selected model's dynamic coordinate RMSE is about 0.844 mm under the stated mask and numerical configuration. These observations also informed model selection, so interpreting the result requires that evidence-use history. [Cable estimation example](calibration-guide.md#cable-example-revising-the-estimate-from-its-residuals)

To reproduce the scores from bundled observations and saved predictions, install the cable extra and run the archive check from a checkout.

```bash
python -m pip install -e ".[cable]"
python -m examples.cable_calibration.verify
```

The verifier checks all 14 archived traces. On the no-plug fitting recording, B* gives 1.598 mm coordinate RMSE and C gives 0.844 mm. [Reproduce the no-plug scores](../examples/cable_calibration/README.md#1-verify-the-reported-results-without-simulation) · [Original run records](../examples/cable_calibration/archive/)

## Work with the skills

You choose the physical task, provide evidence, and resolve experimental assumptions. The skills help organize those decisions, configure the members, and implement an adapter when needed. A configured `Tuning.run()` performs the supported execution stages. [Responsibilities by stage](calibration-guide.md#workflow-and-stage-deliverables) · [Skill installation and usage](codex-skills.md)

## When a run needs attention

| Result or symptom | Next check |
|---|---|
| Parameter readback error | The applied model fields, units, parameter keys, and numerical precision |
| Numerical qualification failure | Timestep, iterations, discretization mapping, initialization, and refinement scope |
| Inconclusive local identifiability | Whether experiments distinguish the selected parameter effects |
| Training errors exceed thresholds | Residuals, bounds, model assumptions, and search budget |
| Held-out prediction fails | Evidence coverage, boundary conditions, and model applicability |
| Final checks are incomplete | Budget use and the saved termination reason |

Use the method guide to decide whether the next step is a new measurement, a model change, a numerical change, or more search. Use the API reference for the corresponding member contract. [Residual diagnosis](calibration-guide.md#using-residuals-to-plan-the-next-iteration) · [Outcome reference](api-reference.md#stopping-reasons-acceptance-and-budgets)
