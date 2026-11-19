<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Newton Calibration API reference

[Quick start](quickstart.md) · [Calibration guide](calibration-guide.md) · [Public exports](../newton_calibration/__init__.py)

These interfaces apply to Newton Calibration 0.2.0. Import core classes from `newton_calibration`. The quick start shows a complete assembly using the public classes and an explicit simulator adapter.

`Tuning` combines ten members named `evidence`, `asset`, `evaluator`, `parameters`, `rollout`, `optimizer`, `decision`, `validator`, `template`, and `play`. It owns `RunState` and returns `TuningResult`. An adapter implements the model construction, parameter binding, and observation readout through `Asset` and `Environment`.

| Responsibility | Interfaces |
|---|---|
| Observations | [Evidence and Experiment](#evidence-and-experiment-store-observations) |
| Search values and fixed conditions | [Parameters](#parameter-and-parameters-define-the-search-space), [Template](#template-fixes-a-runs-configuration) |
| Simulation | [Asset, Environment, Trace](#asset-environment-and-trace-connect-the-simulator), [Rollout](#batchenvironment-rollout-and-episode-execute-candidates) |
| Scoring, search, and stopping | [Evaluator](#evaluator-converts-trajectories-into-objectives-and-constraints), [Optimizer](#optimizer-gridsearch-and-cmaes-propose-candidates), [Decision](#decision-controls-budgets-and-stopping) |
| Candidate checks and presentation | [Validator](#validator-checks-the-exported-candidate), [Play](#play-displays-saved-trajectories) |
| Lifecycle and results | [Tuning](#tuning-assembles-members-and-returns-results), [analysis scope](#lifecycle-and-analysis-scope), [result files](#result-files) |

## Evidence and Experiment store observations

See {py:class}`~newton_calibration.Experiment` and {py:class}`~newton_calibration.Evidence`.

Robot channels might be `[q_1, …, q_n, qd_1, …, qd_n]` with separate position and velocity scales. Cable centerlines can be flattened into channels at fixed arc-length samples, with masks for occlusion and initialization windows. Register independent real recordings as separate `Experiment`s and preserve acquisition relationships with `group`. Document channel order, frames, and units.

## Parameter and Parameters define the search space

See {py:class}`~newton_calibration.Parameter` and {py:class}`~newton_calibration.Parameters`.

For example, `Parameter("friction", (0.05, 0.65), 0.15)` defines dimensionless friction. `Parameter("tau", (0.001, 0.1), 0.02, unit="s", transform="log")` defines a positive time constant spanning two orders of magnitude, provided the material model and adapter implement the mechanism represented by `tau`.

Keep physical definitions, evidence, and dependencies in the [parameter registry](calibration-guide.md#record-parameter-definitions-and-justified-ranges).

## Template fixes a run's configuration

See {py:class}`~newton_calibration.Template`.

The quickstart's sliding example uses `SolverXPBD` on the CPU with a 1/480 s timestep, eight iterations, and float32 state. It fixes gravity with `fixed_parameters={"gravity": 9.81}` and independently checks a 1/960 s timestep and 16 iterations. The adapter consumes these settings when constructing and stepping the Newton solver. A robot environment must also consume the control period and command sequence explicitly. A cable SLS environment must preserve pre-release history and internal-state initialization conditions. [Sliding configuration](../examples/box_friction_calibration/run.py) · [Cable factory](../examples/cable_calibration/run.py)

[Lifecycle and analysis scope](#lifecycle-and-analysis-scope) explains qualification scope and tolerance scaling.

## Asset, Environment, and Trace connect the simulator

See {py:class}`~newton_calibration.Asset`, {py:class}`~newton_calibration.Environment`, and {py:class}`~newton_calibration.Trace`.

The quickstart's `SlidingAsset` constructs a Newton model and `SolverXPBD` through their public APIs. The package's `newton_calibration.examples` module also exports NumPy-based `SlidingAsset` and `OscillatorAsset` demo adapters. The cable factory connects the measured cable study to its pinned solver. [Newton sliding adapter](../examples/box_friction_calibration/backend.py) · [NumPy demo exports](../newton_calibration/examples.py) · [Cable factory](../examples/cable_calibration/run.py)

## BatchEnvironment, Rollout, and Episode execute candidates

See {py:class}`~newton_calibration.Rollout`, {py:class}`~newton_calibration.Episode`, and {py:class}`~newton_calibration.BatchEnvironment`.

See `BatchedOscillatorAsset` for array-based batch execution. [Synthetic example exports](../newton_calibration/examples.py)

## Evaluator converts trajectories into objectives and constraints

See {py:class}`~newton_calibration.Evaluator`.

## Optimizer, GridSearch, and CMAES propose candidates

See {py:class}`~newton_calibration.Optimizer`, {py:class}`~newton_calibration.GridSearch`, {py:class}`~newton_calibration.CMAES`, and {py:class}`~newton_calibration.BatchResult`.

## Decision controls budgets and stopping

See {py:class}`~newton_calibration.Decision`.

For example, `Decision(max_trials=41, max_generations=3, max_rollouts=200)` accommodates a one-dimensional 41-point grid in batches of 16, with episode capacity for analysis and validation. Estimate the full budget using experiments, repeats, parameter dimension, and refinements. `Validator` supplies final acceptance thresholds. See [stopping reasons and acceptance](#stopping-reasons-acceptance-and-budgets).

## Validator checks the exported candidate

See {py:class}`~newton_calibration.Validator`.

## Play displays saved trajectories

See {py:class}`~newton_calibration.Play`.

After a run, inspect `tuning.comparison`. USDA and GL display require position tracks. Run the following after the [complete example](quickstart.md#assemble-the-calibration-yourself) in the quick start.

```python
if tuning.comparison and tuning.comparison["tracks"]:
    path = tuning.play.export_usd(
        tuning.comparison,
        result.run_directory / "comparison-copy.usda",
    )
    print(path)
```

`Tuning.run()` attempts to generate `comparison.usda` automatically. The example above exports another copy. Presentation errors appear in `RunState.errors`. Scoring, validation, and numerical checks determine acceptance.

## Tuning assembles members and returns results

See {py:class}`~newton_calibration.Tuning`, {py:class}`~newton_calibration.RunState`, and {py:class}`~newton_calibration.TuningResult`.

## Contracts to check during integration

Check these contracts before the first complete run. The linked member reference gives the details for each check.

- Evidence ([Evidence and Experiment store observations](#evidence-and-experiment-store-observations)). Supply `[T,D]` reference values, `[T]` times, Boolean `[T,D]` masks, and positive broadcastable scales. `Evidence` rejects a group label shared by training and validation data, but the label itself does not prove independence, so keep the acquisition records. `Evidence` holds only training and validation data. Track development and sealed-data use across runs outside the module, along with image processing, camera calibration, and data collection.
- Parameters and template ([Parameters](#parameter-and-parameters-define-the-search-space) and [Template](#template-fixes-a-runs-configuration)). The asset baseline must equal the `Parameters` baseline, search keys must be disjoint from `Template.fixed_parameters`, and log-transformed parameters need a positive lower bound. Keep provenance, dependencies, and applicability in the Registry or experimental specification.
- Adapter ([Asset, Environment, and Trace connect the simulator](#asset-environment-and-trace-connect-the-simulator)). `create_environment()` reads `Template.experiment_settings[experiment.id]` as inputs, initial states, boundaries, and seeds, builds independent mutable state, and reports values read back from the model through `applied_parameters`. `run()` uses the template's integration settings and returns a `Trace` aligned with the reference times. Raise `SimulationError` only for expected candidate failures, not for implementation errors. The backend must check the deadline itself, because a blocked simulation can run past the thread-based execution deadline.
- Scoring ([Evaluator converts trajectories into objectives and constraints](#evaluator-converts-trajectories-into-objectives-and-constraints)). A custom evaluator returns all five arrays and keeps maximum error in `metrics[:,1]`, which `Tuning` and `Validator` read. Change the affected members together if that column moves.
- Execution ([BatchEnvironment, Rollout, and Episode execute candidates](#batchenvironment-rollout-and-episode-execute-candidates)). Ordinary assets run in independent environments on `Rollout.capacity` threads. A batch environment must preserve candidate order and per-candidate readback and results. GPU batching and distributed execution need backend support; see [running candidates together](#running-candidates-together).

## Lifecycle and analysis scope

`run()` prepares and freezes inputs, evaluates the baseline and numerical qualification, analyzes sensitivity and identifiability, searches, exports/reloads the candidate, retests training experiments, validates, and performs final numerical checks. It then records results and creates a release package when acceptance conditions pass.

Choose the numerical qualification mode explicitly. The default, `qualification_mode="refinement"`, requires at least one refinement that changes numerical or discretization settings. To check repeatability at fixed settings, select `qualification_mode="repeatability"` and `refinements=()`. Independent environments use the same settings and seeds, with scope reported as `fixed_configuration_only`. Check numerical convergence and cross-seed variation separately.

Initial qualification probes the baseline and normalized vectors filled with 0.25 and 0.75. The lifecycle repeats numerical qualification for the final candidate. Both initial and final core numerical checks use training experiments. Check other parameter regions and new validation loads or contact modes separately when they fall within the claimed domain.

`precision_tolerance` limits the maximum absolute change in `Evaluator.features`. Features incorporate residual scales, experiment weights, and valid sample counts. A physical tolerance such as 0.5 mm therefore requires an explicit mapping or an additional physical-unit metric.

Identifiability analysis uses finite differences at the same three anchors in normalized parameter coordinates. It applies SVD to the combined training-feature sensitivity matrix. By default, insufficient rank yields an inconclusive result. With positive retained rank, `allow_rank_reduction=True` and `orthogonalize=True` permit search in the parameter combinations supported by the local sensitivity analysis, with omitted directions fixed at the normalized baseline. Zero rank stops the search. These orthogonalized coordinates describe local parameter combinations. Global physical separation and statistical intervals require further evidence.

## Stopping reasons, acceptance, and budgets

`termination_reason` explains why search stopped. `acceptance` reports completion and outcome of the candidate's required checks. Target, generation, or time stopping conditions are evaluated separately from release eligibility.

| Status | Interpretation and next steps |
|---|---|
| `accepted` | Passed training scoring, exported/reloaded per-experiment training retest and validation, and numerical checks, within the stated conditions, thresholds, and qualification mode |
| `rejected_training` | Training candidate misses acceptance thresholds. Inspect residuals, bounds, model, and search adequacy |
| `rejected_validation` | Frozen-candidate validation fails. Inspect generalization, boundaries, model, and validation report |
| `rejected_numerics` | Qualification or final numerical recheck fails. Inspect solver settings and error budget first |
| `inconclusive` | Analysis cannot support the requested parameter conclusion. Improve identifiability or narrow the claimed scope |
| `unvalidated` | Validation data are absent or finalization was incomplete because of budget limits. Further validation is required for release |
| `no_feasible_candidate`, `run_failed` | No feasible candidate or fatal execution error. Use detailed reasons to investigate implementation, physics, or numerics |

`Validator` and related members define acceptance thresholds. `Decision.target` governs search stopping. The default lifecycle requires nonempty validation data for an accepted release, even with low training error.

`max_trials` limits training candidates. `max_rollouts` includes baseline, analysis, qualification, training, retesting, and validation. `max_seconds` covers the complete run. Leave budget for export/reload, validation, and final numerical checks. Checkpoints retain state for inspection, but a run cannot yet resume from one automatically. The [quick start](quickstart.md#understand-the-result) explains how to read the two result fields together.

## Result files

| File | Purpose |
|---|---|
| `resolved.json` | Observations, frozen configuration, identities, and versions needed to reconstruct the problem |
| `run_state.json` | Budget use, decisions, and phase state |
| `batches/*.json`, `checkpoints/*.pkl` | Candidate evaluation records and optimizer state |
| `candidate_asset.json` | Candidate saved on reaching export, with acceptance recorded separately |
| `result.json` | Stopping reason, acceptance, parameters, and result paths |
| `comparison.usda` | Visualization exported from saved trajectories |
| `release/asset.json`, `release/template.json`, `release/manifest.json` | Accepted asset, reconstruction conditions, and release manifest |

Default asset JSON stores parameters, adapter type, and construction configuration. Rebuilding simulation requires that adapter and the template. `Play` exports animated points/trajectories to USDA and displays saved states. The [quick start](quickstart.md#inspect-the-saved-run) shows how to inspect a completed run.

## Running candidates together

For independent environments, `Rollout(capacity=4)` runs up to four episodes on threads. An episode represents one candidate, experiment, and repeat.

For a simulator that handles multiple candidates in one environment, implement `Asset.create_batch_environment()` and return a `BatchEnvironment`. It reports one applied-parameter mapping per candidate and returns one `Trace` or `SimulationError` per candidate in input order. Each chunk uses one experiment and repeat. `Rollout(worlds=4)` limits a chunk to four candidates. The adapter owns the simulator-specific batching implementation.

The current batch path processes experiments, repeats, and chunks sequentially. Parallel execution occurs inside the batch environment. `capacity` controls the ordinary threaded path, and `worlds` controls the batch path's chunk size. The sliding quickstart shows the worker configuration and the adapter extension required for Newton multi-world execution. [Replace single env rollout with multi env rollout](quickstart.md#replace-single-env-rollout-with-multi-env-rollout)

CPU threads and simulator batching have different memory and scheduling costs. Check candidate isolation, output order, parameter readback, and the backend's concurrency support before increasing capacity. [Bundled batching interfaces](../newton_calibration/examples.py)

## Compatibility and saved assets

Core imports use the `newton_calibration` root. The former core submodule paths remain deprecated compatibility aliases under the project's deprecation policy. The public demo helpers use `newton_calibration.examples`, and the migration utility uses `newton_calibration.migrate_asset`.

Convert an asset exported before the package rename into a new file with

```bash
python -m newton_calibration.migrate_asset old-asset.json new-asset.json
```

The converter updates the asset schema and built-in example class names, preserving parameter values, configuration, and external class names. Load the converted file through the matching asset's `reload()` method and retain the original file with its run records. [Migration and import compatibility](user-guide.md#migrate-a-saved-asset-from-task_tuning)
