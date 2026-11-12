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
| Observations | [Evidence and Experiment](#evidence-and-experiment-store-observations), [FileSource](#filesource-fetches-pinned-source-files) |
| Search values and fixed conditions | [Parameters](#parameter-and-parameters-define-the-search-space), [Template](#template-fixes-a-runs-configuration) |
| Simulation | [Asset, Environment, Trace](#asset-environment-and-trace-connect-the-simulator), [Rollout](#batchenvironment-rollout-and-episode-execute-candidates) |
| Scoring, search, and stopping | [Evaluator](#evaluator-converts-trajectories-into-objectives-and-constraints), [Optimizer](#optimizer-gridsearch-and-cmaes-propose-candidates), [Decision](#decision-controls-budgets-and-stopping) |
| Candidate checks and presentation | [Validator](#validator-checks-the-exported-candidate), [Play](#play-displays-saved-trajectories) |
| Lifecycle and results | [Tuning](#tuning-assembles-members-and-returns-results), [analysis scope](#lifecycle-and-analysis-scope), [result files](#result-files) |

## Evidence and Experiment store observations

`Experiment` holds one experiment's reference outputs and scoring conditions. `Evidence` groups experiments into training and validation data and records their sources. Put control commands, loads, and initial states in `Template.experiment_settings` under the same experiment ID; put measured ones, such as recorded commands, in `Experiment.inputs` so they are part of the evidence. The environment uses both to run the forward problem.

```text
Experiment(id, group, reference, times=None, scale=1.0,
           mask=None, weight=1.0, repeats=1, inputs=None)
Evidence(train_data, validation_data=(), *, sources=None)
```

| Argument | Type or default | Usage |
|---|---|---|
| `id` | Required string | Unique across all training and validation experiments, and a key in `experiment_settings` |
| `group` | Required string | Real acquisition group. Training and validation group labels must be disjoint |
| `reference` | Nonempty `[T,D]` array | `T` times and `D` observation channels, converted to NumPy `float64` during preparation |
| `times` | `None` or `[T]` array | Finite, nonnegative, strictly increasing observation times. Record units, such as seconds. Omitted times become `0, 1, …, T-1` |
| `scale` | `1.0` | Positive finite scalar, `[D]`, or array broadcastable to `[T,D]`, used to normalize residuals |
| `mask` | `None` | Boolean `[T,D]`, with `True` selecting observations for scoring. Defaults to all valid, and must retain at least one element |
| `weight` | `1.0` | Positive finite relative experiment weight in the combined objective |
| `repeats` | `1` | Simulation repeats against the same reference, executed with experiment `seed + repeat` |
| `inputs` | `None` | Measured per-experiment inputs, such as recorded commands or an initial state, as a nonempty mapping of named finite arrays. They are converted to `float64`, included in the fingerprint and split with the reference. `Template` passes them to `create_environment()` and `create_batch_environment()` as an additional `inputs=` keyword |
| `train_data` | Experiment sequence | At least one experiment for analysis and search |
| `validation_data` | `()` | Held-out experiments for individual evaluation after freezing the candidate |
| `sources` | `None` | Provenance dictionary for data versions, processing protocols, and measurement sources, included in the fingerprint |

`Evidence.prepare()` normalizes inputs, checks shapes, finiteness, IDs, and groups, then creates `evidence_id`. `Tuning` calls it during a complete run. For a standalone data check, construct `Evidence`, call `prepare()`, and inspect `train_data`, `validation_data`, and `fingerprint()`.

Robot channels might be `[q_1, …, q_n, qd_1, …, qd_n]` with separate position and velocity scales. Cable centerlines can be flattened into channels at fixed arc-length samples, with masks for occlusion and initialization windows. Register independent real recordings as separate `Experiment`s and preserve acquisition relationships with `group`. Document channel order, frames, and units.

## FileSource fetches pinned source files

`newton_calibration.sources` locates evidence and asset files by relative path and checks each against a declared SHA-256 before use, so a run reads exactly the recorded bytes. Import it from that module; it is not part of the root namespace.

```text
FileSourceLocal(root)
FileSourceHuggingFace(repo_id, revision, *, repo_type="dataset")
FileSource.fetch(relative, sha256) -> Path
FileSource.fetch_all({relative: sha256, ...}) -> {relative: Path}
FileSource.provenance -> dict
file_sha256(path) -> str
```

| Member | Usage |
|---|---|
| `FileSourceLocal` | Files under `root`, for example an offline copy of a repository. Paths that leave `root` are rejected |
| `FileSourceHuggingFace` | Downloads into the Hugging Face cache through `huggingface_hub`, imported on the first fetch (`huggingface` extra). `revision` must be a full 40-digit commit hash; branch and tag names are rejected because they can move. `repo_type` is `"dataset"`, `"model"` or `"space"` |
| `fetch` | Returns the local path after checking the lowercase 64-digit digest; a missing or different digest raises `ValueError` |
| `provenance` | Provider, repository, type and revision, for `Evidence(sources=...)` |

Subclass `FileSource` and implement `locate()` and `provenance` for another provider. The [actuator example](../examples/actuator_calibration/data.py) reads its recordings and robot USD this way.

## Parameter and Parameters define the search space

`Parameter` defines one scalar to estimate. `Parameters` keeps the vector in the supplied order and converts between physical values, normalized values, and search coordinates for parameter combinations. `Asset` applies the values to the solver.

```text
Parameter(name, bounds, baseline, unit="1", transform="linear")
Parameters(definitions)
```

| Argument | Requirements and meaning |
|---|---|
| `name` | Nonempty and unique, matching the asset's parameter key |
| `bounds` | Finite `(lower, upper)` with `lower < upper` |
| `baseline` | Finite and within the closed bounds, representing the pre-calibration value |
| `unit` | Unit description, default `"1"`. Data processing and adapters perform the agreed conversions |
| `transform` | `"linear"` or `"log"`. Logarithmic normalization requires a positive lower bound |
| `definitions` | Nonempty `Parameter` sequence whose order defines physical-vector columns |

For example, `Parameter("friction", (0.05, 0.65), 0.15)` defines dimensionless friction. `Parameter("tau", (0.001, 0.1), 0.02, unit="s", transform="log")` defines a positive time constant spanning two orders of magnitude, provided the material model and adapter implement the mechanism represented by `tau`.

| Interface | Result or purpose |
|---|---|
| `names`, `bounds`, `baseline` | Names, `[P,2]` bounds, and `[P]` baseline |
| `as_dict(row)` | Map one physical row to names, commonly for constructing the asset baseline |
| `to_normalized(values)` / `from_normalized(values)` | Convert between physical values and per-parameter `[0,1]` coordinates |
| `encode(values)` / `decode(values)` | Convert between physical values and the current combination search space, whose size is `dimension` |
| `check_bounds(values)` | Boolean validity of each physical row against finiteness and bounds |
| `snapshot()`, `parameter_space_id` | Record definitions, combination mapping, and identity |

The initial mapping normalizes each parameter independently. After identifiability analysis, `Tuning` can call `set_mapping(origin, basis)` to establish combination coordinates, then `freeze()` to lock the space. Decoded candidates still undergo physical-bound checks. Construct new `Parameters` when changing bounds or baseline for another run. Keep physical definitions, evidence, and dependencies in the [parameter registry](calibration-guide.md#record-parameter-definitions-and-justified-ranges).

## Template fixes a run's configuration

`Template` stores the physical conditions, numerical settings, and experiment inputs shared by all candidates. When constructing an environment, it merges fixed and candidate parameters and passes the experiment's settings to the asset adapter. The adapter defines and interprets the dictionary keys.

```text
Template(*, numeric_settings, discretization, fixed_parameters,
         experiment_settings, refinements=(), precision_tolerance=0.05,
         qualification_mode="refinement")
```

| Argument | Usage |
|---|---|
| `numeric_settings` | Required nonempty dictionary for integrator, timestep, substeps, iterations, and related settings, using keys understood by the environment |
| `discretization` | Required dictionary for rod segments, mesh resolution, or degrees of freedom. May be empty when appropriate |
| `fixed_parameters` | Required dictionary of fixed physical scalars, with keys disjoint from search parameters |
| `experiment_settings` | Required nonempty dictionary keyed by every experiment `id`. Each entry explicitly includes `seed` and adapter-specific inputs |
| `refinements` | Default `()`. Each configuration override may contain only `numeric_settings` and `discretization` |
| `precision_tolerance` | Default `0.05`, positive and finite, limiting maximum feature changes in numerical checks |
| `qualification_mode` | Default `"refinement"`, requiring valid refinement settings during a run. `"repeatability"` with empty `refinements` checks a fixed configuration |

The quickstart's sliding example uses `SolverXPBD` on the CPU with a 1/480 s timestep, eight iterations, and float32 state. It fixes gravity with `fixed_parameters={"gravity": 9.81}` and independently checks a 1/960 s timestep and 16 iterations. The adapter consumes these settings when constructing and stepping the Newton solver. A robot environment must also consume the control period and command sequence explicitly. A cable SLS environment must preserve pre-release history and internal-state initialization conditions. [Sliding configuration](../examples/box_friction_calibration/run.py) · [Cable factory](../examples/cable_calibration/run.py)

`resolve()` checks configuration, `freeze()` locks it, `snapshot()` returns serializable configuration data, and `template_id` identifies it. `refined(change)` creates a frozen refined template, with each refinement applied independently to the original. `instantiate(asset, parameter_values, experiment)` constructs one environment, and `instantiate_batch(...)` constructs a batch. `Rollout` normally calls these methods. [Lifecycle and analysis scope](#lifecycle-and-analysis-scope) explains qualification scope and tolerance scaling.

## Asset, Environment, and Trace connect the simulator

`Asset` holds construction settings, baseline parameters, and parameter-binding logic. `Environment` owns the mutable simulation state for one candidate and experiment and returns a `Trace`. When adding a robot or material model, implement model construction, parameter application, initial-state reset, and observation readout in these classes.

```text
Asset(baseline_parameters, *, config=None)
Asset.create_environment(template, parameter_values, experiment_settings)
Environment.run(times, *, seed, deadline)
Trace(times, values, positions=None)
```

`Asset` and `Environment` are abstract base classes. Use existing subclasses or implement their abstract methods above. The asset's `baseline_parameters` must exactly match `parameters.as_dict(parameters.baseline)`. Fixed physical values belong to the template. `config` stores construction settings such as asset locations and model options, and is retained by default JSON export.

1. `create_environment()` receives merged fixed and candidate parameters, constructs an independent model, solver, and state, and applies experiment boundaries and initial conditions.
2. `applied_parameters` reports the effective values, read back from fields actually used in computation. Its key set must exactly match the combined fixed and search parameter keys.
3. `run()` samples at the requested `times`. `seed` applies to the current repeat. `deadline` is an absolute `time.monotonic()` deadline checked by the environment at interruptible points.
4. `close()` releases environment resources. `Rollout` calls it after execution.

The returned `Trace` uses the following layout.

| Field | Shape | Meaning |
|---|---|---|
| `times` | `[T]` | Matches reference times, checked by the built-in evaluator with absolute tolerance `1e-9` |
| `values` | `[T,D]` | Same channel order, units, and frame semantics as reference observations, finite wherever the mask selects data |
| `positions` | `None` or `[T,N,3]` | Optional display points. Default USDA output uses meters and Z-up |

Raise `SimulationError` for expected physical or numerical candidate failures and `TimeoutError` for deadlines. The lifecycle records other exceptions as execution errors and handles the failure. Include enough context to locate the model, input, or numerical issue.

Assets also provide `load_or_create()`, `snapshot()`, `asset_id`, `export(parameter_values, path)`, and `reload(path)`. Default export writes `newton-calibration-asset/1` JSON with the asset type, candidate parameters, and construction configuration. Calling `reload(path)` on the original asset returns an instance of the same type with the candidate baseline. `Tuning` compares the reloaded baseline with the selected candidate within the Rollout readback tolerances and records the comparison as `analysis_reports["export_roundtrip"]`. `release(candidate_path, directory)` publishes an accepted export; the default copies it to `asset<suffix>`, keeping the export's file type. Its `create_environment()` must still construct fresh mutable state. Subclasses that cache mutable models must handle rebuilding and isolation.

The quickstart's `SlidingAsset` constructs a Newton model and `SolverXPBD` through their public APIs. The package's `newton_calibration.examples` module also exports NumPy-based `SlidingAsset` and `OscillatorAsset` demo adapters. The cable factory connects the measured cable study to its pinned solver. [Newton sliding adapter](../examples/box_friction_calibration/backend.py) · [NumPy demo exports](../newton_calibration/examples.py) · [Cable factory](../examples/cable_calibration/run.py)

## BatchEnvironment, Rollout, and Episode execute candidates

`Rollout` runs the candidate/experiment combinations, checks parameter readback, and collects results. Each candidate, experiment, and repeat produces an `Episode`, identified by `(trial_id, experiment_id, repeat)`.

```text
Rollout(capacity=1, *, worlds=None,
        parameter_rtol=1e-10, parameter_atol=1e-12)
Rollout.execute_batch(template, asset, trials, experiments, *, deadline=None)
```

| Argument | Behavior |
|---|---|
| `capacity` | Positive capacity, limiting worker threads for ordinary independent environments |
| `worlds` | `None` or a positive integer limiting candidates per batch environment. `None` puts the current batch's valid candidates in one environment |
| `parameter_rtol` | Readback relative tolerance, default `1e-10` |
| `parameter_atol` | Readback absolute tolerance, default `1e-12` |

For direct `execute_batch()` calls, supply `trials` as `(trial_id, physical_parameter_dict)` pairs, prepared `experiments`, and a frozen `template`. In a complete run, `Tuning` decodes candidates and calls the method. It returns an `Episode` tuple in candidate, experiment, and repeat order.

| Episode field | Meaning |
|---|---|
| `trial_id`, `experiment_id`, `repeat` | Candidate, experiment, and repetition identity |
| `status` | `completed`, or a failure such as `invalid_parameters`, `numerical_failure`, `timeout`, or `execution_error` |
| `trace` | Successful simulation `Trace`, default `None` |
| `error` | Error description, default `None` |
| `applied_parameters` | Full readback mapping, default empty dictionary |

Overriding `Asset.create_batch_environment(template, parameter_values, experiment_settings)` makes `asset.supports_batch` true and selects the batch path. `parameter_values` is a sequence of complete parameter dictionaries. Both `BatchEnvironment.applied_parameters` and `run(times, seed=..., deadline=...)` results preserve candidate order.

Each successful batch member returns a `Trace`. For a physical failure, return a `SimulationError` instance in that member's slot. Raising an exception from the batch call affects the entire current chunk. Each experiment and repeat gets a fresh batch environment. `worlds` controls chunking. NumPy, Warp, or another backend implements parallelism within each chunk. The ordinary path uses `capacity` threads. See `BatchedOscillatorAsset` for array-based batch execution. [Synthetic example exports](../newton_calibration/examples.py)

For example, 8 candidates, 2 training experiments, and `repeats=3` produce 48 episodes. The rollout budget counts all 48 even when the backend simulates candidates together.

## Evaluator converts trajectories into objectives and constraints

`Evaluator` matches the complete set of episodes to reference data by experiment, candidate, and repeat. It returns an objective, metrics, constraints, and residual features for each candidate. The objective ranks candidates, constraints determine feasibility, and features support sensitivity and numerical qualification.

```text
Evaluator(*, loss="squared_error", max_normalized_error=None)
Evaluator.score_batch(episodes, trial_ids, experiments)
```

| Argument | Behavior |
|---|---|
| `loss="squared_error"` | Aggregate normalized squared residuals using experiment weights, repeats, and valid observation counts. See [loss construction](calibration-guide.md#construct-losses-from-evidence) |
| `loss="cross_entropy"` | Score probability rows. References and predictions must be nonnegative normalized rows, with whole rows selected or excluded by the mask |
| `max_normalized_error=None` | Default constraint matrix has zero columns. A positive finite threshold adds `max_error / threshold - 1 <= 0` |

For batch size `B`, constraint count `K`, and concatenated feature count `F`, `score_batch()` returns five arrays in this order.

| Return value | Shape | Meaning |
|---|---|---|
| `objectives` | `[B]` | Per-candidate objective, positive infinity for invalid candidates |
| `metrics` | `[B,2]` | Default columns `normalized_rmse` and `normalized_max_error`, with maximum error at column 1 |
| `constraints` | `[B,K]` | Each entry must be `<= 0` for satisfaction. Built-in `K` is 0 or 1 |
| `valid_mask` | `[B]` | True when traces, times, finite valid observations, and loss-specific inputs meet the contract |
| `features` | `[B,F]` | Normalized residuals concatenated in fixed experiment/repeat order, scaled by square-root factors for experiment weights, repeats, and valid sample counts |

With squared error, `objectives[i]` equals the sum of squares of `features[i]`, and `metrics[i,0]` is its square root. With cross-entropy, the objective uses probabilities while features and both default error metrics still use scaled prediction-minus-reference residuals. Interpret `scale` accordingly.

For positions in meters, `scale=0.03` and `max_normalized_error=2.0` impose an absolute-error limit of 0.06 m on valid position channels. With different channel scales, convert the threshold separately for each channel.

`metric_spec` records the scoring protocol and `evaluation_id` identifies it. Custom evaluators can override `score_batch()` and `metric_spec`, including objectives, units, constraints, and every adjustable scoring setting in the protocol. Preserve array shapes, order, and feature count. Current `Tuning` and `Validator` read maximum error from `metrics[:,1]`, so retain that column's contract or adapt the affected members together.

## Optimizer, GridSearch, and CMAES propose candidates

`Optimizer` uses an ask/tell protocol. After parameter analysis, `Tuning` calls `initialize(parameters)`. Once the budget allows a batch, it calls `ask(batch_size)` and returns scores through `tell(result)`. The optimizer must receive feedback for the pending batch before proposing another.

```text
Optimizer(batch_size=8, seed=42)
GridSearch(levels=11, batch_size=16)
CMAES(batch_size=8, sigma=0.2, seed=42)
```

| Member or argument | Meaning |
|---|---|
| `GridSearch.levels` | Equally spaced points per search dimension, at least 2, with `levels ** parameters.dimension` total points |
| `GridSearch.batch_size` | Requested candidates per batch. Remaining budget and grid points can reduce the last batch |
| `CMAES.batch_size` | Population per generation, at least 4. `full_batch=True` requires complete-generation evaluation |
| `CMAES.sigma` | Initial step size in the current normalized search coordinates, with `0 < sigma < 1` |
| `CMAES.seed` | Sampling seed, default 42, configured separately from experimental simulation seeds |

Use `GridSearch` to explore a few parameters and inspect responses across their ranges. It generates a grid in the current search coordinates and maps it to physical values with `Parameters.decode()`. `CMAES` requires the `cma` extra and at least two search dimensions. If rank reduction is enabled, choose the optimizer for the reduced `parameters.dimension`.

`ask()` returns `TrialBatch(trial_ids, search_values)` with `[B,Q]` values in `[0,1]`, where `Q` is search dimension. `tell()` receives the corresponding `BatchResult`, whose candidate IDs, search values, and `purpose="train"` must match.

`BatchResult` stores search and physical values, all five scoring outputs, `episode_records`, problem `identity`, and `purpose`. `feasible` combines validity and constraints. `feedback` maps feasible candidates below 1, constraint violations to `[1,2)`, and invalid candidates to 3. Built-in CMA-ES updates from `feedback`, while run records retain the raw objective.

To implement another algorithm, subclass `Optimizer` and implement `ask()`. Extend `initialize()`, `tell()`, `diagnostics()`, and `state_dict()` as needed, and optionally provide `checkpoint()` as CMA-ES does. `diagnostics()["exhausted"]` informs `Decision` about search-space exhaustion. Shared stopping and budget decisions remain in `Decision.next_action()`.

## Decision controls budgets and stopping

Before baseline, analysis, training, and validation work, `Decision` checks state and budgets and returns `DecisionResult(action, reason, batch_size=0)`. `Tuning` uses that decision to start the work or finish the run.

```text
Decision(max_trials=512, max_rollouts=10000, max_generations=100,
         max_seconds=300.0, target=1e-5, patience=None,
         improvement_tolerance=1e-10, relative_improvement_tolerance=None)
```

| Argument | Scope and meaning |
|---|---|
| `max_trials` | Training-candidate limit. Baseline and analysis probes are recorded in their own phases |
| `max_rollouts` | Episode limit across all phases, including scheduled failed or invalid candidates |
| `max_generations` | Training-batch limit, corresponding to generations for CMA-ES and chunks for GridSearch |
| `max_seconds` | Total wall-clock limit from preparation, handled through cooperative backend deadlines |
| `target` | Stop search when the best feasible training objective reaches this value |
| `patience` | Default `None`. When enabled, detect stagnation from improvement across the specified number of recent batches |
| `improvement_tolerance` | Threshold for old best objective minus new best objective when patience is enabled |
| `relative_improvement_tolerance` | Default `None`. When set, the stagnation threshold is the larger of `improvement_tolerance` and this fraction of the old best objective |

Budget limits are positive, `target` is finite and nonnegative, and enabled `patience` is a positive batch count. `next_action(state, request)` receives a read-only state snapshot and work request, then returns `action="continue"` or `"stop"`. Its `reason` enters decision history.

For example, `Decision(max_trials=41, max_generations=3, max_rollouts=200)` accommodates a one-dimensional 41-point grid in batches of 16, with episode capacity for analysis and validation. Estimate the full budget using experiments, repeats, parameter dimension, and refinements. `Validator` supplies final acceptance thresholds. See [stopping reasons and acceptance](#stopping-reasons-acceptance-and-budgets).

## Validator checks the exported candidate

`Validator` checks acceptance for each experiment. `Tuning` exports and reloads the selected asset, then retests it on training experiments and evaluates it on held-out validation experiments. Both checks call `validate()` and produce separate reports.

```text
Validator(max_objective=0.01, max_error=1.0)
Validator.accepts(result)
Validator.validate(candidate_asset, validation_data, evaluate)
```

| Argument or method | Usage |
|---|---|
| `max_objective` | Finite nonnegative objective limit, default `0.01` |
| `max_error` | Finite nonnegative limit on `metrics[:,1]`, default `1.0` |
| `accepts(result)` | True when every candidate is feasible and within objective and maximum-error limits |
| `validate(...)` | Call `evaluate(candidate_asset, (experiment,))` for each experiment and return a dictionary with `passed`, `status`, and `experiments` |

With default squared error, `max_objective=0.01` requires normalized RMSE of at most 0.1 for that evaluation. The maximum-error condition must also pass. Retesting each experiment separately checks even those given low weight during search.

`Tuning` supplies the `evaluate` callback, locks candidate identity, and performs budgeting, simulation, and scoring. Custom validators can add condition grouping or historical-evidence restrictions while preserving report fields and candidate identity. The default validator returns `status="unvalidated"` for empty validation data. `Tuning` organizes numerical rechecks separately and includes them in the final report.

## Play displays saved trajectories

`Play()` uses saved `Trace.positions` from episodes, matching before/after trajectories by `(experiment_id, repeat)`. The default comparison contains `baseline` and `calibrated` tracks and an acceptance flag.

| Method | Arguments and result |
|---|---|
| `compare(baseline, calibrated, references=None, *, accepted=False)` | Accept two episode sequences and return `accepted` and `tracks`. Matched traces require positions on both sides and exactly equal time axes |
| `export_usd(comparison, output_path)` | Write USDA point animation and return its path, with second-based time codes, meters, and Z-up |
| `show_gl(comparison, *, max_frames=None, screenshot=None, headless=False)` | Use Pyglet from the `gl` extra to project xyz onto xy, optionally limit frames and save a screenshot |

`references` is currently a reserved argument. Default tracks come from the two simulated position sequences. Extend comparison or presentation to overlay measurements or plot joints and torques.

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

Pass the ten members to `Tuning` to run preparation, analysis, search, export/reload, retesting, and validation. Compose members and implement case adapters for your application.

```text
Tuning(*, evidence, asset, evaluator, parameters, rollout, optimizer,
       decision, validator, template, play, workspace="workspace",
       analysis_step=0.03, rank_tolerance=1e-3,
       allow_rank_reduction=False, orthogonalize=True)
```

| Additional argument | Default and behavior |
|---|---|
| `workspace` | `"workspace"`, the parent directory for distinct run directories with timestamp and random suffix |
| `analysis_step` | `0.03`, finite-difference step in per-parameter normalized coordinates, requiring `0 < step <= 0.25` |
| `rank_tolerance` | `1e-3`, relative rank threshold with `0 < tolerance < 1`. The singular-value cutoff is `max(s_max * tolerance, 1e-10)` |
| `allow_rank_reduction` | `False`, allowing search in retained identifiable combinations of nonzero rank when enabled |
| `orthogonalize` | `True`, building search coordinates from local SVD combinations. Required for rank reduction |

Start with `result = tuning.run()`. Calling `prepare()`, `sensitivity()`, `identifiability(report)`, and `evaluate_batch(...)` manually requires following the lifecycle, freezing, and budget protocols. Let `run()` manage those steps, using a fresh instance for each run.

`tuning.state` is a `RunState` containing `phase`, `identity`, `budget_usage`, `batch_history`, `best_candidate`, `analysis_reports`, `optimizer_snapshot`, `decision_history`, `validation_report`, `artifact_refs`, and errors. `state.view()` supplies a read-only decision snapshot. Inspect `run_state.json` after execution to trace these records.

`run()` returns a `TuningResult` with five fields.

| Field | Meaning |
|---|---|
| `acceptance` | Acceptance status, such as `accepted`, `rejected_training`, or `rejected_validation` |
| `termination_reason` | Search or run stopping reason, such as `target_reached` or `budget_exhausted` |
| `parameters` | Best candidate's physical parameter dictionary, or `None` before a candidate exists. Interpret it within the acceptance scope |
| `asset_path` | Accepted `release/asset.json` path, otherwise `None` |
| `run_directory` | Directory containing this run's records and results |

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
