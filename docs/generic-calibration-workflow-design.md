# Object-oriented calibration workflow

The generic implementation includes two synthetic examples; see [capabilities and limits](implementation-status.md) for verified scope and limitations. The [cable example](../examples/cable_calibration/README.md) extends members without replacing the lifecycle.

## One Tuning owns one calibration

Tuning composes ten members and a RunState data object. The parent owns run(), sensitivity(), identifiability(), and evaluate_batch(). Case code selects implementations and configuration. It must not introduce a second search, stopping, validation, or recording loop.

| Member | Responsibility | Boundary |
|---|---|---|
| Evidence | Prepare references, alignment, provenance, train/validation groups | Simulation inputs are separate from expected observations; no validation target reaches search |
| Asset | Load/create the baseline, bind physical parameters, export a candidate | Preserve baseline; exported configuration must recreate a simulatable asset |
| Evaluator | Batch loss, metrics, constraints, analysis features | No simulation, parameter changes, or stopping |
| Parameters | Definitions, units, bounds, selection, combinations, encode/decode | No solver state or analysis orchestration |
| Rollout | Instantiate, reset, execute, collect independent environments | No search policy; stable identities across batches |
| Optimizer | Propose through ask(), update through tell(), report facts/state | No stopping or budget authority |
| Decision | Authorize work, continue/stop, record reasons from current and historical state | Sole stopping authority |
| Validator | Assess frozen, exported/reloaded candidates on declared evidence | No training feedback or implicit retraining |
| Template | Resolve/freeze all conditions of one environment; instantiate | No trial history or mutable optimizer state |
| Play | Compare before/after trajectories; GL or USD presentation | No score or acceptance changes |
| RunState | Record identities, budgets, history, reports, artifacts | Data owned and updated by Tuning, not a separate service |

Solver binding, camera projection, and conversions remain internal member implementation details. See the [class diagram](tuning-class-diagram.md).

## Template: a complete single-environment definition

Template describes how a trial runs, not an already-running shared environment. Resolve and record:

- Numerical settings: steps, substeps, iterations, tolerances, integration/solver modes, precision, collision and constraint settings.
- Discretization: segments, mesh, topology, order, contact approximation.
- Known and non-tuned quantities: geometry, mass, gravity, fixed material values.
- Asset, backend versions, device, coordinates/units, construction and reset rules.
- Per-experiment initial state, boundaries, controls, events, duration, seeds, and randomization.

Resolve hidden external defaults explicitly. Each quantity has one owner: Parameters for tuned variables, Template for fixed conditions. Record derived solver mappings instead of independently assigning both a source parameter and its derived field.

Preparation checks completeness, binding, reset isolation, numerical accuracy, and performance before search. Frozen definitions are shared; mutable environments are independent. Recreate environments when safe in-place updates are unsupported. Numerical refinements use separate template copies and IDs; their scores do not compete in the same search ranking. Changing the formal template requires a new run or explicit stage.

Asset defines what the asset is; Template defines construction and conditions; Rollout controls execution and scheduling. All use one consistent binding path. Values frozen after an earlier stage retain provenance and uncertainty.

## Analysis and parameter combinations

Tuning.sensitivity() perturbs normalized parameters using the same rollout and evaluator as training. It analyzes observation residuals/features rather than differentiating only one aggregate score. Analysis consumes authorized budget and is recorded in RunState.

Normalize parameters by declared ranges or log scales and observations by declared noise/tolerances. Check perturbation sizes and repeat noise; use feasible one-sided differences near boundaries. Failed perturbations mean uncertainty, not zero sensitivity.

Tuning.identifiability() distinguishes sensitivity from separability. Several parameters can affect observations while only a sum, ratio, or combination is observable. A local construction is:

1. Form the weighted observation Jacobian J with respect to normalized physical parameters.
2. Stack representative experiments/anchors using predetermined weights; inspect correlations, singular values, rank, and noise.
3. Compute J = U S Vᵀ. Keep supported directions V_r and map combination coordinates by x = x0 + V_r z.
4. Prefer an interpretable physical combination when supported by the same checks.
5. Recheck response correlation over the declared region and freeze the mapping, dimension, fixed directions, and parameter-space ID before search.

This is local linear analysis, not global identifiability or a statistical confidence interval. Coordinate rotation cannot create missing information. Distribution-loss features do not automatically justify interpreting JᵀJ as statistical information.

For rank deficiency, acquire informative evidence, use justified fixed values/priors, or report inconclusive. Directions fixed at baseline are explicitly unidentified; a reduced representative solution does not prove unique physical parameters. Combination-space feasibility is generally nonrectangular. Parameters enforces original bounds; do not silently clip coordinates. Changing mappings requires a fresh optimizer stage.

## Evaluation, stopping, and records

The sole evaluation path is:

```text
search-space trial
  -> Parameters decode and check physical bounds
  -> Rollout instantiate and execute under Template
  -> Evaluator batch score
  -> Tuning assemble BatchResult and update RunState
```

Optimizer diagnostics such as convergence, exhaustion, inability to propose, and numerical errors are facts for Decision. Do not invoke a library's automatic stopping loop. Decision checks budgets before work and determines allowed batch sizes afterward. Complete-generation algorithms must not receive unauthorized partial generations.

Exceptions, cancellation, and cooperative deadlines are recorded. Rollout may interrupt work at an authorized deadline but cannot declare calibration success. External process termination must not be represented as normal completion.

| BatchResult field | Meaning |
|---|---|
| trial_ids | Stable identity independent of temporary environment slots |
| search_values [B,K] | Proposed combinations |
| parameter_values [B,P] | Decoded physical parameters |
| objectives [B] | Consistently directed aggregate objectives |
| metrics [B,M], constraints [B,C] | Separate diagnostics and constraints |
| valid_mask [B], status | Success, failure, missing results, timeout |
| episode records | Trial/experiment/repeat identity and traces |
| identities | Template, parameter mapping, evidence, evaluator, purpose |

Use Torch tensors for numeric batch results with explicit dtype/device/layout/masks. Solvers need not use Torch internally. Convert at optimizer boundaries as necessary. One trial may span multiple experiments/repeats and several execution batches. Missing, duplicate, extra, or failed episodes cannot shrink denominators or become zero error. Protocol-defined physical failure penalties differ from infrastructure failure.

RunState records phases, run ID, versions, frozen configuration, mappings, seeds, budgets by purpose, batches, historical best, analysis, optimizer checkpoints, decisions, failures, validation, and artifact references. Large arrays may live separately with hashes. Checkpoint persistence alone does not establish resume support or cross-machine replay.

## Finalization and presentation

The lifecycle prepares evidence/assets/templates, records baseline and qualification, performs analysis, freezes parameter combinations, searches under Decision, exports the best feasible candidate, reloads it, retests training, validates, and generates comparisons. Early failure without an evaluable candidate skips export and validation.

Stopping is separate from acceptance. Validator failure does not silently restart training. Holdout evidence must not influence sensitivity, parameter selection, search, or training stopping. Once its outcomes guide model changes, report it as development evidence.

Play normally uses saved baseline and candidate traces under the same experiment, coordinates, and physical times. A display rerun uses the frozen template/seed and does not enter training history. Display resampling does not change simulation time steps. Rendering failure does not alter physical acceptance. Asset.export() produces simulatable configuration; Play.export_usd() produces time-sampled visualization, a separate artifact.

## Acceptance and migration boundaries

The generic implementation has no dependency on the former development workflow or its file formats. Core checks include independent synthetic parameter recovery, rank-deficiency behavior, bound-preserving combinations, Decision ownership, immutable templates, isolation, stable identities, nonfinite/missing failures, validation separation, export/reload, traceability, and correctly timed presentation.

Synthetic success does not establish a real backend's physics, hidden defaults, numerical convergence, or performance. Those remain case-specific qualification obligations. See [implementation status](implementation-status.md) and the [user guide](user-guide.md).
