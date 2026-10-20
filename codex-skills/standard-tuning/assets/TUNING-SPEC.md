# Tuning task specification

Record configuration rationale here. Tuning's RunState and workspace remain authoritative for execution state. Remove instructions and irrelevant fields when filling this template; retain unresolved issues rather than presenting prompts as conclusions.

## Goal and scope

- Asset and conditions to predict:
- Deliverable: design / construction / calibration / local change
- Transferable physics or deployment-specific equivalent parameters:
- Prediction accuracy, cost, deployment constraints:
- Current status and limits:

## Sources

| Content | Path/version/acquisition group | Known facts | Uncertainty |
|---|---|---|---|
| Reference data and preprocessing | | | |
| Asset and measurements | | | |
| task_tuning / solver / interpreter | | | |

## Shared decisions

| Issue | Conclusion/status | Source | Rationale/alternatives | Affected members | Verification/invalidation |
|---|---|---|---|---|---|

Separate resolved facts, pending checks, unresolved issues, and limited assumptions; distinguish user answers from technical experiments.

## Observations, scoring, acceptance

- Meaning, correspondence, coordinates/units, physical time, windows/exclusions:
- Training/validation experiments and independent groups; missing or development-used holdout:
- Noise, normalization scale, masks, weights, aggregation:
- Loss, analysis features, metric semantics, failed-batch handling:
- Engineering tolerance, training retest and holdout requirements, distinct stopping target:

## Physics, experiment, numerics

| Layer | Setting | Value/unit | Source/reason fixed | Backend binding | Verification |
|---|---|---|---|---|---|
| Physical model | | | | | |
| Experiment | | | | | |
| Numerics | | | | | |

- Unknown but fixed quantities and impact:
- Equilibrium, dynamic initial state, driving/release:
- Independent benchmarks and representative time/space/iteration checks:
- Qualification policy (`refinement` or explicitly selected fixed-configuration `repeatability`) and claim limits:
- Physical error budget to feature-space tolerance:
- Rollout capacity/isolation/readback/device/failure/deadline:
- Play data, physical time/coordinates, GL/USD limits:

## Parameters, analysis, search

| Parameter/combination | Unit | Bounds/baseline | Source | Transform/binding | Confounding |
|---|---|---|---|---|---|

- Joint/staged strategy and upstream uncertainty:
- Tuning sensitivity/identifiability evidence or plan; unidentified directions:
- Actual optimizer/dependencies, rationale, dimension, full-batch requirements:
- Decision stopping and total budget:
- Qualification/baseline/analysis/search/retest/validation allocation:
- Cost of checks outside run():

## Members and construction

| Member | Actual class/module | Settings/rationale | Existing/adapted | Check |
|---|---|---|---|---|
| evidence | | | | |
| asset | | | | |
| evaluator | | | | |
| parameters | | | | |
| rollout | | | | |
| optimizer | | | | |
| decision | | | | |
| validator | | | | |
| template | | | | |
| play | | | | |

- Factory, import/construction checks, execution scope:
- Workspace, specification hash, source/dependency identities:
- State: Tuning's own RunState; no parallel state manager.
- Dependencies and next step:

## Results (only after execution)

- run_directory, termination_reason, acceptance:
- Analysis, retest, validation, qualification evidence:
- Export/reload and presentation artifacts:
- Supported scope, unverified conditions, further evidence needed:
