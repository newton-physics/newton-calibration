# Capabilities and limits

The core provides two synthetic systems and the [real cable example](../examples/cable_calibration/README.md). A successful synthetic calibration does not imply cable acceptance.

## Implemented

| Requirement | Current implementation |
|---|---|
| Ten members and RunState | Both synthetic cases compose the same Tuning and run() |
| Sensitivity | Finite differences at three anchors using observation features |
| Identifiability | SVD, effective rank, mappings, anchor response correlation; rank deficiency defaults to inconclusive |
| Explicit reduction | Unidentified directions fixed at baseline only when allowed; representative solution |
| Frozen Template | Deep immutable configuration, no tuned/fixed overlap, separate refinement templates |
| Batch execution/scoring | Independent threaded environments, Torch CPU/CUDA scoring |
| Binding checks | Readback of tuned/fixed quantities; missing or inconsistent bindings fail |
| Optimizers | GridSearch and CMAES initialize/ask/tell/state/diagnostics; no stop authority |
| Decision | Work authorization and shared rollout/time budgets with purpose-specific accounting |
| Records | Tensor batches, configuration/code fingerprints, decision history, CMAES checkpoint |
| Validation | Export/reload, training retest, holdout evaluation, final precision checks |
| Play | Saved-trace GL and USDA point animation with identities and parameters |

## Verification scope

Run the checks in [CONTRIBUTING.md](../CONTRIBUTING.md). Tests cover mappings, immutability, identifiability, data splits, result validity, budgets, binding checks, reload behavior, qualification policies, and workspace preservation. Hardware-dependent tests may skip on a CPU-only installation.

## Limits

1. Generic Asset exports JSON parameters/construction information, not a universal native solver asset. Reconstruction requires the correct adapter and template.
2. Evidence accepts processed observations; it does not implement general video tracking, camera calibration, or teleoperation import.
3. Multi-anchor Jacobian/SVD diagnostics do not provide repeat-noise estimates, confidence intervals, or global identifiability guarantees.
4. Each real backend needs independent benchmarks, hidden-default checks, reset tests, performance checks, and meaningful observation correspondence under refinement.
5. Threads are not process isolation. Deadlines are cooperative. Torch CUDA scoring does not imply GPU-batched Newton environments.
6. Play supports XY GL traces and 3D point USD, without complete meshes, materials, or camera replay.
7. Bayesian optimization and PBT are not implemented.
8. Checkpoints are saved, but automatic resume and cross-machine replay are not implemented.
