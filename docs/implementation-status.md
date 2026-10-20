# Generic implementation status

Version 0.2.0. Original implementation record: 2026-10-12. Repository documentation update: 2026-10-14.

The core lifecycle was first qualified using two independent synthetic systems. The later real-cable integration is documented separately in the [cable example](../examples/cable_calibration/README.md); core acceptance does not imply cable acceptance.

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

## Historical generic qualification

The original Python 3.11 environment used Torch 2.9.1+cu128, CUDA, pytest, pyglet, USD, and cma 4.5.0. Initial verification passed 26 tests, including CUDA scoring and USD reload; the subsequent workspace change passed all 30 tests. Editable installation, wheel creation, Ruff, formal CLI execution, headless GL, safe tensor loading, and USD time samples were checked.

These are historical results from that environment. The current checkout can be tested using the [user guide](user-guide.md); hardware-dependent tests may skip on a CPU-only installation.

| Synthetic case | Optimizer | True values | Historical fitted values | Validation objective | Acceptance |
|---|---|---|---|---:|---|
| Mass-spring-damper | CMAES | k=5.0, c=0.7 | k=5.0046336, c=0.6966863 | 0.0001897223 | accepted |
| Sliding friction | GridSearch | friction=0.35 | friction=0.35 | approximately 1.41e-30 | accepted |

Both stopped with target_reached. CMAES evaluated 136 training candidates and sliding evaluated 32. Reference observations came from analytical formulas; simulation used independent numerical integration.

| Purpose | Oscillator rollouts | Sliding rollouts |
|---|---:|---:|
| Baseline | 3 | 2 |
| Numerical qualification | 18 | 12 |
| Sensitivity | 36 | 12 |
| Search | 408 | 64 |
| Reloaded training retest | 3 | 2 |
| Validation and final precision | 4 | 3 |

Historical run directories are intentionally not repository dependencies. New runs allocate unique subdirectories under the selected workspace, containing batches, analysis, checkpoints, candidate configuration, results, and visualization.

Tests cover bounds/mappings, immutability, collinearity and explicit reduction, acquisition-group separation, execution capacity, nonfinite/missing/duplicate results, cross entropy, budgets, no validation leakage, qualification failure, binding failure, export tampering, changed reloaded dynamics, and workspace preservation.

## Limits

1. Generic Asset exports JSON parameters/construction information, not a universal native solver asset. Reconstruction requires the correct adapter and template.
2. Evidence accepts processed observations; it does not implement general video tracking, camera calibration, or teleoperation import.
3. Multi-anchor Jacobian/SVD diagnostics do not provide repeat-noise estimates, confidence intervals, or global identifiability guarantees.
4. Each real backend needs independent benchmarks, hidden-default checks, reset tests, performance checks, and meaningful observation correspondence under refinement.
5. Threads are not process isolation. Deadlines are cooperative. Torch CUDA scoring does not imply GPU-batched Newton environments.
6. Play supports XY GL traces and 3D point USD, without complete meshes, materials, or camera replay.
7. Bayesian optimization and PBT are not implemented.
8. Checkpoints are saved, but automatic resume and cross-machine replay are not implemented.
