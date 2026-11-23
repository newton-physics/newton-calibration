<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Robot actuator calibration contract

`TuningRobot` derives from `Tuning` and only assembles members. `run()`, numerical
qualification, sensitivity/identifiability, search and acceptance use the core
lifecycle. All robot-specific semantics remain in this example.

| Member | Implementation and contract |
| --- | --- |
| Asset | `AssetRobot`: freeze MJCF/URDF, hashes, source-name mappings, immutable construction config; candidate resources remain locatable when gains change |
| Environment | `EnvironmentRobot` (CPU) or experimental `BatchEnvironmentRobotGPU` (CUDA): fresh model/control/state per episode or batch, independent per-world gains and state, known inertias/passive damping, no contacts, simulated-state feedback, actual coefficient readback |
| Evidence | Standard `Evidence`/`Experiment`, named multi-rate signals, masks for absent observations, no derived velocity, whole acquisition groups split before fitting |
| Parameters | Selected scalar joint kp/kd with explicit nonnegative bounds; inertia, damping, gear, limits and controller period fixed |
| Template | Explicit fixed gains/limits/period, initial state and commands, selected device, physics step and refinement; control period preserved under refinement |
| Rollout | Standard `Rollout(capacity=1, worlds=...)`; CPU is serial, CUDA uses independent SolverMuJoCo worlds in chunks, with identity/order/readback/deadline checks |
| Evaluator | `EvaluatorRobot`: core normalized objective/features plus per-joint physical RMSE/max, worst per-record/repeat aggregation |
| Optimizer | CMAES for multiple gains with optional initial `optimizer.sigma` (default 0.2, normalized coordinates, seed 42); GridSearch for a single gain rejects sigma |
| Decision | Explicit all-phase budgets and search stopping rules, with core validation reserves |
| Validator | `ValidatorRobot`: normalized and per-channel physical gates, each independent acquisition checked separately |
| Play/report | Core saved trajectory playback plus per-joint HTML response/residual plots from persisted traces; reporting does not simulate |

Equivalent PD applies `clip(kp*(q_ref-q)-kd*qd)` at controller ticks and holds joint
torque between them. Native position mode holds a command instead, links gain to
position bias, and updates damping bias in MuJoCo. Its native actuator coordinates,
gear and input/force limits are explicit. These are different control contracts.

The independently generated two-link reference uses known fixture mass/inertia,
passive damping and coupled Lagrange dynamics. Recovery tests use three distinct
initial gain vectors and require all four gains within 2% and held-out acceptance.
Native-actuator bridge tests use direct MuJoCo execution with non-unit signed gear,
control saturation and force saturation. Both are synthetic integration checks.

Opt-in CUDA tests compare both controller modes against CPU, verify per-world
gain/bias readback, A/B/A isolation, chunking, initial states, off-tick commands,
saturation, fixed control timing under refinement, NaN failure isolation,
deadlines and fresh-process candidate reconstruction. GPU benchmarks use training
prefixes only, not validation observations, and do not run an optimizer. Batch
capacity is separate from optimizer population and requires workload-specific
memory/throughput measurements. No shared core lifecycle changes are required.

Real-robot selection, measured records, source controller semantics and engineering
thresholds are external inputs. Until supplied and independently validated, no
hardware acceptance is established. A historical summary or simulated robot model
does not substitute for measured acquisition records. Frozen JSON is runtime
configuration for this adapter and its recorded dependencies, not a new USD schema.
