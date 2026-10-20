---
name: tuning-environment
description: Design or revise Asset, Template, Rollout, and Play for task_tuning calibration, including physical and experimental conditions, numerical accuracy, parameter binding, parallel simulation, and comparisons. Use for environment and execution decisions; never promote an old case's numerics to universal defaults.
---

# Tuning Environment

Read the shared specification and actual interfaces. Separate three layers:

| Layer | Contents |
|---|---|
| Physical model | Geometry, mass, inertia, rest state, joints, contact, boundary conditions |
| Experiment | Gravity, clamp, initial state, controls, releases, sampling, observation mapping |
| Numerics | Time step, integrator, discretization, iterations, tolerances, precision, stabilization, device, seed |

For every setting record value, units, source, backend field, reason for fixing it, and qualification status. Unknown physics must not silently become a numerical setting. Establish whether the target is transferable material properties or equivalent parameters for a specified deployment model; retain an existing answer.

## Qualification

Check actual solver binding and readback, derived parameter mappings, and independent physical benchmarks. Qualify time step, space, and iterations across representative loads, attachments, contacts, and parameter ranges. A smaller time step is not automatically more accurate. Define static equilibrium, dynamic initialization, release, and physical versus numerical damping separately.

Translate physical error budgets to Evaluator features with the actual scale, weights, mask, and sample count. A frozen dictionary does not establish backend correctness. If a required locked setting cannot pass qualification, report the gap; do not manufacture a pass.

Template defaults to `qualification_mode="refinement"` and requires actual refinements. Only an explicit fixed-configuration decision permits `qualification_mode="repeatability"` with no refinements. That mode repeats the same settings and seeds; it does not prove convergence or cross-seed robustness. Record its `fixed_configuration_only` scope and retain independent accuracy checks as separate requirements. Never choose repeatability merely to bypass failed refinement.

Run qualification only within the authorized scope and budget. Record standalone setup experiments separately: they are not automatically counted by RunState.

## Execution and presentation

Asset must create fresh state and support export/reload. Read applied parameters from the bound model, not an echo of input values. Test A/B/A isolation, single versus batch consistency, stable trial/experiment/repeat identity, actual capacity and device behavior, failures, and cooperative deadlines.

Play should consume saved trajectories with consistent coordinates and physical times. Current generic GL is an XY point/trajectory view; USD is point animation, not a complete asset renderer. Presentation does not affect optimizer feedback or acceptance.

Deliver Asset/Template/Rollout/Play configurations, qualification evidence or pending checks, and necessary adapters. Use tuning-adapter only when existing constructors cannot express the required behavior.
