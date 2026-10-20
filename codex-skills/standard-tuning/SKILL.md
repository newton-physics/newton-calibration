---
name: standard-tuning
description: Create, configure, or use task_tuning for simulation parameter calibration. Resolve consequential unknowns from existing evidence and targeted questions, select member objects, and construct a real Tuning instance. Use for new calibration tasks, measured or teleoperation data, and configuration changes across modules. Do not start this workflow for score explanations, theory questions, or ordinary debugging.
---

# Standard Tuning

Compose one traceable `task_tuning.Tuning` using its existing lifecycle. This skill and its four sibling skills form a module; they require neither another scheduler nor multiple agents.

## Establish the requested scope

Preserve the user's scope: design, construction, calibration execution, or a local change. Design requests do not require implementation; construction requests do not authorize simulation. When calibration is already authorized and the required inputs and checks are ready, proceed without asking for the same authorization again.

Read the conversation, project notes, existing TUNING-SPEC.md, asset and data metadata, interfaces, and historical results. Locate the actual task_tuning and solver imports with the user's interpreter. Do not hardcode the skill author's environment or treat old example settings as verified defaults. Report missing dependencies honestly; never substitute empty look-alike classes.

## Shared decisions and routing

Maintain one case-level TUNING-SPEC.md. For initial setup, recovery across sessions, or conflicting sources, read [intake rules](references/intake.md). Adapt the [specification template](assets/TUNING-SPEC.md), or retain an equivalent existing format.

| Unresolved topic | Skill | Members |
|---|---|---|
| Observations, data splits, loss, acceptance | [tuning-evidence](../tuning-evidence/SKILL.md) | Evidence / Evaluator / Validator |
| Asset, physics, numerics, execution, presentation | [tuning-environment](../tuning-environment/SKILL.md) | Asset / Template / Rollout / Play |
| Parameters, identifiability, strategy, budget | [tuning-search](../tuning-search/SKILL.md) | Parameters / Optimizer / Decision |
| Behavior unsupported by existing types | [tuning-adapter](../tuning-adapter/SKILL.md) | Minimal subclass or solver adapter |

Keep all five skill directories at the same level. If a relative skill is missing, search available skills by name, then report a missing dependency if unresolved. Use only the relevant scope when a child skill is invoked directly.

Ask about intent, unknown facts, and consequential tradeoffs. Inspect code and perform authorized qualification experiments yourself. Usually ask one to three short questions around the largest dependency while continuing independent work. Defaults, silence, and old code do not resolve high-impact unknowns.

## Assembly and handoff

Inspect current interfaces before generating code. The [contract snapshot](references/contracts.md) is a checklist, not an authoritative API for future versions.

1. Record the actual class, constructor settings, and rationale for all ten members: evidence, asset, evaluator, parameters, rollout, optimizer, decision, validator, template, and play. Tuning creates its own RunState.
2. Provide an importable case factory, normally `build_tuning(workspace=...)` in build_tuning.py, returning a fresh Tuning each time. Reuse appropriate factories and adapters. Create adapters only for a demonstrated gap; do not add a configuration compiler or dynamic class registry.
3. Importing and constructing must not simulate. Guard execution with a main entry point. Resolve data and dependencies in the user's environment.
4. Treat workspace as a reusable root; never overwrite previous runs. Associate the specification hash, source/dependency versions, and decision sources with the run using existing metadata or case records. Do not create a second RunState.
5. In a fresh process, call the factory and check real member types, baselines, mappings, fixed/tuned sets, experiment IDs/groups, times/shapes, dependencies, paths, and devices. For construction-only requests, do not call prepare(), run(), or Environment.run().
6. For authorized execution, call run() once on a fresh instance. Sensitivity and identifiability remain Tuning methods; batch scores remain Torch tensors; Optimizer updates state and Decision alone decides whether work continues.
7. Inspect termination, training retest, validation, numerical qualification, and asset export/reload. Report loss improvement, stopping, and acceptance separately.

Currently prepare() creates directories, freezes data, and runs baseline and numerical qualification simulations. It is not a static check. Do not call prepare() and then run() on the same instance.

## Delivery and changes

Deliver the specification, factory entry point, required adapters, and workspace location. Distinguish unresolved configuration, configured, instantiated, qualified, and completed with an acceptance result. Do not claim unexecuted simulation or generalization checks.

Changes to loss/scale/mask invalidate threshold conversions, analysis, and acceptance assumptions. Asset/boundary/binding changes require physical and numerical requalification. Parameter changes require analysis and budget review. Use a fresh instance and run for new configurations. Once holdout results influence model changes, those data are development evidence rather than independent validation.

When modifying this skill module, review the [acceptance scenarios](references/acceptance-scenarios.md). Distinguish format checks, manual review, actual execution, and independent behavioral evaluation.
