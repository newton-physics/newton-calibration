# task_tuning contracts and version checks

This is an interface snapshot, not a fixed SDK. Use the selected interpreter to inspect task_tuning.__file__, signatures, and callers. Follow the actual implementation and record differences; do not force a version rollback.

## Objects and data

- Tuning takes keyword-only evidence, asset, evaluator, parameters, rollout, optimizer, decision, validator, template, play, workspace, and analysis settings. It creates RunState.
- Evidence takes train_data, validation_data, and sources. Experiment contains id, group, reference [T,D], times [T], scale, mask, weight, and repeats. References normalize to CPU float64. Check broadcasting, finite values, time rules, and acquisition-group separation. Raw video/image/teleoperation ingestion belongs in preprocessing or adapters.
- Asset(baseline_parameters, config) creates fresh Environment objects through create_environment(template, parameter_values, experiment_settings). Environment.run(times, seed, deadline) returns Trace. Check close() requirements. applied_parameters must reflect the bound model.
- Template takes numeric_settings, discretization, fixed_parameters, experiment_settings, refinements, precision_tolerance, and qualification_mode. The default refinement mode requires actual numerical changes; explicit repeatability requires no refinements and only checks repeated fixed settings/seeds. Freezing dictionaries cannot prove physics. Refinements represent separate numerical/discretization comparisons.
- Parameter defines name, bounds, baseline, unit, and transform. Parameters handles continuous scalar linear/log mappings and combination directions. Use its mapping API; do not implement another SVD path.
- Evaluator.score_batch returns losses, metrics, constraints, valid, and features. Generic modes are squared_error/cross_entropy; metrics are normalized_rmse and normalized_max_error. New semantics require matching metric_spec/fingerprints and Validator. Read source for exact layout, aggregation, and failures.
- Default Rollout uses threads. Do not infer isolation, device, or effective capacity from its name; preserve batch-result protocols.
- GridSearch/CMAES implement initialize, ask, tell, diagnostics, and state_dict. CMAES requires cma and full generations. Check finite-feedback/failure conventions at the caller boundary. Other algorithms require actual implementations.
- Decision owns continuation/stopping with trial, rollout, generation, seconds, target, and patience limits. Optimizer.stop() must not acquire this responsibility.
- Validator uses max_objective/max_error; the maximum-error check depends on the second metric column. Custom metrics cannot blindly reuse that index.
- Generic Play supports saved XY GL traces and point-trajectory USD, not full asset rendering.

## Construction versus execution

Construction plans a unique run_directory and creates RunState; execution creates the directory. Reuse workspace roots while preserving historical runs.

prepare() freezes data and runs baseline/numerical qualification. run() calls prepare() internally, then analysis, search, retest, validation, and export. Therefore:
- Never call prepare() for a build-only check.
- Never call prepare() followed by run() on the same instance.
- Do not silently run sensitivity or identifiability during static diagnostics.
- Execute once through run() on a fresh object.

RunState tracks formal stage costs; standalone setup checks are separate. Search termination and candidate acceptance are distinct. Inspect whether validation actually ran, budget was sufficient, and export/reload succeeded.

## Minimal construction evidence

Call the real factory. Check Tuning, all ten members, RunState, baseline/fixed/tuned conflicts, IDs/groups/times/shapes, dependencies, and paths. Class names in strings are not evidence of instantiation.

Static checks do not establish numerical accuracy, actual concurrency, solver binding, or generalization. Execute those checks only within authorized scope or mark them pending. Factory module top level should import/define and read necessary configuration without optimizing or creating workspaces.
