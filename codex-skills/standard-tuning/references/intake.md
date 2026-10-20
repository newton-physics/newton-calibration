# Intake, decisions, and recovery

## Evidence and responsibility

| Situation | Action |
|---|---|
| The user already answered, or a reliable fact exists | Cite the answer/source; do not ask again |
| A fact can be checked in data or code | Inspect imports, metadata, binding, and existing checks first |
| Intent or consequential tradeoff is unknown | Explain the impact and offer concise, justified options |
| A technical choice needs validation | Propose and test it; user agreement cannot establish numerical accuracy |
| A physical quantity is unknown | Measure, leave unresolved, or use an explicitly limited diagnostic assumption |

Existing configuration proves what was used, not why it is correct. An unknown answer, silence, or elapsed time is not confirmation. Investigate conflicting sources before publishing a formal runnable configuration that depends on them.

## Questions

Usually ask one to three short questions about the largest current dependency. Include known facts, what is missing, why it matters, and the main tradeoffs. Avoid field-by-field questionnaires or repeating established geometry, mass, or version information.

Examples:
- Geometry and mass are known; clarify whether the goal is transferable material properties or deployment-specific equivalent parameters.
- Video supports projected shape, but material correspondence is unresolved; establish whether geometry alone satisfies the target.
- Old code provides time steps without convergence evidence; inspect binding and refinement yourself, while asking the user about deployment constraints and allowed error/cost.

Proceed when existing intent is sufficient. Record low-impact presentation choices yourself; do not invent recommended physical facts.

## Shared record

For each decision record the issue, conclusion/status, source, rationale/alternatives, affected members, verification, and invalidation conditions. Separate user confirmation from technical verification. Status may be resolved, pending verification, unresolved, or a limited assumption.

All child skills update one TUNING-SPEC.md and return actual configuration/classes, rationale, and remaining questions. Preserve source paths, acquisition groups/times, or code locations; a skill recommendation is not measured evidence. Do not rely on temporary in-memory objects across sessions.

Reopen only new or invalidated decisions:
- Loss/scale/mask changes: thresholds, identifiability, acceptance.
- Physical boundary/asset/binding changes: physical benchmarks and numerical qualification.
- Parameter changes: analysis, search dimensions, budget.
- Model changes after inspecting holdout results: those data are now development evidence.

## Scope and blockers

Design requests produce design; construction requests produce and instantiate code; authorized calibration proceeds through run() when prerequisites are met. Do not expand design into simulation or repeatedly request existing authorization.

Continue independent checks while necessary facts are pending. Label diagnostic assumptions and limits. Report actual dependency, permission, or data blockers without fabricated checks or placeholder implementations.
