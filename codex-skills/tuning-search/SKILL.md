---
name: tuning-search
description: Configure Parameters, Optimizer, and Decision for task_tuning calibration, including parameter ranges and combinations, identifiability, search strategy, budgets, and stopping. Use for task configuration changes, not ordinary optimization theory questions.
---

# Tuning Search

Read the shared specification, actual interfaces, and evidence/environment decisions.

## Parameters and identifiability

Record units, bounds, baseline and sources, transforms, backend/discretization mappings, coupling, and unknown quantities temporarily fixed. Current parameters are continuous scalars; do not imply automatic mixed/discrete optimization. Explain joint versus staged fitting and conditional uncertainty in values frozen upstream.

Sensitivity and identifiability belong to Tuning. Use its evaluation path rather than a second analysis loop. Construction-only work leaves simulations pending. For execution, inspect perturbation steps, observation noise, multiple anchors, singular values, effective rank, and response correlation. Orthogonal coordinates do not prove physical identifiability. For rank deficiency, propose informative experiments, reduce fitted dimensions with explicit fixed assumptions, or limit the conclusion. A representative solution is not a unique physical solution.

## Strategy and stopping

Check implemented algorithms and dependencies. GridSearch costs grow as levels to the dimension. CMAES needs appropriate scaling, bounded mappings, and complete generations. Bayesian optimization or PBT must not be represented by empty stubs; PBT may require persistent-state contract changes beyond an optimizer subclass.

Optimizer implements initialize/ask/tell, state, and factual diagnostics. It never owns stopping. Decision alone interprets budget, target, patience, exhaustion, failures, and algorithm diagnostics. Do not call an optimizer library's automatic stopping loop.

Measure cost only within authorized execution. Budget qualification, baseline, analysis, complete search batches, export/reload, training retest, validation, and final checks. Record standalone setup costs separately from RunState. Do not start an incomplete generation if the remaining budget cannot authorize it.

Deliver actual Parameters/Optimizer/Decision configurations, selection rationale, analysis evidence or pending work, budget allocation, and stopping semantics in the shared specification. Never copy another case's bounds, tolerances, or budget as verified defaults.
