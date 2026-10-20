---
name: tuning-evidence
description: Configure or revise Evidence, Evaluator, and Validator for task_tuning calibration, including measured or teleoperation observations, train/holdout splits, loss, weights, and acceptance thresholds. Use for configuration decisions, not merely reading a score or explaining an existing loss.
---

# Tuning Evidence

Read the case specification and actual member interfaces. Update the shared TUNING-SPEC.md; do not create a separate workflow.

## Establish what is observed

Distinguish material points, spatial curves, image outlines, time series, and probability distributions. Record acquisition groups, coordinates, units, physical time, correspondence, valid windows, masks, repeats, and uncertainty. Resolve contradictory units or timing before treating inputs as formal evidence. A static shape is not automatically material-point correspondence; video does not directly establish accurate velocity.

Split by independent acquisition or condition. Adjacent frames, resampling, or another seed do not create independent measured evidence. If no independent holdout exists, report that limitation instead of copying training data.

## Choose scoring with explicit meaning

| Evidence | Candidate approach | Required check |
|---|---|---|
| Corresponding points or states | Weighted squared residuals | Units, scales, correspondence, point and experiment weights |
| Unmatched outlines or curves | Geometric distance | Lost material correspondence and identifiability |
| Outliers | Robust loss or predefined masks | Evidence-based exclusions, never candidate-dependent deletion |
| Distributions | Cross entropy | Support, normalization, target entropy |
| Dynamics | Fixed-time trajectories and declared features | Synchronization, initial conditions, physical sampling |

The generic Evaluator currently implements squared_error and cross_entropy. Other semantics require an actual adapter, not just a new name. Distinguish measurement noise, normalization scale, and engineering tolerance. Define aggregation, weights, valid sample counts, and failure penalties. Lower is better; cross-entropy targets need not have zero loss.

Separate training score, Decision stopping target, and Validator acceptance. Do not relax tolerances automatically when a fit fails. Convert physical numerical tolerances into the actual feature/metric space, including scale, weights, masks, and sample count. Keep analysis features, metric_spec/fingerprints, and Validator metric indices consistent.

## Deliver and verify

Record Evidence/Evaluator/Validator types and settings, source and preprocessing provenance, acquisition groups, splits, thresholds, and remaining uncertainty. For new metrics, verify an independently calculated expected value and failure cases.

Configuration-only work must not call prepare() or run(). If holdout outcomes guide model revisions, label them development evidence and plan genuinely independent validation.
