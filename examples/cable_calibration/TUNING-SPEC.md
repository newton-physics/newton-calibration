<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Cable calibration specification and decision record

This specification describes the no-plug reproduction. The original run IDs and evidence fingerprints are in [provenance](archive/provenance.json), the stage decisions and diagnostics are in the [archive](archive/), and the runnable settings are in [configs](configs/).

## Scope

Fit equivalent parameters for a 32-segment Newton VBD cable using the two measured cantilever shapes and the unforced no-plug release. Reproduce the coordinate errors and the matched B* and C fitting objectives on that recording.

The study used static targets of 3 mm RMSE and 9 mm maximum coordinate error, and dynamic targets of 4 mm RMSE and 12 mm maximum error. The numerical refinement target was 0.5 mm for the deployed 32-segment model. The 64-segment run provided a separate discretization diagnostic.

## Confirmed facts and assumptions

| Issue | Decision and source | Consequence |
|---|---|---|
| Filename length17mm | User clarified 17 cm free cable | Use 0.17 m from clamp to free tip |
| Rest image | User identified unloaded flat natural shape | Fit nonstraight rest geometry |
| Left/right | Same clamp, whole board flipped | Share material and mount-angle parameter; apply orientation sign |
| Contact/motion | User-approved no-board-contact and planar XZ assumptions | Results conditional on these assumptions |
| Dynamic excitation | User described manual tip lift then free release | Initialize from release frames; no ongoing trajectory forcing |
| Physical asset | Measured length 20 cm, mass 10.4 g and diameter 7 mm | Linear density 0.052 kg/m |
| Additional experiment | Unavailable during this study | Retain boundary and observation uncertainties |

## Evidence and scoring

The no-plug records in [experiments.json](data/experiments.json) are `noplug_left`, `noplug_right`, and `noplug_release`. Each static observation has 30 x/z coordinates from 15 points at 10–150 mm. The release window is 7.84–8.44 s with 31 frames sampled every 0.02 s. Scoring excludes the first three frames.

A and B fit the two static shapes and the release. B* and C fit the release alone. Their shared objective weights the normalized MSE across all coordinates and the normalized z MSE at 130, 140 and 150 mm equally, using a 4 mm scale and the same mask. The reported RMSE in mm uses all scored coordinates.

## Physical and numerical model

| Layer | Setting and rationale |
|---|---|
| Physical | 0.17 m free cable, measured line density, natural curvature, gravity and a free tip |
| Boundary | Shared mount angle and rigid clamp in the final searches. Clamp compliance is fixed to zero by assumption. |
| Discretization | 32 segments, physical-to-joint EI/h mapping with doubled root stiffness; damping tied to this discretization |
| Numerics | Per-stage dt, iterations, frame orientation, fixed axial/shear settings and refinements in configs |
| Initialization | Static settling; measured dynamic release state and estimated initial motion |
| SLS | Maxwell branch parallel to EI∞; 0.6 s measured prehistory with piecewise exact state initialization |

The final templates use axial and shear settings of 1e5 and 1e4, respectively, following the cantilever benchmark. Static settling uses numerical damping distinct from fitted physical damping. A's archived dynamic settings differ from B/C; always use the matching source snapshot and configuration.

C's winning no-plug time/iteration changes are approximately 0.179/0.003 mm. Its 32 →64 difference is approximately 0.229 mm, diagnostic under the stated deployment scope.

## Stage decisions

| Stage | Active parameters | Evidence/problem | Decision and limitation |
|---|---|---|---|
| A | EI, discrete bend damping | Static left/right tradeoff; dynamic max 12.084 mm | Test one shared mount-angle correction against the two static shapes |
| B | EI, damping, shared mount angle | Uncertain clamp tangent and flipped-board consistency | Fit EI, damping and a shared angle with clamp compliance fixed to zero |
| B* | Damping only | Static fit supports conditional freeze; dynamic amplitude/period tradeoff persists | Freeze EI/angle locally, use distal-weighted dynamic objective |
| C | ΔEI, τ | Viscous damping cannot match both retained oscillation and frequency | Fit the SLS branch with EI∞ and angle fixed, giving a no-plug release RMSE of 0.844 mm |

Winning values and per-stage residuals are tabulated in [README](README.md). A →B is not a controlled ablation: angle set to zero at B's other values worsens the objective only about 1.8%. A static-focused follow-up found no better incumbent but exhausted finalization budget. The freeze therefore remains conditional, without confidence intervals.

C reduces the directly comparable B* objective by 76.0088%. Improved first-cycle timing and decay support further study; remaining later-cycle mismatch prevents identifying SLS as a unique explanation.

## Members, budgets, and provenance

[run.py](run.py) constructs:
- Evidence from exported normalized experiments and explicit source hashes.
- Historical CableAsset or SLSCableAsset, with portable parameters and fixed template.
- CableEvaluator for A/B, DistalDynamicEvaluator for B*/C.
- Parameters from stage definitions; Template from the actual frozen configuration.
- NewtonRollout with capacity 8.
- CMAES (batch 8, sigma 0.2, seed 42), except B* GridSearch (17 levels, batch 8).
- Decision from archived run limits.
- HistoricalValidator retaining the original evidence groups and acceptance protocol.
- Generic Play; Tuning creates its own RunState.

The original user allocations were 30-minute rounds followed by additional one-hour investigations. Formal archived run limits are A 740 s /16 trials, B 2180 s /192 trials, B* 480 s /17 trials, and C 2200 s /128 trials. B reached its target after 24 candidates. Standalone qualification and diagnostics consumed additional historical allocation outside these run counters.

The factory only imports/constructs unless --run is supplied. Fresh execution requalifies numerics and uses a unique workspace subdirectory. It does not reuse qualification passes or overwrite historical records. Three solver files are checked against pinned hashes; complete dependency/hardware equivalence is not guaranteed.

## Outstanding evidence

Independent release recordings would test transfer to other amplitudes and clamp settings. The reported parameters and errors describe the 32-segment model on the fitting recording.
