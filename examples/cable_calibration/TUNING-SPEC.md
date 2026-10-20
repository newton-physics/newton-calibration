# Cable calibration specification and decision record

This is a curated English export of the original case specification. Historical outcomes remain in [archive](archive/); exact runnable configuration is in [configs](configs/). The original workspace and records are unchanged. See [provenance](archive/provenance.json) for source and export hashes.

## Scope

Fit equivalent parameters for the specified 32-segment Newton VBD cable, using measured static shapes and release dynamics. Preserve one Tuning lifecycle with case-specific Asset, Rollout, Evaluator, and Validator implementations. The repository export provides archived scoring and fresh construction/execution, not automatic resume.

Targets: static coordinate RMSE ≤3 mm and maximum ≤9 mm; dynamic RMSE ≤4 mm and maximum ≤12 mm. Numerical changes should remain below 0.5 mm for the qualified deployment conditions. The user later explicitly accepted 32-segment equivalent parameters, with 32 →64 differences as diagnostics rather than a search-blocking material-transferability test.

Historical conclusion: A rejected_training; B/B*/C rejected_validation. No general-purpose asset was accepted.

## Confirmed facts and assumptions

| Issue | Decision and source | Consequence |
|---|---|---|
| Filename length17mm | User clarified 17 cm free cable | Use 0.17 m; exclude plug length |
| Rest image | User identified unloaded flat natural shape | Fit nonstraight rest geometry |
| Left/right | Same clamp, whole board flipped | Share material and mount-angle parameter; apply orientation sign |
| Contact/motion | User-approved no-board-contact and planar XZ assumptions | Results conditional on these assumptions |
| Dynamic excitation | User described manual tip lift then free release | Initialize from release frames; no ongoing trajectory forcing |
| Physical asset | Measured 20 cm /10.4 g, 7 mm diameter, 7.1 g /45 mm plug | Linear density 0.052 kg/m; plug inertia still approximate |
| Additional experiment | Unavailable during this study | Retain boundary and observation uncertainties |

The plug uses a 13.4 ×16.8 ×45 mm uniform-box proxy. Its actual center of mass, inertia, and connection compliance were not separately identified.

## Evidence and scoring

[data/experiments.json](data/experiments.json) contains six normalized experiments: no-plug and with-plug left/right static shapes and release trajectories. Static observations have one frame and 30 coordinates from 15 points at 10–150 mm. Dynamics contain 31 samples spaced by 0.02 s, with the first three masked from scoring.

No-plug release uses 7.84–8.44 s; with-plug uses 9.50–10.10 s. Video timing follows presentation timestamps, without independent speed verification. Ruler-based projected geometry is not full camera calibration. Extrapolation covers the model beyond 150 mm; no statistical uncertainty intervals were obtained.

A/B jointly fit no-plug left/right/static and dynamic evidence. Plugged experiments initially formed validation groups, but later inspection influenced model development. They cannot now establish independent generalization.

B*/C train only on no-plug dynamics. Their objective is 0.5 times full-coordinate normalized MSE plus 0.5 times distal-z normalized MSE at 130/140/150 mm, using 4 mm scale and the same mask. This permits a direct B* →C comparison; A/B objective changes do not support a controlled angle-only attribution.

## Physical and numerical model

| Layer | Setting and rationale |
|---|---|
| Physical | 0.17 m free cable, measured line density, natural curvature, gravity, optional plug proxy |
| Boundary | Shared mount angle, rigid clamp in final searches; compliance fixed to zero after failed qualification, not measured zero |
| Discretization | 32 segments, physical-to-joint EI/h mapping with doubled root stiffness; damping tied to this discretization |
| Numerics | Per-stage dt, iterations, frame orientation, fixed axial/shear settings and refinements in configs |
| Initialization | Static settling; measured dynamic release state and estimated initial motion |
| SLS | Maxwell branch parallel to EI∞; 0.6 s measured prehistory with piecewise exact state initialization |

The original 3e7 stretch/shear values failed cantilever convergence. The user authorized validated template changes; later 1e5 axial/1e4 shear settings were benchmarked. Static settling uses numerical damping distinct from fitted physical damping. A's archived dynamic settings differ from B/C; always use the matching source snapshot and configuration.

C's winning no-plug time/iteration changes are approximately 0.179/0.003 mm. Its 32 →64 difference is approximately 0.229 mm, diagnostic under the stated deployment scope. Loaded time refinements fail 0.5 mm with nonmonotonic changes up to approximately 2.762 mm; loaded conclusions remain numerically unqualified.

## Stage decisions

| Stage | Active parameters | Evidence/problem | Decision and limitation |
|---|---|---|---|
| A | EI, discrete bend damping | Static left/right tradeoff; dynamic max 12.084 mm | Investigate shared boundary error; budget-limited failure does not prove model impossibility |
| B | EI, damping, shared mount angle | Uncertain clamp tangent and flipped-board consistency | Three-parameter model after flexible-root qualification failed; joint fit improves, plug fails |
| B* | Damping only | Static fit supports conditional freeze; dynamic amplitude/period tradeoff persists | Freeze EI/angle locally, use distal-weighted dynamic objective |
| C | ΔEI, τ | Viscous damping cannot match both retained oscillation and frequency | Add SLS, freeze EI∞/angle, set old physical damping to zero; no-plug fit improves, plug still fails |

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
- HistoricalValidator preserving the diagnostic status of plugged evidence.
- Generic Play; Tuning creates its own RunState.

The original user allocations were 30-minute rounds followed by additional one-hour investigations. Formal archived run limits are A 740 s /16 trials, B 2180 s /192 trials, B* 480 s /17 trials, and C 2200 s /128 trials. B reached its target after 24 candidates. Standalone qualification and diagnostics consumed additional historical allocation outside these run counters.

The factory only imports/constructs unless --run is supplied. Fresh execution requalifies numerics and uses a unique workspace subdirectory. It does not reuse qualification passes or overwrite historical records. Three solver files are checked against pinned hashes; complete dependency/hardware equivalence is not guaranteed.

## Outstanding evidence

Independent measurements are needed to assess camera/timing uncertainty, boundary behavior, loaded mass/inertia/connection, and model generalization. Resolve loaded numerical convergence before attributing plugged discrepancies to constitutive physics. Cross-discretization material transfer and statistical identifiability are outside the established result.
