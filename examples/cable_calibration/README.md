<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Reproducing the no-plug cable release

[Repository](../../README.md) · [Specification](TUNING-SPEC.md) · [English HTML report](reports/cable_calibration_en.html) · [Chinese HTML report](reports/cable_calibration_zh.html)

A 17 cm cable was clamped at one end, lifted by its free end and released without a plug. The left and right static shapes constrain bending rigidity and clamp angle. The release recording compares two damping models for the same 32-segment cable.

## 1. Verify the reported results without simulation

From the repository root, with the core package installed, run:

```bash
python - <<'PYTHON'
from examples.cable_calibration.verify import verify

result = verify()
for stage in ("Bstar", "C"):
    score = result["metrics"][f"{stage}__noplug_release"]
    print(f"{stage}: {score['coordinate_rmse_mm']:.3f} mm RMSE")
print(f"Objective reduction: {result['sls_objective_reduction_percent']:.4f}%")
PYTHON
```

Expected output:

```text
Bstar: 1.598 mm RMSE
C: 0.844 mm RMSE
Objective reduction: 76.0088%
```

The verifier checks all 14 archived traces on CPU. The displayed scores use the recording that B* and C were fitted to. For the full verifier output, run `python -m examples.cable_calibration.verify`. The [archive](archive/) contains the original stage decisions, diagnostics and outcomes.

## 2. Experiment and observation pipeline

The cable measured 20 cm and 10.4 g, with a 7 mm diameter. The free length from the clamp to the bare tip was **17 cm**. The rest photograph provides the natural centerline. The same clamp held the cable as the board was flipped for the left and right cantilever photographs. The model assumes planar XZ motion and no board contact.

A ruler in the images sets the scale. Each observation has x and z coordinates at 15 locations from 10 to 150 mm along the cable. The measured release window runs from 7.84 to 8.44 s, with 31 frames spaced by 0.02 s. The scoring mask excludes the first three frames, leaving 28. Stage C initializes its material state from 0.6 s of pre-release shape history.

The [processed measurements](data/) and saved predictions provide the inputs to the CPU check. Sampling times come from video presentation timestamps. The final 20 mm of the model extends beyond the measured points.

## 3. Progress and the reason for each model change

All errors below are coordinate errors over the declared masks, not Euclidean point distances.

| Model | Left static RMSE / max (mm) | Right static RMSE / max (mm) | No-plug dynamic RMSE / max (mm) |
|---|---:|---:|---:|
| Initial baseline | 3.486 / 9.760 | 0.421 / 0.895 | 6.366 / 25.127 |
| A: EI + damping | 0.699 / 2.329 | 2.395 / 6.557 | 3.230 / 12.084 |
| B: add shared mount angle | 0.238 / 0.484 | 0.887 / 1.873 | 1.617 / 9.016 |
| B*: freeze static values, retune damping | — | — | 1.598 / 9.353 |
| C: add SLS | 0.241 / — | 0.886 / — | 0.844 / 3.203 |

### A: fit bending rigidity and damping

The first search varied EI and the 32-segment discrete bending damping coefficient. It improved the joint objective from 1.300964 to 0.447918 with 16 candidates over two generations.

| Quantity | Baseline | A result |
|---|---:|---:|
| EI (N m²) | 0.00816 | 0.00343942844749 |
| Discrete bend damping (N m s/rad, 32 segments) | 0.012 | 0.0105463968453 |

The two-parameter fit reduced the release RMSE from 6.366 to 3.230 mm. The opposite trends in the left and right static residuals motivated a closer look at the clamp angle.

**Why add a boundary parameter?** Shared material parameters traded off opposite orientations, while the photographed clamp tangent and effective boundary were uncertain. The same clamp was used for both orientations, motivating one shared installation-angle correction with orientation-dependent sign, rather than unrelated left/right material parameters.

### B: add a shared installation angle

The angle-stage work started from prior fitted results and qualification/warm-start iterations; the archived B configuration is the final search's actual warm start, not simply A's final parameter vector. Its winning values are:

| Quantity | B result |
|---|---:|
| EI (N m²) | 0.00222895992123 |
| Discrete bend damping (N m s/rad, 32 segments) | 0.0171558962616 |
| Shared mount-angle offset | +0.290562502288° |
| Clamp compliance | Fixed to zero, not measured as zero |

B evaluated 24 candidates over three generations with clamp compliance fixed to zero. At its selected parameters, the static RMSE was 0.238 mm on the left and 0.887 mm on the right. The release RMSE was 1.617 mm.

The improvement in both static orientations motivated fixing EI and the mount angle for the B* damping search.

**Attribution limit:** A and B also differ in backend/numerical settings and search history. At B's other fitted values, setting the angle to zero changes the joint objective only from 0.0856535 to 0.0872235 (about 1.8%). The entire A → B improvement cannot be attributed to angle alone.

### B*: isolate the remaining dynamic mismatch

EI and angle were frozen at B's values, and a 17-point damping grid used a dynamic objective that equally weights full-coordinate normalized MSE and distal-z MSE. Its best damping was 0.0147141547819.

Lower damping preserved oscillation but predicted periods around 0.24–0.26 s versus approximately 0.16 s in the measurement. Higher damping suppressed oscillations. This amplitude/frequency tradeoff motivated additional rate-dependent stiffness rather than another adjustment of viscous damping alone.

### C: standard linear solid bending

SLS combines long-time rigidity EI∞ with a relaxing Maxwell branch. EI∞ and the shared angle remain frozen; the previous physical viscous bending coefficient is set to zero. The search fits transient rigidity ΔEI and relaxation time τ:

| Quantity | C result |
|---|---:|
| EI∞ (N m²), frozen | 0.00222895992123 |
| ΔEI (N m²) | 0.00269860632061 |
| τ (s) | 0.0343552374996 |
| Instantaneous EI₀ = EI∞ + ΔEI (N m²) | 0.00492756624184 |

The Maxwell internal state is initialized from observed pre-release history using piecewise exact integration. The earlier unknown history changes the winning prediction by approximately 0.003205 mm in the recorded sensitivity check.

C evaluated 128 candidates over 16 generations. Under the same dynamic observations, mask, scale, and objective as B*, the objective fell from 0.509256746 to 0.122176896, a **76.0% reduction**. The first trough/peak timing improved to 0.08/0.16 s, but later peak intervals of 0.20/0.18 s still exceed the measured 0.16/0.16 s.

This supports SLS as a useful model extension for this window; it does not uniquely identify viscoelasticity as the only cause or establish transferable material constants.

## 4. Numerical accuracy and remaining limits

The reported values apply to the measured no-plug release, the recorded clamp and the 32-segment Newton VBD model. Stage C's [archived diagnostics](archive/C-final-diagnostics.json) record maximum coordinate changes of 0.179 mm under timestep refinement and 0.003 mm under iteration refinement. The 32-to-64-segment comparison gave 0.229 mm.

Replaying Stage C's archived candidate on the [current pinned solver](solver/current-solver.json) gives **0.844 mm coordinate RMSE** on the same recording. Halving the timestep changes the trajectory by 0.182 mm, and doubling the iteration count changes it by 0.0024 mm. Both changes are below the 0.5 mm numerical target.

## 5. Construct or rerun a stage with Newton

### Recorded simulation environment

Newton executes the cable physics, Warp supplies its GPU runtime, and TaskTuning/PyTorch manage calibration and scoring in the same Python process.

| Component | Recorded version or setting | Role |
|---|---|---|
| OS / Python | Linux x86_64 / Python 3.12.13 | Historical execution platform |
| TaskTuning | 0.2.0 | Calibration lifecycle |
| Newton | 1.6.0.dev0, base commit `d37f4d3d341ccce1e06a1dff21e9a054759b4855` | VBD cable dynamics for A/B/B* |
| Warp | 1.17.0 | Newton execution on `cuda:0` |
| GPU | NVIDIA RTX 5070 Ti Laptop GPU | Historical simulation device |
| PyTorch | 2.9.1+cpu | CPU scoring and calibration tensors |
| NumPy / CMA | 2.4.3 / 4.5.0 | Numerical utilities and CMA-ES |
| Stage C solver | Pinned Newton base plus `solver/sls-adapted.patch` | SLS constitutive extension |

Python/Torch/NumPy/CMA versions are retained in each [stage configuration](configs/); solver revisions and patch identity are in [SLS provenance](solver/sls-provenance.json). Warp and GPU settings were recorded in the original experiment specification. This table is a historical environment record, not a complete dependency lockfile or a claim of exact replay on every system.

The `versions.device=cpu` field in exported configurations describes TaskTuning scoring. The separate `template.numeric_settings.device=cuda:0` selects Newton/Warp simulation. A CPU-only PyTorch build can therefore coexist with GPU cable physics.

### Solver compatibility

New runs use Newton `1.9.0.dev0` at commit `c53345412d401232beff0bc71dcb39ce554fdf96` with Warp `1.17.0`. Stage C additionally requires [sls-current.patch](solver/sls-current.patch). The factory checks the rod builder, rod geometry, and VBD solver/test hashes against [current-solver.json](solver/current-solver.json), as well as the actual import location. A released package or another branch is not interchangeable with this pinned source.

The current adapters use `newton.Rod` and the six-DOF rod layout: shear on axes 0–1, stretch on axis 2, bending on axes 3–4, and twist on axis 5. Both bending axes receive the clamp half-span correction and parameter readback checks. The current SLS patch preserves the relaxing bending branch and measured material-history initialization.

The table above, stage configurations, archived traces, [sls-adapted.patch](solver/sls-adapted.patch), and [expected-hashes.json](solver/expected-hashes.json) describe the historical runs. They remain unchanged. New run metadata records the current solver revision, file hashes, and patch identity; historical scores and qualification passes do not apply automatically to the new runtime.

The adapter uses `compute_geometric_rod_kappa_cached_z`, the solver moment/strain arrays, and `_rigid_pose_rebaseline_mask` to initialize and preserve material history from observed pre-release motion. These private bindings are specific to the pinned solver. A replacement solver or public API must reproduce the same initialization and trajectory behavior before its revision and hashes are updated.

### Install the pinned solver

Use a separate environment if an existing Newton installation must remain intact. Python 3.12 matches the recorded minor version. Activate that environment and use its `python -m pip` for all installations below. These instructions target the recorded Linux setup; other platforms have not been qualified for this cable example.

The `cable` extra installs SciPy, not Newton. A plain `pip install newton` or the current tip of the SLS branch does not select the tested solver sources. Install the pinned checkout and apply the current adaptation as shown, rather than substituting a same-named package or another revision.

From the repository root:

```bash
python -m pip install -e ".[cable,cma]"
git clone https://github.com/newton-physics/newton.git ../newton-calibration-base
git -C ../newton-calibration-base checkout --detach c53345412d401232beff0bc71dcb39ce554fdf96
python -m pip install -e ../newton-calibration-base "warp-lang==1.17.0"

git -C ../newton-calibration-base worktree add --detach ../newton-calibration-sls c53345412d401232beff0bc71dcb39ce554fdf96
git -C ../newton-calibration-sls apply "$PWD/examples/cable_calibration/solver/sls-current.patch"

python -m examples.cable_calibration.run --stage A --newton-source ../newton-calibration-base
python -m examples.cable_calibration.run --stage B --newton-source ../newton-calibration-base
python -m examples.cable_calibration.run --stage Bstar --newton-source ../newton-calibration-base
python -m examples.cable_calibration.run --stage C --newton-source ../newton-calibration-sls
```

### Check the active interpreter and runtime

After installation, check the interpreter, imported solver path, package versions, and Warp devices:

```bash
python - <<'PY'
import sys
import newton
import warp as wp

print("Python:", sys.version)
print("Interpreter:", sys.executable)
print("Newton:", newton.__file__)
print("Warp:", wp.__version__)
print("Warp devices:", wp.get_devices())
PY
```

For the archived CUDA settings, Warp must discover a CUDA device and the selected `cuda:0` must be usable with its NVIDIA driver. The factory separately checks the requested Newton source and pinned VBD hashes. Stage C imports the patched checkout selected by `--newton-source`, even when the base checkout is the environment's editable installation.

This preflight initializes device discovery but does not simulate the cable. Successful imports/device discovery and factory construction are preliminary checks; actual numerical qualification occurs inside a new authorized `run()`.

### Construction versus a new simulation

Without --run these commands only construct a fresh real Tuning. They check the pinned runtime-file hashes and import provenance; they do not simulate or create a run directory. Run stages in separate Python processes because Newton imports are process-global.

To launch a **new** calibration, add --run and select a workspace:

```bash
python -m examples.cable_calibration.run --stage C \
  --newton-source ../newton-calibration-sls \
  --workspace workspace/cable-sls --run
```

The archived Decision configurations cap their own run time at 740 s (A), 2180 s (B), 480 s (B*), and 2200 s (C). These are formal run budgets, not all historical setup/diagnostic costs. New qualification can stop a run before search. Runtime, recovered values, and acceptance are not guaranteed to reproduce exactly across hardware or dependencies. A nonaccepted result returns CLI exit code 1.

The recorded simulations used Warp CUDA on an RTX 5070 Ti Laptop GPU; scoring runs on the CPU. --device cpu is an explicit change of numerical conditions and requires new qualification. The factory never reuses a historical qualification pass.

Core tests, archive scoring, factory construction, and trajectory replay serve different purposes. None substitutes for a new numerical qualification and independent validation of a changed Newton/CUDA environment.

## 6. Repository contents and provenance

| Path | Purpose |
|---|---|
| [data](data/) | Processed observations, normalized experiments, prehistory, compressed numeric traces |
| [configs](configs/) | Portable stage configurations, parameters, fixed settings, splits, budgets |
| [archive](archive/) | Original outcomes, summaries, diagnostics, expected metrics, provenance |
| [backends](backends/) | Stage adapters for the current pinned Newton runtime |
| [solver](solver/) | Current and historical SLS patches, runtime hashes, upstream provenance, Newton license |
| [reports](reports/) | English and Chinese reports on the no-plug release |
| [verify.py](verify.py) | CPU archive verification |
| [run.py](run.py) | Factory/CLI using the standard Tuning lifecycle |

The factory selects the adapter required by each stage:

| Stage | Simulation adapter and rollout | Scoring |
|---|---|---|
| A | `backends/a/` | Shared `CableEvaluator` |
| B | `backends/b/` | Shared `CableEvaluator` |
| B* | `backends/b/` (identical to B) | Shared `DistalDynamicEvaluator` |
| C | `backends/c/`, including the SLS extension | Shared `DistalDynamicEvaluator` |

All stages share `backends/validation.py`; B* and C share
`backends/staged_evaluator.py`. B and B* use the same simulation adapter.
A retains its original boundary/frame handling and readback tolerances; C retains
its SLS initialization hooks and environment selection. `newton_backend.py` builds
and advances the physical model, while `rollout.py` batches candidates and handles
execution status. They serve different responsibilities.

The factory assembles these adapters with portable configuration. It performs a
fresh run and does not pretend to resume a historical checkpoint. New run source
hashes include both the selected stage directory and shared scoring modules.

[provenance.json](archive/provenance.json) records the four historical run IDs,
hashes of their original resolved configurations and results, and fingerprints of
the original observation, prehistory, and asset-description files. These hashes
identify records before portable export; they are not checksums of the current
bundled files. Git tracks bundled-file revisions and the earlier export manifest.
Historical outcomes remain in the stage result files; solver provenance is in
[solver](solver/).

The specification and asset note are curated English exports. The HTML reports
embed their figures and can be opened locally from [reports](reports/). Source
links resolve within the checkout; keep both reports in the same folder for
language switching. See [local preview instructions](../../docs/publishing-results.md).

The bundled SLS patch is the tested adaptation onto the pinned Newton base. Its original source revision `72eca36c70e88d1eb0cc540da7a7a4ea44fb4b98`, adaptation history, and patch checksum are retained as provenance; reconstruction uses the official base and bundled patch. Newton-derived code retains its upstream notices and [Apache 2.0 license](solver/NEWTON-LICENSE.md); see [SLS provenance](solver/sls-provenance.json).
