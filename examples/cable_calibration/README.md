# Cable calibration: A → B → C

[Repository](../../README.md) · [Specification](TUNING-SPEC.md) · [English HTML report](reports/cable_calibration_en.html) · [Chinese HTML report](reports/cable_calibration_zh.html)

This example follows a real 17 cm free cable from a stiffness/damping fit, through a shared clamp-angle correction, to a standard linear solid (SLS) bending model. It includes processed measurements, archived predictions, the actual stage adapters, and factories for new runs.

**The final result is a useful no-plug fit, not an accepted general cable model.** The plugged case remains inaccurate and numerically unqualified. Historical rejection statuses are preserved.

## 1. Verify the reported results without simulation

From the repository root, after installing TaskTuning:

```bash
python -m examples.cable_calibration.verify
```

This CPU command checks 14 archived traces against processed measurements, recomputes coordinate RMSE and maximum error, runs the archived evaluators, checks train/validation group definitions, and reproduces the B* → C objective reduction. It does not import Newton, start an optimizer, or provide new independent validation.

Expected headline: 14 verified traces and approximately 76.0088% reduction in the no-plug dynamic objective from B* to C. Original statuses remain rejected_training for A and rejected_validation for B, B*, and C.

## 2. Experiment and observation pipeline

The measured cable is 20 cm long and 10.4 g, with a 7 mm diameter. Its experimental free length is **17 cm from the clamp to the cable/plug connection**, excluding the 45 mm plug. The plug is 7.1 g. The original filename text “17mm” was a typo.

Six photographs show unloaded rest, left cantilever, and right cantilever configurations with/without the plug. Two left-facing videos show a manually raised free end followed by release. The clamp remained unchanged while the whole board was flipped between left and right. The model assumes planar XZ motion and no board contact; these are stated experimental assumptions.

Processing uses the visible ruler to map projected positions to metric coordinates. Static shapes contain 15 sampled locations at 10–150 mm along the cable, flattened into 30 x/z channels. The last 20 mm of the model depends on extrapolation. Rest photographs provide the natural curved shape.

Each dynamic window has 31 frames at 0.02 s intervals:
- No plug: video time 7.84–8.44 s.
- With plug: video time 9.50–10.10 s.
- The first three frames are excluded from scoring to handle uncertain release/initial velocity; 28 frames remain.
- SLS additionally uses 0.6 s of observed pre-release shape history.

Video presentation timestamps support 50 Hz sampling; recording speed was not independently verified. Ruler-based geometry is not a full camera calibration. Measurement uncertainty and statistical confidence intervals were not established. See [processed data](data/) and the HTML reports for extraction details and photographs.

Raw MOV files, complete search histories, and the original extraction scripts are not bundled. The example reproduces scoring from the exported processed evidence; it does not reproduce raw-image/video preprocessing from scratch.

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

The left static fit improved while the right deteriorated. Dynamic maximum error was 12.084 mm against a 12 mm limit, so A was **rejected_training**. More search could improve this budget-limited result; A alone does not prove the model is incapable of fitting the measurements.

**Why add a boundary parameter?** Shared material parameters traded off opposite orientations, while the photographed clamp tangent and effective boundary were uncertain. The same clamp was used for both orientations, motivating one shared installation-angle correction with orientation-dependent sign, rather than unrelated left/right material parameters.

### B: add a shared installation angle

The angle-stage work started from prior fitted results and qualification/warm-start iterations; the archived B configuration is the final search's actual warm start, not simply A's final parameter vector. Its winning values are:

| Quantity | B result |
|---|---:|
| EI (N m²) | 0.00222895992123 |
| Discrete bend damping (N m s/rad, 32 segments) | 0.0171558962616 |
| Shared mount-angle offset | +0.290562502288° |
| Clamp compliance | Fixed to zero, not measured as zero |

A four-parameter flexible-clamp attempt failed time/space qualification within its budget. The user selected a three-parameter search with compliance fixed to zero. B completed 24 candidates over three generations and met its training stopping target. The plugged dynamic RMSE/max remained 10.281/42.946 mm; formal acceptance was **rejected_validation**.

Both static orientations improved, supporting a conditional freeze of EI and angle for the next dynamic investigation. A 48-candidate static-focused follow-up found no better incumbent but did not complete formal finalization within its 500 s budget. It is supporting local evidence, not a separate accepted calibration.

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

The target is **equivalent parameters for the specified 32-segment model**. Original stretch/shear settings of 3e7 did not converge for the cantilever case; approved template changes used axial/shear settings of 1e5/1e4 with independent benchmark checks. The archived configurations record each stage's exact settings.

At C's no-plug optimum, the recorded maximum changes were approximately 0.179 mm under time refinement and 0.003 mm under iteration refinement, below the 0.5 mm numerical target. The 32 → 64 segment difference was 0.229 mm; spatial differences are retained as diagnostics under the user-approved 32-segment interpretation.

The plugged C prediction remains 10.301 mm RMSE / 43.609 mm maximum error. Loaded time-refinement changes of approximately 0.690, 0.780, and 2.762 mm are nonmonotonic and fail qualification. Plug center of mass, inertia, attachment compliance, effective boundary, and release initialization also add uncertainty. Numerical error prevents assigning the discrepancy entirely to missing physics.

The plugged data initially served as holdout, but later model decisions used their outcomes. They are now historical development diagnostics, not fresh independent validation. The archived HistoricalValidator deliberately prevents a new run from claiming acceptance on this already-used evidence.

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

Stage C requires the recorded base plus `solver/sls-adapted.patch`. The factory checks three solver-file hashes and the actual import location. A released package or another branch is not interchangeable with this pinned source.

The adapter uses `compute_geometric_rod_kappa_cached_z`, the solver moment/strain arrays, and `_rigid_pose_rebaseline_mask` to initialize and preserve material history from observed pre-release motion. These private bindings are specific to the pinned solver. A replacement solver or public API must reproduce the same initialization and trajectory behavior before its revision and hashes are updated.

### Install the pinned solver

Use a separate environment if an existing Newton installation must remain intact. Python 3.12 matches the recorded minor version. Activate that environment and use its `python -m pip` for all installations below. These instructions target the recorded Linux setup; other platforms have not been qualified for this cable example.

The TaskTuning `cable` extra installs SciPy, not Newton. A plain `pip install newton` or the current tip of the SLS branch does not select the tested solver sources. Install the pinned checkout and apply the archived adaptation as shown, rather than substituting a same-named package or another revision.

From the TaskTuning repository root:

```bash
python -m pip install -e ".[cable,cma]"
git clone https://github.com/newton-physics/newton.git ../newton-task-tuning-base
git -C ../newton-task-tuning-base checkout --detach d37f4d3d341ccce1e06a1dff21e9a054759b4855
python -m pip install -e ../newton-task-tuning-base "warp-lang==1.17.0"

git -C ../newton-task-tuning-base worktree add --detach ../newton-task-tuning-sls d37f4d3d341ccce1e06a1dff21e9a054759b4855
git -C ../newton-task-tuning-sls apply "$PWD/examples/cable_calibration/solver/sls-adapted.patch"

python -m examples.cable_calibration.run --stage A --newton-source ../newton-task-tuning-base
python -m examples.cable_calibration.run --stage B --newton-source ../newton-task-tuning-base
python -m examples.cable_calibration.run --stage Bstar --newton-source ../newton-task-tuning-base
python -m examples.cable_calibration.run --stage C --newton-source ../newton-task-tuning-sls
```

### Check the active interpreter and runtime

After installation, check the interpreter, imported solver path, package versions, and Warp devices:

```bash
python - <<'PY'
import sys
import newton
import torch
import warp as wp

print("Python:", sys.version)
print("Interpreter:", sys.executable)
print("Newton:", newton.__file__)
print("Torch:", torch.__version__, "Torch CUDA available:", torch.cuda.is_available())
print("Warp:", wp.__version__)
print("Warp devices:", wp.get_devices())
PY
```

For the archived CUDA settings, Warp must discover a CUDA device and the selected `cuda:0` must be usable with its NVIDIA driver. `torch.cuda.is_available() == False` is expected with CPU PyTorch and does not diagnose Warp's CUDA support. The factory separately checks the requested Newton source and pinned VBD hashes. Stage C imports the patched checkout selected by `--newton-source`, even when the base checkout is the environment's editable installation.

This preflight initializes device discovery but does not simulate the cable. Successful imports/device discovery and factory construction are preliminary checks; actual numerical qualification occurs inside a new authorized `run()`.

### Construction versus a new simulation

Without --run these commands only construct a fresh real Tuning. They check three pinned solver-file hashes and import provenance; they do not simulate or create a run directory. Run stages in separate Python processes because Newton imports are process-global.

To launch a **new** calibration, add --run and select a workspace:

```bash
python -m examples.cable_calibration.run --stage C \
  --newton-source ../newton-task-tuning-sls \
  --workspace workspace/cable-sls --run
```

The archived Decision configurations cap their own run time at 740 s (A), 2180 s (B), 480 s (B*), and 2200 s (C). These are formal run budgets, not all historical setup/diagnostic costs. New qualification can stop a run before search. Runtime, recovered values, and acceptance are not guaranteed to reproduce exactly across hardware or dependencies. A nonaccepted result returns CLI exit code 1.

The recorded simulations used Warp CUDA on an RTX 5070 Ti Laptop GPU; generic Torch scoring can remain on CPU. --device cpu is an explicit change of numerical conditions and requires new qualification. The factory never reuses a historical qualification pass.

Core tests, archive scoring, factory construction, and trajectory replay serve different purposes. None substitutes for a new numerical qualification and independent validation of a changed Newton/CUDA environment.

## 6. Repository contents and provenance

| Path | Purpose |
|---|---|
| [data](data/) | Processed observations, normalized experiments, prehistory, compressed numeric traces |
| [configs](configs/) | Portable stage configurations, parameters, fixed settings, splits, budgets |
| [archive](archive/) | Original outcomes, summaries, diagnostics, expected metrics, provenance |
| [backends](backends/) | Historical adapters with byte-identical stage copies shared |
| [solver](solver/) | Adapted SLS patch, upstream provenance, expected hashes, Newton license |
| [reports](reports/) | Self-contained English and retained Chinese HTML reports |
| [verify.py](verify.py) | CPU archive verification |
| [run.py](run.py) | Factory/CLI using the standard Tuning lifecycle |

The factory selects the historical implementation required by each stage:

| Stage | Simulation adapter and rollout | Scoring |
|---|---|---|
| A | `backends/a/` | Shared `CableEvaluator` |
| B | `backends/b/` | Shared `CableEvaluator` |
| B* | `backends/b/` (identical to B) | Shared `DistalDynamicEvaluator` |
| C | `backends/c/`, including the SLS extension | Shared `DistalDynamicEvaluator` |

All stages share `backends/validation.py`; B* and C share
`backends/staged_evaluator.py`. Only byte-identical copies are consolidated.
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
