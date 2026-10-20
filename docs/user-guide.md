# TaskTuning user guide

[Home](../README.md) · [Skills](codex-skills.md) · [Real cable example](../examples/cable_calibration/README.md)

## 1. Install

Use Python 3.11 or newer in an environment of your choice:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cma,dev]"
```

Run these commands from the repository root. On Windows, activate `.venv\Scripts\activate` instead. No particular Conda environment or developer-machine directory is required.

| Extra | Purpose |
|---|---|
| no extra | NumPy + PyTorch, GridSearch, headless scoring and USDA export |
| cma | CMA-ES optimizer |
| gl | Pyglet playback of saved traces |
| dev | Pytest, Ruff, pre-commit and package build tools |
| cable | SciPy for the optional real-cable simulator adapters; Newton is installed separately |

The examples under `examples/` are repository examples, not part of the installed core API. Run them from a checkout. CPU Torch is sufficient for the generic demos and archive verification. The real cable calibration uses Newton VBD with Warp CUDA for physics and CPU Torch for scoring. Install the pinned Newton checkout (plus the SLS patch for stage C) using the [cable environment instructions](../examples/cable_calibration/README.md#5-construct-or-rerun-a-stage-with-newton). The cable extra alone does not install this solver. The recorded environment is documented there separately from the generic package requirements.

## 2. Run a complete synthetic calibration

```bash
task-tuning demo --case sliding --optimizer grid --workspace workspace/sliding
task-tuning demo --case oscillator --optimizer cma_es --capacity 4 --workspace workspace/oscillator
task-tuning inspect workspace/oscillator/<run-id>
```

Replace `<run-id>` with the created UTC timestamp/random-suffix directory. Sliding fits friction; the oscillator fits stiffness and damping against independently calculated analytical observations.

The workspace root is reusable. Every new Tuning instance allocates a new run directory and leaves earlier runs intact.

CLI exit code 0 means the result is accepted; 1 means that a completed demo did not produce an accepted result. Inspect both `acceptance` and `termination_reason` instead of assuming every nonzero exit is a program crash.

## 3. Use the Python API

```python
from task_tuning.examples import make_tuning

tuning = make_tuning(
    "oscillator",
    workspace="workspace/api_example",
    optimizer="cma_es",
    capacity=4,
)
# Construction creates no simulation and does not start optimization.
result = tuning.run()
print(result.acceptance)
print(result.termination_reason)
print(result.parameters)
print(result.run_directory)
print(result.asset_path)
```

Use a fresh instance for each run. Do not call `prepare()` followed by `run()` on the same instance: preparation already executes baseline and numerical qualification and starts the lifecycle.

`Tuning.sensitivity()` and `Tuning.identifiability()` belong to this lifecycle; they are not separate scheduler objects. Rank deficiency normally stops the task as inconclusive. Opting into rank reduction restricts the searchable combinations and fixes unidentifiable directions; it does not identify every original physical parameter.

## 4. Assemble your own case

Use [the synthetic factories](../src/task_tuning/examples.py) as the smallest complete examples, and [the cable factory](../examples/cable_calibration/run.py) for real observations and custom solver members.

1. **Evidence:** create Experiment objects with reference `[T,D]`, times `[T]`, broadcastable positive scales, a Boolean `[T,D]` mask, weights and acquisition-group IDs. Separate training and validation acquisition groups. Slicing one recording does not create an independent experiment.
2. **Parameters:** define units, bounds, baselines and linear/log transforms. The asset baseline must match exactly. Define only the parameters you are actually fitting.
3. **Template:** declare fixed physics, initial states, excitations, seeds, numerical settings and discretization. Add refinement templates for timestep/iteration accuracy; decide explicitly whether spatial refinement is a gate or diagnostic.
4. **Asset / Environment:** implement `Asset.create_environment(template, parameter_values, experiment_settings)`. Each environment owns its mutable simulator state. `applied_parameters` must read back actual simulator values, not echo input.
5. **Environment.run(times, seed, deadline):** return a Trace aligned to the requested times. Values are `[T,D]`; optional positions are `[T,N,3]`. Use the template integration timestep and honor the monotonic-clock deadline. Raise SimulationError for candidate numerical failures; distinguish implementation errors.
6. **Rollout / Evaluator:** choose capacity and backend readback tolerances. Keep episode identity and batch order intact. Custom evaluators must preserve losses, metrics, constraints, valid masks and residual-feature contracts.
7. **Optimizer / Decision:** choose GridSearch or CMAES and set all-phase budgets. Decision alone authorizes continuation.
8. **Validator / Play:** retest a reloaded frozen asset on training data and separate validation; save visual comparisons from recorded traces. Do not send validation scores back to the optimizer.
9. Compose a Tuning and call its existing `run()` once.

Asset's default JSON export contains parameters, adapter type and constructor configuration. It still needs the same adapter and template to create a simulator. Override export/reload for a different physical asset format.

### Numerical qualification policy

`Template` defaults to `qualification_mode="refinement"`. At least one actual change to numerical settings or discretization is required; an empty refinement list never silently disables qualification.

For a deliberately fixed solver configuration, select `qualification_mode="repeatability"` and `refinements=()`. The lifecycle runs independent environments with identical settings and seeds at representative parameters, then repeats the selected candidate after export/reload. The tolerance still compares normalized residual features. This checks repeatability with fixed seeds, not variability across random seeds or numerical convergence. Search is rejected if the comparison exceeds the tolerance.

The numerical analysis records `mode="repeatability"` and `scope="fixed_configuration_only"`; the final validation report and saved template also retain the selected policy. An accepted result is conditional on that policy. Budget reservation includes the final repeated evaluation. Combining this mode with refinements raises an error. It is not selected automatically for cable cases.

## 5. Scores and thresholds

Default squared-error loss normalizes residuals by observation scales and weights. Do not confuse a dimensionless objective, physical coordinate RMSE and maximum coordinate error.

The evaluator also supports probability-distribution cross-entropy; its attainable minimum includes the reference entropy and should not automatically be set to zero.

`BatchResult` preserves:
- `search_values [B,K]`, `parameter_values [B,P]`;
- `objectives [B]`, `valid_mask [B]`;
- `metrics [B,M]`, `constraints [B,C]`, `features [B,F]`;
- episode records identifying trial, experiment and repeat.

Failed/missing simulations do not receive zero loss. Candidates outside the original physical bounds are invalid rather than clipped. Optimizer feedback uses feasibility penalties; original scores remain stored.

The cable case demonstrates why average coordinate error can pass while distal oscillations remain wrong. It also shows why changing the loss requires an explicitly named comparison protocol.

## 6. Budgets and acceptance

| Setting | Counts |
|---|---|
| max_trials | Training candidates |
| max_rollouts | Baseline, analysis, qualification, training, retest and validation episodes |
| max_generations | Completed optimizer generations |
| max_seconds | Wall time across the complete Tuning run |

Reserve budget for closing checks. A good training candidate can remain unvalidated if time or rollout budget is exhausted before reload/retest/refinement. Threaded deadlines are cooperative; a blocked simulator cannot be forcibly killed by the current rollout.

Typical outcomes include `accepted`, `rejected_training`, `rejected_validation`, `unvalidated` and `inconclusive`. Read the result and RunState for the reason. A numerical qualification failure is a reason to examine the backend/settings, not a physical measurement of the optimum.

## 7. Read the artifacts

```text
workspace/
  <run-id>/
    resolved.json
    run_state.json
    batches/*.pt
    checkpoints/*.pkl
    candidate_asset.json       # once candidate export is reached
    result.json
    comparison.usda           # when comparison data are available
    release/                  # accepted results only
      asset.json
      template.json
      manifest.json
```

Batch/checkpoint references are relative to the run directory. Read tensor archives with `torch.load(path, weights_only=True)`. Checkpoints capture optimizer state; automatic resume is not implemented.

`resolved.json` stores observations, frozen settings, identities and versions. `run_state.json` records budget use and decisions. Release files include rebuilding conditions; candidate parameters alone are insufficient.

## 8. Playback

```bash
python -m pip install -e ".[gl]"
task-tuning demo --case sliding --optimizer grid --workspace workspace/view --gl
task-tuning demo --case sliding --optimizer grid --workspace workspace/headless --headless-gl --gl-frames 30
```

These CLI commands run a new demo, then display saved comparison traces. Playback itself does not advance physics. GL currently uses an XY projection, and USDA export is animated points/trajectories, not a mesh or physical Newton scene. Headless GL still needs a working graphics context.

## 9. Troubleshooting

| Symptom | Check |
|---|---|
| Missing cma | Install the cma extra, or choose GridSearch |
| Parameter readback mismatch | Backend binding, units, root/segment scaling, float precision |
| Numerical qualification fails | Timestep, iterations, initialization, solver frame and refinement protocol |
| Rank deficient | Excitation/observations and parameter coupling; do not silently add more parameters |
| Low training loss but no release | Training retest, validation, numerical checks and remaining budget |
| Cable source mismatch | Use the specified Newton checkout/patch and a fresh Python process |
| Archive verification succeeds | Only confirms saved traces and scores agree; it is not a new calibration |

## 10. Checks

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m examples.cable_calibration.verify
ruff check src tests examples/cable_calibration/run.py examples/cable_calibration/verify.py
```

The frozen cable backend files preserve historical source bytes and formatting for provenance. New factory/verification code is checked separately. See the case README for optional solver-dependent construction checks and their limits.
