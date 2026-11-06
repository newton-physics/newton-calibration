# Newton Calibration

Newton Calibration is a toolbox for fitting simulation parameters to measured data. It supplies components, for example `Evidence`, `Parameters`, `Asset`, `Evaluator`, `Optimizer` and `Validator`. You select, configure and extend these components to build a calibration workflow for your own system.

The goal is a collection of recipes: complete calibration workflows for typical problems that you can adapt to your own case.

> **Pre-release.** This package has not had a release yet. The API, file formats and command line can change without notice.

## Requirements

- Python 3.11 or later. NumPy is the only core dependency.
- The [cable example](examples/cable_calibration/README.md) needs a pinned Newton revision to simulate. Its archived results used a CUDA GPU. A different device changes the numerical conditions.

## Quick start

```bash
git clone https://github.com/newton-physics/newton-calibration.git
cd newton-calibration
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cma,dev]"

# A complete CPU calibration using an independent analytical reference:
newton-calibration demo --case sliding --optimizer grid --workspace workspace/sliding

# Two parameters, CMA-ES:
newton-calibration demo --case oscillator --optimizer cma_es --workspace workspace/oscillator

# Re-score the real cable's archived results, without Newton or simulation:
python -m examples.cable_calibration.verify
```

Use `newton-calibration inspect workspace/sliding/<run-id>` to read a result.
The demo CLI exits with code 0 only for an accepted result. See the [user guide](docs/user-guide.md) for installation, API usage, outputs and failure interpretation.

## Architecture

Ten members compose a Tuning instance; `RunState` is owned by Tuning.

| Member | Responsibility |
|---|---|
| Evidence | Observations, masks, scales, acquisition groups and train/validation split |
| Asset | Fresh environments, applied parameter readback, export/reload |
| Template | Fixed experimental conditions, discretization and numerical refinements |
| Parameters | Units, bounds, baselines and linear/log transforms |
| Rollout | Isolated trial × experiment × repeat execution |
| Evaluator | Batched NumPy losses, metrics, constraints and residual features |
| Optimizer | GridSearch / CMA-ES ask and tell |
| Decision | Sole continue/stop authority and all-phase budgets |
| Validator | Frozen-candidate training retest and validation |
| Play | Saved-trajectory comparison and USDA point animation |

```text
freeze inputs → baseline & numerical qualification → local sensitivity / SVD
→ Decision → ask → rollout → score → tell → stop
→ export/reload → training retest → validation → final numerical checks
→ release only if accepted
```

Numerical qualification defaults to explicitly declared refinements. `Template(qualification_mode="repeatability", refinements=())` instead checks repeated execution of the same fixed configuration. Reports record this narrower scope; repeatability does not establish timestep or discretization convergence.

A new case composes these objects. It does not replace `Tuning.run()`.
See the [class diagram](docs/tuning-class-diagram.md) and [design](docs/generic-calibration-workflow-design.md).

## Real example: cable calibration

[![Cable calibration results: errors and measured/predicted dynamics](examples/cable_calibration/reports/cable_calibration_preview.png)](examples/cable_calibration/reports/cable_calibration_en.html)

**[English HTML report](examples/cable_calibration/reports/cable_calibration_en.html)** · [Chinese report](examples/cable_calibration/reports/cable_calibration_zh.html)

The cable calibration was performed with **Newton VBD, using NVIDIA Warp on a CUDA GPU**. TaskTuning managed the calibration lifecycle and scored observations with PyTorch. The repository includes processed observations, 14 archived traces, stage-specific Newton adapters, portable configurations, and bilingual HTML reports.

| Stage | What changed | No-plug dynamic RMSE | Formal historical result |
|---|---|---:|---|
| A | Fit bending rigidity and damping | 3.230 mm | rejected_training |
| B | Add one shared mount-angle correction | 1.617 mm | rejected_validation |
| B* | Freeze static parameters; re-fit damping with distal weighting | 1.598 mm | rejected_validation |
| C | Replace viscous bending damping with SLS transient rigidity and relaxation time | 0.844 mm | rejected_validation |

A → B addresses a boundary hypothesis prompted by left/right residuals. It is not a controlled angle-only ablation. B* → C uses the same observations and objective; SLS reduces that objective by **76.0%**, but the plugged case still fails and remains numerically unqualified.

- [Case walkthrough and rerun instructions](examples/cable_calibration/README.md)
- [Downloadable English HTML](examples/cable_calibration/reports/cable_calibration_en.html) · [Downloadable Chinese HTML](examples/cable_calibration/reports/cable_calibration_zh.html) — download the reports or use the [local preview](docs/publishing-results.md).
- [Historical results](examples/cable_calibration/archive/) · [portable stage configurations](examples/cable_calibration/configs/)

### Which environment do I need?

| Task | Required environment |
|---|---|
| Generic synthetic demos | Newton Calibration with NumPy; no Newton |
| Re-score archived cable traces | Core dependencies on CPU; no Newton or new simulation |
| Construct or rerun cable calibration | Newton Calibration, SciPy, pinned Newton source and Warp; the recorded simulations used CUDA |
| Run stage C (SLS) | The same stack plus the bundled SLS adaptation applied to the pinned Newton base |

The historical cable environment used Python 3.12.13, Newton 1.6.0.dev0 at commit `d37f4d3d341ccce1e06a1dff21e9a054759b4855`, Warp 1.17.0, and an RTX 5070 Ti Laptop GPU. PyTorch 2.9.1+cpu handled scoring: **CPU PyTorch does not imply CPU Newton simulation**.

Installing `.[cable,cma]` does not install Newton or apply the SLS patch. Follow the [Newton environment and stage setup](examples/cable_calibration/README.md#5-construct-or-rerun-a-stage-with-newton) before using `--run`. Generic quick-start success and archive verification do not validate a new Newton environment. Raw MOV files, complete search histories and the full Newton source tree are not bundled.

## Codex skills

The five sibling directories in [codex-skills/](codex-skills/) form one coordinated configuration module:

- `standard-tuning`: intake, decisions, a shared specification and a real Tuning factory.
- `tuning-evidence`: Evidence / Evaluator / Validator.
- `tuning-environment`: Asset / Template / Rollout / Play.
- `tuning-search`: Parameters / Optimizer / Decision.
- `tuning-adapter`: minimal member adapters when existing implementations are insufficient.

See [installation and usage](docs/codex-skills.md). Skills do not install Python dependencies or create a second calibration loop.

## Development and limits

```bash
python -m pip install -e ".[cma,dev]"
python -m newton_calibration.tests
python -m examples.cable_calibration.verify
ruff check newton_calibration examples/cable_calibration/run.py examples/cable_calibration/verify.py
```

The core currently supports synchronous batches, threaded environments, optional multi-candidate environments and NumPy scoring. Thread speedup depends on the backend. It does not provide process-level forced termination, distributed multi-GPU execution, automatic resume, Bayesian optimization or PBT.

Local SVD diagnostics are not confidence intervals or proofs of global identifiability. USDA output is a trajectory visualization, not a complete physical USD asset. **Search termination, training fit and accepted release are different outcomes.**

See [capabilities and limits](docs/implementation-status.md) and [skills usage](docs/tuning-skills-implementation.md).

Reports are bundled for [local viewing](docs/publishing-results.md). See [development and validation](CONTRIBUTING.md).

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
