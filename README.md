<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Newton Calibration

Newton Calibration is a toolbox for fitting simulation parameters to measured data. It supplies components, for example `Evidence`, `Parameters`, `Asset`, `Evaluator`, `Optimizer` and `Validator`. You select, configure and extend these components to build a calibration workflow for your own system.

The goal is a collection of recipes: complete calibration workflows for typical problems that you can adapt to your own case.

> **Pre-release.** This package has not had a release yet. The API, file formats and command line can change without notice.

Start with a Newton XPBD box friction example, then connect your own measurements and model. The cable study shows how static and dynamic experiments guide parameter estimation and model refinement.

| Guide | Use it to |
|---|---|
| [Quick start](docs/quickstart.md) | Run a Newton calibration example, inspect the outputs, and connect your own observations and model |
| [Calibration workflow](docs/calibration-guide.md) | Define the problem, design experiments, and investigate calibration failures |
| [API reference](docs/api-reference.md) | Look up member arguments, interfaces, and contracts |
| [Cable case study](examples/cable_calibration/README.md) | Follow the measured cable case, inspect its results, and reproduce the workflow |

## Quickstart

Fit the friction of a sliding box with Newton's `SolverXPBD`. This example runs on CPU and compares simulated motion with analytical reference trajectories whose friction is known.

The core package requires Python 3.11+ and NumPy. From a repository checkout, activate a Python environment and install the Newton example dependencies.

```bash
python -m pip install -e ".[newton]"
python -m examples.box_friction_calibration.run --workspace workspace/newton-sliding
```

The `newton` extra selects the tested Newton 1.9.0.dev0 revision and Warp 1.17.0 from `pyproject.toml`. The first run compiles Warp kernels. See the [setup and scope](examples/box_friction_calibration/README.md) for environment details.

Expect `accepted` and a friction coefficient close to `0.35`. The command prints the run directory and accepted asset path. Open `result.json` for the outcome and `run_state.json` for the analysis and validation reports.

Continue with the **[step-by-step quickstart](docs/quickstart.md)** to inspect the saved run, assemble the calibration yourself, replace the simulator adapter, or switch from a single environment to multiple environments.

## Examples

| Example | What it demonstrates | Start here |
|---|---|---|
| Sliding box · Newton XPBD | Fit one physical parameter against analytical evidence, check numerical accuracy, and validate at a held-out initial speed | [Run and inspect](docs/quickstart.md) · [Source](examples/box_friction_calibration/) |
| Measured cable · Newton VBD | Use static shapes and release motion to fit parameters, interpret residuals, and refine the model | [Calibration workflow](docs/calibration-guide.md) · [Case study](examples/cable_calibration/README.md) |

### Cable release experiment

[![No-plug cable release: measured vertical motion compared with viscous and SLS models, with coordinate RMSE of 1.598 mm and 0.844 mm respectively](examples/cable_calibration/reports/cable_calibration_preview.png)](examples/cable_calibration/README.md)

On the measured release of a **17 cm free cable without a plug**, the viscous model (B*) gives a coordinate RMSE of **1.598 mm** and the standard linear solid model (C) gives **0.844 mm**. Both models were fitted to the same recording. Re-scoring their archived predictions over the same 28 frames and 30 x/z coordinates reproduces a **76.0% reduction in the fitting objective**.

[Reproduce the scores on CPU](examples/cable_calibration/README.md#1-verify-the-reported-results-without-simulation) · [English report](examples/cable_calibration/reports/cable_calibration_en.html) · [Chinese report](examples/cable_calibration/reports/cable_calibration_zh.html) · [Full run records](examples/cable_calibration/archive/)

For lightweight sliding and oscillator demos using NumPy, see the [user guide](docs/user-guide.md).

## How it works

A `Tuning` instance brings together the evidence, simulator adapter, physical parameters, and search and validation rules:

```text
Define evidence and parameters → check baseline and numerical accuracy
→ search physical parameters with fixed numerical settings
→ reload the candidate → validate and recheck numerical accuracy
→ release an asset if accepted
```

The adapter owns model construction, parameter application, simulation, and observations. The calibration members handle execution, scoring, search, and validation through one `Tuning.run()` lifecycle.

- **[Calibration workflow](docs/calibration-guide.md):** choose the goal, design experiments, and decide what the evidence supports.
- **[Quickstart](docs/quickstart.md):** connect observations and a simulator, then run the workflow.
- **[API reference](docs/api-reference.md):** configure members, implement adapters, and read result files.
- **[Capabilities and limits](docs/implementation-status.md):** check supported execution modes and the scope of diagnostics.

## Development

```bash
python -m pip install -e ".[cma,dev]"
python -m newton_calibration.tests
python -m examples.cable_calibration.verify
uvx pre-commit run -a
```

See [Contributing](CONTRIBUTING.md) for development and validation guidelines, and [Codex skills](docs/codex-skills.md) for configuring calibration tasks with the bundled skills.

[Class diagram](docs/tuning-class-diagram.md) · [Workflow design](docs/generic-calibration-workflow-design.md) · [Repository instructions](AGENTS.md)

## License

Source code uses Apache-2.0 and project documentation uses CC-BY-4.0. See the license texts and attribution notice. [Apache-2.0](LICENSE) · [CC-BY-4.0](LICENSE-DOCS) · [NOTICE](NOTICE)
