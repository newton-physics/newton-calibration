# Codex skills: installation and workflow

[Home](../README.md) · [Skill sources](../codex-skills/) · [Design](tuning-skills-module-design.md)

The five skill directories are a coordinated configuration module for TaskTuning. They are text instructions, not a Python dependency, MCP server or second runtime.

## Install the complete group

Install the Python project first:

```bash
python -m pip install -e ".[cma]"
```

From the repository root, copy all five directories into your existing Codex skills location. This example uses the same `~/.codex/skills` layout as development; change the destination if your installation uses a different location.

```python
from pathlib import Path
from shutil import copytree

source = Path("codex-skills")
destination = Path.home() / ".codex" / "skills"
names = (
    "standard-tuning",
    "tuning-evidence",
    "tuning-environment",
    "tuning-search",
    "tuning-adapter",
)
existing = [name for name in names if (destination / name).exists()]
if existing:
    raise SystemExit(f"Already installed; review/update these explicitly: {existing}")
destination.mkdir(parents=True, exist_ok=True)
for name in names:
    copytree(source / name, destination / name)
```

The snippet intentionally does not overwrite existing skills. Keep the five folders as siblings because their links are relative. Confirm that your Codex session lists them before invoking them; installation does not grant access to an unavailable interpreter, solver or dataset.

## Which skill owns what?

| Skill | Scope |
|---|---|
| standard-tuning | Shared intake, decisions, one TUNING-SPEC, real factory and execution handoff |
| tuning-evidence | Observation definitions, loss/masks/scales, data groups, acceptance |
| tuning-environment | Physical boundaries, numerics, binding, parallel rollout and playback |
| tuning-search | Parameter selection/bounds, identifiability, optimizer and budgets |
| tuning-adapter | Minimal adapter only when existing members cannot express the requirement |

[standard-tuning](../codex-skills/standard-tuning/SKILL.md) routes only the relevant unresolved questions. It does not require multiple agents.

## Example prompts

Create a configuration without starting a simulation:

```text
$standard-tuning
Use this asset specification and measured dataset to create a fresh Tuning factory.
Use my selected Python interpreter. Inspect the data first and ask about unresolved
experimental conditions. Deliver TUNING-SPEC.md and build_tuning(workspace=...).
Construct and inspect the instance, but do not call prepare() or run().
```

Execute an explicitly scoped calibration:

```text
$standard-tuning
Run the configured calibration with a 30-minute total simulation budget.
Include numerical qualification and final validation in that budget.
Keep all old runs and report acceptance separately from search termination.
```

Revise the evidence contract:

```text
$tuning-evidence
Review the masks and train/validation acquisition groups for this release video.
Propose a distal-motion metric without silently changing the existing thresholds.
```

## What the module should deliver

- One case specification recording facts, decisions, assumptions and provenance.
- A factory returning a real, fresh Tuning instance.
- Explicit member classes, parameter/fixed-setting split and observation shapes.
- Solver/interpreter paths resolved in the user's environment, with no developer-machine fallback.
- No simulation during module import or factory construction.
- A new run only when execution is in scope; no replacement of Tuning.run().
- A report of actual termination, retest, validation and release status.

In the cable example, the critical user decisions included the 17 cm length correction, unchanged clamp during board reversal, fitting a shared angle, using 32-segment equivalent parameters, and later testing SLS. Numerical accuracy and actual parameter binding were checked in implementation, not assumed from an old template.

See the [acceptance scenarios](../codex-skills/standard-tuning/references/acceptance-scenarios.md). See [usage and validation scope](tuning-skills-implementation.md); format checks are not benchmarks of automatic skill-selection accuracy.
