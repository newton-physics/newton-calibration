# Tuning skills usage and validation

The five source directories are distributed in [codex-skills](../codex-skills/). Install them as siblings using the [installation guide](codex-skills.md); the repository does not assume they are already installed on another machine.

| Skill | Responsibility |
|---|---|
| standard-tuning | Shared decisions, specification, real object assembly, handoff |
| tuning-evidence | Evidence / Evaluator / Validator |
| tuning-environment | Asset / Template / Rollout / Play |
| tuning-search | Parameters / Optimizer / Decision |
| tuning-adapter | Minimal missing member behavior and contract checks |

The entry skill includes intake rules, interface checkpoints, a TUNING-SPEC.md template, and behavioral acceptance scenarios. Each skill includes agents/openai.yaml and permits implicit invocation. All cross-skill links assume sibling directories.

No scheduler, configuration compiler, plugin background service, or alternative runtime was added. Factories are case-specific and constructed against current interfaces; there is no global factory containing cable defaults.

## Usage

```text
Use $standard-tuning to create a real Tuning instance from the available project information.
```

```text
Use $standard-tuning to calibrate this asset against the measured data.
```

Use a child skill for a narrower configuration request. Descriptions guide selection; they are not deterministic keyword hooks. If newly installed skills are absent from a session, start a new session and invoke one explicitly.

## Validation scope

Use the [acceptance scenarios](../codex-skills/standard-tuning/references/acceptance-scenarios.md) to check construction boundaries, fresh run identities, lifecycle execution, and reporting. Format checks and a successful synthetic demo do not establish dialogue-selection accuracy or cable acceptance. Changes to repository skills do not automatically update installed copies.
