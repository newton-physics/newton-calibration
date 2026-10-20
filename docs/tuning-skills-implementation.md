# Tuning skills delivery and validation

Original implementation record: 2026-10-12. Repository English documentation update: 2026-10-14.

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

## Historical checks

- All five skills passed skill-creator quick_validate.py.
- UI metadata, 25–64 character descriptions, prompts, invocation settings, and relative links were checked.
- A temporary non-cable factory reused a fully specified sliding case. During import/construction, prepare(), run(), and SlidingEnvironment.run() were replaced by fail-on-call guards. Construction returned a real Tuning, ten members, and RunState without creating workspace. Two constructions produced distinct run identities.
- A fresh instance then completed the original Tuning.run() lifecycle, recovered friction=0.35 with rank 1, passed export/reload validation, generated USD, and returned accepted.
- Installation checked for existing destinations before copying; the original 14 installed files matched source hashes.

Temporary scripts and workspace artifacts were not retained as runtime dependencies. These checks established synthetic construction/execution boundaries in the original session; they were not independent agent evaluations of all dialogue behavior or measurements of natural-language trigger accuracy.

The repository copies are now English. This documentation update does not modify skills previously installed on the development machine. Remaining scenarios are listed in the [acceptance table](../codex-skills/standard-tuning/references/acceptance-scenarios.md). Do not infer cable acceptance from skill format checks or a successful synthetic demo.
