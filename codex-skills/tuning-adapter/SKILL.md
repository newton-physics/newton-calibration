---
name: tuning-adapter
description: Implement minimal task_tuning member subclasses or solver adapters when existing members cannot express an already specified calibration requirement. Cover assets, environments, batching, scoring, optimization, validation, or presentation contracts. Do not use for constructor-only changes or replace Tuning.run() with another calibration loop.
---

# Tuning Adapter

Read the shared specification, current base classes, and callers. Prefer constructor configuration, then a minimal case-specific subclass, then a shared change only for a demonstrated common gap. State the unsupported behavior, affected methods, and tests before editing. Avoid an unnecessary registry or framework.

Route unresolved semantics to tuning-evidence, tuning-environment, or tuning-search.

| Member | Contract to preserve |
|---|---|
| Asset / Environment | Fresh state, real binding/readback, physical time, close, export/reload, independent physical benchmarks |
| Rollout | Stable identities/order, A/B/A isolation, single/batch equivalence, deadlines and failure records |
| Evaluator | Shapes, nonfinite/failed inputs, metric_spec/fingerprint, analysis features and consistent validation |
| Optimizer | Pending complete batches, finite feedback protocol, required state, factual diagnostics only |
| Validator | Frozen candidate, no holdout leakage into training, explicit acceptance |
| Play | Saved trajectories, consistent times/coordinates, no search-state changes |
| Other members | Existing ownership and lifecycle invariants |

When changing loss semantics, update features, numerical tolerances, fingerprints, and validation together. Applied-parameter readback must come from the model, not supplied inputs. Persistent population algorithms need explicit state contracts.

Never override Tuning.run() to create a second loop or maintain a second RunState. Tests should include independently expected values, failures, boundary cases, and physical qualification where applicable; merely completing a run is insufficient.

Return importable actual types, constructor guidance, tests, remaining limitations, and the shared specification update. Adapter implementation success is separate from calibration acceptance.
