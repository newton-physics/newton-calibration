# Newton integration status

[Home](../README.md) · [Contributing](../CONTRIBUTING.md) · [Cable example](../examples/cable_calibration/README.md)

## Initial seed

The initial framework is contributed from Lightwheel's TaskTuning. It preserves
`task_tuning` imports, the `task-tuning` command, ten composable members, synchronous
batch evaluation, ask/tell optimization, numerical qualification, identifiability,
export/reload validation, reporting and the five configuration skills.

The historical cable example retains its four stages, processed evidence, failed
acceptance outcomes, backend snapshots and original reports. Generic CPU examples
and historical trace scoring work without Newton. The source remains auditable
through the existing provenance records.

`refinement` remains the default numerical qualification policy.
The explicit `repeatability` policy checks fixed settings and seeds only; it is
recorded in reports and does not claim convergence. Existing cable stage settings
are not silently changed to this policy.

## Temporary solver dependency

The cable adapters use Newton base
`d37f4d3d341ccce1e06a1dff21e9a054759b4855`.
Stage C additionally applies the recorded `solver/sls-adapted.patch`.
The factory checks the three solver-file hashes and actual import location before
constructing a tuning. These checks remain enforced.

[Newton PR #4825](https://github.com/newton-physics/newton/pull/4825) proposes SLS bending
on a newer upstream base. It is a separate contribution, not a replacement for the
recorded cable environment. Its status and eventual merge commit must be checked
when switching dependencies.

The temporary private dependency is isolated in
[`backends/c/sls_backend.py`](../examples/cable_calibration/backends/c/sls_backend.py):

- `compute_geometric_rod_kappa_cached_z` supplies the solver's own bending strain.
- Solver moment/strain arrays receive the measured pre-release material history.
- `_rigid_pose_rebaseline_mask` is cleared so the first step preserves that history.

The constitutive-model PR does not yet introduce an arbitrary pre-stressed history
initialization API. Its merge alone therefore does not justify deleting these bindings.
The current adapter is intentionally retained until public interfaces cover the same
behavior. The core package has no private Newton imports.

## Subsequent contributions

1. Agree with Jimmy on a batched evaluation interface around `Rollout` and structured
   `BatchResult`: per-experiment metrics, validity, constraints and diagnostics.
2. Integrate the upstream GPU-batched implementation without replacing the lifecycle
   or changing optimizer/Decision ownership.
3. Establish a public solver-history initialization path, then migrate the cable adapter
   in a separate change. Compare parameter readback, strain/history initialization,
   trajectories and numerical qualification before recording a new supported revision.
4. Consider video as a raw evidence source with explicit observation extraction,
   timestamps, calibration, uncertainty and split provenance. The current seed does not
   implement direct ego-video ingestion or a new extraction API.

These are integration topics, not implemented features or committed upstream APIs.
