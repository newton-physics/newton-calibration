.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. _source-code-guidelines:

Source Code and Public API Guidelines
=====================================

This document is the canonical source code and public API guidance for Newton
Calibration contributors and reviewers. It applies prospectively to new code
and to existing code that is being changed substantially. Do not request
unrelated cleanup or break a supported API merely to make existing code
conform.

It is adapted from Newton's guidelines (``newton-physics/newton``,
``CODING_GUIDELINES.rst`` at commit ``cea666826``). The differences are the
package boundary, the module names and the dependency rules.

Its scope is limited to source-code and public-API conventions. Repository
contribution and pull-request workflows are documented in
`CONTRIBUTING.md <CONTRIBUTING.md>`__, which also covers environment setup.
General review concerns such as requirements fidelity, project fit,
reviewability, and maintenance burden are covered by the
:ref:`review-guidelines`.

In this document, *must* identifies a requirement, *should* identifies the
normal choice from which a well-justified exception may be made, and *may*
identifies an allowed choice. Breaking changes must follow the
:ref:`deprecation-policy` even when the replacement would better follow these
guidelines.

.. _deprecation-policy:

Public API compatibility and deprecation
----------------------------------------

This project has no fixed deprecation period. A pull request proposing an
incompatible public API change must describe the affected API, its replacement,
the migration steps, and the proposed transition and removal timing. Maintainers
must agree to that plan during review before the change is merged. Retain
deprecated compatibility paths until the agreed removal point, and document
the migration alongside the affected API.

Public API surface
------------------

Expose each public symbol once
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each public module must define ``__all__`` as the authoritative list of the
symbols it supports. Each public symbol should have exactly one canonical,
documented import path and should appear in the ``__all__`` of exactly one
public module. Do not re-export the same symbol from multiple public submodules.

Compatibility exports retained by the deprecation policy are the exception.
Keep such aliases out of the preferred API documentation, mark them as
deprecated, and remove them only after the required deprecation period.

Python may still allow users to reach implementation details through other
paths, but accessibility does not make those paths public or stable. The core
public API is what :mod:`newton_calibration` exports in its ``__all__``.
:mod:`newton_calibration.examples` separately exports the bundled synthetic demo
helpers, and :mod:`newton_calibration.migrate_asset` exports the asset migration
utility. Their explicit export lists do not duplicate core objects.

Implementation lives under ``newton_calibration/_src/``. Public modules are thin
re-export layers; CLI modules only forward execution to the implementation.
Names with a leading underscore, and everything under the repository's
``examples/`` directory, are internal. Examples and documentation must import
core objects through :mod:`newton_calibration`. The frozen cable backends retain
their historical imports through compatibility modules to preserve source hashes.

Update the user guide whenever a public symbol or module is added, removed,
or renamed.

Keep namespaces shallow and purposeful
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Balance a discoverable top-level namespace against excessively deep import
paths:

- Reserve the :mod:`newton_calibration` root for concepts needed in every
  calibration: the lifecycle, its members and the data records. Adding a root
  symbol requires justification that it belongs in typical user code rather
  than a specialized domain.
- Put families of related types in a public submodule, even when one member of
  the family could plausibly be exposed at the root.
- Put specialized APIs in stable domain submodules, for example a submodule
  per simulator backend or per evidence type, once more than one exists.
- Do not add nesting solely to categorize a small number of names. Additional
  levels should represent a durable boundary such as a backend, provider, or
  cohesive experimental subsystem.
- Group provider-specific integrations or implementations under an appropriate
  domain namespace rather than distributing their symbols throughout the
  library.
- Do not use a general-purpose module such as ``utils`` as a substitute for a
  clear domain boundary.

An internal implementation may be deeply organized when that improves
maintenance. These restrictions concern the public import path, not the layout
of private modules.

Keep public imports lightweight. Importing :mod:`newton_calibration` or a lightweight
public module must not eagerly initialize optional backends or load heavy,
provider-specific dependencies. Use lazy loading at optional-dependency and
backend boundaries, and account for import-time cost when assigning a symbol
to a public module.

Mark experimental public API explicitly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mark every user-facing experimental API in the public docstring or concept
page where users encounter it. The marker is part of the compatibility
contract: it must identify the exact module, type, callable, parameter, or mode
that may change without the normal deprecation process. Describe relevant
limitations alongside the marker.

Use a domain-local experimental namespace only for a cohesive new subsystem
that can reasonably live behind an opt-in import path. Do not move an existing
public type into an experimental namespace merely to describe implementation
maturity. Document the marker in the user guide next to the feature.

Naming
------

Prefer common prefixes
^^^^^^^^^^^^^^^^^^^^^^

Name related public symbols prefix-first so they cluster in autocomplete,
documentation, and search. Put the shared concept before its specialization:

- ``ViewerUSD`` and ``ViewerGL``;
- ``SolverMuJoCo`` and ``SolverXPBD``;
- ``IKObjectivePosition`` and ``IKObjectiveRotation``;
- ``add_shape_sphere()`` rather than ``add_sphere_shape()``.

Use the same principle for attributes and parameters. Prefer a stable noun
followed by its qualifier, such as ``geom_count`` rather than ``num_geoms``.
Reuse the terminology already established by the corresponding public model
instead of introducing synonyms.

Avoid overly general public names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A public name should remain understandable when it appears without its module
path in generated documentation, an error message, or search results. Include
the domain or concept family when a name could plausibly refer to several parts
of the simulation pipeline. For example, prefer ``IKObjectivePosition`` to
``PositionObjective``.

Avoid redundant prefixes for private helpers whose context is unambiguous and
which are not re-exported or documented independently.

Minimize new concepts
^^^^^^^^^^^^^^^^^^^^^

Before introducing a new public noun, determine whether an established
concept of this package or of Newton describes the same responsibility. Prefer extending or composing
existing concepts to adding vague abstractions such as ``Entity`` or
``Manager``.

When a new concept is necessary, the pull request should explain:

- why existing terminology is inaccurate;
- how the concept relates to the existing model; and
- which component owns its lifecycle and responsibilities.

Types and signatures
--------------------

Keep API families consistent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When adding to or changing a family of related public types, methods, or
functions, follow the vocabulary and signature structure already established
by that family. In particular:

- use the same parameter names for the same concepts rather than introducing
  synonyms;
- keep shared parameters in a consistent relative order and group
  operation-specific parameters predictably;
- support the family's common options when they apply, and document any
  deliberate omission; and
- use ``None`` for optional objects that must be constructed at call time,
  especially mutable or runtime-owned objects, rather than constructing them
  in the signature.

For example, a new ``ModelBuilder.add_shape_*`` method should follow the
existing shape methods' use and placement of terms such as ``body``,
``xform``, and ``cfg``, along with applicable shared options. Existing family
members are the detailed precedent; do not copy an example signature without
checking the current API.

Prefer enums for closed categories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a new public closed set of categorical values, prefer an :class:`enum.Enum`
family over a collection of module-level constants. Enums improve type safety,
group related values, and produce clearer API documentation. Use an appropriate
enum variant when integer interoperability or bitwise flags are part of the
contract.

When an integer enum includes a ``NONE`` sentinel, define ``NONE = 0`` first.
Append later real values after existing values so their integer identities
remain stable.

This preference does not apply to numerical constants, default values,
tolerances, or sentinels. It is also constrained by Warp code generation: until
Warp supports an enum type natively at a particular code-generation boundary,
keep the kernel-facing representation compatible and translate at the public
Python boundary when practical. Do not replace existing public constants
without the required deprecation.

Use keyword-only optional arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep the positional portion of a new public callable to the minimal, stable set
of primary operands. Parameters with defaults should normally be keyword-only
so new options can be added and optional parameters can be reorganized without
breaking callers.

For example:

.. code-block:: python

    def create_mesh(vertices, indices, *, normals=None, uvs=None):
        ...

Established Python idioms may justify an exception. Changing an existing
positional parameter to keyword-only is a breaking change and must use the
deprecation process.

Put reusable math at the right layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A generally useful operation on Warp-native types should normally be proposed
upstream in Warp or Newton rather than becoming a concept of this package. If
this package needs the operation before it is available upstream:

- prefer a private compatibility helper linked to the upstream issue;
- put a package-owned public math operation in a named math module, not in a
  catch-all utility namespace; and
- treat any temporary public helper as supported API, including the normal
  deprecation requirements when migrating users to Warp.

Python source conventions
-------------------------

- Follow PEP 8 for Python code.
- Prefer a nested helper class or enum when it is self-contained within one
  owning class and has no independent public meaning.
- Use PEP 604 union syntax, such as ``x | None``, instead of
  ``typing.Optional``.
- Annotate Warp arrays with bracket syntax, such as ``wp.array[wp.vec3]``,
  ``wp.array2d[float]``, and ``wp.array[Any]``. Do not use the parenthesized
  ``wp.array(dtype=...)`` form. Use ``wp.array[X]`` for one-dimensional arrays,
  not ``wp.array1d[X]``.
- Use kebab-case for command-line arguments, such as ``--use-cuda-graph``
  rather than ``--use_cuda_graph``.
- Avoid new required dependencies. The core depends on NumPy and the standard
  library only; simulators, optimizers and image libraries are optional extras
  imported inside the functions that use them.
- A pull request that introduces a required or optional dependency must
  identify its license and verify that it is compatible with this
  repository's Apache-2.0 distribution. Update license metadata and notices when required. Treat an
  unknown or undeclared dependency license as requiring review before merge.

Documentation and comments
--------------------------

- Use Google-style docstrings. Put types in annotations, not in docstrings, and
  write arguments as ``name: description`` under ``Args:``.
- Put a dataclass field's docstring on the line immediately after the field.
- Use the shortest useful Sphinx cross-reference target and prefer public API
  paths. Never reference private modules from user-facing documentation.
- State SI units for physical quantities in public API docstrings, for example
  ``Particle positions [m], shape [particle_count, 3].`` Use ``[m or rad]``
  for joint-dependent values and ``[N, N·m]`` for spatial force vectors.
  Describe compound arrays per component. Do not add units to non-physical
  fields.
- Keep inline comments brief and reserve them for non-obvious intent,
  constraints, or edge cases. Explain *why*, not what the code already states.
  Prefer an appropriate cross-reference to repeating a longer explanation.
- Before relying on or changing a documented claim, inspect its internal
  cross-references and external primary sources. Verify Newton and solver
  behavior against the current code of the pinned Newton revision. If a source is unavailable, state that
  limitation instead of assuming its support.

Tests and repository conventions
--------------------------------

- Write calibration tests as ``unittest.TestCase`` classes and run them with
  ``python -m newton_calibration.tests``.
- Give every test function or method a triple-double-quoted docstring. Begin
  with a concise, imperative summary of the behavior being verified. Add a
  Google-style body after a blank line only when the test needs more context.
- Do not call ``wp.synchronize()`` or ``wp.synchronize_device()`` immediately
  before calling ``.numpy()`` on a Warp array; ``.numpy()`` already performs a
  synchronous device-to-host copy.
- Pin GitHub Actions by commit SHA and retain the version comment in the form
  ``action@<sha>  # vX.Y.Z``. Use the hashes already allowlisted under
  ``.github/workflows`` when applicable.
- In SPDX copyright lines, use the year the file was first created. Do not use
  date ranges or update the year when modifying an existing file.

Public API review checklist
---------------------------

When reviewing a new or substantially changed public API, check that:

- each public symbol has one canonical export and the relevant ``__all__`` is
  updated;
- the namespace is shallow, purposeful, and not unnecessarily added to
  the :mod:`newton_calibration` root;
- names cluster with their concept family and remain clear out of context;
- members of an API family use consistent vocabulary, parameter grouping,
  options, and safe defaults;
- new categorical types use an enum where the execution boundary supports it;
- new public nouns are necessary and have clear ownership;
- optional parameters are keyword-only unless an established idiom justifies
  positional use;
- reusable Warp-native math is placed at the appropriate ownership layer; and
- compatibility changes follow the deprecation policy and update generated API
  documentation.

Prefer deterministic lint or test coverage for mechanically checkable rules.
Use code review for semantic and architectural judgment rather than as the only
enforcement of an invariant.
