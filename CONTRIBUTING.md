<!--
SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
SPDX-License-Identifier: CC-BY-4.0
-->

# Overview

Newton is a project of the Linux Foundation and aims to be governed in a transparent, accessible way for the benefit of the community. All participation in this project is open and not bound to corporate affiliation. Participants are all bound to the Linux Foundation [Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

# General Guidelines and Legal

Please refer to [the contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md) in the `newton-governance` repository for general information, project membership, and legal requirements for making contributions to Newton.

## Corporate Contributions and EasyCLA

If you contribute as part of your employment, your organization must be set up in LFX EasyCLA for Newton and complete a Corporate Contributor License Agreement (CCLA). Opening a pull request against an EasyCLA-enabled Newton repository triggers the EasyCLA check. If authorization is required, the check displays the prompt and link for starting or completing the appropriate CLA workflow.

If your organization is not listed, the EasyCLA workflow lets you add it and identify the representative authorized to coordinate the CCLA signing process. See [Contributor License Agreements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#contributor-license-agreements) for the full process.

# Contributing to Newton Calibration

Newton Calibration provides the `newton_calibration` package and `newton-calibration` CLI. In order to avoid any surprises and to increase the chance of contributions being merged, we encourage contributors to communicate their plans proactively by opening a GitHub issue. Keep each pull request focused on a single feature or bug fix, and write clear, concise commit messages.

Use [AGENTS.md](AGENTS.md) for environment setup and development workflows. Code and public API changes must follow the [source code and public API guidelines](CODING_GUIDELINES.rst). Reviewers may use the suggested [review guidelines](REVIEW_GUIDELINES.rst).

There are several ways to participate:

## Questions, Discussions, Suggestions

* Use [GitHub Issues](https://github.com/newton-physics/newton-calibration/issues) for questions, bug reports and feature proposals.

## Reporting a Bug

* Check in the [GitHub Issues](https://github.com/newton-physics/newton-calibration/issues) if a report for the bug already exists.
* If the bug has not been reported yet, open a new Issue.
* Use a short, descriptive title and write a clear description of the bug.
* Document the Git hash or release version where the bug is present, the Newton revision in use, and the hardware and environment by including the output of `nvidia-smi`.
* Add executable code samples or test cases with instructions for reproducing the bug.

## Documentation Issues

* Create a new issue if there is no existing report, or
* directly submit a fix following the "Fixing a Bug" workflow below.

## Fixing a Bug

* Ensure that the bug report issue has no assignee yet. If the issue is assigned and there is no linked PR, you're welcome to ask about the current status by commenting on the issue.
* Write a fix and regression unit test for the bug following the [source code guidelines](CODING_GUIDELINES.rst).
* Open a new pull request for the fix and test.
* Write a description of the bug and the fix.
* Mention related issues in the description: E.g. if the patch fixes Issue \#33, write Fixes \#33.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the codebase.

## Improving Performance

* Write an optimization and measure it on a case in `examples/` or on a synthetic demo, following the [source code guidelines](CODING_GUIDELINES.rst).
* Open a new pull request with the optimization and the measurement.
* Write a description of the performance optimization.
* Mention related issues in the description: E.g. if the optimization addresses Issue \#42, write Addresses \#42.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the codebase.

## Adding a Feature

* Discuss your proposal ideally before starting with implementation. Open a GitHub Issue or Discussion to:
  * propose and motivate the new feature;
  * detail technical specifications;
  * and list changes or additions to the public `newton_calibration` API.
* Wait for feedback from [Project Members](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) before proceeding.
* Implement the feature following the [source code guidelines](CODING_GUIDELINES.rst).
* Add comprehensive testing for the new feature.
* Ensure all existing tests pass and that the archived cable scoring (`python -m examples.cable_calibration.verify`) still reproduces its recorded metrics.
* Update or add documentation for the new feature.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the codebase.

## Adding a Framework Member

The calibration lifecycle is `Tuning.run()` composed of its members: `Evidence`, `Asset`, `Evaluator`, `Parameters`, `Rollout`, `Optimizer`, `Decision`, `Validator`, `Template` and `Play`. Case-specific behavior goes into members, not into a second loop.

A new member implementation or a new public hook on a member is a candidate for the core package when:

* it applies to more than one case or simulator rather than to one asset or experiment;
* it cannot be expressed by composing or subclassing existing members;
* it has clear long-term maintainers within the project.

Do not add a second search, stopping, validation or recording loop; `Decision` alone stops work, and `Tuning` owns all writes to the run state. Open a GitHub Issue or Discussion and wait for feedback from [Project Members](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) before changing the public package.

## Adding a Calibration Case

Cases live under `examples/`, one directory each, and use the public `newton_calibration` API only. A case consists of:

* a backend that builds the scene and implements `Asset` and `Environment`, with parameter readback from the model;
* a data loader that turns the recordings into `Experiment`s and experiment settings from one source;
* a configuration with parameters, bounds, fixed values, numerical settings, refinements and budgets;
* a script that generates the recordings, or a documented way to obtain them, with their content hashes (see "Assets and Recordings" below);
* a `README.md` that states the physical setup, units, the Newton revision, the hardware the case was run on, and what was and was not validated.

Use public Newton APIs; do not import from `newton._src`. The one existing exception is the archived cable stage C, which binds to a patched solver; see the [cable example README](examples/cable_calibration/README.md#5-construct-or-rerun-a-stage-with-newton). Do not silently replace a pinned solver revision or patch with a released package; record the change and rerun the affected checks.

Measured data, numerical settings, archived scores and acceptance statuses are evidence. Preserve them when reorganizing a case, and never reinterpret a historical rejection as acceptance.

## Assets and Recordings

Git holds code, configurations and documentation. It does not hold recorded or rendered data: frames, masks, videos, trajectories, fitted results and the meshes and robot descriptions a case consumes. Such files bloat the history permanently, and every regeneration adds another copy.

* Generic simulation assets, such as robot URDFs and meshes, belong in the [newton-assets](https://github.com/newton-physics/newton-assets) repository and are downloaded at run time.
* Recordings and results are kept outside git and shared through a case-specific access-controlled location. Document access instructions, file sizes and hashes alongside the case. A case records the content hash of every file it depends on, so a synced copy can be checked.
* A small, deliberately selected set of example recordings may be committed when a fresh checkout or CI needs them to run a case end to end. Choose the smallest set that serves that purpose, prefer regenerable synthetic data, and state in the pull request why each file is needed.

Before proposing to add project-maintained assets to `newton-assets`:

* Make sure that the assets are properly licensed for use and distribution. If you are unsure about the license, open a new discussion.
* If a pull request in this repository relies on new assets, open a corresponding pull request in the [newton-assets](https://github.com/newton-physics/newton-assets) repository.
* Follow the instructions in the [README](https://github.com/newton-physics/newton-assets) of the `newton-assets` repository.
* Have a signed CLA on file (see [Legal Requirements](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#legal-requirements)).
* Have the `newton-assets` pull request approved by a [Project Member](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md#project-members) and merged into the asset repository.

## Licensing

Code and bundled software example fixtures, including archived reports, use
[Apache-2.0](LICENSE); project documentation uses [CC-BY-4.0](LICENSE-DOCS).
CI checks SPDX headers and accepts both `The Newton Developers` and `Lightwheel`
as copyright owners. Preserve existing notices and example/solver provenance
(see [NOTICE](NOTICE)), and identify the license of every new dependency;
dependencies retain their own licenses.
