# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import pytest

from task_tuning import Template
from task_tuning.examples import make_tuning


def fixed_tuning(tmp_path):
    tuning = make_tuning("sliding", tmp_path, optimizer="grid")
    settings = tuning.template.snapshot()
    settings.update(qualification_mode="repeatability", refinements=[])
    tuning.template = Template(**settings)
    return tuning


def test_fixed_configuration_runs_without_claiming_refinement(tmp_path):
    tuning = fixed_tuning(tmp_path)
    observed = []
    original = tuning.asset.create_environment

    def create(template, values, settings):
        observed.append(template.numeric_settings["dt"])
        return original(template, values, settings)

    tuning.asset.create_environment = create
    result = tuning.run()
    assert result.acceptance == "accepted", tuning.state.errors
    report = tuning.state.analysis_reports["numerical"]
    assert report["passed"] and report["scope"] == "fixed_configuration_only"
    assert report["mode"] == "repeatability"
    assert len(set(observed)) == 1
    assert tuning.state.validation_report["qualification_mode"] == "repeatability"
    assert len(tuning.state.validation_report["numerical_retest"]) == 1


def test_repeatability_rejects_unstable_observations(tmp_path):
    tuning = fixed_tuning(tmp_path)
    original = tuning.asset.create_environment
    count = 0

    def create(template, values, settings):
        nonlocal count
        count += 1
        environment = original(template, values, settings)
        run = environment.run
        shift = count * 100.0

        def unstable(*args, **kwargs):
            trace = run(*args, **kwargs)
            trace.values += shift
            return trace

        environment.run = unstable
        return environment

    tuning.asset.create_environment = create
    result = tuning.run()
    assert result.acceptance == "rejected_numerics"
    assert "train" not in tuning.state.budget_usage


def test_repeatability_cannot_hide_changed_settings(tmp_path):
    tuning = fixed_tuning(tmp_path)
    settings = tuning.template.snapshot()
    settings["refinements"] = [{"numeric_settings": {"dt": 0.001}}]
    with pytest.raises(ValueError, match="cannot include refinements"):
        Template(**settings).freeze()


def test_default_mode_still_requires_real_refinements(tmp_path):
    """Reject an empty refinement list instead of silently choosing repeatability."""
    tuning = make_tuning("sliding", tmp_path, optimizer="grid")
    settings = tuning.template.snapshot()
    settings["refinements"] = []
    tuning.template = Template(**settings)
    result = tuning.run()
    assert result.acceptance == "rejected_numerics"
    assert "train" not in tuning.state.budget_usage


def test_unknown_qualification_mode_is_rejected(tmp_path):
    """Reject misspelled policies before executing a simulation."""
    tuning = fixed_tuning(tmp_path)
    settings = tuning.template.snapshot()
    settings["qualification_mode"] = "repeatibility"
    with pytest.raises(ValueError, match="unknown qualification mode"):
        Template(**settings).freeze()
