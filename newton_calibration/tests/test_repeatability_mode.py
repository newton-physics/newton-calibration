# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton_calibration import Template
from newton_calibration.examples import make_tuning
from newton_calibration.tests.unittest_utils import TempPathTestCase


def fixed_tuning(tmp_path):
    tuning = make_tuning("sliding", tmp_path, optimizer="grid")
    settings = tuning.template.snapshot()
    settings.update(qualification_mode="repeatability", refinements=[])
    tuning.template = Template(**settings)
    return tuning


class TestRepeatabilityMode(TempPathTestCase):
def test_fixed_configuration_runs_without_claiming_refinement(self):
    tuning = fixed_tuning(self.tmp_path)
    observed = []
    original = tuning.asset.create_environment

    def create(template, values, settings):
        observed.append(template.numeric_settings["dt"])
        return original(template, values, settings)

    tuning.asset.create_environment = create
    result = tuning.run()
    self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
    report = tuning.state.analysis_reports["numerical"]
    self.assertTrue(report["passed"])
    self.assertEqual(report["scope"], "fixed_configuration_only")
    self.assertEqual(report["mode"], "repeatability")
    self.assertEqual(len(set(observed)), 1)
    self.assertEqual(tuning.state.validation_report["qualification_mode"], "repeatability")
    self.assertEqual(len(tuning.state.validation_report["numerical_retest"]), 1)

def test_repeatability_rejects_unstable_observations(self):
    tuning = fixed_tuning(self.tmp_path)
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
    self.assertEqual(result.acceptance, "rejected_numerics")
    self.assertNotIn("train", tuning.state.budget_usage)

def test_repeatability_cannot_hide_changed_settings(self):
    tuning = fixed_tuning(self.tmp_path)
    settings = tuning.template.snapshot()
    settings["refinements"] = [{"numeric_settings": {"dt": 0.001}}]
    with self.assertRaisesRegex(ValueError, "cannot include refinements"):
        Template(**settings).freeze()

def test_default_mode_still_requires_real_refinements(self):
    """Reject an empty refinement list instead of silently choosing repeatability."""
    tuning = make_tuning("sliding", self.tmp_path, optimizer="grid")
    settings = tuning.template.snapshot()
    settings["refinements"] = []
    tuning.template = Template(**settings)
    result = tuning.run()
    self.assertEqual(result.acceptance, "rejected_numerics")
    self.assertNotIn("train", tuning.state.budget_usage)

def test_unknown_qualification_mode_is_rejected(self):
    """Reject misspelled policies before executing a simulation."""
    tuning = fixed_tuning(self.tmp_path)
    settings = tuning.template.snapshot()
    settings["qualification_mode"] = "repeatibility"
    with self.assertRaisesRegex(ValueError, "unknown qualification mode"):
        Template(**settings).freeze()


if __name__ == "__main__":
    unittest.main(verbosity=2)
