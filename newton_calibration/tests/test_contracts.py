# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from newton_calibration import Evidence, Experiment, Parameter, Parameters, Template


class TestContracts(unittest.TestCase):
def test_parameter_invalid_bounds_and_log_domain(self):
    with self.assertRaises(ValueError):
        Parameter("k", (3.0, 1.0), 2.0)
    with self.assertRaises(ValueError):
        Parameter("k", (-1.0, 2.0), 1.0, transform="log")

def test_rotated_parameters_roundtrip_and_no_silent_clipping(self):
    parameters = Parameters([Parameter("a", (0, 10), 5), Parameter("b", (0, 2), 1)])
    basis = np.array([[1.0, 1.0], [1.0, -1.0]]) / 2**0.5
    parameters.set_mapping(np.array([0.5, 0.5]), basis)
    value = np.array([[7.0, 1.2]])
    np.testing.assert_allclose(parameters.decode(parameters.encode(value)), value)
    self.assertFalse(parameters.check_bounds(parameters.decode(np.array([[1.0, 1.0]]))).all())
    parameters.freeze()
    with self.assertRaises(RuntimeError):
        parameters.set_mapping(np.zeros(2), np.eye(2))

def test_template_freezes_nested_settings_and_separates_parameters(self):
    template = Template(
        numeric_settings={"dt": 0.01, "iterations": 4},
        discretization={"segments": [1, 2]},
        fixed_parameters={"mass": 1.0},
        experiment_settings={"train": {"seed": 1}},
    )
    template.freeze()
    with self.assertRaises(TypeError):
        template.numeric_settings["dt"] = 0.02
    with self.assertRaises((TypeError, AttributeError)):
        template.discretization["segments"].append(3)
    with self.assertRaises(AttributeError):
        template.fixed_parameters = {}
    self.assertTrue(template.template_id)

def test_evidence_rejects_same_acquisition_group_across_splits(self):
    a = Experiment("a", "same_capture", np.array([[1.0]]))
    b = Experiment("b", "same_capture", np.array([[2.0]]))
    with self.assertRaisesRegex(ValueError, "group"):
        Evidence([a], [b]).prepare()

def test_evidence_rejects_missing_valid_observations(self):
    bad = Experiment("bad", "g", np.array([[np.nan]]))
    with self.assertRaises(ValueError):
        Evidence([bad]).prepare()

def test_parameter_freeze_does_not_expose_mutable_bounds_or_mapping(self):
    parameters = Parameters([Parameter("x", (1.0, 3.0), 2.0)])
    parameters.freeze()
    fingerprint = parameters.parameter_space_id
    parameters.bounds[...] = 0
    parameters.baseline[...] = 0
    parameters.snapshot()["basis"][...] = 0
    self.assertEqual(parameters.parameter_space_id, fingerprint)
    self.assertEqual(float(parameters.bounds[0, 0]), 1.0)
    self.assertEqual(float(parameters.baseline[0]), 2.0)
    with self.assertRaisesRegex(AttributeError, "frozen"):
        parameters.names = ("replacement",)


if __name__ == "__main__":
    unittest.main(verbosity=2)
