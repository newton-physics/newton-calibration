# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise cable adapters with the optional Newton runtime and SLS patch."""

import importlib
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

from newton_calibration import Experiment, Template

CASE = Path(__file__).resolve().parents[2] / "examples" / "cable_calibration"
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("newton", "scipy"))


@unittest.skipUnless(HAS_RUNTIME, "install the newton and cable extras")
class TestNewtonCable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import newton

        builder = newton.ModelBuilder()
        newton.solvers.SolverVBD.register_custom_attributes(builder)
        cls.has_sls = "vbd:visco_bend_ke" in builder.custom_attributes

    def _case(self, stage):
        config = json.loads((CASE / "configs" / f"{stage}.json").read_text())
        # Short CPU checks exercise construction and stepping, not qualification.
        config["template"]["numeric_settings"].update(device="cpu", iterations=2, dynamic_iterations=2, block_steps=2)
        config["template"]["discretization"]["segments"] = 4
        template = Template(**config["template"])
        parameters = {entry["name"]: entry["baseline"] for entry in config["parameters"]}
        package = "examples.cable_calibration.backends." + ("b" if stage == "Bstar" else stage.lower())
        backend = importlib.import_module(package + (".sls_backend" if stage == "C" else ".newton_backend"))
        asset_type = backend.SLSCableAsset if stage == "C" else backend.CableAsset
        asset = asset_type(parameters, config=config["asset_config"])
        rollout = importlib.import_module(package + ".rollout").NewtonRollout
        return asset, template, parameters, rollout

    def _check_bindings(self, stage):
        asset, template, parameters, _ = self._case(stage)
        expected = {**template.fixed_parameters, **parameters}
        segments = template.discretization["segments"]
        h = expected["free_length"] / segments
        # The fixed-to-free joint spans half a segment; interior joints span h.
        spans = np.array([h / 2] + [h] * (segments - 1))
        for experiment_id in ("noplug_left", "noplug_release", "withplug_left", "withplug_release"):
            with self.subTest(stage=stage, experiment=experiment_id):
                env = asset.create_environment(template, expected, template.experiment_settings[experiment_id])
                try:
                    for name, value in expected.items():
                        self.assertAlmostEqual(env.applied_parameters[name], value, delta=max(abs(value) * 2e-6, 2e-5))
                    joints = env.joints[:segments]
                    slots = env.model.joint_qd_start.numpy()[joints]
                    coefficients = env.model.joint_target_ke.numpy()
                    for axis in (3, 4):
                        np.testing.assert_allclose(
                            coefficients[slots + axis], expected["bending_rigidity"] / spans, rtol=2e-6, atol=1e-8
                        )
                    np.testing.assert_allclose(
                        coefficients[slots + 2], expected["stretch_stiffness"] * (expected["free_length"] / 32) / h
                    )
                    if stage == "C":
                        np.testing.assert_allclose(
                            env.model.vbd.visco_bend_ke.numpy()[joints],
                            expected["sls_transient_rigidity"] / spans,
                            rtol=2e-6,
                            atol=1e-8,
                        )
                        np.testing.assert_allclose(
                            env.model.vbd.visco_bend_tau.numpy()[joints], expected["sls_relaxation_time"], rtol=2e-6
                        )
                        self.assertTrue(env.solver.enable_viscoelasticity)
                finally:
                    env.close()

    def _check_batch(self, stage):
        asset, template, parameters, rollout = self._case(stage)
        # Allow the response to separate beyond the serial/packed comparison tolerance.
        times = np.array([0.0, 8.0, 16.0]) * template.numeric_settings["dt"]
        experiment = Experiment("noplug_release", "release", np.zeros((3, 30)), times=times)
        name = {
            "A": "bending_rigidity",
            "B": "bending_rigidity",
            "Bstar": "bend_damping",
            "C": "sls_transient_rigidity",
        }[stage]
        perturbed = {**parameters, name: parameters[name] * 1.2}
        trials = [("baseline", parameters), ("perturbed", perturbed)]
        serial = rollout(capacity=1).execute_batch(template, asset, trials, [experiment])
        for episode in serial:
            self.assertEqual(episode.status, "completed", episode.error)
        self.assertGreater(
            np.max(np.abs(serial[1].trace.values - serial[0].trace.values)),
            1e-5,  # 10 micrometres, above the 2-micrometre comparison tolerance.
            f"{stage}: perturbing {name} must change the trajectory",
        )
        packed = rollout(capacity=2).execute_batch(template, asset, trials, [experiment])
        for expected, actual in zip(serial, packed, strict=True):
            with self.subTest(stage=stage, trial=expected.trial_id):
                self.assertEqual(actual.status, "completed", actual.error)
                self.assertEqual(actual.key, expected.key)
                np.testing.assert_array_equal(actual.trace.times, times)
                np.testing.assert_allclose(actual.trace.values, expected.trace.values, atol=2e-6, rtol=2e-6)
                self.assertTrue(np.isfinite(actual.trace.positions).all())
                for name, value in expected.applied_parameters.items():
                    self.assertAlmostEqual(actual.applied_parameters[name], value, delta=max(abs(value) * 2e-6, 2e-5))

    def test_vbd_bindings(self):
        """Read physical coefficients from the six-DOF rods, including the clamp and plug."""
        for stage in ("A", "B", "Bstar"):
            self._check_bindings(stage)

    def test_vbd_batch_matches_serial(self):
        """Match independent and packed dynamic traces for two distinct candidates."""
        for stage in ("A", "B", "Bstar"):
            self._check_batch(stage)

    def test_sls_bindings_and_batch(self):
        """Check SLS coefficients and candidate isolation in the packed adapter."""
        if not self.has_sls:
            self.skipTest("apply the cable SLS patch to Newton")
        self._check_bindings("C")
        self._check_batch("C")

    def test_sls_prehistory_matches_expected_moment(self):
        """Initialize the signed SLS moment from a known bend ramp followed by a hold."""
        if not self.has_sls:
            self.skipTest("apply the cable SLS patch to Newton")
        asset, template, parameters, _ = self._case("C")
        parameters = {**template.fixed_parameters, **parameters}
        segments = template.discretization["segments"]
        length = parameters["free_length"]
        h = length / segments
        tau, ramp_duration, hold_duration = 0.5, 0.25, 0.5
        parameters.update(sls_transient_rigidity=h, sls_relaxation_time=tau, mount_clamp_angle_offset_deg=0.0)
        settings = dict(template.experiment_settings["noplug_release"])
        for name in ("rest_centerline", "rest_arclengths", "initial_centerline_velocity", "clamp_angle_parameter"):
            settings.pop(name, None)
        arclengths = np.linspace(0.0, length, segments + 1)
        theta = 0.1
        straight = np.outer(arclengths, [1.0, 0.0, 0.0]).tolist()
        bent = np.outer(arclengths, [np.cos(theta), 0.0, np.sin(theta)]).tolist()
        settings.update(
            clamp_angle=0.0,
            initial_arclengths=arclengths.tolist(),
            initial_centerline=bent,
            sls_history_initial_stress_factor=0.0,
            sls_history=[
                {"time_s": 0.0, "centerline": straight, "valid": True},
                {"time_s": ramp_duration, "centerline": bent, "valid": True},
                {"time_s": ramp_duration + hold_duration, "centerline": bent, "valid": True},
            ],
        )
        # All free segments rotate together: only the clamp bends, about local -Y.
        expected_kappa = np.zeros((segments, 3))
        expected_kappa[0, 1] = -2.0 * np.tan(theta / 2.0)
        clamp_stiffness = 2.0 * parameters["sls_transient_rigidity"] / h
        # Exact Maxwell response to a linear strain ramp, then exponential relaxation.
        expected_moment = (
            clamp_stiffness
            * expected_kappa
            * tau
            / ramp_duration
            * (1.0 - np.exp(-ramp_duration / tau))
            * np.exp(-hold_duration / tau)
        )
        env = asset.create_environment(template, parameters, settings)
        try:
            joints = env.joints[:segments]
            np.testing.assert_allclose(
                env.solver.joint_visco_moment_prev.numpy()[joints], expected_moment, rtol=2e-5, atol=1e-7
            )
            np.testing.assert_allclose(
                env.solver.joint_visco_kappa_prev.numpy()[joints], expected_kappa, rtol=2e-5, atol=1e-7
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
