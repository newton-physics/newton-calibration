# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check robot optimizer configuration before any calibration search."""

import importlib.util
import unittest

import numpy as np

from examples.actuator.robot.asset import AssetRobot
from examples.actuator.robot.reference import synthetic_config, synthetic_records
from examples.actuator.robot.tuning import TuningRobot
from newton_calibration.tests.test_robot_calibration import BACKEND, FIXTURE
from newton_calibration.tests.unittest_utils import TempPathTestCase


@unittest.skipUnless(BACKEND and importlib.util.find_spec("cma"), "install the robot and cma extras")
class TestRobotOptimizer(TempPathTestCase):
    def test_sigma_changes_actual_sampling_scale(self):
        """Scale matched CMA proposals while preserving seed, center and caller input."""
        asset = AssetRobot.from_file(FIXTURE, bundle_dir=self.tmp_path / "bundle", gravity=[0, 0, 0])
        records = synthetic_records(duration=0.01)
        directions = []
        for sigma in (0.02, 0.05):
            config = synthetic_config()
            for name, definition in config["parameters"].items():
                midpoint = sum(definition["bounds"]) / 2
                definition["baseline"] = midpoint
                joint, field = name.rsplit(".", 1)
                config["controller"]["drives"][joint][field] = midpoint
            config["optimizer"] = {"sigma": sigma}
            tuning = TuningRobot(asset, records, config, workspace=self.tmp_path / "runs", sources={})
            self.assertEqual(tuning.optimizer.sigma, sigma)
            tuning.optimizer.initialize(tuning.parameters)
            self.assertEqual(tuning.optimizer.diagnostics()["sigma"], sigma)
            center = tuning.parameters.encode(tuning.parameters.baseline)
            rows = tuning.optimizer.ask(tuning.optimizer.batch_size).search_values
            directions.append((rows - center) / sigma)
            self.assertEqual(config["optimizer"], {"sigma": sigma})
        np.testing.assert_allclose(directions[0], directions[1], atol=1e-12, rtol=1e-12)
        self.assertGreater(np.linalg.norm(directions[0]), 1)

    def test_default_sigma_remains_compatible(self):
        """Keep the historical initial sigma when optimizer settings are absent."""
        asset = AssetRobot.from_file(FIXTURE, bundle_dir=self.tmp_path / "bundle", gravity=[0, 0, 0])
        config = synthetic_config()
        tuning = TuningRobot(
            asset, synthetic_records(duration=0.01), config, workspace=self.tmp_path / "runs", sources={}
        )
        self.assertEqual(tuning.optimizer.sigma, 0.2)
        self.assertEqual(tuning.optimizer.batch_size, 8)
        self.assertNotIn("optimizer", config)

    def test_reject_invalid_or_inapplicable_sigma(self):
        """Reject malformed, ignored and out-of-range settings before loading a model."""
        invalid = [None, [], {"sigam": 0.02}, *({"sigma": x} for x in (None, True, "0.02", 0, -0.1, 1, 2))]
        for settings in invalid:
            with self.subTest(settings=settings):
                config = synthetic_config()
                config["optimizer"] = settings
                with self.assertRaises(ValueError):
                    TuningRobot(None, [], config, workspace=self.tmp_path / "runs", sources={})
        config = synthetic_config()
        config["parameters"] = {"shoulder.kp": config["parameters"]["shoulder.kp"]}
        config["optimizer"] = {"sigma": 0.02}
        with self.assertRaisesRegex(ValueError, "CMA-ES"):
            TuningRobot(None, [], config, workspace=self.tmp_path / "runs", sources={})

    def test_seed_controls_actual_proposals(self):
        """Reproduce matching seeds and change proposals with a different seed."""
        asset = AssetRobot.from_file(FIXTURE, bundle_dir=self.tmp_path / "bundle", gravity=[0, 0, 0])
        rows = []
        for seed in (42, 7, 42):
            config = synthetic_config()
            config["optimizer"] = {"sigma": 0.02, "seed": seed}
            tuning = TuningRobot(
                asset, synthetic_records(duration=0.01), config, workspace=self.tmp_path / "runs", sources={}
            )
            tuning.optimizer.initialize(tuning.parameters)
            rows.append(tuning.optimizer.ask(tuning.optimizer.batch_size).search_values)
        np.testing.assert_array_equal(rows[0], rows[2])
        self.assertFalse(np.array_equal(rows[0], rows[1]))

    def test_reject_invalid_seed(self):
        """Reject invalid seeds before attempting to construct a robot."""
        for seed in (None, True, -1, 2**32, 1.5, "7"):
            config = synthetic_config()
            config["optimizer"] = {"seed": seed}
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                TuningRobot(None, [], config, workspace=self.tmp_path / "runs", sources={})
