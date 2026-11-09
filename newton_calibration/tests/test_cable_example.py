# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the portable cable evidence example."""

import importlib
import unittest
from unittest import mock

import numpy as np

from newton_calibration.tests.unittest_utils import TempPathTestCase


class TestCableExample(TempPathTestCase):
    def test_archived_cable_scores_preserve_failed_acceptance(self):
        archive = importlib.import_module("examples.cable_calibration.verify")
        result = archive.verify()
        self.assertEqual(result["verified_traces"], 14)
        self.assertIs(result["simulation_executed"], False)
        np.testing.assert_allclose(result["sls_objective_reduction_percent"], 76.0087819876, rtol=1e-6)
        self.assertEqual(
            result["historical_acceptance"],
            {
                "A": "rejected_training",
                "B": "rejected_validation",
                "Bstar": "rejected_validation",
                "C": "rejected_validation",
            },
        )

    def test_archive_verification_rejects_changed_prediction(self):
        archive = importlib.import_module("examples.cable_calibration.verify")
        original = archive.CASE
        for name in ("data/experiments.json", "archive/expected-metrics.json"):
            target = self.tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((original / name).read_bytes())
        with np.load(original / "data/traces.npz", allow_pickle=False) as saved:
            traces = {key: saved[key].copy() for key in saved.files}
        # Perturb every observation, including all valid samples, by a physical 1 cm.
        key = next(iter(traces))
        traces[key] += 0.01
        np.savez_compressed(self.tmp_path / "data/traces.npz", **traces)
        with mock.patch.object(archive, "CASE", self.tmp_path), self.assertRaises(AssertionError):
            archive.verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
