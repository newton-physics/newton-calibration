# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import unittest

import numpy as np

from newton_calibration.examples import make_tuning
from newton_calibration.tests.unittest_utils import TempPathTestCase


class TestBatchRecords(TempPathTestCase):
    def _run(self, name, record_traces):
        tuning = make_tuning("oscillator", self.tmp_path / name, optimizer="grid")
        tuning.record_traces = record_traces
        return tuning, tuning.run()

    def test_summary_records_omit_traces_without_changing_the_result(self):
        """Store scores and statuses only, and move the Jacobians to a hashed archive."""
        _, full = self._run("full", True)
        tuning, summary = self._run("summary", False)
        self.assertEqual(summary.acceptance, full.acceptance)
        self.assertEqual(summary.parameters, full.parameters)

        batches = sorted((summary.run_directory / "batches").glob("*.json"))
        self.assertTrue(batches)
        for path in batches:
            record = json.loads(path.read_text())
            self.assertEqual(record["records"], "summary")
            self.assertNotIn("features", record)
            for episode in record["episode_records"]:
                self.assertNotIn("trace", episode)
                self.assertIn("status", episode)
        full_record = json.loads(sorted((full.run_directory / "batches").glob("*.json"))[0].read_text())
        self.assertIn("features", full_record)
        self.assertIn("trace", full_record["episode_records"][0])

        state = json.loads((summary.run_directory / "run_state.json").read_text())
        sensitivity = state["analysis_reports"]["sensitivity"]
        self.assertNotIn("jacobian", sensitivity)
        archive = summary.run_directory / sensitivity["arrays"]["path"]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), sensitivity["arrays"]["sha256"])
        full_state = json.loads((full.run_directory / "run_state.json").read_text())
        with np.load(archive) as arrays:
            np.testing.assert_allclose(arrays["jacobian"], full_state["analysis_reports"]["sensitivity"]["jacobian"])
        resolved = json.loads((summary.run_directory / "resolved.json").read_text())
        self.assertEqual(resolved["batch_records"], "summary")
        self.assertTrue((summary.run_directory / "comparison.usda").exists() or tuning.comparison is None)


if __name__ == "__main__":
    unittest.main()
