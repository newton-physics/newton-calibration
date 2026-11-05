# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import sys
import unittest
from unittest import mock

import numpy as np

from newton_calibration.__main__ import main
from newton_calibration.examples import make_tuning
from newton_calibration.tests.unittest_utils import TempPathTestCase


class TestWorkflow(TempPathTestCase):
def test_generic_run_recovers_parameters_and_reloads_asset(self):
    for case in ("oscillator", "sliding"):
        with self.subTest(case=case):
            tuning = make_tuning(case, self.tmp_path / case, optimizer="grid")
            result = tuning.run()
            self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
            self.assertEqual(
                tuning.state.analysis_reports["identifiability"]["rank"],
                len(tuning.parameters.names),
            )
            self.assertTrue(tuning.state.decision_history[-1]["reason"])
            self.assertTrue(result.asset_path.exists())
            self.assertTrue((result.run_directory / "comparison.usda").exists())
            report = json.loads((result.run_directory / "run_state.json").read_text())
            self.assertTrue(report["validation_report"]["passed"])
            self.assertGreater(report["budget_usage"]["analysis"]["rollouts"], 0)
            self.assertTrue(report["template_snapshot"]["numeric_settings"])

def test_sliding_cli_runs_without_an_optimizer_flag(self):
    """Run the sliding demo with its default optimizer and accept the fitted asset."""
    argv = ["newton-calibration", "demo", "--case", "sliding", "--workspace", str(self.tmp_path)]
    output = io.StringIO()
    with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
        self.assertEqual(main(), 0)
    self.assertEqual(json.loads(output.getvalue())["acceptance"], "accepted")

def test_changing_validation_does_not_change_training(self):
    first = make_tuning("sliding", self.tmp_path / "first", optimizer="grid")
    second = make_tuning("sliding", self.tmp_path / "second", optimizer="grid")
    second.evidence.validation_data[0].reference[...] += 10
    result_a, result_b = first.run(), second.run()
    self.assertEqual(result_a.parameters, result_b.parameters)
    self.assertEqual(result_a.acceptance, "accepted")
    self.assertEqual(result_b.acceptance, "rejected_validation")
    a = [x["objectives"] for x in first.state.batch_history if x["purpose"] == "train"]
    b = [x["objectives"] for x in second.state.batch_history if x["purpose"] == "train"]
    self.assertEqual(len(a), len(b))
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)

def test_workspace_keeps_repeated_runs_and_their_intermediate_files(self):
    workspace = self.tmp_path / "shared workspace"
    first = make_tuning("sliding", workspace, optimizer="grid")
    second = make_tuning("sliding", workspace, optimizer="grid")
    self.assertFalse(workspace.exists())
    result_a = first.run()
    saved_state = (result_a.run_directory / "run_state.json").read_bytes()
    result_b = second.run()
    self.assertNotEqual(result_a.run_directory, result_b.run_directory)
    self.assertEqual((result_a.run_directory / "run_state.json").read_bytes(), saved_state)
    for tuning, result in ((first, result_a), (second, result_b)):
        self.assertEqual(result.acceptance, "accepted")
        self.assertEqual(result.run_directory.parent, workspace)
        resolved = json.loads((result.run_directory / "resolved.json").read_text())
        self.assertEqual(resolved["workspace"], str(workspace))
        self.assertEqual(resolved["run_directory"], str(result.run_directory))
        self.assertTrue((result.run_directory / "candidate_asset.json").is_file())
        self.assertTrue(result.asset_path.is_relative_to(result.run_directory))
        for record in tuning.state.batch_history:
            batch_path = result.run_directory / record["artifact"]
            batch = json.loads(batch_path.read_text())
            self.assertEqual(batch["trial_ids"], list(record["trial_ids"]))
        self.assertLessEqual(
            {"baseline", "analysis", "train", "validation"},
            {record["purpose"] for record in tuning.state.batch_history},
        )

def test_default_workspace_is_resolved_at_construction(self):
    with contextlib.chdir(self.tmp_path):
        tuning = make_tuning("sliding", optimizer="grid")
    self.assertEqual(tuning.workspace, self.tmp_path / "workspace")
    self.assertEqual(tuning.run_directory.parent, tuning.workspace)
    self.assertFalse(tuning.workspace.exists())

def test_workspace_file_is_not_overwritten(self):
    workspace = self.tmp_path / "file"
    workspace.write_text("keep")
    tuning = make_tuning("sliding", workspace, optimizer="grid")
    with self.assertRaises(FileExistsError):
        tuning.run()
    self.assertEqual(workspace.read_text(), "keep")

def test_existing_generated_run_directory_is_not_reused(self):
    tuning = make_tuning("sliding", self.tmp_path, optimizer="grid")
    tuning.run_directory.mkdir()
    with self.assertRaises(FileExistsError):
        tuning.run()
    self.assertEqual(list(tuning.run_directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
