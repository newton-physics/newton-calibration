# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Multi-joint PD identification against independent Lagrange dynamics."""

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.actuator.robot.asset import AssetRobot
from examples.actuator.robot.recording import load_records
from examples.actuator.robot.reference import synthetic_config, synthetic_records
from examples.actuator.robot.tuning import TuningRobot
from newton_calibration import BatchResult, Episode, SimulationError, Trace
from newton_calibration.tests.unittest_utils import TempPathTestCase

FIXTURE = Path(__file__).resolve().parents[2] / "examples/actuator/robot/fixtures/two_link.xml"
BACKEND = all(
    importlib.util.find_spec(name) for name in ("newton", "warp", "mujoco", "mujoco_warp", "scipy", "trimesh", "PIL")
)
TRUTH = {"shoulder.kp": 18.0, "shoulder.kd": 2.0, "elbow.kp": 6.0, "elbow.kd": 0.5}


@unittest.skipUnless(BACKEND, "install the robot extra")
class TestRobotCalibration(TempPathTestCase):
    def make_case(self, *, duration=0.2, records=None):
        asset = AssetRobot.from_file(FIXTURE, bundle_dir=self.tmp_path / "bundle", gravity=[0, 0, 0])
        return TuningRobot(
            asset,
            records or synthetic_records(duration=duration),
            synthetic_config(),
            workspace=self.tmp_path / "runs",
            sources={"provenance": {"kind": "synthetic", "reference": "Lagrange"}},
        )

    def simulate(self, tuning, *, parameters=None, template=None):
        exp = tuning.evidence.train_data[0]
        env = (template or tuning.template).instantiate(tuning.asset, parameters or TRUTH, exp)
        self.addCleanup(env.close)
        return env, env.run(exp.times, seed=0, deadline=None)

    def test_independent_reference_refinement_and_isolation(self):
        """Recover coupled motion and preserve controller timing under refinement."""
        tuning = self.make_case()
        tuning.template.freeze()
        env, coarse = self.simulate(tuning)
        refined, fine = self.simulate(tuning, template=tuning.template.refined({"numeric_settings": {"dt": 0.000125}}))
        reference = tuning.evidence.train_data[0].reference
        self.assertLess(np.max(np.abs(coarse.values[:, :2] - reference[:, :2])), 0.0002)
        self.assertLess(np.linalg.norm(fine.values - reference), np.linalg.norm(coarse.values - reference))
        self.assertEqual([item["time"] for item in env.history], [item["time"] for item in refined.history])
        _, other = self.simulate(tuning, parameters=dict(tuning.asset.baseline_parameters))
        _, again = self.simulate(tuning)
        np.testing.assert_array_equal(again.values, coarse.values)
        self.assertGreater(np.linalg.norm(other.values - coarse.values), 0.01)

    def test_saved_body_positions_are_observation_snapshots(self):
        """Keep each body's saved position at its observation time when CPU state buffers are reused."""
        tuning = self.make_case()
        tuning.template.freeze()
        env, trace = self.simulate(tuning)
        channels = env.settings["channels"]
        column = next(i for i, (kind, name) in enumerate(channels) if (kind, name) == ("position", "shoulder"))
        q = trace.values[:, column]
        # The fixture's forearm origin is one metre along the rotating upper arm.
        expected = np.column_stack((np.cos(q), np.zeros(len(q)), -np.sin(q)))
        index = env.bindings["bodies"]["forearm"]["newton_body"]
        np.testing.assert_allclose(trace.positions[:, index], expected, atol=1e-6, rtol=0)

    def test_saturation_hold_actual_readback_and_failures(self):
        """Hold each saturated torque and reject timeouts, NaNs and reused state."""
        import mujoco  # noqa: PLC0415

        tuning = self.make_case(duration=0.01)
        tuning.template.freeze()
        exp = tuning.evidence.train_data[0]
        settings = copy.deepcopy(tuning.template.snapshot()["experiment_settings"][exp.id])
        for name, sign in (("shoulder", 1), ("elbow", -1)):
            settings["command"][name] = {"times": [0, 0.001, 0.01], "values": [100 * sign, -100 * sign, -100 * sign]}
        values = {**tuning.template.fixed_parameters, **TRUTH}
        env = tuning.asset.create_environment(tuning.template, values, settings)
        trace = env.run(np.array([0.0, 0.001, 0.002, 0.003, 0.004]), seed=0, deadline=None)
        np.testing.assert_array_equal(env.history[0]["torque"], [25, -8])
        np.testing.assert_array_equal(env.history[1]["torque"], [-25, 8])
        self.assertEqual([item["time"] for item in env.history], [0, 0.002])
        self.assertTrue(np.isfinite(trace.values).all())
        self.assertEqual(env.applied_parameters["shoulder.kp"], 18)
        env.kp[0] = 19
        self.assertEqual(env.applied_parameters["shoulder.kp"], 19)
        self.assertTrue((env.solver.mj_model.actuator_gainprm == 0).all())
        self.assertTrue((env.solver.mj_model.actuator_biasprm == 0).all())
        self.assertTrue(env.solver.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        with self.assertRaises(RuntimeError):
            env.run(exp.times, seed=0, deadline=None)
        fresh = tuning.template.instantiate(tuning.asset, TRUTH, exp)
        with self.assertRaises(TimeoutError):
            fresh.run(exp.times, seed=0, deadline=time.monotonic() - 1)
        with self.assertRaises(SimulationError):
            tuning.template.instantiate(tuning.asset, {**TRUTH, "elbow.kd": float("nan")}, exp)

    def test_multirate_mask_and_per_joint_error_gates(self):
        """Score only original per-channel samples and reject one bad joint."""
        records = synthetic_records(duration=0.02)
        for record in records:
            del record["velocity"]
            for field in ("times", "values"):
                record["position"]["elbow"][field] = record["position"]["elbow"][field][::2]
        tuning = self.make_case(records=records)
        exp = tuning.evidence.train_data[0]
        self.assertEqual(exp.mask[:, 0].sum(), 6)
        self.assertEqual(exp.mask[:, 1].sum(), 11)
        values = exp.reference.copy()
        values[~exp.mask] = 999
        episode = Episode("test", exp.id, 0, "completed", Trace(exp.times, values))
        scores = tuning.evaluator.score_batch((episode,), ("test",), (exp,))
        self.assertEqual(scores[0][0], 0)
        values[exp.mask[:, 0], 0] += 0.04
        scores = tuning.evaluator.score_batch((episode,), ("test",), (exp,))
        result = BatchResult(("test",), np.empty((1, 0)), np.empty((1, 0)), *scores, (episode,), {}, "test")
        self.assertFalse(tuning.validator.accepts(result))
        self.assertAlmostEqual(result.metrics[0, 2], 0.04)

    def test_bad_inputs_and_acquisition_leakage(self):
        """Reject duplicate timestamps, missing control and reused acquisition groups."""
        original = synthetic_records(duration=0.01)
        for mutation in ("timestamp", "command", "nan", "group"):
            records = copy.deepcopy(original)
            if mutation == "timestamp":
                records[0]["position"]["elbow"]["times"][1] = 0
            elif mutation == "command":
                del records[0]["command"]["elbow"]
            elif mutation == "nan":
                records[0]["initial_state"]["velocity"]["elbow"] = float("nan")
            else:
                records[2]["group"] = records[0]["group"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_records(
                    records,
                    active_joints=("elbow", "shoulder"),
                    driven_joints=("elbow", "shoulder"),
                    scales=synthetic_config()["scales"],
                )

    def test_unobserved_driven_joint_and_optional_velocity(self):
        """Reject an unobserved drive and report absent velocity without a false zero."""
        records = synthetic_records(duration=0.02)
        del records[2]["position"]["elbow"]
        with self.assertRaisesRegex(ValueError, "observe position for all driven joints"):
            self.make_case(records=records)
        records = synthetic_records(duration=0.02)
        del records[2]["velocity"]
        asset = AssetRobot.from_file(FIXTURE, bundle_dir=self.tmp_path / "optional", gravity=[0, 0, 0])
        tuning = TuningRobot(
            asset,
            records,
            synthetic_config(),
            workspace=self.tmp_path / "optional-runs",
            sources={"provenance": {"kind": "synthetic"}},
        )

        def evaluate(asset, experiments):
            exp = experiments[0]
            episode = Episode("test", exp.id, 0, "completed", Trace(exp.times, exp.reference.copy()))
            scores = tuning.evaluator.score_batch((episode,), ("test",), experiments)
            return BatchResult(("test",), np.empty((1, 0)), np.empty((1, 0)), *scores, (episode,), {}, "test")

        report = tuning.validator.validate(tuning.asset, tuning.evidence.validation_data, evaluate)
        self.assertTrue(report["passed"])
        item = report["experiments"][0]
        self.assertEqual(set(item["observed_channels"]), {("position", "elbow"), ("position", "shoulder")})
        self.assertIsNone(item["physical_errors"]["elbow.velocity.rmse"])
        self.assertIsNone(item["physical_errors"]["shoulder.velocity.max"])
        self.assertEqual(item["physical_errors"]["elbow.position.rmse"], 0.0)

    def test_changed_candidate_reloads_and_preserves_resources(self):
        """Reload fitted gains without depending on the original baseline identity."""
        tuning = self.make_case()
        tuning.template.freeze()
        _, expected = self.simulate(tuning)
        candidate = tuning.asset.export(TRUTH, self.tmp_path / "candidate/asset.json")
        reloaded = AssetRobot.from_export(candidate)
        release = self.tmp_path / "release/asset.json"
        reloaded.stage_release(candidate, release)
        final = AssetRobot.from_export(release)
        exp = tuning.evidence.train_data[0]
        env = tuning.template.instantiate(final, final.baseline_parameters, exp)
        actual = env.run(exp.times, seed=0, deadline=None)
        np.testing.assert_array_equal(actual.values, expected.values)

    def test_validation_data_does_not_change_training_candidates(self):
        """Keep proposals and training scores unchanged after replacing holdout values."""
        asset = AssetRobot.from_file(FIXTURE, bundle_dir=self.tmp_path / "source", gravity=[0, 0, 0])
        histories = []
        for changed in (False, True):
            records, config = synthetic_records(duration=0.04), synthetic_config()
            if changed:
                for signal in records[2]["position"].values():
                    signal["values"] = [100.0] * len(signal["values"])
            config["parameters"] = {"shoulder.kp": config["parameters"]["shoulder.kp"]}
            config["decision"].update(max_trials=4, target=0.0)
            tuning = TuningRobot(
                asset,
                records,
                config,
                workspace=self.tmp_path / str(changed),
                sources={"provenance": {"kind": "synthetic"}},
            )
            result = tuning.run()
            history = []
            for batch in tuning.state.batch_history:
                if batch["purpose"] == "train":
                    saved = json.loads((result.run_directory / batch["artifact"]).read_text())
                    history.append((saved["parameter_values"], saved["objectives"]))
            self.assertTrue(history, tuning.state.errors)
            histories.append(history)
        self.assertEqual(histories[0], histories[1])

    def test_missing_validation_budget_and_rank_fail_closed(self):
        """Prevent acceptance for absent holdout, insufficient budget or unexcited gains."""
        asset = AssetRobot.from_file(FIXTURE, bundle_dir=self.tmp_path / "source", gravity=[0, 0, 0])
        for mode in ("missing_validation", "budget", "rank"):
            records, config = synthetic_records(duration=0.08), synthetic_config()
            for name, value in TRUTH.items():
                joint, field = name.rsplit(".", 1)
                config["controller"]["drives"][joint][field] = value
                config["parameters"][name]["baseline"] = value
            config["parameters"] = {"shoulder.kp": config["parameters"]["shoulder.kp"]}
            config["decision"]["target"] = 1.0
            if mode == "missing_validation":
                records = records[:2]
            elif mode == "budget":
                config["decision"]["max_rollouts"] = 1
            else:
                for record in records:
                    for values in record["initial_state"].values():
                        values.update(dict.fromkeys(values, 0.0))
                    for kind in ("command", "position", "velocity"):
                        for signal in record[kind].values():
                            signal["values"] = [0.0] * len(signal["values"])
            with self.subTest(mode=mode):
                tuning = TuningRobot(
                    asset,
                    records,
                    config,
                    workspace=self.tmp_path / mode,
                    sources={"provenance": {"kind": "synthetic"}},
                )
                result = tuning.run()
                self.assertNotEqual(result.acceptance, "accepted")
                self.assertIsNone(result.asset_path)
                if mode == "missing_validation":
                    self.assertEqual(result.acceptance, "unvalidated", tuning.state.errors)
                if mode == "rank":
                    self.assertEqual(result.acceptance, "inconclusive", tuning.state.errors)

    def test_new_process_release_replay_and_report(self):
        """Revalidate a moved calibration release and plot only saved trajectories."""
        tuning = self.make_case(duration=0.08)
        # One gain suffices for this lifecycle delivery test; full recovery is separate.
        config = synthetic_config()
        for name, value in TRUTH.items():
            joint, field = name.rsplit(".", 1)
            config["controller"]["drives"][joint][field] = value
            config["parameters"][name]["baseline"] = value
        config["parameters"] = {"shoulder.kp": config["parameters"]["shoulder.kp"]}
        config["decision"]["target"] = 1.0
        tuning = TuningRobot(
            tuning.asset,
            synthetic_records(duration=0.08),
            config,
            workspace=self.tmp_path / "deliver",
            sources={"provenance": {"kind": "synthetic"}},
        )
        result = tuning.run()
        self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
        shutil.copyfile(tuning.run_directory / "resolved.json", result.asset_path.parent / "resolved.json")
        from examples.actuator.robot.report import _plot, write_report  # noqa: PLC0415

        expected = json.loads((tuning.run_directory / tuning.state.best_candidate["batch_ref"]).read_text())[
            "episode_records"
        ]
        with patch("examples.actuator.robot.report._plot", wraps=_plot) as plots:
            report = write_report(tuning.run_directory)
        # Refinement checks must not replace the selected candidate's nominal-step trajectory.
        selected = next(values for label, _, values in plots.call_args_list[0].args[0] if label == "selected candidate")
        first = next(episode for episode in expected if episode["experiment_id"] == tuning.evidence.train_data[0].id)
        np.testing.assert_array_equal(selected, np.asarray(first["trace"]["values"])[:, 0])
        self.assertIn("synthetic", report.read_text())
        self.assertIn("shoulder position", report.read_text())
        destination = self.tmp_path / "moved"
        shutil.move(result.asset_path.parent, destination)
        shutil.rmtree(self.tmp_path / "bundle")
        shutil.rmtree(tuning.run_directory)
        output = self.tmp_path / "replay.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.actuator.robot.calibrate",
                "--replay",
                str(destination),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        replay = json.loads(output.read_text())
        self.assertTrue(replay["validation"]["passed"])
        for episode in expected:
            np.testing.assert_allclose(
                replay["experiments"][episode["experiment_id"]]["values"], episode["trace"]["values"], atol=1e-7, rtol=0
            )


@unittest.skipUnless(BACKEND and os.environ.get("ROBOT_RECOVERY_TESTS") == "1", "opt-in full robot searches")
class TestRobotRecovery(TempPathTestCase):
    def test_three_initial_guesses(self):
        """Recover all four gains within two percent and pass independent validation."""
        root = Path(os.environ.get("ROBOT_TEST_WORKSPACE", str(self.tmp_path)))
        selected = os.environ.get("ROBOT_RECOVERY_START", "all")
        self.assertIn(selected, ("all", "0", "1", "2"))
        records = synthetic_records()
        for index, gains in enumerate(((12, 1.1, 4, 0.3), (26, 3, 8, 0.8), (10, 2.8, 8.5, 0.25))):
            if selected != "all" and selected != str(index):
                continue
            with self.subTest(initial=index):
                case = root / f"start-{index}"
                asset = AssetRobot.from_file(FIXTURE, bundle_dir=case / "input", gravity=[0, 0, 0])
                config = synthetic_config()
                for name, value in zip(TRUTH, gains, strict=True):
                    config["parameters"][name]["baseline"] = value
                    joint, field = name.rsplit(".", 1)
                    config["controller"]["drives"][joint][field] = value
                tuning = TuningRobot(
                    asset,
                    records,
                    config,
                    workspace=case / "runs",
                    sources={"provenance": {"kind": "synthetic", "reference": "independent Lagrange dynamics"}},
                )
                result = tuning.run()
                errors = {
                    name: abs(tuning.state.best_candidate["parameters"][name] / value - 1)
                    for name, value in TRUTH.items()
                }
                (case / "recovery.json").write_text(
                    json.dumps(
                        {
                            "acceptance": result.acceptance,
                            "relative_errors": errors,
                            "run": str(tuning.run_directory),
                            "errors": tuning.state.errors,
                        },
                        indent=2,
                    )
                )
                self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
                self.assertLessEqual(max(errors.values()), 0.02, errors)
