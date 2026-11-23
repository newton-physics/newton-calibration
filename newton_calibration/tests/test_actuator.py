# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contract and independent-physics checks for the actuator recipe."""

import importlib.util
import json
import os
import subprocess
import sys
import time
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.actuator.recording import load_records
from examples.actuator.reference import synthetic_records
from examples.actuator.tuning import TuningActuator, synthetic_config
from newton_calibration import Episode, Tuning
from newton_calibration.tests.unittest_utils import TempPathTestCase


class TestActuator(TempPathTestCase):
    def test_inherits_lifecycle(self):
        """Keep the common lifecycle as the only run implementation."""
        self.assertIs(TuningActuator.run, Tuning.run)

    def test_recording_units_and_missing_velocity(self):
        """Convert degrees and preserve position-only observations."""
        records = synthetic_records(duration=0.05)
        record = records[0]
        record.pop("velocity")
        record["position"]["values"] = (np.asarray(record["position"]["values"]) * 180 / np.pi).tolist()
        record["position"]["unit"] = "deg"
        evidence, settings = load_records(records, scales={"position": 0.01, "velocity": 0.05})
        self.assertEqual(evidence.train_data[0].reference.shape[1], 1)
        self.assertEqual(settings[record["id"]]["channels"], ["position"])

    def test_rejects_invalid_records(self):
        """Reject invalid times, uncovered commands, units and nonfinite samples."""
        for field in ("duplicate_time", "coverage", "unit", "nan"):
            records = synthetic_records(duration=0.05)
            with self.subTest(field=field):
                if field == "duplicate_time":
                    records[0]["position"]["times"][1] = 0
                elif field == "coverage":
                    records[0]["command"]["times"][-1] = 0.049
                elif field == "unit":
                    records[0]["command"]["unit"] = "amp"
                else:
                    records[0]["position"]["values"][1] = float("nan")
                with self.assertRaises(ValueError):
                    load_records(records, scales={"position": 0.01, "velocity": 0.05})

    def test_independent_observation_times(self):
        """Align measured velocity in SI without deriving it from position."""
        records = synthetic_records(duration=0.04)
        record = records[0]
        record["position"]["times"] = [0.001, 0.011, 0.031]
        record["position"]["values"] = [1, 2, 3]
        record["velocity"] = {"times": [0.0, 0.02, 0.04], "values": [0, 20, 40], "unit": "deg/s"}
        evidence, settings = load_records(records, scales={"position": 0.01, "velocity": 0.05})
        np.testing.assert_allclose(evidence.train_data[0].reference[:, 1], np.deg2rad([1, 11, 31]))
        self.assertEqual(settings[record["id"]]["source_units"]["velocity"], "deg/s")
        record["velocity"]["times"][0] = 0.002
        with self.assertRaisesRegex(ValueError, "velocity must cover"):
            load_records(records, scales={"position": 0.01, "velocity": 0.05})

    def test_rejects_missing_protocol_and_leaking_groups(self):
        """Require explicit physical settings and independent capture groups."""
        records = synthetic_records(duration=0.04)
        config = synthetic_config()
        del config["fixed"]["inertia"]
        with self.assertRaises(ValueError):
            TuningActuator(records, config, workspace=self.tmp_path)
        records[2]["group"] = records[0]["group"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            TuningActuator(records, synthetic_config(), workspace=self.tmp_path)

    def test_measurement_mapping_affects_fingerprint(self):
        """Include channel layout and protocol in the recorded identities."""
        records = synthetic_records(duration=0.04)
        a = TuningActuator(records, synthetic_config(), workspace=self.tmp_path)
        records[0].pop("velocity")
        b = TuningActuator(records, synthetic_config(), workspace=self.tmp_path)
        self.assertNotEqual(a.evaluator.evaluation_id, b.evaluator.evaluation_id)
        self.assertNotEqual(a.evidence.fingerprint(), b.evidence.fingerprint())

    def test_invalid_scores_cannot_be_accepted(self):
        """Keep failed, missing and nonfinite episodes out of valid results."""
        tuning = TuningActuator(synthetic_records(duration=0.04), synthetic_config(), workspace=self.tmp_path)
        exp = tuning.evidence.train_data[0]
        episode = Episode("trial", exp.id, 0, "timeout")
        scores = tuning.evaluator.score_batch((episode,), ("trial",), (exp,))
        self.assertFalse(scores[3][0])
        self.assertTrue(np.isinf(scores[0][0]))
        with self.assertRaises(ValueError):
            tuning.evaluator.score_batch((), ("trial",), (exp,))


@unittest.skipUnless(
    importlib.util.find_spec("newton") and importlib.util.find_spec("warp"), "install the newton extra"
)
class TestActuatorNewton(TempPathTestCase):
    def make_case(self, *, duration=0.08):
        config = synthetic_config()
        config["parameters"]["kp"]["baseline"] = 2.0
        config["parameters"]["kd"]["baseline"] = 0.15
        tuning = TuningActuator(synthetic_records(duration=duration), config, workspace=self.tmp_path)
        tuning.template.freeze()
        return tuning

    def simulate(self, tuning, experiment, *, values=None, template=None):
        env = (template or tuning.template).instantiate(
            tuning.asset, values or tuning.asset.baseline_parameters, experiment
        )
        self.addCleanup(env.close)
        return env, env.run(experiment.times, seed=0, deadline=None)

    def test_analytical_reference_and_refinement(self):
        """Match independent held-torque dynamics and improve with a finer timestep."""
        tuning = self.make_case(duration=0.2)
        exp = tuning.evidence.train_data[0]
        env, coarse = self.simulate(tuning, exp)
        fine_env, fine = self.simulate(
            tuning, exp, template=tuning.template.refined({"numeric_settings": {"dt": 0.000125}})
        )
        error = np.abs(coarse.values - exp.reference).max(axis=0)
        self.assertLess(error[0], 0.001)
        self.assertLess(error[1], 0.005)
        self.assertLess(np.linalg.norm(fine.values - exp.reference), np.linalg.norm(coarse.values - exp.reference))
        np.testing.assert_allclose(env.applied_parameters["inertia"], 0.01, rtol=1e-6)
        np.testing.assert_array_equal(env.model.joint_target_ke.numpy(), [0])
        np.testing.assert_array_equal(env.model.joint_target_kd.numpy(), [0])
        self.assertEqual(env.solver.angular_damping, 0)
        np.testing.assert_array_equal(np.asarray(env.control_history)[:, 0], np.asarray(fine_env.control_history)[:, 0])

    def test_hold_saturation_and_tick_grid(self):
        """Hold saturated torque until a controller tick, even across intermediate observations."""
        tuning = self.make_case()
        exp = tuning.evidence.train_data[0]
        settings = dict(tuning.template.experiment_settings[exp.id])
        settings["command_times"] = [0.0, 0.001, 0.08]
        settings["command"] = [0.0, 100.0, 100.0]
        values = {**tuning.template.fixed_parameters, **tuning.asset.baseline_parameters}
        env = tuning.asset.create_environment(tuning.template, values, settings)
        trace = env.run(np.array([0, 0.001, 0.002, 0.003, 0.004]), seed=0, deadline=None)
        self.assertEqual(env.control_history, [(0.0, 0.0), (0.002, 2.0)])
        self.assertEqual(trace.values[2, 0], 0.0)
        self.assertGreater(trace.values[-1, 0], 0.0)

    def test_fresh_initial_state_and_parameter_readback(self):
        """Isolate A/B/A candidates and reject repeated use of an environment."""
        tuning = self.make_case()
        exp = tuning.evidence.train_data[0]
        env, a = self.simulate(tuning, exp)
        _, b = self.simulate(tuning, exp, values={"kp": 4.0, "kd": 0.3})
        _, again = self.simulate(tuning, exp)
        np.testing.assert_array_equal(a.values, again.values)
        self.assertGreater(np.max(np.abs(a.values - b.values)), 0.001)
        env.model.joint_damping.assign(np.array([0.07], dtype=np.float32))
        self.assertAlmostEqual(env.applied_parameters["damping"], 0.07, places=6)
        with self.assertRaisesRegex(RuntimeError, "fresh"):
            env.run(exp.times, seed=0, deadline=None)

    def test_timeout_and_nan_status(self):
        """Classify elapsed deadlines and divergent controller output as failures."""
        tuning = self.make_case()
        exp = tuning.evidence.train_data[0]
        episodes = tuning.rollout.execute_batch(
            tuning.template,
            tuning.asset,
            [("timeout", dict(tuning.asset.baseline_parameters))],
            (exp,),
            deadline=time.monotonic() - 1,
        )
        self.assertEqual(episodes[0].status, "timeout")
        episodes = tuning.rollout.execute_batch(
            tuning.template,
            tuning.asset,
            [("nan", {"kp": float("nan"), "kd": 0.15})],
            (exp,),
        )
        self.assertNotEqual(episodes[0].status, "completed")
        tuning.rollout.close()

    def test_export_replay_in_fresh_process(self):
        """Reconstruct exported model and protocol in a separate interpreter."""
        tuning = self.make_case()
        experiments = tuning.evidence.train_data + tuning.evidence.validation_data
        expected = {e.id: self.simulate(tuning, e)[1].values for e in experiments}
        tuning.asset.export(dict(tuning.asset.baseline_parameters), self.tmp_path / "candidate_asset.json")
        resolved = {
            "template": tuning.template.snapshot(),
            "evidence": {
                "train": [asdict(e) for e in tuning.evidence.train_data],
                "validation": [asdict(e) for e in tuning.evidence.validation_data],
                "sources": tuning.evidence.sources,
            },
        }
        (self.tmp_path / "resolved.json").write_text(json.dumps(resolved, default=lambda x: x.tolist()))
        subprocess.run(
            [sys.executable, "-m", "examples.actuator.run", "--replay", str(self.tmp_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        replay = json.loads((self.tmp_path / "actuator-replay.json").read_text())
        for key, values in expected.items():
            np.testing.assert_allclose(replay["experiments"][key]["values"], values, atol=1e-7, rtol=0)
        self.assertTrue(replay["validation"]["passed"])

    def test_budget_holdout_and_unexcited_data_cannot_release(self):
        """Reject incomplete runs, absent holdout and unidentifiable gains."""
        for kind in ("budget", "holdout", "unexcited"):
            records = synthetic_records(duration=0.08)
            config = synthetic_config()
            config["parameters"]["kp"]["baseline"] = 2.0
            config["parameters"]["kd"]["baseline"] = 0.15
            config["decision"]["target"] = 1.0
            if kind == "budget":
                config["decision"]["max_rollouts"] = 1
            elif kind == "holdout":
                records = records[:2]
            else:
                for record in records:
                    for signal in ("command", "position", "velocity"):
                        record[signal]["values"] = [0.0] * len(record[signal]["times"])
            with self.subTest(kind=kind):
                tuning = TuningActuator(records, config, workspace=self.tmp_path / kind)
                result = tuning.run()
                self.assertNotEqual(result.acceptance, "accepted")
                self.assertIsNone(result.asset_path)
                if kind == "holdout":
                    self.assertEqual(result.acceptance, "unvalidated", tuning.state.errors)
                elif kind == "unexcited":
                    self.assertEqual(result.acceptance, "inconclusive", tuning.state.errors)

    def test_validation_does_not_change_grid_candidates(self):
        """Keep training proposals and scores independent of held-out measurements."""
        histories = []
        for changed in (False, True):
            records = synthetic_records(duration=0.08)
            if changed:
                records[2]["position"]["values"] = [100.0] * len(records[2]["position"]["times"])
            config = synthetic_config()
            config["decision"].update(max_trials=4, target=0.0)
            tuning = TuningActuator(records, config, optimizer="grid", workspace=self.tmp_path / str(changed))
            result = tuning.run()
            history = []
            for batch in tuning.state.batch_history:
                if batch["purpose"] == "train":
                    saved = json.loads((result.run_directory / batch["artifact"]).read_text())
                    history.append((saved["parameter_values"], saved["objectives"]))
            self.assertTrue(history, tuning.state.errors)
            histories.append(history)
        self.assertEqual(histories[0], histories[1])


@unittest.skipUnless(os.environ.get("ACTUATOR_RECOVERY_TESTS") == "1", "opt-in full Newton calibration searches")
class TestActuatorRecovery(TempPathTestCase):
    def recover(self, label, kp, kd):
        config = synthetic_config(solver=os.environ.get("ACTUATOR_TEST_SOLVER", "SolverFeatherstone"))
        config["parameters"]["kp"]["baseline"] = kp
        config["parameters"]["kd"]["baseline"] = kd
        root = Path(os.environ.get("ACTUATOR_TEST_WORKSPACE", str(self.tmp_path))) / label
        tuning = TuningActuator(synthetic_records(), config, workspace=root)
        result = tuning.run()
        print(result, flush=True)
        self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
        np.testing.assert_allclose([result.parameters["kp"], result.parameters["kd"]], [2.0, 0.15], rtol=0.02)
        self.assertTrue(tuning.state.validation_report["passed"])

    def test_initial_nominal(self):
        """Recover both gains from the default initial guess."""
        self.recover("nominal", 1.0, 0.08)

    def test_initial_high(self):
        """Recover both gains when both initial gains exceed the reference."""
        self.recover("high", 4.0, 0.3)

    def test_initial_mixed(self):
        """Recover both gains from a low stiffness and high damping guess."""
        self.recover("mixed", 0.7, 0.3)


if __name__ == "__main__":
    unittest.main()
