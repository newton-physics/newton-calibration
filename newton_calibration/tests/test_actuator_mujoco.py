# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverMuJoCo integration contracts for sampled actuator identification."""

import importlib.util
import json
import subprocess
import sys
import time
import unittest
from contextlib import chdir
from dataclasses import asdict

import numpy as np

from examples.actuator.reference import analytical_response, synthetic_records
from examples.actuator.tuning import TuningActuator, synthetic_config
from newton_calibration import SimulationError
from newton_calibration.tests.unittest_utils import TempPathTestCase


class TestActuatorSolverSelection(TempPathTestCase):
    def test_solver_protocol_is_frozen(self):
        """Record the selected solver and CPU integration options in the asset and template."""
        config = synthetic_config(solver="SolverMuJoCo")
        tuning = TuningActuator(synthetic_records(duration=0.04), config, workspace=self.tmp_path)
        self.assertEqual(tuning.asset.config["solver"], "SolverMuJoCo")
        self.assertTrue(tuning.template.numeric_settings["use_mujoco_cpu"])
        self.assertEqual(tuning.template.numeric_settings["integrator"], "implicitfast")
        with self.assertRaisesRegex(ValueError, "solver"):
            TuningActuator(synthetic_records(duration=0.04), {**config, "solver": "unknown"}, workspace=self.tmp_path)

    def test_legacy_default_and_solver_mismatch(self):
        """Retain old Featherstone protocols and reject switching an exported asset's solver."""
        records = synthetic_records(duration=0.04)
        config = synthetic_config()
        del config["solver"]
        old = TuningActuator(records, config, workspace=self.tmp_path)
        self.assertEqual(old.template.numeric_settings["solver"], "SolverFeatherstone")
        other = TuningActuator(records, synthetic_config(solver="SolverMuJoCo"), workspace=self.tmp_path)
        self.assertNotEqual(old.asset.asset_id, other.asset.asset_id)
        with self.assertRaisesRegex(ValueError, "asset configuration"):
            old.asset.create_environment(
                other.template,
                {**other.template.fixed_parameters, **other.asset.baseline_parameters},
                other.template.experiment_settings[records[0]["id"]],
            )


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("newton", "warp", "mujoco", "mujoco_warp")),
    "install the mujoco extra",
)
class TestActuatorMuJoCo(TempPathTestCase):
    def make_case(self, *, initial=(0.0, 0.0)):
        config = synthetic_config(solver="SolverMuJoCo")
        config["parameters"]["kp"]["baseline"] = 2.0
        config["parameters"]["kd"]["baseline"] = 0.15
        records = synthetic_records(duration=0.2)
        for record in records:
            record["initial_state"] = list(initial)
            states = analytical_response(
                record["command"]["values"],
                control_dt=0.002,
                inertia=0.01,
                damping=0.02,
                kp=2.0,
                kd=0.15,
                torque_limit=2.0,
                initial=initial,
            )
            record["position"]["values"] = states[:, 0].tolist()
            record["velocity"]["values"] = states[:, 1].tolist()
        tuning = TuningActuator(records, config, workspace=self.tmp_path)
        tuning.template.freeze()
        return tuning

    def simulate(self, tuning, *, values=None, template=None):
        exp = tuning.evidence.train_data[0]
        env = (template or tuning.template).instantiate(tuning.asset, values or tuning.asset.baseline_parameters, exp)
        self.addCleanup(env.close)
        return env, env.run(exp.times, seed=0, deadline=None)

    def test_analytical_response_refinement_and_reset(self):
        """Match independent dynamics, preserve controller ticks and reset nonzero initial states."""
        tuning = self.make_case(initial=(0.2, -0.1))
        exp = tuning.evidence.train_data[0]
        env, coarse = self.simulate(tuning)
        refined, fine = self.simulate(tuning, template=tuning.template.refined({"numeric_settings": {"dt": 0.000125}}))
        self.assertLess(np.max(np.abs(coarse.values[:, 0] - exp.reference[:, 0])), 0.001)
        self.assertLess(np.max(np.abs(coarse.values[:, 1] - exp.reference[:, 1])), 0.005)
        self.assertLess(np.linalg.norm(fine.values - exp.reference), np.linalg.norm(coarse.values - exp.reference))
        np.testing.assert_array_equal(np.array(env.control_history)[:, 0], np.array(refined.control_history)[:, 0])
        _, other = self.simulate(tuning, values={"kp": 4.0, "kd": 0.3})
        _, again = self.simulate(tuning)
        np.testing.assert_array_equal(again.values, coarse.values)
        self.assertGreater(np.max(np.abs(other.values - coarse.values)), 0.001)
        np.testing.assert_allclose(coarse.values[0], [0.2, -0.1], atol=1e-7)

    def test_native_readback_and_no_duplicate_drive(self):
        """Read actual MuJoCo coefficients rather than the Newton construction buffers."""
        tuning = self.make_case()
        env, _ = self.simulate(tuning)
        self.assertTrue(env.solver.use_mujoco_cpu)
        self.assertEqual(env.solver.mj_model.nu, 0)
        self.assertAlmostEqual(env.applied_parameters["inertia"], 0.01, places=7)
        self.assertAlmostEqual(env.applied_parameters["damping"], 0.02, places=7)
        env.model.joint_damping.assign(np.array([0.09], dtype=np.float32))
        self.assertAlmostEqual(env.applied_parameters["damping"], 0.02, places=7)
        env.solver.mj_model.dof_damping[0] = 0.07
        self.assertAlmostEqual(env.applied_parameters["damping"], 0.07, places=7)
        env.solver.mj_model.body_inertia[1, 2] = 0.03
        self.assertAlmostEqual(env.applied_parameters["inertia"], 0.03, places=7)

    def test_hold_saturation_and_failures(self):
        """Hold clipped torque across observations and classify deadlines and invalid candidates."""
        tuning = self.make_case()
        exp = tuning.evidence.train_data[0]
        settings = dict(tuning.template.experiment_settings[exp.id])
        settings.update(command_times=[0.0, 0.001, 0.2], command=[0.0, 100.0, 100.0])
        env = tuning.asset.create_environment(
            tuning.template, {**tuning.template.fixed_parameters, **tuning.asset.baseline_parameters}, settings
        )
        self.addCleanup(env.close)
        trace = env.run(np.array([0, 0.001, 0.002, 0.003, 0.004]), seed=0, deadline=None)
        self.assertEqual(env.control_history, [(0.0, 0.0), (0.002, 2.0)])
        self.assertEqual(trace.values[2, 0], 0.0)
        self.assertGreater(trace.values[-1, 0], 0.0)
        for label, values, deadline in (
            ("timeout", dict(tuning.asset.baseline_parameters), time.monotonic() - 1),
            ("nan", {"kp": float("nan"), "kd": 0.15}, None),
        ):
            episodes = tuning.rollout.execute_batch(
                tuning.template, tuning.asset, [(label, values)], (exp,), deadline=deadline
            )
            self.assertNotEqual(episodes[0].status, "completed")

    def test_export_replay_in_new_process(self):
        """Rebuild the selected solver from exported parameters and repeat validation."""
        tuning = self.make_case()
        _, trace = self.simulate(tuning)
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
        np.testing.assert_allclose(replay["experiments"]["capture-0"]["values"], trace.values, atol=1e-7, rtol=0)
        self.assertTrue(replay["validation"]["passed"])

    def test_native_warnings_cannot_be_scored_as_success(self):
        """Reject native divergence warnings even when MuJoCo leaves finite reset state."""
        tuning = self.make_case()
        exp = tuning.evidence.train_data[0]
        env = tuning.template.instantiate(tuning.asset, tuning.asset.baseline_parameters, exp)
        self.addCleanup(env.close)
        env.state_in.joint_qd.assign(np.array([1e20], dtype=np.float32))
        # Keep MuJoCo's diagnostic log inside the test's temporary directory.
        with chdir(self.tmp_path), self.assertRaisesRegex(SimulationError, "MuJoCo reported"):
            env.run(exp.times, seed=0, deadline=None)
        self.assertTrue(np.isfinite(env.solver.mj_data.qvel).all())
