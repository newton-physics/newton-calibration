# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in tests of real CUDA robot worlds, not CPU stand-ins."""

import json
import os
import subprocess
import sys
import time
import unittest

import numpy as np

from examples.actuator.robot.asset import AssetRobot
from examples.actuator.robot.reference import synthetic_config, synthetic_records
from examples.actuator.robot.tuning import TuningRobot
from newton_calibration import Rollout, SimulationError
from newton_calibration.tests.test_robot_calibration import BACKEND, FIXTURE
from newton_calibration.tests.unittest_utils import TempPathTestCase


@unittest.skipUnless(BACKEND and os.environ.get("ROBOT_GPU_TESTS") == "1", "opt-in CUDA robot integration")
class TestRobotGPU(TempPathTestCase):
    def make_case(self, *, native=False):
        source = FIXTURE.with_name("two_link_position.xml") if native else FIXTURE
        asset = AssetRobot.from_file(source, bundle_dir=self.tmp_path / "bundle", gravity=[0, 0, 0])
        config = synthetic_config()
        if native:
            config["controller"].update(mode="native_position", command_space="joint_position")
            for name, kp, kd in (("shoulder", 4.5, 0.5), ("elbow", 2 / 3, 1 / 18)):
                config["controller"]["drives"][name] = {"kp": kp, "kd": kd, "actuator": name + "_drive"}
                for field, value in (("kp", kp), ("kd", kd)):
                    config["parameters"][f"{name}.{field}"] = {"baseline": value, "bounds": [value / 2, value * 2]}
        tuning = TuningRobot(
            asset, synthetic_records(duration=0.04), config, workspace=self.tmp_path / "runs", sources={}
        )
        tuning.template.freeze()
        gpu = tuning.asset.with_device("cuda:0")
        template = tuning.template.refined({"numeric_settings": {"device": "cuda:0"}})
        a = dict(tuning.asset.baseline_parameters)
        b = {name: value * (1.4 if name.endswith(".kp") else 0.6) for name, value in a.items()}
        return tuning, gpu, template, [a, b, a]

    def test_native_batch_matches_cpu_and_isolates_candidates(self):
        """Match signed-gear native CPU dynamics with distinct per-world gains and A/B/A state."""
        self.check_parity(native=True)

    def test_equivalent_pd_matches_cpu_and_isolates_candidates(self):
        """Match held-torque CPU dynamics and retain independent GPU PD gains."""
        self.check_parity(native=False)

    def check_parity(self, *, native):
        tuning, asset, template, candidates = self.make_case(native=native)
        exp = tuning.evidence.train_data[0]
        batch = template.instantiate_batch(asset, candidates, exp)
        self.addCleanup(batch.close)
        self.assertTrue(batch.model.device.is_cuda)
        self.assertFalse(batch.solver.use_mujoco_cpu)
        self.assertEqual(batch.solver.mjw_data.nworld, 3)
        actual = batch.run(exp.times, seed=0, deadline=None)
        np.testing.assert_array_equal(actual[0].values, actual[2].values)
        self.assertGreater(np.max(np.abs(actual[0].values - actual[1].values)), 1e-4)
        for i, values in enumerate(candidates):
            cpu = tuning.template.instantiate(tuning.asset, values, exp)
            expected = cpu.run(exp.times, seed=0, deadline=None)
            np.testing.assert_allclose(actual[i].values, expected.values, atol=3e-5, rtol=0)
            np.testing.assert_allclose(actual[i].positions, expected.positions, atol=3e-5, rtol=0)
            for name, value in values.items():
                np.testing.assert_allclose(batch.applied_parameters[i][name], value, rtol=1e-6, atol=1e-12)
        single = template.instantiate(asset, candidates[0], exp)
        self.addCleanup(single.close)
        np.testing.assert_allclose(
            single.run(exp.times, seed=0, deadline=None).values, actual[0].values, atol=1e-6, rtol=0
        )
        with self.assertRaisesRegex(RuntimeError, "fresh"):
            batch.run(exp.times, seed=0, deadline=None)

    def test_rollout_order_chunking_and_reload(self):
        """Preserve trial identity through partial chunks and rebuild a GPU candidate."""
        tuning, asset, template, candidates = self.make_case()
        exp = tuning.evidence.train_data[0]
        trials = [("a", candidates[0]), ("invalid", None), ("b", candidates[1]), ("again", candidates[2])]
        rollout = Rollout(worlds=2, parameter_rtol=1e-6)
        self.addCleanup(rollout.close)
        episodes = rollout.execute_batch(template, asset, trials, [exp])
        self.assertEqual([e.trial_id for e in episodes], [t[0] for t in trials])
        self.assertEqual([e.status for e in episodes], ["completed", "invalid_parameters", "completed", "completed"])
        np.testing.assert_allclose(episodes[0].trace.values, episodes[3].trace.values, atol=1e-6, rtol=0)
        path = asset.export(candidates[1], self.tmp_path / "candidate/asset.json")
        loaded = AssetRobot.from_export(path)
        self.assertTrue(loaded.supports_batch)
        replay = template.instantiate(loaded, loaded.baseline_parameters, exp)
        self.addCleanup(replay.close)
        np.testing.assert_allclose(
            replay.run(exp.times, seed=0, deadline=None).values, episodes[2].trace.values, atol=1e-6, rtol=0
        )

    def test_failure_and_deadline(self):
        """Fail a NaN world alone and reject execution after the deadline."""
        tuning, asset, template, candidates = self.make_case()
        exp = tuning.evidence.train_data[0]
        batch = template.instantiate_batch(asset, candidates, exp)
        self.addCleanup(batch.close)
        q = batch.state_in.joint_q.numpy()
        q[batch.nq] = np.nan
        batch.state_in.joint_q.assign(q)
        results = batch.run(exp.times, seed=0, deadline=None)
        self.assertIsInstance(results[1], SimulationError)
        self.assertNotIsInstance(results[0], SimulationError)
        self.assertNotIsInstance(results[2], SimulationError)
        timeout = Rollout(worlds=128).execute_batch(
            template, asset, [("a", candidates[0])], [exp], deadline=time.monotonic() - 1
        )
        self.assertEqual(timeout[0].status, "timeout")

    def test_saturation_off_tick_commands_and_refinement(self):
        """Keep controller ticks and saturated held torque unchanged by physics refinement."""
        tuning, asset, template, candidates = self.make_case()
        exp = tuning.evidence.train_data[0]
        settings = template.snapshot()["experiment_settings"][exp.id]
        for name, sign in (("shoulder", 1), ("elbow", -1)):
            settings["command"][name] = {"times": [0, 0.001, 0.04], "values": [100 * sign, -100 * sign, -100 * sign]}
        parameters = [{**template.fixed_parameters, **p} for p in candidates]
        histories = []
        for dt in (0.00025, 0.000125):
            selected = template.refined({"numeric_settings": {"dt": dt}})
            env = asset.create_batch_environment(selected, parameters, settings)
            self.addCleanup(env.close)
            env.run([0, 0.0008, 0.0013], seed=0, deadline=None)
            np.testing.assert_array_equal(env.control.joint_f.numpy().reshape(3, 2), [[25, -8]] * 3)
            histories.append(env.history)
        self.assertEqual(histories[0], histories[1])
        env = asset.create_batch_environment(template, parameters, settings)
        self.addCleanup(env.close)
        env.run([0, 0.001, 0.002, 0.003], seed=0, deadline=None)
        np.testing.assert_array_equal(env.control.joint_f.numpy().reshape(3, 2), [[-25, 8]] * 3)
        self.assertEqual([h["time"] for h in env.history], [0, 0.002])

    def test_native_limits_readback_and_new_process(self):
        """Apply native control/force limits and replay exported GPU gains in a fresh process."""
        tuning, asset, template, candidates = self.make_case(native=True)
        exp = tuning.evidence.train_data[0]
        settings = template.snapshot()["experiment_settings"][exp.id]
        for name in ("shoulder", "elbow"):
            settings["command"][name] = {"times": [0, 0.04], "values": [100, 100]}
        env = asset.create_batch_environment(
            template, [{**template.fixed_parameters, **p} for p in candidates], settings
        )
        self.addCleanup(env.close)
        env.run([0, 0.00025], seed=0, deadline=None)
        np.testing.assert_allclose(np.abs(env.solver.mjw_data.actuator_force.numpy()), [[3, 0.5]] * 3, atol=1e-6)
        gain = env.solver.mjw_model.actuator_gainprm.numpy()
        gain[1, env.native_ids[0], 0] = 123
        env.solver.mjw_model.actuator_gainprm.assign(gain)
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            _ = env.applied_parameters
        path = asset.export(candidates[1], self.tmp_path / "delivery/asset.json")
        inputs = self.tmp_path / "replay-input.json"
        output = self.tmp_path / "replay-output.npy"
        inputs.write_text(
            json.dumps({"template": template.snapshot(), "settings": settings, "times": [0, 0.002, 0.004]})
        )
        code = """
import json, sys, numpy as np
from examples.actuator.robot.asset import AssetRobot
from newton_calibration import Template
asset = AssetRobot.from_export(sys.argv[1])
inputs = json.load(open(sys.argv[2]))
template = Template(**inputs['template']).freeze()
env = asset.create_environment(template, {**template.fixed_parameters, **asset.baseline_parameters}, inputs['settings'])
np.save(sys.argv[3], env.run(inputs['times'], seed=0, deadline=None).values)
env.close()
"""
        subprocess.run(
            [sys.executable, "-c", code, str(path), str(inputs), str(output)],
            check=True,
            capture_output=True,
            timeout=120,
        )
        local = asset.create_environment(template, {**template.fixed_parameters, **candidates[1]}, settings)
        self.addCleanup(local.close)
        np.testing.assert_allclose(
            np.load(output), local.run([0, 0.002, 0.004], seed=0, deadline=None).values, atol=1e-6, rtol=0
        )

    def test_gpu_uses_core_qualification_and_acceptance(self):
        """Complete the inherited lifecycle and independently validate a GPU candidate."""
        tuning, _, _, _ = self.make_case()
        config = synthetic_config()
        for name, value in {"shoulder.kp": 18.0, "shoulder.kd": 2.0, "elbow.kp": 6.0, "elbow.kd": 0.5}.items():
            joint, field = name.split(".")
            config["controller"]["drives"][joint][field] = value
            config["parameters"][name]["baseline"] = value
        config["parameters"] = {"shoulder.kp": config["parameters"]["shoulder.kp"]}
        config["execution"] = {"device": "cuda:0", "worlds": 2, "batch_size": 4}
        config["decision"].update(target=1.0, max_trials=4)
        gpu = TuningRobot(
            tuning.asset,
            synthetic_records(duration=0.04),
            config,
            workspace=self.tmp_path / "gpu-run",
            sources={"provenance": {"kind": "synthetic"}},
        )
        self.assertEqual(gpu.rollout.worlds, 2)
        result = gpu.run()
        self.assertEqual(result.acceptance, "accepted", gpu.state.errors)
        self.assertIsNotNone(result.asset_path)
        self.assertEqual(AssetRobot.from_export(result.asset_path).config["device"], "cuda:0")
