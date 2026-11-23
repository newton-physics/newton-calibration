# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bridge imported native position actuators against direct MuJoCo execution."""

import shutil
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from examples.actuator.robot.asset import AssetRobot
from examples.actuator.robot.reference import synthetic_config, synthetic_records
from examples.actuator.robot.tuning import TuningRobot
from newton_calibration.tests.test_robot_calibration import BACKEND
from newton_calibration.tests.unittest_utils import TempPathTestCase

FIXTURE = Path(__file__).resolve().parents[2] / "examples/actuator/robot/fixtures/two_link_position.xml"


@unittest.skipUnless(BACKEND, "install the robot extra")
class TestRobotNative(TempPathTestCase):
    def make_case(self, *, space="joint_position", source=None):
        asset = AssetRobot.from_file(source or FIXTURE, bundle_dir=self.tmp_path / "input", gravity=[0, 0, 0])
        records = synthetic_records(duration=0.1)
        config = synthetic_config()
        config["controller"].update(mode="native_position", command_space=space)
        for name, kp, kd, gear in (("shoulder", 4.5, 0.5, -2), ("elbow", 2 / 3, 1 / 18, 3)):
            config["controller"]["drives"][name] = {"kp": kp, "kd": kd, "actuator": name + "_drive"}
            for field, value in (("kp", kp), ("kd", kd)):
                config["parameters"][f"{name}.{field}"] = {"baseline": value, "bounds": [value / 2, value * 2]}
            for record in records:
                # Include both ctrl-range and force-range saturation and an off-tick change.
                record["command"][name] = {"times": [0, 0.001, 0.05, 0.1], "values": [2, -0.4, 0.5, 0.5], "unit": "rad"}
                if space == "actuator_control":
                    signal = record["command"].pop(name)
                    signal["values"] = [v * gear for v in signal["values"]]
                    record["command"][name + "_drive"] = signal
        tuning = TuningRobot(
            asset, records, config, workspace=self.tmp_path / "runs", sources={"provenance": {"kind": "synthetic"}}
        )
        tuning.template.freeze()
        return tuning, asset

    def test_general_position_servo_and_non_pd_bias_rejection(self):
        """Accept the explicit general PD form and reject arbitrary affine force laws."""
        source = self.tmp_path / "source"
        shutil.copytree(FIXTURE.parent, source)
        path = source / FIXTURE.name
        tree = ET.parse(path)
        for actuator in tree.getroot().find("actuator"):
            kp, kd = float(actuator.attrib.pop("kp")), float(actuator.attrib.pop("kv"))
            actuator.tag = "general"
            actuator.attrib.update(gaintype="fixed", biastype="affine", gainprm=str(kp), biasprm=f"0 {-kp} {-kd}")
        tree.write(path)
        tuning, _ = self.make_case(source=path)
        exp = tuning.evidence.train_data[0]
        env = tuning.template.instantiate(tuning.asset, tuning.asset.baseline_parameters, exp)
        self.assertEqual(env.applied_parameters["shoulder.kp"], 4.5)
        self.assertTrue(np.isfinite(env.run(exp.times, seed=0, deadline=None).values).all())
        # A fresh import is required for edited source physics.
        tree.getroot().find("actuator")[0].set("biasprm", "1 -4.5 -0.5")
        tree.write(path)
        invalid = AssetRobot.from_file(path, bundle_dir=self.tmp_path / "invalid", gravity=[0, 0, 0])
        invalid = invalid.with_controller(
            dict(tuning.asset.config["controller"]), dict(tuning.asset.baseline_parameters)
        )
        with self.assertRaisesRegex(ValueError, "not a position PD"):
            tuning.template.instantiate(invalid, invalid.baseline_parameters, exp)

    def test_bridge_matches_direct_mujoco_with_gear_and_limits(self):
        """Match standalone MuJoCo at nonzero state, held controls and signed gears."""
        import mujoco  # noqa: PLC0415

        tuning, asset = self.make_case()
        exp = tuning.evidence.train_data[0]
        settings = tuning.template.experiment_settings[exp.id]
        env = tuning.template.instantiate(tuning.asset, tuning.asset.baseline_parameters, exp)
        actual = env.run(exp.times, seed=0, deadline=None)
        model = mujoco.MjModel.from_xml_path(str(asset._resource_root / asset.config["resources"]["entry"]))
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        model.opt.timestep = 0.00025
        data = mujoco.MjData(model)
        for name in ("shoulder", "elbow"):
            joint = model.joint(name)
            data.qpos[joint.qposadr[0]] = settings["initial_state"]["position"][name]
            data.qvel[joint.dofadr[0]] = settings["initial_state"]["velocity"][name]
        expected = []
        torque_seen = []
        step = 0
        for target in exp.times:
            while step * 0.00025 < target - 1e-12:
                if step % 8 == 0:
                    for name in ("shoulder", "elbow"):
                        signal = settings["command"][name]
                        index = np.searchsorted(signal["times"], step * 0.00025 + 1e-12, side="right") - 1
                        actuator = model.actuator(name + "_drive")
                        data.ctrl[actuator.id] = actuator.gear[0] * signal["values"][index]
                mujoco.mj_step(model, data)
                torque_seen.append(data.actuator_force.copy())
                step += 1
            expected.append(
                [
                    float(data.qpos[model.joint(name).qposadr[0]])
                    if kind == "position"
                    else float(data.qvel[model.joint(name).dofadr[0]])
                    for kind, name in settings["channels"]
                ]
            )
        np.testing.assert_allclose(actual.values, expected, atol=2e-5, rtol=0)
        self.assertEqual(env.history[0]["control"], [-4, 6])
        np.testing.assert_array_equal(env.control.joint_f.numpy(), [0, 0])
        self.assertAlmostEqual(np.max(np.abs(np.asarray(torque_seen)[:, 0])), 3)
        self.assertAlmostEqual(np.max(np.abs(np.asarray(torque_seen)[:, 1])), 0.5)

    def test_native_control_space_and_actual_coefficient_readback(self):
        """Use named native control channels and linked gain/bias coefficients."""
        tuning, _ = self.make_case(space="actuator_control")
        exp = tuning.evidence.train_data[0]
        env = tuning.template.instantiate(tuning.asset, tuning.asset.baseline_parameters, exp)
        trace = env.run(exp.times, seed=0, deadline=None)
        self.assertTrue(np.isfinite(trace.values).all())
        self.assertEqual(env.history[0]["control"], [-4, 6])
        aid = env.native_ids[0]
        mj = env.solver.mj_model
        self.assertEqual(mj.actuator_gainprm[aid, 0], 4.5)
        self.assertEqual(mj.actuator_biasprm[aid, 1], -4.5)
        mj.actuator_gainprm[aid, 0] = 3
        mj.actuator_biasprm[aid, 1] = -3
        self.assertEqual(env.applied_parameters["shoulder.kp"], 3)
        mj.actuator_biasprm[aid, 1] = -2
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            _ = env.applied_parameters

    def test_candidate_gains_drive_unsaturated_native_force(self):
        """Apply changed gains to actual native force without saturation hiding the binding."""
        tuning, _ = self.make_case()
        exp = tuning.evidence.train_data[0]
        values = {
            name: value * (1.4 if name.endswith(".kp") else 0.6)
            for name, value in tuning.asset.baseline_parameters.items()
        }
        env = tuning.template.instantiate(tuning.asset, values, exp)
        initial = env.settings["initial_state"]["position"]
        env.settings = {
            **env.settings,
            "command": {name: {"times": [0.0, 0.1], "values": [initial[name] + 0.01] * 2} for name in env.names},
        }
        env.run([0.0, env.dt], seed=0, deadline=None)
        data, model = env.solver.mj_data, env.solver.mj_model
        for name, aid in zip(env.names, env.native_ids, strict=True):
            expected = (
                values[f"{name}.kp"] * (data.ctrl[aid] - data.actuator_length[aid])
                - values[f"{name}.kd"] * data.actuator_velocity[aid]
            )
            self.assertLess(abs(expected), model.actuator_forcerange[aid, 1])
            self.assertAlmostEqual(data.actuator_force[aid], expected, places=7)
            self.assertEqual(env.applied_parameters[f"{name}.kp"], values[f"{name}.kp"])

    def test_native_parameter_reload_and_duplicate_binding_rejection(self):
        """Rebuild native gains from a candidate and reject duplicated actuator routing."""
        tuning, _ = self.make_case()
        values = {name: value * 1.1 for name, value in tuning.asset.baseline_parameters.items()}
        candidate = tuning.asset.export(values, self.tmp_path / "candidate/asset.json")
        loaded = AssetRobot.from_export(candidate)
        exp = tuning.evidence.train_data[0]
        env = tuning.template.instantiate(loaded, values, exp)
        for name, value in values.items():
            self.assertAlmostEqual(env.applied_parameters[name], value)
        controller = synthetic_config()["controller"]
        # Construct a separate invalid routing without mutating frozen configuration.
        controller.update(mode="native_position", command_space="joint_position")
        controller["drives"] = {
            name: {"kp": 1, "kd": 0.1, "actuator": "shoulder_drive"} for name in ("shoulder", "elbow")
        }
        broken = loaded.with_controller(controller, values)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            tuning.template.instantiate(broken, values, exp)
