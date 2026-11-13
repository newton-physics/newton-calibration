# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the actuator example: configuration, data alignment, batching and objective scaling.

These tests do not import Newton or Warp and do not simulate.
"""

import copy
import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

from examples.actuator_calibration import config as actuator_config
from examples.actuator_calibration.asset import _GroupedWorlds, _RecordingGroup, _Worlds, recording_key
from examples.actuator_calibration.config import (
    build_case,
    joint_values,
    load_config,
    make_search,
    make_template,
    precision_tolerance,
)
from examples.actuator_calibration.data import experiment
from examples.actuator_calibration.run import DEFAULT_CONFIG
from newton_calibration import Evaluator, Evidence, Trace
from newton_calibration._src.data import Episode
from newton_calibration.tests.unittest_utils import TempPathTestCase

HAS_PANDAS = importlib.util.find_spec("pandas") is not None
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
HAS_PXR = importlib.util.find_spec("pxr") is not None


class _EchoPopulation:
    """Records assignments; each world returns its stiffness as a constant trace."""

    size = 4

    def __init__(self):
        self.assigned = []

    def assign(self, parameter_values):
        self.assigned.append([dict(v) for v in parameter_values])
        return [dict(v) for v in parameter_values]

    def run(self, times, *, deadline, inputs=None):
        return [Trace(np.asarray(times), np.full((len(times), 1), v["j.stiffness"])) for v in self.assigned[-1]]


class _RecordingPopulation(_EchoPopulation):
    """World w returns 10 * stiffness + its recording's command over that recording's times."""

    def __init__(self, size):
        super().__init__()
        self.size, self.passes = size, 0

    def run_recordings(self, recordings, world_recording, *, deadline):
        self.passes += 1
        traces = []
        for values, index in zip(self.assigned[-1], world_recording, strict=True):
            times, inputs = recordings[index]
            level = 10 * values["j.stiffness"] + float(inputs["command_q"][0, 0])
            traces.append(Trace(np.asarray(times), np.full((len(times), 1), level)))
        return traces


class TestActuatorConfiguration(unittest.TestCase):
    def setUp(self):
        self.so101 = load_config(DEFAULT_CONFIG)

    def test_so101_case_has_independent_parameters_per_joint(self):
        """Give every SO-101 joint its own five actuator parameters."""
        case = build_case(self.so101)
        self.assertEqual(case.joints, ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"))
        self.assertEqual(len(case.parameters.names), 30)
        self.assertIn("Elbow.stiffness", case.parameters.names)
        definitions = {d.name: d for d in case.parameters.definitions}
        self.assertEqual(definitions["Elbow.stiffness"].transform, "log")
        self.assertEqual(definitions["Elbow.dry_friction"].transform, "linear")
        self.assertEqual(case.fixed_parameters["Jaw.effort_limit"], 3.35)
        self.assertEqual(case.fixed_parameters["Jaw.velocity_limit"], 30.0)
        self.assertEqual(case.fixed_parameters["Jaw.delay_steps"], 0.0)
        self.assertEqual(case.buffer_steps, 10)
        values = joint_values(case.parameters.as_dict(case.parameters.baseline), case.joints)
        self.assertEqual(
            values["Pitch"],
            {"stiffness": 40.0, "damping": 2.0, "dry_friction": 0.25, "viscous_friction": 0.75, "armature": 0.05},
        )

    def test_equal_bounds_become_fixed_and_delay_is_in_control_steps(self):
        """Fix parameters with equal bounds and convert delays to control steps."""
        config = copy.deepcopy(self.so101)
        group = config["joint_groups"][0]
        group["parameters"]["armature"] = {"bounds": [0.02, 0.02], "baseline": 0.02}
        group["delay_ms"] = 40
        case = build_case(config)
        self.assertEqual(len(case.parameters.names), 24)
        self.assertEqual(case.fixed_parameters["Elbow.armature"], 0.02)
        self.assertEqual(case.fixed_parameters["Elbow.delay_steps"], 2.0)

    def test_invalid_configurations_are_rejected(self):
        """Reject inconsistent schemas, rates, joints and bounds."""
        cases = [
            (lambda c: c["simulation"].update(physics_freq=275), "integer multiple"),
            (lambda c: c["dataset"].update(dt=0.01), "control timestep"),
            (lambda c: c["joint_groups"].append(copy.deepcopy(c["joint_groups"][0])), "more than"),
            (lambda c: c["joint_groups"][0]["parameters"]["armature"].update(bounds=[-1, 1]), "nonnegative"),
        ]
        for change, message in cases:
            with self.subTest(message=message):
                config = copy.deepcopy(self.so101)
                change(config)
                with self.assertRaisesRegex(ValueError, message):
                    build_case(config)

    def test_template_declares_refinements_and_pointwise_tolerance(self):
        """Declare refinements and express the pointwise tolerance in feature units."""
        case = build_case(self.so101)
        template = make_template(case, ["a", "b"], scored_samples=6000)
        template.freeze()
        self.assertEqual(template.numeric_settings["physics_freq"], 250)
        self.assertEqual(template.discretization["control_freq"], 50)
        refined = template.refined(template.refinements[0])
        self.assertEqual(refined.numeric_settings["physics_freq"], 500)
        self.assertEqual(template.experiment_settings["a"], {"seed": 0, "buffer_steps": 10})
        self.assertTrue(math.isclose(template.precision_tolerance * math.sqrt(6000 * 6), 0.02))
        self.assertEqual(precision_tolerance(case, 6000), template.precision_tolerance)

    def test_search_budget_covers_every_phase(self):
        """Budget rollouts for every lifecycle phase of the search."""
        case = build_case(self.so101)
        optimizer, decision = make_search(case, train_count=6, validation_count=8)
        self.assertEqual((optimizer.batch_size, optimizer.sigma), (256, 0.125))
        self.assertEqual((decision.max_generations, decision.max_trials), (100, 25600))
        templates = 3
        per_replay = 1 + 3 * templates + 6 * 30 + 25600 + templates
        self.assertEqual(decision.max_rollouts, 6 * per_replay + 8)
        self.assertEqual(decision.relative_improvement_tolerance, 0.001)


class TestActuatorData(TempPathTestCase):
    def test_recording_becomes_shifted_masked_experiment(self):
        """Shift the reference one interval and mask the settling buffer."""
        command = np.arange(12, dtype=np.float64).reshape(6, 2)
        measured = command + 0.5
        exp = experiment("rec", command, measured, dt=0.02, buffer_steps=2, group="file").normalized()
        np.testing.assert_array_equal(exp.reference, measured[1:])
        np.testing.assert_allclose(exp.times, [0.02, 0.04, 0.06, 0.08, 0.1])
        self.assertEqual(exp.mask[:2].sum(), 0)
        self.assertTrue(exp.mask[2:].all())
        self.assertEqual((exp.weight, exp.group), (3.0, "file"))
        np.testing.assert_array_equal(exp.inputs["command_q"], command)
        np.testing.assert_array_equal(exp.inputs["initial_q"], measured[0])
        with self.assertRaisesRegex(ValueError, "no scored samples"):
            experiment("short", command, measured, dt=0.02, buffer_steps=5)

    def test_weighted_objective_equals_pooled_position_mse(self):
        """Scored-sample weights reproduce the pooled MSE over all recordings."""
        rng = np.random.default_rng(3)
        lengths, buffer, joints, worlds = (40, 25, 61), 4, 3, 5
        experiments, pooled_sum, pooled_count = [], np.zeros(worlds), 0
        predictions = {}
        for k, length in enumerate(lengths):
            command = rng.normal(size=(length, joints))
            measured = rng.normal(size=(length, joints))
            exp = experiment(f"r{k}", command, measured, dt=0.02, buffer_steps=buffer)
            experiments.append(exp)
            simulated = measured[1:][None] + rng.normal(scale=0.1, size=(worlds, length - 1, joints))
            predictions[exp.id] = simulated
            pooled_sum += np.square(simulated[:, buffer:] - measured[1 + buffer :]).sum(axis=(1, 2))
            pooled_count += (length - 1 - buffer) * joints
        evidence = Evidence(experiments)
        evidence.prepare()
        episodes = [
            Episode(f"t{w}", e.id, 0, "completed", Trace(e.times, predictions[e.id][w]))
            for w in range(worlds)
            for e in evidence.train_data
        ]
        losses, *_ = Evaluator().score_batch(episodes, tuple(f"t{w}" for w in range(worlds)), evidence.train_data)
        np.testing.assert_allclose(losses, pooled_sum / pooled_count, rtol=1e-12)

    @unittest.skipUnless(HAS_PANDAS, "pandas is not installed")
    def test_resampling_uses_the_overlap_of_all_fields(self):
        """Resample every field over the time range all fields share."""

        import pandas as pd  # noqa: PLC0415 (optional actuator extra)

        from examples.actuator_calibration.data import resample  # noqa: PLC0415

        rows = []
        # Field b starts later and ends earlier: the grid spans only the overlap of all fields.
        for t, value in ((0, 0.0), (10, 1.0), (20, 2.0), (30, 3.0)):
            rows.append((t, "exp", "j/command_q", value))
        for t, value in ((5, 10.0), (25, 30.0)):
            rows.append((t, "exp", "j/actual_q", value))
        frame = pd.DataFrame(rows, columns=["time_ns", "experiment", "field", "value"])
        result = resample(frame, 5e-9, {"j": "Joint"})
        times, values = result["exp"]
        np.testing.assert_array_equal(times, [5, 10, 15, 20, 25])
        np.testing.assert_allclose(values["Joint/command_q"], [0.5, 1.0, 1.5, 2.0, 2.5])
        np.testing.assert_allclose(values["Joint/actual_q"], [10, 15, 20, 25, 30])
        with self.assertRaisesRegex(ValueError, "whole number"):
            resample(frame, 1e-10)
        duplicated = pd.concat([frame, frame.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resample(duplicated, 5e-9)
        self.assertTrue(actuator_config.QUANTITIES)  # module imports without Newton

    @unittest.skipUnless(HAS_PANDAS and HAS_PYARROW, "pandas and pyarrow are not installed")
    def test_trajectories_require_command_and_measurement_fields(self):
        """Require commanded and measured positions for every joint."""

        import pandas as pd  # noqa: PLC0415 (optional actuator extra)

        from examples.actuator_calibration.data import trajectories  # noqa: PLC0415

        frame = pd.DataFrame(
            [(t, "e", "a/command_q", 1.0) for t in (0, 10)] + [(t, "e", "a/actual_q", 2.0) for t in (0, 10)],
            columns=["time_ns", "experiment", "field", "value"],
        )
        path = self.tmp_path / "rec.parquet"
        frame.to_parquet(path)
        [(name, command, measured)] = trajectories(path, ["a"], 1e-8)
        self.assertEqual(name, "e")
        self.assertEqual(command.shape, (2, 1))
        self.assertEqual(measured.shape, (2, 1))
        with self.assertRaisesRegex(ValueError, "missing fields"):
            trajectories(path, ["b"], 1e-8)

    def test_robot_usd_comes_from_the_dataset_or_a_local_file_with_the_same_hash(self):
        """Fetch the robot USD from the dataset or accept a local file with its hash."""

        from examples.actuator_calibration.data import resolve_asset  # noqa: PLC0415
        from newton_calibration.sources import file_sha256  # noqa: PLC0415

        source = self.tmp_path / "robot_assets" / "robot.usda"
        source.parent.mkdir()
        source.write_text("#usda 1.0\n")
        config = {
            "asset": {"path": "robot_assets/robot.usda", "usd_sha256": file_sha256(source)},
            "dataset": {"provider": "huggingface"},
        }
        self.assertEqual(resolve_asset(config, data_root=self.tmp_path), source.absolute())
        self.assertEqual(resolve_asset(config, source), source.absolute())
        other = self.tmp_path / "other.usda"
        other.write_text("#usda 1.0\n# changed\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            resolve_asset(config, other)
        with self.assertRaisesRegex(ValueError, "names no dataset asset"):
            resolve_asset({"asset": {"usd_sha256": "0"}, "dataset": {}})


class TestActuatorBatching(unittest.TestCase):
    def test_partial_chunk_pads_unused_worlds_and_returns_only_its_members(self):
        """Pad unused worlds in a partial chunk and return only its members."""
        population = _EchoPopulation()
        values = [{"j.stiffness": 1.0}, {"j.stiffness": 2.0}]
        worlds = _Worlds(population, values, inputs={"command_q": np.zeros(3)})
        self.assertEqual([v["j.stiffness"] for v in population.assigned[-1]], [1.0, 2.0, 2.0, 2.0])
        self.assertEqual(list(worlds.applied_parameters), values)
        traces = worlds.run(np.array([0.1, 0.2]), seed=0, deadline=None)
        self.assertEqual([float(t.values[0, 0]) for t in traces], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "does not fit"):
            _Worlds(population, values * 3, inputs=None)

    def test_recordings_of_a_chunk_replay_together_once(self):
        """Replay every grouped recording in one pass and serve each recording from it."""
        population = _RecordingPopulation(size=2 * 3)
        recordings = {
            f"r{r}": (
                0.1 * np.arange(1, steps + 1),
                {"command_q": np.full((steps + 1, 1), r), "initial_q": np.zeros(1)},
            )
            for r, steps in enumerate((2, 4))
        }
        group = _RecordingGroup(population, recordings, chunk=3)
        values = [{"j.stiffness": 1.0}, {"j.stiffness": 2.0}]
        environments = {name: _GroupedWorlds(group, values, name) for name in recordings}
        self.assertEqual(
            [v["j.stiffness"] for v in population.assigned[-1]],
            [1.0, 2.0, 2.0, 1.0, 2.0, 2.0],
        )
        for name, (times, _) in recordings.items():
            traces = environments[name].run(times, seed=0, deadline=None)
            self.assertEqual(len(traces), 2)
            for trace, expected in zip(traces, values, strict=True):
                np.testing.assert_array_equal(trace.times, times)
                self.assertEqual(float(trace.values[0, 0]), 10 * expected["j.stiffness"] + int(name[1]))
        self.assertEqual(population.passes, 1)
        with self.assertRaisesRegex(ValueError, "control grid"):
            environments["r0"].run(np.array([0.2, 0.4]), seed=0, deadline=None)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            _GroupedWorlds(group, values * 2, "r0")
        self.assertEqual(recording_key(recordings["r0"][1]), recording_key(dict(recordings["r0"][1])))
        self.assertNotEqual(recording_key(recordings["r0"][1]), recording_key(recordings["r1"][1]))


@unittest.skipUnless(HAS_PXR, "usd-core is not installed")
class TestIdentifiedLayer(TempPathTestCase):
    def test_identified_layer_authors_usd_units_and_relocates(self):
        """Author identified values in USD units and relocate the layer on release."""

        from pxr import Sdf, Usd, UsdPhysics

        from examples.actuator_calibration.usd_layer import (  # noqa: PLC0415 (needs OpenUSD)
            relocate_layer,
            sublayer_source,
            write_identified_layer,
        )

        source = self.tmp_path / "robot" / "robot.usda"
        source.parent.mkdir()
        stage = Usd.Stage.CreateNew(str(source))
        root = stage.DefinePrim("/robot", "Xform")
        stage.SetDefaultPrim(root)
        UsdPhysics.RevoluteJoint.Define(stage, "/robot/elbow")
        stage.GetRootLayer().Save()

        parameters = {
            "elbow": {
                "stiffness": 10.0,
                "damping": 0.5,
                "armature": 0.01,
                "dry_friction": 0.2,
                "viscous_friction": 0.3,
            }
        }
        clamps = {"elbow": {"effort_limit": 3.0, "velocity_limit": 2.0}}
        layer = write_identified_layer(source, self.tmp_path / "run" / "candidate.usda", parameters, clamps, "{}")
        self.assertEqual(sublayer_source(layer), source.absolute())
        written = Sdf.Layer.FindOrOpen(str(layer))
        self.assertFalse(Path(written.subLayerPaths[0]).is_absolute())

        identified = Usd.Stage.Open(str(layer))
        joint = identified.GetPrimAtPath("/robot/elbow")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        per_degree = math.pi / 180
        self.assertAlmostEqual(drive.GetStiffnessAttr().Get(), 10.0 * per_degree, places=6)
        self.assertAlmostEqual(drive.GetDampingAttr().Get(), 0.5 * per_degree, places=6)
        self.assertAlmostEqual(drive.GetMaxForceAttr().Get(), 3.0, places=6)
        self.assertTrue(math.isclose(joint.GetAttribute("newton:velocityLimit").Get(), 2.0 / per_degree, rel_tol=1e-6))
        self.assertAlmostEqual(joint.GetAttribute("newton:damping").Get(), 0.3 * per_degree, places=6)

        released = relocate_layer(layer, self.tmp_path / "run" / "release" / "asset.usda")
        self.assertEqual(sublayer_source(released), source.absolute())
        relocated = Usd.Stage.Open(str(released))
        reopened = relocated.GetPrimAtPath("/robot/elbow")
        self.assertEqual(
            UsdPhysics.DriveAPI.Get(reopened, "angular").GetStiffnessAttr().Get(),
            drive.GetStiffnessAttr().Get(),
        )

        with self.assertRaisesRegex(ValueError, "exactly match"):
            write_identified_layer(source, self.tmp_path / "bad.usda", parameters, {}, "{}")


if __name__ == "__main__":
    unittest.main()
