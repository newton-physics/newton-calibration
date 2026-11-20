# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The driven cable case: the clamp follows the recorded TCP, batched equals single.

The tests write small recordings with drawn masks, so they need no recorded data. They
simulate with Newton on a CUDA device and are skipped without one.
"""

import importlib.util
import json
import math
import unittest

import numpy as np

from newton_calibration.tests.unittest_utils import TempPathTestCase

# The grasp point of the test recordings [m], in the drive frame (+Z up).
GRASP = (0.0, 0.0, 0.6)
# A 200 x 200 px camera 1.2 m in front of the grasp point, looking along +Y.
CAMERA = {
    "position_m": [0.0, -1.2, 0.4],
    "quaternion_xyzw": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
    "intrinsics": {"fx": 150.0, "fy": 150.0, "cx": 100.0, "cy": 100.0},
}
OBSERVATION_TIMES = [0.1, 0.2, 0.3]


def _tcp_rotation(angle):
    """TCP rotation with the cable (attachment +Z) hanging down, tilted by ``angle`` [rad] about +Y."""
    import warp as wp

    down = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi)
    return wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle)), down)


def _write_recording(directory, name, tilt_rad):
    """Write a recording whose TCP tilts by ``tilt_rad`` over 0.4 s at a fixed grasp point.

    Each mask is the straight cable along the TCP's +Z at that observation, drawn 6 px wide.
    """
    import warp as wp
    from PIL import Image, ImageDraw

    from examples.cable_driven.chamfer import Camera  # noqa: PLC0415
    from examples.driven_attachment import drive_pose  # noqa: PLC0415

    times = np.linspace(0.0, 0.4, 9)
    drive = {
        "times_s": times.tolist(),
        "positions_m": [list(GRASP)] * len(times),
        "quaternions_xyzw": [list(_tcp_rotation(tilt_rad * t / 0.4)) for t in times],
    }
    camera = Camera(CAMERA)
    (directory / "masks").mkdir(parents=True)
    masks = []
    for k, t in enumerate(OBSERVATION_TIMES):
        position, rotation = drive_pose(drive, t)
        axis = np.array(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)))
        cable = np.array(position) + np.linspace(0.0, 0.48, 2)[:, None] * axis
        image = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(image).line([tuple(p) for p in camera.project(cable)], fill=255, width=6)
        masks.append(f"masks/{k:05d}.png")
        image.save(directory / masks[-1])
    recording = {
        "experiment": {"id": name, "times": OBSERVATION_TIMES},
        "settings": {
            "fps": 60,
            "attachment": {"transform": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], "clamp_position_m": 0.02},
        },
        "drive": drive,
        "camera": CAMERA,
        "images": {"masks": masks},
    }
    (directory / "recording.json").write_text(json.dumps(recording), encoding="utf-8")


def _short_case(tmp_path):
    """The case on two written recordings, ``tilt`` for training and ``hold`` for validation, with a short settle."""
    from examples.cable_driven.run import CONFIG, build_tuning  # noqa: PLC0415

    root = tmp_path / "recordings"
    _write_recording(root / "tilt", "tilt", 1.0)
    _write_recording(root / "hold", "hold", 0.0)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["numeric_settings"]["settle_frames"] = 20
    config["numeric_settings"]["settle_tolerance_m"] = 1.0
    config["train"] = ["tilt"]
    config["validation"] = ["hold"]
    config_path = tmp_path / "cable.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    tuning = build_tuning(config_path, recordings_root=root, workspace=tmp_path / "ws")
    tuning.template.freeze()
    return config, tuning


class TestCableDriven(TempPathTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if importlib.util.find_spec("newton") is None or importlib.util.find_spec("warp") is None:
            raise unittest.SkipTest("newton and warp are not installed")
        import warp as wp

        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA unavailable")

    def test_batched_members_equal_single_members(self):
        """Three candidates in one model give the same traces as three single-member runs."""
        _, tuning = _short_case(self.tmp_path)
        experiments = list(tuning.evidence.train_data)
        baseline = tuning.parameters.as_dict(tuning.parameters.baseline)
        trials = [(f"t{i}", {**baseline, "bending_rigidity": ei}) for i, ei in enumerate((0.1, 0.5, 2.0))]
        # The middle candidate has a curved rest shape, so its build and readback are covered too.
        trials[1][1].update(
            {p.name: 0.5 * p.bounds[1] for p in tuning.parameters.definitions if p.name.startswith("rest_")}
        )
        batched = tuning.rollout.execute_batch(tuning.template, tuning.asset, trials, experiments)
        # One episode per (trial, experiment), in the same order as the batched result.
        singles = [
            episode
            for trial in trials
            for episode in tuning.rollout.execute_batch(tuning.template, tuning.asset, [trial], experiments)
        ]
        for b, s in zip(batched, singles, strict=True):
            self.assertEqual(b.key, s.key)
            self.assertEqual(b.status, "completed", b.error)
            self.assertEqual(s.status, "completed", s.error)
            self.assertEqual(b.applied_parameters, s.applied_parameters)
            np.testing.assert_array_equal(b.trace.values, s.trace.values)
        # The stiffness matters once the tilt bends the cable: the three scores differ.
        last = [e.trace.values[-1] for e in batched if e.experiment_id == "tilt"]
        self.assertFalse(np.allclose(last[0], last[1]))
        self.assertFalse(np.allclose(last[1], last[2]))

    def test_clamp_follows_the_trajectory(self):
        """After a run, the clamp capsule sits at the last TCP pose composed with the attachment."""
        import warp as wp

        from examples.driven_attachment import drive_pose  # noqa: PLC0415

        _, tuning = _short_case(self.tmp_path)
        experiment = tuning.evidence.train_data[0]
        settings = tuning.template.experiment_settings[experiment.id]
        parameters = tuning.parameters.as_dict(tuning.parameters.baseline)
        environment = tuning.template.instantiate_batch(tuning.asset, [parameters], experiment)
        try:
            environment.run(experiment.times.copy(), seed=0, deadline=None)
            pose = environment.state_0.body_q.numpy()[environment.driven_bodies.numpy()[0]]
        finally:
            environment.close()
        position, rotation = drive_pose(settings["drive"], float(experiment.times[-1]))
        expected = np.array(position) + np.array(wp.quat_rotate(rotation, environment.local_position))
        np.testing.assert_allclose(pose[:3], expected, atol=1e-6)
        expected_rotation = np.array(wp.mul(rotation, environment.local_rotation))
        self.assertLess(
            min(np.abs(pose[3:] - expected_rotation).max(), np.abs(pose[3:] + expected_rotation).max()), 1e-6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
