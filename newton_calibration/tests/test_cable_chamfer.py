# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The chamfer score of the driven cable case, against a mask drawn from a known projection."""

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("scipy") and importlib.util.find_spec("PIL"), "scipy and pillow are not installed"
)
class TestCableChamfer(unittest.TestCase):
    def test_score_is_smallest_at_the_true_cable(self):
        from examples.cable_driven.chamfer import MISS, Camera, MaskTarget, chamfer  # noqa: PLC0415

        # Camera 1 m in front of a vertical cable, looking along -Z (Newton camera frame).
        camera = Camera(
            {
                "position_m": [0.0, 0.0, 1.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 200.0, "cy": 200.0},
            }
        )
        cable = np.stack([np.zeros(11), np.linspace(0.2, -0.2, 11), np.zeros(11)], axis=1)
        uv = camera.project(cable)
        np.testing.assert_allclose(uv[0], [200.0, 100.0])
        np.testing.assert_allclose(uv[-1], [200.0, 300.0])

        # The mask: the projected cable, 5 px wide.
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[100:301, 198:203] = 255
        target = MaskTarget(mask)
        self.assertAlmostEqual(target.extent, 200.0)

        at_truth = chamfer(uv, target)
        shifted = chamfer(camera.project(cable + np.array([0.02, 0.0, 0.0])), target)  # 10 px to the side
        short = chamfer(camera.project(cable[:6]), target)  # half the cable
        self.assertLess(at_truth, 0.01)
        self.assertAlmostEqual(shifted, 10.0 / 200.0, delta=0.005)
        self.assertGreater(short, 2.0 * at_truth)
        # A cable behind the camera projects nothing.
        self.assertEqual(chamfer(camera.project(cable + np.array([0.0, 0.0, 2.0])), target), MISS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
