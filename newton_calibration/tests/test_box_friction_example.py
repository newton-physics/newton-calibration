# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from threading import Barrier

import numpy as np


@unittest.skipUnless(find_spec("newton") and find_spec("warp"), "requires the Newton example extra")
class TestBoxFrictionExample(unittest.TestCase):
    def test_four_workers_match_serial_trajectories(self):
        """Build fresh XPBD scenes concurrently and compare their trajectories with serial runs."""
        from examples.box_friction_calibration.backend import SlidingEnvironment  # noqa: PLC0415

        speeds = (1.2, 1.7, 2.1, 2.4)
        times = np.array([0.0, 1.0 / 480.0, 0.02])
        start = Barrier(len(speeds))

        def trajectories(speed):
            positions = []
            for friction in (0.15, 0.35, 0.6, 0.35):
                env = SlidingEnvironment(
                    {"friction": friction, "gravity": 9.81},
                    {"device": "cpu", "dt": 1.0 / 480.0, "iterations": 8},
                    speed,
                    {"half_extents": [0.1, 0.1, 0.05], "density": 1000.0},
                )
                positions.append(env.run(times, seed=1, deadline=None).positions)
            return positions

        def worker(speed):
            start.wait(timeout=30)
            return trajectories(speed)

        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(worker, speeds))
        for speed, positions in zip(speeds, parallel, strict=True):
            np.testing.assert_array_equal(positions, trajectories(speed))
