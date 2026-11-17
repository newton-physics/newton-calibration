# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Connect Newton's rigid-body contact simulation to the calibration members."""

import time
from threading import Lock

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverXPBD

from newton_calibration import Asset, Environment, Trace

# Pipeline construction and collision detection both specialize kernels in a shared Warp module.
_collision_lock = Lock()


class SlidingEnvironment(Environment):
    """Simulate a box on a horizontal plane and observe its center position [m]."""

    def __init__(self, parameters, numeric_settings, initial_speed, config):
        with _collision_lock:
            self.dt = float(numeric_settings["dt"])
            half_x, half_y, half_z = config["half_extents"]
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -parameters["gravity"]))
            material = newton.ModelBuilder.ShapeConfig(
                density=config["density"],
                mu=parameters["friction"],
                mu_torsional=0.0,
                mu_rolling=0.0,
            )
            self.body = builder.add_body(xform=wp.transform((0.0, 0.0, half_z), wp.quat_identity()))
            self.box_shape = builder.add_shape_box(self.body, hx=half_x, hy=half_y, hz=half_z, cfg=material)
            self.ground_shape = builder.add_ground_plane(cfg=material)
            self.model = builder.finalize(device=numeric_settings["device"])
            self.solver = SolverXPBD(self.model, iterations=numeric_settings["iterations"])
            self.state_in, self.state_out = self.model.state(), self.model.state()
            velocities = self.state_in.body_qd.numpy().copy()
            velocities[self.body, 0] = initial_speed
            self.state_in.body_qd.assign(velocities)
            self.collision = newton.CollisionPipeline(self.model)
            self.contacts = self.collision.contacts()

    @property
    def applied_parameters(self):
        materials = self.model.shape_material_mu.numpy()
        if materials[self.box_shape] != materials[self.ground_shape]:
            raise ValueError("the box and ground must share the fitted friction coefficient")
        return {
            "friction": float(materials[self.box_shape]),
            "gravity": -float(self.model.gravity.numpy()[0, 2]),
        }

    def run(self, times, *, seed, deadline):
        current_time = 0.0
        positions = []
        for target in times:
            while current_time < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("sliding simulation deadline")
                step = min(self.dt, target - current_time)
                self.state_in.clear_forces()
                with _collision_lock:
                    self.collision.collide(self.state_in, self.contacts)
                self.solver.step(self.state_in, self.state_out, None, self.contacts, step)
                self.state_in, self.state_out = self.state_out, self.state_in
                current_time += step
            # CPU NumPy views share state memory that later steps overwrite.
            positions.append(self.state_in.body_q.numpy()[self.body, :3].copy())
        positions = np.asarray(positions, dtype=np.float64)[:, None, :]
        return Trace(times.copy(), positions[:, :, 0].copy(), positions)


class SlidingAsset(Asset):
    """Construct a fresh Newton model for each candidate and experiment."""

    def create_environment(self, template, parameter_values, experiment_settings):
        settings = template.numeric_settings
        if settings["solver"] != "SolverXPBD" or settings["dtype"] != "float32":
            raise ValueError("SlidingAsset requires SolverXPBD with float32 state")
        if not np.isfinite(settings["dt"]) or settings["dt"] <= 0:
            raise ValueError("the timestep must be positive and finite")
        if not isinstance(settings["iterations"], int) or settings["iterations"] < 1:
            raise ValueError("solver iterations must be a positive integer")
        return SlidingEnvironment(parameter_values, settings, experiment_settings["initial_speed"], self.config)
