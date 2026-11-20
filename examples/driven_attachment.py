# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive kinematic bodies along a recorded pose trajectory, for the driven examples.

A drive is a pose trajectory, for example a robot TCP: sample times, positions and
rotations in one frame. Each member of a batch has one or more driven bodies, each with a
fixed transform from the drive frame to the body. Before every solver step, a kernel
moves every driven body to the drive pose at the end of the step, composed with that
transform. The pose is interpolated between the samples (linear in position, slerp in
rotation) and held at the end samples outside the recorded span.

Before time zero, the drive holds its first pose for a number of settle frames, so the
asset starts the observed motion at rest. A member that still moves at the end of the
settle phase fails with a SimulationError.

An observation is taken after the solver step that ends nearest to its time, so observation
times need not lie on the frame grid; a camera's own frame stamps can be used directly.

:class:`DrivenBatchEnvironment` holds this sequence. A case subclasses it and supplies
the asset: how to build one member, the solver, the positions to watch and the
observation. The kernel writes the poses into ``body_q``, which maximal-coordinate solvers
such as VBD read for kinematic bodies.
"""

import itertools
import time

import newton
import numpy as np
import warp as wp

from newton_calibration import BatchEnvironment, Environment, SimulationError, Trace

# Window [frames] over which the settle drift is measured.
SETTLE_CHECK_FRAMES = 2


def check_drive(drive):
    """Check that a drive has matching, strictly increasing samples."""
    n = len(drive["times_s"])
    if n < 2 or len(drive["positions_m"]) != n or len(drive["quaternions_xyzw"]) != n:
        raise ValueError("drive times, positions and quaternions must have the same length >= 2")
    if any(b <= a for a, b in itertools.pairwise(drive["times_s"])):
        raise ValueError("drive times must be strictly increasing")


def drive_pose(drive, t):
    """Host-side drive pose at time ``t`` [s]: the same interpolation as :func:`_drive_kernel`."""
    times = drive["times_s"]
    if t <= times[0]:
        i, frac = 0, 0.0
    elif t >= times[-1]:
        i, frac = len(times) - 1, 0.0
    else:
        i = max(j for j in range(len(times) - 1) if times[j] <= t)
        frac = (t - times[i]) / (times[i + 1] - times[i])
    p0 = wp.vec3(*drive["positions_m"][i])
    q0 = wp.quat(*drive["quaternions_xyzw"][i])
    if frac == 0.0:
        return p0, q0
    p1 = wp.vec3(*drive["positions_m"][i + 1])
    q1 = wp.quat(*drive["quaternions_xyzw"][i + 1])
    return wp.lerp(p0, p1, frac), wp.quat_slerp(q0, q1, frac)


@wp.kernel
def _drive_kernel(
    step: wp.array[int],
    time_zero: float,
    dt: float,
    drive_times: wp.array[float],
    drive_positions: wp.array[wp.vec3],
    drive_rotations: wp.array[wp.quat],
    driven_bodies: wp.array[int],
    local_positions: wp.array[wp.vec3],
    local_rotations: wp.array[wp.quat],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Move each driven body to the drive pose at the end of this step, composed with its transform.

    Before time zero the first sample holds, which is the settle phase. The kernel writes
    both states, so the solver sees the same pose in its current and previous buffer.
    """
    tid = wp.tid()
    # The pose at the END of this step, so the driven body and the asset agree at every
    # frame time. Computed from the step count: a float32 clock accumulated by addition
    # drifts by tens of microseconds over a run.
    t = time_zero + float(step[0] + 1) * dt
    n = drive_times.shape[0]
    position = drive_positions[0]
    rotation = drive_rotations[0]
    if t >= drive_times[n - 1]:
        position = drive_positions[n - 1]
        rotation = drive_rotations[n - 1]
    elif t > drive_times[0]:
        i = int(0)
        while drive_times[i + 1] <= t:
            i += 1
        frac = (t - drive_times[i]) / (drive_times[i + 1] - drive_times[i])
        position = wp.lerp(drive_positions[i], drive_positions[i + 1], frac)
        rotation = wp.quat_slerp(drive_rotations[i], drive_rotations[i + 1], frac)
    world_position = position + wp.quat_rotate(rotation, local_positions[tid])
    world_rotation = wp.mul(rotation, local_rotations[tid])
    pose = wp.transform(world_position, world_rotation)
    body = driven_bodies[tid]
    body_q0[body] = pose
    body_q1[body] = pose


@wp.kernel
def _advance_step_kernel(step: wp.array[int]):
    step[0] = step[0] + 1


class SingleEnvironment(Environment):
    """A batch environment with one member, seen as an Environment."""

    def __init__(self, batch):
        self.batch = batch
        self.applied_parameters = dict(batch.applied_parameters[0])

    def run(self, times, *, seed, deadline):
        result = self.batch.run(times, seed=seed, deadline=deadline)[0]
        if isinstance(result, SimulationError):
            raise result
        return result

    def close(self):
        self.batch.close()


class DrivenBatchEnvironment(BatchEnvironment):
    """Several candidates of one experiment, all driven by the same pose trajectory.

    Numeric settings: ``device``, ``substeps`` (solver steps per frame, even),
    ``settle_frames`` (frames the drive holds its first pose before time zero) and
    ``settle_tolerance_m`` (largest movement of a watched position over the last
    ``SETTLE_CHECK_FRAMES`` frames of the settle). Experiment settings: ``fps`` (frames
    per second of the simulation) and ``drive`` (``times_s``, ``positions_m``,
    ``quaternions_xyzw``).

    A subclass sets its own fields, then calls ``super().__init__``, and implements:

    - ``_add_member(builder, parameters, drive_position, drive_rotation)``: add one
      candidate at the drive's first pose and return a dict with at least ``driven``, a
      list of ``(body, local_position, local_rotation)`` from the drive frame to each
      driven body.
    - ``_finalize(builder)``: return ``(model, solver)``.
    - ``_readback(member)``: the parameters as the model holds them.
    - ``_positions()``: watched positions [m], shape (members, points, 3), for the
      settle check, the finite check and the observation.
    - ``_observe(positions, index)``: one observation row from one member's positions at
      observation ``index``.
    """

    def __init__(self, template, parameter_values, settings, *, gravity=(0.0, 0.0, -9.81)):
        numeric = template.numeric_settings
        self.device = wp.get_device(numeric["device"])
        self.substeps = int(numeric["substeps"])
        if self.substeps < 2 or self.substeps % 2:
            raise ValueError("substeps must be an even number of at least 2")
        self.settle_frames = int(numeric["settle_frames"])
        self.settle_tolerance = float(numeric["settle_tolerance_m"])
        self.fps = float(settings["fps"])
        self.frame_dt = 1.0 / self.fps
        self.dt = self.frame_dt / self.substeps
        self.drive = settings["drive"]
        check_drive(self.drive)
        drive_position, drive_rotation = drive_pose(self.drive, self.drive["times_s"][0])

        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(gravity=gravity)
            self.members = [
                self._add_member(builder, parameters, drive_position, drive_rotation) for parameters in parameter_values
            ]
            self.model, self.solver = self._finalize(builder)
            self.state_0, self.state_1 = self.model.state(), self.model.state()
            self.control = self.model.control()
            self.step = wp.zeros(1, dtype=int)
            self.time_zero = -self.settle_frames * self.frame_dt
            self.drive_times = wp.array(np.asarray(self.drive["times_s"], dtype=np.float32), dtype=float)
            self.drive_positions = wp.array(np.asarray(self.drive["positions_m"], dtype=np.float32), dtype=wp.vec3)
            self.drive_rotations = wp.array(np.asarray(self.drive["quaternions_xyzw"], dtype=np.float32), dtype=wp.quat)
            driven = [entry for member in self.members for entry in member["driven"]]
            self.driven_bodies = wp.array([body for body, _, _ in driven], dtype=int)
            self.local_positions = wp.array([position for _, position, _ in driven], dtype=wp.vec3)
            self.local_rotations = wp.array([rotation for _, _, rotation in driven], dtype=wp.quat)
            self.graph = None
            self.substep_graph = None
            if self.device.is_cuda:
                with wp.ScopedCapture(device=self.device) as capture:
                    self._frame()
                self.graph = capture.graph
                with wp.ScopedCapture(device=self.device) as capture:
                    self._substep()
                self.substep_graph = capture.graph
                self._reset()
        self.applied_parameters = [self._readback(member) for member in self.members]

    def _reset(self):
        """Put every member back at its built pose, at rest, with the clock before time zero.

        The solver reset also clears the solver's own history, so every run starts alike.
        """
        self.solver.reset(self.state_0)
        self.state_1.body_q.assign(self.state_0.body_q)
        self.state_1.body_qd.assign(self.state_0.body_qd)
        self.step.zero_()

    def _substep(self):
        """Advance all members by one solver step."""
        wp.launch(
            _drive_kernel,
            dim=self.driven_bodies.shape[0],
            inputs=[
                self.step,
                self.time_zero,
                self.dt,
                self.drive_times,
                self.drive_positions,
                self.drive_rotations,
                self.driven_bodies,
                self.local_positions,
                self.local_rotations,
            ],
            outputs=[self.state_0.body_q, self.state_1.body_q],
        )
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        wp.launch(_advance_step_kernel, dim=1, inputs=[self.step])

    def _frame(self):
        """Advance all members by one camera frame."""
        for _ in range(self.substeps):
            self._substep()

    def _step_frame(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._frame()

    def _step_substep(self):
        if self.substep_graph is not None:
            wp.capture_launch(self.substep_graph)
        else:
            self._substep()

    def _step_indices(self, times):
        """Solver steps after time zero at which each observation is taken: the nearest step end."""
        steps = [round(t / self.dt) for t in times.tolist()]
        if times[0] < 0.0 or any(b <= a for a, b in itertools.pairwise(steps)):
            raise ValueError("observation times must be nonnegative and at least one solver step apart")
        return steps

    def run(self, times, *, seed, deadline):
        steps = self._step_indices(times)
        failures = [None] * len(self.members)
        with wp.ScopedDevice(self.device):
            self._reset()
            before = None
            for frame in range(self.settle_frames):
                self._check_deadline(deadline)
                if frame == self.settle_frames - SETTLE_CHECK_FRAMES:
                    before = self._positions()
                self._step_frame()
            if before is not None:
                drift = np.abs(self._positions() - before).max(axis=(1, 2))
                for i, value in enumerate(drift):
                    if not value < self.settle_tolerance:
                        failures[i] = SimulationError(
                            f"the asset did not settle: moved {value:.2e} m in the last "
                            f"{SETTLE_CHECK_FRAMES} frames, tolerance {self.settle_tolerance:.0e} m"
                        )
            values = [[] for _ in self.members]
            current = 0
            for index, step in enumerate(steps):
                while current + self.substeps <= step:
                    self._check_deadline(deadline)
                    self._step_frame()
                    current += self.substeps
                while current < step:
                    self._step_substep()
                    current += 1
                positions = self._positions()
                for i in range(len(self.members)):
                    if failures[i] is None and not np.isfinite(positions[i]).all():
                        failures[i] = SimulationError(f"nonfinite pose at observation {index}")
                    values[i].append(self._observe(positions[i], index))
        results = []
        for i in range(len(self.members)):
            if failures[i] is not None:
                results.append(failures[i])
            else:
                results.append(Trace(times.copy(), np.array(values[i], dtype=np.float64)))
        return results

    @staticmethod
    def _check_deadline(deadline):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("driven environment deadline")

    def close(self):
        self.graph = None
        self.substep_graph = None
        self.members = []
