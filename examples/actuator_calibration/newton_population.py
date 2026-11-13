# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replicated Newton worlds that replay recorded joint position commands.

Each world simulates one candidate on one recording. Per-world parameters are written to
Newton model arrays and read back from them. Several recordings can be replayed in one
pass, each in its own worlds; shorter recordings hold their last command until the
longest one ends. Commands are uploaded once per pass, and joint positions and body
positions are recorded on the device every control interval and copied to the host once
per pass.
"""

from __future__ import annotations

import math
import time

import newton
import numpy as np
import warp as wp

from newton_calibration import SimulationError, Trace

# Calibrated quantity or fixed clamp -> Newton model array (SI units, per DOF).
MODEL_ARRAYS = {
    "stiffness": "joint_target_ke",
    "damping": "joint_target_kd",
    "armature": "joint_armature",
    "dry_friction": "joint_friction",
    "viscous_friction": "joint_damping",
    "effort_limit": "joint_effort_limit",
    "velocity_limit": "joint_velocity_limit",
}


@wp.kernel
def _write_targets(
    commands: wp.array3d[wp.float32],  # [recording, time, joint]
    world_recording: wp.array[wp.int32],  # [world]
    row: int,
    target_indices: wp.array2d[wp.int32],  # [world, joint]
    joint_target_q: wp.array[wp.float32],
):
    world, joint = wp.tid()
    joint_target_q[target_indices[world, joint]] = commands[world_recording[world], row, joint]


@wp.kernel
def _record_joint_q(
    joint_q: wp.array[wp.float32],
    q_indices: wp.array2d[wp.int32],  # [world, joint]
    row: int,
    output: wp.array3d[wp.float32],  # [time, world, joint]
):
    world, joint = wp.tid()
    output[row, world, joint] = joint_q[q_indices[world, joint]]


@wp.kernel
def _record_body_positions(
    body_q: wp.array[wp.transform],
    body_indices: wp.array2d[wp.int32],  # [world, body]
    row: int,
    output: wp.array3d[wp.vec3],  # [time, world, body]
):
    world, body = wp.tid()
    output[row, world, body] = wp.transform_get_translation(body_q[body_indices[world, body]])


def _matching(labels, requested):
    matches = [i for i, label in enumerate(labels) if label == requested or label.rsplit("/", 1)[-1] == requested]
    if len(matches) != 1:
        raise ValueError(f"Newton asset must contain exactly one joint matching {requested!r}")
    return matches[0]


def builder_dof(builder, joint):
    index = _matching(builder.joint_label, joint)
    q_start, dof_start = int(builder.joint_q_start[index]), int(builder.joint_qd_start[index])
    if index + 1 < len(builder.joint_label):
        q_end, dof_end = builder.joint_q_start[index + 1], builder.joint_qd_start[index + 1]
    else:
        q_end, dof_end = builder.joint_coord_count, builder.joint_dof_count
    if q_end - q_start != 1 or dof_end - dof_start != 1:
        raise ValueError(f"joint {joint!r} must have exactly one coordinate and DOF")
    return dof_start


def create_builder(usd_path, joints, *, fixed_base, values=None):
    """A one-world builder loaded from USD with position-target drives on ``joints``.

    ``values`` optionally overrides the USD-authored quantities per joint, as the baseline
    asset does. A reloaded export passes no values, so the USD alone defines them.
    """
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(source=str(usd_path), floating=not fixed_base)
    position = int(newton.JointTargetMode.POSITION)
    for joint in joints:
        dof = builder_dof(builder, joint)
        builder.joint_target_mode[dof] = position
        for quantity, value in (values or {}).get(joint, {}).items():
            if quantity in MODEL_ARRAYS:
                getattr(builder, MODEL_ARRAYS[quantity])[dof] = float(value)
    return builder


def model_indices(model, joints, world_count):
    """Per-world q, DOF and target indices for each joint, and body indices per world."""
    worlds = model.joint_world.numpy()
    q_start = model.joint_q_start.numpy()
    dof_start = model.joint_qd_start.numpy()
    target_start = model.joint_target_q_start.numpy()
    shape = (world_count, len(joints))
    q, dof, target = (np.empty(shape, dtype=np.int32) for _ in range(3))
    for w in range(world_count):
        world_joints = np.flatnonzero(worlds == w)
        labels = [model.joint_label[i] for i in world_joints]
        for j, joint in enumerate(joints):
            index = int(world_joints[_matching(labels, joint)])
            for starts in (q_start, dof_start, target_start):
                if starts[index + 1] - starts[index] != 1:
                    raise ValueError(f"joint {joint!r} is not scalar")
            q[w, j], dof[w, j], target[w, j] = q_start[index], dof_start[index], target_start[index]
    body_world = model.body_world.numpy()
    bodies = [np.flatnonzero(body_world == w) for w in range(world_count)]
    if len({len(b) for b in bodies}) != 1:
        raise ValueError("replicated worlds must contain the same number of bodies")
    return q, dof, target, np.stack(bodies).astype(np.int32)


class StepExecutor:
    """Advance one control interval; every interval starts and ends in ``state``.

    With an odd number of substeps the final result is copied back into the canonical
    buffer, so the same routine can be captured once as a Warp graph and replayed.
    """

    def __init__(self, model, solver, state, state_next, control, dt, substeps):
        self.model, self.solver, self.control = model, solver, control
        self.state, self.state_next = state, state_next
        self.physics_dt, self.substeps = dt / substeps, substeps
        self.graph = None
        capturable = (
            model.device.is_cuda
            and not getattr(solver, "use_mujoco_cpu", False)
            and getattr(solver, "update_data_interval", 1) in (0, 1)
        )
        if capturable:
            with wp.ScopedCapture(device=model.device) as capture:
                self._interval()
            self.graph = capture.graph

    def _interval(self):
        state, state_next = self.state, self.state_next
        for index in range(self.substeps):
            state.clear_forces()
            self.solver.step(state, state_next, self.control, None, self.physics_dt)
            if self.substeps % 2 and index == self.substeps - 1:
                self.state.assign(state_next)
            else:
                state, state_next = state_next, state

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._interval()


class NewtonActuatorPopulation:
    """``size`` replicated worlds of one Newton model, reused across batches.

    ``assign`` writes one complete parameter mapping per world and returns the values read
    back from the model; ``run`` replays one recording in every world.
    """

    def __init__(
        self,
        usd_path,
        joints,
        numeric_settings,
        discretization,
        size,
        *,
        fixed_base,
        builder_values=None,
        device=None,
    ):
        self.size, self.joints = size, tuple(joints)
        control_freq = int(discretization["control_freq"])
        physics_freq = int(numeric_settings["physics_freq"])
        if physics_freq % control_freq:
            raise ValueError("physics_freq must be an integer multiple of control_freq")
        self.control_dt = 1.0 / control_freq
        template = create_builder(usd_path, joints, fixed_base=fixed_base, values=builder_values)
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(template, size)
        self.model = model = builder.finalize(device=device)
        self.state, self.state_next = model.state(), model.state()
        self.control = model.control()
        self.solver = newton.solvers.SolverMuJoCo(
            model,
            solver=numeric_settings["solver"],
            integrator=numeric_settings["integrator"],
            iterations=int(numeric_settings["iterations"]),
            ls_iterations=int(numeric_settings["ls_iterations"]),
        )
        q, dof, target, bodies = model_indices(model, self.joints, size)
        self._q, self._dof, self._bodies = q, dof, bodies
        self._q_device = wp.array(q, dtype=wp.int32, device=model.device)
        self._target_device = wp.array(target, dtype=wp.int32, device=model.device)
        self._bodies_device = wp.array(bodies, dtype=wp.int32, device=model.device)
        self._base_q = model.joint_q.numpy().copy()
        self._base_qd = model.joint_qd.numpy().copy()
        self._base_arrays = {name: getattr(model, name).numpy().copy() for name in MODEL_ARRAYS.values()}
        self._delay_steps = np.zeros(len(self.joints), dtype=np.int64)
        self._executor = StepExecutor(
            model,
            self.solver,
            self.state,
            self.state_next,
            self.control,
            self.control_dt,
            physics_freq // control_freq,
        )

    def readback(self):
        """Per-world ``<joint>.<quantity>`` values read from the Newton model arrays."""
        arrays = {k: getattr(self.model, v).numpy() for k, v in MODEL_ARRAYS.items()}
        return [
            {
                f"{joint}.{quantity}": float(arrays[quantity][self._dof[w, j]])
                for j, joint in enumerate(self.joints)
                for quantity in MODEL_ARRAYS
            }
            for w in range(self.size)
        ]

    def assign(self, parameter_values):
        if len(parameter_values) != self.size:
            raise ValueError(f"expected {self.size} parameter sets")
        delays = {joint: {float(p[f"{joint}.delay_steps"]) for p in parameter_values} for joint in self.joints}
        if any(len(v) != 1 for v in delays.values()):
            raise ValueError("command delays must be identical in every world")
        self._delay_steps = np.array([int(next(iter(delays[j]))) for j in self.joints])
        for quantity, array_name in MODEL_ARRAYS.items():
            values = self._base_arrays[array_name].copy()
            for w, parameters in enumerate(parameter_values):
                for j, joint in enumerate(self.joints):
                    key = f"{joint}.{quantity}"
                    if key in parameters:
                        values[self._dof[w, j]] = parameters[key]
            getattr(self.model, array_name).assign(values)
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES | newton.ModelFlags.ACTUATOR_PROPERTIES)
        applied = self.readback()
        for w, parameters in enumerate(parameter_values):
            for joint, steps in zip(self.joints, self._delay_steps, strict=True):
                applied[w][f"{joint}.delay_steps"] = float(steps)
            # Report only what was requested; unrequested model quantities are not bindings.
            applied[w] = {k: v for k, v in applied[w].items() if k in parameters}
        return applied

    def _reset(self, initial_q):
        """Start every world at rest from ``initial_q`` [world, joint]."""
        joint_q, joint_qd = self._base_q.copy(), self._base_qd.copy()
        joint_q[self._q] = initial_q
        joint_qd[self._dof] = 0.0
        self.solver.reset(self.state)
        for state in (self.state, self.state_next):
            state.joint_q.assign(joint_q)
            state.joint_qd.assign(joint_qd)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def run(self, times, *, deadline, inputs=None):
        """Replay one recording in every world."""
        return self.run_recordings([(times, inputs)], np.zeros(self.size, dtype=np.int32), deadline=deadline)

    def _delayed(self, times, inputs):
        if inputs is None or {"command_q", "initial_q"} - set(inputs):
            raise ValueError("actuator replay needs command_q and initial_q inputs")
        commands = np.asarray(inputs["command_q"], dtype=np.float64)
        steps = len(commands) - 1
        if commands.shape[1:] != (len(self.joints),) or len(times) != steps:
            raise ValueError("commands, initial state and observation times are misaligned")
        expected = self.control_dt * np.arange(1, steps + 1)
        if not np.allclose(times, expected, rtol=0.0, atol=1e-9):
            raise ValueError("observation times must follow the control interval")
        # Command i is applied during interval i, delayed per joint and held at the start.
        rows = np.maximum(0, np.arange(len(commands))[:, None] - self._delay_steps[None, :])
        return np.take_along_axis(commands, rows, axis=0)[:steps]

    def run_recordings(self, recordings, world_recording, *, deadline):
        """Replay ``recordings`` [(times, inputs)] at once; world w replays ``world_recording[w]``.

        Returns one Trace or SimulationError per world, covering that world's recording.
        """
        world_recording = np.asarray(world_recording, dtype=np.int32)
        if world_recording.shape != (self.size,) or not (0 <= world_recording).all():
            raise ValueError("world_recording must assign one recording to every world")
        if world_recording.max() >= len(recordings):
            raise ValueError("world_recording names a missing recording")
        delayed = [self._delayed(times, inputs) for times, inputs in recordings]
        lengths = np.array([len(d) for d in delayed])
        steps = int(lengths.max())
        # Shorter recordings hold their last command; their extra rows are not returned.
        padded = np.stack([np.concatenate([d, np.repeat(d[-1:], steps - len(d), axis=0)]) for d in delayed])
        initial = np.stack([np.asarray(inputs["initial_q"], dtype=np.float32) for _, inputs in recordings])
        device = self.model.device
        commands_device = wp.array(padded.astype(np.float32), dtype=wp.float32, device=device)
        recording_device = wp.array(world_recording, dtype=wp.int32, device=device)
        joint_trace = wp.zeros((steps, self.size, len(self.joints)), dtype=wp.float32, device=device)
        body_trace = wp.zeros((steps, self.size, self._bodies.shape[1]), dtype=wp.vec3, device=device)
        self._reset(initial[world_recording])
        for row in range(steps):
            if deadline is not None and row % 256 == 0 and time.monotonic() >= deadline:
                raise TimeoutError("actuator replay deadline")
            wp.launch(
                _write_targets,
                dim=(self.size, len(self.joints)),
                inputs=[commands_device, recording_device, row, self._target_device],
                outputs=[self.control.joint_target_q],
                device=device,
            )
            self._executor.step()
            wp.launch(
                _record_joint_q,
                dim=(self.size, len(self.joints)),
                inputs=[self.state.joint_q, self._q_device, row],
                outputs=[joint_trace],
                device=device,
            )
            wp.launch(
                _record_body_positions,
                dim=(self.size, self._bodies.shape[1]),
                inputs=[self.state.body_q, self._bodies_device, row],
                outputs=[body_trace],
                device=device,
            )
        values = joint_trace.numpy()
        positions = body_trace.numpy()
        results = []
        for w in range(self.size):
            index = int(world_recording[w])
            count = int(lengths[index])
            world_values = values[:count, w, :].astype(np.float64)
            if not np.isfinite(world_values).all():
                results.append(SimulationError(f"world {w} produced nonfinite joint positions"))
                continue
            times = np.asarray(recordings[index][0], dtype=np.float64)
            results.append(Trace(times.copy(), world_values, positions[:count, w, :, :].astype(np.float64)))
        return results


def single_world_parameters(usd_path, joints, *, fixed_base, device=None):
    """Read USD-authored joint quantities through Newton's importer, without overrides."""
    builder = create_builder(usd_path, joints, fixed_base=fixed_base)
    values = {}
    for joint in joints:
        dof = builder_dof(builder, joint)
        for quantity, array_name in MODEL_ARRAYS.items():
            values[f"{joint}.{quantity}"] = float(getattr(builder, array_name)[dof])
            if not math.isfinite(values[f"{joint}.{quantity}"]):
                raise ValueError(f"{joint}.{quantity} is not finite in {usd_path}")
    return values
