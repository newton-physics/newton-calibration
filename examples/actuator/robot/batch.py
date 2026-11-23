# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent robot candidates in SolverMuJoCo's CUDA world dimension."""

import time
import warnings

import numpy as np
import warp as wp

from newton_calibration import BatchEnvironment, Environment, SimulationError, Trace

from .environment import EnvironmentRobot


@wp.kernel
def _pd(
    q: wp.array[float],
    v: wp.array[float],
    q_indices: wp.array[int],
    v_indices: wp.array[int],
    refs: wp.array[float],
    kp: wp.array2d[float],
    kd: wp.array2d[float],
    limits: wp.array2d[float],
    nq: int,
    nv: int,
    forces: wp.array[float],
):
    world, drive = wp.tid()
    qi = world * nq + q_indices[drive]
    vi = world * nv + v_indices[drive]
    forces[vi] = wp.clamp(
        kp[world, drive] * (refs[drive] - q[qi]) - kd[world, drive] * v[vi], -limits[world, drive], limits[world, drive]
    )


@wp.kernel
def _check_state(q: wp.array[float], v: wp.array[float], a: wp.array2d[float], nq: int, nv: int, failed: wp.array[int]):
    world = wp.tid()
    for j in range(nq):
        value = q[world * nq + j]
        if not wp.isfinite(value) or wp.abs(value) > 1.0e10:
            failed[world] = 1
    for j in range(nv):
        velocity = v[world * nv + j]
        acceleration = a[world, j]
        if not wp.isfinite(velocity) or wp.abs(velocity) > 1.0e10:
            failed[world] = 1
        if not wp.isfinite(acceleration) or wp.abs(acceleration) > 1.0e10:
            failed[world] = 1


@wp.kernel
def _record(
    q: wp.array[float],
    v: wp.array[float],
    body: wp.array[wp.transform],
    indices: wp.array[int],
    velocity: wp.array[int],
    nq: int,
    nv: int,
    nb: int,
    sample: int,
    values: wp.array3d[float],
    positions: wp.array3d[wp.vec3],
):
    world = wp.tid()
    for c in range(indices.shape[0]):
        if velocity[c] == 1:
            values[world, sample, c] = v[world * nv + indices[c]]
        else:
            values[world, sample, c] = q[world * nq + indices[c]]
    for b in range(nb):
        positions[world, sample, b] = wp.transform_get_translation(body[world * nb + b])


class BatchEnvironmentRobotGPU(BatchEnvironment):
    """Run one experiment with independent gains and state in each GPU world.

    Physics and sampled PD run on CUDA. Observations stay on device until the
    batch finishes; the existing evaluator consumes CPU NumPy traces. Graphs
    cover integration segments between control/observation events.
    """

    def __init__(self, asset, numeric, parameters, settings):
        import mujoco  # noqa: PLC0415
        import newton  # noqa: PLC0415

        self.device = wp.get_device(numeric["device"])
        if not self.device.is_cuda:
            raise ValueError("robot GPU batches require a CUDA device")
        self.count = len(parameters)
        if not self.count:
            raise ValueError("robot batch cannot be empty")
        # Reuse the CPU adapter's import, routing and controller contract checks;
        # this one inspection instance does not simulate any candidate.
        prototype = EnvironmentRobot(asset, {**numeric, "device": "cpu"}, parameters[0], settings)
        self.native, self.names = prototype.native, prototype.names
        self.dt, self.control_dt = prototype.dt, prototype.control_dt
        self.settings, self.bindings = settings, prototype.bindings
        self.command_names = prototype.command_names
        self.nq, self.nv, self.nb = self.bindings["nq"], self.bindings["nv"], len(self.bindings["bodies"])
        self.history, self._graphs, self._used = [], {}, False
        coefficients = {
            field: np.asarray([[p[f"{name}.{field}"] for name in self.names] for p in parameters], dtype=np.float32)
            for field in (("kp", "kd") if self.native else ("kp", "kd", "torque_limit"))
        }
        if any(not np.isfinite(v).all() or (v < 0).any() for v in coefficients.values()):
            raise SimulationError("invalid robot controller parameters")
        if not self.native and (coefficients["torque_limit"] <= 0).any():
            raise SimulationError("invalid robot torque limits")
        if any(p["control_dt"] != self.control_dt for p in parameters):
            raise ValueError("controller period differs from the asset")
        source = asset._build_builder()
        builder = newton.ModelBuilder(gravity=tuple(asset.config["gravity"]))
        builder.replicate(source, self.count)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model = builder.finalize(device=self.device)
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=False,
                separate_worlds=True,
                integrator="implicitfast",
                disable_contacts=True,
                update_data_interval=1,
            )
            if caught:
                raise ValueError("robot GPU conversion warning: " + "; ".join(str(w.message) for w in caught))
        if (self.model.joint_coord_count, self.model.joint_dof_count, self.model.body_count) != (
            self.count * self.nq,
            self.count * self.nv,
            self.count * self.nb,
        ):
            raise ValueError("replicated robot layout differs from inspection model")
        self.control = self.model.control()
        self.state_in, self.state_out = self.model.state(), self.model.state()
        self.state_in.joint_q.assign(np.tile(prototype.state_in.joint_q.numpy(), self.count))
        self.state_in.joint_qd.assign(np.tile(prototype.state_in.joint_qd.numpy(), self.count))
        newton.eval_fk(self.model, self.state_in.joint_q, self.state_in.joint_qd, self.state_in)
        self.failed = wp.zeros(self.count, dtype=int, device=self.device)
        if self.native:
            self.native_ids, self.control_ids = prototype.native_ids, prototype.control_ids
            self.gears, self.command_space = np.asarray(prototype.gears), prototype.command_space
            gain = self.solver.mjw_model.actuator_gainprm.numpy()
            bias = self.solver.mjw_model.actuator_biasprm.numpy()
            for j, aid in enumerate(self.native_ids):
                gain[:, aid, 0] = coefficients["kp"][:, j]
                bias[:, aid, :3] = np.column_stack(
                    (np.zeros(self.count), -coefficients["kp"][:, j], -coefficients["kd"][:, j])
                )
            self.solver.mjw_model.actuator_gainprm.assign(gain)
            self.solver.mjw_model.actuator_biasprm.assign(bias)
        else:
            self.solver.mjw_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            self.solver.mjw_model.actuator_gainprm.zero_()
            self.solver.mjw_model.actuator_biasprm.zero_()
            self.kp, self.kd, self.limits = (
                wp.array(coefficients[field], dtype=float, device=self.device) for field in ("kp", "kd", "torque_limit")
            )
            self.q_indices = wp.array(prototype.q_indices, dtype=int, device=self.device)
            self.v_indices = wp.array(prototype.v_indices, dtype=int, device=self.device)
            self.refs = wp.zeros(len(self.names), device=self.device)
        self.indices = wp.array(
            [
                self.bindings["joints"][name]["q_start" if kind == "position" else "qd_start"]
                for kind, name in settings["channels"]
            ],
            dtype=int,
            device=self.device,
        )
        self.velocity = wp.array(
            [int(kind == "velocity") for kind, _ in settings["channels"]], dtype=int, device=self.device
        )
        prototype.close()

    @property
    def applied_parameters(self):
        """Read each world's actual device coefficients, including linked bias."""
        if self.native:
            gain = self.solver.mjw_model.actuator_gainprm.numpy()[:, self.native_ids, 0]
            bias = self.solver.mjw_model.actuator_biasprm.numpy()[:, self.native_ids, :]
            if not np.array_equal(bias[:, :, 1], -gain):
                raise RuntimeError("native position gain and bias are inconsistent")
            coefficients = {"kp": gain, "kd": -bias[:, :, 2]}
        else:
            coefficients = {"kp": self.kp.numpy(), "kd": self.kd.numpy(), "torque_limit": self.limits.numpy()}
        return [
            {
                "control_dt": self.control_dt,
                **{
                    f"{name}.{field}": float(values[w, j])
                    for field, values in coefficients.items()
                    for j, name in enumerate(self.names)
                },
            }
            for w in range(self.count)
        ]

    def _advance(self, steps):
        # State pointers alternate across segments; both parities need graphs.
        key = (id(self.state_in), tuple(round(dt, 12) for dt in steps))
        if key not in self._graphs:
            with wp.ScopedCapture(device=self.device) as capture:
                for dt in steps:
                    self.state_in.clear_forces()
                    self.solver.step(self.state_in, self.state_out, self.control, None, dt)
                    self.state_in, self.state_out = self.state_out, self.state_in
                    wp.launch(
                        _check_state,
                        dim=self.count,
                        inputs=[
                            self.state_in.joint_q,
                            self.state_in.joint_qd,
                            self.solver.mjw_data.qacc,
                            self.nq,
                            self.nv,
                            self.failed,
                        ],
                        device=self.device,
                    )
            self._graphs[key] = capture.graph, self.state_in, self.state_out
        graph, self.state_in, self.state_out = self._graphs[key]
        wp.capture_launch(graph)

    def run(self, times, *, seed, deadline):
        """Return ordered traces; a divergent world fails only its own slot."""
        if self._used:
            raise RuntimeError("create a fresh robot environment for each rollout")
        self._used = True
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not len(times)
            or not np.isfinite(times).all()
            or times[0] < 0
            or (np.diff(times) <= 0).any()
        ):
            raise ValueError("invalid robot observation times")
        command = self.settings["command"]
        if set(command) != set(self.command_names):
            raise ValueError("commands must cover exactly the driven joints")
        signals = [(np.asarray(command[n]["times"]), np.asarray(command[n]["values"])) for n in self.command_names]
        if any(
            t.ndim != 1
            or v.shape != t.shape
            or not len(t)
            or not np.isfinite(t).all()
            or not np.isfinite(v).all()
            or (np.diff(t) <= 0).any()
            or t[0] != 0
            or t[-1] < times[-1]
            for t, v in signals
        ):
            raise ValueError("invalid commands or commands do not cover rollout")
        values = wp.empty((self.count, len(times), len(self.settings["channels"])), dtype=float, device=self.device)
        positions = wp.empty((self.count, len(times), self.nb), dtype=wp.vec3, device=self.device)
        current, tick, step = 0.0, 0, 0
        for sample, target in enumerate(times):
            while current < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("robot rollout deadline")
                if current >= tick * self.control_dt - 1e-12:
                    refs = np.asarray(
                        [v[np.searchsorted(t, tick * self.control_dt + 1e-12, side="right") - 1] for t, v in signals]
                    )
                    if self.native:
                        controls = refs * self.gears if self.command_space == "joint_position" else refs
                        ctrl = np.zeros((self.count, self.control.mujoco.ctrl.shape[0] // self.count), dtype=np.float32)
                        ctrl[:, self.control_ids] = controls
                        self.control.mujoco.ctrl.assign(ctrl.ravel())
                    else:
                        self.refs.assign(refs.astype(np.float32))
                        wp.launch(
                            _pd,
                            dim=(self.count, len(self.names)),
                            inputs=[
                                self.state_in.joint_q,
                                self.state_in.joint_qd,
                                self.q_indices,
                                self.v_indices,
                                self.refs,
                                self.kp,
                                self.kd,
                                self.limits,
                                self.nq,
                                self.nv,
                                self.control.joint_f,
                            ],
                            device=self.device,
                        )
                    self.history.append({"time": tick * self.control_dt, "reference": refs.tolist()})
                    tick += 1
                end = min(target, tick * self.control_dt)
                steps = []
                while current < end - 1e-12:
                    next_time = min(end, (step + 1) * self.dt)
                    steps.append(next_time - current)
                    current = next_time
                    if current >= (step + 1) * self.dt - 1e-12:
                        step += 1
                self._advance(steps)
            wp.launch(
                _record,
                dim=self.count,
                inputs=[
                    self.state_in.joint_q,
                    self.state_in.joint_qd,
                    self.state_in.body_q,
                    self.indices,
                    self.velocity,
                    self.nq,
                    self.nv,
                    self.nb,
                    sample,
                    values,
                    positions,
                ],
                device=self.device,
            )
        output, xyz, failed = values.numpy(), positions.numpy(), self.failed.numpy()
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("robot rollout exceeded deadline")
        return [
            SimulationError("nonfinite or divergent robot state")
            if failed[w] or not np.isfinite(output[w]).all() or not np.isfinite(xyz[w]).all()
            else Trace(times.copy(), output[w], xyz[w])
            for w in range(self.count)
        ]

    def close(self):
        self._graphs.clear()


class EnvironmentRobotGPU(Environment):
    """Use the same GPU execution contract for one candidate, including replay."""

    def __init__(self, asset, numeric, parameters, settings):
        self.batch = BatchEnvironmentRobotGPU(asset, numeric, [parameters], settings)

    @property
    def applied_parameters(self):
        return self.batch.applied_parameters[0]

    def run(self, times, *, seed, deadline):
        result = self.batch.run(times, seed=seed, deadline=deadline)[0]
        if isinstance(result, SimulationError):
            raise result
        return result

    def close(self):
        self.batch.close()
