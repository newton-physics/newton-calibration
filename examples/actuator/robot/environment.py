# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fresh multi-joint SolverMuJoCo rollouts with a sampled position controller."""

import math
import time

import numpy as np

from newton_calibration import Environment, SimulationError, Trace

from .bindings import joint_values


class EnvironmentRobot(Environment):
    """Simulate named rotary joints in rad, rad/s and N m on CPU."""

    def __init__(self, asset, numeric, parameters, settings):
        import mujoco  # noqa: PLC0415
        import newton  # noqa: PLC0415

        if numeric["solver"] != "SolverMuJoCo" or numeric["device"] != "cpu":
            raise ValueError("robot rollouts require SolverMuJoCo on CPU")
        controller = asset.config["controller"]
        self.native = controller["mode"] == "native_position"
        if controller["mode"] not in {"equivalent_pd", "native_position"}:
            raise ValueError("unsupported robot controller mode")
        self.dt, self.control_dt = numeric["dt"], controller["control_dt"]
        ratio = self.control_dt / self.dt
        if not 0 < self.dt <= self.control_dt or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("physics timestep must divide the control period")
        self.names = tuple(controller["drives"])
        self.model, self.solver, self.bindings = asset.build_model()
        self.settings = settings
        self.kp = np.array([parameters[f"{name}.kp"] for name in self.names])
        self.kd = np.array([parameters[f"{name}.kd"] for name in self.names])
        self.limits = (
            np.array([parameters[f"{name}.torque_limit"] for name in self.names]) if not self.native else np.empty(0)
        )
        if (
            not np.isfinite([*self.kp, *self.kd, *self.limits]).all()
            or (self.kp < 0).any()
            or (self.kd < 0).any()
            or (self.limits <= 0).any()
        ):
            raise SimulationError("invalid robot controller parameters")
        if parameters["control_dt"] != self.control_dt:
            raise ValueError("controller period differs from the asset")
        active = {name for name, value in self.bindings["joints"].items() if value["mujoco_joint"] is not None}
        if not set(self.names) <= active:
            raise ValueError("controller references inactive or unknown joints")
        self.q_indices = [self.bindings["joints"][name]["q_start"] for name in self.names]
        self.v_indices = [self.bindings["joints"][name]["qd_start"] for name in self.names]
        self.command_names = self.names
        if self.native:
            self.command_space = controller["command_space"]
            if self.command_space not in {"joint_position", "actuator_control"}:
                raise ValueError("declare joint_position or actuator_control command space")
            actuators = [controller["drives"][name]["actuator"] for name in self.names]
            if len(set(actuators)) != len(actuators) or set(actuators) != set(self.bindings["actuators"]):
                raise ValueError("native mode must explicitly control every source actuator exactly once")
            self.native_ids, self.control_ids, self.gears = [], [], []
            mj = self.solver.mj_model
            kinds = asset.config["resources"]["names"].get("actuator_types", {})
            for i, (name, actuator) in enumerate(zip(self.names, actuators, strict=True)):
                binding = self.bindings["actuators"][actuator]
                aid = binding["mujoco_actuator"]
                if kinds.get(actuator) not in {"position", "general"} or binding["joint"] != name:
                    raise ValueError("native mode requires a position actuator bound to its declared joint")
                if mj.actuator_dyntype[aid] != 0 or mj.actuator_gaintype[aid] != 0 or mj.actuator_biastype[aid] != 1:
                    raise ValueError("native position mode requires stateless fixed-gain affine-bias actuators")
                if (
                    mj.actuator_biasprm[aid, 0] != 0
                    or not np.isclose(mj.actuator_biasprm[aid, 1], -mj.actuator_gainprm[aid, 0])
                    or mj.actuator_gainprm[aid, 0] < 0
                    or mj.actuator_biasprm[aid, 2] > 0
                ):
                    raise ValueError("source actuator is not a position PD servo")
                gear = float(mj.actuator_gear[aid, 0])
                if gear == 0 or not np.isfinite(gear):
                    raise ValueError("native actuator gear must be finite and nonzero")
                mj.actuator_gainprm[aid, 0] = self.kp[i]
                mj.actuator_biasprm[aid, :3] = [0, -self.kp[i], -self.kd[i]]
                self.native_ids.append(aid)
                self.control_ids.append(binding["control_index"])
                self.gears.append(gear)
            if self.command_space == "actuator_control":
                self.command_names = tuple(actuators)
        else:
            # External joint torque is the only active drive in equivalent-PD mode.
            self.solver.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
            self.solver.mj_model.actuator_gainprm[:] = 0
            self.solver.mj_model.actuator_biasprm[:] = 0
        self.control = self.model.control()
        self.state_in, self.state_out = self.model.state(), self.model.state()
        initial = settings["initial_state"]
        for field, velocity in (("position", False), ("velocity", True)):
            data = initial[field]
            target = self.state_in.joint_qd if velocity else self.state_in.joint_q
            target.assign(joint_values(list(data.values()), list(data), self.bindings, velocity=velocity))
        newton.eval_fk(self.model, self.state_in.joint_q, self.state_in.joint_qd, self.state_in)
        self.history = []
        self._used = False

    @property
    def applied_parameters(self):
        """Read the actual controller arrays used to compute held torque."""
        result = {"control_dt": self.control_dt}
        for i, name in enumerate(self.names):
            if self.native:
                aid = self.native_ids[i]
                kp = float(self.solver.mj_model.actuator_gainprm[aid, 0])
                if not np.isclose(self.solver.mj_model.actuator_biasprm[aid, 1], -kp):
                    raise RuntimeError("native position gain and bias are inconsistent")
                result.update({f"{name}.kp": kp, f"{name}.kd": float(-self.solver.mj_model.actuator_biasprm[aid, 2])})
            else:
                result.update(
                    {
                        f"{name}.kp": float(self.kp[i]),
                        f"{name}.kd": float(self.kd[i]),
                        f"{name}.torque_limit": float(self.limits[i]),
                    }
                )
        return result

    def run(self, times, *, seed, deadline):
        """Hold torque between controller ticks, independently of observation times."""
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
        signals = [
            (np.asarray(command[name]["times"]), np.asarray(command[name]["values"])) for name in self.command_names
        ]
        if any(t[0] != 0 or t[-1] < times[-1] for t, _ in signals):
            raise ValueError("commands do not cover rollout")
        current, tick, step = 0.0, 0, 0
        output, positions = [], []
        for target in times:
            while current < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("robot rollout deadline")
                if current >= tick * self.control_dt - 1e-12:
                    refs = np.array(
                        [v[np.searchsorted(t, tick * self.control_dt + 1e-12, side="right") - 1] for t, v in signals]
                    )
                    if self.native:
                        controls = refs * np.asarray(self.gears) if self.command_space == "joint_position" else refs
                        values = np.zeros(self.control.mujoco.ctrl.shape[0], dtype=np.float32)
                        values[self.control_ids] = controls
                        self.control.mujoco.ctrl.assign(values)
                        self.history.append(
                            {"time": tick * self.control_dt, "reference": refs.tolist(), "control": controls.tolist()}
                        )
                    else:
                        q = self.state_in.joint_q.numpy()[self.q_indices]
                        v = self.state_in.joint_qd.numpy()[self.v_indices]
                        torque = np.clip(self.kp * (refs - q) - self.kd * v, -self.limits, self.limits)
                        if not np.isfinite(torque).all():
                            raise SimulationError("nonfinite robot torque")
                        forces = np.zeros(self.model.joint_dof_count, dtype=np.float32)
                        forces[self.v_indices] = torque
                        self.control.joint_f.assign(forces)
                        self.history.append(
                            {"time": tick * self.control_dt, "reference": refs.tolist(), "torque": torque.tolist()}
                        )
                    tick += 1
                end = min(target, (step + 1) * self.dt, tick * self.control_dt)
                self.state_in.clear_forces()
                self.solver.step(self.state_in, self.state_out, self.control, None, end - current)
                if any(warning.number for warning in self.solver.mj_data.warning):
                    raise SimulationError("MuJoCo reported a robot simulation warning")
                self.state_in, self.state_out = self.state_out, self.state_in
                current = end
                if current >= (step + 1) * self.dt - 1e-12:
                    step += 1
            q, v = self.state_in.joint_q.numpy(), self.state_in.joint_qd.numpy()
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise SimulationError("nonfinite robot state")
            output.append(
                [
                    (q if kind == "position" else v)[
                        self.bindings["joints"][name]["q_start" if kind == "position" else "qd_start"]
                    ]
                    for kind, name in self.settings["channels"]
                ]
            )
            # CPU .numpy() shares storage with the state reused by later steps.
            positions.append(self.state_in.body_q.numpy()[:, :3].copy())
        return Trace(times, np.asarray(output), np.asarray(positions))
