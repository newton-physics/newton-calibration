# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A sampled position controller driving a Newton rotor on CPU."""

import time

import numpy as np

from newton_calibration import Asset, Environment, SimulationError, Trace


def solver_settings(solver):
    """Return the supported, explicit numerical protocol for each CPU backend."""
    common = {"solver": solver, "device": "cpu"}
    if solver == "SolverFeatherstone":
        return {**common, "angular_damping": 0.0}
    if solver == "SolverMuJoCo":
        return {
            **common,
            "use_mujoco_cpu": True,
            "integrator": "implicitfast",
            "disable_contacts": True,
            "update_data_interval": 1,
        }
    raise ValueError(f"unsupported actuator solver: {solver}")


class EnvironmentActuator(Environment):
    """Own one rotor and controller; position [rad], velocity [rad/s], torque [N m]."""

    def __init__(self, parameters, numeric, settings):
        import newton  # noqa: PLC0415 -- optional backend, imported only when simulating
        import warp as wp  # noqa: PLC0415

        self.dt = numeric["dt"]
        self.solver_name = numeric["solver"]
        self.settings = settings
        # These fields are the controller's actual runtime coefficients.
        self.kp, self.kd = parameters["kp"], parameters["kd"]
        self.torque_limit = parameters["torque_limit"]
        self.control_dt = parameters["control_dt"]
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = parameters["inertia"]
        body = builder.add_link(mass=1.0, inertia=wp.mat33(inertia, 0, 0, 0, inertia, 0, 0, 0, inertia))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=(0.0, 0.0, 1.0),
            label="actuator",
            target_ke=0.0,
            target_kd=0.0,
            damping=parameters["damping"],
            armature=0.0,
            friction=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
        )
        builder.add_articulation([joint])
        self.model = builder.finalize(device="cpu")
        options = {key: value for key, value in numeric.items() if key not in {"dt", "solver", "device"}}
        if self.solver_name == "SolverMuJoCo":
            self.solver = newton.solvers.SolverMuJoCo(self.model, **options)
        else:
            self.solver = newton.solvers.SolverFeatherstone(self.model, **options)
        self.control = self.model.control()
        self.state_in, self.state_out = self.model.state(), self.model.state()
        self.state_in.joint_q.assign(np.array([settings["initial_state"][0]], dtype=np.float32))
        self.state_in.joint_qd.assign(np.array([settings["initial_state"][1]], dtype=np.float32))
        newton.eval_fk(self.model, self.state_in.joint_q, self.state_in.joint_qd, self.state_in)
        self.control_history = []
        self._used = False

    @property
    def applied_parameters(self):
        """Read physical model fields and the controller coefficients used by run."""
        if self.solver_name == "SolverMuJoCo":
            # The CPU solver owns a separate compiled model. This isotropic rotor
            # is its only non-world body and its only degree of freedom.
            inertia = float(self.solver.mj_model.body_inertia[1, 2])
            damping = float(self.solver.mj_model.dof_damping[0])
        else:
            inertia = float(self.model.body_inertia.numpy()[0, 2, 2])
            damping = float(self.model.joint_damping.numpy()[0])
        return {
            "kp": self.kp,
            "kd": self.kd,
            "inertia": inertia,
            "damping": damping,
            "torque_limit": self.torque_limit,
            "control_dt": self.control_dt,
        }

    def run(self, times, *, seed, deadline):
        """Return observations at requested times without moving controller ticks."""
        if self._used:
            raise RuntimeError("create a fresh actuator environment for every episode")
        self._used = True
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not len(times)
            or not np.isfinite(times).all()
            or times[0] < 0
            or (np.diff(times) <= 0).any()
        ):
            raise ValueError("invalid observation times")
        ct, command = np.asarray(self.settings["command_times"]), np.asarray(self.settings["command"])
        if ct[0] != 0 or ct[-1] < times[-1]:
            raise ValueError("command does not cover simulation")
        current, tick, step = 0.0, 0, 0
        output = []
        for target in times:
            while current < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("actuator rollout deadline")
                if current >= tick * self.control_dt - 1e-12:
                    q = float(self.state_in.joint_q.numpy()[0])
                    velocity = float(self.state_in.joint_qd.numpy()[0])
                    # A command exactly at a controller tick is visible at that tick.
                    index = np.searchsorted(ct, tick * self.control_dt + 1e-12, side="right") - 1
                    torque = float(
                        np.clip(
                            self.kp * (command[index] - q) - self.kd * velocity, -self.torque_limit, self.torque_limit
                        )
                    )
                    if not np.isfinite(torque):
                        raise SimulationError("nonfinite controller output")
                    self.control.joint_f.assign(np.array([torque], dtype=np.float32))
                    self.control_history.append((tick * self.control_dt, torque))
                    tick += 1
                while (step + 1) * self.dt <= current + 1e-12:
                    step += 1
                end = min(target, tick * self.control_dt, (step + 1) * self.dt)
                self.state_in.clear_forces()
                self.solver.step(self.state_in, self.state_out, self.control, None, end - current)
                if self.solver_name == "SolverMuJoCo" and np.any(self.solver.mj_data.warning.number):
                    # MuJoCo can reset divergent state to finite values. Such a
                    # rollout must not become a plausible calibration candidate.
                    raise SimulationError("MuJoCo reported a simulation warning")
                self.state_in, self.state_out = self.state_out, self.state_in
                current = end
            output.append((float(self.state_in.joint_q.numpy()[0]), float(self.state_in.joint_qd.numpy()[0])))
        output = np.asarray(output)
        if not np.isfinite(output).all():
            raise SimulationError("nonfinite rotor state")
        columns = [0 if channel == "position" else 1 for channel in self.settings["channels"]]
        positions = np.zeros((len(times), 2, 3))
        positions[:, 1, 0] = 0.1 * np.cos(output[:, 0])
        positions[:, 1, 1] = 0.1 * np.sin(output[:, 0])
        return Trace(times.copy(), output[:, columns], positions)


class AssetActuator(Asset):
    """Reconstruct a CPU rotor and sampled controller from exported configuration."""

    def create_environment(self, template, parameter_values, experiment_settings):
        numeric = template.numeric_settings
        expected_numeric = solver_settings(numeric["solver"])
        expected = {
            "model": "rotor-pd/1",
            "solver": numeric["solver"],
            "device": "cpu",
            "model_parameters": {name: parameter_values[name] for name in ("inertia", "damping")},
        }
        if dict(self.config) != expected:
            raise ValueError("unsupported actuator asset configuration")
        if {key: value for key, value in numeric.items() if key != "dt"} != expected_numeric or not 0 < numeric[
            "dt"
        ] <= parameter_values["control_dt"]:
            raise ValueError("invalid actuator numerical settings")
        if (
            experiment_settings["input_hold"] != "zero_order"
            or experiment_settings["joint"] != "actuator"
            or tuple(experiment_settings["channels"]) not in {("position",), ("position", "velocity")}
        ):
            raise ValueError("unsupported actuator observation or control mapping")
        return EnvironmentActuator(parameter_values, numeric, experiment_settings)
