"""Newton VBD rod adapter. No optimization or scoring logic lives here."""

from __future__ import annotations

import math
import time

import newton
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from task_tuning import Asset, Environment, SimulationError, Trace


def solver_frame_rotation(template):
    """Rotate the entire physical problem; observations stay in laboratory axes."""
    return Rotation.from_euler(
        "y",
        template.numeric_settings.get("solver_frame_y_rotation_deg", 0.0),
        degrees=True,
    )


def planar_rod_quaternions(points):
    """Preserve small XZ-plane slopes without float32 parallel-axis snapping."""
    edges = np.diff(np.asarray(points), axis=0)
    if np.max(np.abs(edges[:, 1])) > 1e-10:
        raise ValueError("cable orientation helper requires the declared XZ plane")
    angles = np.arctan2(edges[:, 0], edges[:, 2])
    return [wp.quat(0.0, math.sin(a / 2), 0.0, math.cos(a / 2)) for a in angles]


class CableAsset(Asset):
    def create_environment(self, template, parameter_values, experiment_settings):
        return CableEnvironment(
            template, parameter_values, experiment_settings, self.config
        )


class CableEnvironment(Environment):
    def __init__(self, template, parameters, experiment, config, *, build_only=False):
        self.template, self.settings = template, dict(experiment)
        self.parameters = dict(parameters)
        numeric, discrete = template.numeric_settings, template.discretization
        self.device = wp.get_device(numeric["device"])
        self.solver_frame = solver_frame_rotation(template)
        self.dt = float(
            numeric.get("dynamic_dt", numeric["dt"])
            if experiment["mode"] == "dynamic"
            else numeric["dt"]
        )
        self.block_steps = int(numeric["block_steps"])
        self.relaxation_damping = (
            float(numeric["static_relaxation_damping"])
            if experiment["mode"] == "static"
            else 0.0
        )
        self.length = parameters["free_length"]
        self.n = int(discrete["segments"])
        self.h = self.length / self.n
        if self.dt <= 0 or self.n < 4 or numeric["dtype"] != "float32":
            raise ValueError("invalid Newton numerical settings")
        if self.settings.get("contact", False):
            raise ValueError(
                "this adapter requires an explicitly contact-free experiment"
            )
        self.samples = np.asarray(experiment["sample_arclengths"], dtype=float)
        if np.any(self.samples < 0) or np.any(self.samples > self.length + 1e-9):
            raise ValueError("observations must lie on the free cable")
        angle = float(experiment["clamp_angle"])
        self.clamp_angle_parameter = experiment.get("clamp_angle_parameter")
        if self.clamp_angle_parameter is not None:
            angle += float(experiment.get("clamp_angle_sign", 1.0)) * math.radians(
                parameters[self.clamp_angle_parameter]
            )
        tangent = np.array([math.cos(angle), 0.0, math.sin(angle)])
        points = np.arange(-1, self.n + 1)[:, None] * self.h * tangent
        if "rest_centerline" in experiment:
            from scipy.interpolate import CubicSpline

            rest = CubicSpline(
                experiment["rest_arclengths"], experiment["rest_centerline"], axis=0
            )
            natural = rest(np.linspace(0, self.length, self.n + 1))
            rotation = Rotation.from_rotvec([0.0, -angle, 0.0])
            points[1:] = rotation.apply(natural)
        points = self.solver_frame.apply(points)
        quaternions = planar_rod_quaternions(points)
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder(
                gravity=tuple(
                    self.solver_frame.apply([0.0, 0.0, -parameters["gravity"]])
                )
            )
            cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0, has_shape_collision=False, has_particle_collision=False
            )
            bodies, joints = builder.add_rod(
                positions=points,
                quaternions=quaternions,
                radius=parameters["radius"],
                cfg=cfg,
                body_frame_origin="com",
                stretch_stiffness=parameters["stretch_stiffness"]
                * (self.length / 32)
                / self.h,
                shear_stiffness=parameters["shear_stiffness"]
                * (self.length / 32)
                / self.h,
                stretch_damping=0.0,
                shear_damping=0.0,
                bend_stiffness=parameters["bending_rigidity"] / self.h,
                bend_damping=(parameters["bend_damping"] + self.relaxation_damping)
                * (self.length / 32)
                / self.h,
                twist_stiffness=parameters["twist_stiffness"]
                * (self.length / 32)
                / self.h,
                twist_damping=parameters["twist_damping"] * (self.length / 32) / self.h,
                wrap_in_articulation=False,
                label="cable",
            )
            # A fixed stub is OUTSIDE the measured free length. The clamp-to-first
            # segment-center dual span is h/2; interior dual spans are h.
            first = builder.joint_qd_start[joints[0]]
            builder.joint_target_ke[first + 2] *= 2.0
            builder.joint_target_kd[first + 2] *= 2.0
            self.clamp_compliance = parameters.get("clamp_rotational_compliance")
            if self.clamp_compliance is not None:
                if self.clamp_compliance < 0:
                    raise ValueError("clamp rotational compliance cannot be negative")
                # Series elastic compliance: free first half-segment + mount.
                rod_ke = 2.0 * parameters["bending_rigidity"] / self.h
                weight = 1.0 / (1.0 + rod_ke * self.clamp_compliance)
                builder.joint_target_ke[first + 2] = rod_ke * weight
                # Low-frequency Kelvin-Voigt equivalent of a damped half-rod
                # in series with an elastic mount. This is an effective 32-rod
                # boundary, not an independently measured clamp rheology.
                physical_kd = (
                    2.0 * parameters["bend_damping"] * (self.length / 32) / self.h
                )
                self.root_relaxation = (
                    float(numeric["static_clamp_relaxation_damping"])
                    if experiment["mode"] == "static"
                    else 0.0
                )
                builder.joint_target_kd[first + 2] = (
                    physical_kd * weight**2 + self.root_relaxation
                )
            mass = parameters["linear_density"] * self.h
            radius = parameters["radius"]
            diagonal = np.array(
                [
                    mass * (3 * radius**2 + self.h**2) / 12,
                    mass * (3 * radius**2 + self.h**2) / 12,
                    mass * radius**2 / 2,
                ]
            )
            for index in bodies:
                fixed = index == bodies[0]
                if fixed:
                    builder.body_flags[index] = int(newton.BodyFlags.KINEMATIC)
                builder.body_mass[index] = 0.0 if fixed else mass
                builder.body_inv_mass[index] = 0.0 if fixed else 1.0 / mass
                builder.body_inertia[index] = wp.mat33(
                    np.zeros((3, 3)) if fixed else np.diag(diagonal)
                )
                builder.body_inv_inertia[index] = wp.mat33(
                    np.zeros((3, 3)) if fixed else np.diag(1 / diagonal)
                )
            self.plug_body = None
            if experiment["with_plug"]:
                plug_length = parameters["plug_length"]
                plug_mass = parameters["plug_mass"]
                width, height = config["plug_cross_section_m"]
                inertia = (
                    plug_mass
                    / 12
                    * np.array(
                        [
                            height**2 + plug_length**2,
                            width**2 + plug_length**2,
                            width**2 + height**2,
                        ]
                    )
                )
                self.plug_body = builder.add_link(
                    xform=wp.transform(
                        wp.vec3(
                            *(
                                points[-1]
                                + Rotation.from_quat(np.asarray(quaternions[-1])).apply(
                                    [0, 0, plug_length / 2]
                                )
                            )
                        ),
                        quaternions[-1],
                    ),
                    mass=plug_mass,
                    inertia=wp.mat33(np.diag(inertia)),
                    label="plug",
                )
                builder.add_shape_box(
                    self.plug_body,
                    hx=width / 2,
                    hy=height / 2,
                    hz=plug_length / 2,
                    cfg=cfg,
                )
                joint = builder.add_joint_fixed(
                    bodies[-1],
                    self.plug_body,
                    parent_xform=wp.transform(
                        wp.vec3(0, 0, self.h / 2), wp.quat_identity()
                    ),
                    child_xform=wp.transform(
                        wp.vec3(0, 0, -plug_length / 2), wp.quat_identity()
                    ),
                )
                joints.append(joint)
            builder.add_articulation(joints)
            if self.plug_body is not None:
                # A rigid attachment needs no extra iterative constraint. Merge
                # mass, COM and inertia exactly while retaining the rod frame.
                remap = builder.collapse_fixed_joints()
                bodies = [remap["body_remap"][body] for body in bodies]
                joints = [remap["joint_remap"][joint] for joint in joints[:-1]]
                self.plug_body = bodies[-1]
            self.builder = builder
            self.bodies = np.asarray(bodies, dtype=int)
            self.joints = np.asarray(joints, dtype=int)
            if not build_only:
                self._initialize(builder, [self])

    def _initialize(self, builder, parts):
        self.parts = parts
        numeric = self.template.numeric_settings
        builder.color()
        self.model = builder.finalize(device=self.device)
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=int(
                numeric.get("dynamic_iterations", numeric["iterations"])
                if self.settings["mode"] == "dynamic"
                else numeric["iterations"]
            ),
            rigid_compliant_alm=numeric["rigid_compliant_alm"],
            rigid_avbd_joint_alpha=numeric["rigid_avbd_joint_alpha"],
            rigid_avbd_gamma=numeric["rigid_avbd_gamma"],
            rigid_joint_linear_ke=numeric["attachment_linear_ke"],
            rigid_joint_angular_ke=numeric["attachment_angular_ke"],
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
            particle_enable_self_contact=False,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        for part in parts:
            part.model = self.model
            part.state_0, part.state_1 = self.state_0, self.state_1
            if "initial_centerline" in part.settings:
                part._set_initial_curve(np.asarray(part.settings["initial_centerline"]))
            part._readback()
        self.graph = None
        self._graph_start_state = self.state_0
        self._initial_q = self.state_0.body_q.numpy().copy()
        self._initial_qd = self.state_0.body_qd.numpy().copy()
        # A block has an even number of swaps, so graph launches always start
        # from the same state buffer. Each environment owns its solver history.
        if self.block_steps % 2:
            raise ValueError("block_steps must be even")
        if self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:
                self._steps(self.block_steps, self.dt)
            self.graph = capture.graph
            self.state_0.body_q.assign(self._initial_q)
            self.state_1.body_q.assign(self._initial_q)
            self.state_0.body_qd.assign(self._initial_qd)
            self.state_1.body_qd.assign(self._initial_qd)

    def _readback(self):
        self.applied_parameters = dict(self.parameters)
        slots = self.model.joint_qd_start.numpy()[self.joints[: self.n]]
        coefficients = self.model.joint_target_ke.numpy()
        dampings = self.model.joint_target_kd.numpy()
        factors = np.ones(self.n) * self.h
        factors[0] /= 2.0
        for name, actual in (
            ("bending_rigidity", coefficients[slots + 2] * factors),
            (
                "bend_damping",
                dampings[slots + 2] * factors / (self.length / 32)
                - self.relaxation_damping,
            ),
        ):
            if self.clamp_compliance is not None:
                actual = actual[1:]
            if not np.allclose(actual, self.parameters[name], rtol=2e-6, atol=1e-8):
                raise RuntimeError(f"actual rod joint {name} binding mismatch")
            self.applied_parameters[name] = float(actual.mean())
        for name, actual in (
            ("stretch_stiffness", coefficients[slots] * self.h / (self.length / 32)),
            ("shear_stiffness", coefficients[slots + 1] * self.h / (self.length / 32)),
            ("twist_stiffness", coefficients[slots + 3] * self.h / (self.length / 32)),
            ("twist_damping", dampings[slots + 3] * self.h / (self.length / 32)),
        ):
            if not np.allclose(actual, self.parameters[name], rtol=2e-6, atol=1e-8):
                raise RuntimeError(f"actual rod joint {name} binding mismatch")
            self.applied_parameters[name] = float(actual.mean())
        if self.clamp_compliance is not None:
            actual_ei = self.applied_parameters["bending_rigidity"]
            actual_compliance = 1.0 / float(coefficients[slots[0] + 2]) - self.h / (
                2.0 * actual_ei
            )
            if not math.isclose(
                actual_compliance, self.clamp_compliance, rel_tol=2e-6, abs_tol=2e-5
            ):
                raise RuntimeError("actual clamp compliance binding mismatch")
            self.applied_parameters["clamp_rotational_compliance"] = actual_compliance
            weight = 1.0 / (1.0 + 2.0 * actual_ei / self.h * self.clamp_compliance)
            expected_kd = (
                2.0
                * self.parameters["bend_damping"]
                * (self.length / 32)
                / self.h
                * weight**2
                + self.root_relaxation
            )
            if not math.isclose(
                float(dampings[slots[0] + 2]), expected_kd, rel_tol=2e-6, abs_tol=1e-8
            ):
                raise RuntimeError("actual clamp damping binding mismatch")
        if self.clamp_angle_parameter is not None:
            # Read the actual kinematic stub orientation, not the requested value.
            stub = self.model.body_q.numpy()[self.bodies[0]]
            direction = self.solver_frame.inv().apply(
                Rotation.from_quat(stub[3:]).apply([0.0, 0.0, 1.0])
            )
            delta = (
                math.atan2(direction[2], direction[0]) - self.settings["clamp_angle"]
            )
            actual = math.degrees(math.atan2(math.sin(delta), math.cos(delta))) / float(
                self.settings.get("clamp_angle_sign", 1.0)
            )
            if not math.isclose(
                actual, self.parameters[self.clamp_angle_parameter], abs_tol=2e-5
            ):
                raise RuntimeError("actual clamp angle binding mismatch")
            self.applied_parameters[self.clamp_angle_parameter] = actual
        masses = self.model.body_mass.numpy()
        cable_bodies = (
            self.bodies[1:-1] if self.plug_body is not None else self.bodies[1:]
        )
        density = float(masses[cable_bodies].mean()) / self.h
        self.applied_parameters["linear_density"] = density
        if self.plug_body is not None:
            self.applied_parameters["plug_mass"] = (
                float(masses[self.plug_body]) - density * self.h
            )

    def _set_initial_curve(self, curve):
        from scipy.interpolate import CubicSpline

        # Natural rest shape was frozen before setting the live release state.
        s = np.asarray(self.settings["initial_arclengths"])
        spline = CubicSpline(s, curve, axis=0)
        points = self.solver_frame.apply(
            spline(np.linspace(0.0, self.length, self.n + 1))
        )
        quats = newton.utils.rod_parallel_transport_quaternions(
            [wp.vec3(*p) for p in points]
        )
        q = self.state_0.body_q.numpy()
        for i, body in enumerate(self.bodies[1:]):
            q[body, :3] = (points[i] + points[i + 1]) / 2
            q[body, 3:] = np.asarray(quats[i])
        self.state_0.body_q.assign(q)
        self.state_1.body_q.assign(q)
        if "initial_centerline_velocity" in self.settings:
            velocity_curve = CubicSpline(
                s, np.asarray(self.settings["initial_centerline_velocity"]), axis=0
            )
            node_s = np.linspace(0.0, self.length, self.n + 1)
            speeds = self.solver_frame.apply(velocity_curve(node_s))
            qd = self.state_0.body_qd.numpy()
            for i, body in enumerate(self.bodies[1:]):
                edge = points[i + 1] - points[i]
                edge_speed = speeds[i + 1] - speeds[i]
                qd[body, :3] = (speeds[i] + speeds[i + 1]) / 2
                qd[body, 3:] = np.cross(edge, edge_speed) / np.dot(edge, edge)
            # Newton stores linear velocity at COM; the merged terminal body
            # has a COM offset from its unchanged rod-midpoint frame.
            offsets = Rotation.from_quat(q[self.bodies[1:], 3:]).apply(
                self.model.body_com.numpy()[self.bodies[1:]]
            )
            qd[self.bodies[1:], :3] += np.cross(qd[self.bodies[1:], 3:], offsets)
            self.state_0.body_qd.assign(qd)
            self.state_1.body_qd.assign(qd)

    def close(self):
        self.graph = None
        self.parts = []

    def _steps(self, count, dt):
        for _ in range(count):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _positions(self):
        from scipy.interpolate import CubicSpline

        q = self.state_0.body_q.numpy()[self.bodies[1:]]
        directions = Rotation.from_quat(q[:, 3:]).apply(np.tile([0, 0, 1], (self.n, 1)))
        left = q[:, :3] - directions * self.h / 2
        right = q[:, :3] + directions * self.h / 2
        nodes = np.vstack([left[0], (right[:-1] + left[1:]) / 2, right[-1]])
        # C2 readout on material arclength, independent of the optimization trial.
        nodes = self.solver_frame.inv().apply(nodes)
        curve = CubicSpline(np.linspace(0, self.length, self.n + 1), nodes, axis=0)
        return curve(self.samples), nodes

    def run(self, times, *, seed, deadline):
        result = self.run_batch(times, seed=seed, deadline=deadline)[0]
        if isinstance(result, SimulationError):
            raise result
        return result

    def run_batch(self, times, *, seed, deadline):
        current = 0.0
        output = [[] for _ in self.parts]
        positions = [[] for _ in self.parts]
        settle = (
            float(self.settings["settle_time_s"])
            if self.settings["mode"] == "static"
            else 0.0
        )
        with wp.ScopedDevice(self.device):
            targets = [settle + value for value in times.tolist()]
            if settle:
                targets.insert(0, settle - 0.1)
            for target in targets:
                while current < target - 1e-10:
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("Newton cable deadline")
                    remaining = target - current
                    block_dt = self.dt * self.block_steps
                    if (
                        self.graph is not None
                        and self.state_0 is self._graph_start_state
                        and remaining >= block_dt - 1e-10
                    ):
                        wp.capture_launch(self.graph)
                        current += block_dt
                    else:
                        dt = min(self.dt, remaining)
                        self._steps(1, dt)
                        current += dt
                for i, part in enumerate(self.parts):
                    part.state_0 = self.state_0
                    points, nodes = part._positions()
                    if (
                        not np.isfinite(points).all()
                        or np.max(np.abs(points)) > 10 * self.length
                    ):
                        raise SimulationError("nonfinite or diverged cable")
                    if np.max(np.abs(nodes[:, 1])) > 1e-4:
                        raise SimulationError("unexpected out-of-plane motion")
                    output[i].append(points[:, [0, 2]].reshape(-1))
                    positions[i].append(points)
        traces = []
        for values, xyz in zip(output, positions, strict=True):
            if settle:
                drift = np.max(np.abs(values[1] - values[0]))
                if drift > 0.0001:
                    traces.append(
                        SimulationError(
                            f"static equilibrium drift {drift:g} m exceeds 0.1 mm"
                        )
                    )
                    continue
                values, xyz = values[1:], xyz[1:]
            traces.append(
                Trace(
                    times.clone(),
                    torch.tensor(np.array(values), dtype=torch.float64),
                    torch.tensor(np.array(xyz), dtype=torch.float64),
                )
            )
        return traces
