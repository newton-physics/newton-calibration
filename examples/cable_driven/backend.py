# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton backend: a cable clamped to a gripper that follows a recorded TCP trajectory.

The cable is a Newton rod (capsule bodies joined by rod joints, solved by VBD). One
capsule, the one that holds the grasp point, is kinematic and driven by the TCP
trajectory through :mod:`examples.driven_attachment`. Every candidate of a batch is one
rod in a shared model; the rods have no contact, so they do not interact.

No optimization or scoring logic lives here.
"""

import math

import newton
import numpy as np
import warp as wp

from newton_calibration import Asset

from ..driven_attachment import DrivenBatchEnvironment, SingleEnvironment
from .chamfer import Camera, chamfer, load_mask_target

PARAMETERS = (
    "bending_rigidity",  # EI [N m^2]
    "bend_damping",  # [N m^2 s], per length
    "stretch_rigidity",  # EA [N]
    "shear_rigidity",  # GA [N]
    "twist_rigidity",  # GJ [N m^2]
    "twist_damping",  # [N m^2 s], per length
    "linear_density",  # [kg/m]
    "radius",  # [m]
)
# Offsets of each response in a rod joint's DOF slice, whose slot order is
# [stretch, shear, bend, twist].
ROD_DOF_OFFSETS = {"stretch": (0,), "shear": (1,), "bend": (2,), "twist": (3,)}
# Largest rest twist [rad] the readback accepts; the rest shape is built without twist.
REST_TWIST_TOLERANCE = 1e-5


def rest_parameter_names(num_elements):
    """Names of the rest-shape parameters: an (x, y) pair for each joint 1 .. num_elements - 1.

    Joint k connects capsule k - 1 to capsule k. Each value is the arc displacement [m]
    at the end of the cable that the joint's rest bend swings: the bend angle times the
    joint's lever (see :func:`rest_levers`).
    """
    return tuple(f"rest_{axis}_{k}" for k in range(1, num_elements) for axis in "xy")


def clamp_segment(clamp_position, cable_length, num_elements):
    """Index of the capsule that holds the grasp point at arc length ``clamp_position`` [m].

    1e-9 lets a grasp on a node pick the capsule that starts there.
    """
    if not 0.0 <= clamp_position <= cable_length:
        raise ValueError("clamp_position_m must lie on the cable")
    return min(int(clamp_position / (cable_length / num_elements) + 1e-9), num_elements - 1)


def cable_layout(cable_length, discretization, clamp_position):
    """Capsule layout of the model: ``(capsules, segment_length, clamp, first_node_arc)``.

    ``discretization["clamp"]`` chooses where the kinematic clamp capsule sits:

    - ``"segment"`` (default): the cable is ``num_elements`` capsules, and the capsule that
      holds the grasp point is the clamp.
    - ``"stub"``: ``num_elements`` capsules model the free cable beyond the grasp point,
      and one more capsule of the same length, inside the gripper, ends at the grasp point
      and is the clamp. The cable inside the gripper is not modelled further.

    ``first_node_arc`` is the arc length [m] along the real cable of the model's first node.
    """
    n = int(discretization["num_elements"])
    mode = discretization.get("clamp", "segment")
    if mode == "segment":
        h = cable_length / n
        return n, h, clamp_segment(clamp_position, cable_length, n), 0.0
    if mode == "stub":
        if not 0.0 <= clamp_position < cable_length:
            raise ValueError("clamp_position_m must lie on the cable")
        h = (cable_length - clamp_position) / n
        return n + 1, h, 0, clamp_position - h
    raise ValueError(f"unknown clamp {mode!r}; expected 'segment' or 'stub'")


def rest_levers(num_elements, segment_length, clamp):
    """Lever [m] of each joint 1 .. num_elements - 1: the cable length it swings.

    That is the length from the joint to the free end on the side away from the clamp
    capsule ``clamp``.
    """
    return [(num_elements - k) * segment_length if k > clamp else k * segment_length for k in range(1, num_elements)]


def bend_quat(alpha, beta):
    """Rest rotation of one joint: the exponential map of the rotation vector (alpha, beta, 0).

    The rotation axis lies in the capsule's cross-section, so the bend adds no twist.
    """
    theta = math.hypot(alpha, beta)
    if theta < 1e-12:
        return wp.quat_identity()
    return wp.quat_from_axis_angle(wp.vec3(alpha / theta, beta / theta, 0.0), theta)


def rotation_vector(q):
    """Rotation vector [rad] of a unit quaternion (x, y, z, w)."""
    v, w = np.asarray(q[:3], dtype=np.float64), float(q[3])
    if w < 0.0:
        v, w = -v, -w
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return 2.0 * v
    return v / s * (2.0 * math.atan2(s, w))


class CableDrivenAsset(Asset):
    """The cable, built as a Newton rod and clamped to the recorded TCP.

    Args:
        baseline_parameters: Start values of the calibrated parameters (see ``PARAMETERS``).
        config: ``cable_length_m``, the cable length [m].
    """

    def create_environment(self, template, parameter_values, experiment_settings):
        batch = self.create_batch_environment(template, [parameter_values], experiment_settings)
        return SingleEnvironment(batch)

    def create_batch_environment(self, template, parameter_values, experiment_settings):
        return CableDrivenBatchEnvironment(template, parameter_values, experiment_settings, self.config)


class CableDrivenBatchEnvironment(DrivenBatchEnvironment):
    """Several cable candidates of one experiment, driven by the same TCP trajectory.

    Numeric settings, besides those of :class:`DrivenBatchEnvironment`: ``iterations``
    (VBD iterations per step). The watched positions are the rod nodes. The discretization
    gives ``num_elements`` and ``clamp`` (see :func:`cable_layout`). Experiment settings,
    besides the drive: ``attachment`` and ``observation``, ``{"camera": {...}, "masks": [...]}``.
    Each observation is one value: the chamfer score (:mod:`.chamfer`) of the projected
    free cable against that observation's mask, by file path.
    """

    def __init__(self, template, parameter_values, settings, config):
        self.iterations = int(template.numeric_settings["iterations"])
        attachment = settings["attachment"]
        clamp_position = float(attachment["clamp_position_m"])
        self.n, self.h, self.clamp_segment, self.first_node_arc = cable_layout(
            float(config["cable_length_m"]), template.discretization, clamp_position
        )
        # Nodes beyond the kinematic capsule when it sits inside the gripper: the visible cable.
        self.visible_from = 1 if template.discretization.get("clamp", "segment") == "stub" else 0
        self.rest_names = rest_parameter_names(self.n)
        expected = set(PARAMETERS) | set(self.rest_names)
        for parameters in parameter_values:
            if set(parameters) != expected:
                raise ValueError(f"parameters must be exactly {sorted(expected)}")

        observation = settings["observation"]
        self.camera = Camera(observation["camera"])
        self.targets = [load_mask_target(path) for path in observation["masks"]]

        self.local_rotation = wp.quat(*(float(v) for v in attachment["transform"][1]))
        self.levers = rest_levers(self.n, self.h, self.clamp_segment)
        # Clamp capsule center, in the attachment frame: the cable runs along local +Z and
        # the grasp point is the frame origin.
        center_offset = self.first_node_arc + (self.clamp_segment + 0.5) * self.h - clamp_position
        self.local_position = wp.vec3(*(float(v) for v in attachment["transform"][0]))
        self.local_position = self.local_position + wp.quat_rotate(
            self.local_rotation, wp.vec3(0.0, 0.0, center_offset)
        )
        super().__init__(template, parameter_values, settings)

    def _add_member(self, builder, parameters, drive_position, drive_rotation):
        rotation = wp.mul(drive_rotation, self.local_rotation)
        origin = drive_position + wp.quat_rotate(drive_rotation, self.local_position)
        # The rod is built in its rest shape. The chain is composed from capsule 0, then
        # moved rigidly so the clamp capsule's center and frame land on the clamp pose.
        frames = [wp.quat_identity()]
        for k, lever in enumerate(self.levers, start=1):
            alpha = float(parameters[f"rest_x_{k}"]) / lever
            beta = float(parameters[f"rest_y_{k}"]) / lever
            frames.append(wp.mul(frames[-1], bend_quat(alpha, beta)))
        points = [wp.vec3(0.0, 0.0, 0.0)]
        for frame in frames:
            points.append(points[-1] + wp.quat_rotate(frame, wp.vec3(0.0, 0.0, self.h)))
        c = self.clamp_segment
        center = (points[c] + points[c + 1]) * 0.5
        to_world = wp.mul(rotation, wp.quat_inverse(frames[c]))
        world_points = [origin + wp.quat_rotate(to_world, p - center) for p in points]
        world_frames = [wp.mul(to_world, frame) for frame in frames]
        radius = float(parameters["radius"])
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        shape_start = builder.shape_count
        bodies, joints = builder.add_rod(
            positions=world_points,
            quaternions=world_frames,
            radius=radius,
            cfg=cfg,
            body_frame_origin="com",
            stretch_stiffness=parameters["stretch_rigidity"] / self.h,
            stretch_damping=0.0,
            shear_stiffness=parameters["shear_rigidity"] / self.h,
            shear_damping=0.0,
            bend_stiffness=parameters["bending_rigidity"] / self.h,
            bend_damping=parameters["bend_damping"] / self.h,
            twist_stiffness=parameters["twist_rigidity"] / self.h,
            twist_damping=parameters["twist_damping"] / self.h,
            wrap_in_articulation=False,
            label="cable",
        )
        # finalize() requires every joint to belong to an articulation.
        builder.add_articulation(list(joints))
        mass = float(parameters["linear_density"]) * self.h
        inertia = np.diag(
            [
                mass * (3.0 * radius**2 + self.h**2) / 12.0,
                mass * (3.0 * radius**2 + self.h**2) / 12.0,
                mass * radius**2 / 2.0,
            ]
        )
        for i, body in enumerate(bodies):
            kinematic = i == self.clamp_segment
            if kinematic:
                builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            builder.body_mass[body] = 0.0 if kinematic else mass
            builder.body_inv_mass[body] = 0.0 if kinematic else 1.0 / mass
            builder.body_inertia[body] = wp.mat33(np.zeros((3, 3)) if kinematic else inertia)
            builder.body_inv_inertia[body] = wp.mat33(np.zeros((3, 3)) if kinematic else np.linalg.inv(inertia))
        return {
            "bodies": np.asarray(bodies, dtype=int),
            "joints": np.asarray(joints, dtype=int),
            "shape": shape_start,
            "driven": [(bodies[self.clamp_segment], self.local_position, self.local_rotation)],
        }

    def _finalize(self, builder):
        builder.color()  # VBD needs the rigid bodies colored.
        model = builder.finalize(device=self.device)
        solver = newton.solvers.SolverVBD(
            model,
            iterations=self.iterations,
            rigid_compliant_alm=True,
            particle_enable_self_contact=False,
        )
        return model, solver

    def _readback(self, member):
        """Return the parameters as the model holds them (float32), in the units of PARAMETERS."""
        slots = self.model.joint_qd_start.numpy()[member["joints"]]
        ke = self.model.joint_target_ke.numpy()
        kd = self.model.joint_target_kd.numpy()
        masses = self.model.body_mass.numpy()[member["bodies"]]
        free = [i for i in range(self.n) if i != self.clamp_segment]

        def uniform(gains, name, mode):
            values = gains[slots[:, None] + np.asarray(ROD_DOF_OFFSETS[mode])]
            if not np.allclose(values, values.flat[0], rtol=1e-6):
                raise RuntimeError(f"{name} differs between the joints or axes of one rod")
            return float(np.mean(values) * self.h)

        # Rest rotation of each joint: VBD takes the rod rest shape from the built body
        # poses, so it is the child body rotation relative to the parent body rotation.
        # Joint k sits at index k - 1 of the rod's joints.
        body_q = self.model.body_q.numpy()
        parents = self.model.joint_parent.numpy()[member["joints"]]
        children = self.model.joint_child.numpy()[member["joints"]]
        rest = {}
        for k, (parent, child, lever) in enumerate(zip(parents, children, self.levers, strict=True), start=1):
            relative = wp.mul(wp.quat_inverse(wp.quat(*body_q[parent][3:])), wp.quat(*body_q[child][3:]))
            rotvec = rotation_vector(np.array(relative))
            if abs(rotvec[2]) > REST_TWIST_TOLERANCE:
                raise RuntimeError(f"rest rotation of joint {k} has a twist of {rotvec[2]:.2e} rad")
            rest[f"rest_x_{k}"] = float(rotvec[0] * lever)
            rest[f"rest_y_{k}"] = float(rotvec[1] * lever)

        return {
            **rest,
            "bending_rigidity": uniform(ke, "bending_rigidity", "bend"),
            "bend_damping": uniform(kd, "bend_damping", "bend"),
            "stretch_rigidity": uniform(ke, "stretch_rigidity", "stretch"),
            "shear_rigidity": uniform(ke, "shear_rigidity", "shear"),
            "twist_rigidity": uniform(ke, "twist_rigidity", "twist"),
            "twist_damping": uniform(kd, "twist_damping", "twist"),
            "linear_density": float(np.mean(masses[free]) / self.h),
            "radius": float(self.model.shape_scale.numpy()[member["shape"]][0]),
        }

    def _positions(self):
        """Return the node positions [m] of every member, shape (members, n + 1, 3)."""
        q = self.state_0.body_q.numpy()
        out = []
        for member in self.members:
            poses = q[member["bodies"]]
            centers = poses[:, :3].astype(np.float64)
            quats = poses[:, 3:].astype(np.float64)
            x, y, z, w = quats.T
            # Local +Z rotated into the world: the segment direction.
            axes = np.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], axis=1)
            starts = centers - 0.5 * self.h * axes
            ends = centers + 0.5 * self.h * axes
            nodes = np.vstack([starts[0], 0.5 * (ends[:-1] + starts[1:]), ends[-1]])
            out.append(nodes)
        return np.stack(out)

    def _observe(self, positions, index):
        """One observation row: the chamfer score (1,)."""
        uv = self.camera.project(positions[self.visible_from :])
        return np.array([chamfer(uv, self.targets[index])])
