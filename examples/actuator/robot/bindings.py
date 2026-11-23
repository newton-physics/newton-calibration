# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolve source names through Newton's compiled MuJoCo mappings."""

import numpy as np


def _index(labels, name):
    matches = [i for i, label in enumerate(labels) if label == name or label.endswith("/" + name)]
    if len(matches) != 1:
        raise ValueError(f"missing or ambiguous imported name: {name}")
    return matches[0]


def inspect_model(model, solver, names):
    """Return JSON-compatible bindings and actual physical properties in SI."""
    mj = solver.mj_model
    joint_map = solver.mjc_jnt_to_newton_jnt.numpy()[0]
    body_map = solver.mjc_body_to_newton.numpy()[0]
    q_starts, qd_starts = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    types = model.joint_type.numpy()
    joints, bodies, actuators = {}, {}, {}
    for name in names["joints"]:
        index = _index(model.joint_label, name)
        native = np.flatnonzero(joint_map == index)
        if len(native) > 1:
            raise ValueError(f"joint must have at most one MuJoCo coordinate: {name}")
        mid = int(native[0]) if len(native) else None
        joints[name] = {
            "newton_joint": index,
            "newton_label": model.joint_label[index],
            "newton_type": int(types[index]),
            "q_start": int(q_starts[index]),
            "qd_start": int(qd_starts[index]),
            "mujoco_joint": mid,
            "mujoco_name": mj.joint(mid).name if mid is not None else None,
        }
        if mid is not None:
            dof = int(mj.jnt_dofadr[mid])
            joints[name].update(
                qpos_adr=int(mj.jnt_qposadr[mid]),
                dof_adr=dof,
                axis=mj.jnt_axis[mid].tolist(),
                position=mj.jnt_pos[mid].tolist(),
                limited=bool(mj.jnt_limited[mid]),
                limits=mj.jnt_range[mid].tolist(),
                damping=float(mj.dof_damping[dof]),
                armature=float(mj.dof_armature[dof]),
                friction=float(mj.dof_frictionloss[dof]),
            )
    for name in names["bodies"]:
        index = _index(model.body_label, name)
        native = np.flatnonzero(body_map == index)
        if len(native) != 1:
            raise ValueError(f"body did not survive import: {name}")
        mid = int(native[0])
        bodies[name] = {
            "newton_body": index,
            "newton_label": model.body_label[index],
            "mujoco_body": mid,
            "mujoco_name": mj.body(mid).name,
            "mass": float(mj.body_mass[mid]),
            "center_of_mass": mj.body_ipos[mid].tolist(),
            "principal_inertia": mj.body_inertia[mid].tolist(),
            "inertia_orientation_wxyz": mj.body_iquat[mid].tolist(),
            "position": mj.body_pos[mid].tolist(),
            "orientation_wxyz": mj.body_quat[mid].tolist(),
            "parent": int(mj.body_parentid[mid]),
        }
    native_joint_names = {
        item["mujoco_joint"]: name for name, item in joints.items() if item["mujoco_joint"] is not None
    }
    ctrl_sources = solver.mjc_actuator_ctrl_source.numpy() if mj.nu else []
    ctrl_indices = solver.mjc_actuator_to_newton_idx.numpy() if mj.nu else []
    for index in range(mj.nu):
        if int(ctrl_sources[index]) != 1:
            raise ValueError("unexpected inferred joint-target actuator")
        # The pinned solver does not retain actuator names in its compiled MjModel.
        name = model.mujoco.actuator_label[int(ctrl_indices[index])]
        if name not in names["actuators"]:
            raise ValueError(f"unexpected actuator produced by import: {name}")
        if int(mj.actuator_trntype[index]) != 0 or int(mj.actuator_trnid[index, 0]) not in native_joint_names:
            raise ValueError(f"unsupported actuator transmission: {name}")
        if native_joint_names[int(mj.actuator_trnid[index, 0])] != names["actuator_joints"][name]:
            raise ValueError(f"actuator joint changed during import: {name}")
        actuators[name] = {
            "mujoco_actuator": index,
            "joint": native_joint_names[int(mj.actuator_trnid[index, 0])],
            "control_source": int(ctrl_sources[index]),
            "control_index": int(ctrl_indices[index]),
            "gear": mj.actuator_gear[index].tolist(),
            "gain": mj.actuator_gainprm[index].tolist(),
            "bias": mj.actuator_biasprm[index].tolist(),
            "ctrl_limited": bool(mj.actuator_ctrllimited[index]),
            "ctrl_range": mj.actuator_ctrlrange[index].tolist(),
            "force_limited": bool(mj.actuator_forcelimited[index]),
            "force_range": mj.actuator_forcerange[index].tolist(),
        }
    if set(actuators) != set(names["actuators"]):
        raise ValueError("actuator names changed or disappeared during import")
    if mj.nq != model.joint_coord_count or mj.nv != model.joint_dof_count:
        raise ValueError("imported scalar coordinates do not cover the Newton state")
    return {
        "joints": joints,
        "bodies": bodies,
        "actuators": actuators,
        "nq": int(mj.nq),
        "nv": int(mj.nv),
        "nu": int(mj.nu),
        "gravity": model.gravity.numpy().tolist(),
        "contact_mode": "disabled",
        "units": {"position": "m", "joint_position": "rad", "mass": "kg", "inertia": "kg*m^2"},
    }


def joint_values(values, names, report, *, velocity=False):
    """Map named scalar rotary values to a complete Newton q or qd vector."""
    values = np.asarray(values, dtype=float)
    if len(names) != len(set(names)) or values.shape != (len(names),):
        raise ValueError("joint names and values must agree without duplicates")
    active = {name: item for name, item in report["joints"].items() if item["mujoco_joint"] is not None}
    if set(names) != set(active):
        raise ValueError("provide all active joint values")
    output = np.zeros(report["nv"] if velocity else report["nq"], dtype=np.float32)
    for name, value in zip(names, values, strict=True):
        output[active[name]["qd_start" if velocity else "q_start"]] = value
    if not np.isfinite(output).all():
        raise ValueError("joint values must be finite")
    return output
