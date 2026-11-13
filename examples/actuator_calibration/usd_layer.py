# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# pyright: reportAttributeAccessIssue=false

"""Sparse USD layers holding identified actuator parameters.

The layer sublayers the source asset through a relative path and authors only joint drive,
armature, friction, damping and limit opinions, converted to USD drive units.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

_ANGULAR_DRIVES = frozenset({"angular", "rotX", "rotY", "rotZ"})
_LINEAR_DRIVES = frozenset({"linear", "transX", "transY", "transZ"})
_PARAMETER_NAMES = frozenset({"stiffness", "damping", "armature", "dry_friction", "viscous_friction"})
_RADIANS_PER_DEGREE = math.pi / 180.0


def write_identified_layer(
    source_usd: str | Path,
    output_path: str | Path,
    parameters: Mapping[str, Mapping[str, float]],
    clamps: Mapping[str, Mapping[str, float]],
    metadata: str,
) -> Path:
    """Write ``parameters[joint][quantity]`` and ``clamps[joint]`` as a sparse layer."""

    source_path = Path(os.path.abspath(Path(source_usd).expanduser()))
    destination = Path(os.path.abspath(Path(output_path).expanduser()))
    if not source_path.is_file():
        raise FileNotFoundError(f"source USD does not exist: {source_path}")
    if source_path == destination:
        raise ValueError("identified USD output must differ from the source USD")
    if not parameters or set(clamps) != set(parameters):
        raise ValueError("controller clamps must exactly match the identified joints")

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise ValueError(f"could not open source USD: {source_path}")
    joints = {name: _find_joint(source_stage, name) for name in parameters}
    drive_gain_scaling = _joint_drive_gain_scaling(source_stage)

    destination.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(destination))
    if layer is None:
        raise RuntimeError(f"could not create identified USD layer: {destination}")
    layer.Clear()
    layer.customLayerData = {"newton_calibration": metadata}
    layer.subLayerPaths = [Path(os.path.relpath(source_path, destination.parent)).as_posix()]

    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise RuntimeError(f"could not open identified USD layer: {destination}")
    _copy_stage_coordinate_metadata(source_stage, stage)
    default_prim = source_stage.GetDefaultPrim()
    if default_prim and (prim := stage.GetPrimAtPath(default_prim.GetPath())):
        stage.SetDefaultPrim(prim)

    for joint_name, values in parameters.items():
        _validate_parameters(joint_name, dict(values))
        limits = clamps[joint_name]
        source_joint = joints[joint_name]
        drive_name, is_angular = _drive(source_joint)
        joint = stage.OverridePrim(source_joint.GetPath())
        joint.AddAppliedSchema("NewtonJointAPI")
        drive = UsdPhysics.DriveAPI.Apply(joint, drive_name)
        if not drive:
            raise RuntimeError(f"could not apply the {drive_name!r} drive to {joint_name!r}")
        scale = _drive_gain_export_scale(drive_name, is_angular=is_angular, joint_drive_gain_scaling=drive_gain_scaling)
        drive.CreateStiffnessAttr(float(values["stiffness"]) * scale)
        drive.CreateDampingAttr(float(values["damping"]) * scale)
        drive.CreateMaxForceAttr(float(limits["effort_limit"]))
        _set_float(joint, "newton:armature", values["armature"])
        _set_float(joint, "newton:friction", values["dry_friction"])
        damping_scale = _RADIANS_PER_DEGREE if is_angular else 1.0
        _set_float(joint, "newton:damping", float(values["viscous_friction"]) * damping_scale)
        velocity_scale = 1.0 / _RADIANS_PER_DEGREE if is_angular else 1.0
        _set_float(joint, "newton:velocityLimit", float(limits["velocity_limit"]) * velocity_scale)

    if not layer.Save():
        raise RuntimeError(f"could not save identified USD layer: {destination}")
    return destination


def sublayer_source(layer_path: str | Path) -> Path:
    """The absolute source asset path referenced by an identified layer."""
    layer_path = Path(os.path.abspath(layer_path))
    layer = Sdf.Layer.FindOrOpen(str(layer_path))
    if layer is None or len(layer.subLayerPaths) != 1:
        raise ValueError(f"{layer_path} is not an identified layer with one source sublayer")
    return Path(os.path.normpath(layer_path.parent / layer.subLayerPaths[0]))


def relocate_layer(layer_path: str | Path, destination: str | Path) -> Path:
    """Copy an identified layer and keep its source sublayer resolvable from the new place."""
    source = sublayer_source(layer_path)
    destination = Path(os.path.abspath(destination))
    layer = Sdf.Layer.FindOrOpen(os.path.abspath(layer_path))
    if not layer.Export(str(destination)):
        raise RuntimeError(f"could not export {layer_path} to {destination}")
    copied = Sdf.Layer.FindOrOpen(str(destination))
    copied.Reload(force=True)
    copied.subLayerPaths = [Path(os.path.relpath(source, destination.parent)).as_posix()]
    if not copied.Save():
        raise RuntimeError(f"could not save relocated layer {destination}")
    return destination


def _copy_stage_coordinate_metadata(source_stage: Any, output_stage: Any) -> None:
    """Preserve the source stage's coordinate system and physical units."""

    UsdGeom.SetStageUpAxis(output_stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(
        output_stage,
        UsdGeom.GetStageMetersPerUnit(source_stage),
    )
    UsdPhysics.SetStageKilogramsPerUnit(
        output_stage,
        UsdPhysics.GetStageKilogramsPerUnit(source_stage),
    )


def _find_joint(stage: Any, requested: str) -> Any:
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint) and (str(prim.GetPath()) == requested or prim.GetName() == requested)
    ]
    if len(matches) != 1:
        raise ValueError(f"source USD must contain exactly one joint matching {requested!r}")
    return matches[0]


def _drive(joint: Any) -> tuple[str, bool]:
    if joint.IsA(UsdPhysics.RevoluteJoint):
        return "angular", True
    if joint.IsA(UsdPhysics.PrismaticJoint):
        return "linear", False

    drive_names = _applied_api_instances(joint, "PhysicsDriveAPI")
    if len(drive_names) == 1:
        drive_name = drive_names[0]
    elif not drive_names:
        free_axes = []
        for axis in _applied_api_instances(joint, "PhysicsLimitAPI"):
            limit = UsdPhysics.LimitAPI.Get(joint, axis)
            low = limit.GetLowAttr().Get()
            high = limit.GetHighAttr().Get()
            if low is not None and high is not None and float(low) < float(high):
                free_axes.append(axis)
        if len(free_axes) != 1:
            raise ValueError(f"joint {str(joint.GetPath())!r} must have exactly one driven axis")
        drive_name = free_axes[0]
    else:
        raise ValueError(f"joint {str(joint.GetPath())!r} must have exactly one driven axis")

    if drive_name in _ANGULAR_DRIVES:
        return drive_name, True
    if drive_name in _LINEAR_DRIVES:
        return drive_name, False
    raise ValueError(f"joint {str(joint.GetPath())!r} has unsupported drive {drive_name!r}")


def _applied_api_instances(prim: Any, schema_name: str) -> list[str]:
    prefix = f"{schema_name}:"
    return [str(schema).removeprefix(prefix) for schema in prim.GetAppliedSchemas() if str(schema).startswith(prefix)]


def _joint_drive_gain_scaling(stage: Any) -> float:
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    scaling = 1.0
    if scenes:
        attribute = scenes[0].GetAttribute("newton:joint_drive_gains_scaling")
        value = attribute.Get() if attribute else None
        if value is not None:
            scaling = float(value)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("newton:joint_drive_gains_scaling must be positive and finite")
    return scaling


def _drive_gain_export_scale(
    drive_name: str,
    *,
    is_angular: bool,
    joint_drive_gain_scaling: float,
) -> float:
    if not is_angular:
        return 1.0
    if drive_name == "angular":
        return _RADIANS_PER_DEGREE / joint_drive_gain_scaling
    return _RADIANS_PER_DEGREE * joint_drive_gain_scaling


def _validate_parameters(joint_name: str, parameters: dict[str, float]) -> None:
    names = set(parameters)
    if names != _PARAMETER_NAMES:
        missing = sorted(_PARAMETER_NAMES - names)
        extra = sorted(names - _PARAMETER_NAMES)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(f"invalid parameters for joint {joint_name!r}: " + ", ".join(details))
    values = [float(parameters[name]) for name in _PARAMETER_NAMES]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"identified parameters for joint {joint_name!r} must be finite and nonnegative")


def _set_float(prim: Any, name: str, value: float) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(float(value))
