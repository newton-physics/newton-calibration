# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose the robot recipe with the existing calibration lifecycle."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from newton_calibration import CMAES, Decision, GridSearch, Parameter, Parameters, Play, Rollout, Template, Tuning

from .evaluation import EvaluatorRobot, ValidatorRobot
from .recording import load_records


class TuningRobot(Tuning):
    """Fit selected joint gains; preserve inherited search, reload and acceptance."""

    def __init__(self, asset, records, config, *, workspace, sources):
        config = json.loads(json.dumps(config, allow_nan=False))
        optimizer_settings = config.get("optimizer", {})
        if not isinstance(optimizer_settings, dict) or set(optimizer_settings) - {"sigma", "seed"}:
            raise ValueError("unknown robot optimizer setting")
        sigma = optimizer_settings.get("sigma", 0.2)
        if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or not 0 < sigma < 1:
            raise ValueError("optimizer sigma must be a finite number between 0 and 1")
        seed = optimizer_settings.get("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("optimizer seed must be an integer in [0, 2**32)")
        if optimizer_settings and len(config["parameters"]) == 1:
            raise ValueError("optimizer settings require multiple parameters for CMA-ES")
        execution = config.get("execution", {})
        if set(execution) - {"device", "worlds", "batch_size"}:
            raise ValueError("unknown robot execution setting")
        device = execution.get("device", "cpu")
        worlds = execution.get("worlds", 128 if device != "cpu" else 1)
        batch_size = execution.get("batch_size", 128 if device != "cpu" else 8)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (worlds, batch_size)):
            raise ValueError("worlds and batch_size must be positive integers")
        if device == "cpu" and worlds != 1:
            raise ValueError("CPU robot execution supports one world")
        asset = asset.with_device(device) if device != asset.config["device"] else asset
        controller = config["controller"]
        native = controller["mode"] == "native_position"
        if controller["mode"] not in {"equivalent_pd", "native_position"}:
            raise ValueError("unsupported robot controller mode")
        period, dt, refined = controller["control_dt"], config["dt"], config["refinement_dt"]
        if not 0 < refined < dt <= period:
            raise ValueError("require 0 < refinement_dt < dt <= control_dt")
        for step in (dt, refined):
            if not math.isclose(period / step, round(period / step), abs_tol=1e-9):
                raise ValueError("control_dt must be divisible by physics timesteps")
        _, _, report = asset.build_model()
        active = {name for name, item in report["joints"].items() if item["mujoco_joint"] is not None}
        if not controller["drives"] or not set(controller["drives"]) <= active:
            raise ValueError("controller requires known active joints")
        if any(not set(controller["drives"]) <= set(record["position"]) for record in records):
            raise ValueError("every capture must observe position for all driven joints")
        fixed = {"control_dt": period}
        for name, drive in controller["drives"].items():
            if set(drive) != ({"kp", "kd", "actuator"} if native else {"kp", "kd", "torque_limit"}):
                raise ValueError("each drive requires kp/kd and native actuator name or equivalent torque_limit")
            for field, value in drive.items():
                if field == "actuator":
                    continue
                if not math.isfinite(value) or (value <= 0 if field == "torque_limit" else value < 0):
                    raise ValueError("invalid fixed controller coefficients")
                fixed[f"{name}.{field}"] = value
        definitions = []
        for name, definition in config["parameters"].items():
            if name not in fixed or not name.endswith((".kp", ".kd")):
                raise ValueError("only selected driven joint kp/kd can be fitted")
            if definition["baseline"] != fixed.pop(name):
                raise ValueError("controller and parameter baselines differ")
            parameter = Parameter(name, **definition, unit="N*m/rad" if name.endswith(".kp") else "N*m*s/rad")
            if parameter.bounds[0] < 0:
                raise ValueError("PD gains must be nonnegative")
            definitions.append(parameter)
        parameters = Parameters(definitions)
        command_names = (
            [drive["actuator"] for drive in controller["drives"].values()]
            if native and controller["command_space"] == "actuator_control"
            else controller["drives"]
        )
        evidence, settings = load_records(
            records,
            active_joints=active,
            driven_joints=command_names,
            scales=config["scales"],
            sources={
                **sources,
                "protocol": config,
                "adapter_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path(__file__).parent.glob("*.py"))
                },
            },
        )
        # Training channels determine the metric layout, independent of holdout values.
        channels = sorted({tuple(channel) for exp in evidence.train_data for channel in settings[exp.id]["channels"]})
        if any(not set(map(tuple, value["channels"])) <= set(channels) for value in settings.values()):
            raise ValueError("validation channels must also be present in training")
        for kind, name in channels:
            limits = config["thresholds"][kind][name]
            if (
                set(limits) != {"rmse", "max"}
                or not np.isfinite(list(limits.values())).all()
                or min(limits.values()) <= 0
            ):
                raise ValueError("declare positive finite physical thresholds per channel")
        super().__init__(
            evidence=evidence,
            asset=asset.with_controller(controller, parameters.as_dict(parameters.baseline)),
            evaluator=EvaluatorRobot(settings, channels),
            parameters=parameters,
            rollout=Rollout(capacity=1, worlds=worlds, parameter_rtol=1e-6),
            optimizer=CMAES(batch_size=batch_size, sigma=sigma, seed=seed)
            if len(definitions) > 1
            else GridSearch(levels=21, batch_size=batch_size),
            decision=Decision(**config["decision"]),
            validator=ValidatorRobot(channels, config["thresholds"], settings=settings),
            template=Template(
                numeric_settings={"solver": "SolverMuJoCo", "device": device, "dt": dt},
                discretization={"joints": report["nq"], "bodies": len(report["bodies"])},
                fixed_parameters=fixed,
                experiment_settings=settings,
                refinements=({"numeric_settings": {"dt": refined}},),
                precision_tolerance=config["precision_tolerance"],
            ),
            play=Play(),
            workspace=workspace,
            orthogonalize=False,
        )
