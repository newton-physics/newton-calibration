# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assemble the actuator recipe without replacing the shared lifecycle."""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

from newton_calibration import CMAES, Decision, GridSearch, Parameter, Parameters, Play, Rollout, Template, Tuning

from .backend import AssetActuator, solver_settings
from .evaluation import EvaluatorActuator, ValidatorActuator
from .recording import load_records


def synthetic_config(*, solver="SolverFeatherstone"):
    """Return the explicit protocol for the independently generated demo."""
    return {
        "solver": solver,
        "fixed": {"inertia": 0.01, "damping": 0.02, "torque_limit": 2.0, "control_dt": 0.002},
        "parameters": {
            "kp": {"bounds": [0.5, 5.0], "baseline": 1.0},
            "kd": {"bounds": [0.03, 0.4], "baseline": 0.08},
        },
        "dt": 0.00025,
        "refinement_dt": 0.000125,
        "scales": {"position": 0.01, "velocity": 0.05},
        "thresholds": {"position_rmse": 0.01, "position_max": 0.03, "velocity_rmse": 0.05, "velocity_max": 0.15},
        "precision_tolerance": 0.05,
        "decision": {
            "max_trials": 512,
            "max_rollouts": 2000,
            "max_generations": 64,
            "max_seconds": 1800,
            "target": 1e-4,
            "patience": 12,
            "improvement_tolerance": 1e-7,
        },
    }


class TuningActuator(Tuning):
    """Configure single-joint calibration; run() is inherited from Tuning."""

    def __init__(self, records, config, *, workspace="workspace/actuator", optimizer="cma_es", sources=None):
        config = json.loads(json.dumps(config, allow_nan=False))
        config.setdefault("solver", "SolverFeatherstone")
        numeric = solver_settings(config["solver"])
        fixed = config["fixed"]
        if set(fixed) != {"inertia", "damping", "torque_limit", "control_dt"}:
            raise ValueError("declare inertia, damping, torque_limit and control_dt")
        for name, value in fixed.items():
            if not math.isfinite(value) or (value < 0 if name == "damping" else value <= 0):
                raise ValueError(f"invalid {name}")
        dt, refinement = config["dt"], config["refinement_dt"]
        if not 0 < refinement < dt <= fixed["control_dt"]:
            raise ValueError("require 0 < refinement_dt < dt <= control_dt")
        for value in (dt, refinement):
            ratio = fixed["control_dt"] / value
            if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
                raise ValueError("control_dt must be an integer multiple of both physics timesteps")
        if set(config["parameters"]) != {"kp", "kd"}:
            raise ValueError("the actuator recipe identifies exactly kp and kd")
        for key in ("position", "velocity"):
            if not math.isfinite(config["scales"][key]) or config["scales"][key] <= 0:
                raise ValueError("observation scales must be positive finite SI values")
        for key in ("position_rmse", "position_max", "velocity_rmse", "velocity_max"):
            if not math.isfinite(config["thresholds"][key]) or config["thresholds"][key] <= 0:
                raise ValueError("physical thresholds must be positive finite SI values")
        definitions = [
            Parameter(name, **config["parameters"][name], unit=unit)
            for name, unit in (("kp", "N*m/rad"), ("kd", "N*m*s/rad"))
        ]
        if any(p.bounds[0] < 0 for p in definitions):
            raise ValueError("PD gains must be nonnegative")
        parameters = Parameters(definitions)
        provenance = {
            **(sources or {}),
            "protocol": config,
            "adapter_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
        versions = {}
        packages = ["newton", "warp-lang"]
        if config["solver"] == "SolverMuJoCo":
            packages.extend(("mujoco", "mujoco-warp"))
        for package in packages:
            try:
                distribution = importlib.metadata.distribution(package)
                versions[package] = {
                    "version": distribution.version,
                    "direct_url": json.loads(distribution.read_text("direct_url.json") or "null"),
                }
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        provenance["backend_versions"] = versions
        evidence, settings = load_records(records, scales=config["scales"], sources=provenance)
        template = Template(
            numeric_settings={**numeric, "dt": dt},
            discretization={"joints": 1, "bodies": 1},
            fixed_parameters=fixed,
            experiment_settings=settings,
            refinements=({"numeric_settings": {"dt": refinement}},),
            precision_tolerance=config["precision_tolerance"],
        )
        if optimizer not in {"cma_es", "grid"}:
            raise ValueError("optimizer must be cma_es or grid")
        strategy = (
            CMAES(batch_size=8, sigma=0.2, seed=42) if optimizer == "cma_es" else GridSearch(levels=21, batch_size=8)
        )
        super().__init__(
            evidence=evidence,
            asset=AssetActuator(
                parameters.as_dict(parameters.baseline),
                config={
                    "model": "rotor-pd/1",
                    "solver": config["solver"],
                    "device": "cpu",
                    "model_parameters": {name: fixed[name] for name in ("inertia", "damping")},
                },
            ),
            evaluator=EvaluatorActuator(settings),
            parameters=parameters,
            rollout=Rollout(capacity=1, parameter_rtol=1e-6),
            optimizer=strategy,
            decision=Decision(**config["decision"]),
            validator=ValidatorActuator(config["thresholds"]),
            template=template,
            play=Play(),
            workspace=workspace,
            orthogonalize=False,
        )
