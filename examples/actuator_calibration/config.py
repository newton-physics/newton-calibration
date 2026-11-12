# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actuator case configuration: joint parameters, fixed controller settings and Template.

Parameter names are ``<joint>.<quantity>``. Parameter groups share bounds; every joint in a
group still receives independent parameters. A bound with equal limits becomes a fixed
parameter. Controller clamps and command delays are fixed parameters as well, so the
Rollout readback checks them in every world.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from newton_calibration import CMAES, Decision, Parameter, Parameters, Template, Validator

SCHEMA = "actuator-calibration/1"
QUANTITIES = ("stiffness", "damping", "dry_friction", "viscous_friction", "armature")
CLAMPS = ("effort_limit", "velocity_limit")
SIMULATION_KEYS = ("physics_freq", "solver", "integrator", "iterations", "ls_iterations")


@dataclass(frozen=True)
class ActuatorCase:
    config: dict
    joints: tuple[str, ...]
    parameters: Parameters
    fixed_parameters: dict[str, float]

    @property
    def control_dt(self):
        return 1.0 / self.config["simulation"]["control_freq"]

    @property
    def buffer_steps(self):
        return round(self.config["dataset"]["buffer_time"] / self.control_dt)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError(f"expected configuration schema {SCHEMA!r}")
    return config


def build_case(config):
    simulation = config["simulation"]
    if missing := [k for k in (*SIMULATION_KEYS, "control_freq") if k not in simulation]:
        raise ValueError(f"simulation settings missing {missing}")
    physics, control = simulation["physics_freq"], simulation["control_freq"]
    if physics % control:
        raise ValueError("physics_freq must be an integer multiple of control_freq")
    if not math.isclose(config["dataset"]["dt"], 1.0 / control, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("dataset dt must equal the control timestep")
    joints, definitions, fixed = [], [], {}
    for group in config["joint_groups"]:
        if set(group["parameters"]) != set(QUANTITIES):
            raise ValueError(f"each joint group must declare exactly {QUANTITIES}")
        delay_steps = round(group.get("delay_ms", 0) / 1000.0 * control)
        for joint in group["names"]:
            if joint in joints:
                raise ValueError(f"joint {joint!r} appears in more than one group")
            joints.append(joint)
            for quantity in QUANTITIES:
                spec = group["parameters"][quantity]
                lower, upper = map(float, spec["bounds"])
                if lower < 0:
                    raise ValueError(f"{joint}.{quantity} bounds must be nonnegative")
                if lower == upper:
                    fixed[f"{joint}.{quantity}"] = lower
                    continue
                definitions.append(
                    Parameter(
                        f"{joint}.{quantity}",
                        (lower, upper),
                        float(spec["baseline"]),
                        unit=spec.get("unit", "1"),
                        transform=spec.get("transform", "linear"),
                    )
                )
            for clamp in CLAMPS:
                fixed[f"{joint}.{clamp}"] = float(group[clamp])
            fixed[f"{joint}.delay_steps"] = float(delay_steps)
    return ActuatorCase(config, tuple(joints), Parameters(definitions), fixed)


def joint_values(values, joints):
    """Split ``<joint>.<quantity>`` dictionaries into per-joint dictionaries."""
    result = {joint: {} for joint in joints}
    for name, value in values.items():
        joint, _, quantity = name.rpartition(".")
        if joint not in result:
            raise ValueError(f"parameter {name!r} does not name a configured joint")
        result[joint][quantity] = float(value)
    return result


def precision_tolerance(case, scored_samples):
    """Express the declared pointwise tolerance in Evaluator feature units.

    Experiment weights equal their scored sample counts and scales are 1 rad, so every
    feature is ``residual / sqrt(N * J)`` for N scored samples over J joints.
    """
    tolerance = float(case.config["pointwise_precision_tolerance_rad"])
    return tolerance / math.sqrt(scored_samples * len(case.joints))


def make_template(case, experiment_ids, scored_samples):
    simulation = case.config["simulation"]
    return Template(
        numeric_settings={k: simulation[k] for k in SIMULATION_KEYS},
        discretization={"control_freq": simulation["control_freq"], "joints": list(case.joints)},
        fixed_parameters=case.fixed_parameters,
        experiment_settings={
            experiment_id: {"seed": 0, "buffer_steps": case.buffer_steps} for experiment_id in experiment_ids
        },
        refinements=tuple(case.config["refinements"]),
        precision_tolerance=precision_tolerance(case, scored_samples),
    )


def make_search(case, train_count, validation_count):
    """CMA-ES over one world per candidate; budgets cover every lifecycle phase exactly."""
    search = case.config["search"]
    world_count = int(search["world_count"])
    generations = int(search["max_generations"])
    optimizer = CMAES(batch_size=world_count, sigma=float(search["sigma"]), seed=search["seed"])
    templates = 1 + len(case.config["refinements"])
    trials_per_training_replay = (
        1  # baseline
        + 3 * templates  # numerical qualification probes
        + 3 * 2 * len(case.parameters.names)  # sensitivity at three anchors
        + world_count * generations  # search
        + templates  # training retest and final numerical retest
    )
    decision = Decision(
        max_trials=world_count * generations,
        max_rollouts=train_count * trials_per_training_replay + validation_count,
        max_generations=generations,
        max_seconds=float(search["max_seconds"]),
        target=0.0,
        patience=search.get("patience"),
        relative_improvement_tolerance=search.get("relative_improvement_tolerance"),
    )
    return optimizer, decision


def make_validator(case):
    acceptance = case.config["acceptance"]
    return Validator(
        max_objective=float(acceptance["max_objective_rad2"]),
        max_error=float(acceptance["max_error_rad"]),
    )
