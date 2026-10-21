# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Two synthetic physics cases assembled from the same public Tuning objects."""

import math
import time

import numpy as np
import torch

from . import (
    CMAES,
    Asset,
    BatchEnvironment,
    Decision,
    Environment,
    Evaluator,
    Evidence,
    Experiment,
    GridSearch,
    Parameter,
    Parameters,
    Play,
    Rollout,
    Template,
    Trace,
    Tuning,
    Validator,
)


def _trace(times, values):
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)
    positions = torch.zeros((len(times), 1, 3), dtype=torch.float64)
    positions[:, 0, 0] = values[:, 0]
    return Trace(times.clone(), values, positions)


class OscillatorEnvironment(Environment):
    def __init__(self, parameters, dt, settings):
        self.applied_parameters = dict(parameters)
        self.dt, self.settings = dt, dict(settings)

    def run(self, times, *, seed, deadline):
        p, s = self.applied_parameters, self.settings
        k, c, mass = p["stiffness"], p["damping"], p["mass"]
        force = s["force"]
        if s["mode"] == "static":
            return _trace(times, np.full(len(times), force / k))
        x, velocity, current = s["initial_x"], s["initial_velocity"], 0.0
        output = []
        for target in times.tolist():
            while current < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("oscillator deadline")
                h = min(self.dt, target - current)

                def acceleration(position, speed):
                    return (force - k * position - c * speed) / mass

                a1 = acceleration(x, velocity)
                v2 = velocity + h * a1 / 2
                a2 = acceleration(x + h * velocity / 2, v2)
                v3 = velocity + h * a2 / 2
                a3 = acceleration(x + h * v2 / 2, v3)
                v4 = velocity + h * a3
                a4 = acceleration(x + h * v3, v4)
                x += h * (velocity + 2 * v2 + 2 * v3 + v4) / 6
                velocity += h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
                current += h
            output.append(x)
        return _trace(times, output)


class OscillatorAsset(Asset):
    def create_environment(self, template, parameter_values, experiment_settings):
        if template.numeric_settings["integrator"] != "rk4":
            raise ValueError("OscillatorAsset supports the declared RK4 integrator")
        return OscillatorEnvironment(
            parameter_values, template.numeric_settings["dt"], experiment_settings
        )


class OscillatorBatchEnvironment(BatchEnvironment):
    """All members integrated together as arrays; the same RK4 steps as OscillatorEnvironment."""

    def __init__(self, parameters, dt, settings):
        self.applied_parameters = [dict(p) for p in parameters]
        self.dt, self.settings = dt, dict(settings)

    def run(self, times, *, seed, deadline):
        s = self.settings
        k = np.array([p["stiffness"] for p in self.applied_parameters], dtype=np.float64)
        c = np.array([p["damping"] for p in self.applied_parameters], dtype=np.float64)
        mass = np.array([p["mass"] for p in self.applied_parameters], dtype=np.float64)
        force = s["force"]
        if s["mode"] == "static":
            equilibrium = force / k
            return [_trace(times, np.full(len(times), value)) for value in equilibrium]
        x = np.full(len(k), float(s["initial_x"]))
        velocity = np.full(len(k), float(s["initial_velocity"]))
        current = 0.0
        output = []
        for target in times.tolist():
            while current < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("oscillator deadline")
                h = min(self.dt, target - current)

                def acceleration(position, speed):
                    return (force - k * position - c * speed) / mass

                a1 = acceleration(x, velocity)
                v2 = velocity + h * a1 / 2
                a2 = acceleration(x + h * velocity / 2, v2)
                v3 = velocity + h * a2 / 2
                a3 = acceleration(x + h * v2 / 2, v3)
                v4 = velocity + h * a3
                a4 = acceleration(x + h * v3, v4)
                x = x + h * (velocity + 2 * v2 + 2 * v3 + v4) / 6
                velocity = velocity + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6
                current += h
            output.append(x.copy())
        series = np.stack(output, axis=1)
        return [_trace(times, series[i]) for i in range(len(k))]


class BatchedOscillatorAsset(OscillatorAsset):
    """OscillatorAsset that also simulates a whole batch in one environment."""

    def create_batch_environment(self, template, parameter_values, experiment_settings):
        if template.numeric_settings["integrator"] != "rk4":
            raise ValueError("OscillatorAsset supports the declared RK4 integrator")
        return OscillatorBatchEnvironment(
            parameter_values, template.numeric_settings["dt"], experiment_settings
        )


def oscillator_reference(times, parameters, settings):
    k, c, mass = parameters["stiffness"], parameters["damping"], parameters["mass"]
    equilibrium = settings["force"] / k
    if settings["mode"] == "static":
        return torch.full((len(times), 1), equilibrium, dtype=torch.float64)
    rate = c / (2 * mass)
    omega = math.sqrt(k / mass - rate**2)
    a = settings["initial_x"] - equilibrium
    b = (settings["initial_velocity"] + rate * a) / omega
    values = equilibrium + torch.exp(-rate * times) * (
        a * torch.cos(omega * times) + b * torch.sin(omega * times)
    )
    return values[:, None]


class SlidingEnvironment(Environment):
    def __init__(self, parameters, dt, settings):
        self.applied_parameters = dict(parameters)
        self.dt, self.settings = dt, dict(settings)

    def run(self, times, *, seed, deadline):
        deceleration = self.applied_parameters["friction"] * self.applied_parameters["gravity"]
        speed = self.settings["initial_speed"]
        x, current = 0.0, 0.0
        values = []
        for target in times.tolist():
            while current < target - 1e-12:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("sliding deadline")
                h = min(self.dt, target - current)
                moving = min(h, speed / deceleration)
                next_speed = max(0.0, speed - deceleration * moving)
                x += (speed + next_speed) * moving / 2
                speed, current = next_speed, current + h
            values.append(x)
        return _trace(times, values)


class SlidingAsset(Asset):
    def create_environment(self, template, parameter_values, experiment_settings):
        if template.numeric_settings["integrator"] != "event_trapezoid":
            raise ValueError("SlidingAsset requires event_trapezoid")
        return SlidingEnvironment(
            parameter_values, template.numeric_settings["dt"], experiment_settings
        )


def make_tuning(
    case, workspace="workspace", *, optimizer="cma_es", capacity=4, batched=False, worlds=None
):
    if case == "oscillator":
        parameters = Parameters(
            [
                Parameter("stiffness", (2.0, 8.0), 4.0, unit="N/m"),
                Parameter("damping", (0.1, 1.3), 0.3, unit="N s/m"),
            ]
        )
        asset_type = BatchedOscillatorAsset if batched else OscillatorAsset
        asset = asset_type(parameters.as_dict(parameters.baseline))
        fixed = {"mass": 1.0}
        truth = {**fixed, "stiffness": 5.0, "damping": 0.7}
        settings = {
            "static": {
                "mode": "static",
                "force": 1.5,
                "initial_x": 0.0,
                "initial_velocity": 0.0,
                "seed": 1,
            },
            "release": {
                "mode": "dynamic",
                "force": 0.0,
                "initial_x": 0.4,
                "initial_velocity": 0.0,
                "seed": 2,
            },
            "driven": {
                "mode": "dynamic",
                "force": 1.0,
                "initial_x": 0.0,
                "initial_velocity": 0.0,
                "seed": 3,
            },
            "holdout": {
                "mode": "dynamic",
                "force": 0.8,
                "initial_x": -0.3,
                "initial_velocity": 0.2,
                "seed": 4,
            },
        }
        experiments = []
        for name, config in settings.items():
            times = (
                torch.tensor([0.0], dtype=torch.float64)
                if name == "static"
                else torch.linspace(0.0, 3.0, 61, dtype=torch.float64)
            )
            experiments.append(
                Experiment(
                    name, name, oscillator_reference(times, truth, config), times, scale=0.05
                )
            )
        numeric = {"dt": 0.02, "integrator": "rk4", "dtype": "float64"}
    elif case == "sliding":
        if batched:
            raise ValueError("the sliding example has no batched asset")
        parameters = Parameters([Parameter("friction", (0.05, 0.65), 0.15)])
        asset = SlidingAsset(parameters.as_dict(parameters.baseline))
        fixed = {"gravity": 9.81}
        settings = {
            "slow": {"initial_speed": 1.2, "seed": 1},
            "fast": {"initial_speed": 2.1, "seed": 2},
            "holdout": {"initial_speed": 1.7, "seed": 3},
        }
        experiments = []
        for name, config in settings.items():
            times = torch.linspace(0.0, 1.0, 41, dtype=torch.float64)
            deceleration = 0.35 * fixed["gravity"]
            moving = times.clamp_max(config["initial_speed"] / deceleration)
            positions = config["initial_speed"] * moving - 0.5 * deceleration * moving**2
            experiments.append(Experiment(name, name, positions[:, None], times, scale=0.03))
        numeric = {"dt": 0.02, "integrator": "event_trapezoid", "dtype": "float64"}
    else:
        raise ValueError(f"unknown example {case}")
    template = Template(
        numeric_settings=numeric,
        discretization={"degrees_of_freedom": 1},
        fixed_parameters=fixed,
        experiment_settings=settings,
        refinements=({"numeric_settings": {"dt": 0.01}},),
        precision_tolerance=0.02,
    )
    strategy = GridSearch(levels=41, batch_size=32) if optimizer == "grid" else CMAES()
    return Tuning(
        evidence=Evidence(
            experiments[:-1],
            experiments[-1:],
            sources={"kind": "independent analytical synthetic references"},
        ),
        asset=asset,
        evaluator=Evaluator(max_normalized_error=2.0),
        parameters=parameters,
        rollout=Rollout(capacity, worlds=worlds),
        optimizer=strategy,
        decision=Decision(
            max_trials=2000 if optimizer == "grid" else 400,
            max_generations=100,
            target=1e-4,
            max_seconds=180,
        ),
        validator=Validator(max_objective=0.04, max_error=0.6),
        template=template,
        play=Play(),
        workspace=workspace,
    )
