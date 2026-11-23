# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent planar two-link dynamics for the bundled robot fixtures."""

import numpy as np


def main():
    """Save reproducible synthetic inputs in a new directory."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Generate independent two-link evidence; no hardware data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(synthetic_config(), indent=2) + "\n")
    document = {
        "schema": "robot-recordings/1",
        "provenance": {"kind": "synthetic", "reference": "two-link Lagrange equations integrated by DOP853"},
        "records": synthetic_records(),
    }
    (args.output / "recordings.json").write_text(json.dumps(document, indent=2) + "\n")


def two_link_response(command, *, control_dt, kp, kd, limits, initial, gravity=0.0):
    """Integrate Lagrange dynamics at held torque; no Newton or MuJoCo calls.

    The fixture uses Y-axis hinges and links along X. Positive angles rotate
    toward -Z. Inertias are about each link's center of mass [kg m²].
    """
    from scipy.integrate import solve_ivp  # noqa: PLC0415 -- optional robot dependency

    state = np.array(initial, dtype=float)
    states = []
    kp, kd, limits = np.asarray(kp), np.asarray(kd), np.asarray(limits)

    def dynamics(_, value, torque):
        q1, q2, v1, v2 = value
        # m1=2, m2=1, l1=1, r1=.5, r2=.3; I1yy=.18, I2yy=.04.
        coupling = 0.3 * np.cos(q2)
        mass = np.array([[1.81 + 2 * coupling, 0.13 + coupling], [0.13 + coupling, 0.13]])
        h = 0.3 * np.sin(q2)
        coriolis = np.array([-h * (2 * v1 * v2 + v2**2), h * v1**2])
        weight = -gravity * np.array([2.0 * np.cos(q1) + 0.3 * np.cos(q1 + q2), 0.3 * np.cos(q1 + q2)])
        acceleration = np.linalg.solve(mass, torque - coriolis - weight - np.array([0.02 * v1, 0.015 * v2]))
        return [v1, v2, *acceleration]

    for target in command:
        states.append(state.copy())
        torque = np.clip(kp * (target - state[:2]) - kd * state[2:], -limits, limits)
        result = solve_ivp(dynamics, (0, control_dt), state, args=(torque,), method="DOP853", rtol=1e-11, atol=1e-13)
        if not result.success:
            raise RuntimeError(result.message)
        state = result.y[:, -1]
    return np.asarray(states)


def synthetic_records(*, duration=1.2):
    """Generate distinct multi-frequency captures with nonzero initial states."""
    times = np.arange(round(duration / 0.002) + 1) * 0.002
    names, records = ("shoulder", "elbow"), []
    for index, (a, b) in enumerate(((0.7, 1.9), (1.1, 2.7), (0.9, 2.3))):
        command = np.column_stack(
            (0.15 + 0.25 * np.sin(2 * np.pi * a * times), -0.1 + 0.2 * np.sin(2 * np.pi * b * times + 0.3))
        )
        initial = [0.1 + 0.03 * index, -0.15 - 0.04 * index, 0.02, -0.03]
        states = two_link_response(command, control_dt=0.002, kp=[18, 6], kd=[2, 0.5], limits=[25, 8], initial=initial)
        record = {
            "id": f"two-link-{index}",
            "group": f"capture-{index}",
            "split": "train" if index < 2 else "validation",
            "initial_state": {
                "position": dict(zip(names, initial[:2], strict=True)),
                "velocity": dict(zip(names, initial[2:], strict=True)),
            },
        }
        for kind, values, unit in (
            ("command", command, "rad"),
            ("position", states[:, :2], "rad"),
            ("velocity", states[:, 2:], "rad/s"),
        ):
            record[kind] = {
                name: {"times": times.tolist(), "values": values[:, i].tolist(), "unit": unit}
                for i, name in enumerate(names)
            }
        records.append(record)
    return records


def synthetic_config():
    """Return fixture-specific bounds and gates; these are not hardware defaults."""
    drives = {
        "shoulder": {"kp": 12.0, "kd": 1.1, "torque_limit": 25.0},
        "elbow": {"kp": 4.0, "kd": 0.3, "torque_limit": 8.0},
    }
    bounds = {"shoulder.kp": [8, 32], "shoulder.kd": [0.8, 3.5], "elbow.kp": [2.5, 10], "elbow.kd": [0.2, 1.0]}
    return {
        "controller": {"mode": "equivalent_pd", "control_dt": 0.002, "drives": drives},
        "parameters": {
            f"{joint}.{key}": {"baseline": drive[key], "bounds": bounds[f"{joint}.{key}"]}
            for joint, drive in drives.items()
            for key in ("kp", "kd")
        },
        "dt": 0.00025,
        "refinement_dt": 0.000125,
        "scales": {kind: dict.fromkeys(drives, value) for kind, value in (("position", 0.01), ("velocity", 0.05))},
        "thresholds": {
            kind: {joint: {"rmse": rmse, "max": maximum} for joint in drives}
            for kind, rmse, maximum in (("position", 0.01, 0.03), ("velocity", 0.05, 0.15))
        },
        "precision_tolerance": 0.05,
        "decision": {
            "max_trials": 800,
            "max_rollouts": 2200,
            "max_generations": 95,
            "max_seconds": 3600,
            "target": 1e-4,
            "patience": 20,
            "improvement_tolerance": 1e-8,
        },
    }


if __name__ == "__main__":
    main()
