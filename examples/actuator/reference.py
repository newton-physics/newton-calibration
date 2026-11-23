# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent analytical reference for a sampled controller and viscous rotor."""

import numpy as np


def analytical_response(command, *, control_dt, inertia, damping, kp, kd, torque_limit, initial=(0.0, 0.0)):
    """Integrate held torque exactly between controller samples; return q, qd in SI."""
    q, velocity = initial
    output = []
    decay = np.exp(-damping * control_dt / inertia)
    for target in command:
        output.append((q, velocity))
        torque = np.clip(kp * (target - q) - kd * velocity, -torque_limit, torque_limit)
        if damping == 0:
            q += velocity * control_dt + torque * control_dt**2 / (2 * inertia)
            velocity += torque * control_dt / inertia
        else:
            equilibrium = torque / damping
            q += equilibrium * control_dt + (velocity - equilibrium) * inertia / damping * (1 - decay)
            velocity = equilibrium + (velocity - equilibrium) * decay
    return np.asarray(output)


def synthetic_records(*, kp=2.0, kd=0.15, duration=1.0):
    """Generate two training captures and one distinct validation excitation."""
    dt = 0.002
    times = np.arange(round(duration / dt) + 1) * dt
    records = []
    for index, (a, b) in enumerate(((1.3, 3.7), (2.1, 6.2), (1.7, 4.9))):
        command = 0.15 * np.sin(2 * np.pi * a * times) + 0.05 * np.sin(2 * np.pi * b * times)
        states = analytical_response(
            command,
            control_dt=dt,
            inertia=0.01,
            damping=0.02,
            kp=kp,
            kd=kd,
            torque_limit=2.0,
        )
        records.append(
            {
                "id": f"capture-{index}",
                "group": f"capture-{index}",
                "split": "train" if index < 2 else "validation",
                "initial_state": [0.0, 0.0],
                "command": {"times": times.tolist(), "values": command.tolist(), "unit": "rad"},
                "position": {"times": times.tolist(), "values": states[:, 0].tolist(), "unit": "rad"},
                "velocity": {"times": times.tolist(), "values": states[:, 1].tolist(), "unit": "rad/s"},
            }
        )
    return records
