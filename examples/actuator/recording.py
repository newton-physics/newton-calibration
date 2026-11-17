# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate offline records and convert named channels to SI observations."""

import hashlib
import json
from pathlib import Path

import numpy as np

from newton_calibration import Evidence, Experiment


def _series(record, units):
    times = np.asarray(record["times"], dtype=float)
    values = np.asarray(record["values"], dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or values.shape != times.shape
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
        or times[0] < 0
        or (np.diff(times) <= 0).any()
    ):
        raise ValueError("signals require finite values and strictly increasing nonnegative times")
    if record["unit"] not in units:
        raise ValueError("unsupported signal unit")
    return times, values * units[record["unit"]]


def load_records(records, *, scales, sources=None):
    """Create Evidence and experiment settings from records; scales use SI units."""
    train, validation, settings = [], [], {}
    for record in records:
        if record["id"] in settings:
            raise ValueError("duplicate experiment ID")
        command_times, command = _series(record["command"], {"rad": 1.0, "deg": np.pi / 180})
        times, q = _series(record["position"], {"rad": 1.0, "deg": np.pi / 180})
        if command_times[0] != 0 or command_times[-1] < times[-1]:
            raise ValueError("command must cover simulation from zero through the last observation")
        channels, columns = ["position"], [q]
        if "velocity" in record:
            vt, velocity = _series(record["velocity"], {"rad/s": 1.0, "deg/s": np.pi / 180})
            if vt[0] > times[0] or vt[-1] < times[-1]:
                raise ValueError("velocity must cover position observation times")
            columns.append(np.interp(times, vt, velocity))
            channels.append("velocity")
        initial = np.asarray(record["initial_state"], dtype=float)
        if initial.shape != (2,) or not np.isfinite(initial).all():
            raise ValueError("initial_state must be finite [q in rad, qd in rad/s] at time zero")
        exp = Experiment(
            id=record["id"],
            group=record["group"],
            times=times,
            reference=np.column_stack(columns),
            scale=np.array([scales[name] for name in channels]),
        ).normalized()
        if record["split"] not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        (train if record["split"] == "train" else validation).append(exp)
        settings[exp.id] = {
            "seed": 0,
            "initial_state": initial.tolist(),
            "channels": channels,
            "command_times": command_times.tolist(),
            "command": command.tolist(),
            "joint": "actuator",
            "input_hold": "zero_order",
            "source_units": {name: record[name]["unit"] for name in ("command", *channels)},
            "observation_units": ["rad" if name == "position" else "rad/s" for name in channels],
            "velocity_alignment": "linear_interpolation_no_extrapolation" if "velocity" in record else "absent",
        }
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()
    evidence = Evidence(train, validation, sources={**(sources or {}), "records_sha256": digest})
    evidence.prepare()
    return evidence, settings


def read_file(path):
    """Read the versioned JSON recording format and retain its content hash."""
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document["schema"] != "actuator-recordings/1":
        raise ValueError("unsupported recording schema")
    return document["records"], {"kind": "offline_recording", "file_sha256": hashlib.sha256(raw).hexdigest()}
