# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Named multi-rate robot observations without invented velocity samples."""

import hashlib
import json
from pathlib import Path

import numpy as np

from newton_calibration import Evidence, Experiment

from ..recording import _series


def load_records(records, *, active_joints, driven_joints, scales, sources=None):
    """Mask independent observation time axes; initial states and scales use SI."""
    train, validation, settings = [], [], {}
    for record in records:
        if record["id"] in settings:
            raise ValueError("duplicate robot experiment ID")
        initial = record["initial_state"]
        if set(initial) != {"position", "velocity"} or any(
            set(initial[kind]) != set(active_joints) for kind in initial
        ):
            raise ValueError("declare position and velocity for every active joint at time zero")
        if not all(np.isfinite(list(values.values())).all() for values in initial.values()):
            raise ValueError("robot initial states must be finite SI values")
        if not record["position"]:
            raise ValueError("robot records require observed positions")
        channels, observations = [], []
        for kind, units in (
            ("position", {"rad": 1, "deg": np.pi / 180}),
            ("velocity", {"rad/s": 1, "deg/s": np.pi / 180}),
        ):
            for name in sorted(record.get(kind, {})):
                if name not in active_joints:
                    raise ValueError("observations reference unknown joints")
                channels.append((kind, name))
                observations.append(_series(record[kind][name], units))
        times = np.unique(np.concatenate([t for t, _ in observations]))
        reference = np.zeros((len(times), len(channels)))
        mask = np.zeros_like(reference, dtype=bool)
        for column, (t, values) in enumerate(observations):
            rows = np.searchsorted(times, t)
            reference[rows, column], mask[rows, column] = values, True
        command = {}
        if set(record["command"]) != set(driven_joints):
            raise ValueError("commands must cover exactly the driven joints")
        for name in driven_joints:
            t, values = _series(record["command"][name], {"rad": 1, "deg": np.pi / 180})
            if t[0] != 0 or t[-1] < times[-1]:
                raise ValueError("command must cover simulation from zero through the last observation")
            command[name] = {"times": t.tolist(), "values": values.tolist()}
        exp = Experiment(
            id=record["id"],
            group=record["group"],
            times=times,
            reference=reference,
            mask=mask,
            scale=np.array([scales[kind][name] for kind, name in channels]),
        ).normalized()
        if record["split"] not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        (train if record["split"] == "train" else validation).append(exp)
        settings[exp.id] = {
            "seed": 0,
            "initial_state": initial,
            "command": command,
            "channels": channels,
            "input_hold": "zero_order",
            "observation_alignment": "union_of_times_with_mask_no_interpolation",
            "source_units": {
                kind: {name: signal["unit"] for name, signal in record.get(kind, {}).items()}
                for kind in ("position", "velocity", "command")
            },
        }
    raw = json.dumps(records, sort_keys=True, allow_nan=False).encode()
    evidence = Evidence(
        train, validation, sources={**(sources or {}), "records_sha256": hashlib.sha256(raw).hexdigest()}
    )
    evidence.prepare()
    return evidence, settings


def read_file(path):
    """Load explicit recording provenance and preserve the original file hash."""
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if document["schema"] != "robot-recordings/1" or document["provenance"]["kind"] not in {"synthetic", "measured"}:
        raise ValueError("expected robot-recordings/1 with explicit synthetic/measured provenance")
    return document["records"], {"provenance": document["provenance"], "file_sha256": hashlib.sha256(raw).hexdigest()}
