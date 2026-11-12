# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Long-form Parquet recordings to Evidence.

Field names are mapped first, then each experiment is linearly interpolated onto a uniform
grid that starts at the latest first sample and ends at the earliest last sample over
*all* recorded fields.

Each recording becomes one Experiment. ``reference`` holds the measured positions after
each control interval, ``inputs`` hold the commands and the measured initial state, and
the first ``buffer_steps`` rows are masked so the replay can settle without being scored.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np

from newton_calibration import Evidence, Experiment
from newton_calibration.sources import FileSourceHuggingFace, FileSourceLocal

REQUIRED_COLUMNS = ("time_ns", "experiment", "field", "value")


def _dt_ns(dt):
    value = Decimal(str(dt)) * 1_000_000_000
    if not value.is_finite() or value <= 0 or value != value.to_integral_value():
        raise ValueError("dt must be a positive whole number of nanoseconds")
    return int(value)


def _mapped(field, joint_name_mapping):
    joint, separator, feature = field.rpartition("/")
    if not separator:
        return field
    return f"{joint_name_mapping.get(joint, joint)}/{feature}"


def resample(frame, dt, joint_name_mapping=None):
    """Return ``{experiment: (time_ns [T], {field: values [T]})}`` in first-seen order."""
    import pandas as pd  # noqa: PLC0415 (optional actuator extra)

    if missing := [c for c in REQUIRED_COLUMNS if c not in frame.columns]:
        raise ValueError(f"recording is missing columns {missing}")
    if frame.loc[:, list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("recording alignment columns contain nulls")
    if not pd.api.types.is_integer_dtype(frame["time_ns"]):
        raise ValueError("time_ns must contain integer nanoseconds")
    step = _dt_ns(dt)
    mapping = dict(joint_name_mapping or {})
    fields = frame["field"].map(lambda x: _mapped(x, mapping))
    frame = frame.assign(field=fields)
    if frame.duplicated(subset=["experiment", "time_ns", "field"]).any():
        raise ValueError("recording contains duplicate field timestamps or mapping collisions")
    result = {}
    for experiment, episode in frame.groupby("experiment", sort=False):
        signals = {field: group.sort_values("time_ns") for field, group in episode.groupby("field", sort=False)}
        start = max(int(s["time_ns"].iloc[0]) for s in signals.values())
        end = min(int(s["time_ns"].iloc[-1]) for s in signals.values())
        if start > end:
            raise ValueError(f"experiment {experiment!r} has no overlapping signal range")
        count = (end - start) // step + 1
        target = start + np.arange(count, dtype=np.int64) * step
        offset = (target - start).astype(np.float64)
        values = {
            field: np.interp(
                offset,
                (s["time_ns"].to_numpy(dtype=np.int64) - start).astype(np.float64),
                s["value"].to_numpy(dtype=np.float64),
            )
            for field, s in signals.items()
        }
        result[str(experiment)] = (target, values)
    return result


def trajectories(path, joints, dt, joint_name_mapping=None):
    """Return ``[(experiment_id, command [T, J], measured [T, J])]`` for one file."""
    import pandas as pd  # noqa: PLC0415 (optional actuator extra)

    episodes = resample(pd.read_parquet(path), dt, joint_name_mapping)
    output = []
    for experiment, (_, values) in episodes.items():
        columns = {}
        for kind in ("command_q", "actual_q"):
            names = [f"{joint}/{kind}" for joint in joints]
            if missing := [n for n in names if n not in values]:
                raise ValueError(f"experiment {experiment!r} is missing fields {missing}")
            columns[kind] = np.stack([values[n] for n in names], axis=1)
        if len(columns["actual_q"]) < 2:
            raise ValueError(f"experiment {experiment!r} needs at least two aligned samples")
        output.append((experiment, columns["command_q"], columns["actual_q"]))
    return output


def experiment(experiment_id, command, measured, *, dt, buffer_steps, group=None):
    """One recording as an Experiment, scored as pooled position error.

    The replay starts from ``measured[0]`` and applies ``command[i]`` during control
    interval i, so ``reference[i] = measured[i + 1]``. Weights equal the scored sample
    count, which makes the weighted objective the pooled position MSE over all recordings.
    """
    transitions = len(measured) - 1
    scored = transitions - buffer_steps
    if scored < 1:
        raise ValueError(f"{experiment_id}: no scored samples after the {buffer_steps}-step buffer")
    mask = np.ones((transitions, measured.shape[1]), dtype=bool)
    mask[:buffer_steps] = False
    return Experiment(
        experiment_id,
        group or experiment_id,
        np.array(measured[1:], dtype=np.float64),
        dt * np.arange(1, transitions + 1, dtype=np.float64),
        scale=1.0,
        mask=mask,
        weight=float(scored),
        inputs={
            "command_q": np.array(command, dtype=np.float64),
            "initial_q": np.array(measured[0], dtype=np.float64),
        },
    )


def file_source(dataset, data_root=None):
    """The dataset's files: from ``data_root`` if given, else the pinned Hugging Face revision."""
    if data_root is not None:
        return FileSourceLocal(data_root)
    if dataset.get("provider") == "huggingface":
        return FileSourceHuggingFace(dataset["repo_id"], dataset["revision"])
    raise ValueError("pass data_root for a dataset without a supported provider")


def resolve_asset(config, usd_path=None, data_root=None):
    """The robot USD: ``usd_path`` if given, else the configured dataset file; sha256-checked."""
    asset = config["asset"]
    if usd_path is None:
        if "path" not in asset:
            raise ValueError("pass a USD path; the configuration names no dataset asset")
        return file_source(config["dataset"], data_root).fetch(asset["path"], asset["usd_sha256"])
    path = Path(usd_path).expanduser().resolve()
    return FileSourceLocal(path.parent).fetch(path.name, asset["usd_sha256"])


def load_evidence(case, data_root=None):
    dataset = case.config["dataset"]
    source = file_source(dataset, data_root)
    splits = {}
    for split in ("train", "validation"):
        experiments = []
        for relative, path in source.fetch_all(dataset[split]).items():
            for experiment_id, command, measured in trajectories(
                path, case.joints, dataset["dt"], dataset.get("joint_name_mapping")
            ):
                # One acquisition group per recording file.
                experiments.append(
                    experiment(
                        experiment_id,
                        command,
                        measured,
                        dt=dataset["dt"],
                        buffer_steps=case.buffer_steps,
                        group=relative,
                    )
                )
        splits[split] = experiments
    sources = {
        key: dataset[key]
        for key in ("provider", "repo_id", "revision", "dt", "buffer_time", "joint_name_mapping")
        if key in dataset
    }
    sources["files"] = {**dataset["train"], **dataset["validation"]}
    sources["resampling"] = "linear interpolation over the overlap of all recorded fields"
    return Evidence(splits["train"], splits["validation"], sources=sources)
