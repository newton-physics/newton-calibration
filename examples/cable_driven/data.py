# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read driven-cable recordings into calibration experiments and experiment settings.

A recording is a directory with a ``recording.json``. It holds the observation times, the
TCP drive that moved the clamp, the attachment of the cable to the TCP, the ``camera``
(pose in the drive frame, Newton camera frame, and pinhole intrinsics) and ``images.masks``
(one cable mask per observation, paths relative to the recording), fitted with the chamfer
score.

``images.rgb`` and ``source`` (where the data came from) are not read here. The
experiments and the experiment settings are built from the recording directly, so the data
exists in one place.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from newton_calibration import Experiment

CASE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Recording:
    """One recorded driven cable motion."""

    directory: Path
    data: dict

    @property
    def id(self):
        return self.data["experiment"]["id"]


def load_recording(directory):
    """Read one recording."""
    directory = Path(directory)
    data = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
    return Recording(directory, data)


def load_recordings(root):
    """Read every recording below ``root``, sorted by experiment id."""
    recordings = [load_recording(path.parent) for path in Path(root).glob("*/recording.json")]
    if not recordings:
        raise ValueError(f"no recordings below {root}")
    ids = [r.id for r in recordings]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate experiment ids below {root}: {sorted(ids)}")
    return sorted(recordings, key=lambda r: r.id)


def experiment(recording, *, scale):
    """Return the calibration experiment of a recording.

    Args:
        recording: The recording.
        scale: Expected chamfer score of a good fit [cable lengths on screen].

    The trace is one chamfer score per observation and the reference is 0. Each recording
    is its own group, so it is never split between training and validation.
    """
    block = recording.data["experiment"]
    return Experiment(
        id=block["id"],
        group=block["id"],
        reference=np.zeros((len(block["times"]), 1)),
        times=block["times"],
        scale=float(scale),
    ).normalized()


def file_sha256(path):
    """Return the SHA-256 of a file's content, as hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment_settings(recording, *, seed):
    """Return the settings the environment needs to replay a recording.

    The scene is deterministic and does not use ``seed``, but the Template requires one.
    The settings hold the mask paths and their SHA-256, so the evidence identity changes
    with the mask content.
    """
    settings = recording.data["settings"]
    if settings.get("attachment") is None:
        raise ValueError(f"{recording.directory}: the recording has no attachment")
    masks = [str((recording.directory / path).resolve()) for path in recording.data["images"]["masks"]]
    if len(masks) != len(recording.data["experiment"]["times"]):
        raise ValueError(f"{recording.directory}: one mask per observation is required")
    observation = {
        "camera": dict(recording.data["camera"]),
        "masks": masks,
        "masks_sha256": [file_sha256(path) for path in masks],
    }
    return {
        "fps": settings["fps"],
        "attachment": dict(settings["attachment"]),
        "observation": observation,
        "drive": dict(recording.data["drive"]),
        "seed": seed,
    }
