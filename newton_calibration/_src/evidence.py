# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared observations; acquisition groups enforce independent holdout data."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .data import identity


@dataclass(frozen=True)
class Experiment:
    id: str
    group: str
    reference: np.ndarray
    times: np.ndarray | None = None
    scale: float | np.ndarray = 1.0
    mask: np.ndarray | None = None
    weight: float = 1.0
    repeats: int = 1
    # Measured per-experiment inputs, e.g. recorded commands or initial states. They are
    # evidence, so they are fingerprinted and split with the reference observations.
    inputs: Mapping[str, np.ndarray] | None = None

    def normalized(self):
        reference = np.array(self.reference, dtype=np.float64)
        if reference.ndim != 2 or not reference.size:
            raise ValueError("reference must be a nonempty [time, channel] array")
        times = (
            np.arange(len(reference), dtype=np.float64)
            if self.times is None
            else np.array(self.times, dtype=np.float64)
        )
        mask = np.ones(reference.shape, dtype=bool) if self.mask is None else np.array(self.mask, dtype=bool)
        scale = np.broadcast_to(np.asarray(self.scale, dtype=np.float64), reference.shape)
        if not self.id or not self.group or self.repeats < 1 or not self.weight > 0 or not math.isfinite(self.weight):
            raise ValueError("invalid experiment identity, weight or repetitions")
        if mask.shape != reference.shape or not mask.any():
            raise ValueError("no valid observations, or mask shape mismatch")
        if not np.isfinite(reference[mask]).all():
            raise ValueError("nonfinite reference observations")
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("observation scales must be positive finite")
        if (
            times.shape != (len(reference),)
            or not np.isfinite(times).all()
            or (times < 0).any()
            or (times[1:] <= times[:-1]).any()
        ):
            raise ValueError("invalid observation time axis")
        inputs = None
        if self.inputs is not None:
            if not isinstance(self.inputs, Mapping) or not self.inputs:
                raise ValueError("experiment inputs must be a nonempty mapping of named arrays")
            inputs = {}
            for name, value in self.inputs.items():
                array = np.array(value, dtype=np.float64)
                if not isinstance(name, str) or not name or not array.size:
                    raise ValueError("experiment inputs need nonempty names and arrays")
                if not np.isfinite(array).all():
                    raise ValueError(f"nonfinite experiment input {name!r}")
                inputs[name] = array
        return Experiment(self.id, self.group, reference, times, scale.copy(), mask, self.weight, self.repeats, inputs)


class Evidence:
    def __init__(self, train_data, validation_data=(), *, sources=None):
        self.train_data = tuple(train_data)
        self.validation_data = tuple(validation_data)
        self.sources = dict(sources or {})

    def prepare(self):
        if not self.train_data:
            raise ValueError("training observations required")
        self.train_data = tuple(x.normalized() for x in self.train_data)
        self.validation_data = tuple(x.normalized() for x in self.validation_data)
        all_data = self.train_data + self.validation_data
        if len({x.id for x in all_data}) != len(all_data):
            raise ValueError("duplicate experiment IDs")
        if {x.group for x in self.train_data} & {x.group for x in self.validation_data}:
            raise ValueError("acquisition group leakage between train and validation")
        self.evidence_id = self.fingerprint()

    def fingerprint(self):
        return identity({"train": self.train_data, "validation": self.validation_data, "sources": self.sources})
