# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared observations; acquisition groups enforce independent holdout data."""

import math
from dataclasses import dataclass

import numpy as np

from .data import identity


@dataclass(frozen=True)
class Experiment:
    """``Experiment`` holds one experiment's reference outputs and scoring conditions.

    Put control commands, loads, and initial states in ``Template.experiment_settings``
    under the same experiment ID. The environment uses those settings to run the
    forward problem.
    """

    id: str
    """Unique across all training and validation experiments, and a key in ``experiment_settings``."""
    group: str
    """Real acquisition group. Training and validation group labels must be disjoint."""
    reference: np.ndarray
    """Nonempty ``[T,D]`` array. ``T`` times and ``D`` observation channels, converted to
    NumPy ``float64`` during preparation."""
    times: np.ndarray | None = None
    """``None`` or ``[T]`` array. Finite, nonnegative, strictly increasing observation times.
    Record units, such as seconds. Omitted times become ``0, 1, …, T-1``."""
    scale: float | np.ndarray = 1.0
    """Positive finite scalar, ``[D]``, or array broadcastable to ``[T,D]``, used to normalize
    residuals."""
    mask: np.ndarray | None = None
    """Boolean ``[T,D]``, with ``True`` selecting observations for scoring. Defaults to all
    valid, and must retain at least one element."""
    weight: float = 1.0
    """Positive finite relative experiment weight in the combined objective."""
    repeats: int = 1
    """Simulation repeats against the same reference, executed with experiment ``seed + repeat``."""

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
        return Experiment(self.id, self.group, reference, times, scale.copy(), mask, self.weight, self.repeats)


class Evidence:
    """``Evidence`` groups experiments into training and validation data and records their sources."""

    def __init__(self, train_data, validation_data=(), *, sources=None):
        """Initialize the evidence.

        Args:
            train_data: Experiment sequence. At least one experiment for analysis and search.
            validation_data: Held-out experiments for individual evaluation after freezing
                the candidate. Defaults to ``()``.
            sources: Provenance dictionary for data versions, processing protocols, and
                measurement sources, included in the fingerprint. Defaults to ``None``.
        """
        self.train_data = tuple(train_data)
        self.validation_data = tuple(validation_data)
        self.sources = dict(sources or {})

    def prepare(self):
        """``Evidence.prepare()`` normalizes inputs, checks shapes, finiteness, IDs, and groups,
        then creates ``evidence_id``.

        ``Tuning`` calls it during a complete run. For a standalone data check, construct
        ``Evidence``, call ``prepare()``, and inspect ``train_data``, ``validation_data``,
        and ``fingerprint()``.
        """
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
