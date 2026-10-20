# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Prepared observations; acquisition groups enforce independent holdout data."""

from dataclasses import dataclass

import torch

from .data import identity


@dataclass(frozen=True)
class Experiment:
    id: str
    group: str
    reference: torch.Tensor
    times: torch.Tensor | None = None
    scale: float | torch.Tensor = 1.0
    mask: torch.Tensor | None = None
    weight: float = 1.0
    repeats: int = 1

    def normalized(self):
        reference = torch.as_tensor(self.reference, dtype=torch.float64).detach().cpu().clone()
        if reference.ndim != 2 or not reference.numel():
            raise ValueError("reference must be a nonempty [time, channel] tensor")
        times = (
            torch.arange(len(reference), dtype=torch.float64)
            if self.times is None
            else torch.as_tensor(self.times, dtype=torch.float64).detach().cpu().clone()
        )
        mask = (
            torch.ones_like(reference, dtype=torch.bool)
            if self.mask is None
            else torch.as_tensor(self.mask, dtype=torch.bool).detach().cpu().clone()
        )
        scale = torch.broadcast_to(
            torch.as_tensor(self.scale, dtype=torch.float64), reference.shape
        )
        if (
            not self.id
            or not self.group
            or self.repeats < 1
            or not self.weight > 0
            or not torch.isfinite(torch.tensor(self.weight))
        ):
            raise ValueError("invalid experiment identity, weight or repetitions")
        if mask.shape != reference.shape or not mask.any():
            raise ValueError("no valid observations, or mask shape mismatch")
        if not torch.isfinite(reference[mask]).all():
            raise ValueError("nonfinite reference observations")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("observation scales must be positive finite")
        if (
            times.shape != (len(reference),)
            or not torch.isfinite(times).all()
            or (times < 0).any()
            or (times[1:] <= times[:-1]).any()
        ):
            raise ValueError("invalid observation time axis")
        return Experiment(
            self.id, self.group, reference, times, scale.clone(), mask, self.weight, self.repeats
        )


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
        return identity(
            {"train": self.train_data, "validation": self.validation_data, "sources": self.sources}
        )
