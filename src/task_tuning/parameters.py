# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Parameter encoding and feasible combinations; analysis belongs to Tuning."""

import math
from dataclasses import dataclass

import torch

from .data import identity


@dataclass(frozen=True)
class Parameter:
    name: str
    bounds: tuple[float, float]
    baseline: float
    unit: str = "1"
    transform: str = "linear"

    def __post_init__(self):
        lo, hi = self.bounds
        if (
            not self.name
            or not all(math.isfinite(x) for x in (lo, hi, self.baseline))
            or not lo < hi
            or not lo <= self.baseline <= hi
        ):
            raise ValueError("invalid parameter bounds or baseline")
        if self.transform not in {"linear", "log"} or (self.transform == "log" and lo <= 0):
            raise ValueError("invalid parameter transform")


class Parameters:
    def __init__(self, definitions):
        self.definitions = tuple(definitions)
        self.names = tuple(x.name for x in self.definitions)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("parameter names must be nonempty and unique")
        self._bounds = torch.tensor([x.bounds for x in self.definitions], dtype=torch.float64)
        self._baseline = torch.tensor([x.baseline for x in self.definitions], dtype=torch.float64)
        self._log = torch.tensor([x.transform == "log" for x in self.definitions])
        self._limits = self.bounds.clone()
        self._limits[self._log] = self.bounds[self._log].log()
        self._frozen = False
        self.set_mapping(torch.zeros(len(self.names)), torch.eye(len(self.names)))

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("Parameters is frozen; create a new parameter space")
        object.__setattr__(self, name, value)

    @property
    def bounds(self):
        return self._bounds.clone()

    @property
    def baseline(self):
        return self._baseline.clone()

    def to_normalized(self, values):
        values = torch.as_tensor(values, dtype=torch.float64).cpu().clone()
        if values.shape[-1] != len(self.names):
            raise ValueError("physical parameter width mismatch")
        values[..., self._log] = values[..., self._log].log()
        return (values - self._limits[:, 0]) / (self._limits[:, 1] - self._limits[:, 0])

    def from_normalized(self, values):
        values = torch.as_tensor(values, dtype=torch.float64).cpu()
        physical = values * (self._limits[:, 1] - self._limits[:, 0]) + self._limits[:, 0]
        physical[..., self._log] = physical[..., self._log].exp()
        return physical

    def set_mapping(self, origin, basis):
        if self._frozen:
            raise RuntimeError("parameter mapping is frozen")
        p = len(self.names)
        origin = torch.as_tensor(origin, dtype=torch.float64).cpu().clone()
        basis = torch.as_tensor(basis, dtype=torch.float64).cpu().clone()
        if (
            origin.shape != (p,)
            or basis.ndim != 2
            or basis.shape[0] != p
            or not 1 <= basis.shape[1] <= p
            or not torch.isfinite(origin).all()
            or not torch.isfinite(basis).all()
        ):
            raise ValueError("invalid combination mapping")
        if not torch.allclose(
            basis.T @ basis, torch.eye(basis.shape[1], dtype=torch.float64), atol=1e-8, rtol=1e-8
        ):
            raise ValueError("combination basis must have orthonormal columns")
        self._origin, self._basis = origin, basis
        self._low = torch.minimum(-origin[:, None] * basis, (1 - origin[:, None]) * basis).sum(
            dim=0
        )
        self._high = torch.maximum(-origin[:, None] * basis, (1 - origin[:, None]) * basis).sum(
            dim=0
        )

    @property
    def dimension(self):
        return self._basis.shape[1]

    def snapshot(self):
        return {
            "definitions": self.definitions,
            "origin": self._origin.clone(),
            "basis": self._basis.clone(),
            "search_low": self._low.clone(),
            "search_high": self._high.clone(),
        }

    @property
    def parameter_space_id(self):
        return identity(self.snapshot())

    def freeze(self):
        self._frozen = True

    def encode(self, values):
        x = self.to_normalized(values) - self._origin
        z = x @ self._basis
        if not torch.allclose(z @ self._basis.T, x, atol=1e-7, rtol=1e-7):
            raise ValueError("physical parameters are outside the selected subspace")
        return (z - self._low) / (self._high - self._low)

    def decode(self, values):
        values = torch.as_tensor(values, dtype=torch.float64).cpu()
        if values.shape[-1] != self.dimension:
            raise ValueError("search dimension mismatch")
        z = self._low + values * (self._high - self._low)
        return self.from_normalized(self._origin + z @ self._basis.T)

    def check_bounds(self, values):
        values = torch.as_tensor(values, dtype=torch.float64).cpu()
        return (
            torch.isfinite(values).all(dim=-1)
            & (values >= self.bounds[:, 0] - 1e-12).all(dim=-1)
            & (values <= self.bounds[:, 1] + 1e-12).all(dim=-1)
        )

    def as_dict(self, row):
        return dict(zip(self.names, map(float, row), strict=True))
