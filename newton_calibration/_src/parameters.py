# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parameter encoding and feasible combinations; analysis belongs to Tuning."""

import math
from dataclasses import dataclass

import numpy as np

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
        self._bounds = np.array([x.bounds for x in self.definitions], dtype=np.float64)
        self._baseline = np.array([x.baseline for x in self.definitions], dtype=np.float64)
        self._log = np.array([x.transform == "log" for x in self.definitions], dtype=bool)
        self._limits = self.bounds
        self._limits[self._log] = np.log(self.bounds[self._log])
        self._frozen = False
        self.set_mapping(np.zeros(len(self.names)), np.eye(len(self.names)))

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("Parameters is frozen; create a new parameter space")
        object.__setattr__(self, name, value)

    @property
    def bounds(self):
        return self._bounds.copy()

    @property
    def baseline(self):
        return self._baseline.copy()

    def to_normalized(self, values):
        values = np.array(values, dtype=np.float64)
        if values.shape[-1] != len(self.names):
            raise ValueError("physical parameter width mismatch")
        values[..., self._log] = np.log(values[..., self._log])
        return (values - self._limits[:, 0]) / (self._limits[:, 1] - self._limits[:, 0])

    def from_normalized(self, values):
        values = np.asarray(values, dtype=np.float64)
        physical = values * (self._limits[:, 1] - self._limits[:, 0]) + self._limits[:, 0]
        physical[..., self._log] = np.exp(physical[..., self._log])
        return physical

    def set_mapping(self, origin, basis):
        if self._frozen:
            raise RuntimeError("parameter mapping is frozen")
        p = len(self.names)
        origin = np.array(origin, dtype=np.float64)
        basis = np.array(basis, dtype=np.float64)
        if (
            origin.shape != (p,)
            or basis.ndim != 2
            or basis.shape[0] != p
            or not 1 <= basis.shape[1] <= p
            or not np.isfinite(origin).all()
            or not np.isfinite(basis).all()
        ):
            raise ValueError("invalid combination mapping")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8, rtol=1e-8):
            raise ValueError("combination basis must have orthonormal columns")
        self._origin, self._basis = origin, basis
        self._low = np.minimum(-origin[:, None] * basis, (1 - origin[:, None]) * basis).sum(axis=0)
        self._high = np.maximum(-origin[:, None] * basis, (1 - origin[:, None]) * basis).sum(axis=0)

    @property
    def dimension(self):
        return self._basis.shape[1]

    def snapshot(self):
        return {
            "definitions": self.definitions,
            "origin": self._origin.copy(),
            "basis": self._basis.copy(),
            "search_low": self._low.copy(),
            "search_high": self._high.copy(),
        }

    @property
    def parameter_space_id(self):
        return identity(self.snapshot())

    def freeze(self):
        self._frozen = True

    def encode(self, values):
        x = self.to_normalized(values) - self._origin
        z = x @ self._basis
        if not np.allclose(z @ self._basis.T, x, atol=1e-7, rtol=1e-7):
            raise ValueError("physical parameters are outside the selected subspace")
        return (z - self._low) / (self._high - self._low)

    def decode(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != self.dimension:
            raise ValueError("search dimension mismatch")
        z = self._low + values * (self._high - self._low)
        return self.from_normalized(self._origin + z @ self._basis.T)

    def check_bounds(self, values):
        values = np.asarray(values, dtype=np.float64)
        return (
            np.isfinite(values).all(axis=-1)
            & (values >= self.bounds[:, 0] - 1e-12).all(axis=-1)
            & (values <= self.bounds[:, 1] + 1e-12).all(axis=-1)
        )

    def as_dict(self, row):
        return dict(zip(self.names, map(float, row), strict=True))
