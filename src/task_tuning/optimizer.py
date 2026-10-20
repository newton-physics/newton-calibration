# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Ask/tell strategies. No optimizer is permitted to stop the Tuning loop."""

import itertools
from abc import ABC, abstractmethod

import numpy as np
import torch

from .data import TrialBatch


class Optimizer(ABC):
    full_batch = False

    def __init__(self, batch_size=8, seed=42):
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self._generation = 0
        self._pending = None

    def initialize(self, parameters):
        self.dimension = parameters.dimension
        self.initial = parameters.encode(parameters.baseline).numpy()
        self._generation = 0
        self._pending = None

    def _batch(self, rows):
        if self._pending is not None:
            raise RuntimeError("tell the pending batch before asking again")
        rows = torch.as_tensor(np.asarray(rows), dtype=torch.float64).reshape(-1, self.dimension)
        if not len(rows) or not torch.isfinite(rows).all() or ((rows < 0) | (rows > 1)).any():
            raise ValueError("optimizer produced invalid normalized proposals")
        ids = tuple(f"trial-{self._generation:05d}-{i:05d}" for i in range(len(rows)))
        self._pending = TrialBatch(ids, rows.clone())
        return TrialBatch(ids, rows.clone())

    def _accept(self, result):
        if (
            self._pending is None
            or result.trial_ids != self._pending.trial_ids
            or not torch.equal(result.search_values.cpu(), self._pending.search_values)
            or result.purpose != "train"
        ):
            raise ValueError("optimizer feedback does not match pending training batch")
        if not torch.isfinite(result.feedback).all():
            raise ValueError("optimizer feedback must be finite")
        self._pending = None
        self._generation += 1

    @abstractmethod
    def ask(self, batch_size): ...

    def tell(self, result):
        self._accept(result)

    def diagnostics(self):
        return {"generation": self._generation, "exhausted": False}

    def state_dict(self):
        return {
            "kind": type(self).__name__,
            "generation": self._generation,
            "seed": self.seed,
            "dimension": self.dimension,
        }


class GridSearch(Optimizer):
    def __init__(self, levels=11, batch_size=16):
        super().__init__(batch_size)
        if levels < 2:
            raise ValueError("grid needs at least two levels")
        self.levels = levels

    def initialize(self, parameters):
        super().initialize(parameters)
        self._iterator = itertools.product(np.linspace(0, 1, self.levels), repeat=self.dimension)
        self._remaining = self.levels**self.dimension

    def ask(self, batch_size):
        rows = list(itertools.islice(self._iterator, min(batch_size, self._remaining)))
        self._remaining -= len(rows)
        return self._batch(rows)

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "exhausted": self._remaining == 0,
            "remaining": self._remaining,
        }

    def state_dict(self):
        return {**super().state_dict(), "remaining": self._remaining, "levels": self.levels}


class CMAES(Optimizer):
    full_batch = True

    def __init__(self, batch_size=8, sigma=0.2, seed=42):
        super().__init__(batch_size, seed)
        if batch_size < 4 or not 0 < sigma < 1:
            raise ValueError("CMA-ES needs population >= 4 and 0 < sigma < 1")
        self.sigma = sigma

    def initialize(self, parameters):
        super().initialize(parameters)
        try:
            import cma
        except ImportError as error:
            raise ImportError("install task-tuning[cma] to use CMAES") from error
        self._rng = np.random.RandomState(self.seed)
        self._strategy = cma.CMAEvolutionStrategy(
            self.initial,
            self.sigma,
            {
                "popsize": self.batch_size,
                "bounds": [0, 1],
                "randn": self._rng.randn,
                "seed": self.seed,
                "verbose": -9,
                "verb_log": 0,
            },
        )
        self._last_rows = None

    def ask(self, batch_size):
        if batch_size != self.batch_size:
            raise ValueError("CMA-ES requires a complete generation")
        self._last_rows = self._strategy.ask()
        return self._batch(self._last_rows)

    def tell(self, result):
        self._accept(result)
        self._strategy.tell(self._last_rows, result.feedback.cpu().tolist())

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "sigma": float(self._strategy.sigma),
            "condition_number": float(self._strategy.condition_number),
        }

    def state_dict(self):
        return {
            **super().state_dict(),
            "mean": self._strategy.mean.tolist(),
            "sigma": float(self._strategy.sigma),
            "covariance": self._strategy.C.tolist(),
            "rng": self._rng.get_state()[1].tolist(),
        }

    def checkpoint(self):
        return self._strategy.pickle_dumps()
