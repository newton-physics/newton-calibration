# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ask/tell strategies. No optimizer is permitted to stop the Tuning loop."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .data import BatchResult, TrialBatch
from .parameters import Parameters


class Optimizer(ABC):
    """``Optimizer`` uses an ask/tell protocol. After parameter analysis, ``Tuning`` calls
    ``initialize(parameters)``. Once the budget allows a batch, it calls ``ask(batch_size)``
    and returns scores through ``tell(result)``. The optimizer must receive feedback for the
    pending batch before proposing another.

    To implement another algorithm, subclass ``Optimizer`` and implement ``ask()``. Extend
    ``initialize()``, ``tell()``, ``diagnostics()``, and ``state_dict()`` as needed, and
    optionally provide ``checkpoint()`` as CMA-ES does. Shared stopping and budget decisions
    remain in ``Decision.next_action()``.
    """

    full_batch = False

    def __init__(self, batch_size: int = 8, seed: int = 42) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self._generation = 0
        self._pending: TrialBatch | None = None

    def initialize(self, parameters: Parameters) -> None:
        """Store the search dimension and the encoded baseline of ``parameters``."""
        self.dimension = parameters.dimension
        self.initial = parameters.encode(parameters.baseline)
        self._generation = 0
        self._pending = None

    def _batch(self, rows: ArrayLike) -> TrialBatch:
        if self._pending is not None:
            raise RuntimeError("tell the pending batch before asking again")
        rows = np.array(rows, dtype=np.float64).reshape(-1, self.dimension)
        if not len(rows) or not np.isfinite(rows).all() or ((rows < 0) | (rows > 1)).any():
            raise ValueError("optimizer produced invalid normalized proposals")
        ids = tuple(f"trial-{self._generation:05d}-{i:05d}" for i in range(len(rows)))
        self._pending = TrialBatch(ids, rows.copy())
        return TrialBatch(ids, rows.copy())

    def _accept(self, result: BatchResult) -> None:
        if (
            self._pending is None
            or result.trial_ids != self._pending.trial_ids
            or not np.array_equal(result.search_values, self._pending.search_values)
            or result.purpose != "train"
        ):
            raise ValueError("optimizer feedback does not match pending training batch")
        if not np.isfinite(result.feedback).all():
            raise ValueError("optimizer feedback must be finite")
        self._pending = None
        self._generation += 1

    @abstractmethod
    def ask(self, batch_size: int) -> TrialBatch:
        """``ask()`` returns ``TrialBatch(trial_ids, search_values)`` with ``[B,Q]`` values in
        ``[0,1]``, where ``Q`` is search dimension."""

    def tell(self, result: BatchResult) -> None:
        """``tell()`` receives the corresponding ``BatchResult``, whose candidate IDs, search
        values, and ``purpose="train"`` must match."""
        self._accept(result)

    def diagnostics(self) -> dict[str, Any]:
        """``diagnostics()["exhausted"]`` informs ``Decision`` about search-space exhaustion."""
        return {"generation": self._generation, "exhausted": False}

    def state_dict(self) -> dict[str, Any]:
        """Return the search state for the run records."""
        return {
            "kind": type(self).__name__,
            "generation": self._generation,
            "seed": self.seed,
            "dimension": self.dimension,
        }


class GridSearch(Optimizer):
    """Use ``GridSearch`` to explore a few parameters and inspect responses across their
    ranges. It generates a grid in the current search coordinates and maps it to physical
    values with ``Parameters.decode()``."""

    def __init__(self, levels: int = 11, batch_size: int = 16) -> None:
        """Initialize the grid search.

        Args:
            levels: Equally spaced points per search dimension, at least 2, with
                ``levels ** parameters.dimension`` total points. Defaults to ``11``.
            batch_size: Requested candidates per batch. Remaining budget and grid points can
                reduce the last batch. Defaults to ``16``.
        """
        super().__init__(batch_size)
        if levels < 2:
            raise ValueError("grid needs at least two levels")
        self.levels = levels

    def initialize(self, parameters: Parameters) -> None:
        """Prepare the grid for the search dimension of ``parameters``."""
        super().initialize(parameters)
        self._iterator = itertools.product(np.linspace(0, 1, self.levels), repeat=self.dimension)
        self._remaining = self.levels**self.dimension

    def ask(self, batch_size: int) -> TrialBatch:
        """Return the next grid points, at most ``batch_size``."""
        rows = list(itertools.islice(self._iterator, min(batch_size, self._remaining)))
        self._remaining -= len(rows)
        return self._batch(rows)

    def diagnostics(self) -> dict[str, Any]:
        """Add the number of ``remaining`` grid points. ``exhausted`` is true after the last point."""
        return {
            **super().diagnostics(),
            "exhausted": self._remaining == 0,
            "remaining": self._remaining,
        }

    def state_dict(self) -> dict[str, Any]:
        """Add the ``remaining`` grid points and ``levels`` to the search state."""
        return {**super().state_dict(), "remaining": self._remaining, "levels": self.levels}


class CMAES(Optimizer):
    """``CMAES`` requires the ``cma`` extra and at least two search dimensions. If rank
    reduction is enabled, choose the optimizer for the reduced ``parameters.dimension``."""

    full_batch = True

    def __init__(self, batch_size: int = 8, sigma: float = 0.2, seed: int = 42) -> None:
        """Initialize the CMA-ES search.

        Args:
            batch_size: Population per generation, at least 4. ``full_batch=True`` requires
                complete-generation evaluation. Defaults to ``8``.
            sigma: Initial step size in the current normalized search coordinates, with
                ``0 < sigma < 1``. Defaults to ``0.2``.
            seed: Sampling seed, default 42, configured separately from experimental
                simulation seeds.
        """
        super().__init__(batch_size, seed)
        if batch_size < 4 or not 0 < sigma < 1:
            raise ValueError("CMA-ES needs population >= 4 and 0 < sigma < 1")
        self.sigma = sigma

    def initialize(self, parameters: Parameters) -> None:
        """Create the ``cma`` strategy, started at the encoded baseline."""
        super().initialize(parameters)
        if self.dimension < 2:
            raise ValueError("CMA-ES needs at least two parameters; use GridSearch")
        try:
            import cma
        except ImportError as error:
            raise ImportError("install newton-calibration[cma] to use CMAES") from error
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

    def ask(self, batch_size: int) -> TrialBatch:
        """Return one complete generation. ``batch_size`` must equal the population size."""
        if batch_size != self.batch_size:
            raise ValueError("CMA-ES requires a complete generation")
        self._last_rows = self._strategy.ask()
        return self._batch(self._last_rows)

    def tell(self, result: BatchResult) -> None:
        """Pass ``result.feedback`` of the pending generation to the ``cma`` strategy."""
        self._accept(result)
        self._strategy.tell(self._last_rows, result.feedback.tolist())

    def diagnostics(self) -> dict[str, Any]:
        """Add the step size ``sigma`` and the ``condition_number`` of the covariance."""
        return {
            **super().diagnostics(),
            "sigma": float(self._strategy.sigma),
            "condition_number": float(self._strategy.condition_number),
        }

    def state_dict(self) -> dict[str, Any]:
        """Add the mean, step size, covariance, and random state to the search state."""
        return {
            **super().state_dict(),
            "mean": self._strategy.mean.tolist(),
            "sigma": float(self._strategy.sigma),
            "covariance": self._strategy.C.tolist(),
            "rng": self._rng.get_state()[1].tolist(),
        }

    def checkpoint(self) -> bytes:
        """Return the pickled ``cma`` strategy. ``Tuning`` saves it after each batch."""
        return self._strategy.pickle_dumps()
