# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent per-experiment validation through a restricted evaluation callback."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .asset import Asset
from .data import BatchResult
from .evidence import Experiment


class Validator:
    """``Validator`` checks acceptance for each experiment. ``Tuning`` exports and reloads the
    selected asset, then retests it on training experiments and evaluates it on held-out
    validation experiments. Both checks call ``validate()`` and produce separate reports.

    Retesting each experiment separately checks even those given low weight during search.

    ``Tuning`` supplies the ``evaluate`` callback, locks candidate identity, and performs
    budgeting, simulation, and scoring. Custom validators can add condition grouping or
    historical-evidence restrictions while preserving report fields and candidate identity.
    """

    def __init__(self, max_objective: float = 0.01, max_error: float = 1.0) -> None:
        """Initialize the validator.

        Args:
            max_objective: Finite nonnegative objective limit, default ``0.01``. With default
                squared error, ``max_objective=0.01`` requires normalized RMSE of at most 0.1
                for that evaluation. The maximum-error condition must also pass.
            max_error: Finite nonnegative limit on ``metrics[:,1]``, default ``1.0``.
        """
        if any(not math.isfinite(x) or x < 0 for x in (max_objective, max_error)):
            raise ValueError("invalid validation thresholds")
        self.max_objective = max_objective
        self.max_error = max_error

    def accepts(self, result: BatchResult) -> bool:
        """True when every candidate is feasible and within objective and maximum-error limits."""
        return bool(
            result.feasible.all()
            and (result.objectives <= self.max_objective).all()
            and (result.metrics[:, 1] <= self.max_error).all()
        )

    def validate(
        self,
        candidate_asset: Asset,
        validation_data: Sequence[Experiment],
        evaluate: Callable[[Asset, tuple[Experiment, ...]], BatchResult],
    ) -> dict[str, Any]:
        """Call ``evaluate(candidate_asset, (experiment,))`` for each experiment and return a
        dictionary with ``passed``, ``status``, and ``experiments``.

        The default validator returns ``status="unvalidated"`` for empty validation data.
        ``Tuning`` organizes numerical rechecks separately and includes them in the final report.
        """
        if not validation_data:
            return {"passed": False, "status": "unvalidated", "experiments": []}
        reports = []
        for experiment in validation_data:
            result = evaluate(candidate_asset, (experiment,))
            reports.append(
                {
                    "experiment": experiment.id,
                    "passed": self.accepts(result),
                    "objective": float(result.objectives[0]),
                    "max_error": float(result.metrics[0, 1]),
                    "valid": bool(result.valid_mask[0]),
                }
            )
        return {
            "passed": all(x["passed"] for x in reports),
            "status": "evaluated",
            "experiments": reports,
        }
