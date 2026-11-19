# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The sole authority for stopping or authorizing work."""

import math
from dataclasses import dataclass

from .data import DecisionResult


@dataclass(frozen=True)
class Decision:
    """Before baseline, analysis, training, and validation work, ``Decision`` checks state
    and budgets and returns ``DecisionResult(action, reason, batch_size=0)``. ``Tuning``
    uses that decision to start the work or finish the run.

    Budget limits are positive, ``target`` is finite and nonnegative, and enabled
    ``patience`` is a positive batch count.
    """

    max_trials: int = 512
    """Training-candidate limit. Baseline and analysis probes are recorded in their own phases."""
    max_rollouts: int = 10000
    """Episode limit across all phases, including scheduled failed or invalid candidates."""
    max_generations: int = 100
    """Training-batch limit, corresponding to generations for CMA-ES and chunks for GridSearch."""
    max_seconds: float = 300.0
    """Total wall-clock limit from preparation, handled through cooperative backend deadlines."""
    target: float = 1e-5
    """Stop search when the best feasible training objective reaches this value."""
    patience: int | None = None
    """Default ``None``. When enabled, detect stagnation from improvement across the specified
    number of recent batches."""
    improvement_tolerance: float = 1e-10
    """Threshold for old best objective minus new best objective when patience is enabled."""

    def __post_init__(self):
        if (
            min(self.max_trials, self.max_rollouts, self.max_generations) < 1
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
            or not math.isfinite(self.target)
            or self.target < 0
            or (self.patience is not None and self.patience < 1)
        ):
            raise ValueError("invalid decision policy")

    def next_action(self, state, request):
        """``next_action(state, request)`` receives a read-only state snapshot and work
        request, then returns ``action="continue"`` or ``"stop"``. Its ``reason`` enters
        decision history."""
        if state["analysis_reports"].get("numerical", {}).get("passed") is False:
            return DecisionResult("stop", "numerical_config_unavailable")
        if state["fatal_error"]:
            return DecisionResult("stop", "execution_error")
        if state["cancelled"]:
            return DecisionResult("stop", "cancelled")
        if state["elapsed_s"] >= self.max_seconds:
            return DecisionResult("stop", "time_budget_exhausted")
        purpose = request["purpose"]
        usage = state["budget_usage"]
        used = sum(item["rollouts"] for item in usage.values())
        remaining = self.max_rollouts - used - request.get("reserve_rollouts", 0)
        count = request["count"]
        per_trial = request["episodes_per_trial"]
        if purpose == "train":
            analysis = state["analysis_reports"].get("identifiability", {})
            if analysis and not analysis["conclusive"]:
                return DecisionResult("stop", "inconclusive_parameters")
            history = [b for b in state["batch_history"] if b["purpose"] == "train"]
            best = state["best_candidate"]
            if best and best["objective"] <= self.target:
                return DecisionResult("stop", "target_reached")
            if state["optimizer_snapshot"].get("diagnostics", {}).get("exhausted", False):
                return DecisionResult("stop", "candidate_space_exhausted")
            if len(history) >= self.max_generations:
                return DecisionResult("stop", "generation_budget_exhausted")
            remaining_trials = self.max_trials - usage.get("train", {}).get("trials", 0)
            count = min(count, remaining_trials)
            if self.patience and len(history) >= self.patience + 1:
                old = history[-self.patience - 1]["best_objective"]
                new = history[-1]["best_objective"]
                if old is not None and new is not None and old - new <= self.improvement_tolerance:
                    return DecisionResult("stop", "stagnation")
        count = min(count, max(0, remaining // per_trial))
        if count < request.get("minimum_batch", 1):
            return DecisionResult("stop", "budget_exhausted")
        return DecisionResult("continue", "work_authorized", count)
