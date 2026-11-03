# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""The sole authority for stopping or authorizing work."""

import math
from dataclasses import dataclass

from .data import DecisionResult


@dataclass(frozen=True)
class Decision:
    max_trials: int = 512
    max_rollouts: int = 10000
    max_generations: int = 100
    max_seconds: float = 300.0
    target: float = 1e-5
    patience: int | None = None
    improvement_tolerance: float = 1e-10

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
