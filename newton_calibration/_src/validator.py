# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Independent per-experiment validation through a restricted evaluation callback."""

import math


class Validator:
    def __init__(self, max_objective=0.01, max_error=1.0):
        if any(not math.isfinite(x) or x < 0 for x in (max_objective, max_error)):
            raise ValueError("invalid validation thresholds")
        self.max_objective = max_objective
        self.max_error = max_error

    def accepts(self, result):
        return bool(
            result.feasible.all()
            and (result.objectives <= self.max_objective).all()
            and (result.metrics[:, 1] <= self.max_error).all()
        )

    def validate(self, candidate_asset, validation_data, evaluate):
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
