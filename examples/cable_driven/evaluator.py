# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluator whose features are the normalized residuals themselves."""

import math

from newton_calibration import Evaluator


class CableDrivenEvaluator(Evaluator):
    """Squared-error evaluator with residual features in channel scales.

    The base evaluator weights each residual by ``sqrt(weight / samples)`` so that the
    squared features sum to the loss. The numerical qualification compares these features,
    so its tolerance would shrink with the amount of data. This evaluator removes the
    weighting from the features: one feature unit is one channel scale. The losses,
    metrics and constraints are unchanged.
    """

    @property
    def metric_spec(self):
        spec = super().metric_spec
        spec["features"] = "normalized residuals in channel scales; tolerance in scales"
        return spec

    def score_batch(self, episodes, trial_ids, experiments):
        losses, metrics, constraints, valid, features = super().score_batch(episodes, trial_ids, experiments)
        total_weight = sum(e.weight for e in experiments)
        column = 0
        for experiment in experiments:
            width = int(experiment.mask.sum().item())
            weight = experiment.weight / total_weight / experiment.repeats
            factor = math.sqrt(width / weight)
            for _ in range(experiment.repeats):
                features[:, column : column + width] *= factor
                column += width
        return losses, metrics, constraints, valid, features
