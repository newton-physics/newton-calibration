# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical per-channel errors alongside the unchanged normalized core metrics."""

import numpy as np

from newton_calibration import Evaluator, Validator


class EvaluatorActuator(Evaluator):
    def __init__(self, settings):
        super().__init__(loss="squared_error")
        self.channels = {key: tuple(value["channels"]) for key, value in settings.items()}

    @property
    def metric_spec(self):
        return {
            **super().metric_spec,
            "channels": self.channels,
            "metrics": [
                "normalized_rmse",
                "normalized_max_error",
                "position_rmse_rad",
                "position_max_rad",
                "velocity_rmse_rad_s",
                "velocity_max_rad_s",
            ],
            "physical_aggregation": "maximum per-experiment/repeat error; absent velocity contributes zero",
        }

    def score_batch(self, episodes, trial_ids, experiments):
        losses, metrics, constraints, valid, features = super().score_batch(episodes, trial_ids, experiments)
        extra = np.zeros((len(trial_ids), 4))
        by_id = {experiment.id: experiment for experiment in experiments}
        rows = {trial: i for i, trial in enumerate(trial_ids)}
        for episode in episodes:
            i = rows[episode.trial_id]
            if not valid[i]:
                extra[i] = np.inf
                continue
            experiment = by_id[episode.experiment_id]
            for column, channel in enumerate(self.channels[experiment.id]):
                residual = episode.trace.values[:, column] - experiment.reference[:, column]
                residual = residual[experiment.mask[:, column]]
                offset = 0 if channel == "position" else 2
                extra[i, offset : offset + 2] = np.maximum(
                    extra[i, offset : offset + 2],
                    [np.sqrt(np.mean(residual**2)), np.max(np.abs(residual))],
                )
        return losses, np.column_stack((metrics, extra)), constraints, valid, features


class ValidatorActuator(Validator):
    def __init__(self, thresholds):
        super().__init__(max_objective=1.0, max_error=3.0)
        self.thresholds = dict(thresholds)

    def accepts(self, result):
        limits = [self.thresholds[name] for name in ("position_rmse", "position_max", "velocity_rmse", "velocity_max")]
        return super().accepts(result) and bool((result.metrics[:, 2:6] <= limits).all())

    def validate(self, candidate_asset, validation_data, evaluate):
        results = []

        def capture(asset, experiments):
            result = evaluate(asset, experiments)
            results.append(result.metrics[0, 2:6].tolist())
            return result

        report = super().validate(candidate_asset, validation_data, capture)
        report["thresholds_si"] = self.thresholds
        for item, metrics in zip(report["experiments"], results, strict=True):
            item["physical_errors"] = metrics
        return report
