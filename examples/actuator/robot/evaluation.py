# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-joint physical error gates while preserving the core residual features."""

import numpy as np

from newton_calibration import Evaluator, Validator


class EvaluatorRobot(Evaluator):
    def __init__(self, settings, channels):
        super().__init__()
        self.channels = tuple(tuple(channel) for channel in channels)
        self.record_channels = {key: tuple(tuple(c) for c in value["channels"]) for key, value in settings.items()}

    @property
    def metric_spec(self):
        return {
            **super().metric_spec,
            "metrics": [
                *super().metric_spec["metrics"],
                *[f"{name}.{kind}.{metric}" for kind, name in self.channels for metric in ("rmse", "max")],
            ],
            "record_channels": self.record_channels,
            "physical_aggregation": "worst per-record/repeat; absent channels contribute zero",
        }

    def score_batch(self, episodes, trial_ids, experiments):
        losses, metrics, constraints, valid, features = super().score_batch(episodes, trial_ids, experiments)
        extra = np.zeros((len(trial_ids), 2 * len(self.channels)))
        extra[~valid] = np.inf
        rows = {trial: i for i, trial in enumerate(trial_ids)}
        by_id = {exp.id: exp for exp in experiments}
        for episode in episodes:
            row = rows[episode.trial_id]
            if not valid[row]:
                continue
            exp = by_id[episode.experiment_id]
            for column, channel in enumerate(self.record_channels[exp.id]):
                residual = (episode.trace.values[:, column] - exp.reference[:, column])[exp.mask[:, column]]
                offset = 2 * self.channels.index(channel)
                extra[row, offset : offset + 2] = np.maximum(
                    extra[row, offset : offset + 2], [np.sqrt(np.mean(residual**2)), np.max(np.abs(residual))]
                )
        return losses, np.column_stack((metrics, extra)), constraints, valid, features


class ValidatorRobot(Validator):
    def __init__(self, channels, thresholds, *, settings):
        super().__init__(max_objective=1.0, max_error=3.0)
        self.labels = [f"{name}.{kind}.{metric}" for kind, name in channels for metric in ("rmse", "max")]
        self.limits = [thresholds[kind][name][metric] for kind, name in channels for metric in ("rmse", "max")]
        self.record_channels = {
            key: tuple(tuple(channel) for channel in value["channels"]) for key, value in settings.items()
        }

    def accepts(self, result):
        return super().accepts(result) and bool((result.metrics[:, 2:] <= self.limits).all())

    def validate(self, candidate_asset, validation_data, evaluate):
        physical = []

        def capture(asset, experiments):
            result = evaluate(asset, experiments)
            physical.append(dict(zip(self.labels, result.metrics[0, 2:].tolist(), strict=True)))
            return result

        report = super().validate(candidate_asset, validation_data, capture)
        report["thresholds_si"] = dict(zip(self.labels, self.limits, strict=True))
        for item, values in zip(report["experiments"], physical, strict=True):
            observed = self.record_channels[item["experiment"]]
            labels = {f"{name}.{kind}.{metric}" for kind, name in observed for metric in ("rmse", "max")}
            item["observed_channels"] = observed
            item["physical_errors"] = {name: value if name in labels else None for name, value in values.items()}
        return report
