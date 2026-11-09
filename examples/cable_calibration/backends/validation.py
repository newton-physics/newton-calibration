# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-experiment tolerances and explicit historical-data limitations."""

import numpy as np

from newton_calibration import Evaluator, Validator


class CableEvaluator(Evaluator):
    @property
    def metric_spec(self):
        spec = super().metric_spec
        spec["metrics"] = spec["metrics"] + ["worst_experiment_mse"]
        spec["features"] = "unweighted normalized coordinate residuals; SVD and numerical checks"
        return spec

    def score_batch(self, episodes, trial_ids, experiments):
        losses, metrics, constraints, valid, features = super().score_batch(
            episodes, trial_ids, experiments
        )
        per_experiment = []
        feature_blocks = []
        for experiment in experiments:
            subset = tuple(e for e in episodes if e.experiment_id == experiment.id)
            score = super().score_batch(subset, trial_ids, (experiment,))
            per_experiment.append(score[0])
            feature_blocks.append(
                score[4] * (experiment.mask.sum().item() * experiment.repeats) ** 0.5
            )
        worst = np.stack(per_experiment).max(axis=0)
        features = np.concatenate(feature_blocks, axis=1)
        return losses, np.concatenate((metrics, worst[:, None]), axis=1), constraints, valid, features


class HistoricalValidator(Validator):
    historical_experiments = frozenset(("withplug_left", "withplug_right", "withplug_release"))

    def accepts(self, result):
        return super().accepts(result) and bool((result.metrics[:, 2] <= 1.0).all())

    def validate(self, candidate_asset, validation_data, evaluate):
        report = super().validate(candidate_asset, validation_data, evaluate)
        if not any(e.id in self.historical_experiments for e in validation_data):
            return report
        report["historical_checks_passed"] = report["passed"]
        report["passed"] = False
        report["status"] = "historical_recheck_only"
        report["reason"] = (
            "Existing with-plug group was used in earlier model development; "
            "new independent evidence required."
        )
        return report
