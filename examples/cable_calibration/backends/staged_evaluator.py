"""Dynamic waveform emphasis while retaining full-coordinate numeric checks."""

from dataclasses import replace

import torch

from .validation import CableEvaluator


class DistalDynamicEvaluator(CableEvaluator):
    @property
    def metric_spec(self):
        spec = super().metric_spec
        spec["objective"] = "0.5 full-coordinate MSE + 0.5 distal-z MSE"
        spec["distal_channels"] = [25, 27, 29]
        spec["distal_arclength_m"] = [0.13, 0.14, 0.15]
        spec["metrics"] += ["distal_z_mse"]
        spec["features"] = (
            "full unweighted coordinate residuals; unchanged numerical gates"
        )
        return spec

    def score_batch(self, episodes, trial_ids, experiments):
        losses, metrics, constraints, valid, features = super().score_batch(
            episodes, trial_ids, experiments
        )
        distal = []
        for experiment in experiments:
            if experiment.reference.shape[1] != 30 or len(experiment.times) < 2:
                raise ValueError(
                    "distal dynamic scoring requires 15 XZ points over time"
                )
            mask = torch.zeros_like(experiment.mask)
            mask[:, [25, 27, 29]] = experiment.mask[:, [25, 27, 29]]
            if not mask.any():
                raise ValueError("no valid distal dynamic observations")
            distal.append(replace(experiment, mask=mask))
        distal_loss, distal_metrics, _, distal_valid, _ = super().score_batch(
            episodes, trial_ids, tuple(distal)
        )
        losses = 0.5 * (losses + distal_loss)
        metrics[:, 0] = losses.sqrt()
        metrics[:, 2] = torch.maximum(metrics[:, 2], distal_metrics[:, 2])
        metrics = torch.cat((metrics, distal_loss[:, None]), dim=1)
        return losses, metrics, constraints, valid & distal_valid, features
