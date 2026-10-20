# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Vectorized tensor scoring with fixed masks and complete episode identities."""

import math

import torch

from .data import identity


class Evaluator:
    def __init__(self, *, loss="squared_error", max_normalized_error=None, device="cpu"):
        if loss not in {"squared_error", "cross_entropy"}:
            raise ValueError("unsupported scoring loss")
        if max_normalized_error is not None and (
            not math.isfinite(max_normalized_error) or max_normalized_error <= 0
        ):
            raise ValueError("invalid constraint threshold")
        self.loss = loss
        self.max_normalized_error = max_normalized_error
        self.device = torch.device(device)

    @property
    def metric_spec(self):
        return {
            "loss": self.loss,
            "max_normalized_error": self.max_normalized_error,
            "device": str(self.device),
            "metrics": ["normalized_rmse", "normalized_max_error"],
        }

    @property
    def evaluation_id(self):
        return identity(self.metric_spec)

    def score_batch(self, episodes, trial_ids, experiments):
        expected = {(t, e.id, r) for t in trial_ids for e in experiments for r in range(e.repeats)}
        received = [item.key for item in episodes]
        if len(received) != len(set(received)) or set(received) != expected:
            raise ValueError("missing, duplicate or extra episode results")
        by_key = {item.key: item for item in episodes}
        b = len(trial_ids)
        valid = torch.ones(b, dtype=torch.bool, device=self.device)
        losses = torch.zeros(b, dtype=torch.float64, device=self.device)
        max_error = torch.zeros_like(losses)
        blocks = []
        total_weight = sum(e.weight for e in experiments)
        for experiment in experiments:
            mask = experiment.mask.to(self.device)
            reference = experiment.reference.to(self.device)
            scale = experiment.scale.to(self.device)
            weight = experiment.weight / total_weight / experiment.repeats
            if self.loss == "cross_entropy":
                if (
                    not (mask.all(dim=1) | ~mask.any(dim=1)).all()
                    or (reference[mask] < 0).any()
                    or not torch.allclose(
                        reference[mask.all(dim=1)].sum(dim=1),
                        torch.ones(mask.all(dim=1).sum(), dtype=torch.float64, device=self.device),
                    )
                ):
                    raise ValueError("cross_entropy needs complete normalized probability rows")
            for repeat in range(experiment.repeats):
                predictions = []
                for i, trial_id in enumerate(trial_ids):
                    episode = by_key[trial_id, experiment.id, repeat]
                    trace = episode.trace
                    ok = episode.status == "completed" and trace is not None
                    if ok:
                        values = torch.as_tensor(
                            trace.values, dtype=torch.float64, device=self.device
                        )
                        times = torch.as_tensor(trace.times, dtype=torch.float64).cpu()
                        ok = (
                            values.shape == reference.shape
                            and times.shape == experiment.times.shape
                            and torch.allclose(times, experiment.times, atol=1e-9, rtol=0)
                            and bool(torch.isfinite(values[mask]).all())
                        )
                    if not ok:
                        valid[i] = False
                        values = torch.full_like(reference, float("nan"))
                    predictions.append(values)
                values = torch.stack(predictions)
                residual = ((values - reference) / scale)[:, mask]
                features = residual * math.sqrt(weight / mask.sum().item())
                blocks.append(features)
                max_error = torch.maximum(max_error, residual.abs().amax(dim=1))
                if self.loss == "squared_error":
                    losses += weight * residual.square().mean(dim=1)
                else:
                    rows = mask.all(dim=1)
                    probability = values[:, rows]
                    probability_ok = (probability >= 0).all(dim=(1, 2)) & torch.isclose(
                        probability.sum(dim=-1),
                        torch.ones_like(probability[..., 0]),
                        atol=1e-8,
                        rtol=1e-8,
                    ).all(dim=1)
                    valid &= probability_ok
                    entropy = -(reference[rows] * probability.clamp_min(1e-15).log()).sum(-1)
                    losses += weight * entropy.mean(dim=1)
        features = torch.cat(blocks, dim=1)
        valid &= torch.isfinite(features).all(dim=1) & torch.isfinite(losses)
        metrics = torch.stack((features.square().sum(dim=1).sqrt(), max_error), dim=1)
        constraints = (
            torch.empty((b, 0), dtype=torch.float64, device=self.device)
            if self.max_normalized_error is None
            else (max_error / self.max_normalized_error - 1).unsqueeze(1)
        )
        losses = torch.where(valid, losses, torch.full_like(losses, float("inf")))
        return losses, metrics, constraints, valid, features
