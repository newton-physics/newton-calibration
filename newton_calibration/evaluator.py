# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Vectorized array scoring with fixed masks and complete episode identities."""

import math

import numpy as np

from .data import identity


class Evaluator:
    def __init__(self, *, loss="squared_error", max_normalized_error=None):
        if loss not in {"squared_error", "cross_entropy"}:
            raise ValueError("unsupported scoring loss")
        if max_normalized_error is not None and (not math.isfinite(max_normalized_error) or max_normalized_error <= 0):
            raise ValueError("invalid constraint threshold")
        self.loss = loss
        self.max_normalized_error = max_normalized_error

    @property
    def metric_spec(self):
        return {
            "loss": self.loss,
            "max_normalized_error": self.max_normalized_error,
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
        valid = np.ones(b, dtype=bool)
        losses = np.zeros(b, dtype=np.float64)
        max_error = np.zeros_like(losses)
        blocks = []
        total_weight = sum(e.weight for e in experiments)
        for experiment in experiments:
            mask = experiment.mask
            reference = experiment.reference
            scale = experiment.scale
            weight = experiment.weight / total_weight / experiment.repeats
            if self.loss == "cross_entropy":
                if (
                    not (mask.all(axis=1) | ~mask.any(axis=1)).all()
                    or (reference[mask] < 0).any()
                    or not np.allclose(reference[mask.all(axis=1)].sum(axis=1), 1.0)
                ):
                    raise ValueError("cross_entropy needs complete normalized probability rows")
            for repeat in range(experiment.repeats):
                predictions = []
                for i, trial_id in enumerate(trial_ids):
                    episode = by_key[trial_id, experiment.id, repeat]
                    trace = episode.trace
                    ok = episode.status == "completed" and trace is not None
                    if ok:
                        values = np.asarray(trace.values, dtype=np.float64)
                        times = np.asarray(trace.times, dtype=np.float64)
                        ok = (
                            values.shape == reference.shape
                            and times.shape == experiment.times.shape
                            and np.allclose(times, experiment.times, atol=1e-9, rtol=0)
                            and bool(np.isfinite(values[mask]).all())
                        )
                    if not ok:
                        valid[i] = False
                        values = np.full_like(reference, np.nan)
                    predictions.append(values)
                values = np.stack(predictions)
                residual = ((values - reference) / scale)[:, mask]
                features = residual * math.sqrt(weight / mask.sum())
                blocks.append(features)
                max_error = np.maximum(max_error, np.abs(residual).max(axis=1))
                if self.loss == "squared_error":
                    losses += weight * np.square(residual).mean(axis=1)
                else:
                    rows = mask.all(axis=1)
                    probability = values[:, rows]
                    probability_ok = (probability >= 0).all(axis=(1, 2)) & np.isclose(
                        probability.sum(axis=-1), 1.0, atol=1e-8, rtol=1e-8
                    ).all(axis=1)
                    valid &= probability_ok
                    entropy = -(reference[rows] * np.log(np.maximum(probability, 1e-15))).sum(-1)
                    losses += weight * entropy.mean(axis=1)
        features = np.concatenate(blocks, axis=1)
        valid &= np.isfinite(features).all(axis=1) & np.isfinite(losses)
        metrics = np.stack((np.sqrt(np.square(features).sum(axis=1)), max_error), axis=1)
        constraints = (
            np.empty((b, 0), dtype=np.float64)
            if self.max_normalized_error is None
            else (max_error / self.max_normalized_error - 1)[:, None]
        )
        losses = np.where(valid, losses, np.inf)
        return losses, metrics, constraints, valid, features
