# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Vectorized array scoring with fixed masks and complete episode identities."""

import math

import numpy as np

from .data import identity


class Evaluator:
    """``Evaluator`` matches the complete set of episodes to reference data by experiment,
    candidate, and repeat. It returns an objective, metrics, constraints, and residual
    features for each candidate. The objective ranks candidates, constraints determine
    feasibility, and features support sensitivity and numerical qualification.

    Custom evaluators can override ``score_batch()`` and ``metric_spec``, including
    objectives, units, constraints, and every adjustable scoring setting in the protocol.
    Preserve array shapes, order, and feature count. Current ``Tuning`` and ``Validator``
    read maximum error from ``metrics[:,1]``, so retain that column's contract or adapt
    the affected members together.
    """

    def __init__(self, *, loss="squared_error", max_normalized_error=None):
        """Initialize the evaluator.

        Args:
            loss: ``"squared_error"``: Aggregate normalized squared residuals using experiment
                weights, repeats, and valid observation counts. See
                :doc:`loss construction </calibration-guide>`.
                ``"cross_entropy"``: Score probability rows. References and predictions must be
                nonnegative normalized rows, with whole rows selected or excluded by the mask.
                Defaults to ``"squared_error"``.
            max_normalized_error: Default constraint matrix has zero columns. A positive finite
                threshold adds ``max_error / threshold - 1 <= 0``. For positions in meters,
                ``scale=0.03`` and ``max_normalized_error=2.0`` impose an absolute-error limit
                of 0.06 m on valid position channels. With different channel scales, convert
                the threshold separately for each channel. Defaults to ``None``.
        """
        if loss not in {"squared_error", "cross_entropy"}:
            raise ValueError("unsupported scoring loss")
        if max_normalized_error is not None and (not math.isfinite(max_normalized_error) or max_normalized_error <= 0):
            raise ValueError("invalid constraint threshold")
        self.loss = loss
        self.max_normalized_error = max_normalized_error

    @property
    def metric_spec(self):
        """``metric_spec`` records the scoring protocol."""
        return {
            "loss": self.loss,
            "max_normalized_error": self.max_normalized_error,
            "metrics": ["normalized_rmse", "normalized_max_error"],
        }

    @property
    def evaluation_id(self):
        """``evaluation_id`` identifies the scoring protocol in ``metric_spec``."""
        return identity(self.metric_spec)

    def score_batch(self, episodes, trial_ids, experiments):
        """For batch size ``B``, constraint count ``K``, and concatenated feature count ``F``,
        ``score_batch()`` returns five arrays in this order.

        - ``objectives``, ``[B]``: Per-candidate objective, positive infinity for invalid
          candidates.
        - ``metrics``, ``[B,2]``: Default columns ``normalized_rmse`` and
          ``normalized_max_error``, with maximum error at column 1.
        - ``constraints``, ``[B,K]``: Each entry must be ``<= 0`` for satisfaction.
          Built-in ``K`` is 0 or 1.
        - ``valid_mask``, ``[B]``: True when traces, times, finite valid observations, and
          loss-specific inputs meet the contract.
        - ``features``, ``[B,F]``: Normalized residuals concatenated in fixed
          experiment/repeat order, scaled by square-root factors for experiment weights,
          repeats, and valid sample counts.

        With squared error, ``objectives[i]`` equals the sum of squares of ``features[i]``,
        and ``metrics[i,0]`` is its square root. With cross-entropy, the objective uses
        probabilities while features and both default error metrics still use scaled
        prediction-minus-reference residuals. Interpret ``scale`` accordingly.
        """
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
