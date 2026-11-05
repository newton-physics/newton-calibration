# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the portable cable evidence example."""

import importlib

import numpy as np
import pytest


def test_archived_cable_scores_preserve_failed_acceptance():
    archive = importlib.import_module("examples.cable_calibration.verify")
    result = archive.verify()
    assert result["verified_traces"] == 14
    assert result["simulation_executed"] is False
    assert result["sls_objective_reduction_percent"] == pytest.approx(76.0087819876)
    assert result["historical_acceptance"] == {
        "A": "rejected_training",
        "B": "rejected_validation",
        "Bstar": "rejected_validation",
        "C": "rejected_validation",
    }


def test_archive_verification_rejects_changed_prediction(tmp_path, monkeypatch):
    archive = importlib.import_module("examples.cable_calibration.verify")
    original = archive.CASE
    for name in ("data/experiments.json", "archive/expected-metrics.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((original / name).read_bytes())
    with np.load(original / "data/traces.npz", allow_pickle=False) as saved:
        traces = {key: saved[key].copy() for key in saved.files}
    # Perturb every observation, including all valid samples, by a physical 1 cm.
    key = next(iter(traces))
    traces[key] += 0.01
    np.savez_compressed(tmp_path / "data/traces.npz", **traces)
    monkeypatch.setattr(archive, "CASE", tmp_path)
    with pytest.raises(AssertionError):
        archive.verify()
