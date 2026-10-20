"""Re-score archived cable traces on CPU; never starts Newton or an optimizer."""

import json
from pathlib import Path

import numpy as np
import torch

from task_tuning import Evidence, Experiment, Trace
from task_tuning.data import Episode

from .backends.a.validation import CableEvaluator
from .backends.c.staged_evaluator import DistalDynamicEvaluator

CASE = Path(__file__).resolve().parent


def load_experiments():
    records = json.loads((CASE / "data/experiments.json").read_text())
    return {key: Experiment(**value).normalized() for key, value in records.items()}


def verify():
    experiments = load_experiments()
    expected = json.loads((CASE / "archive/expected-metrics.json").read_text())
    results = {}
    with np.load(CASE / "data/traces.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("Archive and expected metric keys differ")
        for key, target in expected.items():
            stage, name = key.split("__", 1)
            exp = experiments[name]
            values = torch.from_numpy(archive[key])
            if values.shape != exp.reference.shape or not torch.isfinite(values).all():
                raise ValueError(f"Invalid archived trace: {key}")
            error = (values - exp.reference)[exp.mask] * 1000
            rmse, maximum = float(error.square().mean().sqrt()), float(error.abs().max())
            np.testing.assert_allclose(
                [rmse, maximum], [target["rmse_mm"], target["max_mm"]], rtol=0, atol=1e-8
            )
            evaluator = DistalDynamicEvaluator() if stage in ("Bstar", "C") else CableEvaluator()
            episode = Episode("archived", name, 0, "completed", Trace(exp.times, values))
            score = evaluator.score_batch((episode,), ("archived",), (exp,))
            if not bool(score[3].all()):
                raise ValueError(f"Invalid evaluator result: {key}")
            results[key] = {
                "coordinate_rmse_mm": rmse,
                "max_coordinate_error_mm": maximum,
                "objective": float(score[0][0]),
            }
    for stage in ("A", "B", "Bstar", "C"):
        config = json.loads((CASE / "configs" / f"{stage}.json").read_text())
        Evidence(
            [experiments[x] for x in config["train"]],
            [experiments[x] for x in config["validation"]],
        ).prepare()
    comparison = json.loads((CASE / "archive/C-comparison.json").read_text())
    for stage, label in (("Bstar", "Kelvin best"), ("C", "SLS best")):
        target = comparison["metrics"]["noplug_release"][label]["objective"]
        np.testing.assert_allclose(
            results[f"{stage}__noplug_release"]["objective"], target, rtol=0, atol=1e-10
        )
    before = results["Bstar__noplug_release"]["objective"]
    after = results["C__noplug_release"]["objective"]
    return {
        "mode": "archived_trace_verification",
        "simulation_executed": False,
        "verified_traces": len(results),
        "metrics": results,
        "sls_objective_reduction_percent": 100 * (1 - after / before),
        "historical_acceptance": {
            s: json.loads((CASE / "archive" / f"{s}-result.json").read_text())["acceptance"]
            for s in ("A", "B", "Bstar", "C")
        },
        "scope": "Archive agreement is not new simulation or independent validation.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
