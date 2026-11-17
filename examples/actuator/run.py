# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run synthetic/offline calibration, rebuild a candidate, or render saved results."""

import argparse
import json
from pathlib import Path

import numpy as np

from newton_calibration import BatchResult, Episode, Experiment, Template

from .backend import AssetActuator
from .evaluation import EvaluatorActuator, ValidatorActuator
from .recording import read_file
from .reference import synthetic_records
from .report import write_report
from .tuning import TuningActuator, synthetic_config


def replay(run_directory):
    """Rebuild the exported candidate and evaluate all saved observations."""
    root = Path(run_directory)
    resolved = json.loads((root / "resolved.json").read_text())
    path = root / "candidate_asset.json"
    record = json.loads(path.read_text())
    asset = AssetActuator(record["parameters"], config=record["config"]).reload(path)
    template = Template(**resolved["template"]).freeze()
    experiments = [
        Experiment(**item).normalized() for split in ("train", "validation") for item in resolved["evidence"][split]
    ]
    evaluator = EvaluatorActuator(template.experiment_settings)
    validator = ValidatorActuator(resolved["evidence"]["sources"]["protocol"]["thresholds"])
    output = {}
    batches = {}
    for experiment in experiments:
        environment = template.instantiate(asset, asset.baseline_parameters, experiment)
        try:
            trace = environment.run(experiment.times, seed=0, deadline=None)
        finally:
            environment.close()
        episode = Episode("replay", experiment.id, 0, "completed", trace)
        scores = evaluator.score_batch((episode,), ("replay",), (experiment,))
        batches[experiment.id] = BatchResult(
            trial_ids=("replay",),
            search_values=np.empty((1, 0)),
            parameter_values=np.array([[asset.baseline_parameters[name] for name in ("kp", "kd")]]),
            objectives=scores[0],
            metrics=scores[1],
            constraints=scores[2],
            valid_mask=scores[3],
            features=scores[4],
            episode_records=(episode,),
            identity={},
            purpose="replay",
        )
        output[experiment.id] = {
            "times": trace.times.tolist(),
            "values": trace.values.tolist(),
            "objective": float(scores[0][0]),
            "metrics": scores[1][0].tolist(),
            "valid": bool(scores[3][0]),
        }
    validation_ids = {item["id"] for item in resolved["evidence"]["validation"]}
    validation = validator.validate(
        asset,
        tuple(e for e in experiments if e.id in validation_ids),
        lambda _asset, selected: batches[selected[0].id],
    )
    target = root / "actuator-replay.json"
    target.write_text(
        json.dumps(
            {
                "scope": "diagnostic replay; original acceptance unchanged",
                "thresholds_si": validator.thresholds,
                "validation": validation,
                "experiments": output,
            },
            allow_nan=False,
        )
    )
    return target


def main():
    """Run the recipe; success exit status requires accepted calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recordings", type=Path)
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--report", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path("workspace/actuator"))
    parser.add_argument("--optimizer", choices=("cma_es", "grid"), default="cma_es")
    parser.add_argument(
        "--solver",
        choices=("SolverFeatherstone", "SolverMuJoCo"),
        help="synthetic demo backend; offline records use the solver in --config",
    )
    args = parser.parse_args()
    if args.replay:
        if args.solver:
            parser.error("--replay uses the saved solver")
        print(replay(args.replay))
        return 0
    if args.report:
        if args.solver:
            parser.error("--report uses saved trajectories")
        print(write_report(args.report))
        return 0
    if args.recordings:
        if args.solver:
            parser.error("set the offline solver in --config")
        if args.config is None:
            parser.error("--recordings requires an explicit --config")
        records, sources = read_file(args.recordings)
        config = json.loads(args.config.read_text())
    else:
        if args.config:
            parser.error("--config is only used with --recordings")
        records, config = synthetic_records(), synthetic_config(solver=args.solver or "SolverFeatherstone")
        sources = {"kind": "analytical_synthetic", "true_parameters": {"kp": 2.0, "kd": 0.15}}
    tuning = TuningActuator(records, config, workspace=args.workspace, optimizer=args.optimizer, sources=sources)
    result = tuning.run()
    print(result)
    if (result.run_directory / "resolved.json").exists():
        print(write_report(result.run_directory))
    return 0 if result.acceptance == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
