# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrate imported robots or revalidate a delivered bundle in a new process."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from newton_calibration import BatchResult, Evidence, Experiment, Rollout, Template

from .asset import AssetRobot
from .evaluation import EvaluatorRobot, ValidatorRobot
from .recording import read_file
from .tuning import TuningRobot


def replay(directory, output):
    """Validate saved evidence without searching or changing original acceptance."""
    root = Path(directory)
    resolved = json.loads((root / "resolved.json").read_text())
    candidate = root / ("candidate_asset.json" if (root / "candidate_asset.json").exists() else "asset.json")
    asset = AssetRobot.from_export(candidate)
    template = Template(**resolved["template"]).freeze()
    data = resolved["evidence"]
    evidence = Evidence(
        [Experiment(**item).normalized() for item in data["train"]],
        [Experiment(**item).normalized() for item in data["validation"]],
        sources=data["sources"],
    )
    evidence.prepare()
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        if (
            evidence.fingerprint() != manifest["identity"]["evidence"]
            or template.template_id != manifest["identity"]["template"]
        ):
            raise ValueError("delivered evidence or template identity changed")
        if dict(asset.baseline_parameters) != manifest["parameters"]:
            raise ValueError("delivered parameters differ from the accepted candidate")
    channels = sorted(
        {tuple(channel) for exp in evidence.train_data for channel in template.experiment_settings[exp.id]["channels"]}
    )
    evaluator = EvaluatorRobot(template.experiment_settings, channels)
    validator = ValidatorRobot(
        channels, data["sources"]["protocol"]["thresholds"], settings=template.experiment_settings
    )
    rollout = Rollout(capacity=1, parameter_rtol=1e-6)
    results, traces = {}, {}
    try:
        for exp in evidence.train_data + evidence.validation_data:
            episodes = rollout.execute_batch(template, asset, [("replay", asset.baseline_parameters)], (exp,))
            scores = evaluator.score_batch(episodes, ("replay",), (exp,))
            results[exp.id] = BatchResult(
                ("replay",), np.empty((1, 0)), np.empty((1, 0)), *scores, episodes, {}, "replay"
            )
            if any(episode.status != "completed" for episode in episodes):
                raise RuntimeError(f"replay failed: {[episode.error for episode in episodes]}")
            traces[exp.id] = {
                "times": episodes[0].trace.times.tolist(),
                "values": episodes[0].trace.values.tolist(),
                "metrics": scores[1][0].tolist(),
            }
        validation = validator.validate(
            asset, evidence.validation_data, lambda _asset, selected: results[selected[0].id]
        )
    finally:
        rollout.close()
    report = {
        "scope": "diagnostic replay; original acceptance unchanged",
        "validation": validation,
        "training_passed": all(validator.accepts(results[exp.id]) for exp in evidence.train_data),
        "experiments": traces,
        "provenance": data["sources"].get("provenance"),
    }
    with Path(output).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--asset", type=Path)
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--report", type=Path)
    parser.add_argument("--recordings", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, help="new diagnostic replay JSON")
    args = parser.parse_args()
    if args.replay:
        if args.output is None:
            parser.error("--replay requires --output")
        report = replay(args.replay, args.output)
        return 0 if report["validation"]["passed"] and report["training_passed"] else 1
    if args.report:
        from .report import write_report  # noqa: PLC0415

        print(write_report(args.report))
        return 0
    if not all((args.recordings, args.config, args.workspace)):
        parser.error("--asset requires --recordings, --config and --workspace")
    records, sources = read_file(args.recordings)
    config = json.loads(args.config.read_text())
    tuning = TuningRobot(AssetRobot.from_export(args.asset), records, config, workspace=args.workspace, sources=sources)
    result = tuning.run()
    if result.asset_path is not None:
        # Reference observations accompany the template/control protocol for independent replay.
        shutil.copyfile(tuning.run_directory / "resolved.json", result.asset_path.parent / "resolved.json")
    from .report import write_report  # noqa: PLC0415

    print(write_report(tuning.run_directory))
    return 0 if result.acceptance == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
