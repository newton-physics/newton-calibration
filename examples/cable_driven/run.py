# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose a fresh driven-cable Tuning from a configuration.

Construction does not simulate. CLI execution requires --run.
"""

import argparse
import json
from pathlib import Path

from newton_calibration import (
    CMAES,
    Decision,
    Evidence,
    Parameter,
    Parameters,
    Play,
    Rollout,
    Template,
    Tuning,
    Validator,
)

from .backend import CableDrivenAsset, cable_layout, rest_levers, rest_parameter_names
from .data import CASE, experiment, experiment_settings, file_sha256, load_recordings
from .evaluator import CableDrivenEvaluator

CONFIG = CASE / "configs" / "orange_cable.json"
# The model stores the rod gains as float32, so the readback matches to about 1e-7.
READBACK_RTOL = 1e-6
# The rest displacements [m] pass through float32 rotations; a straight rest reads back as 0.
READBACK_ATOL = 2e-7


def _rest_shape(config, recordings):
    """Return ``(parameters, fixed)`` for the rest shape: fitted, or fixed straight.

    Each fitted value is an arc displacement [m] bounded by ``max_angle_rad`` times its
    joint's lever. All recordings must share the attachment, which sets the levers.
    """
    clamps = {float(r.data["settings"]["attachment"]["clamp_position_m"]) for r in recordings}
    if len(clamps) != 1:
        raise ValueError("the recordings must share the clamp position to share the rest-shape parameters")
    n, h, clamp, _ = cable_layout(float(config["asset"]["cable_length_m"]), config["discretization"], clamps.pop())
    names = rest_parameter_names(n)
    if not config["rest_shape"]["fit"]:
        return [], dict.fromkeys(names, 0.0)
    levers = rest_levers(n, h, clamp)
    limit = float(config["rest_shape"]["max_angle_rad"])
    parameters = [
        Parameter(name, (-limit * lever, limit * lever), 0.0, unit="m")
        for name, lever in zip(names, (lever for lever in levers for _ in "xy"), strict=True)
    ]
    return parameters, {}


def build_tuning(config_path=CONFIG, *, recordings_root, workspace="workspace/cable_driven"):
    """Return a fresh Tuning for the driven cable case. Nothing is simulated."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    recordings = {r.id: r for r in load_recordings(recordings_root)}
    used = [*config["train"], *config["validation"]]
    unknown = sorted(set(used) - set(recordings))
    if unknown:
        raise ValueError(f"config names recordings that do not exist: {unknown}")
    if len(set(used)) != len(used):
        raise ValueError("a recording is in both the training and the validation set")

    experiments = {k: experiment(recordings[k], scale=config["scale"]) for k in used}
    rest_parameters, rest_fixed = _rest_shape(config, [recordings[k] for k in used])
    parameters = Parameters([Parameter(**p) for p in config["parameters"]] + rest_parameters)
    template = Template(
        numeric_settings=config["numeric_settings"],
        discretization=config["discretization"],
        fixed_parameters={**config["fixed_parameters"], **rest_fixed},
        experiment_settings={k: experiment_settings(recordings[k], seed=i) for i, k in enumerate(used)},
        refinements=config["refinements"],
        precision_tolerance=config["precision_tolerance"],
    )
    sources = {
        "config_sha256": file_sha256(config_path),
        "recording_sha256": {k: file_sha256(recordings[k].directory / "recording.json") for k in used},
    }
    return Tuning(
        evidence=Evidence(
            [experiments[k] for k in config["train"]],
            [experiments[k] for k in config["validation"]],
            sources=sources,
        ),
        asset=CableDrivenAsset(parameters.as_dict(parameters.baseline), config=config["asset"]),
        evaluator=CableDrivenEvaluator(**config["evaluator"]),
        parameters=parameters,
        template=template,
        rollout=Rollout(worlds=config["worlds"], parameter_rtol=READBACK_RTOL, parameter_atol=READBACK_ATOL),
        optimizer=CMAES(**config["optimizer"]),
        decision=Decision(**config["decision"]),
        validator=Validator(**config["validator"]),
        play=Play(),
        workspace=workspace,
        # Search in the parameter box itself. Rotating it onto the identifiability
        # directions puts nearly all of the rotated box outside the parameter bounds with
        # 19 parameters, so almost every candidate would be rejected.
        orthogonalize=False,
        rank_tolerance=config["rank_tolerance"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--recordings", type=Path, required=True, help="Directory of recording directories")
    parser.add_argument("--workspace", type=Path, default=Path("workspace/cable_driven"))
    parser.add_argument("--run", action="store_true", help="Start qualification and a NEW search")
    args = parser.parse_args()
    tuning = build_tuning(args.config, recordings_root=args.recordings, workspace=args.workspace)
    print(f"parameters={tuning.parameters.names}, run_directory={tuning.run_directory}")
    if not args.run:
        print("Constructed only. No simulation performed. Add --run to start a new calibration.")
        return 0
    result = tuning.run()
    print(result)
    return 0 if result.acceptance == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
