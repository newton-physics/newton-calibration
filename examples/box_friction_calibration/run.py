# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit Newton XPBD contact friction to analytical sliding trajectories."""

import argparse
from pathlib import Path

import numpy as np

from newton_calibration import (
    Decision,
    Evaluator,
    Evidence,
    Experiment,
    GridSearch,
    Parameter,
    Parameters,
    Play,
    Rollout,
    Template,
    Tuning,
    Validator,
)

from .backend import SlidingAsset


def main():
    """Run the example and print its acceptance, stopping reason, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspace/newton-sliding"))
    parser.add_argument("--capacity", type=int, default=1, help="maximum number of concurrent CPU episodes")
    args = parser.parse_args()

    parameters = Parameters([Parameter("friction", bounds=(0.05, 0.65), baseline=0.15)])
    asset = SlidingAsset(
        parameters.as_dict(parameters.baseline),
        config={"half_extents": [0.1, 0.1, 0.05], "density": 1000.0},
    )
    settings = {
        "slow": {"initial_speed": 1.2, "seed": 1},
        "fast": {"initial_speed": 2.1, "seed": 2},
        "holdout": {"initial_speed": 1.7, "seed": 3},
    }
    times = np.linspace(0.0, 1.0, 41)
    gravity = 9.81
    reference_deceleration = 0.35 * gravity
    experiments = []
    for experiment_id, setting in settings.items():
        speed = setting["initial_speed"]
        moving_time = np.minimum(times, speed / reference_deceleration)
        position = speed * moving_time - 0.5 * reference_deceleration * moving_time**2
        experiments.append(
            Experiment(
                id=experiment_id,
                group=experiment_id,
                reference=position[:, None],
                times=times,
                scale=0.03,
            )
        )

    evidence = Evidence(
        train_data=experiments[:2],
        validation_data=experiments[2:],
        sources={"kind": "analytical synthetic sliding trajectories", "reference_friction": 0.35},
    )
    template = Template(
        numeric_settings={
            "solver": "SolverXPBD",
            "device": "cpu",
            "dt": 1.0 / 480.0,
            "iterations": 8,
            "dtype": "float32",
        },
        discretization={"rigid_bodies": 1, "collision_shapes": 2},
        fixed_parameters={"gravity": gravity},
        experiment_settings=settings,
        refinements=(
            {"numeric_settings": {"dt": 1.0 / 960.0}},
            {"numeric_settings": {"iterations": 16}},
        ),
        # Map a 1.5 mm position change to features for two equally weighted traces.
        precision_tolerance=0.0015 / (0.03 * np.sqrt(2 * len(times))),
        qualification_mode="refinement",
    )
    tuning = Tuning(
        evidence=evidence,
        asset=asset,
        evaluator=Evaluator(loss="squared_error", max_normalized_error=2.0),
        parameters=parameters,
        rollout=Rollout(capacity=args.capacity, parameter_rtol=1e-6),
        optimizer=GridSearch(levels=41, batch_size=8),
        decision=Decision(
            max_trials=41,
            max_rollouts=200,
            max_generations=6,
            max_seconds=180.0,
            target=0.005,
        ),
        validator=Validator(max_objective=0.01, max_error=1.0 / 6.0),
        template=template,
        play=Play(),
        workspace=args.workspace,
    )
    result = tuning.run()
    print(result.acceptance)
    print(result.termination_reason)
    print(result.parameters)
    print(result.run_directory)
    print(result.asset_path)
    return 0 if result.acceptance == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
