# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compose the actuator calibration from the standard Tuning members."""

from __future__ import annotations

from pathlib import Path

from newton_calibration import Evaluator, Play, Rollout, Tuning

from .asset import ArticulationAsset
from .config import build_case, load_config, make_search, make_template, make_validator
from .data import load_evidence, resolve_asset

# Newton stores joint quantities in single precision.
READBACK = {"parameter_rtol": 1e-6, "parameter_atol": 1e-9}


def make_tuning(
    config_path,
    usd_path=None,
    workspace="workspace/actuator",
    *,
    data_root=None,
    device=None,
):
    """Return a Tuning that evaluates each generation in ``search.world_count`` replicated worlds.

    Without ``usd_path`` the robot USD is the configured dataset file at the pinned revision.
    """
    case = build_case(load_config(config_path))
    usd_path = resolve_asset(case.config, usd_path, data_root)
    evidence = load_evidence(case, data_root)
    train, validation = evidence.train_data, evidence.validation_data
    scored = sum(int(e.weight) for e in train)
    template = make_template(case, [e.id for e in train + validation], scored)
    optimizer, decision = make_search(case, len(train), len(validation))
    world_count = int(case.config["search"]["world_count"])
    rollout = Rollout(worlds=world_count, **READBACK)
    asset = ArticulationAsset(
        case.parameters.as_dict(case.parameters.baseline),
        usd_path=usd_path,
        joints=case.joints,
        fixed_parameters=case.fixed_parameters,
        fixed_base=case.config["asset"].get("fixed_base", True),
        world_count=world_count,
        device=device,
    )
    return Tuning(
        evidence=evidence,
        asset=asset,
        evaluator=Evaluator(loss="squared_error"),
        parameters=case.parameters,
        rollout=rollout,
        optimizer=optimizer,
        decision=decision,
        validator=make_validator(case),
        template=template,
        play=Play(),
        workspace=Path(workspace),
        allow_rank_reduction=bool(case.config["analysis"]["allow_rank_reduction"]),
    )
