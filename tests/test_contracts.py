# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from task_tuning import Evidence, Experiment, Parameter, Parameters, Template


def test_parameter_invalid_bounds_and_log_domain():
    with pytest.raises(ValueError):
        Parameter("k", (3.0, 1.0), 2.0)
    with pytest.raises(ValueError):
        Parameter("k", (-1.0, 2.0), 1.0, transform="log")


def test_rotated_parameters_roundtrip_and_no_silent_clipping():
    parameters = Parameters([Parameter("a", (0, 10), 5), Parameter("b", (0, 2), 1)])
    basis = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / 2**0.5
    parameters.set_mapping(torch.tensor([0.5, 0.5]), basis)
    value = torch.tensor([[7.0, 1.2]], dtype=torch.float64)
    torch.testing.assert_close(parameters.decode(parameters.encode(value)), value)
    assert not parameters.check_bounds(parameters.decode(torch.tensor([[1.0, 1.0]]))).all()
    parameters.freeze()
    with pytest.raises(RuntimeError):
        parameters.set_mapping(torch.zeros(2), torch.eye(2))


def test_template_freezes_nested_settings_and_separates_parameters():
    template = Template(
        numeric_settings={"dt": 0.01, "iterations": 4},
        discretization={"segments": [1, 2]},
        fixed_parameters={"mass": 1.0},
        experiment_settings={"train": {"seed": 1}},
    )
    template.freeze()
    with pytest.raises(TypeError):
        template.numeric_settings["dt"] = 0.02
    with pytest.raises((TypeError, AttributeError)):
        template.discretization["segments"].append(3)
    with pytest.raises(AttributeError):
        template.fixed_parameters = {}
    assert template.template_id


def test_evidence_rejects_same_acquisition_group_across_splits():
    a = Experiment("a", "same_capture", torch.tensor([[1.0]]))
    b = Experiment("b", "same_capture", torch.tensor([[2.0]]))
    with pytest.raises(ValueError, match="group"):
        Evidence([a], [b]).prepare()


def test_evidence_rejects_missing_valid_observations():
    bad = Experiment("bad", "g", torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError):
        Evidence([bad]).prepare()


def test_parameter_freeze_does_not_expose_mutable_bounds_or_mapping():
    parameters = Parameters([Parameter("x", (1.0, 3.0), 2.0)])
    parameters.freeze()
    fingerprint = parameters.parameter_space_id
    parameters.bounds.zero_()
    parameters.baseline.zero_()
    parameters.snapshot()["basis"].zero_()
    assert parameters.parameter_space_id == fingerprint
    assert float(parameters.bounds[0, 0]) == 1.0
    assert float(parameters.baseline[0]) == 2.0
    with pytest.raises(AttributeError, match="frozen"):
        parameters.names = ("replacement",)
