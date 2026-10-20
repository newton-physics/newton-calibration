# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from task_tuning.examples import make_tuning


@pytest.mark.parametrize("case", ["oscillator", "sliding"])
def test_generic_run_recovers_parameters_and_reloads_asset(case, tmp_path):
    tuning = make_tuning(case, tmp_path / case, optimizer="grid")
    result = tuning.run()
    assert result.acceptance == "accepted", tuning.state.errors
    assert tuning.state.analysis_reports["identifiability"]["rank"] == len(tuning.parameters.names)
    assert tuning.state.decision_history[-1]["reason"]
    assert result.asset_path.exists()
    assert (result.run_directory / "comparison.usda").exists()
    report = json.loads((result.run_directory / "run_state.json").read_text())
    assert report["validation_report"]["passed"]
    assert report["budget_usage"]["analysis"]["rollouts"] > 0
    assert report["template_snapshot"]["numeric_settings"]


def test_changing_validation_does_not_change_training(tmp_path):
    first = make_tuning("sliding", tmp_path / "first", optimizer="grid")
    second = make_tuning("sliding", tmp_path / "second", optimizer="grid")
    second.evidence.validation_data[0].reference.add_(10)
    result_a, result_b = first.run(), second.run()
    assert result_a.parameters == result_b.parameters
    assert result_a.acceptance == "accepted"
    assert result_b.acceptance == "rejected_validation"
    a = [x["objectives"] for x in first.state.batch_history if x["purpose"] == "train"]
    b = [x["objectives"] for x in second.state.batch_history if x["purpose"] == "train"]
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_workspace_keeps_repeated_runs_and_their_intermediate_files(tmp_path):
    workspace = tmp_path / "shared workspace"
    first = make_tuning("sliding", workspace, optimizer="grid")
    second = make_tuning("sliding", workspace, optimizer="grid")
    assert not workspace.exists()
    result_a = first.run()
    saved_state = (result_a.run_directory / "run_state.json").read_bytes()
    result_b = second.run()
    assert result_a.run_directory != result_b.run_directory
    assert (result_a.run_directory / "run_state.json").read_bytes() == saved_state
    for tuning, result in ((first, result_a), (second, result_b)):
        assert result.acceptance == "accepted"
        assert result.run_directory.parent == workspace
        resolved = json.loads((result.run_directory / "resolved.json").read_text())
        assert resolved["workspace"] == str(workspace)
        assert resolved["run_directory"] == str(result.run_directory)
        assert (result.run_directory / "candidate_asset.json").is_file()
        assert result.asset_path.is_relative_to(result.run_directory)
        for record in tuning.state.batch_history:
            batch_path = result.run_directory / record["artifact"]
            batch = torch.load(batch_path, weights_only=True)
            assert batch["trial_ids"] == record["trial_ids"]
        assert {"baseline", "analysis", "train", "validation"} <= {
            record["purpose"] for record in tuning.state.batch_history
        }


def test_default_workspace_is_resolved_at_construction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tuning = make_tuning("sliding", optimizer="grid")
    assert tuning.workspace == tmp_path / "workspace"
    assert tuning.run_directory.parent == tuning.workspace
    assert not tuning.workspace.exists()


def test_workspace_file_is_not_overwritten(tmp_path):
    workspace = tmp_path / "file"
    workspace.write_text("keep")
    tuning = make_tuning("sliding", workspace, optimizer="grid")
    with pytest.raises(FileExistsError):
        tuning.run()
    assert workspace.read_text() == "keep"


def test_existing_generated_run_directory_is_not_reused(tmp_path):
    tuning = make_tuning("sliding", tmp_path, optimizer="grid")
    tuning.run_directory.mkdir()
    with pytest.raises(FileExistsError):
        tuning.run()
    assert list(tuning.run_directory.iterdir()) == []
