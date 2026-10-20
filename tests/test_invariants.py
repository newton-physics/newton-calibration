# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import replace

import pytest
import torch

from task_tuning import (
    CMAES,
    Asset,
    Decision,
    Environment,
    Evaluator,
    Evidence,
    Experiment,
    GridSearch,
    Parameter,
    Parameters,
    Play,
    Rollout,
    Template,
    Trace,
    Tuning,
    Validator,
)
from task_tuning.data import Episode
from task_tuning.examples import make_tuning


class SumEnvironment(Environment):
    def __init__(self, values):
        self.applied_parameters = dict(values)

    def run(self, times, *, seed, deadline):
        value = sum(self.applied_parameters.values())
        return Trace(
            times,
            torch.full((len(times), 1), value, dtype=torch.float64),
            torch.zeros((len(times), 1, 3), dtype=torch.float64),
        )


class SumAsset(Asset):
    def create_environment(self, template, parameter_values, experiment_settings):
        return SumEnvironment(parameter_values)


def make_sum(workspace, allow_reduction=False):
    parameters = Parameters([Parameter("a", (0.5, 3.5), 1.5), Parameter("b", (0.5, 3.5), 1.5)])
    evidence = Evidence(
        [Experiment("train", "capture1", torch.tensor([[4.0]]))],
        [Experiment("val", "capture2", torch.tensor([[4.0]]))],
    )
    template = Template(
        numeric_settings={"dt": 0.1},
        discretization={"size": 1},
        fixed_parameters={},
        experiment_settings={"train": {"seed": 0}, "val": {"seed": 1}},
        refinements=({"numeric_settings": {"dt": 0.05}},),
    )
    return Tuning(
        evidence=evidence,
        asset=SumAsset(parameters.as_dict(parameters.baseline)),
        template=template,
        parameters=parameters,
        evaluator=Evaluator(),
        optimizer=GridSearch(levels=31, batch_size=8),
        decision=Decision(target=1e-3),
        rollout=Rollout(2),
        validator=Validator(0.01, 0.15),
        play=Play(),
        workspace=workspace,
        allow_rank_reduction=allow_reduction,
    )


def test_rank_deficiency_is_not_mistaken_for_unique_parameter_recovery(tmp_path):
    tuning = make_sum(tmp_path / "rank")
    result = tuning.run()
    assert result.acceptance == "inconclusive"
    assert result.termination_reason == "inconclusive_parameters"
    assert "train" not in tuning.state.budget_usage
    assert tuning.state.analysis_reports["identifiability"]["rank"] == 1
    assert result.asset_path is None


def test_explicit_reduced_space_preserves_fixed_null_direction(tmp_path):
    tuning = make_sum(tmp_path / "reduced", True)
    result = tuning.run()
    assert result.acceptance == "accepted", tuning.state.errors
    assert tuning.parameters.dimension == 1
    assert math.isclose(result.parameters["a"], result.parameters["b"], abs_tol=1e-8)
    assert abs(sum(result.parameters.values()) - 4.0) < 0.15
    report = tuning.state.analysis_reports["identifiability"]
    assert report["fixed_unidentified_directions"] == 1
    with pytest.raises(ValueError, match="subspace"):
        tuning.parameters.encode(torch.tensor([[1.0, 3.0]]))


def test_batch_capacity_does_not_change_trials_or_scores(tmp_path):
    a = make_tuning("sliding", tmp_path / "one", optimizer="grid", capacity=1)
    b = make_tuning("sliding", tmp_path / "many", optimizer="grid", capacity=4)
    ra, rb = a.run(), b.run()
    assert ra.parameters == rb.parameters
    for x, y in zip(a.state.batch_history, b.state.batch_history, strict=True):
        assert x["trial_ids"] == y["trial_ids"]
        torch.testing.assert_close(x["objectives"], y["objectives"], rtol=0, atol=0)


def test_decision_owns_budget_and_optimizer_has_no_stop(tmp_path):
    tuning = make_tuning("sliding", tmp_path / "budget", optimizer="grid")
    tuning.decision = replace(tuning.decision, max_trials=3, target=0)
    result = tuning.run()
    assert result.termination_reason == "budget_exhausted"
    assert tuning.state.budget_usage["train"]["trials"] == 3
    assert not hasattr(tuning.optimizer, "stop")
    stops = [d for d in tuning.state.decision_history if d["action"] == "stop"]
    assert stops and stops[0]["reason"] == result.termination_reason


def test_analysis_budget_is_not_free(tmp_path):
    tuning = make_tuning("sliding", tmp_path / "analysis_budget", optimizer="grid")
    tuning.decision = replace(tuning.decision, max_rollouts=5)
    result = tuning.run()
    assert result.termination_reason == "budget_exhausted"
    assert sum(x["rollouts"] for x in tuning.state.budget_usage.values()) <= 5
    assert "train" not in tuning.state.budget_usage


def test_no_validation_cannot_publish(tmp_path):
    tuning = make_tuning("sliding", tmp_path / "no_val", optimizer="grid")
    tuning.evidence.validation_data = ()
    result = tuning.run()
    assert result.acceptance == "unvalidated"
    assert result.asset_path is None
    assert not (tuning.run_directory / "release").exists()


def test_failed_numerical_qualification_is_rejected(tmp_path):
    tuning = make_tuning("oscillator", tmp_path / "coarse", optimizer="grid")
    tuning.template.numeric_settings["dt"] = 0.5
    tuning.template.precision_tolerance = 1e-12
    result = tuning.run()
    assert result.acceptance == "rejected_numerics"
    assert result.termination_reason == "numerical_config_unavailable"
    assert "train" not in tuning.state.budget_usage


def test_existing_run_is_not_overwritten(tmp_path):
    directory = tmp_path / "used"
    directory.mkdir()
    marker = directory / "marker"
    marker.write_text("keep")
    tuning = make_tuning("sliding", directory, optimizer="grid")
    result = tuning.run()
    assert result.acceptance == "accepted"
    assert marker.read_text() == "keep"
    assert result.run_directory.parent == directory
    assert set(directory.iterdir()) == {marker, result.run_directory}


class BrokenReadback(SumAsset):
    def create_environment(self, template, parameter_values, experiment_settings):
        env = super().create_environment(template, parameter_values, experiment_settings)
        env.applied_parameters["a"] += 1
        return env


def test_parameter_write_without_readback_does_not_qualify(tmp_path):
    tuning = make_sum(tmp_path / "bad_binding")
    tuning.asset = BrokenReadback(tuning.asset.baseline_parameters)
    result = tuning.run()
    assert result.acceptance == "run_failed"
    assert any("readback" in e for e in tuning.state.errors)
    assert result.asset_path is None


class CorruptExport(SumAsset):
    def reload(self, path):
        loaded = super().reload(path)
        loaded.baseline_parameters = {"a": 100.0, "b": 100.0}
        return loaded


def test_validation_uses_exported_asset_and_detects_corruption(tmp_path):
    tuning = make_sum(tmp_path / "bad_export", True)
    tuning.asset = CorruptExport(tuning.asset.baseline_parameters)
    result = tuning.run()
    assert result.acceptance == "run_failed"
    assert any("reloaded" in e for e in tuning.state.errors)
    assert not (tuning.run_directory / "release").exists()


def test_scoring_rejects_missing_duplicate_and_nan():
    evidence = Evidence([Experiment("e", "g", torch.tensor([[0.0], [1.0]]))])
    evidence.prepare()
    experiment = evidence.train_data[0]
    trace = Trace(experiment.times, experiment.reference.clone())
    episode = Episode("trial", "e", 0, "completed", trace)
    evaluator = Evaluator()
    for episodes in [(), (episode, episode)]:
        with pytest.raises(ValueError, match="episode"):
            evaluator.score_batch(episodes, ("trial",), (experiment,))
    trace.values[0, 0] = float("nan")
    loss, _, _, valid, _ = evaluator.score_batch((episode,), ("trial",), (experiment,))
    assert not valid[0]
    assert not torch.isfinite(loss[0])


def test_cross_entropy_scores_probability_distributions():
    evidence = Evidence([Experiment("e", "g", torch.tensor([[0.25, 0.75]]))])
    evidence.prepare()
    experiment = evidence.train_data[0]
    episode = Episode(
        "trial", "e", 0, "completed", Trace(experiment.times, experiment.reference.clone())
    )
    evaluator = Evaluator(loss="cross_entropy")
    loss, _, _, valid, _ = evaluator.score_batch((episode,), ("trial",), (experiment,))
    assert valid[0]
    assert float(loss[0]) == pytest.approx(-0.25 * math.log(0.25) - 0.75 * math.log(0.75))
    episode.trace.values = torch.tensor([[0.3, 0.3]])
    assert not evaluator.score_batch((episode,), ("trial",), (experiment,))[3][0]


def test_decision_sees_immutable_state_snapshot(tmp_path):
    tuning = make_sum(tmp_path / "state")
    view = tuning.state.view()
    with pytest.raises(TypeError):
        view["phase"] = "hijacked"
    with pytest.raises(TypeError):
        view["budget_usage"]["train"] = {}


def test_gl_usd_outputs_do_not_affect_acceptance_or_optimizer(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    tuning = make_tuning("sliding", tmp_path / "play", optimizer="grid")
    result = tuning.run()
    assert result.acceptance == "accepted"
    stage = Usd.Stage.Open(str(tuning.run_directory / "comparison.usda"))
    assert stage and stage.GetTimeCodesPerSecond() == 1
    import json

    provenance = json.loads(stage.GetDefaultPrim().GetAttribute("provenance").Get())
    assert provenance["identity"]["template"] == tuning.template.template_id
    assert provenance["calibrated_parameters"] == result.parameters
    point_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Points)]
    assert len(point_prims) == 4
    for prim in point_prims:
        points = UsdGeom.Points(prim).GetPointsAttr()
        assert len(points.GetTimeSamples()) == 41
    before = tuning.optimizer.state_dict()
    tuning.play.export_usd(tuning.comparison, tmp_path / "second.usda")
    assert tuning.optimizer.state_dict() == before


def test_cma_es_recovers_two_parameters_with_same_parent(tmp_path):
    pytest.importorskip("cma")
    tuning = make_tuning("oscillator", tmp_path / "cma", optimizer="cma_es")
    assert isinstance(tuning.optimizer, CMAES)
    result = tuning.run()
    assert result.acceptance == "accepted", tuning.state.errors
    assert result.parameters["stiffness"] == pytest.approx(5.0, rel=0.02)
    assert result.parameters["damping"] == pytest.approx(0.7, rel=0.03)
    checkpoint = tuning.state.optimizer_snapshot["checkpoint"]
    assert (tuning.run_directory / checkpoint).stat().st_size > 0


def test_scoring_device_can_be_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    evidence = Evidence([Experiment("e", "g", torch.tensor([[1.0]]))])
    evidence.prepare()
    exp = evidence.train_data[0]
    raw = (Episode("t", "e", 0, "completed", Trace(exp.times, exp.reference.clone())),)
    scores = Evaluator(device="cuda").score_batch(raw, ("t",), (exp,))
    assert scores[0].device.type == "cuda"
    assert float(scores[0][0]) == 0


class CorruptDynamics(SumAsset):
    def reload(self, path):
        loaded = super().reload(path)
        original = loaded.create_environment

        def changed(template, parameter_values, experiment_settings):
            env = original(template, parameter_values, experiment_settings)
            original_run = env.run

            def biased(times, *, seed, deadline):
                trace = original_run(times, seed=seed, deadline=deadline)
                trace.values += 100.0
                return trace

            env.run = biased
            return env

        loaded.create_environment = changed
        return loaded


def test_reloaded_asset_is_resimulated_even_when_parameter_metadata_matches(tmp_path):
    tuning = make_sum(tmp_path / "changed_dynamics", True)
    tuning.asset = CorruptDynamics(tuning.asset.baseline_parameters)
    result = tuning.run()
    assert result.acceptance == "rejected_training"
    assert not tuning.state.analysis_reports["training_retest"]["passed"]
    assert "training_retest" in tuning.state.budget_usage
    assert not (tuning.run_directory / "release").exists()
