# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import math
import unittest
from dataclasses import replace

import numpy as np

from newton_calibration import (
    CMAES,
    Asset,
    Decision,
    Environment,
    Episode,
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
from newton_calibration.examples import make_tuning
from newton_calibration.tests.unittest_utils import TempPathTestCase


class SumEnvironment(Environment):
    def __init__(self, values):
        self.applied_parameters = dict(values)

    def run(self, times, *, seed, deadline):
        value = sum(self.applied_parameters.values())
        return Trace(
            times,
            np.full((len(times), 1), value, dtype=np.float64),
            np.zeros((len(times), 1, 3), dtype=np.float64),
        )


class SumAsset(Asset):
    def create_environment(self, template, parameter_values, experiment_settings):
        return SumEnvironment(parameter_values)


def make_sum(workspace, allow_reduction=False):
    parameters = Parameters([Parameter("a", (0.5, 3.5), 1.5), Parameter("b", (0.5, 3.5), 1.5)])
    evidence = Evidence(
        [Experiment("train", "capture1", np.array([[4.0]]))],
        [Experiment("val", "capture2", np.array([[4.0]]))],
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


class BrokenReadback(SumAsset):
    def create_environment(self, template, parameter_values, experiment_settings):
        env = super().create_environment(template, parameter_values, experiment_settings)
        env.applied_parameters["a"] += 1
        return env


class CorruptExport(SumAsset):
    def reload(self, path):
        loaded = super().reload(path)
        loaded.baseline_parameters = {"a": 100.0, "b": 100.0}
        return loaded


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


class TestInvariants(TempPathTestCase):
    def test_rank_deficiency_is_not_mistaken_for_unique_parameter_recovery(self):
        tuning = make_sum(self.tmp_path / "rank")
        result = tuning.run()
        self.assertEqual(result.acceptance, "inconclusive")
        self.assertEqual(result.termination_reason, "inconclusive_parameters")
        self.assertNotIn("train", tuning.state.budget_usage)
        self.assertEqual(tuning.state.analysis_reports["identifiability"]["rank"], 1)
        self.assertIsNone(result.asset_path)

    def test_explicit_reduced_space_preserves_fixed_null_direction(self):
        tuning = make_sum(self.tmp_path / "reduced", True)
        result = tuning.run()
        self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
        self.assertEqual(tuning.parameters.dimension, 1)
        self.assertTrue(math.isclose(result.parameters["a"], result.parameters["b"], abs_tol=1e-8))
        self.assertLess(abs(sum(result.parameters.values()) - 4.0), 0.15)
        report = tuning.state.analysis_reports["identifiability"]
        self.assertEqual(report["fixed_unidentified_directions"], 1)
        with self.assertRaisesRegex(ValueError, "subspace"):
            tuning.parameters.encode(np.array([[1.0, 3.0]]))

    def test_batch_capacity_does_not_change_trials_or_scores(self):
        a = make_tuning("sliding", self.tmp_path / "one", optimizer="grid", capacity=1)
        b = make_tuning("sliding", self.tmp_path / "many", optimizer="grid", capacity=4)
        ra, rb = a.run(), b.run()
        self.assertEqual(ra.parameters, rb.parameters)
        for x, y in zip(a.state.batch_history, b.state.batch_history, strict=True):
            self.assertEqual(x["trial_ids"], y["trial_ids"])
            np.testing.assert_array_equal(x["objectives"], y["objectives"])

    def test_decision_owns_budget_and_optimizer_has_no_stop(self):
        tuning = make_tuning("sliding", self.tmp_path / "budget", optimizer="grid")
        tuning.decision = replace(tuning.decision, max_trials=3, target=0)
        result = tuning.run()
        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertEqual(tuning.state.budget_usage["train"]["trials"], 3)
        self.assertFalse(hasattr(tuning.optimizer, "stop"))
        stops = [d for d in tuning.state.decision_history if d["action"] == "stop"]
        self.assertTrue(stops)
        self.assertEqual(stops[0]["reason"], result.termination_reason)

    def test_analysis_budget_is_not_free(self):
        tuning = make_tuning("sliding", self.tmp_path / "analysis_budget", optimizer="grid")
        tuning.decision = replace(tuning.decision, max_rollouts=5)
        result = tuning.run()
        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertLessEqual(sum(x["rollouts"] for x in tuning.state.budget_usage.values()), 5)
        self.assertNotIn("train", tuning.state.budget_usage)

    def test_no_validation_cannot_publish(self):
        tuning = make_tuning("sliding", self.tmp_path / "no_val", optimizer="grid")
        tuning.evidence.validation_data = ()
        result = tuning.run()
        self.assertEqual(result.acceptance, "unvalidated")
        self.assertIsNone(result.asset_path)
        self.assertFalse((tuning.run_directory / "release").exists())

    def test_failed_numerical_qualification_is_rejected(self):
        tuning = make_tuning("oscillator", self.tmp_path / "coarse", optimizer="grid")
        tuning.template.numeric_settings["dt"] = 0.5
        tuning.template.precision_tolerance = 1e-12
        result = tuning.run()
        self.assertEqual(result.acceptance, "rejected_numerics")
        self.assertEqual(result.termination_reason, "numerical_config_unavailable")
        self.assertNotIn("train", tuning.state.budget_usage)

    def test_existing_run_is_not_overwritten(self):
        directory = self.tmp_path / "used"
        directory.mkdir()
        marker = directory / "marker"
        marker.write_text("keep")
        tuning = make_tuning("sliding", directory, optimizer="grid")
        result = tuning.run()
        self.assertEqual(result.acceptance, "accepted")
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(result.run_directory.parent, directory)
        self.assertEqual(set(directory.iterdir()), {marker, result.run_directory})

    def test_parameter_write_without_readback_does_not_qualify(self):
        tuning = make_sum(self.tmp_path / "bad_binding")
        tuning.asset = BrokenReadback(tuning.asset.baseline_parameters)
        result = tuning.run()
        self.assertEqual(result.acceptance, "run_failed")
        self.assertTrue(any("readback" in e for e in tuning.state.errors))
        self.assertIsNone(result.asset_path)

    def test_validation_uses_exported_asset_and_detects_corruption(self):
        tuning = make_sum(self.tmp_path / "bad_export", True)
        tuning.asset = CorruptExport(tuning.asset.baseline_parameters)
        result = tuning.run()
        self.assertEqual(result.acceptance, "run_failed")
        self.assertTrue(any("reloaded" in e for e in tuning.state.errors))
        self.assertFalse((tuning.run_directory / "release").exists())

    def test_scoring_rejects_missing_duplicate_and_nan(self):
        evidence = Evidence([Experiment("e", "g", np.array([[0.0], [1.0]]))])
        evidence.prepare()
        experiment = evidence.train_data[0]
        trace = Trace(experiment.times, experiment.reference.copy())
        episode = Episode("trial", "e", 0, "completed", trace)
        evaluator = Evaluator()
        for episodes in [(), (episode, episode)]:
            with self.subTest(episodes=len(episodes)), self.assertRaisesRegex(ValueError, "episode"):
                evaluator.score_batch(episodes, ("trial",), (experiment,))
        trace.values[0, 0] = float("nan")
        loss, _, _, valid, _ = evaluator.score_batch((episode,), ("trial",), (experiment,))
        self.assertFalse(valid[0])
        self.assertFalse(np.isfinite(loss[0]))

    def test_cross_entropy_scores_probability_distributions(self):
        evidence = Evidence([Experiment("e", "g", np.array([[0.25, 0.75]]))])
        evidence.prepare()
        experiment = evidence.train_data[0]
        episode = Episode("trial", "e", 0, "completed", Trace(experiment.times, experiment.reference.copy()))
        evaluator = Evaluator(loss="cross_entropy")
        loss, _, _, valid, _ = evaluator.score_batch((episode,), ("trial",), (experiment,))
        self.assertTrue(valid[0])
        np.testing.assert_allclose(float(loss[0]), -0.25 * math.log(0.25) - 0.75 * math.log(0.75), rtol=1e-6)
        episode.trace.values = np.array([[0.3, 0.3]])
        self.assertFalse(evaluator.score_batch((episode,), ("trial",), (experiment,))[3][0])

    def test_decision_sees_immutable_state_snapshot(self):
        tuning = make_sum(self.tmp_path / "state")
        view = tuning.state.view()
        with self.assertRaises(TypeError):
            view["phase"] = "hijacked"
        with self.assertRaises(TypeError):
            view["budget_usage"]["train"] = {}

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "usd-core is not installed")
    def test_gl_usd_outputs_do_not_affect_acceptance_or_optimizer(self):
        from pxr import Usd, UsdGeom

        tuning = make_tuning("sliding", self.tmp_path / "play", optimizer="grid")
        result = tuning.run()
        self.assertEqual(result.acceptance, "accepted")
        stage = Usd.Stage.Open(str(tuning.run_directory / "comparison.usda"))
        self.assertTrue(stage)
        self.assertEqual(stage.GetTimeCodesPerSecond(), 1)

        provenance = json.loads(stage.GetDefaultPrim().GetAttribute("provenance").Get())
        self.assertEqual(provenance["identity"]["template"], tuning.template.template_id)
        self.assertEqual(provenance["calibrated_parameters"], result.parameters)
        point_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Points)]
        self.assertEqual(len(point_prims), 4)
        for prim in point_prims:
            points = UsdGeom.Points(prim).GetPointsAttr()
            self.assertEqual(len(points.GetTimeSamples()), 41)
        before = tuning.optimizer.state_dict()
        tuning.play.export_usd(tuning.comparison, self.tmp_path / "second.usda")
        self.assertEqual(tuning.optimizer.state_dict(), before)

    @unittest.skipUnless(importlib.util.find_spec("cma"), "cma is not installed")
    def test_cma_es_recovers_two_parameters_with_same_parent(self):
        tuning = make_tuning("oscillator", self.tmp_path / "cma", optimizer="cma_es")
        self.assertIsInstance(tuning.optimizer, CMAES)
        result = tuning.run()
        self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
        np.testing.assert_allclose(result.parameters["stiffness"], 5.0, rtol=0.02)
        np.testing.assert_allclose(result.parameters["damping"], 0.7, rtol=0.03)
        checkpoint = tuning.state.optimizer_snapshot["checkpoint"]
        self.assertGreater((tuning.run_directory / checkpoint).stat().st_size, 0)

    def test_reloaded_asset_is_resimulated_even_when_parameter_metadata_matches(self):
        tuning = make_sum(self.tmp_path / "changed_dynamics", True)
        tuning.asset = CorruptDynamics(tuning.asset.baseline_parameters)
        result = tuning.run()
        self.assertEqual(result.acceptance, "rejected_training")
        self.assertFalse(tuning.state.analysis_reports["training_retest"]["passed"])
        self.assertIn("training_retest", tuning.state.budget_usage)
        self.assertFalse((tuning.run_directory / "release").exists())

    def test_cma_es_rejects_one_parameter(self):
        """Reject an unsupported search before asking pycma for a generation."""
        parameters = Parameters([Parameter("k", (0.0005, 0.05), 0.003, "m", "log")])
        with self.assertRaisesRegex(ValueError, "GridSearch"):
            CMAES(batch_size=4).initialize(parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
