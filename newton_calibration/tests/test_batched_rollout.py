# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The batched Rollout path: one BatchEnvironment per experiment, repeat and chunk of trials."""

import unittest
from itertools import product
from unittest import mock

import numpy as np

from newton_calibration import Rollout, SimulationError
from newton_calibration.examples import (
    BatchedOscillatorAsset,
    OscillatorBatchEnvironment,
    make_tuning,
)
from newton_calibration.tests.unittest_utils import TempPathTestCase


def _rollout_inputs(tmp_path, **kwargs):
    tuning = make_tuning("oscillator", tmp_path, optimizer="grid", **kwargs)
    tuning.template.freeze()
    grid = np.array(list(product(np.linspace(0, 1, 3), repeat=2)))
    values = tuning.parameters.decode(grid)
    trials = [(f"t{i}", tuning.parameters.as_dict(v)) for i, v in enumerate(values)]
    trials.insert(2, ("bad", None))
    return tuning, trials, tuning.evidence.train_data


def _episodes(tuning, trials, experiments):
    return tuning.rollout.execute_batch(tuning.template, tuning.asset, trials, experiments)


class TestBatchedRollout(TempPathTestCase):
    def test_batched_matches_per_episode(self):
        """A batched asset gives the same episodes, traces and readback as per-episode environments."""
        single, trials, experiments = _rollout_inputs(self.tmp_path / "single")
        batched, _, _ = _rollout_inputs(self.tmp_path / "batched", batched=True)
        a = _episodes(single, trials, experiments)
        b = _episodes(batched, iter(trials), iter(experiments))
        self.assertEqual([e.key for e in a], [e.key for e in b])
        self.assertEqual([e.status for e in a], [e.status for e in b])
        for x, y in zip(a, b, strict=True):
            self.assertEqual(x.applied_parameters, y.applied_parameters)
            if x.status == "completed":
                np.testing.assert_allclose(x.trace.values, y.trace.values, rtol=0, atol=0)
                np.testing.assert_allclose(x.trace.times, y.trace.times, rtol=0, atol=0)
        self.assertEqual(
            [e.status for e in b if e.trial_id == "bad"],
            ["invalid_parameters"] * len(experiments),
        )

    def test_worlds_splits_a_generation_without_changing_results(self):
        """Chunking a generation into several batch environments does not change any episode."""
        whole, trials, experiments = _rollout_inputs(self.tmp_path / "whole", batched=True)
        chunked, _, _ = _rollout_inputs(self.tmp_path / "chunked", batched=True, worlds=4)
        self.assertEqual(chunked.rollout.worlds, 4)
        a, b = _episodes(whole, trials, experiments), _episodes(chunked, trials, experiments)
        for x, y in zip(a, b, strict=True):
            self.assertEqual((x.key, x.status), (y.key, y.status))
            if x.status == "completed":
                np.testing.assert_allclose(x.trace.values, y.trace.values, rtol=0, atol=0)

    def test_failing_member_fails_alone(self):
        """A SimulationError in one member's slot marks only that episode as a numerical failure."""

        class OneBadMember(OscillatorBatchEnvironment):
            def run(self, times, *, seed, deadline):
                results = super().run(times, seed=seed, deadline=deadline)
                results[1] = SimulationError("member diverged")
                return results

        class OneBadAsset(BatchedOscillatorAsset):
            def create_batch_environment(self, template, parameter_values, experiment_settings):
                return OneBadMember(parameter_values, template.numeric_settings["dt"], experiment_settings)

        tuning, trials, experiments = _rollout_inputs(self.tmp_path, batched=True)
        tuning.asset = OneBadAsset(tuning.asset.baseline_parameters)
        live = [t for t in trials if t[1] is not None]
        episodes = _episodes(tuning, live, experiments[:1])
        self.assertEqual(
            [e.status for e in episodes],
            ["completed", "numerical_failure"] + ["completed"] * (len(live) - 2),
        )
        self.assertEqual(episodes[1].error, "member diverged")
        self.assertIsNone(episodes[1].trace)

    def test_readback_mismatch_fails_the_chunk(self):
        """A wrong readback in one member fails every episode of its chunk, and the run fails."""

        class WrongReadback(OscillatorBatchEnvironment):
            def __init__(self, parameters, dt, settings):
                super().__init__(parameters, dt, settings)
                self.applied_parameters[0] = {**self.applied_parameters[0], "stiffness": 999.0}

        class WrongAsset(BatchedOscillatorAsset):
            def create_batch_environment(self, template, parameter_values, experiment_settings):
                return WrongReadback(parameter_values, template.numeric_settings["dt"], experiment_settings)

        tuning, trials, experiments = _rollout_inputs(self.tmp_path / "chunk", batched=True)
        tuning.asset = WrongAsset(tuning.asset.baseline_parameters)
        live = [t for t in trials if t[1] is not None]
        episodes = _episodes(tuning, live, experiments[:1])
        self.assertEqual({e.status for e in episodes}, {"execution_error"})
        for e in episodes:
            self.assertIn("readback", e.error)

        run = make_tuning("oscillator", self.tmp_path / "run", optimizer="grid", batched=True)
        run.asset = WrongAsset(run.asset.baseline_parameters)
        result = run.run()
        self.assertEqual(result.acceptance, "run_failed")

    def test_batched_oscillator_run_is_accepted(self):
        """The full lifecycle accepts the batched oscillator and matches the per-episode run."""
        single = make_tuning("oscillator", self.tmp_path / "single", optimizer="grid")
        batched = make_tuning("oscillator", self.tmp_path / "batched", optimizer="grid", batched=True, worlds=16)
        a, b = single.run(), batched.run()
        self.assertEqual(b.acceptance, "accepted", batched.state.errors)
        self.assertEqual(a.parameters, b.parameters)
        for x, y in zip(single.state.batch_history, batched.state.batch_history, strict=True):
            self.assertEqual(x["trial_ids"], y["trial_ids"])
            np.testing.assert_allclose(x["objectives"], y["objectives"], rtol=0, atol=0)

    def test_rollout_rejects_bad_worlds(self):
        """worlds must be None or a positive integer."""
        for worlds in (0, -1, 2.5, True):
            with self.subTest(worlds=worlds), self.assertRaises(ValueError):
                Rollout(worlds=worlds)

    def test_invalid_batch_result_is_an_execution_error(self):
        tuning, trials, experiments = _rollout_inputs(self.tmp_path, batched=True)
        live = [trial for trial in trials if trial[1] is not None]
        with mock.patch.object(
            OscillatorBatchEnvironment,
            "run",
            lambda self, *args, **kwargs: [None] * len(self.applied_parameters),
        ):
            episodes = _episodes(tuning, live, experiments[:1])
        self.assertTrue(episodes)
        for e in episodes:
            self.assertEqual(e.status, "execution_error")
            self.assertIsNone(e.trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
