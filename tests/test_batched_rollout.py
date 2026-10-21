# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""The batched Rollout path: one BatchEnvironment per experiment, repeat and chunk of trials."""

import pytest
import torch

from task_tuning import Rollout, SimulationError
from task_tuning.examples import BatchedOscillatorAsset, OscillatorBatchEnvironment, make_tuning


def _rollout_inputs(tmp_path, **kwargs):
    tuning = make_tuning("oscillator", tmp_path, optimizer="grid", **kwargs)
    tuning.template.freeze()
    grid = torch.cartesian_prod(torch.linspace(0, 1, 3), torch.linspace(0, 1, 3))
    values = tuning.parameters.decode(grid)
    trials = [(f"t{i}", tuning.parameters.as_dict(v)) for i, v in enumerate(values)]
    trials.insert(2, ("bad", None))
    return tuning, trials, tuning.evidence.train_data


def _episodes(tuning, trials, experiments):
    return tuning.rollout.execute_batch(tuning.template, tuning.asset, trials, experiments)


def test_batched_matches_per_episode(tmp_path):
    """A batched asset gives the same episodes, traces and readback as per-episode environments."""
    single, trials, experiments = _rollout_inputs(tmp_path / "single")
    batched, _, _ = _rollout_inputs(tmp_path / "batched", batched=True)
    a = _episodes(single, trials, experiments)
    b = _episodes(batched, trials, experiments)
    assert [e.key for e in a] == [e.key for e in b]
    assert [e.status for e in a] == [e.status for e in b]
    for x, y in zip(a, b, strict=True):
        assert x.applied_parameters == y.applied_parameters
        if x.status == "completed":
            torch.testing.assert_close(x.trace.values, y.trace.values, rtol=0, atol=0)
            torch.testing.assert_close(x.trace.times, y.trace.times, rtol=0, atol=0)
    assert [e.status for e in b if e.trial_id == "bad"] == ["invalid_parameters"] * len(experiments)


def test_worlds_splits_a_generation_without_changing_results(tmp_path):
    """Chunking a generation into several batch environments does not change any episode."""
    whole, trials, experiments = _rollout_inputs(tmp_path / "whole", batched=True)
    chunked, _, _ = _rollout_inputs(tmp_path / "chunked", batched=True, worlds=4)
    assert chunked.rollout.worlds == 4
    a, b = _episodes(whole, trials, experiments), _episodes(chunked, trials, experiments)
    for x, y in zip(a, b, strict=True):
        assert (x.key, x.status) == (y.key, y.status)
        if x.status == "completed":
            torch.testing.assert_close(x.trace.values, y.trace.values, rtol=0, atol=0)


def test_failing_member_fails_alone(tmp_path):
    """A SimulationError in one member's slot marks only that episode as a numerical failure."""

    class OneBadMember(OscillatorBatchEnvironment):
        def run(self, times, *, seed, deadline):
            results = super().run(times, seed=seed, deadline=deadline)
            results[1] = SimulationError("member diverged")
            return results

    class OneBadAsset(BatchedOscillatorAsset):
        def create_batch_environment(self, template, parameter_values, experiment_settings):
            return OneBadMember(
                parameter_values, template.numeric_settings["dt"], experiment_settings
            )

    tuning, trials, experiments = _rollout_inputs(tmp_path, batched=True)
    tuning.asset = OneBadAsset(tuning.asset.baseline_parameters)
    live = [t for t in trials if t[1] is not None]
    episodes = _episodes(tuning, live, experiments[:1])
    assert [e.status for e in episodes] == ["completed", "numerical_failure"] + ["completed"] * (
        len(live) - 2
    )
    assert episodes[1].error == "member diverged" and episodes[1].trace is None


def test_readback_mismatch_fails_the_chunk(tmp_path):
    """A wrong readback in one member fails every episode of its chunk, and the run fails."""

    class WrongReadback(OscillatorBatchEnvironment):
        def __init__(self, parameters, dt, settings):
            super().__init__(parameters, dt, settings)
            self.applied_parameters[0] = {**self.applied_parameters[0], "stiffness": 999.0}

    class WrongAsset(BatchedOscillatorAsset):
        def create_batch_environment(self, template, parameter_values, experiment_settings):
            return WrongReadback(
                parameter_values, template.numeric_settings["dt"], experiment_settings
            )

    tuning, trials, experiments = _rollout_inputs(tmp_path / "chunk", batched=True)
    tuning.asset = WrongAsset(tuning.asset.baseline_parameters)
    live = [t for t in trials if t[1] is not None]
    episodes = _episodes(tuning, live, experiments[:1])
    assert {e.status for e in episodes} == {"execution_error"}
    assert all("readback" in e.error for e in episodes)

    run = make_tuning("oscillator", tmp_path / "run", optimizer="grid", batched=True)
    run.asset = WrongAsset(run.asset.baseline_parameters)
    result = run.run()
    assert result.acceptance == "run_failed"


def test_batched_oscillator_run_is_accepted(tmp_path):
    """The full lifecycle accepts the batched oscillator and matches the per-episode run."""
    single = make_tuning("oscillator", tmp_path / "single", optimizer="grid")
    batched = make_tuning(
        "oscillator", tmp_path / "batched", optimizer="grid", batched=True, worlds=16
    )
    a, b = single.run(), batched.run()
    assert b.acceptance == "accepted", batched.state.errors
    assert a.parameters == b.parameters
    for x, y in zip(single.state.batch_history, batched.state.batch_history, strict=True):
        assert x["trial_ids"] == y["trial_ids"]
        torch.testing.assert_close(x["objectives"], y["objectives"], rtol=0, atol=0)


def test_rollout_rejects_bad_worlds():
    """worlds must be None or a positive integer."""
    for worlds in (0, -1, 2.5):
        with pytest.raises(ValueError):
            Rollout(worlds=worlds)
