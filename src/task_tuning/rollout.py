# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Environment batches: one BatchEnvironment per experiment, or threaded per-episode instances."""

import math
import time
from concurrent.futures import ThreadPoolExecutor

from .asset import SimulationError
from .data import Episode


class Rollout:
    """Run the episodes of a trial batch and return one Episode per (trial, experiment, repeat).

    An asset that overrides ``create_batch_environment`` gets one BatchEnvironment per
    experiment, repeat and chunk of at most ``worlds`` trials. Any other asset gets one
    Environment per episode on ``capacity`` threads.
    """

    def __init__(self, capacity=1, *, worlds=None, parameter_rtol=1e-10, parameter_atol=1e-12):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if worlds is not None and (not isinstance(worlds, int) or worlds < 1):
            raise ValueError("worlds must be None or a positive integer")
        if any(not math.isfinite(x) or x < 0 for x in (parameter_rtol, parameter_atol)):
            raise ValueError("parameter readback tolerances must be finite and nonnegative")
        self.parameter_rtol, self.parameter_atol = parameter_rtol, parameter_atol
        self.capacity = capacity
        self.worlds = worlds
        self._pool = None

    def execute_batch(self, template, asset, trials, experiments, *, deadline=None):
        if asset.supports_batch:
            return self._execute_batched(template, asset, trials, experiments, deadline)
        return self._execute_threaded(template, asset, trials, experiments, deadline)

    def _check_readback(self, template, values, applied):
        expected = {**dict(template.fixed_parameters), **values}
        if expected.keys() != applied.keys() or any(
            not math.isclose(
                float(v),
                float(applied[k]),
                rel_tol=self.parameter_rtol,
                abs_tol=self.parameter_atol,
            )
            for k, v in expected.items()
        ):
            raise RuntimeError("parameter readback does not match requested values")

    def _execute_batched(self, template, asset, trials, experiments, deadline):
        by_key = {}
        for exp in experiments:
            for repeat in range(exp.repeats):
                live = []
                for trial_id, values in trials:
                    episode = Episode(trial_id, exp.id, repeat, "completed")
                    by_key[episode.key] = episode
                    if values is None:
                        episode.status = "invalid_parameters"
                    else:
                        live.append((episode, values))
                size = self.worlds or max(len(live), 1)
                for start in range(0, len(live), size):
                    self._run_chunk(
                        template, asset, exp, repeat, live[start : start + size], deadline
                    )
        return tuple(
            by_key[trial_id, exp.id, repeat]
            for trial_id, _ in trials
            for exp in experiments
            for repeat in range(exp.repeats)
        )

    def _run_chunk(self, template, asset, exp, repeat, chunk, deadline):
        environment = None
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("rollout deadline reached")
            environment = template.instantiate_batch(asset, [values for _, values in chunk], exp)
            applied = list(environment.applied_parameters)
            if len(applied) != len(chunk):
                raise RuntimeError(
                    "batch environment did not report one parameter mapping per member"
                )
            for (episode, values), actual in zip(chunk, applied, strict=True):
                actual = dict(actual)
                self._check_readback(template, values, actual)
                episode.applied_parameters = actual
            seed = int(template.experiment_settings[exp.id]["seed"]) + repeat
            results = list(environment.run(exp.times.clone(), seed=seed, deadline=deadline))
            if len(results) != len(chunk):
                raise RuntimeError("batch environment did not return one result per member")
            for (episode, _), result in zip(chunk, results, strict=True):
                if isinstance(result, SimulationError):
                    episode.status, episode.error = "numerical_failure", str(result)
                else:
                    episode.trace = result
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("rollout exceeded cooperative deadline")
        except TimeoutError as error:
            for episode, _ in chunk:
                episode.status, episode.error, episode.trace = "timeout", str(error), None
        except SimulationError as error:
            for episode, _ in chunk:
                episode.status, episode.error, episode.trace = "numerical_failure", str(error), None
        except Exception as error:
            for episode, _ in chunk:
                episode.status, episode.trace = "execution_error", None
                episode.error = f"{type(error).__name__}: {error}"
        finally:
            if environment is not None:
                try:
                    environment.close()
                except Exception as error:
                    for episode, _ in chunk:
                        episode.status, episode.error = "execution_error", f"close: {error}"

    def _execute_threaded(self, template, asset, trials, experiments, deadline):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.capacity)
        requests = [
            (trial_id, values, exp, repeat)
            for trial_id, values in trials
            for exp in experiments
            for repeat in range(exp.repeats)
        ]

        def execute(request):
            trial_id, values, exp, repeat = request
            episode = Episode(trial_id, exp.id, repeat, "completed")
            if values is None:
                episode.status = "invalid_parameters"
                return episode
            environment = None
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("rollout deadline reached")
                environment = template.instantiate(asset, values, exp)
                applied = dict(environment.applied_parameters)
                self._check_readback(template, values, applied)
                episode.applied_parameters = applied
                seed = int(template.experiment_settings[exp.id]["seed"]) + repeat
                episode.trace = environment.run(exp.times.clone(), seed=seed, deadline=deadline)
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("rollout exceeded cooperative deadline")
            except TimeoutError as error:
                episode.status, episode.error = "timeout", str(error)
            except SimulationError as error:
                episode.status, episode.error = "numerical_failure", str(error)
            except Exception as error:
                episode.status = "execution_error"
                episode.error = f"{type(error).__name__}: {error}"
            finally:
                if environment is not None:
                    try:
                        environment.close()
                    except Exception as error:
                        episode.status, episode.error = "execution_error", f"close: {error}"
            return episode

        return tuple(self._pool.map(execute, requests))

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
