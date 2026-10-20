# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Threaded environment batches with fresh per-episode instances."""

import math
import time
from concurrent.futures import ThreadPoolExecutor

from .asset import SimulationError
from .data import Episode


class Rollout:
    def __init__(self, capacity=1, *, parameter_rtol=1e-10, parameter_atol=1e-12):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if any(not math.isfinite(x) or x < 0 for x in (parameter_rtol, parameter_atol)):
            raise ValueError("parameter readback tolerances must be finite and nonnegative")
        self.parameter_rtol, self.parameter_atol = parameter_rtol, parameter_atol
        self.capacity = capacity
        self._pool = None

    def execute_batch(self, template, asset, trials, experiments, *, deadline=None):
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
                expected = {**dict(template.fixed_parameters), **values}
                applied = dict(environment.applied_parameters)
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
