"""Pack independent, contact-free rods into one Newton model per experiment batch."""

import math
import time

import newton
import warp as wp
from task_tuning.data import Episode

from task_tuning import Rollout, SimulationError

from .newton_backend import CableEnvironment


class NewtonRollout(Rollout):
    def __init__(self, capacity=16):
        super().__init__(capacity, parameter_rtol=2e-6, parameter_atol=1e-8)

    def execute_batch(self, template, asset, trials, experiments, *, deadline=None):
        by_key = {}
        for exp in experiments:
            settings = template.experiment_settings[exp.id]
            for repeat in range(exp.repeats):
                for start in range(0, len(trials), self.capacity):
                    chunk = trials[start : start + self.capacity]
                    live = []
                    for trial_id, values in chunk:
                        episode = Episode(trial_id, exp.id, repeat, "completed")
                        by_key[episode.key] = episode
                        if values is None:
                            episode.status = "invalid_parameters"
                        else:
                            live.append((episode, {**dict(template.fixed_parameters), **values}))
                    if not live:
                        continue
                    parts = []
                    try:
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError("Newton batch deadline")
                        with wp.ScopedDevice(template.numeric_settings["device"]):
                            builder = newton.ModelBuilder(gravity=(0, 0, -live[0][1]["gravity"]))
                            parts = []
                            for episode, parameters in live:
                                part = CableEnvironment(
                                    template, parameters, settings, asset.config, build_only=True
                                )
                                body_offset, joint_offset = builder.body_count, builder.joint_count
                                builder.add_builder(part.builder)
                                part.bodies = part.bodies + body_offset
                                part.joints = part.joints + joint_offset
                                if part.plug_body is not None:
                                    part.plug_body += body_offset
                                parts.append(part)
                            runner = parts[0]
                            runner._initialize(builder, parts)
                            for (episode, expected), part in zip(live, parts, strict=True):
                                actual = part.applied_parameters
                                if expected.keys() != actual.keys() or any(
                                    not math.isclose(
                                        float(v),
                                        float(actual[k]),
                                        rel_tol=self.parameter_rtol,
                                        abs_tol=self.parameter_atol,
                                    )
                                    for k, v in expected.items()
                                ):
                                    raise RuntimeError("packed model parameter readback mismatch")
                                episode.applied_parameters = dict(actual)
                            traces = runner.run_batch(
                                exp.times.clone(),
                                seed=int(settings["seed"]) + repeat,
                                deadline=deadline,
                            )
                            for (episode, _), trace in zip(live, traces, strict=True):
                                if isinstance(trace, SimulationError):
                                    episode.status = "numerical_failure"
                                    episode.error = str(trace)
                                else:
                                    episode.trace = trace
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError("Newton batch exceeded deadline")
                    except TimeoutError as error:
                        for episode, _ in live:
                            episode.status, episode.error = "timeout", str(error)
                    except SimulationError as error:
                        for episode, _ in live:
                            episode.status, episode.error = "numerical_failure", str(error)
                    except Exception as error:
                        for episode, _ in live:
                            episode.status = "execution_error"
                            episode.error = f"{type(error).__name__}: {error}"
                    finally:
                        for part in parts:
                            part.close()
        return tuple(
            by_key[(trial_id, exp.id, repeat)]
            for trial_id, _ in trials
            for exp in experiments
            for repeat in range(exp.repeats)
        )
