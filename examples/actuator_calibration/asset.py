# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A USD articulation whose actuator quantities are calibrated in Newton.

The baseline asset is the source USD with the configured baseline values applied to the
builder. ``export`` writes a sparse identified USD layer. ``reload`` reads that layer back
through Newton's USD importer without overrides, so the retest and validation simulate
exactly what a downstream user of the layer would load.

With ``recordings``, a chunk of candidates replays all of those recordings in one pass of
a model with one world per candidate and recording. Rollout still asks for one recording
at a time; the first request for a chunk runs the pass, and the others are served from it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from newton_calibration import Asset, BatchEnvironment, Environment
from newton_calibration.sources import file_sha256

from .config import CLAMPS, joint_values


class _SingleWorld(Environment):
    """Per-episode path: one single-world Newton model per episode (slow, but equivalent)."""

    def __init__(self, population, parameter_values, inputs):
        self.population, self.inputs = population, inputs
        self.applied_parameters = population.assign([dict(parameter_values)])[0]

    def run(self, times, *, seed, deadline):
        result = self.population.run(times, deadline=deadline, inputs=self.inputs)[0]
        if isinstance(result, Exception):
            raise result
        return result


class _Worlds(BatchEnvironment):
    """One chunk of candidates in a cached population; unused worlds repeat the last candidate."""

    def __init__(self, population, parameter_values, inputs):
        count = len(parameter_values)
        if not 1 <= count <= population.size:
            raise ValueError(f"a batch of {count} candidates does not fit {population.size} worlds")
        padded = [dict(v) for v in parameter_values]
        padded += [padded[-1]] * (population.size - count)
        self.population, self.inputs, self.count = population, inputs, count
        self.applied_parameters = population.assign(padded)[:count]

    def run(self, times, *, seed, deadline):
        return self.population.run(times, deadline=deadline, inputs=self.inputs)[: self.count]


def recording_key(inputs):
    """Identify a recording by its replay inputs: commanded positions and initial state."""
    digest = hashlib.sha256()
    for name in ("command_q", "initial_q"):
        array = np.ascontiguousarray(inputs[name], dtype=np.float64)
        digest.update(f"{name}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class _RecordingGroup:
    """Replays every recording of the group for one chunk of candidates in a single pass.

    World ``r * chunk + c`` replays recording r for candidate c. The pass runs on the
    first ``run`` for a chunk and serves the chunk's other recordings. The two most recent
    chunks are kept, so a batch larger than one chunk replays each chunk once.
    """

    def __init__(self, population, recordings, chunk):
        self.population, self.chunk = population, chunk
        self.keys = list(recordings)
        self.recordings = [recordings[k] for k in self.keys]
        self.world_recording = np.repeat(np.arange(len(self.keys), dtype=np.int32), chunk)
        self._passes = OrderedDict()

    def _assign(self, entry):
        padded = entry["requested"] + [entry["requested"][-1]] * (self.chunk - len(entry["requested"]))
        return self.population.assign(padded * len(self.keys))

    def prepare(self, parameter_values):
        requested = [dict(v) for v in parameter_values]
        key = tuple(tuple(sorted(v.items())) for v in requested)
        entry = self._passes.get(key)
        if entry is None:
            entry = {"requested": requested, "results": None}
            entry["applied"] = self._assign(entry)[: len(requested)]
            self._passes[key] = entry
            while len(self._passes) > 2:
                self._passes.popitem(last=False)
        self._passes.move_to_end(key)
        return entry

    def results(self, entry, recording, *, deadline):
        if entry["results"] is None:
            self._assign(entry)  # another chunk may have been assigned since
            entry["results"] = self.population.run_recordings(self.recordings, self.world_recording, deadline=deadline)
        start = self.keys.index(recording) * self.chunk
        return entry["results"][start : start + len(entry["requested"])]


class _GroupedWorlds(BatchEnvironment):
    """One recording of a chunk whose replay is shared with the group's other recordings."""

    def __init__(self, group, parameter_values, recording):
        if not 1 <= len(parameter_values) <= group.chunk:
            raise ValueError(f"a batch of {len(parameter_values)} candidates does not fit {group.chunk} worlds")
        self.group, self.recording = group, recording
        self.entry = group.prepare(parameter_values)
        self.applied_parameters = self.entry["applied"]

    def run(self, times, *, seed, deadline):
        expected = self.group.recordings[self.group.keys.index(self.recording)][0]
        if not np.array_equal(np.asarray(times, dtype=np.float64), expected):
            raise ValueError("observation times differ from the grouped recording's control grid")
        return self.group.results(self.entry, self.recording, deadline=deadline)


class ArticulationAsset(Asset):
    def __init__(
        self,
        baseline_parameters,
        *,
        usd_path,
        joints,
        fixed_parameters,
        fixed_base=True,
        apply_baseline=True,
        world_count=1,
        recordings=None,
        device=None,
    ):
        """``recordings`` lists replay inputs (``command_q``, ``initial_q``) replayed together."""
        usd_path = Path(os.path.abspath(Path(usd_path).expanduser()))
        super().__init__(
            baseline_parameters,
            config={
                "usd_sha256": file_sha256(usd_path),
                "joints": list(joints),
                "fixed_base": bool(fixed_base),
                "apply_baseline": bool(apply_baseline),
            },
        )
        self.usd_path = usd_path
        self.joints = tuple(joints)
        self.fixed_parameters = dict(fixed_parameters)
        self.fixed_base = bool(fixed_base)
        self.apply_baseline = bool(apply_baseline)
        self.world_count = int(world_count)
        self.device = device
        # An execution strategy, not part of the asset's identity.
        self.recordings = {recording_key(inputs): inputs for inputs in (recordings or ())}
        self._populations = {}  # template_id -> NewtonActuatorPopulation, built once per template
        self._groups = {}  # template_id -> _RecordingGroup over self._recordings

    def _builder_values(self):
        if not self.apply_baseline:
            return None
        values = {**self.fixed_parameters, **dict(self.baseline_parameters)}
        return joint_values({k: v for k, v in values.items() if "delay_steps" not in k}, self.joints)

    def _population(self, template, size):
        from .newton_population import NewtonActuatorPopulation  # noqa: PLC0415 (needs Newton or OpenUSD)

        return NewtonActuatorPopulation(
            self.usd_path,
            self.joints,
            template.numeric_settings,
            template.discretization,
            size,
            fixed_base=self.fixed_base,
            builder_values=self._builder_values(),
            device=self.device,
        )

    def create_environment(self, template, parameter_values, experiment_settings, *, inputs):
        return _SingleWorld(self._population(template, 1), parameter_values, inputs)

    def create_batch_environment(self, template, parameter_values, experiment_settings, *, inputs):
        recording = recording_key(inputs)
        if recording in self.recordings:
            group = self._groups.get(template.template_id)
            if group is None:
                size = self.world_count * len(self.recordings)
                times = {k: self._times(template, v) for k, v in self.recordings.items()}
                group = self._groups[template.template_id] = _RecordingGroup(
                    self._population(template, size),
                    {k: (times[k], v) for k, v in self.recordings.items()},
                    self.world_count,
                )
            return _GroupedWorlds(group, parameter_values, recording)
        # Building and capturing a replicated model is expensive; reuse it for every chunk.
        population = self._populations.get(template.template_id)
        if population is None:
            population = self._populations[template.template_id] = self._population(template, self.world_count)
        return _Worlds(population, parameter_values, inputs)

    @staticmethod
    def _times(template, inputs):
        control_dt = 1.0 / float(template.discretization["control_freq"])
        return control_dt * np.arange(1, len(inputs["command_q"]), dtype=np.float64)

    def export(self, parameter_values, path):
        from .usd_layer import write_identified_layer  # noqa: PLC0415 (needs Newton or OpenUSD)

        values = joint_values(parameter_values, self.joints)
        clamps = joint_values(
            {k: v for k, v in self.fixed_parameters.items() if k.rpartition(".")[2] in CLAMPS},
            self.joints,
        )
        metadata = json.dumps(
            {
                "schema": "newton-calibration-actuator-layer/1",
                "parameters": dict(parameter_values),
                "fixed_parameters": self.fixed_parameters,
                "source_usd_sha256": self.config["usd_sha256"],
            },
            sort_keys=True,
        )
        return write_identified_layer(self.usd_path, Path(path).with_suffix(".usda"), values, clamps, metadata)

    def reload(self, path, *, rtol=1e-6, atol=1e-9):
        from .newton_population import single_world_parameters  # noqa: PLC0415 (needs Newton or OpenUSD)
        from .usd_layer import sublayer_source  # noqa: PLC0415 (needs Newton or OpenUSD)

        if file_sha256(sublayer_source(path)) != self.config["usd_sha256"]:
            raise ValueError("identified layer does not reference the calibrated source USD")
        read = single_world_parameters(path, self.joints, fixed_base=self.fixed_base)
        for name, expected in self.fixed_parameters.items():
            if name.endswith(".delay_steps"):
                continue  # A controller setting, not a USD quantity.
            if not math.isclose(read[name], expected, rel_tol=rtol, abs_tol=atol):
                raise ValueError(f"reloaded {name}={read[name]} differs from fixed {expected}")
        return ArticulationAsset(
            {name: read[name] for name in self.baseline_parameters},
            usd_path=path,
            joints=self.joints,
            fixed_parameters=self.fixed_parameters,
            fixed_base=self.fixed_base,
            apply_baseline=False,
            world_count=self.world_count,
            recordings=list(self.recordings.values()),
            device=self.device,
        )

    def release(self, candidate_path, directory):
        from .usd_layer import relocate_layer  # noqa: PLC0415 (needs Newton or OpenUSD)

        return relocate_layer(candidate_path, Path(directory) / "asset.usda")
