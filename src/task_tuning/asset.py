# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Assets own physical bindings; a reloaded candidate constructs fresh environments."""

import copy
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from .data import identity, immutable, plain, write_json


class SimulationError(RuntimeError):
    """Expected physical failure, as distinct from an implementation/infrastructure error."""


class Environment(ABC):
    applied_parameters: dict[str, float]

    @abstractmethod
    def run(self, times, *, seed, deadline):
        """Return a Trace at requested observation times; obey the monotonic deadline."""

    def close(self):
        pass


class BatchEnvironment(ABC):
    """Several candidates of one experiment, simulated together.

    ``applied_parameters`` holds one mapping per member, read back from the model.
    """

    applied_parameters: Sequence[dict[str, float]]

    @abstractmethod
    def run(self, times, *, seed, deadline):
        """Return one Trace per member at the requested times, in member order.

        A member that fails physically returns a SimulationError in its slot; the
        other members still return their traces. An exception fails the whole batch.
        """

    def close(self):
        pass


class Asset(ABC):
    def __init__(self, baseline_parameters, *, config=None):
        self.baseline_parameters = immutable(dict(baseline_parameters))
        self.config = immutable(dict(config or {}))

    def load_or_create(self):
        return self

    def snapshot(self):
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "baseline_parameters": self.baseline_parameters,
            "config": self.config,
        }

    @property
    def asset_id(self):
        return identity(self.snapshot())

    @abstractmethod
    def create_environment(self, template, parameter_values, experiment_settings):
        """One fresh environment, with the applied parameter values available for readback."""

    def create_batch_environment(self, template, parameter_values, experiment_settings):
        """One fresh BatchEnvironment for a sequence of parameter mappings.

        Rollout uses this when the asset overrides it; otherwise it creates one
        environment per episode.
        """
        raise NotImplementedError

    @property
    def supports_batch(self):
        return type(self).create_batch_environment is not Asset.create_batch_environment

    def export(self, parameter_values, path):
        path = Path(path)
        write_json(
            path,
            {
                "schema": "task-tuning-asset/1",
                "asset_type": self.snapshot()["type"],
                "parameters": parameter_values,
                "config": self.config,
            },
        )
        return path

    def reload(self, path):
        data = json.loads(Path(path).read_text())
        if (
            data["schema"] != "task-tuning-asset/1"
            or data["asset_type"] != self.snapshot()["type"]
            or data["config"] != plain(self.config)
        ):
            raise ValueError("asset type or construction settings changed during export")
        loaded = copy.copy(self)
        loaded.baseline_parameters = immutable(data["parameters"])
        return loaded
