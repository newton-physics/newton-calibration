# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Assets own physical bindings; a reloaded candidate constructs fresh environments."""

from __future__ import annotations

import copy
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .data import Trace, identity, immutable, plain, write_json

if TYPE_CHECKING:
    from .template import Template


class SimulationError(RuntimeError):
    """Expected physical failure, as distinct from an implementation/infrastructure error."""


class Environment(ABC):
    """``Environment`` owns the mutable simulation state for one candidate and experiment
    and returns a ``Trace``."""

    applied_parameters: dict[str, float]
    """``applied_parameters`` reports the effective values, read back from fields actually
    used in computation. Its key set must exactly match the combined fixed and search
    parameter keys."""

    @abstractmethod
    def run(self, times: np.ndarray, *, seed: int, deadline: float | None) -> Trace:
        """Return a Trace at requested observation times; obey the monotonic deadline.

        ``run()`` samples at the requested ``times``. ``seed`` applies to the current repeat.
        ``deadline`` is an absolute ``time.monotonic()`` deadline checked by the environment
        at interruptible points.

        Raise ``SimulationError`` for expected physical or numerical candidate failures and
        ``TimeoutError`` for deadlines. The lifecycle records other exceptions as execution
        errors and handles the failure. Include enough context to locate the model, input,
        or numerical issue.
        """

    def close(self) -> None:  # noqa: B027
        """``close()`` releases environment resources. ``Rollout`` calls it after execution."""


class BatchEnvironment(ABC):
    """Several candidates of one experiment, simulated together.

    ``applied_parameters`` holds one mapping per member, read back from the model.

    Both ``BatchEnvironment.applied_parameters`` and ``run(times, seed=..., deadline=...)``
    results preserve candidate order.
    """

    applied_parameters: Sequence[dict[str, float]]

    @abstractmethod
    def run(self, times: np.ndarray, *, seed: int, deadline: float | None) -> Sequence[Trace | SimulationError]:
        """Return one Trace per member at the requested times, in member order.

        A member that fails physically returns a SimulationError in its slot; the
        other members still return their traces. An exception fails the whole batch.

        Each successful batch member returns a ``Trace``. For a physical failure, return a
        ``SimulationError`` instance in that member's slot. Raising an exception from the
        batch call affects the entire current chunk.
        """

    def close(self) -> None:
        return None


class Asset(ABC):
    """``Asset`` holds construction settings, baseline parameters, and parameter-binding
    logic. When adding a robot or material model, implement model construction, parameter
    application, initial-state reset, and observation readout in these classes.

    ``Asset`` and ``Environment`` are abstract base classes. Use existing subclasses or
    implement their abstract methods.
    """

    def __init__(self, baseline_parameters: Mapping[str, float], *, config: Mapping[str, Any] | None = None) -> None:
        """Initialize the asset.

        Args:
            baseline_parameters: The asset's ``baseline_parameters`` must exactly match
                ``parameters.as_dict(parameters.baseline)``. Fixed physical values belong to
                the template.
            config: ``config`` stores construction settings such as asset locations and model
                options, and is retained by default JSON export. Defaults to ``None``.
        """
        self.baseline_parameters = immutable(dict(baseline_parameters))
        self.config = immutable(dict(config or {}))

    def load_or_create(self) -> Asset:
        """Return the asset to calibrate. ``Tuning`` calls it at the start of a run. The
        default returns this asset."""
        return self

    def snapshot(self) -> dict[str, Any]:
        """Return the asset type, baseline parameters, and ``config`` as a record."""
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "baseline_parameters": self.baseline_parameters,
            "config": self.config,
        }

    @property
    def asset_id(self) -> str:
        """Identity of ``snapshot()``."""
        return identity(self.snapshot())

    @abstractmethod
    def create_environment(
        self, template: Template, parameter_values: Mapping[str, float], experiment_settings: Mapping[str, Any]
    ) -> Environment:
        """One fresh environment, with the applied parameter values available for readback.

        ``create_environment()`` receives merged fixed and candidate parameters, constructs
        an independent model, solver, and state, and applies experiment boundaries and
        initial conditions.
        """

    def create_batch_environment(
        self,
        template: Template,
        parameter_values: Sequence[Mapping[str, float]],
        experiment_settings: Mapping[str, Any],
    ) -> BatchEnvironment:
        """One fresh BatchEnvironment for a sequence of parameter mappings.

        Rollout uses this when the asset overrides it; otherwise it creates one
        environment per episode.

        Overriding ``Asset.create_batch_environment(template, parameter_values,
        experiment_settings)`` makes ``asset.supports_batch`` true and selects the batch
        path. ``parameter_values`` is a sequence of complete parameter dictionaries.
        """
        raise NotImplementedError

    @property
    def supports_batch(self) -> bool:
        return type(self).create_batch_environment is not Asset.create_batch_environment

    def export(self, parameter_values: Mapping[str, float], path: str | os.PathLike[str]) -> Path:
        """Default export writes ``newton-calibration-asset/1`` JSON with the asset type,
        candidate parameters, and construction configuration."""
        path = Path(path)
        write_json(
            path,
            {
                "schema": "newton-calibration-asset/1",
                "asset_type": self.snapshot()["type"],
                "parameters": parameter_values,
                "config": self.config,
            },
        )
        return path

    def reload(self, path: str | os.PathLike[str]) -> Self:
        """Calling ``reload(path)`` on the original asset returns an instance of the same type
        with the candidate baseline. Its ``create_environment()`` must still construct fresh
        mutable state. Subclasses that cache mutable models must handle rebuilding and
        isolation."""
        data = json.loads(Path(path).read_text())
        if (
            data["schema"] != "newton-calibration-asset/1"
            or data["asset_type"] != self.snapshot()["type"]
            or data["config"] != plain(self.config)
        ):
            raise ValueError("asset type or construction settings changed during export")
        loaded = copy.copy(self)
        loaded.baseline_parameters = immutable(data["parameters"])
        return loaded
