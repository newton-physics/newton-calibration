# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Immutable, explicit single-environment construction conditions."""

import math

from .data import identity, immutable, plain


class Template:
    """``Template`` stores the physical conditions, numerical settings, and experiment
    inputs shared by all candidates. When constructing an environment, it merges fixed
    and candidate parameters and passes the experiment's settings to the asset adapter.
    The adapter defines and interprets the dictionary keys.
    """

    def __init__(
        self,
        *,
        numeric_settings,
        discretization,
        fixed_parameters,
        experiment_settings,
        refinements=(),
        precision_tolerance=0.05,
        qualification_mode="refinement",
    ):
        """Initialize the template.

        Args:
            numeric_settings: Required nonempty dictionary for integrator, timestep, substeps,
                iterations, and related settings, using keys understood by the environment.
            discretization: Required dictionary for rod segments, mesh resolution, or degrees
                of freedom. May be empty when appropriate.
            fixed_parameters: Required dictionary of fixed physical scalars, with keys disjoint
                from search parameters.
            experiment_settings: Required nonempty dictionary keyed by every experiment ``id``.
                Each entry explicitly includes ``seed`` and adapter-specific inputs.
            refinements: Each configuration override may contain only ``numeric_settings`` and
                ``discretization``. Defaults to ``()``.
            precision_tolerance: Positive and finite, limiting maximum feature changes in
                numerical checks. Defaults to ``0.05``.
            qualification_mode: ``"refinement"`` requires valid refinement settings during a
                run. ``"repeatability"`` with empty ``refinements`` checks a fixed configuration.
                Defaults to ``"refinement"``.
        """
        self.numeric_settings = dict(numeric_settings)
        self.discretization = dict(discretization)
        self.fixed_parameters = dict(fixed_parameters)
        self.experiment_settings = dict(experiment_settings)
        self.refinements = tuple(refinements)
        self.precision_tolerance = precision_tolerance
        self.qualification_mode = qualification_mode
        self._frozen = False

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("Template is frozen; create a new version")
        object.__setattr__(self, name, value)

    def resolve(self):
        """``resolve()`` checks configuration."""
        if self.qualification_mode not in {"refinement", "repeatability"}:
            raise ValueError("unknown qualification mode")
        if self.qualification_mode == "repeatability" and self.refinements:
            raise ValueError("fixed-configuration repeatability cannot include refinements")
        if not self.numeric_settings or not self.experiment_settings:
            raise ValueError("numeric settings and experiment definitions must be explicit")
        if not math.isfinite(self.precision_tolerance) or self.precision_tolerance <= 0:
            raise ValueError("invalid numerical precision tolerance")

        def check(value):
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite template setting")

        check(self.snapshot())
        return self

    def snapshot(self):
        """``snapshot()`` returns serializable configuration data."""
        return {
            k: plain(getattr(self, k))
            for k in (
                "numeric_settings",
                "discretization",
                "fixed_parameters",
                "experiment_settings",
                "refinements",
                "precision_tolerance",
                "qualification_mode",
            )
        }

    @property
    def template_id(self):
        """``template_id`` identifies the configuration."""
        return identity(self.snapshot())

    def freeze(self):
        """``freeze()`` locks the configuration."""
        self.resolve()
        for key in (
            "numeric_settings",
            "discretization",
            "fixed_parameters",
            "experiment_settings",
            "refinements",
        ):
            setattr(self, key, immutable(getattr(self, key)))
        self._frozen = True
        return self

    def refined(self, change):
        """``refined(change)`` creates a frozen refined template, with each refinement applied
        independently to the original."""
        data = self.snapshot()
        for key, values in change.items():
            if key not in {"numeric_settings", "discretization"}:
                raise ValueError("numerical refinement cannot alter physical parameters")
            data[key].update(plain(values))
        data["refinements"] = []
        return Template(**data).freeze()

    def instantiate(self, asset, parameter_values, experiment):
        """``instantiate(asset, parameter_values, experiment)`` constructs one environment."""
        if not self._frozen:
            raise RuntimeError("freeze Template before executing")
        if set(parameter_values) & set(self.fixed_parameters):
            raise ValueError("a parameter cannot be both calibrated and fixed")
        settings = self.experiment_settings[experiment.id]
        if "seed" not in settings:
            raise ValueError("each experiment must explicitly declare its seed")
        complete = {**dict(self.fixed_parameters), **parameter_values}
        return asset.create_environment(self, complete, settings)

    def instantiate_batch(self, asset, parameter_values, experiment):
        """Create one BatchEnvironment for several candidates, with the checks of instantiate.

        ``instantiate_batch(...)`` constructs a batch. ``Rollout`` normally calls these methods.
        """
        if not self._frozen:
            raise RuntimeError("freeze Template before executing")
        settings = self.experiment_settings[experiment.id]
        if "seed" not in settings:
            raise ValueError("each experiment must explicitly declare its seed")
        complete = []
        for values in parameter_values:
            if set(values) & set(self.fixed_parameters):
                raise ValueError("a parameter cannot be both calibrated and fixed")
            complete.append({**dict(self.fixed_parameters), **values})
        return asset.create_batch_environment(self, tuple(complete), settings)
