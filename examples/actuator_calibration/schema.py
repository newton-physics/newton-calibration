# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Typed actuator configuration (``actuator-calibration/1``) and its JSON Schema.

The pydantic models are the contract for configuration files. The JSON Schema in
``actuator-calibration.schema.json`` is generated from them for editors and other tools;
regenerate it after changing a model::

    python -m examples.actuator_calibration.schema
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator

SCHEMA = "actuator-calibration/1"
SCHEMA_PATH = Path(__file__).parent / "actuator-calibration.schema.json"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)


class ParameterSpec(_Model):
    """Search range of one actuator quantity, applied to every joint of its group."""

    bounds: tuple[NonNegativeFloat, NonNegativeFloat]
    """Lower and upper physical bound; equal bounds fix the quantity at that value."""
    baseline: NonNegativeFloat
    """Starting value inside the bounds."""
    unit: str = "1"
    """Physical unit, recorded with the parameter."""
    transform: Literal["linear", "log"] = "linear"
    """Search coordinate; ``log`` requires a positive lower bound."""

    @model_validator(mode="after")
    def _ordered(self):
        lower, upper = self.bounds
        if lower > upper:
            raise ValueError("bounds must be [lower, upper] with lower <= upper")
        if not lower <= self.baseline <= upper:
            raise ValueError("baseline must lie within the bounds")
        if self.transform == "log" and lower <= 0:
            raise ValueError("a log transform needs a positive lower bound")
        return self


class JointParameters(_Model):
    """The five quantities fitted for every joint of a group."""

    stiffness: ParameterSpec
    """PD drive stiffness ``joint_target_ke`` [N m/rad or N/m]."""
    damping: ParameterSpec
    """PD drive damping ``joint_target_kd`` [N m s/rad or N s/m]."""
    dry_friction: ParameterSpec
    """Joint dry friction ``joint_friction`` [N m or N]."""
    viscous_friction: ParameterSpec
    """Joint viscous damping ``joint_damping`` [N m s/rad or N s/m]."""
    armature: ParameterSpec
    """Joint armature ``joint_armature`` [kg m² or kg]."""


class JointGroup(_Model):
    """Joints that share bounds, baselines and fixed controller settings.

    Every joint still receives its own parameters.
    """

    names: list[str] = Field(min_length=1)
    """USD joint names."""
    delay_ms: NonNegativeFloat = 0.0
    """Command delay [ms], rounded to whole control steps."""
    effort_limit: PositiveFloat
    """Fixed drive effort clamp [N m or N]."""
    velocity_limit: PositiveFloat
    """Fixed joint velocity clamp [rad/s or m/s]."""
    parameters: JointParameters


class AssetConfig(_Model):
    """The robot USD."""

    path: str | None = None
    """USD path relative to the dataset root; omit when ``--usd`` always supplies the file."""
    usd_sha256: Sha256 | None = None
    """Required sha256 of the USD; omit to record the hash of whichever file is used."""
    provenance: str | None = None
    """Where the asset comes from."""
    fixed_base: bool = True
    """Attach the articulation root to the world."""


class NumericSettings(_Model):
    """Overrides of the MuJoCo solver settings for one numerical refinement."""

    physics_freq: PositiveInt | None = None
    """Physics frequency [Hz]."""
    solver: str | None = None
    """``SolverMuJoCo`` constraint solver."""
    integrator: str | None = None
    """``SolverMuJoCo`` integrator."""
    iterations: PositiveInt | None = None
    """Solver iterations."""
    ls_iterations: PositiveInt | None = None
    """Line-search iterations."""


class SimulationConfig(_Model):
    """Newton ``SolverMuJoCo`` settings for the replay."""

    physics_freq: PositiveInt
    """Physics frequency [Hz]; an integer multiple of ``control_freq``."""
    control_freq: PositiveInt
    """Command rate [Hz]; equals the resampled data rate ``1 / dataset.dt``."""
    solver: str
    """``SolverMuJoCo`` constraint solver, for example ``newton``."""
    integrator: str
    """``SolverMuJoCo`` integrator, for example ``implicitfast``."""
    iterations: PositiveInt
    """Solver iterations."""
    ls_iterations: PositiveInt
    """Line-search iterations."""


class Refinement(_Model):
    """One numerical refinement used for qualification; physical parameters stay fixed."""

    numeric_settings: NumericSettings


class DatasetConfig(_Model):
    """Recordings and how they are resampled."""

    provider: Literal["huggingface"] | None = None
    """Remote dataset provider; without one, pass ``--data-root``."""
    repo_id: str | None = None
    """Hugging Face dataset repository."""
    revision: str | None = None
    """Pinned dataset revision."""
    dt: PositiveFloat
    """Resampling interval [s]."""
    buffer_time: NonNegativeFloat = 0.0
    """Unscored settling time at the start of each recording [s]."""
    joint_name_mapping: dict[str, str] | None = None
    """Recording joint name to USD joint name."""
    train: dict[str, Sha256 | None] | list[str]
    """Search recordings: ``{path: sha256 or null}`` or a list of paths, relative to the dataset root."""
    validation: dict[str, Sha256 | None] | list[str]
    """Held-out recordings, in the same form as ``train``."""

    @model_validator(mode="after")
    def _provider_location(self):
        if self.provider == "huggingface" and not (self.repo_id and self.revision):
            raise ValueError("a huggingface dataset needs repo_id and revision")
        return self

    def files(self, split: Literal["train", "validation"]) -> dict[str, str | None]:
        """``{relative path: sha256 or None}`` of one split."""
        files = getattr(self, split)
        return dict(files) if isinstance(files, dict) else dict.fromkeys(files)


class SearchConfig(_Model):
    """CMA-ES search and its budgets."""

    world_count: PositiveInt
    """Candidates per generation, each simulated in its own world."""
    sigma: PositiveFloat
    """Initial CMA-ES step size in normalized coordinates."""
    seed: int | None = None
    """CMA-ES random seed."""
    max_generations: PositiveInt
    """Generation budget."""
    patience: PositiveInt | None = None
    """Generations without relative improvement before stopping."""
    relative_improvement_tolerance: NonNegativeFloat | None = None
    """Relative loss improvement that resets ``patience``."""
    max_seconds: PositiveFloat
    """Wall-clock budget [s]."""


class AnalysisConfig(_Model):
    """Identifiability decisions."""

    allow_rank_reduction: bool
    """Search a reduced subspace instead of stopping as inconclusive on rank deficiency."""
    orthogonalize: bool = True
    """Search orthogonalized combinations instead of normalized parameters."""


class AcceptanceConfig(_Model):
    """Per-recording thresholds for the training retest and held-out recordings."""

    max_objective_rad2: PositiveFloat
    """Maximum position MSE [rad²]."""
    max_error_rad: PositiveFloat
    """Maximum absolute position error [rad]."""


class ActuatorConfig(_Model):
    """Actuator calibration configuration."""

    json_schema: str | None = Field(None, alias="$schema")
    """Optional JSON Schema reference for editors; ignored by the loader."""
    schema_: Literal["actuator-calibration/1"] = Field(alias="schema")
    """Configuration format version."""
    description: str | None = None
    asset: AssetConfig
    joint_groups: list[JointGroup] = Field(min_length=1)
    simulation: SimulationConfig
    refinements: list[Refinement] = []
    pointwise_precision_tolerance_rad: PositiveFloat
    """Pointwise numerical qualification tolerance [rad]."""
    dataset: DatasetConfig
    search: SearchConfig
    analysis: AnalysisConfig
    acceptance: AcceptanceConfig
    record_traces: bool = False
    """Store per-candidate trajectories in batch files (large for long recordings)."""

    @model_validator(mode="after")
    def _consistent(self):
        control = self.simulation.control_freq
        physics = [self.simulation.physics_freq]
        physics += [r.numeric_settings.physics_freq for r in self.refinements if r.numeric_settings.physics_freq]
        if any(f % control for f in physics):
            raise ValueError("physics_freq must be an integer multiple of control_freq")
        if not math.isclose(self.dataset.dt, 1.0 / control, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("dataset dt must equal the control timestep")
        joints = [joint for group in self.joint_groups for joint in group.names]
        if duplicates := sorted({joint for joint in joints if joints.count(joint) > 1}):
            raise ValueError(f"joints {duplicates} appear more than once")
        return self


def json_schema() -> str:
    """The JSON Schema of :class:`ActuatorConfig`, as written to ``SCHEMA_PATH``."""
    schema = ActuatorConfig.model_json_schema(by_alias=True)
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    SCHEMA_PATH.write_text(json_schema(), encoding="utf-8")
    print(f"wrote {SCHEMA_PATH}")
