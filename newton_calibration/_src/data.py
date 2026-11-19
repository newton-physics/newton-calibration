# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Records shared by the ten members; only Tuning mutates RunState."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def identity(value: Any) -> str:
    data = json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()[:20]


def immutable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: immutable(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable(v) for v in value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass
class Trace:
    times: np.ndarray
    """``[T]``. Matches reference times, checked by the built-in evaluator with absolute
    tolerance ``1e-9``."""
    values: np.ndarray
    """``[T,D]``. Same channel order, units, and frame semantics as reference observations,
    finite wherever the mask selects data."""
    positions: np.ndarray | None = None  # [time, point, xyz], optional visualization
    """``None`` or ``[T,N,3]``. Optional display points. Default USDA output uses meters and Z-up."""


@dataclass
class Episode:
    """Each candidate, experiment, and repeat produces an ``Episode``, identified by
    ``(trial_id, experiment_id, repeat)``.
    """

    trial_id: str
    """Candidate identity."""
    experiment_id: str
    """Experiment identity."""
    repeat: int
    """Repetition identity."""
    status: str
    """``completed``, or a failure such as ``invalid_parameters``, ``numerical_failure``,
    ``timeout``, or ``execution_error``."""
    trace: Trace | None = None
    """Successful simulation ``Trace``, default ``None``."""
    error: str | None = None
    """Error description, default ``None``."""
    applied_parameters: dict[str, float] = field(default_factory=dict)
    """Full readback mapping, default empty dictionary."""

    @property
    def key(self):
        return self.trial_id, self.experiment_id, self.repeat


@dataclass
class TrialBatch:
    trial_ids: tuple[str, ...]
    search_values: np.ndarray


@dataclass
class BatchResult:
    """``BatchResult`` stores search and physical values, all five scoring outputs,
    ``episode_records``, problem ``identity``, and ``purpose``."""

    trial_ids: tuple[str, ...]
    search_values: np.ndarray
    parameter_values: np.ndarray
    objectives: np.ndarray
    metrics: np.ndarray
    constraints: np.ndarray
    valid_mask: np.ndarray
    features: np.ndarray
    episode_records: tuple[Episode, ...]
    identity: dict[str, str]
    purpose: str

    @property
    def feasible(self):
        """``feasible`` combines validity and constraints."""
        return self.valid_mask & (self.constraints <= 0).all(axis=1)

    @property
    def feedback(self):
        """``feedback`` maps feasible candidates below 1, constraint violations to ``[1,2)``,
        and invalid candidates to 3. Built-in CMA-ES updates from ``feedback``, while run
        records retain the raw objective."""
        # Feasible < 1, constraint violation in [1, 2), failed == 3.
        # Invalid rows have infinite objectives; zero them so inf / inf is never formed.
        loss = np.where(self.valid_mask, np.maximum(self.objectives, 0), 0.0)
        scores = loss / (1 + loss)
        if self.constraints.shape[1]:
            violation = np.maximum(self.constraints, 0).max(axis=1)
            scores = np.where(self.feasible, scores, 1 + violation / (1 + violation))
        return np.where(self.valid_mask, scores, 3.0)


@dataclass(frozen=True)
class DecisionResult:
    action: str
    reason: str
    batch_size: int = 0


@dataclass
class RunState:
    """``tuning.state`` is a ``RunState`` containing ``phase``, ``identity``,
    ``budget_usage``, ``batch_history``, ``best_candidate``, ``analysis_reports``,
    ``optimizer_snapshot``, ``decision_history``, ``validation_report``,
    ``artifact_refs``, and errors. Inspect ``run_state.json`` after execution to trace
    these records.
    """

    run_id: str = ""
    phase: str = "created"
    identity: dict[str, str] = field(default_factory=dict)
    template_snapshot: dict = field(default_factory=dict)
    budget_usage: dict = field(default_factory=dict)
    batch_history: list[dict] = field(default_factory=list)
    best_candidate: dict | None = None
    analysis_reports: dict = field(default_factory=dict)
    optimizer_snapshot: dict = field(default_factory=dict)
    decision_history: list[dict] = field(default_factory=list)
    validation_report: dict | None = None
    artifact_refs: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    fatal_error: str | None = None
    cancelled: bool = False
    termination_reason: str | None = None
    elapsed_s: float = 0.0

    def view(self):
        """``state.view()`` supplies a read-only decision snapshot."""
        return immutable(plain(self))


@dataclass(frozen=True)
class TuningResult:
    """``run()`` returns a ``TuningResult`` with five fields."""

    acceptance: str
    """Acceptance status, such as ``accepted``, ``rejected_training``, or ``rejected_validation``."""
    termination_reason: str
    """Search or run stopping reason, such as ``target_reached`` or ``budget_exhausted``."""
    parameters: dict[str, float] | None
    """Best candidate's physical parameter dictionary, or ``None`` before a candidate exists.
    Interpret it within the acceptance scope."""
    asset_path: Path | None
    """Accepted ``release/asset.json`` path, otherwise ``None``."""
    run_directory: Path
    """Directory containing this run's records and results."""
