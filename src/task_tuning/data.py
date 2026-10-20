# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
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

import torch


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
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
    times: torch.Tensor
    values: torch.Tensor
    positions: torch.Tensor | None = None  # [time, point, xyz], optional visualization


@dataclass
class Episode:
    trial_id: str
    experiment_id: str
    repeat: int
    status: str
    trace: Trace | None = None
    error: str | None = None
    applied_parameters: dict[str, float] = field(default_factory=dict)

    @property
    def key(self):
        return self.trial_id, self.experiment_id, self.repeat


@dataclass
class TrialBatch:
    trial_ids: tuple[str, ...]
    search_values: torch.Tensor


@dataclass
class BatchResult:
    trial_ids: tuple[str, ...]
    search_values: torch.Tensor
    parameter_values: torch.Tensor
    objectives: torch.Tensor
    metrics: torch.Tensor
    constraints: torch.Tensor
    valid_mask: torch.Tensor
    features: torch.Tensor
    episode_records: tuple[Episode, ...]
    identity: dict[str, str]
    purpose: str

    @property
    def feasible(self):
        return self.valid_mask & (self.constraints <= 0).all(dim=1)

    @property
    def feedback(self):
        # Feasible < 1, constraint violation in [1, 2), failed == 3.
        loss = self.objectives.clamp_min(0)
        scores = loss / (1 + loss)
        if self.constraints.shape[1]:
            violation = self.constraints.clamp_min(0).amax(dim=1)
            scores = torch.where(self.feasible, scores, 1 + violation / (1 + violation))
        return torch.where(self.valid_mask, scores, torch.full_like(scores, 3.0))


@dataclass(frozen=True)
class DecisionResult:
    action: str
    reason: str
    batch_size: int = 0


@dataclass
class RunState:
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
        return immutable(plain(self))


@dataclass(frozen=True)
class TuningResult:
    acceptance: str
    termination_reason: str
    parameters: dict[str, float] | None
    asset_path: Path | None
    run_directory: Path
