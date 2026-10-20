# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""The common lifecycle. Cases compose members and do not replace run()."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

from .data import BatchResult, RunState, TrialBatch, TuningResult, identity, plain, write_json


class WorkStopped(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Tuning:
    def __init__(
        self,
        *,
        evidence,
        asset,
        evaluator,
        parameters,
        rollout,
        optimizer,
        decision,
        validator,
        template,
        play,
        workspace="workspace",
        analysis_step=0.03,
        rank_tolerance=1e-3,
        allow_rank_reduction=False,
        orthogonalize=True,
    ):
        self.evidence, self.asset, self.evaluator = evidence, asset, evaluator
        self.parameters, self.rollout, self.optimizer = parameters, rollout, optimizer
        self.decision, self.validator, self.template, self.play = (
            decision,
            validator,
            template,
            play,
        )
        self.workspace = Path(workspace).expanduser().resolve()
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.run_directory = self.workspace / run_name
        if not 0 < analysis_step <= 0.25 or not 0 < rank_tolerance < 1:
            raise ValueError("invalid analysis step or rank tolerance")
        self.analysis_step = analysis_step
        self.rank_tolerance = rank_tolerance
        self.allow_rank_reduction = allow_rank_reduction
        self.orthogonalize = orthogonalize
        self.state = RunState()
        self._started = False
        self._clock = None
        self._counter = 0
        self._baseline = None
        self._best = None
        self.comparison = None
        self._parameter_lock = None
        self._replayed_episodes = []

    def _save(self):
        self.state.elapsed_s = time.monotonic() - self._clock
        write_json(self.run_directory / "run_state.json", self.state)

    def _authorize(self, purpose, count, experiments, *, minimum_batch=None):
        self.state.elapsed_s = time.monotonic() - self._clock
        self.state.phase = purpose
        request = {
            "purpose": purpose,
            "count": count,
            "episodes_per_trial": sum(e.repeats for e in experiments if e is not None) or 1,
            "minimum_batch": count if minimum_batch is None else minimum_batch,
            "reserve_rollouts": (
                sum(e.repeats for e in self.evidence.validation_data)
                + (
                    1
                    + (
                        1
                        if self.template.qualification_mode == "repeatability"
                        else len(self.template.refinements)
                    )
                )
                * sum(e.repeats for e in self.evidence.train_data)
                if purpose not in {"validation", "training_retest", "failure"}
                else 0
            ),
        }
        action = self.decision.next_action(self.state.view(), request)
        self.state.decision_history.append({"request": request, **asdict(action)})
        self._save()
        if action.action == "stop":
            raise WorkStopped(action.reason)
        if action.action != "continue" or not 1 <= action.batch_size <= count:
            raise ValueError("Decision returned an invalid work authorization")
        return action.batch_size

    def _check_locked(self):
        if self.template.template_id != self.state.identity["template"]:
            raise RuntimeError("template changed during run")
        if self.evidence.fingerprint() != self.state.identity["evidence"]:
            raise RuntimeError("evidence changed during run")
        if self.asset.asset_id != self.state.identity["asset"]:
            raise RuntimeError("baseline asset changed during run")
        if self.evaluator.evaluation_id != self.state.identity["evaluation"]:
            raise RuntimeError("evaluation protocol changed during run")
        if self._parameter_lock and self.parameters.parameter_space_id != self._parameter_lock:
            raise RuntimeError("parameter space changed during run")

    def evaluate_batch(self, trials, experiments, purpose, *, asset=None, template=None):
        self._authorize(purpose, len(trials.trial_ids), experiments)
        return self._execute(trials, experiments, purpose, asset=asset, template=template)

    def _execute(self, trials, experiments, purpose, *, asset=None, template=None):
        self._check_locked()
        asset, template = asset or self.asset, template or self.template
        if (
            len(trials.trial_ids) == 0
            or len(set(trials.trial_ids)) != len(trials.trial_ids)
            or trials.search_values.ndim != 2
            or len(trials.search_values) != len(trials.trial_ids)
            or not torch.isfinite(trials.search_values).all()
            or ((trials.search_values < 0) | (trials.search_values > 1)).any()
        ):
            raise ValueError("invalid trial batch")
        values = self.parameters.decode(trials.search_values)
        within_bounds = self.parameters.check_bounds(values)
        requests = [
            (trial_id, self.parameters.as_dict(value) if valid else None)
            for trial_id, value, valid in zip(trials.trial_ids, values, within_bounds, strict=True)
        ]
        usage = self.state.budget_usage.setdefault(purpose, {"trials": 0, "rollouts": 0})
        usage["trials"] += len(requests)
        # Account all scheduled requests, even rejected or failed ones.
        usage["rollouts"] += len(requests) * sum(e.repeats for e in experiments)
        began = time.monotonic()
        episodes = self.rollout.execute_batch(
            template, asset, requests, experiments, deadline=self._clock + self.decision.max_seconds
        )
        scores = self.evaluator.score_batch(episodes, trials.trial_ids, experiments)
        result = BatchResult(
            trials.trial_ids,
            trials.search_values.clone(),
            values.clone(),
            *scores,
            episodes,
            {
                **self.state.identity,
                "template": template.template_id,
                "parameter_space": self.parameters.parameter_space_id,
                "asset": asset.asset_id,
            },
            purpose,
        )
        batch_id = f"batch-{self._counter:05d}"
        self._counter += 1
        artifact = self.run_directory / "batches" / f"{batch_id}.pt"
        artifact.parent.mkdir(exist_ok=True)
        temporary = artifact.with_suffix(".tmp")
        torch.save(asdict(result), temporary)
        temporary.replace(artifact)
        if purpose in {"train", "baseline"}:
            for i in range(len(result.trial_ids)):
                if result.feasible[i] and (
                    self.state.best_candidate is None
                    or float(result.objectives[i]) < self.state.best_candidate["objective"]
                ):
                    self.state.best_candidate = {
                        "trial_id": result.trial_ids[i],
                        "parameters": self.parameters.as_dict(values[i]),
                        "objective": float(result.objectives[i]),
                        "max_error": float(result.metrics[i, 1]),
                        "batch_ref": str(artifact.relative_to(self.run_directory)),
                    }
                    self._best = (result, i)
        self.state.batch_history.append(
            {
                "batch_id": batch_id,
                "purpose": purpose,
                "trial_ids": result.trial_ids,
                "objectives": result.objectives.clone(),
                "valid_mask": result.valid_mask.clone(),
                "identity": dict(result.identity),
                "artifact": str(artifact.relative_to(self.run_directory)),
                "wall_s": time.monotonic() - began,
                "best_objective": (
                    self.state.best_candidate["objective"] if self.state.best_candidate else None
                ),
            }
        )
        self._save()
        infrastructure = [e.error for e in episodes if e.status == "execution_error"]
        if infrastructure:
            raise RuntimeError("; ".join(infrastructure))
        return result

    def _physical_trials(self, values, prefix):
        rows = torch.atleast_2d(torch.as_tensor(values, dtype=torch.float64))
        return TrialBatch(
            tuple(f"{prefix}-{i}" for i in range(len(rows))), self.parameters.encode(rows)
        )

    def prepare(self):
        if self._started:
            raise RuntimeError("each Tuning object runs once; construct another for a new run")
        self.workspace.mkdir(parents=True, exist_ok=True)
        # Reserve a new directory atomically; never reuse an existing run.
        self.run_directory.mkdir()
        self._started = True
        self._clock = time.monotonic()
        self.evidence.prepare()
        self.asset = self.asset.load_or_create()
        if dict(self.asset.baseline_parameters) != self.parameters.as_dict(
            self.parameters.baseline
        ):
            raise ValueError("asset baseline and Parameters baseline disagree")
        if set(self.parameters.names) & set(self.template.fixed_parameters):
            raise ValueError("calibrated and fixed parameter sets overlap")
        self.template.resolve()
        for e in self.evidence.train_data + self.evidence.validation_data:
            if e.id not in self.template.experiment_settings:
                raise ValueError(f"template missing experiment {e.id}")
        self.template.freeze()
        package_sources = {
            p.name: identity(p.read_text()) for p in Path(__file__).parent.glob("*.py")
        }
        self.state.identity = {
            "template": self.template.template_id,
            "asset": self.asset.asset_id,
            "evidence": self.evidence.evidence_id,
            "evaluation": self.evaluator.evaluation_id,
            "code": identity(package_sources),
        }
        self.state.run_id = identity(
            {
                **self.state.identity,
                "parameters": self.parameters.snapshot(),
                "decision": self.decision,
                "optimizer": type(self.optimizer).__name__,
                "optimizer_config": vars(self.optimizer),
                "analysis": [
                    self.analysis_step,
                    self.rank_tolerance,
                    self.allow_rank_reduction,
                    self.orthogonalize,
                ],
            }
        )
        self.state.template_snapshot = self.template.snapshot()
        versions = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": str(self.evaluator.device),
        }
        for name in ("numpy", "cma", "pyglet"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        write_json(
            self.run_directory / "resolved.json",
            {
                "schema": "task-tuning/0.2",
                "workspace": str(self.workspace),
                "run_directory": str(self.run_directory),
                "identity": self.state.identity,
                "template": self.template.snapshot(),
                "parameters": self.parameters.snapshot(),
                "evidence": {
                    "train": self.evidence.train_data,
                    "validation": self.evidence.validation_data,
                    "sources": self.evidence.sources,
                },
                "asset": self.asset.snapshot(),
                "evaluator": self.evaluator.metric_spec,
                "versions": versions,
                "source_hashes": package_sources,
                "decision": self.decision,
            },
        )
        baseline = self._physical_trials(self.parameters.baseline, "baseline")
        self._baseline = self.evaluate_batch(baseline, self.evidence.train_data, "baseline")
        if not self._baseline.valid_mask.all():
            raise RuntimeError("baseline did not produce complete finite observations")
        if self.template.qualification_mode == "refinement" and not self.template.refinements:
            self.state.analysis_reports["numerical"] = {
                "passed": False,
                "reason": "declare at least one numerical refinement",
            }
            raise ValueError("numerical qualification requires explicit refinement settings")
        # Include representative points; qualification is never fed to the optimizer.
        anchors = torch.stack(
            [
                self.parameters.baseline,
                self.parameters.from_normalized(torch.full_like(self.parameters.baseline, 0.25)),
                self.parameters.from_normalized(torch.full_like(self.parameters.baseline, 0.75)),
            ]
        )
        probes = self._physical_trials(anchors, "qualification")
        coarse = self.evaluate_batch(probes, self.evidence.train_data, "qualification")
        reports = []
        changes = (
            (None,)
            if self.template.qualification_mode == "repeatability"
            else self.template.refinements
        )
        for change in changes:
            refined_template = self.template if change is None else self.template.refined(change)
            if (
                change is not None
                and refined_template.numeric_settings == self.template.numeric_settings
                and (refined_template.discretization == self.template.discretization)
            ):
                raise ValueError("refinement must actually change numerical or discrete settings")
            refined = self.evaluate_batch(
                probes, self.evidence.train_data, "qualification", template=refined_template
            )
            good = bool(coarse.valid_mask.all() and refined.valid_mask.all())
            difference = float((coarse.features - refined.features).abs().max()) if good else None
            passed = good and difference <= self.template.precision_tolerance
            reports.append(
                {
                    "template": refined_template.template_id,
                    "feature_difference_max": difference,
                    "passed": passed,
                }
            )
        self.state.analysis_reports["numerical"] = {
            "mode": self.template.qualification_mode,
            "scope": "fixed_configuration_only"
            if self.template.qualification_mode == "repeatability"
            else "declared_refinements",
            "passed": all(x["passed"] for x in reports),
            "probes": reports,
            "tolerance": self.template.precision_tolerance,
        }
        if not self.state.analysis_reports["numerical"]["passed"]:
            raise ValueError(f"{self.template.qualification_mode} difference exceeds tolerance")
        self._save()

    def sensitivity(self):
        self._check_locked()
        p = len(self.parameters.names)
        anchors = [
            self.parameters.to_normalized(self.parameters.baseline),
            torch.full((p,), 0.25, dtype=torch.float64),
            torch.full((p,), 0.75, dtype=torch.float64),
        ]
        jacobians = []
        for a, anchor in enumerate(anchors):
            perturbations = []
            spans = []
            for j in range(p):
                left, right = anchor.clone(), anchor.clone()
                left[j] = max(0, float(anchor[j]) - self.analysis_step)
                right[j] = min(1, float(anchor[j]) + self.analysis_step)
                perturbations.extend([left, right])
                spans.append(float(right[j] - left[j]))
            physical = self.parameters.from_normalized(torch.stack(perturbations))
            result = self.evaluate_batch(
                self._physical_trials(physical, f"sensitivity-{a}"),
                self.evidence.train_data,
                "analysis",
            )
            if not result.valid_mask.all():
                raise ValueError("sensitivity probes failed; do not treat them as zero sensitivity")
            columns = [
                (result.features[2 * j + 1] - result.features[2 * j]) / spans[j] for j in range(p)
            ]
            jacobians.append(torch.stack(columns, dim=1).cpu())
        stacked = torch.cat(jacobians) / len(anchors) ** 0.5
        report = {
            "anchors": torch.stack(anchors),
            "jacobian": stacked,
            "per_anchor_jacobians": jacobians,
            "column_norms": stacked.norm(dim=0),
            "step": self.analysis_step,
            "scope": "local finite differences on training data",
        }
        self.state.analysis_reports["sensitivity"] = plain(report)
        self._save()
        return report

    def identifiability(self, sensitivity_report):
        j = sensitivity_report["jacobian"]
        _, singular_values, vh = torch.linalg.svd(j, full_matrices=False)
        threshold = max(float(singular_values[0]) * self.rank_tolerance, 1e-10)
        rank = int((singular_values > threshold).sum())
        p = len(self.parameters.names)
        conclusive = rank == p or (rank > 0 and self.allow_rank_reduction)
        basis = vh[:rank].T
        transformed = j @ basis
        norms = transformed.norm(dim=0)
        correlation = (transformed.T @ transformed) / (norms[:, None] * norms[None, :]).clamp_min(
            1e-30
        )
        per_anchor = []
        for local in sensitivity_report["per_anchor_jacobians"]:
            effect = local @ basis
            n = effect.norm(dim=0)
            c = effect.T @ effect / (n[:, None] * n[None, :]).clamp_min(1e-30)
            per_anchor.append(c)
        report = {
            "rank": rank,
            "raw_dimension": p,
            "conclusive": conclusive,
            "singular_values": singular_values,
            "threshold": threshold,
            "combined_response_correlation": correlation,
            "per_anchor_response_correlation": per_anchor,
            "rank_reduction_authorized": self.allow_rank_reduction,
            "fixed_unidentified_directions": p - rank,
            "scope": "local; not a proof of global uniqueness",
        }
        if conclusive and self.orthogonalize:
            self.parameters.set_mapping(
                self.parameters.to_normalized(self.parameters.baseline), basis
            )
        elif conclusive and rank < p:
            raise ValueError("rank reduction requires combination mapping")
        self.parameters.freeze()
        self._parameter_lock = self.parameters.parameter_space_id
        report["mapping"] = self.parameters.snapshot()
        self.state.identity["parameter_space"] = self._parameter_lock
        self.state.analysis_reports["identifiability"] = plain(report)
        self._save()
        return report

    def _failure(self, error):
        self.state.fatal_error = f"{type(error).__name__}: {error}"
        self.state.errors.append(self.state.fatal_error)
        if self._clock is None:
            self._clock = time.monotonic()
        try:
            self._authorize("failure", 1, self.evidence.train_data or (None,))
        except WorkStopped as stop:
            return stop.reason
        raise RuntimeError("Decision must stop when fatal_error is set")

    def _finish(self):
        candidate = self.state.best_candidate
        if self.state.analysis_reports.get("numerical", {}).get("passed") is False:
            return "rejected_numerics", None
        if self.state.fatal_error:
            return "run_failed", None
        if candidate is None:
            return "no_feasible_candidate", None
        analysis = self.state.analysis_reports.get("identifiability", {})
        if not analysis.get("conclusive", False):
            return "inconclusive", None
        if (
            candidate["objective"] > self.validator.max_objective
            or candidate["max_error"] > self.validator.max_error
        ):
            return "rejected_training", None
        candidate_path = self.asset.export(
            candidate["parameters"], self.run_directory / "candidate_asset.json"
        )
        self.state.artifact_refs["candidate_asset"] = str(candidate_path)
        loaded = self.asset.reload(candidate_path)
        if dict(loaded.baseline_parameters) != candidate["parameters"]:
            raise ValueError("exported/reloaded asset parameters differ from selected candidate")
        trials = self._physical_trials(
            torch.tensor(
                [loaded.baseline_parameters[k] for k in self.parameters.names], dtype=torch.float64
            ),
            "validation",
        )

        def evaluate(reloaded, experiments):
            if reloaded is not loaded:
                raise ValueError("validator cannot substitute an unfrozen candidate")
            return self.evaluate_batch(trials, experiments, "validation", asset=loaded)

        def evaluate_training(reloaded, experiments):
            if reloaded is not loaded:
                raise ValueError("training retest must use the reloaded candidate")
            replay = self.evaluate_batch(trials, experiments, "training_retest", asset=loaded)
            self._replayed_episodes.extend(replay.episode_records)
            return replay

        try:
            training = self.validator.validate(loaded, self.evidence.train_data, evaluate_training)
            self.state.analysis_reports["training_retest"] = training
            if not training["passed"]:
                return "rejected_training", None
            report = self.validator.validate(loaded, self.evidence.validation_data, evaluate)
            # Retest under the explicitly selected qualification policy.
            final_checks = []
            changes = (
                (None,)
                if self.template.qualification_mode == "repeatability"
                else self.template.refinements
            )
            for change in changes:
                finer = self.evaluate_batch(
                    trials,
                    self.evidence.train_data,
                    "validation",
                    asset=loaded,
                    template=self.template if change is None else self.template.refined(change),
                )
                best_result, best_index = self._best
                error = float((finer.features[0] - best_result.features[best_index]).abs().max())
                final_checks.append(
                    {
                        "passed": bool(finer.valid_mask.all())
                        and error <= self.template.precision_tolerance,
                        "feature_difference_max": error,
                    }
                )
            report["numerical_retest"] = final_checks
            report["qualification_mode"] = self.template.qualification_mode
            report["passed"] = report["passed"] and all(c["passed"] for c in final_checks)
            self.state.validation_report = report
            acceptance = (
                "accepted"
                if report["passed"]
                else "unvalidated"
                if report["status"] == "unvalidated"
                else "rejected_numerics"
                if not all(c["passed"] for c in final_checks)
                else "rejected_validation"
            )
        except WorkStopped as stop:
            self.state.validation_report = {
                "passed": False,
                "status": "incomplete",
                "reason": stop.reason,
            }
            acceptance = "unvalidated"
        published = None
        if acceptance == "accepted":
            published = self.run_directory / "release" / "asset.json"
            published.parent.mkdir(exist_ok=True)
            shutil.copyfile(candidate_path, published)
            write_json(published.parent / "template.json", self.template.snapshot())
            write_json(
                published.parent / "manifest.json",
                {
                    "asset": "asset.json",
                    "template": "template.json",
                    "identity": self.state.identity,
                    "parameters": candidate["parameters"],
                    "analysis": analysis,
                    "validation": self.state.validation_report,
                },
            )
            self.state.artifact_refs["released_asset"] = str(published)
        return acceptance, published

    def run(self):
        if self._started:
            raise RuntimeError("Tuning objects cannot be run twice")
        acceptance, published = "run_failed", None
        try:
            try:
                self.prepare()
                self.identifiability(self.sensitivity())
                self.optimizer.initialize(self.parameters)
                while True:
                    self.state.optimizer_snapshot.update(
                        {
                            "state": self.optimizer.state_dict(),
                            "diagnostics": self.optimizer.diagnostics(),
                        }
                    )
                    count = self._authorize(
                        "train",
                        self.optimizer.batch_size,
                        self.evidence.train_data,
                        minimum_batch=self.optimizer.batch_size if self.optimizer.full_batch else 1,
                    )
                    trials = self.optimizer.ask(count)
                    result = self._execute(trials, self.evidence.train_data, "train")
                    self.optimizer.tell(result)
                    self.state.optimizer_snapshot = {
                        "state": self.optimizer.state_dict(),
                        "diagnostics": self.optimizer.diagnostics(),
                    }
                    if hasattr(self.optimizer, "checkpoint"):
                        checkpoint = self.run_directory / "checkpoints" / f"{self._counter:05d}.pkl"
                        checkpoint.parent.mkdir(exist_ok=True)
                        temporary = checkpoint.with_suffix(".tmp")
                        temporary.write_bytes(self.optimizer.checkpoint())
                        temporary.replace(checkpoint)
                        self.state.optimizer_snapshot["checkpoint"] = str(
                            checkpoint.relative_to(self.run_directory)
                        )
                    self._save()
            except WorkStopped as stop:
                self.state.termination_reason = stop.reason
            except Exception as error:
                # Do not modify existing files after workspace preparation fails.
                if not self._started:
                    raise
                self.state.termination_reason = self._failure(error)
            try:
                acceptance, published = self._finish()
            except Exception as error:
                self._failure(error)
                acceptance, published = "run_failed", None
            if self._baseline is not None and self._best is not None:
                try:
                    best, index = self._best
                    calibrated = tuple(self._replayed_episodes) or tuple(
                        e for e in best.episode_records if e.trial_id == best.trial_ids[index]
                    )
                    self.comparison = self.play.compare(
                        self._baseline.episode_records,
                        calibrated,
                        self.evidence.train_data,
                        accepted=acceptance == "accepted",
                    )
                    self.comparison["identity"] = plain(self.state.identity)
                    self.comparison["baseline_parameters"] = dict(self.asset.baseline_parameters)
                    self.comparison["calibrated_parameters"] = self.state.best_candidate[
                        "parameters"
                    ]
                    self.play.export_usd(self.comparison, self.run_directory / "comparison.usda")
                    self.state.artifact_refs["comparison_usd"] = "comparison.usda"
                except Exception as error:
                    self.state.errors.append(f"presentation: {type(error).__name__}: {error}")
            self.state.phase = "completed"
            self._save()
            result = TuningResult(
                acceptance,
                self.state.termination_reason or "unknown",
                self.state.best_candidate["parameters"] if self.state.best_candidate else None,
                published,
                self.run_directory,
            )
            write_json(self.run_directory / "result.json", result)
            return result
        finally:
            self.rollout.close()
