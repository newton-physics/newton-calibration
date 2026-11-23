# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a fixed Go2 hyperparameter grid, with one isolated process per search."""

import argparse
import copy
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .prepare import checksum, write_json


def job_config(base, *, batch, sigma, seed, candidates, worlds, device, max_seconds):
    """Change search/execution settings while retaining the physical protocol."""
    if min(batch, candidates, worlds) < 1 or candidates % batch:
        raise ValueError("positive candidate budget must be divisible by population size")
    if not math.isfinite(sigma) or not 0 < sigma < 1 or not 0 <= seed < 2**32:
        raise ValueError("require 0 < sigma < 1 and 0 <= seed < 2**32")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    if device == "cpu" and worlds != 1:
        raise ValueError("CPU execution requires worlds=1")
    config = copy.deepcopy(base)
    config["optimizer"] = {"sigma": sigma, "seed": seed}
    config["execution"] = {"device": device, "worlds": worlds, "batch_size": batch}
    config["decision"] = {
        "max_trials": candidates,
        "max_generations": candidates // batch,
        "max_rollouts": 2 * candidates + 500,
        "max_seconds": max_seconds,
        "target": 0.0,
        "patience": None,
    }
    return config


def run_process(command, log, timeout):
    """Bound an owned child process, including its descendants on POSIX systems."""
    with Path(log).open("x") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name == "posix"
        )
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            raise


def replay_difference(summary, replay):
    """Compare every capture's squared normalized RMSE with its original retest."""
    expected = {
        item["experiment"]: item["objective"]
        for report in (summary["training_retest"], summary["validation"])
        for item in report["experiments"]
    }
    if set(expected) != set(replay["experiments"]):
        raise ValueError("replay experiments differ from the accepted run")
    differences = [
        abs(replay["experiments"][name]["metrics"][0] ** 2 - objective) for name, objective in expected.items()
    ]
    if not differences or not all(math.isfinite(value) for value in differences):
        raise ValueError("replay contains missing or nonfinite scores")
    return max(differences)


def worker(prepared, job):
    from ..asset import AssetRobot  # noqa: PLC0415
    from ..recording import read_file  # noqa: PLC0415
    from ..report import write_report  # noqa: PLC0415
    from ..tuning import TuningRobot  # noqa: PLC0415

    config = json.loads((job / "config.json").read_text())
    records, sources = read_file(prepared / "recordings.json")
    if not any(record["split"] == "validation" for record in records):
        raise ValueError("the study requires an independent validation recording")
    tuning = TuningRobot(
        AssetRobot.from_export(prepared / "imported/asset.json"),
        records,
        config,
        workspace=job / "runs",
        sources=sources,
    )
    result = tuning.run()
    root = result.run_directory
    state = json.loads((root / "run_state.json").read_text())
    history = [item for item in state["batch_history"] if item["purpose"] == "train"]
    summary = {
        "acceptance": result.acceptance,
        "termination_reason": result.termination_reason,
        "run_directory": str(root.relative_to(job)),
        "elapsed_s": state["elapsed_s"],
        "best_candidate": state["best_candidate"],
        "generations": len(history),
        "search_candidates": sum(len(item["trial_ids"]) for item in history),
        "history": [{key: item[key] for key in ("best_objective", "wall_s")} for item in history],
        "training_retest": state["analysis_reports"].get("training_retest"),
        "validation": state["validation_report"],
        "errors": state["errors"],
    }
    # Persist the calibration outcome even if report rendering or a later replay fails.
    write_json(job / "summary.json", summary)
    if result.asset_path is not None:
        shutil.copyfile(root / "resolved.json", result.asset_path.parent / "resolved.json")
    write_report(root)
    return 1 if result.acceptance == "run_failed" else 0


def study(args):
    prepared = args.prepared.resolve()
    base = json.loads((prepared / "config.json").read_text())
    jobs = []
    for batch, sigma, seed in itertools.product(args.batch_sizes, args.sigmas, args.seeds):
        config = job_config(
            base,
            batch=batch,
            sigma=sigma,
            seed=seed,
            candidates=args.candidates,
            worlds=args.worlds,
            device=args.device,
            max_seconds=args.max_seconds,
        )
        name = f"b{batch}-s{sigma:g}-seed{seed}"
        jobs.append((name, config))
    if len({name for name, _ in jobs}) != len(jobs):
        raise ValueError("duplicate study configuration")
    identity = {name: checksum(prepared / name) for name in ("config.json", "recordings.json", "imported/asset.json")}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "study.json",
        {
            "inputs_sha256": identity,
            "jobs": [{"id": name, "config": config} for name, config in jobs],
            "selection": "fixed grid; each search uses training only; validation never selects a later job",
        },
    )
    rows = []
    for name, config in jobs:
        job = output / name
        job.mkdir()
        write_json(job / "config.json", config)
        print(f"Starting {name}", flush=True)
        began = time.monotonic()
        row = {"id": name}
        try:
            # Changes to prepared inputs between jobs must fail, not contaminate a comparison.
            if any(checksum(prepared / path) != digest for path, digest in identity.items()):
                raise ValueError("prepared inputs changed during the study")
            command = [sys.executable, "-m", __spec__.name, "--prepared", str(prepared), "--worker", str(job)]
            row["exit_code"] = run_process(command, job / "worker.log", args.max_seconds + 1200)
            summary_file = job / "summary.json"
            if summary_file.exists():
                row.update(json.loads(summary_file.read_text()))
            if row["exit_code"] == 0 and not summary_file.exists():
                raise RuntimeError("worker exited without a calibration summary")
            if row.get("acceptance") == "accepted" and row["exit_code"] == 0:
                release = job / row["run_directory"] / "release"
                replay = job / "replay.json"
                command = [
                    sys.executable,
                    "-m",
                    "examples.actuator.robot.calibrate",
                    "--replay",
                    str(release),
                    "--output",
                    str(replay),
                ]
                row["replay_exit_code"] = run_process(command, job / "replay.log", 1200)
                row["replay_passed"] = False
                if row["replay_exit_code"] == 0:
                    difference = replay_difference(row, json.loads(replay.read_text()))
                    row["replay_objective_difference"] = difference
                    row["replay_tolerance"] = 1e-6
                    row["replay_passed"] = difference <= row["replay_tolerance"]
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["wall_s"] = time.monotonic() - began
        rows.append(row)
        write_json(output / "summary.json", rows)
        print(f"Finished {name}: {row.get('acceptance', row.get('error', 'worker failed'))}", flush=True)
    return int(any(row.get("error") or row.get("exit_code") != 0 or not row.get("replay_passed", True) for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="new study directory (existing output is never overwritten)")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-seconds", type=float, default=10800, help="per-search time budget; watchdog adds 1200 s for finalization"
    )
    args = parser.parse_args()
    return worker(args.prepared.resolve(), args.worker.resolve()) if args.worker else study(args)


if __name__ == "__main__":
    raise SystemExit(main())
