# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure GPU robot throughput on a training-record prefix without optimizing."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .asset import AssetRobot
from .recording import read_file
from .tuning import TuningRobot


def main():
    import warp as wp  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new benchmark JSON")
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=2, help="fresh batches after one discarded warmup")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.worlds < 1 or args.repeats < 1 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("positive worlds, repeats and duration are required")
    if args.output.exists():
        parser.error("output already exists")
    records, sources = read_file(args.recordings)
    config = json.loads(args.config.read_text())
    config["execution"] = {"device": args.device, "worlds": args.worlds, "batch_size": args.worlds}
    tuning = TuningRobot(
        AssetRobot.from_export(args.asset), records, config, workspace=args.output.parent, sources=sources
    )
    tuning.template.freeze()
    exp = tuning.evidence.train_data[0]
    times = exp.times[exp.times <= args.duration]
    if len(times) < 2:
        parser.error("duration must include at least two training observation times")
    # Distinct candidates exercise per-world gains; holdout observations are unused.
    candidates = []
    for i in range(args.worlds):
        candidates.append(
            {
                name: float(
                    np.clip(definition["baseline"] * (0.9 + 0.2 * i / max(args.worlds - 1, 1)), *definition["bounds"])
                )
                for name, definition in config["parameters"].items()
            }
        )
    cpu_asset = tuning.asset.with_device("cpu")
    cpu_template = tuning.template.refined({"numeric_settings": {"device": "cpu"}})
    expected, cpu_costs = [], []
    for values in (candidates[0], candidates[-1]):
        start = time.perf_counter()
        env = cpu_template.instantiate(cpu_asset, values, exp)
        try:
            expected.append(env.run(times, seed=0, deadline=None).values)
        finally:
            env.close()
        cpu_costs.append(time.perf_counter() - start)
    measurements = []
    for repeat in range(args.repeats + 1):
        start = time.perf_counter()
        env = tuning.template.instantiate_batch(tuning.asset, candidates, exp)
        wp.synchronize_device(args.device)
        built = time.perf_counter()
        try:
            readback = env.applied_parameters
            for requested, actual in zip(candidates, readback, strict=True):
                for name, value in requested.items():
                    if not np.isclose(value, actual[name], rtol=1e-6, atol=1e-12):
                        raise RuntimeError("GPU parameter readback mismatch")
            traces = env.run(times, seed=0, deadline=None)
            if any(not hasattr(trace, "values") for trace in traces):
                raise RuntimeError("GPU benchmark member failed")
            errors = [
                float(np.max(np.abs(traces[index].values - reference)))
                for index, reference in zip((0, -1), expected, strict=True)
            ]
            # This is a bridge diagnostic, not a hardware acceptance threshold.
            np.testing.assert_allclose(traces[0].values, expected[0], atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(traces[-1].values, expected[-1], atol=1e-4, rtol=1e-4)
            finished = time.perf_counter()
            measurement = {
                "warmup": repeat == 0,
                "build_seconds": built - start,
                "run_seconds": finished - built,
                "total_seconds": finished - start,
                "candidate_simulated_seconds_per_wall_second": args.worlds * float(times[-1]) / (finished - start),
                "cpu_gpu_max_abs_channel_error": errors,
                "warp_mempool_peak_bytes_process": wp.get_mempool_used_mem_high(args.device),
            }
            print(json.dumps(measurement), flush=True)
            measurements.append(measurement)
        finally:
            env.close()
        del env, traces
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "scope": "training-prefix throughput and CPU/GPU bridge; no search or hardware acceptance",
                "device": args.device,
                "gpu_name": wp.get_device(args.device).name,
                "worlds": args.worlds,
                "duration": float(times[-1]),
                "observations": len(times),
                "dt": config["dt"],
                "control_dt": config["controller"]["control_dt"],
                "cpu_single_candidate_seconds": cpu_costs,
                "measurements": measurements,
            },
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
