# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a standalone HTML/SVG report exclusively from saved run records."""

import html
import json
from pathlib import Path

import numpy as np


def _plot(series):
    colors = ("#707070", "#2166ac", "#d6604d", "#238b45")
    points = [(name, np.asarray(times), np.asarray(values)) for name, times, values in series]
    t0 = min(times.min() for _, times, _ in points)
    t1 = max(times.max() for _, times, _ in points)
    low = min(values.min() for _, _, values in points)
    high = max(values.max() for _, _, values in points)
    span = max(high - low, 1e-9)
    paths = []
    for (_name, times, values), color in zip(points, colors, strict=False):
        xy = zip(50 + 690 * (times - t0) / max(t1 - t0, 1e-9), 180 - 150 * (values - low) / span, strict=True)
        paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="'
            + " ".join(f"{x:.2f},{y:.2f}" for x, y in xy)
            + '"/>'
        )
    legend = " · ".join(
        f'<span style="color:{color}">{html.escape(name)}</span>'
        for (name, _, _), color in zip(points, colors, strict=False)
    )
    return (
        f'<div>{legend}</div><svg viewBox="0 0 780 220" role="img" aria-label="Signal comparison">'
        f'<text x="5" y="25">{high:.4g}</text><text x="5" y="185">{low:.4g}</text>'
        f'<text x="50" y="210">{t0:.3g} s</text><text x="700" y="210">{t1:.3g} s</text>' + "".join(paths) + "</svg>"
    )


def write_report(run_directory):
    """Render comparisons from JSON batches without constructing a simulator."""
    root = Path(run_directory)
    resolved = json.loads((root / "resolved.json").read_text())
    state = json.loads((root / "run_state.json").read_text())
    result = json.loads((root / "result.json").read_text())
    best = state["best_candidate"]
    body = [
        f"<h1>Actuator calibration: {html.escape(result['acceptance'])}</h1>",
        "<p>Equivalent position-PD gains. Offline fit; no hardware qualification is implied.</p>",
        "<p>Solver: " + html.escape(resolved["template"]["numeric_settings"]["solver"]) + " (CPU)</p>",
        "<pre>" + html.escape(json.dumps(result, indent=2)) + "</pre>",
    ]
    if best:
        baseline_ref = next(batch["artifact"] for batch in state["batch_history"] if batch["purpose"] == "baseline")
        baseline = json.loads((root / baseline_ref).read_text())["episode_records"]
        fitted = json.loads((root / best["batch_ref"]).read_text())["episode_records"]
        fitted = {e["experiment_id"]: e for e in fitted if e["trial_id"] == best["trial_id"] and e["repeat"] == 0}
        baseline = {e["experiment_id"]: e for e in baseline if e["repeat"] == 0}
        for batch in state["batch_history"]:
            if batch["purpose"] == "validation" and batch["identity"]["template"] == state["identity"]["template"]:
                episodes = json.loads((root / batch["artifact"]).read_text())["episode_records"]
                fitted.update({e["experiment_id"]: e for e in episodes if e["repeat"] == 0})
        validation_ids = {e["id"] for e in resolved["evidence"]["validation"]}
        for experiment in resolved["evidence"]["train"] + resolved["evidence"]["validation"]:
            key, times = experiment["id"], experiment["times"]
            if key not in fitted:
                continue
            settings = resolved["template"]["experiment_settings"][key]
            before, after = baseline.get(key, {}).get("trace"), fitted[key]["trace"]
            if after is None:
                continue
            split = "validation" if key in validation_ids else "training"
            body.append("<h2>" + html.escape(key) + f" ({split})</h2>")
            for col, channel in enumerate(settings["channels"]):
                reference = np.asarray(experiment["reference"])[:, col]
                calibrated = np.asarray(after["values"])[:, col]
                unit = "rad" if channel == "position" else "rad/s"
                body.append(f"<h3>{channel} [{unit}]</h3>")
                signals = [
                    ("measured/reference", times, reference),
                    ("fitted", times, calibrated),
                ]
                residuals = [("fitted - reference", times, calibrated - reference)]
                if before is not None:
                    initial = np.asarray(before["values"])[:, col]
                    signals.append(("initial", times, initial))
                    residuals.append(("initial - reference", times, initial - reference))
                if channel == "position":
                    # Repeat timestamps to draw a zero-order-held command.
                    ct, command = np.asarray(settings["command_times"]), np.asarray(settings["command"])
                    signals.append(("command", np.repeat(ct, 2)[1:], np.repeat(command, 2)[:-1]))
                body.append(_plot(signals))
                body.append(f"<h3>{channel} residual [{unit}]</h3>")
                body.append(_plot(residuals))
    body.extend(
        [
            "<h2>Validation and numerical scope</h2>",
            "<pre>",
            html.escape(
                json.dumps(
                    {"numerical": state["analysis_reports"].get("numerical"), "validation": state["validation_report"]},
                    indent=2,
                )
            ),
            "</pre>",
        ]
    )
    path = root / "actuator-report.html"
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Actuator calibration</title>'
        "<style>body{font:16px system-ui;max-width:980px;margin:32px auto;padding:0 20px}"
        "svg{width:100%;background:#f7f8fa}pre{white-space:pre-wrap}</style>" + "".join(body) + "</html>",
        encoding="utf-8",
    )
    return path
