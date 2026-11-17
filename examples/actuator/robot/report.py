# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render saved per-joint response and residuals without running physics."""

import html
import json
from pathlib import Path

import numpy as np

from ..report import _plot


def write_report(directory):
    root = Path(directory)
    resolved = json.loads((root / "resolved.json").read_text())
    state = json.loads((root / "run_state.json").read_text())
    result = json.loads((root / "result.json").read_text())
    sources = resolved["evidence"]["sources"]
    controller = sources["protocol"]["controller"]
    native_control = controller["mode"] == "native_position" and controller["command_space"] == "actuator_control"
    provenance = sources.get("provenance", {"kind": "unspecified"})
    best = state["best_candidate"]
    body = [
        "<h1>Robot actuator calibration</h1>",
        "<p>Evidence: " + html.escape(provenance.get("kind", "unspecified")) + "</p>",
        "<p>Acceptance: "
        + html.escape(result["acceptance"])
        + "; controller: "
        + html.escape(controller["mode"])
        + "</p>",
        "<p>Saved trajectories only. Acceptance is limited to the recorded model, protocol and evidence.</p>",
        "<p>" + html.escape(provenance.get("scope", "")) + "</p>",
        "<details><summary>Evidence provenance and limitations</summary><pre>"
        + html.escape(json.dumps(provenance, indent=2))
        + "</pre></details>",
        "<details><summary>Selected parameters and training score</summary><pre>"
        + html.escape(json.dumps(best, indent=2))
        + "</pre></details>",
    ]
    if best and any(
        item["purpose"] == "baseline" and item["artifact"] == best["batch_ref"] for item in state["batch_history"]
    ):
        body.append("<p>The selected candidate is the initial baseline; no improvement is claimed.</p>")
    baseline, candidate = {}, {}
    for item in state["batch_history"]:
        if (
            item["purpose"] == "baseline"
            or (best and item["artifact"] == best["batch_ref"])
            or (item["purpose"] == "validation" and item["identity"]["template"] == state["identity"]["template"])
        ):
            episodes = json.loads((root / item["artifact"]).read_text())["episode_records"]
            for episode in episodes:
                if episode["repeat"] != 0 or episode["trace"] is None:
                    continue
                if item["purpose"] == "baseline":
                    baseline[episode["experiment_id"]] = episode["trace"]
                if best and (episode["trial_id"] == best["trial_id"] or item["purpose"] == "validation"):
                    candidate[episode["experiment_id"]] = episode["trace"]
    for split in ("train", "validation"):
        for exp in resolved["evidence"][split]:
            eid = exp["id"]
            if eid not in candidate:
                continue
            settings = resolved["template"]["experiment_settings"][eid]
            body.append("<h2>" + html.escape(f"{eid} ({split})") + "</h2>")
            for column, (kind, name) in enumerate(settings["channels"]):
                mask = np.asarray(exp["mask"])[:, column]
                times, reference = np.asarray(exp["times"])[mask], np.asarray(exp["reference"])[mask, column]
                after = np.asarray(candidate[eid]["values"])[mask, column]
                signals = [("reference", times, reference), ("selected candidate", times, after)]
                residuals = [("candidate - reference", times, after - reference)]
                if eid in baseline:
                    before = np.asarray(baseline[eid]["values"])[mask, column]
                    signals.append(("baseline", times, before))
                    residuals.append(("baseline - reference", times, before - reference))
                command_name = name
                if native_control:
                    command_name = controller["drives"].get(name, {}).get("actuator")
                if kind == "position" and command_name in settings["command"]:
                    command = settings["command"][command_name]
                    if not native_control:
                        signals.append(
                            (
                                "position target",
                                np.repeat(command["times"], 2)[1:],
                                np.repeat(command["values"], 2)[:-1],
                            )
                        )
                    else:
                        # Native actuator-length controls have different scaling from joint angles.
                        body.append("<p>Native actuator control: " + html.escape(command_name) + "</p>")
                        body.append(
                            _plot(
                                [
                                    (
                                        "native control",
                                        np.repeat(command["times"], 2)[1:],
                                        np.repeat(command["values"], 2)[:-1],
                                    )
                                ]
                            )
                        )
                body.append(
                    "<h3>" + html.escape(f"{name} {kind} [{'rad' if kind == 'position' else 'rad/s'}]") + "</h3>"
                )
                body.extend((_plot(signals), _plot(residuals)))
    body.append(
        "<h2>Numerical and independent validation gates</h2><pre>"
        + html.escape(
            json.dumps(
                {"numerical": state["analysis_reports"].get("numerical"), "validation": state["validation_report"]},
                indent=2,
            )
        )
        + "</pre>"
    )
    target = root / "robot-report.html"
    target.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Robot calibration</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:32px auto}svg{width:100%}pre{white-space:pre-wrap}</style>"
        + "".join(body)
        + "</html>",
        encoding="utf-8",
    )
    return target
