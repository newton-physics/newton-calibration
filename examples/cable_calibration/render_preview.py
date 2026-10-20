"""Render the README/Pages preview from archived measurements and predictions.

Run from the repository root:
    python -m pip install matplotlib
    python -m examples.cable_calibration.render_preview
No Newton simulation is performed.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE = Path(__file__).resolve().parent


def main():
    metrics = json.loads((CASE / "archive/expected-metrics.json").read_text())
    experiment = json.loads((CASE / "data/experiments.json").read_text())["noplug_release"]
    times = np.asarray(experiment["times"])
    reference = np.asarray(experiment["reference"])
    mask = np.asarray(experiment["mask"], dtype=bool)
    stages = ["A", "B", "Bstar", "C"]
    colors = ["#8c6a50", "#527eb1", "#7b75a3", "#087e83"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#172e40",
            "axes.labelcolor": "#172e40",
        }
    )
    fig, (bars, traces) = plt.subplots(
        1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.5]}
    )
    fig.patch.set_facecolor("#f7f5ef")
    for ax in (bars, traces):
        ax.set_facecolor("#f7f5ef")
        ax.grid(axis="y", color="#dbe0e0", linewidth=0.7)
        ax.set_axisbelow(True)
    values = [metrics[f"{stage}__noplug_release"]["rmse_mm"] for stage in stages]
    bars.bar(["A", "B", "B*", "C"], values, color=colors, width=0.58)
    for i, value in enumerate(values):
        bars.text(i, value + 0.10, f"{value:.3f}", ha="center", weight="bold")
    bars.set_ylim(0, 4.0)
    bars.set_ylabel("Coordinate RMSE (mm)")
    bars.set_title("No-plug dynamic error", loc="left", pad=15, weight="bold")

    valid = mask[:, 29]
    traces.plot(
        times[valid],
        reference[valid, 29] * 1000,
        color="#172e40",
        linewidth=2.5,
        marker=".",
        label="Measured",
    )
    with np.load(CASE / "data/traces.npz", allow_pickle=False) as archive:
        for stage, color, label in zip(stages, colors, ["A", "B", "B*", "C (SLS)"]):
            traces.plot(
                times[valid],
                archive[f"{stage}__noplug_release"][valid, 29] * 1000,
                color=color,
                linewidth=1.7,
                label=label,
            )
    traces.set_title("Vertical motion at 150 mm along the cable", loc="left", pad=15, weight="bold")
    traces.set_xlabel("Time after release (s)")
    traces.set_ylabel("z (mm)")
    traces.legend(frameon=False, ncol=3, loc="lower right", fontsize=9)
    fig.suptitle(
        "CABLE CALIBRATION  /  A → B → C", x=0.065, y=0.965, ha="left", fontsize=17, weight="bold"
    )
    fig.text(
        0.065,
        0.885,
        "Newton VBD · 17 cm free length · 32-segment equivalent parameters",
        fontsize=11,
        color="#50616e",
    )
    fig.text(
        0.065,
        0.105,
        "A  EI + damping     B  + shared clamp angle     B*  damping control     C  SLS",
        fontsize=10,
    )
    fig.text(
        0.065,
        0.055,
        "B* → C: 76.0% lower matched objective. Plugged validation remains unresolved.",
        fontsize=10,
        color="#795234",
    )
    fig.subplots_adjust(left=0.075, right=0.97, top=0.78, bottom=0.24, wspace=0.30)
    output = CASE / "reports/cable_calibration_preview.png"
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
