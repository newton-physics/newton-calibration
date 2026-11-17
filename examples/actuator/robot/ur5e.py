# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fetch a pinned, licensed UR5e model for robot integration checks."""

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np

REVISION = "370c43965323c0944b433b202aac5058b5f907a9"
SOURCE = "https://github.com/google-deepmind/mujoco_menagerie"


def fetch(output):
    """Download original model resources and BSD-3-Clause notice; never rewrite them."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/universal_robots_ur5e/"
    hashes = {}

    def download(name):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid upstream resource path")
        with urllib.request.urlopen(base + name, timeout=60) as response:
            raw = response.read()
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return raw

    root = ET.fromstring(download("ur5e.xml"))
    download("LICENSE")
    download("README.md")
    for mesh in root.findall("asset/mesh"):
        download("assets/" + mesh.attrib["file"])
    provenance = {
        "repository": SOURCE,
        "revision": REVISION,
        "model": "universal_robots_ur5e/ur5e.xml",
        "license": "BSD-3-Clause",
        "sha256": hashes,
        "scope": "public robot description; no measured recordings",
    }
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return output / "ur5e.xml"


def check(source, output):
    """Compare three six-joint rollouts with direct MuJoCo; do not claim calibration."""
    import mujoco  # noqa: PLC0415

    from .asset import AssetRobot  # noqa: PLC0415
    from .tuning import TuningRobot  # noqa: PLC0415

    source, output = Path(source), Path(output)
    provenance = json.loads((source / "source.json").read_text())
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed upstream model resource: {name}")
    output.mkdir(parents=True, exist_ok=False)
    asset = AssetRobot.from_file(
        source / "ur5e.xml",
        bundle_dir=output / "input",
        gravity=[0, 0, -9.81],
        obj_materials="mjcf",
        provenance=provenance,
    )
    reference = mujoco.MjModel.from_xml_path(str(source / "ur5e.xml"))
    reference.opt.timestep = 0.0005
    reference.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    home = reference.key("home").qpos.copy()
    names = [reference.joint(i).name for i in range(reference.njnt)]
    if reference.njnt != 6 or reference.nu != 6:
        raise ValueError("expected the pinned six-joint UR5e")
    drives = {}
    for i in range(reference.nu):
        joint = reference.joint(int(reference.actuator_trnid[i, 0])).name
        drives[joint] = {
            "actuator": reference.actuator(i).name,
            "kp": float(reference.actuator_gainprm[i, 0]),
            "kd": float(-reference.actuator_biasprm[i, 2]),
        }
    times = np.arange(81) * 0.002
    records = []
    for capture in range(3):
        q0 = home + 0.04 * capture * np.array([1, -1, 1, -1, 1, -1])
        v0 = np.linspace(-0.015, 0.015, 6)
        command = q0 + 0.08 * np.sin(2 * np.pi * times[:, None] * (np.arange(6) + 1) * (0.4 + 0.13 * capture))
        data = mujoco.MjData(reference)
        data.qpos[:] = q0
        data.qvel[:] = v0
        observed = []
        for target in command:
            observed.append(np.r_[data.qpos, data.qvel].copy())
            for i in range(reference.nu):
                data.ctrl[i] = target[int(reference.actuator_trnid[i, 0])]
            for _ in range(4):
                mujoco.mj_step(reference, data)
        observed = np.array(observed)
        record = {
            "id": f"ur5e-{capture}",
            "group": f"synthetic-capture-{capture}",
            "split": "validation" if capture == 2 else "train",
            "initial_state": {
                "position": dict(zip(names, q0.tolist(), strict=True)),
                "velocity": dict(zip(names, v0.tolist(), strict=True)),
            },
        }
        for kind, values, unit in (
            ("command", command, "rad"),
            ("position", observed[:, :6], "rad"),
            ("velocity", observed[:, 6:], "rad/s"),
        ):
            record[kind] = {
                name: {"times": times.tolist(), "values": values[:, i].tolist(), "unit": unit}
                for i, name in enumerate(names)
            }
        records.append(record)
    config = {
        "controller": {
            "mode": "native_position",
            "command_space": "joint_position",
            "control_dt": 0.002,
            "drives": drives,
        },
        "parameters": {
            f"{name}.{field}": {"baseline": drive[field], "bounds": [0.5 * drive[field], 2 * drive[field]]}
            for name, drive in drives.items()
            for field in ("kp", "kd")
        },
        "dt": 0.0005,
        "refinement_dt": 0.00025,
        "precision_tolerance": 0.05,
        "scales": {kind: dict.fromkeys(names, scale) for kind, scale in (("position", 0.01), ("velocity", 0.05))},
        "thresholds": {
            kind: {name: {"rmse": rmse, "max": maximum} for name in names}
            for kind, rmse, maximum in (("position", 0.0002, 0.0002), ("velocity", 0.005, 0.005))
        },
        "decision": {"max_trials": 800, "max_rollouts": 2200, "max_generations": 95, "max_seconds": 3600},
    }
    sources = {
        "provenance": {
            "kind": "synthetic",
            "reference": "direct MuJoCo on original upstream UR5e model",
            "model": provenance,
        }
    }
    tuning = TuningRobot(asset, records, config, workspace=output / "unused-runs", sources=sources)
    tuning.template.freeze()
    summary = []
    for exp in tuning.evidence.train_data + tuning.evidence.validation_data:
        env = tuning.template.instantiate(tuning.asset, tuning.asset.baseline_parameters, exp)
        try:
            trace = env.run(exp.times, seed=0, deadline=None)
        finally:
            env.close()
        error = np.abs(trace.values - exp.reference)
        kinds = [channel[0] for channel in tuning.template.experiment_settings[exp.id]["channels"]]
        position = float(error[:, np.array(kinds) == "position"].max())
        velocity = float(error[:, np.array(kinds) == "velocity"].max())
        summary.append(
            {
                "experiment": exp.id,
                "position_max_rad": position,
                "velocity_max_rad_s": velocity,
                "passed": position <= 0.0002 and velocity <= 0.005,
            }
        )
    tuning.asset.export(dict(tuning.asset.baseline_parameters), output / "delivery/asset.json")
    resolved = {
        "template": tuning.template.snapshot(),
        "evidence": {
            "train": [asdict(exp) for exp in tuning.evidence.train_data],
            "validation": [asdict(exp) for exp in tuning.evidence.validation_data],
            "sources": tuning.evidence.sources,
        },
    }
    (output / "delivery/resolved.json").write_text(
        json.dumps(resolved, default=lambda value: value.tolist(), indent=2) + "\n"
    )
    (output / "recordings.json").write_text(
        json.dumps({"schema": "robot-recordings/1", "provenance": sources["provenance"], "records": records}, indent=2)
        + "\n"
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "scope": "six-joint model integration; no optimization or measured hardware acceptance",
        "passed": all(item["passed"] for item in summary),
        "experiments": summary,
        "source_revision": provenance["revision"],
        "native_actuators": 6,
        "calibration_executed": False,
    }
    (output / "integration-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("UR5e direct-MuJoCo bridge comparison failed")
    return output / "integration-report.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for original model files")
    parser.add_argument("--check", type=Path, help="downloaded source directory for six-joint bridge checks")
    args = parser.parse_args()
    print(check(args.check, args.output) if args.check else fetch(args.output))


if __name__ == "__main__":
    main()
