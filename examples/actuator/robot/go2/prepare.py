# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Download hash-pinned public evidence and derive the suspended Go2 model."""

import argparse
import hashlib
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JOINTS = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fetch_file(root, entry):
    path = root / entry["path"]
    if path.exists():
        if checksum(path) != entry["sha256"]:
            raise ValueError(f"changed download: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(entry["url"], timeout=60) as response:
        raw = response.read(entry["size"] + 1)
    if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"upstream size/hash mismatch: {entry['url']}")
    path.write_bytes(raw)


def read_capture(path, expected_hash):
    """Read only the known pinned tensor-storage layout, without executing pickle."""
    path = Path(path)
    if checksum(path) != expected_hash:
        raise ValueError("capture hash mismatch")
    # These exact archives contain contiguous little-endian float32 arrays.
    # The pinned hash binds the field-to-storage mapping; this is not a generic .pt loader.
    with zipfile.ZipFile(path) as archive:
        prefix = path.stem
        if archive.read(f"{prefix}/byteorder") != b"little":
            raise ValueError("unexpected tensor byte order")
        arrays = {}
        for field, storage, shape in (
            ("position", 1, (4800, 12)),
            ("velocity", 2, (4800, 12)),
            ("command", 3, (4800, 12)),
        ):
            raw = archive.read(f"{prefix}/data/{storage}")
            if len(raw) != 4 * np.prod(shape):
                raise ValueError("unexpected pinned tensor shape")
            array = np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float64)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite observations")
            arrays[field] = array
    return arrays


def convert_records(root, sources):
    records = []
    for capture, split in (("excitation_1", "train"), ("excitation_2", "train"), ("excitation_3", "validation")):
        entry = sources["files"][f"data/datasets/go2/{capture}.pt"]
        arrays = read_capture(root / entry["path"], entry["sha256"])
        times = np.arange(4800, dtype=np.float64) / 200.0
        observed = np.unique(np.r_[np.arange(0, 4800, 4), 4799]) if split == "train" else np.arange(4800)
        record = {
            "id": capture,
            "group": capture,
            "split": split,
            "initial_state": {
                kind: dict(zip(JOINTS, arrays[kind][0].tolist(), strict=True)) for kind in ("position", "velocity")
            },
        }
        for kind, values in arrays.items():
            index = np.arange(4800) if kind == "command" else observed
            record[kind] = {
                name: {
                    "times": times[index].tolist(),
                    "values": values[index, column].tolist(),
                    "unit": "rad/s" if kind == "velocity" else "rad",
                }
                for column, name in enumerate(JOINTS)
            }
        records.append(record)
    return {
        "schema": "robot-recordings/1",
        "provenance": {
            "kind": "measured",
            "source": sources["data_repository"],
            "revision": sources["data_revision"],
            "timestamp_policy": "nominal 200 Hz index clock in float64; actual event timing unavailable",
            "training_observations": "every fourth sample plus final sample; no interpolation or filtering",
            "validation_observations": "complete excitation_3 capture; all samples",
            "control_samples": "all 200 Hz targets retained",
            "joint_order_source": "collector JOINT_NAMES: FL, FR, RL, RR; hip, thigh, calf",
            "scope": "fixed suspended base, assumed level mount; effective PD gains, no hardware qualification",
            "original_files": {key: value for key, value in sources["files"].items() if key.startswith("data/")},
        },
        "records": records,
    }


def derive_model(original, destination):
    shutil.copytree(original, destination)
    path = destination / "go2.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"freejoint", "keyframe"}:
                parent.remove(child)
    root.find("default/default/joint").set("damping", "0")
    for geom in root.iter("geom"):
        if "margin" in geom.attrib:
            geom.set("margin", "0")
    for actuator in root.findall("actuator/motor"):
        actuator.tag = "general"
        actuator.attrib.update(
            gaintype="fixed",
            biastype="affine",
            dyntype="none",
            gainprm="25.0",
            biasprm="0 -25.0 -1.5",
            ctrllimited="false",
            forcelimited="true",
            forcerange="-45.43 45.43" if actuator.get("class") == "knee" else "-23.7 23.7",
        )
    tree.write(path, encoding="unicode")
    return path


def default_config():
    return {
        "controller": {
            "mode": "native_position",
            "command_space": "joint_position",
            "control_dt": 0.005,
            "drives": {name: {"actuator": name.removesuffix("_joint"), "kp": 25.0, "kd": 1.5} for name in JOINTS},
        },
        "parameters": {
            f"{name}.{field}": {"baseline": baseline, "bounds": bounds}
            for name in JOINTS
            for field, baseline, bounds in (("kp", 25.0, [12.5, 50.0]), ("kd", 1.5, [0.25, 4.0]))
        },
        "dt": 0.0005,
        "refinement_dt": 0.00025,
        "precision_tolerance": 0.0002,
        "scales": {kind: dict.fromkeys(JOINTS, scale) for kind, scale in (("position", 0.05), ("velocity", 0.5))},
        "thresholds": {
            kind: {name: {"rmse": rmse, "max": maximum} for name in JOINTS}
            for kind, rmse, maximum in (("position", 0.05, 0.15), ("velocity", 0.5, 1.5))
        },
        "optimizer": {"sigma": 0.02, "seed": 42},
        "execution": {"device": "cuda:0", "worlds": 1024, "batch_size": 64},
        "decision": {
            "max_trials": 4096,
            "max_rollouts": 8692,
            "max_generations": 64,
            "max_seconds": 10800,
            "target": 0.0,
            "patience": None,
        },
    }


def prepare(output):
    from ..asset import AssetRobot  # noqa: PLC0415

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("derived", "imported", "recordings.json")):
        raise FileExistsError("use a new output directory; prepared inputs are immutable")
    sources = json.loads((HERE / "sources.json").read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda entry: fetch_file(output, entry), sources["files"].values()))
    write_json(output / "sources.json", sources)
    recordings = convert_records(output, sources)
    records_hash = hashlib.sha256(
        json.dumps(recordings["records"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if records_hash != sources["records_sha256"]:
        raise ValueError("converted records differ from the pinned historical protocol")
    model = derive_model(output / "model/original", output / "derived")
    recipe = json.loads((HERE / "recipe.json").read_text())
    write_json(output / "derived/recipe.json", recipe)
    asset = AssetRobot.from_file(
        model,
        bundle_dir=output / "imported/resources",
        gravity=[0, 0, -9.81],
        obj_materials="mjcf",
        provenance={"repository": sources["model_repository"], "revision": sources["model_revision"], "recipe": recipe},
    )
    asset.export({}, output / "imported/asset.json")
    write_json(output / "recordings.json", recordings)
    write_json(output / "config.json", default_config())
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
