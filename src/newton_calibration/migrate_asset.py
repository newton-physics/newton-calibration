# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert an exported task_tuning asset to the newton_calibration file format."""

import argparse
import json
from pathlib import Path


def migrate_asset(source: Path, destination: Path) -> None:
    """Write a migrated copy of a legacy asset to a new file."""
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != "task-tuning-asset/1":
        raise ValueError("expected a task-tuning-asset/1 export")
    asset_type = record.get("asset_type")
    if not isinstance(asset_type, str) or not asset_type.strip():
        raise ValueError("expected an asset_type name")
    if not all(isinstance(record.get(field), dict) for field in ("config", "parameters")):
        raise ValueError("expected config and parameters mappings")

    if asset_type in {
        "task_tuning.examples.OscillatorAsset",
        "task_tuning.examples.SlidingAsset",
        "task_tuning.examples.BatchedOscillatorAsset",
    }:
        record["asset_type"] = asset_type.replace("task_tuning.", "newton_calibration.", 1)
    record["schema"] = "newton-calibration-asset/1"
    payload = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as output:
        try:
            output.write(payload)
            output.flush()
        except OSError:
            output.close()
            destination.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing legacy asset JSON")
    parser.add_argument("destination", type=Path, help="new output file")
    args = parser.parse_args()
    try:
        migrate_asset(args.source, args.destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
