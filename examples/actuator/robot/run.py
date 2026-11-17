# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Import or independently inspect a frozen robot; no calibration is performed."""

import argparse
import json
import tempfile
from pathlib import Path

from .asset import AssetRobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", type=Path, help="local MJCF or URDF")
    mode.add_argument("--reload", type=Path, help="previously exported asset.json")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--gravity", type=float, nargs=3, help="explicit acceleration [m/s²]")
    parser.add_argument("--package", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--obj-materials", choices=("reject", "mjcf"), help="explicit OBJ material handling")
    parser.add_argument("--source-provenance", type=Path, help="model source/revision metadata JSON")
    args = parser.parse_args()
    if args.model and args.gravity is None:
        parser.error("--model requires --gravity")
    if args.reload and (args.gravity is not None or args.package or args.obj_materials or args.source_provenance):
        parser.error("--reload uses the frozen configuration; omit import options")
    packages = {}
    for value in args.package:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory or name in packages:
            parser.error("--package requires unique NAME=DIR entries")
        packages[name] = directory
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        if args.model:
            asset = AssetRobot.from_file(
                args.model,
                bundle_dir=Path(temporary) / "bundle",
                gravity=args.gravity,
                package_map=packages,
                obj_materials=args.obj_materials or "reject",
                provenance=json.loads(args.source_provenance.read_text()) if args.source_provenance else None,
            )
        else:
            asset = AssetRobot.from_export(args.reload)
        _, _, bindings = asset.build_model()
        asset.export({}, args.output / "asset.json")
    report = {"status": "imported", "asset_id": asset.asset_id, "bindings": bindings}
    (args.output / "import-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Imported {bindings['nq']} coordinates; report: {args.output / 'import-report.json'}")


if __name__ == "__main__":
    main()
