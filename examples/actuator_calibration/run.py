# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the actuator calibration through Tuning.run().

Requires Newton, Warp, OpenUSD, pandas and pyarrow; see the example README.
"""

import argparse
from pathlib import Path

from .case import make_tuning

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "so101.json"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m examples.actuator_calibration.run")
    parser.add_argument(
        "--usd",
        type=Path,
        help="local robot USD (sha256-checked); default uses the configured dataset file",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workspace", type=Path, default=Path("workspace/actuator"))
    parser.add_argument(
        "--data-root",
        type=Path,
        help="local copy of the dataset paths; default downloads the pinned revision",
    )
    parser.add_argument("--device", help="Warp device, e.g. cuda:0 or cpu")
    args = parser.parse_args(argv)
    tuning = make_tuning(
        args.config,
        args.usd,
        args.workspace,
        data_root=args.data_root,
        device=args.device,
    )
    result = tuning.run()
    for line in (
        result.acceptance,
        result.termination_reason,
        result.parameters,
        result.run_directory,
        result.asset_path,
    ):
        print(line)
    return 0 if result.acceptance == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
