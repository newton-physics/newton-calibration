# SPDX-FileCopyrightText: Copyright (c) 2026 Lightwheel
# SPDX-License-Identifier: Apache-2.0

"""Small example CLI; custom cases use the public object API."""

import argparse
import json
from pathlib import Path

from .data import plain


def main():
    parser = argparse.ArgumentParser(prog="task-tuning")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run a non-cable synthetic qualification case")
    demo.add_argument("--case", choices=("oscillator", "sliding"), default="oscillator")
    demo.add_argument("--optimizer", choices=("grid", "cma_es"), default="cma_es")
    demo.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="root directory for isolated tuning runs (default: workspace)",
    )
    demo.add_argument("--capacity", type=int, default=4)
    demo.add_argument("--gl", action="store_true")
    demo.add_argument("--headless-gl", action="store_true")
    demo.add_argument("--gl-frames", type=int)
    inspect = commands.add_parser("inspect", help="read a recorded result")
    inspect.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        print((args.run_directory / "result.json").read_text())
        return 0
    from .examples import make_tuning

    tuning = make_tuning(
        args.case, args.workspace, optimizer=args.optimizer, capacity=args.capacity
    )
    result = tuning.run()
    print(json.dumps(plain(result), indent=2, ensure_ascii=False))
    if args.gl or args.headless_gl:
        tuning.play.show_gl(
            tuning.comparison,
            max_frames=args.gl_frames,
            headless=args.headless_gl,
            screenshot=result.run_directory / "comparison.png",
        )
    return 0 if result.acceptance == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
