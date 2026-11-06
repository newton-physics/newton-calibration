# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point for Newton Calibration."""

from ._src.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
