# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convert an exported task_tuning asset to the newton_calibration file format."""

from ._src.migrate_asset import main, migrate_asset

__all__ = ["migrate_asset"]


if __name__ == "__main__":
    main()
