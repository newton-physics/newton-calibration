# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent

if __name__ == "__main__":
    unittest.main(
        module=None,
        argv=[sys.argv[0], "discover", "-s", str(TESTS), "-t", str(TESTS.parents[1]), *sys.argv[1:]],
    )
