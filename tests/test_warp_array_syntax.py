# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from scripts.check_warp_array_syntax import legacy_array_lines


def test_checks_annotations_without_flagging_array_construction():
    source = (
        "def allocate(values: list[wp.array(dtype=wp.float32)]) -> wp.array1d[float]:\n"
        "    buffer: wp.array2d(dtype=wp.float32) = wp.array(dtype=wp.float32)\n"
        "    current: wp.array[wp.float32] = buffer\n"
        "    label = 'wp.array(dtype=wp.float32)'\n"
        "    return wp.array(dtype=wp.float32)\n"
    )
    assert legacy_array_lines(source) == [1, 2]
