# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Report legacy Warp array syntax in Python annotations."""

import ast
import sys
from pathlib import Path


def legacy_array_lines(source):
    tree = ast.parse(source)
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
            annotations.append(node.returns)
        if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation:
            annotations.append(node.annotation)

    lines = set()
    for annotation in annotations:
        for node in ast.walk(annotation):
            if isinstance(node, ast.Call):
                target = node.func
                names = {"array", "array1d", "array2d", "array3d", "array4d"}
            elif isinstance(node, ast.Subscript):
                target = node.value
                names = {"array1d"}
            else:
                continue
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "wp"
                and target.attr in names
            ):
                lines.add(node.lineno)
    return sorted(lines)


def main():
    failed = False
    for arg in sys.argv[1:]:
        path = Path(arg)
        for lineno in legacy_array_lines(path.read_text(encoding="utf-8")):
            print(f"{path}:{lineno}: use bracket annotations such as wp.array[wp.float32]")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
