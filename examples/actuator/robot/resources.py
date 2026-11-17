# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Freeze local robot sources and their supported file dependencies."""

import hashlib
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_resources(root, hashes):
    """Reject missing, modified or escaping bundle members before using them."""
    root = Path(root).resolve()
    members = list(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise ValueError("resource bundle must not contain symlinks")
    if {path.relative_to(root).as_posix() for path in members if path.is_file()} != set(hashes):
        raise ValueError("resource bundle has missing or unexpected files")
    for name, expected in hashes.items():
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(root):
            raise ValueError(f"invalid resource path: {name}")
        if not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"resource checksum mismatch: {name}")


def _resolve(value, parent, packages):
    if value.startswith("package://"):
        package, separator, relative = value[10:].partition("/")
        if not separator or package not in packages:
            raise ValueError(f"unresolved package: {package}")
        base = Path(packages[package]).resolve()
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError("resource escapes its package")
    elif "://" in value:
        raise ValueError(f"only local paths and package:// resources are supported: {value}")
    else:
        path = (parent / value).resolve()
    if not path.is_file():
        raise ValueError(f"missing resource: {path}")
    return path


def _names_and_validate(root, kind):
    """Limit import to named, fixed-base open chains with scalar rotary joints."""
    unsupported = {
        "freejoint",
        "mimic",
        "transmission",
        "tendon",
        "equality",
        "flex",
        "flexcomp",
        "composite",
        "replicate",
        "attach",
        "plugin",
    }
    for element in root.iter():
        if element.tag in unsupported:
            raise ValueError(f"unsupported robot structure: {element.tag}")
    names = {}
    tags = {"bodies": "body" if kind == "mjcf" else "link", "joints": "joint"}
    for category, tag in tags.items():
        # MJCF default/joint is a template, not a physical joint.
        scope = [root] if kind == "urdf" else list(root.findall("worldbody"))
        elements = [element for tree in scope for element in tree.iter(tag)]
        values = [element.get("name") for element in elements]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"missing or duplicate {category} names")
        names[category] = values
        for element in elements:
            if tag == "joint":
                allowed = {"hinge"} if kind == "mjcf" else {"revolute", "continuous", "fixed"}
                if element.get("type", "hinge") not in allowed:
                    raise ValueError("only fixed and single-DOF rotary joints are supported")
            if element.get("mocap", "false") not in {"false", "0"}:
                raise ValueError("mocap bodies are not supported")
    if not names["bodies"] or not names["joints"]:
        raise ValueError("robot requires named bodies and joints")
    for default in root.findall(".//default/joint"):
        if default.get("type", "hinge") != "hinge":
            raise ValueError("non-hinge joint defaults are not supported")
    names["actuators"] = []
    names["actuator_joints"] = {}
    names["actuator_types"] = {}
    for section in root.findall("actuator"):
        for actuator in section:
            if actuator.tag not in {"motor", "position", "velocity", "general"} or not actuator.get("joint"):
                raise ValueError("only direct joint motor/position/velocity/general actuators are supported")
            if actuator.get("joint") not in names["joints"]:
                raise ValueError("actuator references an unknown joint")
            names["actuators"].append(actuator.get("name"))
            names["actuator_joints"][actuator.get("name")] = actuator.get("joint")
            names["actuator_types"][actuator.get("name")] = actuator.tag
    if any(not n for n in names["actuators"]) or len(set(names["actuators"])) != len(names["actuators"]):
        raise ValueError("missing or duplicate actuator names")
    if kind == "urdf":
        parents = {}
        for joint in root.findall("joint"):
            parent, child = joint.find("parent"), joint.find("child")
            if parent is None or child is None:
                raise ValueError("URDF joint requires parent and child links")
            parent, child = parent.get("link"), child.get("link")
            if parent not in names["bodies"] or child not in names["bodies"]:
                raise ValueError("URDF joint references an unknown link")
            if child in parents:
                raise ValueError("URDF must be an open tree")
            parents[child] = parent
        if len(set(names["bodies"]) - set(parents)) != 1:
            raise ValueError("URDF must have one root link")
        for name in parents:
            visited = set()
            current = name
            while current in parents:
                if current in visited:
                    raise ValueError("cyclic URDF joints")
                visited.add(current)
                current = parents[current]
    return names


def freeze_resources(source, destination, *, package_map=None, obj_materials="reject"):
    """Copy originals and normalize local references without editing source files.

    Supports MJCF includes, OBJ/STL geometry, and PNG/JPEG textures. The explicit
    mjcf material policy strips OBJ material directives while retaining originals.
    Other file dependency formats fail instead of omitting transitive resources.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if obj_materials not in {"reject", "mjcf"}:
        raise ValueError("unknown OBJ material policy")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    packages = package_map or {}
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        originals, visited, mesh_normalization = {}, set(), {}

        def read_xml(path):
            raw = path.read_bytes()
            if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                raise ValueError("XML entities are not supported")
            name = f"originals/{digest(raw)}{path.suffix.lower()}"
            (bundle / name).parent.mkdir(exist_ok=True)
            (bundle / name).write_bytes(raw)
            originals[name] = digest(raw)
            return ET.fromstring(raw)

        def copy_asset(path, mesh):
            extension = path.suffix.lower()
            allowed = {".obj", ".stl"} if mesh else {".png", ".jpg", ".jpeg"}
            if extension not in allowed:
                raise ValueError(f"unsupported resource format: {extension}")
            raw = path.read_bytes()
            original = None
            if extension == ".obj":
                lines = raw.splitlines(keepends=True)
                external = any(line.split(maxsplit=1)[:1] == [b"mtllib"] for line in lines)
                if external:
                    if obj_materials == "reject":
                        raise ValueError("OBJ external material libraries require the explicit mjcf material policy")
                    original = f"originals/{digest(raw)}{extension}"
                    (bundle / original).write_bytes(raw)
                    originals[original] = digest(raw)
                    raw = b"".join(
                        line for line in lines if line.split(maxsplit=1)[:1] not in ([b"mtllib"], [b"usemtl"])
                    )
            name = f"files/{digest(raw)}{extension}"
            (bundle / name).parent.mkdir(exist_ok=True)
            (bundle / name).write_bytes(raw)
            if original is not None:
                mesh_normalization[name] = {
                    "original": original,
                    "removed": ["mtllib", "usemtl"],
                    "geometry": "unchanged",
                }
            return name

        def expand_mjcf(path):
            if path in visited:
                raise ValueError("duplicate or cyclic MJCF include")
            visited.add(path)
            root = read_xml(path)
            compiler = root.find("compiler")
            attributes = dict(compiler.attrib) if compiler is not None else {}
            assetdir = attributes.get("assetdir")
            meshdir = attributes.get("meshdir", assetdir or ".")
            texturedir = attributes.get("texturedir", assetdir or meshdir)
            if attributes.get("strippath", "false") not in {"false", "0"}:
                raise ValueError("MJCF strippath is not supported")
            if attributes.get("fusestatic", "false") not in {"false", "0"}:
                raise ValueError("MJCF fusestatic is not supported")
            for element in root.iter():
                if element.tag == "include":
                    continue
                if element.get("file"):
                    if element.tag not in {"mesh", "texture"}:
                        raise ValueError(f"unsupported file reference: {element.tag}")
                    directory = meshdir if element.tag == "mesh" else texturedir
                    value = element.get("file")
                    resolved = _resolve(value, path.parent / directory, packages)
                    # MuJoCo derives unnamed asset names from the original file
                    # stem; hashed filenames must not change geom references.
                    if not element.get("name"):
                        element.set("name", resolved.stem)
                    element.set("file", copy_asset(resolved, element.tag == "mesh"))
            if compiler is not None:
                for key in ("assetdir", "meshdir", "texturedir"):
                    compiler.attrib.pop(key, None)
            for parent in list(root.iter()):
                for child in list(parent):
                    if child.tag == "include":
                        included = expand_mjcf(_resolve(child.attrib["file"], path.parent, packages))
                        index = list(parent).index(child)
                        parent.remove(child)
                        for offset, item in enumerate(included):
                            parent.insert(index + offset, item)
            return root

        header = read_xml(source)
        kind = {"mujoco": "mjcf", "robot": "urdf"}.get(header.tag)
        if kind is None:
            raise ValueError("expected a mujoco or robot XML root")
        if kind != "mjcf" and obj_materials != "reject":
            raise ValueError("the mjcf OBJ material policy is only valid for MJCF sources")
        if kind == "mjcf":
            root = expand_mjcf(source)
        else:
            root = read_xml(source)
            for element in root.iter():
                if element.get("filename"):
                    if element.tag not in {"mesh", "texture"}:
                        raise ValueError(f"unsupported file reference: {element.tag}")
                    resolved = _resolve(element.get("filename"), source.parent, packages)
                    element.set("filename", copy_asset(resolved, element.tag == "mesh"))
        names = _names_and_validate(root, kind)
        entry = "model.xml" if kind == "mjcf" else "model.urdf"
        ET.indent(root)
        ET.ElementTree(root).write(bundle / entry, encoding="utf-8", xml_declaration=True)
        for filename in ("LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE"):
            notice = source.parent / filename
            if notice.is_file():
                (bundle / "notices").mkdir(exist_ok=True)
                shutil.copyfile(notice, bundle / "notices" / filename)
        hashes = {
            p.relative_to(bundle).as_posix(): digest(p.read_bytes()) for p in sorted(bundle.rglob("*")) if p.is_file()
        }
        shutil.move(bundle, destination)
    return {
        "format": kind,
        "entry": entry,
        "files": hashes,
        "originals": originals,
        "names": names,
        "obj_materials": obj_materials,
        "mesh_normalization": mesh_normalization,
    }
