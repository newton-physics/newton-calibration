# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot resource delivery and CPU import contracts, before parameter fitting."""

import importlib.util
import json
import shutil
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from examples.actuator.robot.asset import AssetRobot
from examples.actuator.robot.bindings import joint_values
from examples.actuator.robot.resources import freeze_resources, verify_resources
from newton_calibration.tests.unittest_utils import TempPathTestCase

FIXTURES = Path(__file__).resolve().parents[2] / "examples/actuator/robot/fixtures"


class TestRobotResources(TempPathTestCase):
    def test_explicit_mjcf_material_policy_preserves_geometry_and_notices(self):
        """Preserve raw OBJ and license while normalizing only external material directives."""
        source = self.tmp_path / "source"
        shutil.copytree(FIXTURES, source)
        mesh = source / "meshes/marker.obj"
        geometry = mesh.read_bytes()
        original = b"mtllib\tmissing.mtl\nusemtl fixture\n" + geometry
        mesh.write_bytes(original)
        (source / "LICENSE").write_text("fixture notice")
        with self.assertRaisesRegex(ValueError, "explicit mjcf"):
            freeze_resources(source / "two_link.xml", self.tmp_path / "rejected")
        bundle = self.tmp_path / "normalized"
        resources = freeze_resources(source / "two_link.xml", bundle, obj_materials="mjcf")
        self.assertEqual(len(resources["mesh_normalization"]), 1)
        for name, record in resources["mesh_normalization"].items():
            self.assertEqual((bundle / name).read_bytes(), geometry)
            self.assertEqual((bundle / record["original"]).read_bytes(), original)
        self.assertEqual((bundle / "notices/LICENSE").read_text(), "fixture notice")
        verify_resources(bundle, resources["files"])
        with self.assertRaisesRegex(ValueError, "only valid for MJCF"):
            freeze_resources(source / "two_link.urdf", self.tmp_path / "invalid-urdf", obj_materials="mjcf")

    def test_include_package_and_original_hashes(self):
        """Freeze includes and package meshes with relative, verified references."""
        for extension in ("xml", "urdf"):
            with self.subTest(extension=extension):
                target = self.tmp_path / extension
                record = freeze_resources(
                    FIXTURES / f"two_link.{extension}", target, package_map={"two_link": FIXTURES}
                )
                verify_resources(target, record["files"])
                text = (target / record["entry"]).read_text()
                self.assertNotIn("<include", text)
                self.assertNotIn("package://", text)
                self.assertNotIn(str(FIXTURES), text)
                self.assertIn("files/", text)
                self.assertEqual(len(record["originals"]), 2 if extension == "xml" else 1)
                self.assertEqual(set(record["names"]["joints"]), {"shoulder", "elbow"})

    def test_missing_modified_extra_and_symlink_resources(self):
        """Reject incomplete, modified or unmanifested resource bundles."""
        root = self.tmp_path / "bundle"
        record = freeze_resources(FIXTURES / "two_link.xml", root)
        member = root / next(name for name in record["files"] if name.startswith("files/"))
        raw = member.read_bytes()
        for operation in ("modified", "missing", "extra", "symlink"):
            with self.subTest(operation=operation):
                extra = root / "extra"
                if operation == "modified":
                    member.write_bytes(b"modified")
                elif operation == "missing":
                    member.unlink()
                elif operation == "extra":
                    extra.write_text("unlisted")
                else:
                    extra.symlink_to(member)
                with self.assertRaises(ValueError):
                    verify_resources(root, record["files"])
                member.write_bytes(raw)
                extra.unlink(missing_ok=True)

    def test_invalid_structures_fail_without_creating_bundle(self):
        """Reject ambiguous names, unsupported joints and broken tree references."""
        source = self.tmp_path / "input.urdf"
        original = (FIXTURES / "two_link.urdf").read_text()
        invalid = [
            original.replace('name="elbow"', 'name="shoulder"'),
            original.replace('type="revolute"', 'type="prismatic"'),
            original.replace('<parent link="base"/>', '<parent link="missing"/>'),
            original.replace('<child link="upper"/>', ""),
            original.replace('<parent link="upper"/>', '<parent link="forearm"/>'),
            original.replace('<child link="forearm"/>', '<child link="upper"/>'),
            original.replace("</robot>", '<transmission name="ignored"/></robot>'),
        ]
        for index, contents in enumerate(invalid):
            with self.subTest(index=index):
                source.write_text(contents)
                target = self.tmp_path / "invalid"
                with self.assertRaises(ValueError):
                    freeze_resources(source, target, package_map={"two_link": FIXTURES})
                self.assertFalse(target.exists())

    def test_unresolved_package_and_recursive_include(self):
        """Fail explicitly on missing inputs and cyclic XML includes."""
        with self.assertRaisesRegex(ValueError, "unresolved package"):
            freeze_resources(FIXTURES / "two_link.urdf", self.tmp_path / "missing")
        source = self.tmp_path / "cycle.xml"
        source.write_text('<mujoco><include file="cycle.xml"/></mujoco>')
        with self.assertRaisesRegex(ValueError, "cyclic"):
            freeze_resources(source, self.tmp_path / "cycle")
        source.write_text('<mujoco><include file="absent.xml"/></mujoco>')
        with self.assertRaisesRegex(ValueError, "missing resource"):
            freeze_resources(source, self.tmp_path / "absent")
        source.write_text(
            '<mujoco><worldbody><body name="base"><freejoint name="floating"/></body></worldbody></mujoco>'
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            freeze_resources(source, self.tmp_path / "floating")


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name)
        for name in ("newton", "warp", "mujoco", "mujoco_warp", "trimesh", "scipy", "PIL")
    ),
    "install the robot extra",
)
class TestRobotImport(TempPathTestCase):
    def make_asset(self, extension, *, source=None, label=None):
        return AssetRobot.from_file(
            source or FIXTURES / f"two_link.{extension}",
            bundle_dir=self.tmp_path / (label or extension),
            gravity=(0.0, 0.0, -9.81),
            package_map={"two_link": FIXTURES},
        )

    def test_implicit_mesh_name_survives_hashed_resource_filename(self):
        """Preserve MuJoCo's filename-derived mesh name when normalizing resources."""
        source = self.tmp_path / "source"
        shutil.copytree(FIXTURES, source)
        include = source / "parts/links.xml"
        include.write_text(include.read_text().replace('mesh name="marker"', "mesh"))
        model, _, report = self.make_asset("xml", source=source / "two_link.xml").build_model()
        self.assertEqual(model.shape_count, 3)
        self.assertEqual(report["nq"], 2)

    def test_obj_texture_coordinates_import_with_robot_dependencies(self):
        """Import UV-bearing OBJ geometry through the declared robot dependencies."""
        source = self.tmp_path / "source"
        shutil.copytree(FIXTURES, source)
        mesh = source / "meshes/marker.obj"
        lines = mesh.read_text().splitlines()
        geometry = [line for line in lines if not line.startswith("f ")]
        faces = [
            "f " + " ".join(f"{index}/{index}" for index in line.split()[1:]) for line in lines if line.startswith("f ")
        ]
        mesh.write_text("\n".join([*geometry, "vt 0 0", "vt 1 0", "vt 0 1", "vt 1 1", *faces]) + "\n")
        model, _, report = self.make_asset("xml", source=source / "two_link.xml").build_model()
        self.assertEqual(model.shape_count, 3)
        self.assertEqual(report["nq"], 2)

    def test_solver_conversion_warnings_fail_closed(self):
        """Reject solver-side model changes that would otherwise only emit a warning."""
        source = self.tmp_path / "source"
        shutil.copytree(FIXTURES, source)
        include = source / "parts/links.xml"
        tree = ET.parse(include)
        geom = tree.getroot().find("worldbody/body/geom")
        geom.attrib.pop("mesh")
        geom.attrib.update(type="box", size="0.1 0.1 0.1", margin="0.001", contype="1", conaffinity="1")
        tree.write(include)
        with self.assertRaisesRegex(ValueError, "solver conversion warning"):
            self.make_asset("xml", source=source / "two_link.xml").build_model()

    def test_physical_properties_match_known_two_link_model(self):
        """Verify masses, full inertia tensors, joint geometry and passive damping."""
        import mujoco  # noqa: PLC0415
        import newton

        for extension in ("xml", "urdf"):
            with self.subTest(extension=extension):
                model, solver, report = self.make_asset(extension).build_model()
                self.assertEqual((report["nq"], report["nv"]), (2, 2))
                self.assertEqual(model.shape_count, 3)
                self.assertIn(newton.GeoType.MESH, model.shape_type.numpy())
                expected = {
                    "base": (1, [0, 0, 0], [0.05, 0.05, 0.05]),
                    "upper": (2, [0.5, 0, 0], [0.02, 0.18, 0.18]),
                    "forearm": (1, [0.3, 0, 0], [0.01, 0.04, 0.04]),
                }
                for name, (mass, com, diagonal) in expected.items():
                    body = report["bodies"][name]
                    self.assertAlmostEqual(body["mass"], mass)
                    np.testing.assert_allclose(body["center_of_mass"], com, atol=1e-7)
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, np.array(body["inertia_orientation_wxyz"]))
                    rotation = rotation.reshape(3, 3)
                    tensor = rotation @ np.diag(body["principal_inertia"]) @ rotation.T
                    np.testing.assert_allclose(tensor, np.diag(diagonal), atol=5e-7, rtol=0)
                for name, damping in (("shoulder", 0.02), ("elbow", 0.015)):
                    joint = report["joints"][name]
                    np.testing.assert_allclose(joint["axis"], [0, 1, 0])
                    np.testing.assert_allclose(joint["limits"], [-2, 2])
                    self.assertAlmostEqual(joint["damping"], damping, places=7)
                    self.assertEqual(joint["armature"], 0)
                np.testing.assert_allclose(solver.mj_model.opt.gravity, [0, 0, -9.81], atol=1e-6)
                if extension == "urdf":
                    self.assertEqual(solver.mj_model.nu, 0)
                    np.testing.assert_array_equal(model.joint_target_kd.numpy(), [0, 0])
                else:
                    for name in ("shoulder", "elbow"):
                        binding = report["actuators"][name + "_drive"]
                        self.assertEqual(binding["joint"], name)
                        self.assertEqual(binding["control_source"], 1)

    def advance(self, asset, *, names=("elbow", "shoulder"), values=(-0.2, 0.3)):
        import newton

        model, solver, report = asset.build_model()
        state, next_state = model.state(), model.state()
        state.joint_q.assign(joint_values(values, names, report))
        state.joint_qd.assign(joint_values([0.1, -0.15], ["shoulder", "elbow"], report, velocity=True))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        elbow = report["bodies"]["forearm"]["newton_body"]
        np.testing.assert_allclose(state.body_q.numpy()[elbow, :3], [np.cos(0.3), 0, -np.sin(0.3)], atol=1e-6)
        control = model.control()
        control.joint_f.assign(joint_values([0.15, -0.04], ["shoulder", "elbow"], report, velocity=True))
        for _ in range(40):
            state.clear_forces()
            solver.step(state, next_state, control, None, 0.00025)
            state, next_state = next_state, state
        return np.array(
            [
                [
                    state.joint_q.numpy()[report["joints"][name]["q_start"]],
                    state.joint_qd.numpy()[report["joints"][name]["qd_start"]],
                ]
                for name in ("shoulder", "elbow")
            ]
        )

    def test_formats_and_column_order_preserve_motion(self):
        """Match nonzero-state forced motion across formats and shuffled columns."""
        mjcf, urdf = self.make_asset("xml"), self.make_asset("urdf")
        expected = self.advance(mjcf)
        np.testing.assert_allclose(self.advance(urdf), expected, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(self.advance(mjcf, names=("shoulder", "elbow"), values=(0.3, -0.2)), expected)
        tree = ET.parse(FIXTURES / "two_link.urdf")
        joints = tree.getroot().findall("joint")
        for joint in joints:
            tree.getroot().remove(joint)
        tree.getroot().extend(reversed(joints))
        source = self.tmp_path / "reordered.urdf"
        tree.write(source)
        reordered = self.make_asset("urdf", source=source, label="reordered")
        np.testing.assert_allclose(self.advance(reordered), expected, atol=1e-6, rtol=0)

    def test_named_vectors_reject_ambiguous_or_missing_values(self):
        """Reject partial, duplicated and nonfinite joint-state inputs."""
        _, _, report = self.make_asset("xml").build_model()
        for names, values in (
            (["shoulder"], [0]),
            (["elbow", "elbow"], [0, 0]),
            (["elbow", "shoulder"], [0, float("nan")]),
            (["elbow", "shoulder"], [[0, 0]]),
        ):
            with self.subTest(names=names, values=values), self.assertRaises(ValueError):
                joint_values(values, names, report)

    def test_actuator_declaration_order_preserves_names_and_targets(self):
        """Resolve native actuator bindings through actual control indices."""
        source = self.tmp_path / "source"
        shutil.copytree(FIXTURES, source)
        tree = ET.parse(source / "two_link.xml")
        section = tree.getroot().find("actuator")
        section[:] = list(reversed(list(section)))
        section[0].set("name", "custom_elbow")
        tree.write(source / "two_link.xml")
        _, _, report = self.make_asset("xml", source=source / "two_link.xml").build_model()
        self.assertEqual(report["actuators"]["custom_elbow"]["joint"], "elbow")
        self.assertEqual(report["actuators"]["custom_elbow"]["control_index"], 0)
        self.assertEqual(report["actuators"]["shoulder_drive"]["control_index"], 1)

    def test_release_reloads_in_new_process_without_source_directory(self):
        """Rebuild a relocated release after removing source and candidate bundles."""
        source = self.tmp_path / "source"
        shutil.copytree(FIXTURES, source)
        asset = self.make_asset("xml", source=source / "two_link.xml")
        _, _, expected = asset.build_model()
        candidate = asset.export({}, self.tmp_path / "candidate/asset.json")
        release = self.tmp_path / "release/asset.json"
        asset.stage_release(candidate, release)
        self.assertEqual(candidate.read_bytes(), release.read_bytes())
        shutil.rmtree(source)
        shutil.rmtree(self.tmp_path / "xml")
        shutil.rmtree(candidate.parent)
        moved = self.tmp_path / "moved"
        shutil.move(release.parent, moved)
        output = self.tmp_path / "reloaded"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "examples.actuator.robot.run",
                "--reload",
                str(moved / "asset.json"),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        actual = json.loads((output / "import-report.json").read_text())
        self.assertEqual(actual["status"], "imported")
        self.assertEqual(actual["bindings"], expected)

    def test_backend_drift_and_resource_changes_prevent_rebuild(self):
        """Reject changed dependencies and mesh content before constructing physics."""
        asset = self.make_asset("xml")
        path = asset.export({}, self.tmp_path / "candidate/asset.json")
        record = json.loads(path.read_text())
        config = record["config"]
        config["versions"]["mujoco"]["version"] = "unexpected"
        changed = AssetRobot({}, config=config, resource_root=self.tmp_path / "xml")
        with self.assertRaisesRegex(ValueError, "version changed"):
            changed.build_model()
        member = next((path.parent / asset.bundle_name / "files").iterdir())
        member.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "checksum"):
            AssetRobot.from_export(path)
