# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check portable Go2 input preparation and unattended study contracts."""

import copy
import hashlib
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from examples.actuator.robot.go2.prepare import JOINTS, checksum, default_config, derive_model, fetch_file, read_capture
from examples.actuator.robot.go2.study import job_config, replay_difference, run_process
from newton_calibration.tests.unittest_utils import TempPathTestCase


class TestGo2Study(TempPathTestCase):
    def test_grid_preserves_physics_and_full_generations(self):
        """Vary population and sigma without changing the physical or evidence protocol."""
        base = default_config()
        before = copy.deepcopy(base)
        for batch in (64, 512):
            config = job_config(
                base, batch=batch, sigma=0.05, seed=7, candidates=4096, worlds=1024, device="cuda:0", max_seconds=10800
            )
            self.assertEqual(config["decision"]["max_generations"] * batch, 4096)
            self.assertEqual(config["optimizer"], {"sigma": 0.05, "seed": 7})
            self.assertEqual(config["decision"]["target"], 0)
            self.assertIsNone(config["decision"]["patience"])
            for key in base.keys() - {"optimizer", "execution", "decision"}:
                self.assertEqual(config[key], base[key])
        self.assertEqual(base, before)

    def test_invalid_study_settings(self):
        """Reject incomplete generations and invalid search controls before launching workers."""
        settings = {
            "batch": 64,
            "sigma": 0.02,
            "seed": 42,
            "candidates": 2048,
            "worlds": 1024,
            "device": "cuda:0",
            "max_seconds": 10800,
        }
        for override in (
            {"candidates": 100},
            {"batch": 0},
            {"sigma": float("nan")},
            {"seed": -1},
            {"worlds": 0},
            {"device": "cpu"},
            {"max_seconds": 0},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                job_config(default_config(), **(settings | override))

    def test_download_checks_bytes_before_writing(self):
        """Accept pinned bytes and reject changed cached or upstream content."""
        source = self.tmp_path / "source"
        source.write_bytes(b"known source")
        entry = {
            "url": source.as_uri(),
            "path": "download/file",
            "size": source.stat().st_size,
            "sha256": checksum(source),
        }
        fetch_file(self.tmp_path, entry)
        fetch_file(self.tmp_path, entry)
        (self.tmp_path / entry["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed download"):
            fetch_file(self.tmp_path, entry)
        entry["path"] = "invalid"
        entry["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "mismatch"):
            fetch_file(self.tmp_path, entry)
        self.assertFalse((self.tmp_path / "invalid").exists())

    def test_tensor_storage_is_hash_bound_without_pickle(self):
        """Read numeric storage only and reject any change to the pinned archive."""
        path = self.tmp_path / "capture.pt"
        values = np.arange(4800 * 12, dtype="<f4").reshape(4800, 12)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("capture/byteorder", "little")
            archive.writestr("capture/data.pkl", b"deliberately invalid pickle")
            for number in (1, 2, 3):
                archive.writestr(f"capture/data/{number}", (values + number).tobytes())
        digest = checksum(path)
        arrays = read_capture(path, digest)
        np.testing.assert_array_equal(arrays["position"], values + 1)
        np.testing.assert_array_equal(arrays["command"], values + 3)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            read_capture(path, hashlib.sha256(b"different").hexdigest())

    def test_model_recipe_preserves_inertia_and_limits(self):
        """Freeze the base and replace motor control while retaining load and torque limits."""
        source = self.tmp_path / "original"
        source.mkdir()
        (source / "go2.xml").write_text("""<mujoco><default><default><joint damping="2" armature=".01"/>
        <geom margin=".001"/></default></default><worldbody><body><freejoint/><inertial mass="1"/>
        </body></worldbody><actuator><motor name="hip" class="hip"/><motor name="knee" class="knee"/>
        </actuator><keyframe/></mujoco>""")
        original_hash = checksum(source / "go2.xml")
        root = ET.parse(derive_model(source, self.tmp_path / "derived")).getroot()
        self.assertIsNone(root.find(".//freejoint"))
        self.assertIsNone(root.find("keyframe"))
        self.assertEqual(root.find(".//inertial").get("mass"), "1")
        self.assertEqual(root.find("default/default/joint").get("damping"), "0")
        self.assertEqual(root.find("default/default/joint").get("armature"), ".01")
        self.assertEqual(
            [a.get("forcerange") for a in root.findall("actuator/general")], ["-23.7 23.7", "-45.43 45.43"]
        )
        self.assertEqual(checksum(source / "go2.xml"), original_hash)
        self.assertEqual(len(JOINTS), 12)

    def test_replay_compares_scores_and_capture_identity(self):
        """Detect altered replay scores even if they still satisfy physical gates."""
        summary = {
            "training_retest": {"experiments": [{"experiment": "train", "objective": 0.04}]},
            "validation": {"experiments": [{"experiment": "validation", "objective": 0.09}]},
        }
        replay = {"experiments": {"train": {"metrics": [0.2]}, "validation": {"metrics": [0.3]}}}
        self.assertLess(replay_difference(summary, replay), 1e-12)
        replay["experiments"]["train"]["metrics"][0] = math.sqrt(0.041)
        self.assertGreater(replay_difference(summary, replay), 1e-6)
        del replay["experiments"]["validation"]
        with self.assertRaisesRegex(ValueError, "experiments differ"):
            replay_difference(summary, replay)

    def test_worker_failure_and_timeout_are_visible(self):
        """Preserve nonzero status and stop a hung child instead of claiming completion."""
        self.assertEqual(
            run_process([sys.executable, "-c", "raise SystemExit(3)"], self.tmp_path / "failed.log", 10), 3
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            run_process([sys.executable, "-c", "import time; time.sleep(30)"], self.tmp_path / "timeout.log", 0.1)
