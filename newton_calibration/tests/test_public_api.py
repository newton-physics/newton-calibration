# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import pickle
import subprocess
import sys
import unittest
from unittest import mock

import newton_calibration as calibration
from newton_calibration.examples import BatchedOscillatorAsset, OscillatorAsset, SlidingAsset, make_tuning
from newton_calibration.tests.unittest_utils import TempPathTestCase


class TestPublicAPI(TempPathTestCase):
    def test_core_exports_and_compatibility_imports(self):
        """Keep core imports stable and expose Episode for custom rollouts."""
        exports = {
            "asset": ("Asset", "BatchEnvironment", "Environment", "SimulationError"),
            "data": ("BatchResult", "DecisionResult", "Episode", "RunState", "Trace", "TrialBatch", "TuningResult"),
            "decision": ("Decision",),
            "evaluator": ("Evaluator",),
            "evidence": ("Evidence", "Experiment"),
            "optimizer": ("CMAES", "GridSearch", "Optimizer"),
            "parameters": ("Parameter", "Parameters"),
            "play": ("Play",),
            "rollout": ("Rollout",),
            "template": ("Template",),
            "tuning": ("Tuning",),
            "validator": ("Validator",),
        }
        expected = {name for names in exports.values() for name in names}
        self.assertCountEqual(calibration.__all__, expected)
        for module, names in exports.items():
            legacy = importlib.import_module(f"newton_calibration.{module}")
            self.assertCountEqual(legacy.__all__, names)
            for name in names:
                with self.subTest(module=module, name=name):
                    self.assertIs(getattr(legacy, name), getattr(calibration, name))
        episode = calibration.Episode("trial", "experiment", 0, "completed")
        self.assertEqual(episode.key, ("trial", "experiment", 0))
        for helper in ("plain", "identity", "immutable", "write_json", "WorkStopped"):
            self.assertFalse(hasattr(calibration, helper))

    def test_import_without_optional_dependencies(self):
        """Import the core and demo factory without loading optional backends."""
        script = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cma', 'scipy', 'warp', 'pxr', 'pyglet'}:
            raise AssertionError(f'Unexpected optional import: {fullname}')

sys.meta_path.insert(0, BlockOptional())
import newton_calibration
from newton_calibration.examples import make_tuning
from newton_calibration.migrate_asset import migrate_asset
for name in newton_calibration.__all__:
    assert getattr(newton_calibration, name) is not None
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_saved_builtin_asset_identities(self):
        """Reload pre-reorganization assets and retain their serialized identities."""
        for asset_type in (SlidingAsset, OscillatorAsset, BatchedOscillatorAsset):
            with self.subTest(asset_type=asset_type.__name__):
                name = f"newton_calibration.examples.{asset_type.__name__}"
                record = {
                    "schema": "newton-calibration-asset/1",
                    "asset_type": name,
                    "parameters": {"value": 2.0},
                    "config": {"label": "saved configuration"},
                }
                path = self.tmp_path / f"{asset_type.__name__}.json"
                path.write_text(json.dumps(record))
                asset = asset_type({"value": 1.0}, config=record["config"])
                loaded = asset.reload(path)
                self.assertEqual(loaded.snapshot()["type"], name)
                self.assertEqual(dict(loaded.baseline_parameters), record["parameters"])
                loaded.export(record["parameters"], path)
                self.assertEqual(json.loads(path.read_text()), record)
                self.assertIs(pickle.loads(pickle.dumps(asset_type)), asset_type)

    def test_external_asset_identity(self):
        """Keep a downstream subclass identified by its own module and class."""

        class CustomAsset(SlidingAsset):
            pass

        asset = CustomAsset({"friction": 0.2})
        self.assertEqual(asset.snapshot()["type"], f"{__name__}.{CustomAsset.__qualname__}")
        path = asset.export({"friction": 0.3}, self.tmp_path / "custom.json")
        self.assertEqual(asset.reload(path).baseline_parameters["friction"], 0.3)

    def test_source_hashes_cover_public_and_nested_implementation(self):
        """Track production changes across directories while excluding test sources."""
        package = self.tmp_path / "package"
        sources = {
            "__init__.py": "# public entry\n",
            "asset.py": "# compatibility entry\n",
            "_src/tuning.py": "# implementation\n",
            "_src/backend/kernel.py": "# nested implementation\n",
            "tests/test_example.py": "# test only\n",
        }
        for name, content in sources.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        implementation = importlib.import_module(calibration.Tuning.__module__)

        def run():
            tuning = make_tuning("sliding", self.tmp_path / "runs", optimizer="grid")
            with mock.patch.object(implementation, "__file__", str(package / "_src/tuning.py")):
                result = tuning.run()
            self.assertEqual(result.acceptance, "accepted")
            return json.loads((result.run_directory / "resolved.json").read_text())

        initial = run()
        expected = set(sources) - {"tests/test_example.py"}
        self.assertEqual(set(initial["source_hashes"]), expected)
        for name in ("__init__.py", "_src/backend/kernel.py"):
            with self.subTest(changed=name):
                (package / name).write_text("# changed production source\n")
                changed = run()
                self.assertNotEqual(changed["identity"]["code"], initial["identity"]["code"])
                self.assertNotEqual(changed["source_hashes"][name], initial["source_hashes"][name])
                (package / name).write_text(sources[name])
        (package / "tests/test_example.py").write_text("# changed test source\n")
        self.assertEqual(run()["identity"]["code"], initial["identity"]["code"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
