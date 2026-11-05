# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from pathlib import Path

from newton_calibration.examples import SlidingAsset
from newton_calibration.migrate_asset import migrate_asset
from newton_calibration.tests.unittest_utils import TempPathTestCase

LEGACY = Path(__file__).parent / "fixtures" / "legacy-sliding-asset.json"


class TestAssetMigration(TempPathTestCase):
def test_reload_migrated_export(self):
    """Reload an asset exported by the pre-rename package."""
    output = self.tmp_path / "converted.json"
    original = LEGACY.read_bytes()
    migrate_asset(LEGACY, output)
    asset = SlidingAsset({"friction": 0.15}).reload(output)
    self.assertEqual(dict(asset.baseline_parameters), {"friction": 0.35})
    self.assertEqual(LEGACY.read_bytes(), original)
    with self.assertRaises(FileExistsError):
        migrate_asset(LEGACY, output)

def test_external_asset_and_config_preserved(self):
    """Preserve third-party identities and opaque configuration."""
    record = json.loads(LEGACY.read_text())
    record.update(asset_type="custom.CableAsset", config={"label": "task_tuning.examples λ \ud800"})
    source, dest = self.tmp_path / "source.json", self.tmp_path / "out.json"
    source.write_text(json.dumps(record))
    with self.assertRaises(FileExistsError):
        migrate_asset(source, source)
    migrate_asset(source, dest)
    self.assertTrue(dest.read_bytes().isascii())
    self.assertEqual(json.loads(dest.read_text()), {**record, "schema": "newton-calibration-asset/1"})

def test_reject_invalid_input(self):
    """Reject incompatible inputs before creating the destination."""
    for change in ({"schema": "unknown"}, {"asset_type": None}, {"config": []}):
        with self.subTest(change=change):
            source, dest = self.tmp_path / "invalid.json", self.tmp_path / "out.json"
            source.write_text(json.dumps({**json.loads(LEGACY.read_text()), **change}))
            with self.assertRaises(ValueError):
                migrate_asset(source, dest)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
