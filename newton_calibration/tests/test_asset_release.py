# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Release staging through the shared calibration lifecycle."""

from pathlib import Path

from newton_calibration.tests.test_invariants import SumAsset, make_sum
from newton_calibration.tests.unittest_utils import TempPathTestCase


class ResourceAsset(SumAsset):
    def stage_release(self, candidate_path, release_path):
        super().stage_release(candidate_path, release_path)
        Path(release_path).with_suffix(".resource").write_text("frozen resource")


class FailedRelease(SumAsset):
    def stage_release(self, candidate_path, release_path):
        raise ValueError("resource checksum mismatch")


class TestAssetRelease(TempPathTestCase):
    def test_default_copy(self):
        """Preserve default JSON contents when staging a plain asset."""
        asset = SumAsset({"a": 1.0})
        source = asset.export({"a": 2.0}, self.tmp_path / "candidate.json")
        target = self.tmp_path / "release" / "asset.json"
        asset.stage_release(source, target)
        self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_lifecycle_uses_release_hook(self):
        """Stage dependent files only after an accepted calibration."""
        tuning = make_sum(self.tmp_path, allow_reduction=True)
        tuning.asset = ResourceAsset(tuning.asset.baseline_parameters)
        result = tuning.run()
        self.assertEqual(result.acceptance, "accepted", tuning.state.errors)
        self.assertEqual(result.asset_path.with_suffix(".resource").read_text(), "frozen resource")

    def test_failed_staging_cannot_be_accepted(self):
        """Reject a release whose resources cannot be staged."""
        tuning = make_sum(self.tmp_path, allow_reduction=True)
        tuning.asset = FailedRelease(tuning.asset.baseline_parameters)
        result = tuning.run()
        self.assertNotEqual(result.acceptance, "accepted")
        self.assertIsNone(result.asset_path)
