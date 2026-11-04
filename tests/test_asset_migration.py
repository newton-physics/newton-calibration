# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from newton_calibration.examples import SlidingAsset
from newton_calibration.migrate_asset import migrate_asset

LEGACY = Path(__file__).parent / "fixtures" / "legacy-sliding-asset.json"


def test_reload_migrated_export(tmp_path):
    """Reload an asset exported by the pre-rename package."""
    output = tmp_path / "converted.json"
    original = LEGACY.read_bytes()
    migrate_asset(LEGACY, output)
    asset = SlidingAsset({"friction": 0.15}).reload(output)
    assert dict(asset.baseline_parameters) == {"friction": 0.35}
    assert LEGACY.read_bytes() == original
    with pytest.raises(FileExistsError):
        migrate_asset(LEGACY, output)


def test_external_asset_and_config_preserved(tmp_path):
    """Preserve third-party identities and opaque configuration."""
    record = json.loads(LEGACY.read_text())
    record.update(asset_type="custom.CableAsset", config={"label": "task_tuning.examples \u03bb \ud800"})
    source, dest = tmp_path / "source.json", tmp_path / "out.json"
    source.write_text(json.dumps(record))
    with pytest.raises(FileExistsError):
        migrate_asset(source, source)
    migrate_asset(source, dest)
    assert dest.read_bytes().isascii()
    assert json.loads(dest.read_text()) == {**record, "schema": "newton-calibration-asset/1"}


@pytest.mark.parametrize("change", [{"schema": "unknown"}, {"asset_type": None}, {"config": []}])
def test_reject_invalid_input(tmp_path, change):
    """Reject incompatible inputs before creating the destination."""
    source, dest = tmp_path / "invalid.json", tmp_path / "out.json"
    source.write_text(json.dumps({**json.loads(LEGACY.read_text()), **change}))
    with pytest.raises(ValueError):
        migrate_asset(source, dest)
    assert not dest.exists()
