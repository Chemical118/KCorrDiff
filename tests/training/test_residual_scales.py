from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kcorrdiff.training.residual_scales import (
    ResidualScaleAccumulator,
    scale_lookup,
    write_residual_scales,
)


def test_scales_are_weighted_rms_with_floor_and_provenance(tmp_path: Path) -> None:
    accumulator = ResidualScaleAccumulator(epsilon_scale=0.25)
    accumulator.update(
        lead_hours=0.5,
        condition_signature="era5",
        target_z=np.asarray([1.0, 2.0]),
        mu_z_oof=np.asarray([0.0, 0.0]),
        target_validity=np.asarray([True, True]),
        omega=2.0,
    )
    accumulator.update(
        lead_hours=1.0,
        condition_signature="era5",
        target_z=np.zeros(2),
        mu_z_oof=np.zeros(2),
        target_validity=np.asarray([True, True]),
        omega=1.0,
    )
    path = tmp_path / "scales.json"
    digest = write_residual_scales(
        path,
        accumulator,
        oof_manifest_sha256="a" * 64,
        regression_checkpoint_set_sha256="b" * 64,
    )
    assert len(digest) == 64
    raw = json.loads(path.read_text())
    lookup = scale_lookup(raw)
    assert lookup[(0.5, "era5")] == pytest.approx(np.sqrt(2.5))
    assert lookup[(1.0, "era5")] == 0.25
    assert raw["records"][1]["epsilon_applied"] is True


def test_scale_accumulator_rejects_invalid_or_empty_cells() -> None:
    accumulator = ResidualScaleAccumulator()
    with pytest.raises(ValueError, match="no residual"):
        accumulator.result()
    with pytest.raises(ValueError, match="official 12"):
        accumulator.update(
            lead_hours=0.25,
            condition_signature="era5",
            target_z=[0.0],
            mu_z_oof=[0.0],
            target_validity=[True],
            omega=1.0,
        )


def test_scale_multiplicity_replays_duplicate_training_draw_rows() -> None:
    accumulator = ResidualScaleAccumulator()
    accumulator.update(
        lead_hours=0.5,
        condition_signature="era5",
        target_z=np.asarray([1.0, 3.0]),
        mu_z_oof=np.asarray([0.0, 0.0]),
        target_validity=np.asarray([True, True]),
        omega=2.0,
        multiplicity=4,
    )
    record = accumulator.result()["records"][0]  # type: ignore[index]
    assert record["items"] == 4
    assert record["valid_pixels"] == 8
    assert record["importance_weight_sum"] == 16.0
    assert record["raw_rms"] == pytest.approx(np.sqrt(5.0))
    with pytest.raises(ValueError, match="multiplicity"):
        accumulator.update(
            lead_hours=0.5,
            condition_signature="era5",
            target_z=[1.0],
            mu_z_oof=[0.0],
            target_validity=[True],
            omega=1.0,
            multiplicity=0,
        )
