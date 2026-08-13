from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from kcorrdiff.data.static_loader import StaticNormalization, load_target_static


def normalization() -> StaticNormalization:
    return StaticNormalization(
        means={
            "elevation": 1.0,
            "slope_x": 0.0,
            "slope_y": 0.0,
            "local_relief": 2.0,
        },
        standard_deviations={
            "elevation": 2.0,
            "slope_x": 1.0,
            "slope_y": 1.0,
            "local_relief": 2.0,
        },
        fit_split="outer_train",
        fit_manifest_sha256="a" * 64,
    )


def write_static(path: Path) -> None:
    shape = (256, 256)
    np.savez(
        path,
        elevation=np.full(shape, 3.0, np.float32),
        slope_x=np.zeros(shape, np.float32),
        slope_y=np.zeros(shape, np.float32),
        local_relief=np.full(shape, 4.0, np.float32),
        land_mask=np.ones(shape, np.uint8),
        lcc_x_normalized=np.broadcast_to(np.linspace(-1, 1, 256), shape),
        lcc_y_normalized=np.broadcast_to(np.linspace(-1, 1, 256)[:, None], shape),
        coverage=np.ones(shape, np.uint8),
        lcc_x_m=np.arange(256, dtype=np.float64) * 500,
        lcc_y_m=np.arange(256, dtype=np.float64) * 500,
    )


def test_static_loader_uses_declared_outer_train_stats_and_names(tmp_path: Path) -> None:
    path = tmp_path / "static.npz"
    write_static(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = load_target_static(
        path, normalization=normalization(), expected_sha256=digest
    )
    assert result.values.shape == (7, 256, 256)
    assert result.values.dtype == np.float32
    assert result.values[0, 0, 0] == 1.0
    assert result.values[3, 0, 0] == 1.0
    assert result.coverage.all()
    assert result.x_lcc_km[-1] == 127.5


def test_static_loader_rejects_unhashed_or_nontrain_stats(tmp_path: Path) -> None:
    path = tmp_path / "static.npz"
    write_static(path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_target_static(
            path, normalization=normalization(), expected_sha256="bad"
        )
    with pytest.raises(ValueError, match="outer_train"):
        StaticNormalization(
            means=normalization().means,
            standard_deviations=normalization().standard_deviations,
            fit_split="calibration",
            fit_manifest_sha256="a",
        )
