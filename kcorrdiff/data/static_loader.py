"""Load and normalize the immutable DEM/static artifact for model input."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


TARGET_STATIC_CHANNEL_NAMES = (
    "dem_elevation",
    "terrain_slope_x",
    "terrain_slope_y",
    "local_terrain_relief",
    "land_sea_mask",
    "x_lcc_shared",
    "y_lcc_shared",
)
_SOURCE_KEYS = (
    "elevation",
    "slope_x",
    "slope_y",
    "local_relief",
    "land_mask",
    "lcc_x_normalized",
    "lcc_y_normalized",
)
_TRAIN_NORMALIZED = ("elevation", "slope_x", "slope_y", "local_relief")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StaticNormalization:
    means: Mapping[str, float]
    standard_deviations: Mapping[str, float]
    fit_split: str
    fit_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.fit_split != "outer_train" or not self.fit_manifest_sha256:
            raise ValueError("static normalization must be fit on hashed outer_train")
        if set(self.means) != set(_TRAIN_NORMALIZED) or set(
            self.standard_deviations
        ) != set(_TRAIN_NORMALIZED):
            raise ValueError("static normalization schema is incomplete")
        for key in _TRAIN_NORMALIZED:
            mean = float(self.means[key])
            standard = float(self.standard_deviations[key])
            if not np.isfinite(mean) or not np.isfinite(standard) or standard <= 0.0:
                raise ValueError(f"invalid static normalization for {key}")


@dataclass(frozen=True, slots=True)
class TargetStaticArtifact:
    channel_names: tuple[str, ...]
    values: NDArray[np.float32]
    coverage: NDArray[np.bool_]
    x_lcc_km: NDArray[np.float64]
    y_lcc_km: NDArray[np.float64]
    source_sha256: str
    normalization: StaticNormalization


def load_target_static(
    path: Path,
    *,
    normalization: StaticNormalization,
    expected_sha256: str | None = None,
) -> TargetStaticArtifact:
    """Return named model fields; never derive statistics from holdout labels."""

    path = path.resolve()
    digest = _hash_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("target static artifact SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set((*_SOURCE_KEYS, "coverage", "lcc_x_m", "lcc_y_m")) - set(archive.files))
        if missing:
            raise ValueError(f"target static artifact is incomplete: {missing}")
        channels: list[NDArray[np.float32]] = []
        for key in _SOURCE_KEYS:
            values = np.asarray(archive[key], dtype=np.float32)
            if key in _TRAIN_NORMALIZED:
                values = (
                    values - np.float32(normalization.means[key])
                ) / np.float32(normalization.standard_deviations[key])
            channels.append(values)
        coverage = np.asarray(archive["coverage"], dtype=np.bool_)
        x_lcc_km = np.asarray(archive["lcc_x_m"], dtype=np.float64) / 1000.0
        y_lcc_km = np.asarray(archive["lcc_y_m"], dtype=np.float64) / 1000.0
    stacked = np.stack(channels).astype(np.float32, copy=False)
    if stacked.shape != (7, 256, 256) or coverage.shape != (256, 256):
        raise ValueError("target static artifact must describe the 256 x 256 target")
    if not np.all(np.isfinite(stacked)):
        raise ValueError("target static model channels contain a non-finite value")
    if not np.all(np.diff(x_lcc_km) > 0.0) or not np.all(np.diff(y_lcc_km) > 0.0):
        raise ValueError("target LCC axes must be strictly increasing")
    return TargetStaticArtifact(
        channel_names=TARGET_STATIC_CHANNEL_NAMES,
        values=stacked,
        coverage=coverage,
        x_lcc_km=x_lcc_km,
        y_lcc_km=y_lcc_km,
        source_sha256=digest,
        normalization=normalization,
    )


__all__ = [
    "TARGET_STATIC_CHANNEL_NAMES",
    "StaticNormalization",
    "TargetStaticArtifact",
    "load_target_static",
]
