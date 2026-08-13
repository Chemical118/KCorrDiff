"""Seven-scan target construction for K-CorrDiff v1.1.3b."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .radar_values import (
    A0_MM,
    A_WET_MM,
    censor_accumulation,
    cprecnet_normalized_to_rain_rate,
)


TARGET_SCAN_COUNT = 7
SCAN_INTERVAL_HOURS = 5.0 / 60.0
TRAPEZOID_WEIGHTS = np.asarray(
    [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5], dtype=np.float64
)

InputEncoding: TypeAlias = Literal["rain_rate", "cprecnet_normalized"]
Scan: TypeAlias = ArrayLike | None


class MissingTimestampError(ValueError):
    """A required target timestamp is wholly absent and the sample must drop."""


@dataclass(frozen=True, slots=True)
class AccumulationTarget:
    """A target and its loss/metric-only seven-scan validity intersection.

    Invalid pixels are neutral-filled with zero in all value fields.  They are
    not dry observations: consumers must exclude them using ``valid_mask``.
    The future-derived mask must never be forwarded to a model condition or an
    inference API.
    """

    raw_accumulation_mm: NDArray[np.float64]
    model_accumulation_mm: NDArray[np.float64]
    wet: NDArray[np.bool_]
    z: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]


def _scan_sequence(scans: Sequence[Scan] | NDArray[np.generic]) -> tuple[Scan, ...]:
    if isinstance(scans, np.ndarray):
        if scans.ndim == 0:
            raise ValueError("scans must have a leading seven-scan axis")
        return tuple(scans[index] for index in range(scans.shape[0]))
    return tuple(scans)


def _numeric_scan(scan: ArrayLike, *, index: int) -> NDArray[np.float64]:
    raw = np.asarray(scan)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError(f"scan {index} must be a real numeric array")
    return np.asarray(raw, dtype=np.float64)


def _boolean_mask(
    mask: ArrayLike, *, name: str, shape: tuple[int, ...]
) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if array.dtype == np.bool_:
        return np.asarray(array, dtype=np.bool_)
    if not np.issubdtype(array.dtype, np.number) or not np.all(
        (array == 0) | (array == 1)
    ):
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    return np.asarray(array, dtype=np.bool_)


def _validity_stack(
    masks: Sequence[ArrayLike] | NDArray[np.generic] | None,
    *,
    shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    if masks is None:
        return np.ones((TARGET_SCAN_COUNT, *shape), dtype=np.bool_)
    if isinstance(masks, np.ndarray):
        if masks.ndim == 0:
            raise ValueError("pixel_validity must have a leading seven-scan axis")
        sequence = tuple(masks[index] for index in range(masks.shape[0]))
    else:
        sequence = tuple(masks)
    if len(sequence) != TARGET_SCAN_COUNT:
        raise ValueError(
            f"pixel_validity must contain exactly {TARGET_SCAN_COUNT} masks, "
            f"got {len(sequence)}"
        )
    return np.stack(
        [
            _boolean_mask(mask, name=f"pixel_validity[{index}]", shape=shape)
            for index, mask in enumerate(sequence)
        ],
        axis=0,
    )


def build_accumulation_target(
    scans: Sequence[Scan] | NDArray[np.generic],
    *,
    input_encoding: InputEncoding,
    pixel_validity: Sequence[ArrayLike] | NDArray[np.generic] | None = None,
    static_coverage: ArrayLike | None = None,
    a_wet_mm: float = A_WET_MM,
    a0_mm: float = A0_MM,
) -> AccumulationTarget:
    """Build one censored 30-minute target from exactly seven 5-minute scans.

    ``None`` denotes a wholly missing timestamp and raises
    :class:`MissingTimestampError`; callers must drop that sample.  A present
    scan with invalid pixels is instead represented by ``pixel_validity``.
    Invalid pixels are never treated as extra zero-rain integration time and
    the six remaining scans are never reweighted.

    CPrecNet archives have no dynamic per-pixel validity.  For that encoding,
    the contract therefore requires ``static_coverage`` and rejects a dynamic
    validity argument.  Linear, already-QCed input may omit pixel validity to
    declare that every supplied pixel is valid.
    """

    sequence = _scan_sequence(scans)
    if len(sequence) != TARGET_SCAN_COUNT:
        raise ValueError(
            f"scans must contain exactly {TARGET_SCAN_COUNT} timestamps, "
            f"got {len(sequence)}"
        )
    missing = [index for index, scan in enumerate(sequence) if scan is None]
    if missing:
        indices = ", ".join(str(index) for index in missing)
        raise MissingTimestampError(
            f"required target scan timestamp(s) wholly missing at index: {indices}"
        )

    arrays = [
        _numeric_scan(scan, index=index)
        for index, scan in enumerate(sequence)
        if scan is not None
    ]
    shape = arrays[0].shape
    for index, array in enumerate(arrays[1:], start=1):
        if array.shape != shape:
            raise ValueError(f"scan {index} has shape {array.shape}, expected {shape}")

    if input_encoding == "cprecnet_normalized":
        if pixel_validity is not None:
            raise ValueError(
                "CPrecNet has no dynamic pixel validity; use static_coverage only"
            )
        if static_coverage is None:
            raise ValueError(
                "CPrecNet targets require an explicit static_coverage mask"
            )
    elif input_encoding != "rain_rate":
        raise ValueError(f"unsupported input_encoding: {input_encoding!r}")

    validity = _validity_stack(pixel_validity, shape=shape)
    coverage = (
        np.ones(shape, dtype=np.bool_)
        if static_coverage is None
        else _boolean_mask(static_coverage, name="static_coverage", shape=shape)
    )

    raw_stack = np.stack(arrays, axis=0)
    values_to_check = raw_stack[validity]
    if not np.all(np.isfinite(values_to_check)):
        raise ValueError("a valid scan pixel contains a non-finite value")

    if input_encoding == "cprecnet_normalized":
        rain_rate = cprecnet_normalized_to_rain_rate(raw_stack)
    else:
        if np.any(values_to_check < 0.0):
            raise ValueError("a valid linear rain-rate pixel is negative")
        # Invalid raw sentinels/non-finite values have no target meaning.
        rain_rate = np.where(validity, raw_stack, 0.0)

    target_valid = np.asarray(coverage & np.all(validity, axis=0), dtype=np.bool_)
    accumulated = SCAN_INTERVAL_HOURS * np.tensordot(
        TRAPEZOID_WEIGHTS, rain_rate, axes=(0, 0)
    )
    # A partially accumulated invalid pixel must not masquerade as a target.
    accumulated = np.where(target_valid, accumulated, 0.0)
    censored = censor_accumulation(
        accumulated, a_wet_mm=a_wet_mm, a0_mm=a0_mm
    )
    return AccumulationTarget(
        raw_accumulation_mm=censored.raw_mm,
        model_accumulation_mm=censored.model_mm,
        wet=censored.wet,
        z=censored.z,
        valid_mask=target_valid,
    )


__all__ = [
    "SCAN_INTERVAL_HOURS",
    "TARGET_SCAN_COUNT",
    "TRAPEZOID_WEIGHTS",
    "AccumulationTarget",
    "InputEncoding",
    "MissingTimestampError",
    "build_accumulation_target",
]
