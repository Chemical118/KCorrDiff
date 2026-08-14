"""Value-space contracts for the v1.1.3b K-CorrDiff radar target.

CPrecNet archives store a normalized logarithmic rain-rate value rather than
``mm h-1``.  Accumulation and spatial regridding must happen after converting
that value back to linear rain rate.  This module also owns the single
censoring/transform contract shared by training, residual construction, and
verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p

import numpy as np
from numpy.typing import ArrayLike, NDArray


CPRECNET_RAIN_DB_MIN = -15.0
"""Lower endpoint of the CPrecNet ``10 log10(R)`` normalization."""

CPRECNET_RAIN_DB_SPAN = 45.0
"""Width of the CPrecNet normalized logarithmic rain-rate interval."""

CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR = 10.0 ** (
    (CPRECNET_RAIN_DB_MIN + CPRECNET_RAIN_DB_SPAN) / 10.0
)
"""Largest linear rain rate representable by the CPrecNet transform."""

CPRECNET_RAIN_RATE_MAX_RTOL = 1.0e-6
"""Relative tolerance for float32 rounding at the rain-rate upper bound."""

A_WET_MM = 0.1
"""Official wet threshold in millimetres per 30 minutes."""

A0_MM = 1.0
"""Scale of the official ``log1p(A / A0)`` target transform."""

Z_WET = log1p(A_WET_MM / A0_MM)
"""Lower support of a wet transformed target."""


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CensoredAccumulation:
    """The four coupled target values defined by the data contract.

    ``raw_mm`` remains available for diagnostics.  ``model_mm`` and ``z`` are
    exactly zero at dry pixels, while every wet ``z`` is at least ``Z_WET``.
    Arrays use float64 so threshold decisions do not depend on an implicit
    down-cast performed inside this utility.
    """

    raw_mm: FloatArray
    model_mm: FloatArray
    wet: BoolArray
    z: FloatArray


def _finite_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError(f"{name} must be a real numeric array")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def cprecnet_normalized_to_rain_rate(values: ArrayLike) -> FloatArray:
    """Convert CPrecNet normalized values to linear rain rate in ``mm h-1``.

    The archive contract is intentionally discontinuous at zero: an exact
    stored zero is no echo/below threshold and maps to exactly zero rain rate;
    every positive value is inverted using the documented logarithmic map.
    Values outside ``[0, 1]`` are contract violations rather than candidates
    for silent clipping.
    """

    normalized = _finite_float_array(values, name="CPrecNet normalized values")
    if np.any((normalized < 0.0) | (normalized > 1.0)):
        raise ValueError("CPrecNet normalized values must lie in [0, 1]")

    rain_rate = np.zeros(normalized.shape, dtype=np.float64)
    positive = normalized > 0.0
    rain_db = CPRECNET_RAIN_DB_SPAN * normalized[positive] + CPRECNET_RAIN_DB_MIN
    rain_rate[positive] = np.power(10.0, rain_db / 10.0)
    return rain_rate


def rain_rate_to_cprecnet_normalized(
    rain_rate_mm_per_hour: ArrayLike,
) -> FloatArray:
    """Convert linear rain rate in ``mm h-1`` to CPrecNet normalized values.

    Inverse of :func:`cprecnet_normalized_to_rain_rate` on its image, used to
    re-apply the model-input transform after linear-space context regridding.
    Rates at or below the archive's minimum representable rate collapse to
    exactly zero, mirroring the archive's below-threshold censoring.  Rates
    above the representable maximum are contract violations; a relative
    tolerance absorbs float32 regrid rounding before the final clip.
    """

    rate = _finite_float_array(rain_rate_mm_per_hour, name="rain rate")
    if np.any(rate < 0.0):
        raise ValueError("rain rate cannot be negative")
    maximum_with_tolerance = CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR * (
        1.0 + CPRECNET_RAIN_RATE_MAX_RTOL
    )
    if np.any(rate > maximum_with_tolerance):
        raise ValueError("rain rate exceeds the CPrecNet representable maximum")

    normalized = np.zeros(rate.shape, dtype=np.float64)
    positive = rate > 0.0
    rain_db = 10.0 * np.log10(rate[positive])
    normalized[positive] = (
        rain_db - CPRECNET_RAIN_DB_MIN
    ) / CPRECNET_RAIN_DB_SPAN
    return np.clip(normalized, 0.0, 1.0)


def censor_accumulation(
    raw_accumulation_mm: ArrayLike,
    *,
    a_wet_mm: float = A_WET_MM,
    a0_mm: float = A0_MM,
) -> CensoredAccumulation:
    """Apply the v1.1.3b wet censor and transformed-target mapping.

    Wetness is inclusive at ``a_wet_mm``.  Censoring precedes ``log1p`` so the
    returned values satisfy ``wet == False`` iff ``z == 0`` for valid target
    pixels.
    """

    if not np.isfinite(a_wet_mm) or a_wet_mm <= 0.0:
        raise ValueError("a_wet_mm must be finite and positive")
    if not np.isfinite(a0_mm) or a0_mm <= 0.0:
        raise ValueError("a0_mm must be finite and positive")

    raw_mm = _finite_float_array(raw_accumulation_mm, name="raw accumulation")
    if np.any(raw_mm < 0.0):
        raise ValueError("raw accumulation cannot be negative")

    wet = np.asarray(raw_mm >= a_wet_mm, dtype=np.bool_)
    model_mm = np.where(wet, raw_mm, 0.0)
    z = np.log1p(model_mm / a0_mm)
    return CensoredAccumulation(
        raw_mm=raw_mm,
        model_mm=np.asarray(model_mm, dtype=np.float64),
        wet=wet,
        z=np.asarray(z, dtype=np.float64),
    )


__all__ = [
    "A0_MM",
    "A_WET_MM",
    "CPRECNET_RAIN_DB_MIN",
    "CPRECNET_RAIN_DB_SPAN",
    "CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR",
    "CPRECNET_RAIN_RATE_MAX_RTOL",
    "Z_WET",
    "CensoredAccumulation",
    "censor_accumulation",
    "cprecnet_normalized_to_rain_rate",
    "rain_rate_to_cprecnet_normalized",
]
