from __future__ import annotations

import math

import numpy as np
import pytest

from kcorrdiff.data.radar_values import (
    A0_MM,
    A_WET_MM,
    Z_WET,
    censor_accumulation,
    cprecnet_normalized_to_rain_rate,
)


def test_cprecnet_inverse_uses_exact_zero_branch_and_logarithmic_mapping() -> None:
    values = np.asarray([0.0, 1.0 / 3.0, 1.0], dtype=np.float64)

    rain_rate = cprecnet_normalized_to_rain_rate(values)

    np.testing.assert_allclose(rain_rate, [0.0, 1.0, 1000.0], rtol=1e-14)
    assert rain_rate[0] == 0.0


def test_any_positive_cprecnet_value_uses_positive_log_endpoint() -> None:
    smallest_positive = np.nextafter(0.0, 1.0)

    rain_rate = cprecnet_normalized_to_rain_rate([0.0, smallest_positive])

    assert rain_rate[0] == 0.0
    assert rain_rate[1] == pytest.approx(10.0**-1.5)


@pytest.mark.parametrize(
    "bad_values",
    [
        [-1e-12],
        [1.0 + 1e-12],
        [np.nan],
        [np.inf],
        np.asarray([1.0 + 1.0j]),
    ],
)
def test_cprecnet_inverse_rejects_contract_violations(bad_values: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        cprecnet_normalized_to_rain_rate(bad_values)


def test_censor_is_inclusive_and_censoring_precedes_transform() -> None:
    raw = np.asarray([0.0, np.nextafter(A_WET_MM, 0.0), A_WET_MM, 2.0])

    target = censor_accumulation(raw)

    np.testing.assert_array_equal(target.wet, [False, False, True, True])
    np.testing.assert_array_equal(target.model_mm[:2], [0.0, 0.0])
    np.testing.assert_allclose(target.model_mm[2:], [A_WET_MM, 2.0])
    np.testing.assert_allclose(
        target.z,
        [0.0, 0.0, Z_WET, math.log1p(2.0 / A0_MM)],
    )
    np.testing.assert_array_equal(target.wet, target.z > 0.0)


@pytest.mark.parametrize(
    ("raw", "a_wet_mm", "a0_mm"),
    [
        ([-0.01], A_WET_MM, A0_MM),
        ([0.0], 0.0, A0_MM),
        ([0.0], A_WET_MM, 0.0),
        ([np.nan], A_WET_MM, A0_MM),
    ],
)
def test_censor_rejects_invalid_physical_contract(
    raw: list[float], a_wet_mm: float, a0_mm: float
) -> None:
    with pytest.raises(ValueError):
        censor_accumulation(raw, a_wet_mm=a_wet_mm, a0_mm=a0_mm)
