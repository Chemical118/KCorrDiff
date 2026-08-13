from __future__ import annotations

import numpy as np
import pytest

from kcorrdiff.data.accumulation import (
    MissingTimestampError,
    build_accumulation_target,
)


def test_constant_rain_rate_integrates_to_half_hour_amount() -> None:
    scans = np.full((7, 2, 3), 2.0, dtype=np.float64)

    target = build_accumulation_target(scans, input_encoding="rain_rate")

    np.testing.assert_allclose(target.raw_accumulation_mm, 1.0)
    np.testing.assert_allclose(target.model_accumulation_mm, 1.0)
    np.testing.assert_allclose(target.z, np.log(2.0))
    assert np.all(target.wet)
    assert np.all(target.valid_mask)


def test_trapezoid_gives_endpoints_half_the_weight_of_interior_scans() -> None:
    endpoint = np.zeros((7, 1), dtype=np.float64)
    endpoint[0] = 12.0
    interior = np.zeros((7, 1), dtype=np.float64)
    interior[1] = 12.0

    endpoint_target = build_accumulation_target(
        endpoint, input_encoding="rain_rate"
    )
    interior_target = build_accumulation_target(
        interior, input_encoding="rain_rate"
    )

    np.testing.assert_allclose(endpoint_target.raw_accumulation_mm, [0.5])
    np.testing.assert_allclose(interior_target.raw_accumulation_mm, [1.0])


def test_adjacent_targets_share_boundary_scan_as_two_half_weights() -> None:
    scans = np.arange(1.0, 14.0, dtype=np.float64).reshape(13, 1)

    first = build_accumulation_target(
        scans[:7], input_encoding="rain_rate"
    ).raw_accumulation_mm
    second = build_accumulation_target(
        scans[6:], input_encoding="rain_rate"
    ).raw_accumulation_mm
    direct_hour = (5.0 / 60.0) * (
        0.5 * scans[0] + scans[1:12].sum(axis=0) + 0.5 * scans[12]
    )

    np.testing.assert_allclose(first + second, direct_hour)


def test_cprecnet_is_converted_to_linear_space_before_accumulation() -> None:
    # v=1/3 maps to exactly 1 mm/h; seven constant scans integrate to 0.5 mm.
    scans = np.full((7, 2), 1.0 / 3.0, dtype=np.float64)
    coverage = np.asarray([1, 0], dtype=np.uint8)

    target = build_accumulation_target(
        scans,
        input_encoding="cprecnet_normalized",
        static_coverage=coverage,
    )

    np.testing.assert_allclose(target.raw_accumulation_mm, [0.5, 0.0])
    np.testing.assert_array_equal(target.valid_mask, [True, False])
    np.testing.assert_array_equal(target.wet, [True, False])


def test_wholly_missing_timestamp_raises_sample_drop_signal() -> None:
    scans: list[np.ndarray | None] = [np.ones((2, 2)) for _ in range(7)]
    scans[3] = None

    with pytest.raises(MissingTimestampError, match="index: 3"):
        build_accumulation_target(scans, input_encoding="rain_rate")


def test_six_scans_are_not_renormalized() -> None:
    with pytest.raises(ValueError, match="exactly 7"):
        build_accumulation_target(
            np.ones((6, 2, 2)), input_encoding="rain_rate"
        )


def test_present_invalid_pixel_is_masked_not_called_dry_observation() -> None:
    scans = np.full((7, 2), 2.0, dtype=np.float64)
    scans[2, 1] = -25_000.0  # an invalid raw sentinel is harmless when masked
    validity = np.ones((7, 2), dtype=np.bool_)
    validity[2, 1] = False

    target = build_accumulation_target(
        scans,
        input_encoding="rain_rate",
        pixel_validity=validity,
    )

    assert target.valid_mask.tolist() == [True, False]
    assert target.raw_accumulation_mm.tolist() == [1.0, 0.0]
    assert target.wet.tolist() == [True, False]
    # The zero at index 1 is neutral fill, distinguishable from valid dry data.
    assert not target.valid_mask[1]


def test_valid_negative_or_nonfinite_linear_rain_rate_is_rejected() -> None:
    negative = np.ones((7, 1), dtype=np.float64)
    negative[0, 0] = -1.0
    nonfinite = np.ones((7, 1), dtype=np.float64)
    nonfinite[0, 0] = np.nan

    with pytest.raises(ValueError, match="negative"):
        build_accumulation_target(negative, input_encoding="rain_rate")
    with pytest.raises(ValueError, match="non-finite"):
        build_accumulation_target(nonfinite, input_encoding="rain_rate")


def test_cprecnet_requires_static_coverage_and_forbids_dynamic_mask() -> None:
    scans = np.zeros((7, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="static_coverage"):
        build_accumulation_target(
            scans, input_encoding="cprecnet_normalized"
        )
    with pytest.raises(ValueError, match="no dynamic pixel validity"):
        build_accumulation_target(
            scans,
            input_encoding="cprecnet_normalized",
            pixel_validity=np.ones_like(scans, dtype=np.bool_),
            static_coverage=np.ones((1,), dtype=np.bool_),
        )


def test_shape_and_mask_contracts_are_strict() -> None:
    scans = [np.ones((2, 2)) for _ in range(7)]
    scans[4] = np.ones((3, 2))
    with pytest.raises(ValueError, match="shape"):
        build_accumulation_target(scans, input_encoding="rain_rate")

    with pytest.raises(ValueError, match="only boolean or 0/1"):
        build_accumulation_target(
            np.ones((7, 2, 2)),
            input_encoding="rain_rate",
            static_coverage=np.full((2, 2), 2),
        )
