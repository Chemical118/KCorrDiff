import numpy as np
import pytest
from scipy.sparse import issparse

from kcorrdiff.data.context_regrid import (
    build_area_integrated_weights,
    build_context_regrid_operator,
)


def test_axis_operator_is_sparse_and_reproduces_constant_and_linear_fields() -> None:
    source = np.array([0.0, 0.7, 1.9, 4.2, 5.0])
    destination = np.linspace(source[0], source[-1], 9)
    weights = build_area_integrated_weights(source, destination)

    assert issparse(weights)
    np.testing.assert_allclose(weights @ np.ones_like(source), 1.0, atol=1.0e-13)
    np.testing.assert_allclose(
        weights @ (2.5 * source - 0.75),
        2.5 * destination - 0.75,
        atol=1.0e-12,
    )
    assert weights.nnz < weights.shape[0] * weights.shape[1]


def test_separable_operator_preserves_constant_and_affine_rain_rate() -> None:
    source_x = np.array([0.0, 1.0, 2.0, 4.0, 7.0])
    source_y = np.array([-3.0, -1.0, 0.0, 0.8, 3.0])
    destination_x = np.linspace(0.0, 7.0, 8)
    destination_y = np.linspace(-3.0, 3.0, 7)
    operator = build_context_regrid_operator(
        source_x,
        source_y,
        destination_x_km=destination_x,
        destination_y_km=destination_y,
    )

    constant = np.full(operator.source_shape, 7.25)
    np.testing.assert_allclose(operator.apply_linear(constant), 7.25, atol=1.0e-12)

    xx, yy = np.meshgrid(source_x, source_y, indexing="xy")
    affine = 10.0 + 0.4 * xx + 0.2 * yy
    expected_xx, expected_yy = np.meshgrid(
        destination_x, destination_y, indexing="xy"
    )
    expected = 10.0 + 0.4 * expected_xx + 0.2 * expected_yy
    np.testing.assert_allclose(operator.apply_linear(affine), expected, atol=2.0e-12)


def test_area_integration_antialiases_instead_of_point_sampling() -> None:
    source = np.arange(-2.0, 3.0)
    destination = np.array([-2.0, 0.0, 2.0])
    weights = build_area_integrated_weights(source, destination)
    impulse = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    integrated = np.asarray(weights @ impulse).ravel()
    assert integrated[1] == pytest.approx(0.5)
    assert integrated[1] != pytest.approx(impulse[2])


def test_regrid_separates_missing_from_dry_and_preserves_valid_constant() -> None:
    source_x = np.array([0.0, 1.0, 2.0, 4.0, 6.0])
    source_y = np.array([0.0, 1.0, 3.0, 4.0, 6.0])
    operator = build_context_regrid_operator(
        source_x, source_y, destination_shape=(7, 7)
    )
    rate = np.full(operator.source_shape, 5.0)
    valid = np.ones(operator.source_shape, dtype=bool)
    valid[2, 2] = False
    rate[2, 2] = np.nan

    result = operator.regrid(rate, valid_mask=valid)
    available = result.valid_fraction > 1.0e-12
    np.testing.assert_allclose(result.mean_rate_mm_per_hour[available], 5.0)
    assert np.any(result.valid_fraction < 1.0)
    assert np.all(result.valid_fraction >= 0.0)
    assert np.all(result.valid_fraction <= 1.0)
    assert np.all(result.interpolation_confidence <= result.valid_fraction)
    assert np.all(result.wet_mask[available])


def test_local_max_preserves_core_and_nearest_mode_is_explicit() -> None:
    source = np.arange(-2.0, 3.0)
    destination = np.array([-2.0, 0.0, 2.0])
    operator = build_context_regrid_operator(
        source,
        source,
        destination_x_km=destination,
        destination_y_km=destination,
    )
    rate = np.zeros(operator.source_shape)
    rate[2, 2] = 20.0

    local_max = operator.regrid(rate, detail_mode="local_max")
    nearest = operator.regrid(rate, detail_mode="nearest")

    assert local_max.mean_rate_mm_per_hour[1, 1] == pytest.approx(5.0)
    assert local_max.detail_rate_mm_per_hour[1, 1] == pytest.approx(20.0)
    assert nearest.detail_rate_mm_per_hour[1, 1] == pytest.approx(20.0)
    assert local_max.detail_mode == "local_max"
    assert nearest.detail_mode == "nearest"
    assert np.all(local_max.detail_valid)


def test_geometry_channels_have_physical_meaning_and_valid_ranges() -> None:
    source_x = np.array([0.0, 1.0, 3.0, 6.0])
    source_y = np.array([-2.0, 0.0, 1.0, 5.0])
    operator = build_context_regrid_operator(
        source_x, source_y, destination_shape=(7, 7)
    )

    assert operator.source_dx_km.shape == (7, 7)
    assert operator.source_dy_km.shape == (7, 7)
    assert operator.nearest_source_distance_km.shape == (7, 7)
    assert np.all(operator.source_dx_km > 0.0)
    assert np.all(operator.source_dy_km > 0.0)
    assert np.all((operator.geometry_confidence > 0.0))
    assert np.all((operator.geometry_confidence <= 1.0))
    assert operator.nearest_source_distance_km[0, 0] == pytest.approx(0.0)
    assert operator.geometry_confidence[0, 0] == pytest.approx(1.0)


def test_negative_valid_linear_rain_rate_is_rejected() -> None:
    source = np.arange(3.0)
    operator = build_context_regrid_operator(source, source)
    rate = np.zeros((3, 3))
    rate[1, 1] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        operator.regrid(rate)


def test_unsafe_boundary_extrapolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative weights"):
        build_area_integrated_weights(
            [0.0, 0.1, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        )
