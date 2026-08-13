import numpy as np
import pytest

from kcorrdiff.data.coordinates import (
    build_era_latlon_geometry,
    build_token_geometry,
    normalize_lcc_coordinates,
    target_center_from_axes,
    token_footprints,
    wgs84_latlon_to_kma_lcc,
)


def test_common_coordinates_use_target_center_and_fixed_scale() -> None:
    target_x = np.arange(45.0, 173.0, 0.5)
    target_y = np.arange(-160.0, -32.0, 0.5)
    center_x, center_y = target_center_from_axes(target_x, target_y)

    target_shared_x, _ = normalize_lcc_coordinates(
        target_x,
        target_y,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
    )
    context_x = np.array([-45.0, center_x, 262.0])
    context_shared_x, _ = normalize_lcc_coordinates(
        context_x,
        np.array([center_y]),
        target_center_x_km=center_x,
        target_center_y_km=center_y,
    )

    assert context_shared_x[1] == pytest.approx(0.0)
    assert context_shared_x[0] == pytest.approx((-45.0 - center_x) / 100.0)
    assert target_shared_x[-1] == pytest.approx((172.5 - center_x) / 100.0)
    # In particular, neither domain's extrema have independently become +/-1.
    assert not np.isclose(context_shared_x[0], -1.0)
    assert not np.isclose(context_shared_x[-1], 1.0)
    assert not np.isclose(target_shared_x[0], -1.0)


def test_kma_lcc_projection_matches_supplied_target_coordinate_contract() -> None:
    latitude = np.array(
        [36.52129247315956, 37.10054845605839, 37.679542821396375]
    )
    longitude = np.array(
        [126.51392607616683, 127.25037572808765, 128.00615342157317]
    )
    x_m, y_m = wgs84_latlon_to_kma_lcc(latitude, longitude)
    np.testing.assert_allclose(x_m, [45000.0, 108500.0, 172500.0], atol=1.0e-5)
    np.testing.assert_allclose(y_m, [-160000.0, -96500.0, -32500.0], atol=1.0e-5)


def test_era_latlon_geometry_retains_curvilinear_physical_positions() -> None:
    latitude = np.linspace(32.0, 40.0, 33)
    longitude = np.linspace(123.0, 131.0, 33)
    geometry = build_era_latlon_geometry(
        latitude,
        longitude,
        target_center_x_km=108.75,
        target_center_y_km=-96.25,
        valid_fraction=np.ones((33, 33)),
    )
    assert geometry.shape == (33, 33)
    assert geometry.valid_fraction is not None
    np.testing.assert_array_equal(geometry.valid_fraction, 1.0)
    assert np.all(geometry.footprint_width_km > 0.0)
    assert np.all(geometry.footprint_height_km > 0.0)
    # A projected regular lat/lon grid is not a separable x/y tensor product.
    assert not np.allclose(geometry.x_shared[0], geometry.x_shared[-1])
    assert not np.allclose(geometry.y_shared[:, 0], geometry.y_shared[:, -1])


def test_token_geometry_tracks_physical_footprints_across_levels() -> None:
    x = np.arange(8, dtype=np.float64) * 0.5
    y = np.arange(8, dtype=np.float64) * 0.5

    level_zero = build_token_geometry(
        x,
        y,
        target_center_x_km=1.75,
        target_center_y_km=1.75,
    )
    level_one = build_token_geometry(
        x,
        y,
        target_center_x_km=1.75,
        target_center_y_km=1.75,
        block_shape=2,
    )

    assert level_zero.shape == (8, 8)
    assert level_one.shape == (4, 4)
    np.testing.assert_allclose(level_zero.footprint_width_km, 0.5)
    np.testing.assert_allclose(level_zero.footprint_height_km, 0.5)
    np.testing.assert_allclose(level_one.footprint_width_km, 1.0)
    np.testing.assert_allclose(level_one.footprint_height_km, 1.0)
    np.testing.assert_allclose(level_one.footprint_width_shared, 0.01)
    assert level_one.x_shared[0, 0] == pytest.approx(-0.015)
    assert level_one.y_shared[0, 0] == pytest.approx(-0.015)


def test_nonuniform_footprints_and_validity_are_area_weighted() -> None:
    x = np.array([0.0, 1.0, 3.0, 6.0])
    y = np.array([0.0, 2.0, 5.0, 9.0])
    widths, heights = token_footprints(x, y)
    np.testing.assert_allclose(widths[0], [1.0, 1.5, 2.5, 3.0])
    np.testing.assert_allclose(heights[:, 0], [2.0, 2.5, 3.5, 4.0])

    validity = np.ones((4, 4), dtype=np.float64)
    validity[0, 0] = 0.0
    geometry = build_token_geometry(
        x,
        y,
        target_center_x_km=3.0,
        target_center_y_km=4.5,
        block_shape=(2, 2),
        valid_fraction=validity,
    )
    source_area_00 = widths[0, 0] * heights[0, 0]
    expected = 1.0 - source_area_00 / (
        geometry.footprint_width_km[0, 0]
        * geometry.footprint_height_km[0, 0]
    )
    assert geometry.valid_fraction is not None
    assert geometry.valid_fraction[0, 0] == pytest.approx(expected)
    np.testing.assert_allclose(geometry.valid_fraction[1, :], 1.0)


def test_coordinate_axes_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        target_center_from_axes([0.0, 0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="divisible"):
        build_token_geometry(
            np.arange(5.0),
            np.arange(4.0),
            target_center_x_km=2.0,
            target_center_y_km=1.5,
            block_shape=2,
        )
