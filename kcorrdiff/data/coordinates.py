"""Physical coordinate metadata shared by all K-CorrDiff spatial streams.

The v1.1.3b contract deliberately does not normalize each domain to its own
``[-1, 1]`` range.  Target, context, DEM, and ERA positions are translated by
the same target-domain centre and divided by the fixed 100 km length scale.
Footprints remain in kilometres so their physical size is not ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


DEFAULT_COORDINATE_SCALE_KM = 100.0

# Exact CRS of the supplied KMA HSR coordinate artifacts.
KMA_LCC_LATITUDE_OF_ORIGIN_DEGREES = 38.0
KMA_LCC_CENTRAL_LONGITUDE_DEGREES = 126.0
KMA_LCC_STANDARD_PARALLELS_DEGREES = (30.0, 60.0)
WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_INVERSE_FLATTENING = 298.257223563


def wgs84_latlon_to_kma_lcc(
    latitude_degrees: npt.ArrayLike,
    longitude_degrees: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 coordinates into the KMA HSR LCC CRS, in metres.

    This dependency-free ellipsoidal LCC implementation keeps ERA geometry
    reproducible across training images without selecting a system CRS by
    name or relying on a domain-extrema normalization.
    """

    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    longitude = np.asarray(longitude_degrees, dtype=np.float64)
    try:
        latitude, longitude = np.broadcast_arrays(latitude, longitude)
    except ValueError as error:
        raise ValueError("latitude and longitude must be broadcast-compatible") from error
    if not np.all(np.isfinite(latitude)) or not np.all(np.isfinite(longitude)):
        raise ValueError("latitude and longitude must be finite")
    if np.any((latitude <= -90.0) | (latitude >= 90.0)):
        raise ValueError("LCC projection requires latitude strictly inside (-90, 90)")

    flattening = 1.0 / WGS84_INVERSE_FLATTENING
    eccentricity = np.sqrt(flattening * (2.0 - flattening))

    def meridional_scale(phi: np.ndarray) -> np.ndarray:
        return np.cos(phi) / np.sqrt(
            1.0 - eccentricity**2 * np.sin(phi) ** 2
        )

    def conformal_t(phi: np.ndarray) -> np.ndarray:
        sin_phi = np.sin(phi)
        ratio = (
            (1.0 - eccentricity * sin_phi)
            / (1.0 + eccentricity * sin_phi)
        ) ** (eccentricity / 2.0)
        return np.tan(np.pi / 4.0 - phi / 2.0) / ratio

    phi = np.deg2rad(latitude)
    longitude_radians = np.deg2rad(longitude)
    phi0 = np.deg2rad(KMA_LCC_LATITUDE_OF_ORIGIN_DEGREES)
    lambda0 = np.deg2rad(KMA_LCC_CENTRAL_LONGITUDE_DEGREES)
    phi1, phi2 = np.deg2rad(KMA_LCC_STANDARD_PARALLELS_DEGREES)
    cone = (
        np.log(meridional_scale(phi1)) - np.log(meridional_scale(phi2))
    ) / (np.log(conformal_t(phi1)) - np.log(conformal_t(phi2)))
    factor = meridional_scale(phi1) / (cone * conformal_t(phi1) ** cone)
    radius = WGS84_SEMI_MAJOR_AXIS_M * factor * conformal_t(phi) ** cone
    radius0 = WGS84_SEMI_MAJOR_AXIS_M * factor * conformal_t(phi0) ** cone
    theta = cone * (longitude_radians - lambda0)
    return (
        np.asarray(radius * np.sin(theta), dtype=np.float64),
        np.asarray(radius0 - radius * np.cos(theta), dtype=np.float64),
    )


def _curvilinear_footprints_km(
    x_m: np.ndarray, y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if x_m.ndim != 2 or y_m.shape != x_m.shape or min(x_m.shape) < 2:
        raise ValueError("projected coordinate grids must share a >=2 by >=2 shape")
    horizontal = np.hypot(np.diff(x_m, axis=1), np.diff(y_m, axis=1)) / 1000.0
    vertical = np.hypot(np.diff(x_m, axis=0), np.diff(y_m, axis=0)) / 1000.0
    width = np.empty_like(x_m, dtype=np.float64)
    height = np.empty_like(y_m, dtype=np.float64)
    width[:, 0] = horizontal[:, 0]
    width[:, -1] = horizontal[:, -1]
    width[:, 1:-1] = 0.5 * (horizontal[:, :-1] + horizontal[:, 1:])
    height[0] = vertical[0]
    height[-1] = vertical[-1]
    height[1:-1] = 0.5 * (vertical[:-1] + vertical[1:])
    if np.any(width <= 0.0) or np.any(height <= 0.0):
        raise ValueError("projected grid contains a zero-size footprint")
    return width, height


def _centres_1d(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    centres = np.asarray(values, dtype=np.float64)
    if centres.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if centres.size < 2:
        raise ValueError(f"{name} must contain at least two centres")
    if not np.all(np.isfinite(centres)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(np.diff(centres) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return centres


def cell_edges_from_centers(centers: npt.ArrayLike) -> np.ndarray:
    """Infer cell edges from strictly increasing tensor-product centres.

    Interior edges are adjacent-centre midpoints.  Each outer edge is a
    half-step extrapolation, which is the natural cell footprint convention
    for both the uniform target grid and the NON_UNI context grid.
    """

    centres = _centres_1d(centers, name="centers")
    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges


def target_center_from_axes(
    target_x_lcc_km: npt.ArrayLike,
    target_y_lcc_km: npt.ArrayLike,
) -> tuple[float, float]:
    """Return the LCC centre of a target grid from its coordinate axes."""

    x = _centres_1d(target_x_lcc_km, name="target_x_lcc_km")
    y = _centres_1d(target_y_lcc_km, name="target_y_lcc_km")
    return (float(0.5 * (x[0] + x[-1])), float(0.5 * (y[0] + y[-1])))


def normalize_lcc_coordinates(
    x_lcc_km: npt.ArrayLike,
    y_lcc_km: npt.ArrayLike,
    *,
    target_center_x_km: float,
    target_center_y_km: float,
    scale_km: float = DEFAULT_COORDINATE_SCALE_KM,
) -> tuple[np.ndarray, np.ndarray]:
    """Map LCC coordinates into the common target-centred reference frame.

    Inputs may be scalars, axes, or already-meshed arrays.  No extrema are
    inspected and no domain-specific scaling is performed.
    """

    if not np.isfinite(target_center_x_km) or not np.isfinite(target_center_y_km):
        raise ValueError("target centre must be finite")
    if not np.isfinite(scale_km) or scale_km <= 0.0:
        raise ValueError("scale_km must be finite and positive")

    x = np.asarray(x_lcc_km, dtype=np.float64)
    y = np.asarray(y_lcc_km, dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("LCC coordinates must be finite")
    return (
        (x - float(target_center_x_km)) / float(scale_km),
        (y - float(target_center_y_km)) / float(scale_km),
    )


# A descriptive alias used by callers that want to emphasize the shared frame.
common_lcc_coordinates = normalize_lcc_coordinates


def _block_shape(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        block_y = block_x = int(value)
    else:
        if len(value) != 2:
            raise ValueError("block_shape must be an int or a (y, x) pair")
        block_y, block_x = (int(value[0]), int(value[1]))
    if block_y <= 0 or block_x <= 0:
        raise ValueError("block dimensions must be positive")
    return block_y, block_x


def token_footprints(
    x_centers_km: npt.ArrayLike,
    y_centers_km: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D physical width and height for tensor-product grid cells."""

    x_edges = cell_edges_from_centers(x_centers_km)
    y_edges = cell_edges_from_centers(y_centers_km)
    widths = np.diff(x_edges)
    heights = np.diff(y_edges)
    shape = (heights.size, widths.size)
    return (
        np.broadcast_to(widths[None, :], shape).copy(),
        np.broadcast_to(heights[:, None], shape).copy(),
    )


@dataclass(frozen=True)
class TokenGeometry:
    """Geometry carried alongside a spatial encoder feature map.

    ``x_shared`` and ``y_shared`` are dimensionless common-frame positions.
    Footprint sizes are physical kilometres.  ``valid_fraction`` is an
    area-weighted source-support diagnostic when supplied by the builder.  A
    block token measures support inside its footprint; a padded-convolution
    token measures support inside the explicitly documented downsampler
    receptive field, which is deliberately distinct from feature-cell size.
    """

    x_shared: np.ndarray
    y_shared: np.ndarray
    footprint_width_km: np.ndarray
    footprint_height_km: np.ndarray
    valid_fraction: np.ndarray | None
    coordinate_scale_km: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.x_shared.shape

    @property
    def footprint_width_shared(self) -> np.ndarray:
        return self.footprint_width_km / self.coordinate_scale_km

    @property
    def footprint_height_shared(self) -> np.ndarray:
        return self.footprint_height_km / self.coordinate_scale_km


def build_token_geometry(
    x_centers_km: npt.ArrayLike,
    y_centers_km: npt.ArrayLike,
    *,
    target_center_x_km: float,
    target_center_y_km: float,
    block_shape: int | Sequence[int] = 1,
    valid_fraction: npt.ArrayLike | None = None,
    scale_km: float = DEFAULT_COORDINATE_SCALE_KM,
) -> TokenGeometry:
    """Build centre, footprint, and optional validity metadata for tokens.

    ``block_shape`` describes non-overlapping source cells represented by one
    token, for example 2 at encoder L1.  A token centre is the midpoint of its
    complete physical footprint, not the mean of independently normalized
    source indices.  Validity aggregation is weighted by physical cell area.
    """

    x = _centres_1d(x_centers_km, name="x_centers_km")
    y = _centres_1d(y_centers_km, name="y_centers_km")
    block_y, block_x = _block_shape(block_shape)
    if y.size % block_y or x.size % block_x:
        raise ValueError("grid shape must be divisible by block_shape")

    x_edges = cell_edges_from_centers(x)
    y_edges = cell_edges_from_centers(y)
    token_x_edges = x_edges[::block_x]
    token_y_edges = y_edges[::block_y]
    token_x = 0.5 * (token_x_edges[:-1] + token_x_edges[1:])
    token_y = 0.5 * (token_y_edges[:-1] + token_y_edges[1:])
    x_lcc, y_lcc = np.meshgrid(token_x, token_y, indexing="xy")
    x_shared, y_shared = normalize_lcc_coordinates(
        x_lcc,
        y_lcc,
        target_center_x_km=target_center_x_km,
        target_center_y_km=target_center_y_km,
        scale_km=scale_km,
    )

    widths = np.diff(token_x_edges)
    heights = np.diff(token_y_edges)
    shape = (heights.size, widths.size)
    footprint_width = np.broadcast_to(widths[None, :], shape).copy()
    footprint_height = np.broadcast_to(heights[:, None], shape).copy()

    token_validity: np.ndarray | None = None
    if valid_fraction is not None:
        validity = np.asarray(valid_fraction, dtype=np.float64)
        if validity.shape != (y.size, x.size):
            raise ValueError(
                "valid_fraction must have shape "
                f"{(y.size, x.size)}, got {validity.shape}"
            )
        if not np.all(np.isfinite(validity)):
            raise ValueError("valid_fraction must be finite")
        if np.any((validity < 0.0) | (validity > 1.0)):
            raise ValueError("valid_fraction values must lie in [0, 1]")

        source_widths = np.diff(x_edges)
        source_heights = np.diff(y_edges)
        source_area = source_heights[:, None] * source_widths[None, :]
        grouped_valid_area = (validity * source_area).reshape(
            y.size // block_y,
            block_y,
            x.size // block_x,
            block_x,
        ).sum(axis=(1, 3))
        token_area = footprint_height * footprint_width
        token_validity = grouped_valid_area / token_area

    return TokenGeometry(
        x_shared=x_shared,
        y_shared=y_shared,
        footprint_width_km=footprint_width,
        footprint_height_km=footprint_height,
        valid_fraction=token_validity,
        coordinate_scale_km=float(scale_km),
    )


def _extended_edge(edges: np.ndarray, index: int) -> float:
    """Return a cell edge while extending the boundary cell spacing."""

    cells = edges.size - 1
    if index < 0:
        return float(edges[0] + index * (edges[1] - edges[0]))
    if index > cells:
        return float(edges[-1] + (index - cells) * (edges[-1] - edges[-2]))
    return float(edges[index])


def build_repeated_stride2_geometry(
    x_centers_km: npt.ArrayLike,
    y_centers_km: npt.ArrayLike,
    *,
    target_center_x_km: float,
    target_center_y_km: float,
    downsampling_levels: int,
    valid_fraction: npt.ArrayLike | None = None,
    scale_km: float = DEFAULT_COORDINATE_SCALE_KM,
) -> TokenGeometry:
    """Geometry for repeated ``Conv2d(k=3, stride=2, padding=1)`` levels.

    After ``L`` convolutions, output index ``j`` is centred on input index
    ``j * 2**L`` and its nominal receptive field covers input indices from
    ``j*2**L-(2**L-1)`` through ``j*2**L+(2**L-1)``.  This differs from a
    non-overlapping ``2**L``-cell block: the first output stays centred on the
    first input cell and adjacent receptive fields overlap.

    Token footprints retain the architecture's feature-cell convention: one
    local input-cell width/height times the lattice jump (4/8 km for target
    L3/L4).  This must not be confused with the network receptive field, which
    also grows through stride-one residual convolutions.  Separately, the
    returned validity diagnoses the nominal *downsampler* receptive support
    (``2**(L+1)-1`` input cells), including unsupported structural padding.
    It is therefore present even when no extra validity mask is supplied.
    """

    x = _centres_1d(x_centers_km, name="x_centers_km")
    y = _centres_1d(y_centers_km, name="y_centers_km")
    if isinstance(downsampling_levels, bool) or not isinstance(
        downsampling_levels, (int, np.integer)
    ):
        raise TypeError("downsampling_levels must be an integer")
    levels = int(downsampling_levels)
    if levels < 0:
        raise ValueError("downsampling_levels must be non-negative")

    jump = 1 << levels
    radius = jump - 1
    output_x_indices = np.arange(0, x.size, jump, dtype=np.int64)
    output_y_indices = np.arange(0, y.size, jump, dtype=np.int64)
    x_edges = cell_edges_from_centers(x)
    y_edges = cell_edges_from_centers(y)
    source_widths = np.diff(x_edges)
    source_heights = np.diff(y_edges)

    x_lcc, y_lcc = np.meshgrid(
        x[output_x_indices], y[output_y_indices], indexing="xy"
    )
    x_shared, y_shared = normalize_lcc_coordinates(
        x_lcc,
        y_lcc,
        target_center_x_km=target_center_x_km,
        target_center_y_km=target_center_y_km,
        scale_km=scale_km,
    )

    x_bounds = tuple(
        (int(index) - radius, int(index) + radius + 1)
        for index in output_x_indices
    )
    y_bounds = tuple(
        (int(index) - radius, int(index) + radius + 1)
        for index in output_y_indices
    )
    receptive_widths = np.asarray(
        [
            _extended_edge(x_edges, high) - _extended_edge(x_edges, low)
            for low, high in x_bounds
        ],
        dtype=np.float64,
    )
    receptive_heights = np.asarray(
        [
            _extended_edge(y_edges, high) - _extended_edge(y_edges, low)
            for low, high in y_bounds
        ],
        dtype=np.float64,
    )
    shape = (output_y_indices.size, output_x_indices.size)
    token_widths = jump * source_widths[output_x_indices]
    token_heights = jump * source_heights[output_y_indices]
    footprint_width = np.broadcast_to(token_widths[None, :], shape).copy()
    footprint_height = np.broadcast_to(token_heights[:, None], shape).copy()

    if valid_fraction is None:
        validity = np.ones((y.size, x.size), dtype=np.float64)
    else:
        validity = np.asarray(valid_fraction, dtype=np.float64)
        if validity.shape != (y.size, x.size):
            raise ValueError(
                "valid_fraction must have shape "
                f"{(y.size, x.size)}, got {validity.shape}"
            )
        if not np.all(np.isfinite(validity)):
            raise ValueError("valid_fraction must be finite")
        if np.any((validity < 0.0) | (validity > 1.0)):
            raise ValueError("valid_fraction values must lie in [0, 1]")

    source_area = source_heights[:, None] * source_widths[None, :]
    supported = np.empty(shape, dtype=np.float64)
    for output_y, (low_y, high_y) in enumerate(y_bounds):
        clipped_low_y = max(low_y, 0)
        clipped_high_y = min(high_y, y.size)
        for output_x, (low_x, high_x) in enumerate(x_bounds):
            clipped_low_x = max(low_x, 0)
            clipped_high_x = min(high_x, x.size)
            valid_area = (
                validity[
                    clipped_low_y:clipped_high_y,
                    clipped_low_x:clipped_high_x,
                ]
                * source_area[
                    clipped_low_y:clipped_high_y,
                    clipped_low_x:clipped_high_x,
                ]
            ).sum()
            nominal_area = (
                receptive_heights[output_y] * receptive_widths[output_x]
            )
            supported[output_y, output_x] = valid_area / nominal_area

    return TokenGeometry(
        x_shared=x_shared,
        y_shared=y_shared,
        footprint_width_km=footprint_width,
        footprint_height_km=footprint_height,
        valid_fraction=supported,
        coordinate_scale_km=float(scale_km),
    )


def build_era_latlon_geometry(
    latitude_degrees: npt.ArrayLike,
    longitude_degrees: npt.ArrayLike,
    *,
    target_center_x_km: float,
    target_center_y_km: float,
    valid_fraction: npt.ArrayLike | None = None,
    scale_km: float = DEFAULT_COORDINATE_SCALE_KM,
) -> TokenGeometry:
    """Build two-dimensional common-LCC geometry for an ERA lat/lon grid."""

    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    longitude = np.asarray(longitude_degrees, dtype=np.float64)
    if latitude.ndim == 1 and longitude.ndim == 1:
        longitude, latitude = np.meshgrid(longitude, latitude, indexing="xy")
    elif latitude.ndim != 2 or latitude.shape != longitude.shape:
        raise ValueError("ERA latitude/longitude must be two axes or equal 2-D grids")
    x_m, y_m = wgs84_latlon_to_kma_lcc(latitude, longitude)
    width_km, height_km = _curvilinear_footprints_km(x_m, y_m)
    x_shared, y_shared = normalize_lcc_coordinates(
        x_m / 1000.0,
        y_m / 1000.0,
        target_center_x_km=target_center_x_km,
        target_center_y_km=target_center_y_km,
        scale_km=scale_km,
    )
    validity: np.ndarray | None = None
    if valid_fraction is not None:
        validity = np.asarray(valid_fraction, dtype=np.float64)
        if validity.shape != x_m.shape or not np.all(np.isfinite(validity)):
            raise ValueError("ERA valid_fraction must be finite and match the grid")
        if np.any((validity < 0.0) | (validity > 1.0)):
            raise ValueError("ERA valid_fraction must lie in [0, 1]")
        validity = validity.copy()
    return TokenGeometry(
        x_shared=x_shared,
        y_shared=y_shared,
        footprint_width_km=width_km,
        footprint_height_km=height_km,
        valid_fraction=validity,
        coordinate_scale_km=float(scale_km),
    )


__all__ = [
    "DEFAULT_COORDINATE_SCALE_KM",
    "TokenGeometry",
    "build_era_latlon_geometry",
    "build_repeated_stride2_geometry",
    "build_token_geometry",
    "cell_edges_from_centers",
    "common_lcc_coordinates",
    "normalize_lcc_coordinates",
    "target_center_from_axes",
    "token_footprints",
    "wgs84_latlon_to_kma_lcc",
]
