"""Area-integrated regridding for the NON_UNI radar context grid.

The operator integrates the piecewise-linear source reconstruction over every
destination cell.  It is separable because the source is a non-uniform
tensor-product grid::

    destination = W_y @ source @ W_x.T

Rain rates passed to this module are always linear ``mm/h`` values.  Applying
a log/normalization transform before conservative spatial regridding violates
the K-CorrDiff v1.1.3b data contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from .coordinates import cell_edges_from_centers


DetailMode = Literal["local_max", "nearest"]


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


def uniform_centers_like(
    source_centers_km: npt.ArrayLike,
    *,
    count: int | None = None,
) -> np.ndarray:
    """Make a uniform axis retaining the first and last source centres."""

    source = _centres_1d(source_centers_km, name="source_centers_km")
    size = source.size if count is None else int(count)
    if size < 2:
        raise ValueError("count must be at least two")
    return np.linspace(source[0], source[-1], size, dtype=np.float64)


def _integrate_linear_segment(
    left: float,
    right: float,
    source_left: float,
    source_right: float,
) -> tuple[float, float]:
    """Integrate the two local linear basis functions on one interval."""

    delta = source_right - source_left
    length = right - left
    # Each basis is affine on this segment, so its integral is exactly the
    # interval length times its midpoint value.  This form avoids cancellation
    # from subtracting squared absolute LCC coordinates.
    midpoint = left + 0.5 * length
    left_integral = length * (source_right - midpoint) / delta
    right_integral = length * (midpoint - source_left) / delta
    return left_integral, right_integral


def build_area_integrated_weights(
    source_centers_km: npt.ArrayLike,
    destination_centers_km: npt.ArrayLike,
    *,
    zero_tolerance: float = 1.0e-14,
) -> csr_matrix:
    """Build sparse 1-D cell-mean weights for a piecewise-linear field.

    For destination cell ``j`` and source nodal basis ``phi_i``, the returned
    matrix contains ``integral(phi_i) / destination_width``.  Destination
    centres must remain inside the source-centre extent.  The half destination
    cells outside the first/last source centre use the adjacent linear segment
    as an extrapolant; their symmetric integral consequently reproduces the
    endpoint value and preserves affine fields exactly.
    """

    source = _centres_1d(source_centers_km, name="source_centers_km")
    destination = _centres_1d(
        destination_centers_km, name="destination_centers_km"
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, np.max(np.abs(source)), np.max(np.abs(destination))
    )
    if destination[0] < source[0] - tolerance or destination[-1] > source[-1] + tolerance:
        raise ValueError(
            "destination centres must lie within the source-centre extent"
        )
    destination_edges = cell_edges_from_centers(destination)
    if (
        source[0] - destination_edges[0] > source[1] - source[0] + tolerance
        or destination_edges[-1] - source[-1]
        > source[-1] - source[-2] + tolerance
    ):
        raise ValueError(
            "a boundary destination half-cell exceeds its adjacent source "
            "interval; linear extrapolation would produce negative weights"
        )
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    for row, (cell_left, cell_right) in enumerate(
        zip(destination_edges[:-1], destination_edges[1:], strict=True)
    ):
        width = cell_right - cell_left
        internal_breaks = source[(source > cell_left) & (source < cell_right)]
        breaks = np.concatenate(([cell_left], internal_breaks, [cell_right]))
        accumulator: dict[int, float] = {}

        for left, right in zip(breaks[:-1], breaks[1:], strict=True):
            midpoint = 0.5 * (left + right)
            source_index = int(np.searchsorted(source, midpoint, side="right") - 1)
            source_index = min(max(source_index, 0), source.size - 2)
            left_integral, right_integral = _integrate_linear_segment(
                float(left),
                float(right),
                float(source[source_index]),
                float(source[source_index + 1]),
            )
            accumulator[source_index] = (
                accumulator.get(source_index, 0.0) + left_integral / width
            )
            accumulator[source_index + 1] = (
                accumulator.get(source_index + 1, 0.0) + right_integral / width
            )

        for column in sorted(accumulator):
            value = accumulator[column]
            if abs(value) > zero_tolerance:
                rows.append(row)
                columns.append(column)
                values.append(value)

    weights = csr_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
        shape=(destination.size, source.size),
        dtype=np.float64,
    )
    weights.sum_duplicates()
    weights.sort_indices()

    # These are construction invariants, rather than optional numerical tests.
    if weights.data.size and np.min(weights.data) < -2.0e-12:
        raise RuntimeError("area-integrated weights must be non-negative")
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    reproduced_centres = np.asarray(weights @ source).ravel()
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=2.0e-12):
        raise RuntimeError("area-integrated weights do not preserve constants")
    if not np.allclose(
        reproduced_centres, destination, rtol=0.0, atol=2.0e-11
    ):
        raise RuntimeError("area-integrated weights do not reproduce linear fields")
    return weights


# More explicit aliases for callers and serialized-operator builders.
build_area_integrated_operator_1d = build_area_integrated_weights
build_axis_regrid_weights = build_area_integrated_weights


def _nearest_indices(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    upper = np.searchsorted(source, destination, side="left")
    upper = np.clip(upper, 0, source.size - 1)
    lower = np.clip(upper - 1, 0, source.size - 1)
    choose_upper = np.abs(source[upper] - destination) < np.abs(
        source[lower] - destination
    )
    return np.where(choose_upper, upper, lower).astype(np.int64)


def _source_spacing_at_centres(source: np.ndarray) -> np.ndarray:
    return np.diff(cell_edges_from_centers(source))


def _axis_supports(weights: csr_matrix) -> tuple[np.ndarray, ...]:
    supports: list[np.ndarray] = []
    for row in range(weights.shape[0]):
        start, stop = weights.indptr[row : row + 2]
        row_indices = weights.indices[start:stop]
        row_values = weights.data[start:stop]
        support = row_indices[row_values > 0.0]
        if support.size == 0:
            # This can only arise from severe floating-point cancellation at
            # an extrapolated boundary; retain every nonzero contributor.
            support = row_indices
        supports.append(support)
    return tuple(supports)


def _apply_separable(
    field: np.ndarray,
    weights_y: csr_matrix,
    weights_x: csr_matrix,
) -> np.ndarray:
    intermediate = np.asarray(weights_y @ field, dtype=np.float64)
    return np.asarray(weights_x @ intermediate.T, dtype=np.float64).T


def _support_shift_offsets(
    supports: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...] | None:
    """Per-shift gather indices for contiguous supports, or None.

    Element ``shift`` holds, for every destination row, the index of that
    row's ``min(shift, support size - 1)``-th support member, so an
    elementwise-maximum sweep over all shifts visits every support member of
    every row.  Area integration yields contiguous supports; None (an empty
    or non-contiguous support) keeps callers on the per-row path.
    """

    if not supports or any(support.size == 0 for support in supports):
        return None
    for support in supports:
        if not np.array_equal(
            support, np.arange(support[0], support[0] + support.size)
        ):
            return None
    sizes = np.asarray([support.size for support in supports], dtype=np.intp)
    starts = np.asarray([support[0] for support in supports], dtype=np.intp)
    return tuple(
        starts + np.minimum(shift, sizes - 1) for shift in range(int(sizes.max()))
    )


@dataclass(frozen=True)
class ContextRegridResult:
    """Dynamic and confidence channels on the uniform context grid."""

    mean_rate_mm_per_hour: np.ndarray
    detail_rate_mm_per_hour: np.ndarray
    wet_mask: np.ndarray
    valid_fraction: np.ndarray
    detail_valid: np.ndarray
    source_dx_km: np.ndarray
    source_dy_km: np.ndarray
    nearest_source_distance_km: np.ndarray
    interpolation_confidence: np.ndarray
    detail_mode: DetailMode

    @property
    def area_mean_rate(self) -> np.ndarray:
        return self.mean_rate_mm_per_hour

    @property
    def local_max_or_nearest_rate(self) -> np.ndarray:
        return self.detail_rate_mm_per_hour


@dataclass(frozen=True)
class ContextRegridOperator:
    """Reusable sparse operators and geometry for context radar frames."""

    source_x_km: np.ndarray
    source_y_km: np.ndarray
    destination_x_km: np.ndarray
    destination_y_km: np.ndarray
    weights_x: csr_matrix
    weights_y: csr_matrix
    nearest_x_indices: np.ndarray
    nearest_y_indices: np.ndarray
    source_dx_km: np.ndarray
    source_dy_km: np.ndarray
    nearest_source_distance_km: np.ndarray
    geometry_confidence: np.ndarray
    _x_supports: tuple[np.ndarray, ...]
    _y_supports: tuple[np.ndarray, ...]

    @property
    def source_shape(self) -> tuple[int, int]:
        return (self.source_y_km.size, self.source_x_km.size)

    @property
    def destination_shape(self) -> tuple[int, int]:
        return (self.destination_y_km.size, self.destination_x_km.size)

    @property
    def W_x(self) -> csr_matrix:
        return self.weights_x

    @property
    def W_y(self) -> csr_matrix:
        return self.weights_y

    @cached_property
    def _all_valid_fraction(self) -> np.ndarray:
        """Regridded basis mass of an entirely valid frame; read-only.

        The array is shared across every all-valid ``regrid`` call, so it must
        never be mutated in place.
        """

        ones = np.ones(self.source_shape, dtype=np.float64)
        fraction = np.clip(
            _apply_separable(ones, self.weights_y, self.weights_x), 0.0, 1.0
        )
        fraction.setflags(write=False)
        return fraction

    @cached_property
    def _x_support_offsets(self) -> tuple[np.ndarray, ...] | None:
        return _support_shift_offsets(self._x_supports)

    @cached_property
    def _y_support_offsets(self) -> tuple[np.ndarray, ...] | None:
        return _support_shift_offsets(self._y_supports)

    def apply_linear(self, field: npt.ArrayLike) -> np.ndarray:
        """Apply ``W_y @ field @ W_x.T`` without validity renormalization."""

        values = np.asarray(field, dtype=np.float64)
        if values.shape != self.source_shape:
            raise ValueError(
                f"field must have shape {self.source_shape}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("field must contain only finite values")
        return _apply_separable(values, self.weights_y, self.weights_x)

    def _local_max(
        self,
        values: np.ndarray,
        valid: np.ndarray,
        *,
        fill_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        masked = np.where(valid, values, -np.inf)
        x_offsets = self._x_support_offsets
        y_offsets = self._y_support_offsets
        if x_offsets is not None and y_offsets is not None:
            along_x = np.take(masked, x_offsets[0], axis=1)
            for offsets in x_offsets[1:]:
                np.maximum(along_x, np.take(masked, offsets, axis=1), out=along_x)
            detail = np.take(along_x, y_offsets[0], axis=0)
            for offsets in y_offsets[1:]:
                np.maximum(detail, np.take(along_x, offsets, axis=0), out=detail)
        else:
            along_x = np.empty(
                (self.source_y_km.size, self.destination_x_km.size),
                dtype=np.float64,
            )
            for destination_x, source_indices in enumerate(self._x_supports):
                along_x[:, destination_x] = np.max(masked[:, source_indices], axis=1)

            detail = np.empty(self.destination_shape, dtype=np.float64)
            for destination_y, source_indices in enumerate(self._y_supports):
                detail[destination_y, :] = np.max(
                    along_x[source_indices, :], axis=0
                )
        detail_valid = np.isfinite(detail)
        return np.where(detail_valid, detail, fill_value), detail_valid

    def _nearest(
        self,
        values: np.ndarray,
        valid: np.ndarray,
        *,
        fill_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        index = np.ix_(self.nearest_y_indices, self.nearest_x_indices)
        detail = values[index]
        detail_valid = valid[index]
        return np.where(detail_valid, detail, fill_value), detail_valid

    def regrid(
        self,
        rain_rate_mm_per_hour: npt.ArrayLike,
        *,
        valid_mask: npt.ArrayLike | None = None,
        detail_mode: DetailMode = "local_max",
        wet_threshold_mm_per_hour: float = 0.0,
        minimum_valid_fraction: float = 1.0e-12,
        fill_value: float = 0.0,
    ) -> ContextRegridResult:
        """Regrid one linear rain-rate frame and emit auxiliary channels.

        Missing samples contribute to ``valid_fraction`` rather than being
        interpreted as dry.  The area mean is renormalized over valid basis
        mass where any is available.  ``interpolation_confidence`` combines a
        geometry term with that dynamic valid fraction.
        """

        values = np.asarray(rain_rate_mm_per_hour, dtype=np.float64)
        if values.shape != self.source_shape:
            raise ValueError(
                "rain_rate_mm_per_hour must have shape "
                f"{self.source_shape}, got {values.shape}"
            )
        if valid_mask is None:
            valid = np.isfinite(values)
        else:
            supplied_validity = np.asarray(valid_mask)
            if supplied_validity.shape != self.source_shape:
                raise ValueError(
                    f"valid_mask must have shape {self.source_shape}, "
                    f"got {supplied_validity.shape}"
                )
            valid = supplied_validity.astype(bool, copy=False) & np.isfinite(values)
        all_valid = bool(valid.all())
        if np.any((values if all_valid else values[valid]) < 0.0):
            raise ValueError("valid linear rain rates must be non-negative")
        if detail_mode not in ("local_max", "nearest"):
            raise ValueError("detail_mode must be 'local_max' or 'nearest'")
        if not np.isfinite(wet_threshold_mm_per_hour) or wet_threshold_mm_per_hour < 0.0:
            raise ValueError("wet_threshold_mm_per_hour must be finite and non-negative")
        if not np.isfinite(minimum_valid_fraction) or not (
            0.0 <= minimum_valid_fraction <= 1.0
        ):
            raise ValueError("minimum_valid_fraction must lie in [0, 1]")
        if not np.isfinite(fill_value):
            raise ValueError("fill_value must be finite")

        safe_values = values if all_valid else np.where(valid, values, 0.0)
        if all_valid:
            valid_fraction = self._all_valid_fraction
        else:
            valid_fraction = _apply_separable(
                valid.astype(np.float64), self.weights_y, self.weights_x
            )
            # Roundoff at the endpoint extrapolation must not create impossible
            # validity values such as 1 + 2e-16.
            valid_fraction = np.clip(valid_fraction, 0.0, 1.0)
        numerator = _apply_separable(safe_values, self.weights_y, self.weights_x)
        has_valid_mean = valid_fraction > minimum_valid_fraction
        mean_rate = np.full(self.destination_shape, fill_value, dtype=np.float64)
        np.divide(
            numerator,
            valid_fraction,
            out=mean_rate,
            where=has_valid_mean,
        )
        # Tiny negative values can be produced only by floating-point endpoint
        # cancellation.  Substantial negatives indicate a violated rate input.
        mean_rate[np.abs(mean_rate) < 1.0e-14] = 0.0

        if detail_mode == "local_max":
            detail, detail_valid = self._local_max(
                values, valid, fill_value=fill_value
            )
        else:
            detail, detail_valid = self._nearest(
                values, valid, fill_value=fill_value
            )

        wet_mask = has_valid_mean & (mean_rate > wet_threshold_mm_per_hour)
        interpolation_confidence = self.geometry_confidence * valid_fraction
        return ContextRegridResult(
            mean_rate_mm_per_hour=mean_rate,
            detail_rate_mm_per_hour=detail,
            wet_mask=wet_mask,
            valid_fraction=valid_fraction,
            detail_valid=detail_valid,
            source_dx_km=self.source_dx_km,
            source_dy_km=self.source_dy_km,
            nearest_source_distance_km=self.nearest_source_distance_km,
            interpolation_confidence=interpolation_confidence,
            detail_mode=detail_mode,
        )

    __call__ = regrid


def build_context_regrid_operator(
    source_x_km: npt.ArrayLike,
    source_y_km: npt.ArrayLike,
    *,
    destination_x_km: npt.ArrayLike | None = None,
    destination_y_km: npt.ArrayLike | None = None,
    destination_shape: tuple[int, int] | None = None,
) -> ContextRegridOperator:
    """Build reusable separable regrid and source-confidence metadata.

    When destination axes are omitted, uniform axes retain each source axis's
    first/last centre.  ``destination_shape`` is ``(y, x)`` and defaults to the
    source shape (the v1.1.3b NON_UNI 256 x 256 -> uniform 256 x 256 case).
    """

    source_x = _centres_1d(source_x_km, name="source_x_km")
    source_y = _centres_1d(source_y_km, name="source_y_km")
    if destination_shape is not None:
        if len(destination_shape) != 2:
            raise ValueError("destination_shape must be a (y, x) pair")
        destination_height, destination_width = map(int, destination_shape)
        if destination_height < 2 or destination_width < 2:
            raise ValueError("destination dimensions must each be at least two")
    else:
        destination_height, destination_width = source_y.size, source_x.size

    if destination_x_km is None:
        destination_x = uniform_centers_like(source_x, count=destination_width)
    else:
        destination_x = _centres_1d(
            destination_x_km, name="destination_x_km"
        )
        if destination_shape is not None and destination_x.size != destination_width:
            raise ValueError("destination_x_km length disagrees with destination_shape")
    if destination_y_km is None:
        destination_y = uniform_centers_like(source_y, count=destination_height)
    else:
        destination_y = _centres_1d(
            destination_y_km, name="destination_y_km"
        )
        if destination_shape is not None and destination_y.size != destination_height:
            raise ValueError("destination_y_km length disagrees with destination_shape")

    weights_x = build_area_integrated_weights(source_x, destination_x)
    weights_y = build_area_integrated_weights(source_y, destination_y)
    nearest_x = _nearest_indices(source_x, destination_x)
    nearest_y = _nearest_indices(source_y, destination_y)

    source_spacing_x = _source_spacing_at_centres(source_x)
    source_spacing_y = _source_spacing_at_centres(source_y)
    destination_source_dx = np.interp(destination_x, source_x, source_spacing_x)
    destination_source_dy = np.interp(destination_y, source_y, source_spacing_y)
    source_dx = np.broadcast_to(
        destination_source_dx[None, :],
        (destination_y.size, destination_x.size),
    ).copy()
    source_dy = np.broadcast_to(
        destination_source_dy[:, None],
        (destination_y.size, destination_x.size),
    ).copy()

    nearest_delta_x = np.abs(destination_x - source_x[nearest_x])
    nearest_delta_y = np.abs(destination_y - source_y[nearest_y])
    distance = np.hypot(nearest_delta_y[:, None], nearest_delta_x[None, :])
    # A dimensionless, bounded confidence with a clear physical scale: one
    # local source-cell diagonal reduces confidence to 1/2, while a source
    # sample itself has confidence 1.  Dynamic validity is multiplied later.
    local_diagonal = np.hypot(source_dy, source_dx)
    geometry_confidence = 1.0 / (1.0 + np.square(distance / local_diagonal))

    return ContextRegridOperator(
        source_x_km=source_x.copy(),
        source_y_km=source_y.copy(),
        destination_x_km=destination_x.copy(),
        destination_y_km=destination_y.copy(),
        weights_x=weights_x,
        weights_y=weights_y,
        nearest_x_indices=nearest_x,
        nearest_y_indices=nearest_y,
        source_dx_km=source_dx,
        source_dy_km=source_dy,
        nearest_source_distance_km=distance,
        geometry_confidence=geometry_confidence,
        _x_supports=_axis_supports(weights_x),
        _y_supports=_axis_supports(weights_y),
    )


def regrid_context(
    rain_rate_mm_per_hour: npt.ArrayLike,
    source_x_km: npt.ArrayLike,
    source_y_km: npt.ArrayLike,
    **kwargs: object,
) -> ContextRegridResult:
    """Convenience wrapper for one frame; reuse an operator for production."""

    operator_keys = {
        "destination_x_km",
        "destination_y_km",
        "destination_shape",
    }
    operator_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in operator_keys}
    operator = build_context_regrid_operator(
        source_x_km, source_y_km, **operator_kwargs
    )
    return operator.regrid(rain_rate_mm_per_hour, **kwargs)


__all__ = [
    "ContextRegridOperator",
    "ContextRegridResult",
    "DetailMode",
    "build_area_integrated_operator_1d",
    "build_area_integrated_weights",
    "build_axis_regrid_weights",
    "build_context_regrid_operator",
    "regrid_context",
    "uniform_centers_like",
]
