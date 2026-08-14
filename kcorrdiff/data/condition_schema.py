"""Named, issue-time-only condition schemas for :mod:`kcorrdiff.data.dataset`.

These dataclasses make the condition boundary explicit.  In particular, none
of them has a field for the future-derived target validity mask, future wet
label, or transformed target.  Those values belong exclusively to
``TrainingLabel`` in ``dataset.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

import numpy as np
import numpy.typing as npt

from .context_regrid import ContextRegridOperator, DetailMode
from .era5_reader import ERA5_NATIVE_HOURS, Era5Window, Era5WindowProvenance
from .provider_adapter import (
    ERA5_CHANNEL_NAMES,
    ConditionSignature,
)
from .radar_values import cprecnet_normalized_to_rain_rate


CONTEXT_DYNAMIC_CHANNEL_NAMES = (
    "area_mean_rate_mm_per_hour",
    "detail_rate_mm_per_hour",
    "wet_mask",
    "valid_fraction",
    "interpolation_confidence",
)

_FUTURE_ONLY_STATIC_NAMES = frozenset(
    {
        "future_target",
        "future_target_validity",
        "m_target",
        "m_target_tau",
        "target_label",
        "target_validity",
        "target_wet_label",
        "target_z",
        "training_label",
    }
)


def _canonical_channel_name(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", "_").split())


def _readonly_copy(
    values: npt.ArrayLike,
    *,
    dtype: npt.DTypeLike,
    name: str,
) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class NamedSpatialFields:
    """A channel-first static tensor with an explicit, immutable schema."""

    channel_names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        names = tuple(_canonical_channel_name(name) for name in self.channel_names)
        if not names or any(not name for name in names):
            raise ValueError("static channel names must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("static channel names must be unique")
        forbidden = sorted(set(names) & _FUTURE_ONLY_STATIC_NAMES)
        if forbidden:
            raise ValueError(
                "future-derived fields are forbidden in static conditions: "
                + ", ".join(forbidden)
            )
        values = _readonly_copy(
            self.values, dtype=np.float32, name="static spatial fields"
        )
        if values.ndim != 3:
            raise ValueError("static spatial fields must have shape [C,H,W]")
        if values.shape[0] != len(names):
            raise ValueError(
                "static channel_names length does not match the channel axis"
            )
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "values", values)

    def channel(self, name: str) -> np.ndarray:
        """Return one named ``[H,W]`` field without copying it."""

        canonical = _canonical_channel_name(name)
        try:
            index = self.channel_names.index(canonical)
        except ValueError as error:
            raise KeyError(f"unknown static channel: {name!r}") from error
        return self.values[index]


@dataclass(frozen=True, slots=True)
class ContextStaticConditions:
    """Source-density/confidence geometry fixed for a regrid operator."""

    source_dx_km: np.ndarray
    source_dy_km: np.ndarray
    nearest_source_distance_km: np.ndarray
    geometry_confidence: np.ndarray

    @classmethod
    def from_operator(
        cls, operator: ContextRegridOperator
    ) -> "ContextStaticConditions":
        return cls(
            source_dx_km=_readonly_copy(
                operator.source_dx_km, dtype=np.float32, name="context source dx"
            ),
            source_dy_km=_readonly_copy(
                operator.source_dy_km, dtype=np.float32, name="context source dy"
            ),
            nearest_source_distance_km=_readonly_copy(
                operator.nearest_source_distance_km,
                dtype=np.float32,
                name="nearest context source distance",
            ),
            geometry_confidence=_readonly_copy(
                operator.geometry_confidence,
                dtype=np.float32,
                name="context geometry confidence",
            ),
        )

    def __post_init__(self) -> None:
        arrays = (
            self.source_dx_km,
            self.source_dy_km,
            self.nearest_source_distance_km,
            self.geometry_confidence,
        )
        shape = arrays[0].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays):
            raise ValueError("context static fields must share one [H,W] shape")
        if np.any(self.source_dx_km <= 0.0) or np.any(self.source_dy_km <= 0.0):
            raise ValueError("context source spacing must be positive")
        if np.any(self.nearest_source_distance_km < 0.0):
            raise ValueError("nearest source distance cannot be negative")
        if np.any(
            (self.geometry_confidence < 0.0)
            | (self.geometry_confidence > 1.0)
        ):
            raise ValueError("context geometry confidence must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class StaticConditions:
    """Known-at-issue-time spatial fields; deliberately contains no labels."""

    target: NamedSpatialFields
    target_static_coverage: np.ndarray
    context: ContextStaticConditions

    def __post_init__(self) -> None:
        raw_coverage = np.asarray(self.target_static_coverage)
        if raw_coverage.dtype != np.bool_:
            if not np.issubdtype(raw_coverage.dtype, np.number) or not np.all(
                (raw_coverage == 0) | (raw_coverage == 1)
            ):
                raise ValueError(
                    "target static coverage must contain boolean or 0/1 values"
                )
        coverage = np.array(raw_coverage, dtype=np.bool_, copy=True, order="C")
        if coverage.shape != self.target.values.shape[1:]:
            raise ValueError(
                "target static coverage must match target static spatial shape"
            )
        coverage.setflags(write=False)
        object.__setattr__(self, "target_static_coverage", coverage)

    @classmethod
    def from_arrays(
        cls,
        *,
        target_values: npt.ArrayLike,
        target_static_coverage: npt.ArrayLike,
        context_regrid_operator: ContextRegridOperator,
        target_channel_names: Sequence[str] | None = None,
    ) -> "StaticConditions":
        values = np.asarray(target_values)
        if values.ndim != 3:
            raise ValueError("target_values must have shape [C,H,W]")
        names = (
            tuple(target_channel_names)
            if target_channel_names is not None
            else tuple(f"static_{index}" for index in range(values.shape[0]))
        )
        return cls(
            target=NamedSpatialFields(names, values),
            target_static_coverage=np.asarray(target_static_coverage),
            context=ContextStaticConditions.from_operator(
                context_regrid_operator
            ),
        )


@dataclass(frozen=True, slots=True)
class ContextRadarConditions:
    """Twelve regridded context frames with named dynamic channels."""

    mean_rate_mm_per_hour: np.ndarray
    detail_rate_mm_per_hour: np.ndarray
    wet_mask: np.ndarray
    valid_fraction: np.ndarray
    interpolation_confidence: np.ndarray
    detail_valid: np.ndarray
    detail_mode: DetailMode

    def __post_init__(self) -> None:
        arrays = (
            self.mean_rate_mm_per_hour,
            self.detail_rate_mm_per_hour,
            self.wet_mask,
            self.valid_fraction,
            self.interpolation_confidence,
            self.detail_valid,
        )
        shape = arrays[0].shape
        if len(shape) != 3 or any(array.shape != shape for array in arrays):
            raise ValueError("context dynamic fields must share shape [T,H,W]")
        if shape[0] != 12:
            raise ValueError("context radar conditions require exactly 12 frames")
        if np.any(self.mean_rate_mm_per_hour < 0.0) or np.any(
            self.detail_rate_mm_per_hour < 0.0
        ):
            raise ValueError("context rain rates cannot be negative")
        if np.any((self.valid_fraction < 0.0) | (self.valid_fraction > 1.0)):
            raise ValueError("context valid fraction must lie in [0, 1]")
        if np.any(
            (self.interpolation_confidence < 0.0)
            | (self.interpolation_confidence > 1.0)
        ):
            raise ValueError("context interpolation confidence must lie in [0, 1]")
        if self.detail_mode not in ("local_max", "nearest"):
            raise ValueError("unsupported context detail mode")

    @property
    def channel_names(self) -> tuple[str, ...]:
        return CONTEXT_DYNAMIC_CHANNEL_NAMES

    def as_array(self, *, dtype: npt.DTypeLike = np.float32) -> np.ndarray:
        """Return batch-facing ``[T,5,H,W]`` values in named channel order.

        Rate channels stay in linear ``mm h-1`` because the advection path
        aliases this tensor; ``ContextEncoder`` re-applies the CPrecNet
        model-input transform to them at its own boundary.
        """

        return np.stack(
            (
                self.mean_rate_mm_per_hour,
                self.detail_rate_mm_per_hour,
                self.wet_mask,
                self.valid_fraction,
                self.interpolation_confidence,
            ),
            axis=1,
        ).astype(dtype, copy=False)


def build_context_radar_conditions(
    normalized_history: npt.ArrayLike,
    *,
    operator: ContextRegridOperator,
    detail_mode: DetailMode = "local_max",
    wet_threshold_mm_per_hour: float = 0.0,
) -> ContextRadarConditions:
    """Convert CPrecNet history to linear R, then regrid every frame."""

    normalized = np.asarray(normalized_history)
    if normalized.shape != (12, *operator.source_shape):
        raise ValueError(
            "normalized context history must have shape "
            f"{(12, *operator.source_shape)}, got {normalized.shape}"
        )
    # Conversion happens once for the complete history and always precedes the
    # spatial operator.  CPrecNet has no dynamic validity field: non-finite or
    # out-of-range archive values are contract errors, not missing/dry pixels.
    linear_rate = cprecnet_normalized_to_rain_rate(normalized)
    valid = np.ones(operator.source_shape, dtype=np.bool_)
    frames = tuple(
        operator.regrid(
            frame,
            valid_mask=valid,
            detail_mode=detail_mode,
            wet_threshold_mm_per_hour=wet_threshold_mm_per_hour,
        )
        for frame in linear_rate
    )
    return ContextRadarConditions(
        mean_rate_mm_per_hour=np.stack(
            [frame.mean_rate_mm_per_hour for frame in frames]
        ).astype(np.float32),
        detail_rate_mm_per_hour=np.stack(
            [frame.detail_rate_mm_per_hour for frame in frames]
        ).astype(np.float32),
        wet_mask=np.stack([frame.wet_mask for frame in frames]),
        valid_fraction=np.stack(
            [frame.valid_fraction for frame in frames]
        ).astype(np.float32),
        interpolation_confidence=np.stack(
            [frame.interpolation_confidence for frame in frames]
        ).astype(np.float32),
        detail_valid=np.stack([frame.detail_valid for frame in frames]),
        detail_mode=detail_mode,
    )


def parse_condition_signature(
    value: str | ConditionSignature,
) -> ConditionSignature:
    """Parse one canonical ``provider:era=N:tp=N:access`` signature."""

    if isinstance(value, ConditionSignature):
        return value.canonical()
    if not isinstance(value, str):
        raise TypeError("condition signature must be a string or ConditionSignature")
    parts = value.rsplit(":", 3)
    if len(parts) != 4 or not parts[0]:
        raise ValueError(f"invalid condition signature: {value!r}")
    provider, era_part, tp_part, access = parts
    if access not in {
        "full_trajectory",
        "target_end_causal",
        "no_era_access",
    }:
        raise ValueError(
            f"invalid temporal access mode in condition signature: {access!r}"
        )

    def flag(part: str, name: str) -> bool:
        if part not in (f"{name}=0", f"{name}=1"):
            raise ValueError(f"invalid {name} flag in condition signature: {part!r}")
        return part.endswith("1")

    signature = ConditionSignature(
        provider_track=provider,
        era_present=flag(era_part, "era"),
        tp_present=flag(tp_part, "tp"),
        temporal_access_mode=access,  # type: ignore[arg-type]
    ).canonical()
    if signature.key != value:
        raise ValueError(
            f"condition signature is not canonical: {value!r}; use {signature.key!r}"
        )
    return signature


@dataclass(frozen=True, slots=True)
class Era5Conditions:
    """Canonical ERA window and all masks kept as separate model inputs."""

    values: np.ndarray
    valid_times_utc: tuple[datetime, ...]
    delta_hours: np.ndarray
    tp_interval_center_delta_hours: np.ndarray
    data_valid_inst: np.ndarray
    tp_valid: np.ndarray
    trajectory_window_mask: np.ndarray
    temporal_access_mask: np.ndarray
    era_present: bool
    tp_present: bool
    condition_signature: str
    lead_hours: float
    provenance: Era5WindowProvenance

    @property
    def channel_names(self) -> tuple[str, ...]:
        return ERA5_CHANNEL_NAMES

    @classmethod
    def from_window(
        cls,
        window: Era5Window,
        *,
        requested_signature: ConditionSignature,
        lead_hours: float,
    ) -> "Era5Conditions":
        signature = requested_signature.canonical()
        if not math.isfinite(lead_hours) or lead_hours <= 0.0:
            raise ValueError("lead_hours must be finite and positive")
        if window.provenance.condition_signature != signature.key:
            raise ValueError(
                "ERA window provenance/signature mismatch: "
                f"{window.provenance.condition_signature!r} != {signature.key!r}"
            )
        if window.era_present != signature.era_present or window.tp_present != (
            signature.era_present and signature.tp_present
        ):
            raise ValueError("ERA window source-presence flags mismatch signature")
        values = np.asarray(window.values, dtype=np.float32)
        if values.ndim != 4 or values.shape[:2] != (
            ERA5_NATIVE_HOURS,
            len(ERA5_CHANNEL_NAMES),
        ):
            raise ValueError(
                "ERA values must have canonical shape [8,24,H,W], got "
                f"{values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("ERA values contain non-finite values")
        mask_names = (
            "data_valid_inst",
            "tp_valid",
            "trajectory_window_mask",
            "temporal_access_mask",
        )
        masks: dict[str, np.ndarray] = {}
        for name in mask_names:
            mask = np.asarray(getattr(window, name), dtype=np.bool_)
            if mask.shape != (ERA5_NATIVE_HOURS,):
                raise ValueError(f"ERA {name} must have shape [8]")
            masks[name] = mask
        delta = np.asarray(window.delta_hours, dtype=np.float32)
        tp_delta = np.asarray(
            window.tp_interval_center_delta_hours, dtype=np.float32
        )
        if delta.shape != (ERA5_NATIVE_HOURS,) or tp_delta.shape != (
            ERA5_NATIVE_HOURS,
        ):
            raise ValueError("ERA time deltas must have shape [8]")
        if not np.all(np.isfinite(delta)) or not np.all(np.isfinite(tp_delta)):
            raise ValueError("ERA time deltas must be finite")
        if len(window.valid_times_utc) != ERA5_NATIVE_HOURS:
            raise ValueError("ERA valid times must contain exactly eight hours")
        return cls(
            values=values,
            valid_times_utc=tuple(window.valid_times_utc),
            delta_hours=delta,
            tp_interval_center_delta_hours=tp_delta,
            data_valid_inst=masks["data_valid_inst"],
            tp_valid=masks["tp_valid"],
            trajectory_window_mask=masks["trajectory_window_mask"],
            temporal_access_mask=masks["temporal_access_mask"],
            era_present=bool(window.era_present),
            tp_present=bool(window.tp_present),
            condition_signature=signature.key,
            lead_hours=float(lead_hours),
            provenance=window.provenance,
        )


__all__ = [
    "CONTEXT_DYNAMIC_CHANNEL_NAMES",
    "ContextRadarConditions",
    "ContextStaticConditions",
    "Era5Conditions",
    "NamedSpatialFields",
    "StaticConditions",
    "build_context_radar_conditions",
    "parse_condition_signature",
]
