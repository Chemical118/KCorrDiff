"""Strict, causal ``TrainingSample`` to Torch batch boundary.

The dataset deliberately keeps NumPy/PVC concerns outside the model.  This
module is the one production boundary that turns those samples into FP32
Torch conditions and a separate loss-only label object.  It does not infer
physical coordinates from tensor indices and it does not silently normalize
ERA fields: both geometry and an optional normalization artifact are explicit
constructor inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
import math
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from kcorrdiff.data.advection import (
    AdvectionGeometry,
    CausalAdvectionInput,
    causal_history_times,
)
from kcorrdiff.data.condition_schema import (
    CONTEXT_DYNAMIC_CHANNEL_NAMES,
    Era5Conditions,
    parse_condition_signature,
)
from kcorrdiff.data.context_regrid import ContextRegridOperator
from kcorrdiff.data.coordinates import (
    DEFAULT_COORDINATE_SCALE_KM,
    TokenGeometry,
    build_era_latlon_geometry,
    build_repeated_stride2_geometry,
    normalize_lcc_coordinates,
    target_center_from_axes,
)
from kcorrdiff.data.dataset import TrainingSample
from kcorrdiff.data.era5_reader import (
    ERA5_NATIVE_HOURS,
    Era5MmapCache,
    Era5WindowProvenance,
    native_hour_times,
    temporal_access_mask as expected_temporal_access_mask,
    trajectory_window_mask as expected_trajectory_window_mask,
)
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES
from kcorrdiff.data.radar_values import cprecnet_normalized_to_rain_rate
from kcorrdiff.data.static_loader import TargetStaticArtifact
from kcorrdiff.data.time_features import build_verification_time_features
from kcorrdiff.models.condition_bank import ConditionBankInputs
from kcorrdiff.models.context_encoder import (
    CONTEXT_STATIC_CHANNEL_NAMES,
    ContextEncoderInputs,
)
from kcorrdiff.models.embeddings import ConditionEmbeddingInputs
from kcorrdiff.models.physical_attention import PhysicalTokenGeometry
from kcorrdiff.models.regression import RegressionGeometry
from kcorrdiff.models.target_encoder import (
    TARGET_STATIC_CHANNEL_NAMES,
    TargetEncoderInputs,
)


EraValueSpace = Literal["physical_raw", "explicitly_normalized"]

_FUTURE_MODEL_FIELDS = frozenset(
    {
        "future",
        "future_target",
        "label",
        "labels",
        "m_target",
        "target_validity",
        "target_wet",
        "target_z",
        "training_label",
        "wet_label",
        "z_tau",
    }
)


def _as_readonly_axis(values: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    axis = np.array(values, dtype=np.float64, copy=True, order="C")
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with >=2 cells")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    axis.setflags(write=False)
    return axis


def _as_grid(values: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    grid = np.array(values, dtype=np.float64, copy=True, order="C")
    if grid.ndim not in (1, 2) or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must be a finite one- or two-dimensional grid")
    grid.setflags(write=False)
    return grid


@dataclass(frozen=True, slots=True)
class PhysicalGridSpec:
    """Actual source coordinates required to construct physical model tokens.

    Target and context axes are increasing KMA-LCC kilometres.  ERA
    coordinates use canonical south-to-north latitude and west-to-east
    longitude. Grid sizes are experiment choices; geometry remains explicit.
    """

    target_x_lcc_km: np.ndarray | Sequence[float]
    target_y_lcc_km: np.ndarray | Sequence[float]
    context_x_lcc_km: np.ndarray | Sequence[float]
    context_y_lcc_km: np.ndarray | Sequence[float]
    era_latitude_degrees: np.ndarray | Sequence[float]
    era_longitude_degrees: np.ndarray | Sequence[float]
    allow_test_override: bool = False

    def __post_init__(self) -> None:
        target_x = _as_readonly_axis(self.target_x_lcc_km, name="target_x_lcc_km")
        target_y = _as_readonly_axis(self.target_y_lcc_km, name="target_y_lcc_km")
        context_x = _as_readonly_axis(self.context_x_lcc_km, name="context_x_lcc_km")
        context_y = _as_readonly_axis(self.context_y_lcc_km, name="context_y_lcc_km")
        latitude = _as_grid(self.era_latitude_degrees, name="era_latitude_degrees")
        longitude = _as_grid(self.era_longitude_degrees, name="era_longitude_degrees")

        size = target_x.size
        if target_y.size != size or context_x.size != size or context_y.size != size:
            raise ValueError("target/context model grids must share one square size")
        if size % 16:
            raise ValueError("target/context grid size must be divisible by 16")
        if latitude.ndim == longitude.ndim == 1:
            if latitude.shape != (33,) or longitude.shape != (33,):
                raise ValueError("ERA axes must describe the native 33 x 33 grid")
            if not np.all(np.diff(latitude) > 0.0):
                raise ValueError("ERA latitude must be canonical south-to-north")
            if not np.all(np.diff(longitude) > 0.0):
                raise ValueError("ERA longitude must be canonical west-to-east")
        elif latitude.ndim == longitude.ndim == 2:
            if latitude.shape != (33, 33) or longitude.shape != (33, 33):
                raise ValueError("ERA coordinate grids must be exactly 33 x 33")
            if not np.all(np.diff(latitude, axis=0) > 0.0):
                raise ValueError("ERA latitude rows must run south-to-north")
            if not np.all(np.diff(longitude, axis=1) > 0.0):
                raise ValueError("ERA longitude columns must run west-to-east")
        else:
            raise ValueError("ERA latitude and longitude must use matching ranks")

        # This also freezes uniform-spacing and containment requirements used
        # by the physical RK2 advection kernel.
        AdvectionGeometry(
            target_x.copy(), target_y.copy(), context_x.copy(), context_y.copy()
        )
        object.__setattr__(self, "target_x_lcc_km", target_x)
        object.__setattr__(self, "target_y_lcc_km", target_y)
        object.__setattr__(self, "context_x_lcc_km", context_x)
        object.__setattr__(self, "context_y_lcc_km", context_y)
        object.__setattr__(self, "era_latitude_degrees", latitude)
        object.__setattr__(self, "era_longitude_degrees", longitude)

    @property
    def input_size(self) -> int:
        return int(np.asarray(self.target_x_lcc_km).size)

    @property
    def target_center_lcc_km(self) -> tuple[float, float]:
        return target_center_from_axes(self.target_x_lcc_km, self.target_y_lcc_km)

    @property
    def advection_geometry(self) -> AdvectionGeometry:
        return AdvectionGeometry(
            np.asarray(self.target_x_lcc_km).copy(),
            np.asarray(self.target_y_lcc_km).copy(),
            np.asarray(self.context_x_lcc_km).copy(),
            np.asarray(self.context_y_lcc_km).copy(),
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        target_static: TargetStaticArtifact,
        context_operator: ContextRegridOperator,
        era5_cache: Era5MmapCache,
        allow_test_override: bool = False,
    ) -> "PhysicalGridSpec":
        """Construct the production geometry from authoritative artifacts."""

        return cls(
            target_x_lcc_km=target_static.x_lcc_km,
            target_y_lcc_km=target_static.y_lcc_km,
            context_x_lcc_km=context_operator.destination_x_km,
            context_y_lcc_km=context_operator.destination_y_km,
            era_latitude_degrees=np.asarray(era5_cache.latitude),
            era_longitude_degrees=np.asarray(era5_cache.longitude),
            allow_test_override=allow_test_override,
        )


@runtime_checkable
class EraNormalization(Protocol):
    """Explicit train-fit ERA transformation injected into the collator."""

    @property
    def artifact_id(self) -> str:
        """Stable hash/version identifying the fitted normalization."""

    def normalize(
        self,
        values: np.ndarray,
        *,
        channel_names: tuple[str, ...],
        data_valid_inst: np.ndarray,
        tp_valid: np.ndarray,
        era_present: np.ndarray,
        tp_present: np.ndarray,
        provenance: tuple[object, ...],
    ) -> np.ndarray:
        """Return a new float32 array with the same ``[B,8,24,33,33]`` shape."""


def _torch_float(values: np.ndarray, *, device: torch.device) -> Tensor:
    array = np.asarray(values, dtype=np.float32, order="C")
    return torch.from_numpy(np.array(array, copy=True, order="C")).to(device=device)


def _torch_bool(values: np.ndarray, *, device: torch.device) -> Tensor:
    array = np.asarray(values, dtype=np.bool_, order="C")
    return torch.from_numpy(np.array(array, copy=True, order="C")).to(device=device)


def _move_geometry(
    geometry: PhysicalTokenGeometry, device: torch.device, *, non_blocking: bool
) -> PhysicalTokenGeometry:
    return PhysicalTokenGeometry(
        x_shared=geometry.x_shared.to(device, non_blocking=non_blocking),
        y_shared=geometry.y_shared.to(device, non_blocking=non_blocking),
        footprint_width=geometry.footprint_width.to(
            device, non_blocking=non_blocking
        ),
        footprint_height=geometry.footprint_height.to(
            device, non_blocking=non_blocking
        ),
    )


def _pin_geometry(geometry: PhysicalTokenGeometry) -> PhysicalTokenGeometry:
    return PhysicalTokenGeometry(
        x_shared=geometry.x_shared.pin_memory(),
        y_shared=geometry.y_shared.pin_memory(),
        footprint_width=geometry.footprint_width.pin_memory(),
        footprint_height=geometry.footprint_height.pin_memory(),
    )


def _physical_geometry(
    geometry: TokenGeometry, *, device: torch.device
) -> PhysicalTokenGeometry:
    return PhysicalTokenGeometry(
        x_shared=_torch_float(geometry.x_shared, device=device),
        y_shared=_torch_float(geometry.y_shared, device=device),
        footprint_width=_torch_float(geometry.footprint_width_km, device=device),
        footprint_height=_torch_float(geometry.footprint_height_km, device=device),
    )


def _regression_geometry(
    spec: PhysicalGridSpec, *, device: torch.device
) -> RegressionGeometry:
    center_x, center_y = spec.target_center_lcc_km
    target_l3 = build_repeated_stride2_geometry(
        spec.target_x_lcc_km,
        spec.target_y_lcc_km,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
        downsampling_levels=3,
    )
    target_l4 = build_repeated_stride2_geometry(
        spec.target_x_lcc_km,
        spec.target_y_lcc_km,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
        downsampling_levels=4,
    )
    context_l3 = build_repeated_stride2_geometry(
        spec.context_x_lcc_km,
        spec.context_y_lcc_km,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
        downsampling_levels=3,
    )
    context_l4 = build_repeated_stride2_geometry(
        spec.context_x_lcc_km,
        spec.context_y_lcc_km,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
        downsampling_levels=4,
    )
    era = build_era_latlon_geometry(
        spec.era_latitude_degrees,
        spec.era_longitude_degrees,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
    )
    return RegressionGeometry(
        target_l3=_physical_geometry(target_l3, device=device),
        target_l4=_physical_geometry(target_l4, device=device),
        context_l3=_physical_geometry(context_l3, device=device),
        context_l4=_physical_geometry(context_l4, device=device),
        era_native=_physical_geometry(era, device=device),
    )


@dataclass(frozen=True, slots=True)
class EraModelInputs:
    """Named ERA tensors, masks, signatures, and immutable source provenance."""

    values: Tensor
    delta_hours: Tensor
    tp_interval_center_delta_hours: Tensor
    data_valid_inst: Tensor
    tp_valid: Tensor
    trajectory_window_mask: Tensor
    temporal_access_mask: Tensor
    era_present: Tensor
    tp_present: Tensor
    valid_times_utc: tuple[tuple[datetime, ...], ...]
    tp_intervals_utc: tuple[tuple[tuple[datetime, datetime], ...], ...]
    condition_signatures: tuple[str, ...]
    provenance: tuple[object, ...]
    channel_names: tuple[str, ...] = ERA5_CHANNEL_NAMES
    value_space: EraValueSpace = "physical_raw"
    normalization_artifact_id: str | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or tuple(self.values.shape[1:3]) != (
            ERA5_NATIVE_HOURS,
            len(ERA5_CHANNEL_NAMES),
        ):
            raise ValueError("ERA values must have shape [B,8,24,H,W]")
        if tuple(self.values.shape[-2:]) != (33, 33):
            raise ValueError("ERA values must retain the native 33 x 33 grid")
        if self.values.dtype is not torch.float32 or not torch.isfinite(
            self.values
        ).all():
            raise TypeError("ERA values must be finite float32")
        batch = int(self.values.shape[0])
        for name, value in (
            ("delta_hours", self.delta_hours),
            ("tp_interval_center_delta_hours", self.tp_interval_center_delta_hours),
        ):
            if value.shape != (batch, 8) or value.dtype is not torch.float32:
                raise TypeError(f"ERA {name} must be float32 [B,8]")
            if value.device != self.values.device or not torch.isfinite(value).all():
                raise ValueError(f"ERA {name} must be finite on the ERA device")
        if not torch.allclose(
            self.tp_interval_center_delta_hours,
            self.delta_hours - 0.5,
            rtol=0.0,
            atol=2.0e-6,
        ):
            raise ValueError("ERA tp interval centres must equal delta_hours - 0.5")
        for name, value, shape in (
            ("data_valid_inst", self.data_valid_inst, (batch, 8)),
            ("tp_valid", self.tp_valid, (batch, 8)),
            ("trajectory_window_mask", self.trajectory_window_mask, (batch, 8)),
            ("temporal_access_mask", self.temporal_access_mask, (batch, 8)),
            ("era_present", self.era_present, (batch,)),
            ("tp_present", self.tp_present, (batch,)),
        ):
            if value.shape != shape or value.dtype is not torch.bool:
                raise TypeError(f"ERA {name} must be bool with shape {shape}")
            if value.device != self.values.device:
                raise ValueError(f"ERA {name} must share the ERA device")
        if torch.any(self.tp_present & ~self.era_present):
            raise ValueError("tp_present cannot be true without ERA")
        if tuple(self.channel_names) != ERA5_CHANNEL_NAMES:
            raise ValueError("ERA channel names/order do not match the 24-channel schema")
        metadata = (
            self.valid_times_utc,
            self.tp_intervals_utc,
            self.condition_signatures,
            self.provenance,
        )
        if any(len(value) != batch for value in metadata):
            raise ValueError("ERA metadata requires one entry per batch item")
        for index, raw_signature in enumerate(self.condition_signatures):
            signature = parse_condition_signature(raw_signature)
            if bool(self.era_present[index].item()) != signature.era_present or bool(
                self.tp_present[index].item()
            ) != (signature.era_present and signature.tp_present):
                raise ValueError("ERA source-presence flags mismatch condition signature")
            access = self.temporal_access_mask[index]
            trajectory = self.trajectory_window_mask[index]
            if signature.temporal_access_mode == "no_era_access":
                if bool(access.any().item()):
                    raise ValueError("no-era condition must have no temporal access")
            elif signature.temporal_access_mode == "full_trajectory":
                if not torch.equal(access, trajectory):
                    raise ValueError(
                        "full-trajectory condition must expose its trajectory window"
                    )
            elif bool((access & ~trajectory).any().item()):
                raise ValueError("causal temporal access must stay inside the trajectory")
            provenance = self.provenance[index]
            if not isinstance(provenance, Era5WindowProvenance):
                raise TypeError("ERA provenance must be Era5WindowProvenance")
            if provenance.condition_signature != signature.key:
                raise ValueError("ERA provenance mismatch condition signature")
            if (
                provenance.valid_times_utc != self.valid_times_utc[index]
                or provenance.tp_intervals_utc != self.tp_intervals_utc[index]
            ):
                raise ValueError("ERA provenance time metadata mismatch")
            if (
                signature.temporal_access_mode != "no_era_access"
                and provenance.access_mode != signature.temporal_access_mode
            ):
                raise ValueError("ERA provenance access mode mismatch signature")
        if self.value_space == "physical_raw":
            if self.normalization_artifact_id is not None:
                raise ValueError("raw ERA values cannot claim a normalization artifact")
        elif self.value_space == "explicitly_normalized":
            if not self.normalization_artifact_id:
                raise ValueError("normalized ERA values require an artifact id")
        else:
            raise ValueError(f"unsupported ERA value space: {self.value_space!r}")

    def validate(self) -> None:
        """Revalidate mutable tensor contents at a model/inference boundary."""

        self.__post_init__()

    @property
    def instantaneous(self) -> Tensor:
        return self.values[:, :, :23]

    @property
    def precipitation(self) -> Tensor:
        return self.values[:, :, 23:24]

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "EraModelInputs":
        selected = torch.device(device)
        return EraModelInputs(
            values=self.values.to(selected, non_blocking=non_blocking),
            delta_hours=self.delta_hours.to(selected, non_blocking=non_blocking),
            tp_interval_center_delta_hours=self.tp_interval_center_delta_hours.to(
                selected, non_blocking=non_blocking
            ),
            data_valid_inst=self.data_valid_inst.to(selected, non_blocking=non_blocking),
            tp_valid=self.tp_valid.to(selected, non_blocking=non_blocking),
            trajectory_window_mask=self.trajectory_window_mask.to(
                selected, non_blocking=non_blocking
            ),
            temporal_access_mask=self.temporal_access_mask.to(
                selected, non_blocking=non_blocking
            ),
            era_present=self.era_present.to(selected, non_blocking=non_blocking),
            tp_present=self.tp_present.to(selected, non_blocking=non_blocking),
            valid_times_utc=self.valid_times_utc,
            tp_intervals_utc=self.tp_intervals_utc,
            condition_signatures=self.condition_signatures,
            provenance=self.provenance,
            channel_names=self.channel_names,
            value_space=self.value_space,
            normalization_artifact_id=self.normalization_artifact_id,
        )

    def pin_memory(self) -> "EraModelInputs":
        return EraModelInputs(
            values=self.values.pin_memory(),
            delta_hours=self.delta_hours.pin_memory(),
            tp_interval_center_delta_hours=(
                self.tp_interval_center_delta_hours.pin_memory()
            ),
            data_valid_inst=self.data_valid_inst.pin_memory(),
            tp_valid=self.tp_valid.pin_memory(),
            trajectory_window_mask=self.trajectory_window_mask.pin_memory(),
            temporal_access_mask=self.temporal_access_mask.pin_memory(),
            era_present=self.era_present.pin_memory(),
            tp_present=self.tp_present.pin_memory(),
            valid_times_utc=self.valid_times_utc,
            tp_intervals_utc=self.tp_intervals_utc,
            condition_signatures=self.condition_signatures,
            provenance=self.provenance,
            channel_names=self.channel_names,
            value_space=self.value_space,
            normalization_artifact_id=self.normalization_artifact_id,
        )


@dataclass(frozen=True, slots=True)
class AdvectionModelInputs:
    """Physical causal histories and the requested half-hour lead index."""

    causal: CausalAdvectionInput
    geometry: AdvectionGeometry
    lead_indices: Tensor
    context_channels_alias_condition_bank: bool = False

    def __post_init__(self) -> None:
        if self.lead_indices.shape != (self.causal.batch_size,):
            raise ValueError("advection lead_indices must have shape [B]")
        if self.lead_indices.dtype is not torch.int64:
            raise TypeError("advection lead_indices must be int64")
        if self.lead_indices.device != self.causal.target_rate_mm_per_hour.device:
            raise ValueError("advection lead indices must share the input device")
        if not isinstance(self.context_channels_alias_condition_bank, bool):
            raise TypeError("context alias declaration must be boolean")
        if bool(((self.lead_indices < 0) | (self.lead_indices >= 12)).any().item()):
            raise ValueError("advection lead indices must lie in [0,11]")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "AdvectionModelInputs":
        selected = torch.device(device)
        causal = CausalAdvectionInput(
            t0_utc=self.causal.t0_utc,
            history_times_utc=self.causal.history_times_utc,
            target_rate_mm_per_hour=self.causal.target_rate_mm_per_hour.to(
                selected, non_blocking=non_blocking
            ),
            target_valid=self.causal.target_valid.to(
                selected, non_blocking=non_blocking
            ),
            context_rate_mm_per_hour=self.causal.context_rate_mm_per_hour.to(
                selected, non_blocking=non_blocking
            ),
            context_valid_fraction=self.causal.context_valid_fraction.to(
                selected, non_blocking=non_blocking
            ),
            context_interpolation_confidence=(
                self.causal.context_interpolation_confidence.to(
                    selected, non_blocking=non_blocking
                )
            ),
        )
        return AdvectionModelInputs(
            causal=causal,
            geometry=self.geometry,
            lead_indices=self.lead_indices.to(selected, non_blocking=non_blocking),
            context_channels_alias_condition_bank=(
                self.context_channels_alias_condition_bank
            ),
        )

    def pin_memory(self) -> "AdvectionModelInputs":
        causal = self.causal
        return AdvectionModelInputs(
            causal=CausalAdvectionInput(
                t0_utc=causal.t0_utc,
                history_times_utc=causal.history_times_utc,
                target_rate_mm_per_hour=(
                    causal.target_rate_mm_per_hour.pin_memory()
                ),
                target_valid=causal.target_valid.pin_memory(),
                context_rate_mm_per_hour=(
                    causal.context_rate_mm_per_hour.pin_memory()
                ),
                context_valid_fraction=(
                    causal.context_valid_fraction.pin_memory()
                ),
                context_interpolation_confidence=(
                    causal.context_interpolation_confidence.pin_memory()
                ),
            ),
            geometry=self.geometry,
            lead_indices=self.lead_indices.pin_memory(),
            context_channels_alias_condition_bank=(
                self.context_channels_alias_condition_bank
            ),
        )


@dataclass(frozen=True, slots=True)
class BatchProvenance:
    sample_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    fold_ids: tuple[int | None, ...]
    t0_utc: tuple[datetime, ...]
    history_times_utc: tuple[tuple[datetime, ...], ...]
    condition_signatures: tuple[str, ...]
    duplicate_condition_groups: tuple[tuple[int, ...], ...]
    explicit_history_copies: bool = True


@dataclass(frozen=True, slots=True)
class RegressionModelBatch:
    """Complete issue-time model boundary; no loss/future field can fit here."""

    condition_bank: ConditionBankInputs
    embedding: ConditionEmbeddingInputs
    era: EraModelInputs
    advection: AdvectionModelInputs
    geometry: RegressionGeometry
    provenance: BatchProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, ConditionEmbeddingInputs):
            raise TypeError("embedding must be ConditionEmbeddingInputs")
        self.embedding.validate()
        forbidden = sorted(
            {field.name.lower() for field in fields(self)} & _FUTURE_MODEL_FIELDS
        )
        if forbidden:
            raise ValueError("future-derived fields entered model batch: " + ", ".join(forbidden))
        batch = int(self.embedding.lead_hours.shape[0])
        if self.condition_bank.target.radar_history.shape[0] != batch:
            raise ValueError("condition bank and embedding batches differ")
        if self.era.values.shape[0] != batch or self.advection.causal.batch_size != batch:
            raise ValueError("ERA/advection/model batches differ")
        if len(self.provenance.sample_ids) != batch:
            raise ValueError("batch provenance length differs from model tensors")
        if any(
            len(value) != batch
            for value in (
                self.provenance.block_ids,
                self.provenance.fold_ids,
                self.provenance.t0_utc,
                self.provenance.history_times_utc,
                self.provenance.condition_signatures,
            )
        ):
            raise ValueError("all batch provenance fields require one item per row")
        if self.era.condition_signatures != self.provenance.condition_signatures:
            raise ValueError("ERA and batch condition signatures differ")
        for index, raw_signature in enumerate(self.era.condition_signatures):
            signature = parse_condition_signature(raw_signature)
            t0 = self.provenance.t0_utc[index]
            expected_cyclic = torch.tensor(
                build_verification_time_features(
                    t0, float(self.embedding.lead_hours[index].item())
                ).cyclic_features,
                dtype=torch.float32,
                device=self.embedding.verification_cyclic.device,
            )
            if not torch.equal(
                self.embedding.verification_cyclic[index], expected_cyclic
            ):
                raise ValueError(
                    "embedding verification_cyclic disagrees with canonical "
                    "FP32 t0/lead features"
                )
            expected_times = native_hour_times(t0)
            if self.era.valid_times_utc[index] != expected_times:
                raise ValueError("ERA valid times disagree with the batch issue time")
            expected_intervals = tuple(
                (value - timedelta(hours=1), value) for value in expected_times
            )
            if self.era.tp_intervals_utc[index] != expected_intervals:
                raise ValueError("ERA precipitation intervals disagree with valid times")
            expected_delta = torch.tensor(
                [
                    (value - t0.astimezone(UTC)).total_seconds() / 3600.0
                    for value in expected_times
                ],
                dtype=torch.float32,
                device=self.era.values.device,
            )
            if not torch.allclose(
                self.era.delta_hours[index], expected_delta, rtol=0.0, atol=1.0e-6
            ):
                raise ValueError("ERA time deltas disagree with the batch issue time")
            expected_trajectory = torch.as_tensor(
                expected_trajectory_window_mask(t0),
                dtype=torch.bool,
                device=self.era.values.device,
            )
            if not torch.equal(
                self.era.trajectory_window_mask[index], expected_trajectory
            ):
                raise ValueError("ERA trajectory mask disagrees with the issue time")
            if signature.temporal_access_mode == "no_era_access":
                expected_access = torch.zeros_like(expected_trajectory)
            else:
                expected_access = torch.as_tensor(
                    expected_temporal_access_mask(
                        t0,
                        lead_hours=float(self.embedding.lead_hours[index].item()),
                        access_mode=signature.temporal_access_mode,  # type: ignore[arg-type]
                    ),
                    dtype=torch.bool,
                    device=self.era.values.device,
                )
            if not torch.equal(self.era.temporal_access_mask[index], expected_access):
                raise ValueError("ERA temporal access disagrees with signature and lead")
        devices = (
            self.condition_bank.target.radar_history.device,
            self.condition_bank.context.dynamic_fields.device,
            self.embedding.lead_hours.device,
            self.era.values.device,
            self.advection.causal.target_rate_mm_per_hour.device,
            self.geometry.target_l3.x_shared.device,
        )
        if any(device != devices[0] for device in devices):
            raise ValueError("all model conditions must share one device")

    def validate(self) -> None:
        """Revalidate semantic condition identity after possible tensor mutation."""

        self.embedding.validate()
        self.era.validate()
        self.__post_init__()

    @property
    def device(self) -> torch.device:
        return self.embedding.lead_hours.device

    @property
    def batch_size(self) -> int:
        return int(self.embedding.lead_hours.shape[0])

    def select_rows(self, indices: Sequence[int]) -> "RegressionModelBatch":
        """Select model-only rows, retaining storage views for one-row cells."""

        rows = tuple(indices)
        if not rows:
            raise ValueError("row selection cannot be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= self.batch_size
            for index in rows
        ):
            raise IndexError("row selection contains an invalid batch index")
        index = torch.tensor(rows, dtype=torch.int64, device=self.device)

        def selected(value: Tensor) -> Tensor:
            if len(rows) == 1:
                return value[rows[0] : rows[0] + 1]
            return value.index_select(0, index)

        def geometry_rows(value: PhysicalTokenGeometry) -> PhysicalTokenGeometry:
            def field_rows(field: Tensor) -> Tensor:
                return selected(field) if field.ndim == 3 else field

            return PhysicalTokenGeometry(
                x_shared=field_rows(value.x_shared),
                y_shared=field_rows(value.y_shared),
                footprint_width=field_rows(value.footprint_width),
                footprint_height=field_rows(value.footprint_height),
            )

        target = self.condition_bank.target
        context = self.condition_bank.context
        context_dynamic = selected(context.dynamic_fields)
        bank = ConditionBankInputs(
            target=TargetEncoderInputs(
                radar_history=selected(target.radar_history),
                history_validity=selected(target.history_validity),
                static_fields=selected(target.static_fields),
                static_coverage=selected(target.static_coverage),
                static_channel_names=target.static_channel_names,
            ),
            context=ContextEncoderInputs(
                dynamic_fields=context_dynamic,
                detail_validity=selected(context.detail_validity),
                static_fields=selected(context.static_fields),
                dynamic_channel_names=context.dynamic_channel_names,
                static_channel_names=context.static_channel_names,
            ),
        )
        era = self.era
        era_rows = EraModelInputs(
            values=selected(era.values),
            delta_hours=selected(era.delta_hours),
            tp_interval_center_delta_hours=selected(
                era.tp_interval_center_delta_hours
            ),
            data_valid_inst=selected(era.data_valid_inst),
            tp_valid=selected(era.tp_valid),
            trajectory_window_mask=selected(era.trajectory_window_mask),
            temporal_access_mask=selected(era.temporal_access_mask),
            era_present=selected(era.era_present),
            tp_present=selected(era.tp_present),
            valid_times_utc=tuple(era.valid_times_utc[row] for row in rows),
            tp_intervals_utc=tuple(era.tp_intervals_utc[row] for row in rows),
            condition_signatures=tuple(
                era.condition_signatures[row] for row in rows
            ),
            provenance=tuple(era.provenance[row] for row in rows),
            channel_names=era.channel_names,
            value_space=era.value_space,
            normalization_artifact_id=era.normalization_artifact_id,
        )
        causal = self.advection.causal
        if self.advection.context_channels_alias_condition_bank:
            context_rate = context_dynamic[:, :, 0]
            context_valid = context_dynamic[:, :, 3]
            context_confidence = context_dynamic[:, :, 4]
        else:
            context_rate = selected(causal.context_rate_mm_per_hour)
            context_valid = selected(causal.context_valid_fraction)
            context_confidence = selected(causal.context_interpolation_confidence)
        selected_t0 = tuple(causal.t0_utc[row] for row in rows)
        selected_history = tuple(causal.history_times_utc[row] for row in rows)
        selected_signatures = tuple(
            self.provenance.condition_signatures[row] for row in rows
        )
        groups: dict[tuple[datetime, str], list[int]] = {}
        for new_index, key in enumerate(zip(selected_t0, selected_signatures, strict=True)):
            groups.setdefault(key, []).append(new_index)
        duplicate_groups = tuple(
            tuple(group) for group in groups.values() if len(group) > 1
        )
        return RegressionModelBatch(
            condition_bank=bank,
            embedding=ConditionEmbeddingInputs(
                lead_hours=selected(self.embedding.lead_hours),
                verification_cyclic=selected(self.embedding.verification_cyclic),
            ),
            era=era_rows,
            advection=AdvectionModelInputs(
                causal=CausalAdvectionInput(
                    t0_utc=selected_t0,
                    history_times_utc=selected_history,
                    target_rate_mm_per_hour=selected(
                        causal.target_rate_mm_per_hour
                    ),
                    target_valid=selected(causal.target_valid),
                    context_rate_mm_per_hour=context_rate,
                    context_valid_fraction=context_valid,
                    context_interpolation_confidence=context_confidence,
                ),
                geometry=self.advection.geometry,
                lead_indices=selected(self.advection.lead_indices),
                context_channels_alias_condition_bank=(
                    self.advection.context_channels_alias_condition_bank
                ),
            ),
            geometry=RegressionGeometry(
                target_l3=geometry_rows(self.geometry.target_l3),
                target_l4=geometry_rows(self.geometry.target_l4),
                context_l3=geometry_rows(self.geometry.context_l3),
                context_l4=geometry_rows(self.geometry.context_l4),
                era_native=geometry_rows(self.geometry.era_native),
            ),
            provenance=BatchProvenance(
                sample_ids=tuple(self.provenance.sample_ids[row] for row in rows),
                block_ids=tuple(self.provenance.block_ids[row] for row in rows),
                fold_ids=tuple(self.provenance.fold_ids[row] for row in rows),
                t0_utc=tuple(self.provenance.t0_utc[row] for row in rows),
                history_times_utc=tuple(
                    self.provenance.history_times_utc[row] for row in rows
                ),
                condition_signatures=selected_signatures,
                duplicate_condition_groups=duplicate_groups,
                explicit_history_copies=self.provenance.explicit_history_copies,
            ),
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "RegressionModelBatch":
        selected = torch.device(device)
        target = self.condition_bank.target
        context = self.condition_bank.context
        context_dynamic = context.dynamic_fields.to(
            selected, non_blocking=non_blocking
        )
        bank = ConditionBankInputs(
            target=TargetEncoderInputs(
                radar_history=target.radar_history.to(selected, non_blocking=non_blocking),
                history_validity=target.history_validity.to(selected, non_blocking=non_blocking),
                static_fields=target.static_fields.to(selected, non_blocking=non_blocking),
                static_coverage=target.static_coverage.to(selected, non_blocking=non_blocking),
                static_channel_names=target.static_channel_names,
            ),
            context=ContextEncoderInputs(
                dynamic_fields=context_dynamic,
                detail_validity=context.detail_validity.to(selected, non_blocking=non_blocking),
                static_fields=context.static_fields.to(selected, non_blocking=non_blocking),
                dynamic_channel_names=context.dynamic_channel_names,
                static_channel_names=context.static_channel_names,
            ),
        )
        original_causal = self.advection.causal
        if self.advection.context_channels_alias_condition_bank:
            context_rate = context_dynamic[:, :, 0]
            context_valid_fraction = context_dynamic[:, :, 3]
            context_confidence = context_dynamic[:, :, 4]
        else:
            context_rate = original_causal.context_rate_mm_per_hour.to(
                selected, non_blocking=non_blocking
            )
            context_valid_fraction = original_causal.context_valid_fraction.to(
                selected, non_blocking=non_blocking
            )
            context_confidence = (
                original_causal.context_interpolation_confidence.to(
                    selected, non_blocking=non_blocking
                )
            )
        moved_causal = CausalAdvectionInput(
            t0_utc=original_causal.t0_utc,
            history_times_utc=original_causal.history_times_utc,
            target_rate_mm_per_hour=original_causal.target_rate_mm_per_hour.to(
                selected, non_blocking=non_blocking
            ),
            target_valid=original_causal.target_valid.to(
                selected, non_blocking=non_blocking
            ),
            context_rate_mm_per_hour=context_rate,
            context_valid_fraction=context_valid_fraction,
            context_interpolation_confidence=context_confidence,
        )
        geometry = RegressionGeometry(
            target_l3=_move_geometry(self.geometry.target_l3, selected, non_blocking=non_blocking),
            target_l4=_move_geometry(self.geometry.target_l4, selected, non_blocking=non_blocking),
            context_l3=_move_geometry(self.geometry.context_l3, selected, non_blocking=non_blocking),
            context_l4=_move_geometry(self.geometry.context_l4, selected, non_blocking=non_blocking),
            era_native=_move_geometry(self.geometry.era_native, selected, non_blocking=non_blocking),
        )
        return RegressionModelBatch(
            condition_bank=bank,
            embedding=ConditionEmbeddingInputs(
                lead_hours=self.embedding.lead_hours.to(selected, non_blocking=non_blocking),
                verification_cyclic=self.embedding.verification_cyclic.to(
                    selected, non_blocking=non_blocking
                ),
            ),
            era=self.era.to(selected, non_blocking=non_blocking),
            advection=AdvectionModelInputs(
                causal=moved_causal,
                geometry=self.advection.geometry,
                lead_indices=self.advection.lead_indices.to(
                    selected, non_blocking=non_blocking
                ),
                context_channels_alias_condition_bank=(
                    self.advection.context_channels_alias_condition_bank
                ),
            ),
            geometry=geometry,
            provenance=self.provenance,
        )

    def pin_memory(self) -> "RegressionModelBatch":
        """Pin once per storage and reconstruct documented context aliases."""

        target = self.condition_bank.target
        context = self.condition_bank.context
        pinned_context_dynamic = context.dynamic_fields.pin_memory()
        bank = ConditionBankInputs(
            target=TargetEncoderInputs(
                radar_history=target.radar_history.pin_memory(),
                history_validity=target.history_validity.pin_memory(),
                static_fields=target.static_fields.pin_memory(),
                static_coverage=target.static_coverage.pin_memory(),
                static_channel_names=target.static_channel_names,
            ),
            context=ContextEncoderInputs(
                dynamic_fields=pinned_context_dynamic,
                detail_validity=context.detail_validity.pin_memory(),
                static_fields=context.static_fields.pin_memory(),
                dynamic_channel_names=context.dynamic_channel_names,
                static_channel_names=context.static_channel_names,
            ),
        )
        causal = self.advection.causal
        if self.advection.context_channels_alias_condition_bank:
            context_rate = pinned_context_dynamic[:, :, 0]
            context_valid_fraction = pinned_context_dynamic[:, :, 3]
            context_confidence = pinned_context_dynamic[:, :, 4]
        else:
            context_rate = causal.context_rate_mm_per_hour.pin_memory()
            context_valid_fraction = causal.context_valid_fraction.pin_memory()
            context_confidence = (
                causal.context_interpolation_confidence.pin_memory()
            )
        pinned_causal = CausalAdvectionInput(
            t0_utc=causal.t0_utc,
            history_times_utc=causal.history_times_utc,
            target_rate_mm_per_hour=causal.target_rate_mm_per_hour.pin_memory(),
            target_valid=causal.target_valid.pin_memory(),
            context_rate_mm_per_hour=context_rate,
            context_valid_fraction=context_valid_fraction,
            context_interpolation_confidence=context_confidence,
        )
        return RegressionModelBatch(
            condition_bank=bank,
            embedding=ConditionEmbeddingInputs(
                lead_hours=self.embedding.lead_hours.pin_memory(),
                verification_cyclic=self.embedding.verification_cyclic.pin_memory(),
            ),
            era=self.era.pin_memory(),
            advection=AdvectionModelInputs(
                causal=pinned_causal,
                geometry=self.advection.geometry,
                lead_indices=self.advection.lead_indices.pin_memory(),
                context_channels_alias_condition_bank=(
                    self.advection.context_channels_alias_condition_bank
                ),
            ),
            geometry=RegressionGeometry(
                target_l3=_pin_geometry(self.geometry.target_l3),
                target_l4=_pin_geometry(self.geometry.target_l4),
                context_l3=_pin_geometry(self.geometry.context_l3),
                context_l4=_pin_geometry(self.geometry.context_l4),
                era_native=_pin_geometry(self.geometry.era_native),
            ),
            provenance=self.provenance,
        )


@dataclass(frozen=True, slots=True)
class LossOnlyLabels:
    """Future-derived tensors accepted by losses, never by model forward."""

    target_z: Tensor
    target_wet: Tensor
    raw_target_mm: Tensor
    model_target_mm: Tensor
    target_validity: Tensor
    omega: Tensor

    def __post_init__(self) -> None:
        if self.target_z.ndim != 4 or self.target_z.shape[1] != 1:
            raise ValueError("loss target fields must have shape [B,1,H,W]")
        shape = self.target_z.shape
        for name, value in (
            ("raw_target_mm", self.raw_target_mm),
            ("model_target_mm", self.model_target_mm),
        ):
            if value.shape != shape or value.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32 and match target_z")
        if self.target_z.dtype is not torch.float32:
            raise TypeError("target_z must be float32")
        for name, value in (
            ("target_wet", self.target_wet),
            ("target_validity", self.target_validity),
        ):
            if value.shape != shape or value.dtype is not torch.bool:
                raise TypeError(f"{name} must be bool and match target_z")
        if self.omega.shape != (shape[0],) or self.omega.dtype is not torch.float32:
            raise TypeError("omega must be float32 [B]")
        tensors = (
            self.target_z,
            self.target_wet,
            self.raw_target_mm,
            self.model_target_mm,
            self.target_validity,
            self.omega,
        )
        if any(value.device != self.target_z.device for value in tensors):
            raise ValueError("all loss tensors must share one device")
        if not torch.isfinite(self.omega).all() or torch.any(self.omega <= 0.0):
            raise ValueError("omega must be finite and strictly positive")
        valid = self.target_validity
        for name, value in (
            ("target_z", self.target_z),
            ("raw_target_mm", self.raw_target_mm),
            ("model_target_mm", self.model_target_mm),
        ):
            if not torch.isfinite(value[valid]).all() or torch.any(value[valid] < 0.0):
                raise ValueError(f"valid {name} values must be finite and non-negative")

    def hurdle_kwargs(self) -> dict[str, Tensor]:
        return {
            "target_z": self.target_z,
            "target_wet": self.target_wet,
            "target_validity": self.target_validity,
            "omega": self.omega,
        }

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "LossOnlyLabels":
        selected = torch.device(device)
        return LossOnlyLabels(
            target_z=self.target_z.to(selected, non_blocking=non_blocking),
            target_wet=self.target_wet.to(selected, non_blocking=non_blocking),
            raw_target_mm=self.raw_target_mm.to(selected, non_blocking=non_blocking),
            model_target_mm=self.model_target_mm.to(selected, non_blocking=non_blocking),
            target_validity=self.target_validity.to(selected, non_blocking=non_blocking),
            omega=self.omega.to(selected, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "LossOnlyLabels":
        return LossOnlyLabels(
            target_z=self.target_z.pin_memory(),
            target_wet=self.target_wet.pin_memory(),
            raw_target_mm=self.raw_target_mm.pin_memory(),
            model_target_mm=self.model_target_mm.pin_memory(),
            target_validity=self.target_validity.pin_memory(),
            omega=self.omega.pin_memory(),
        )


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """A model-only object plus a structurally separate loss-only object."""

    model: RegressionModelBatch
    labels: LossOnlyLabels

    def __post_init__(self) -> None:
        if self.model.batch_size != self.labels.target_z.shape[0]:
            raise ValueError("model and label batch sizes differ")
        if self.model.device != self.labels.target_z.device:
            raise ValueError("model and labels must share one device")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "TrainingBatch":
        return TrainingBatch(
            model=self.model.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "TrainingBatch":
        return TrainingBatch(
            model=self.model.pin_memory(),
            labels=self.labels.pin_memory(),
        )


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def _validate_duplicate_conditions(samples: Sequence[TrainingSample]) -> tuple[tuple[int, ...], ...]:
    groups: dict[tuple[datetime, str], list[int]] = {}
    for index, sample in enumerate(samples):
        t0 = sample.conditions.t0_utc.astimezone(UTC)
        key = (t0, sample.conditions.condition_signature)
        groups.setdefault(key, []).append(index)
    duplicates = tuple(tuple(indices) for indices in groups.values() if len(indices) > 1)
    for indices in duplicates:
        reference = samples[indices[0]].conditions
        for index in indices[1:]:
            candidate = samples[index].conditions
            named_arrays = (
                ("target_history", reference.target_history, candidate.target_history),
                ("history_validity", reference.history_validity, candidate.history_validity),
                ("context_mean", reference.context.mean_rate_mm_per_hour, candidate.context.mean_rate_mm_per_hour),
                ("context_detail", reference.context.detail_rate_mm_per_hour, candidate.context.detail_rate_mm_per_hour),
                ("context_wet", reference.context.wet_mask, candidate.context.wet_mask),
                ("context_valid", reference.context.valid_fraction, candidate.context.valid_fraction),
                ("context_confidence", reference.context.interpolation_confidence, candidate.context.interpolation_confidence),
                ("context_detail_valid", reference.context.detail_valid, candidate.context.detail_valid),
                ("target_static", reference.static.target.values, candidate.static.target.values),
                ("target_coverage", reference.static.target_static_coverage, candidate.static.target_static_coverage),
                ("era_values", reference.era5.values, candidate.era5.values),
                ("era_delta", reference.era5.delta_hours, candidate.era5.delta_hours),
                ("era_inst_valid", reference.era5.data_valid_inst, candidate.era5.data_valid_inst),
                ("era_tp_valid", reference.era5.tp_valid, candidate.era5.tp_valid),
                ("era_trajectory", reference.era5.trajectory_window_mask, candidate.era5.trajectory_window_mask),
            )
            mismatch = [name for name, left, right in named_arrays if not _same_array(left, right)]
            if mismatch:
                raise ValueError(
                    "duplicate t0/signature histories disagree for "
                    f"batch rows {indices[0]} and {index}: {', '.join(mismatch)}"
                )
            if (
                reference.context.detail_mode != candidate.context.detail_mode
                or reference.static.target.channel_names != candidate.static.target.channel_names
                or reference.era5.valid_times_utc != candidate.era5.valid_times_utc
                or reference.era5.era_present != candidate.era5.era_present
                or reference.era5.tp_present != candidate.era5.tp_present
            ):
                raise ValueError("duplicate t0/signature condition metadata disagree")
    return duplicates


def _validate_sample(sample: TrainingSample, spec: PhysicalGridSpec) -> None:
    conditions = sample.conditions
    size = spec.input_size
    if conditions.t0_utc.tzinfo is None or conditions.t0_utc.utcoffset() is None:
        raise ValueError("sample t0 must be timezone-aware")
    if conditions.t0_utc.minute % 5 or conditions.t0_utc.second or conditions.t0_utc.microsecond:
        raise ValueError("sample t0 must lie on a five-minute slot")
    expected_shape = (12, size, size)
    if conditions.target_history.shape != expected_shape:
        raise ValueError(f"target history must have shape {expected_shape}")
    if conditions.history_validity.shape != expected_shape:
        raise ValueError("target history validity must match target history")
    target = np.asarray(conditions.target_history)
    if target.dtype != np.float32 or not np.all(np.isfinite(target)):
        raise TypeError("target history must be finite archive float32")
    if np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("normalized target history must lie in [0,1]")
    if np.asarray(conditions.history_validity).dtype != np.bool_:
        raise TypeError("target history validity must be boolean")
    if tuple(conditions.static.target.channel_names) != TARGET_STATIC_CHANNEL_NAMES:
        raise ValueError("target static channels must match the production schema")
    if conditions.static.target.values.shape != (7, size, size):
        raise ValueError("target static fields must have shape [7,H,W]")
    if conditions.static.context.source_dx_km.shape != (size, size):
        raise ValueError("context static fields must match the model grid")

    center_x, center_y = spec.target_center_lcc_km
    x_grid, y_grid = np.meshgrid(
        spec.target_x_lcc_km, spec.target_y_lcc_km, indexing="xy"
    )
    expected_x, expected_y = normalize_lcc_coordinates(
        x_grid,
        y_grid,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
        scale_km=DEFAULT_COORDINATE_SCALE_KM,
    )
    if not np.allclose(
        conditions.static.target.channel("x_lcc_shared"), expected_x, rtol=0.0, atol=2.0e-5
    ) or not np.allclose(
        conditions.static.target.channel("y_lcc_shared"), expected_y, rtol=0.0, atol=2.0e-5
    ):
        raise ValueError("target static coordinates disagree with actual LCC axes")

    context = conditions.context
    context_arrays = (
        context.mean_rate_mm_per_hour,
        context.detail_rate_mm_per_hour,
        context.wet_mask,
        context.valid_fraction,
        context.interpolation_confidence,
        context.detail_valid,
    )
    if any(value.shape != expected_shape for value in context_arrays):
        raise ValueError("context dynamic fields must share [12,H,W]")
    era = conditions.era5
    if era.values.shape != (8, 24, 33, 33):
        raise ValueError("ERA conditions must retain [8,24,33,33]")
    if era.condition_signature != conditions.condition_signature:
        raise ValueError("ERA and radar condition signatures differ")
    if not math.isclose(era.lead_hours, conditions.lead_hours, abs_tol=1.0e-12):
        raise ValueError("ERA and radar condition leads differ")
    if era.provenance.condition_signature != conditions.condition_signature:
        raise ValueError("ERA provenance and condition signatures differ")
    expected_time = build_verification_time_features(
        conditions.t0_utc, conditions.lead_hours
    ).cyclic_features
    if not np.allclose(conditions.time_features, expected_time, rtol=0.0, atol=2.0e-7):
        raise ValueError("verification-time features do not match t0/lead")

    label = sample.label
    for name, value in (
        ("z", label.z),
        ("wet", label.wet),
        ("raw_accumulation_mm", label.raw_accumulation_mm),
        ("model_accumulation_mm", label.model_accumulation_mm),
        ("target_validity", label.target_validity),
    ):
        if np.asarray(value).shape != (size, size):
            raise ValueError(f"label {name} must match the target grid")
    if np.asarray(label.wet).dtype != np.bool_ or np.asarray(label.target_validity).dtype != np.bool_:
        raise TypeError("wet and target validity labels must be boolean")
    if not math.isfinite(float(label.omega)) or float(label.omega) <= 0.0:
        raise ValueError("sample importance weight must be finite and positive")


class TrainingBatchCollator:
    """Pickle-safe production collator with explicit geometry/normalization."""

    def __init__(
        self,
        geometry: PhysicalGridSpec,
        *,
        era_normalization: EraNormalization | None = None,
        device: torch.device | str = "cpu",
        validate_duplicate_histories: bool = True,
        allow_test_override: bool = False,
    ) -> None:
        if not isinstance(geometry, PhysicalGridSpec):
            raise TypeError("geometry must be a PhysicalGridSpec")
        if era_normalization is not None:
            artifact_id = getattr(era_normalization, "artifact_id", None)
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("ERA normalization requires a non-empty artifact_id")
            if not callable(getattr(era_normalization, "normalize", None)):
                raise TypeError("ERA normalization must implement normalize()")
        self.geometry = geometry
        self.era_normalization = era_normalization
        self.device = torch.device(device)
        self.validate_duplicate_histories = bool(validate_duplicate_histories)
        self.allow_test_override = bool(allow_test_override)

    def __call__(self, samples: Sequence[TrainingSample]) -> TrainingBatch:
        values = tuple(samples)
        if not values:
            raise ValueError("cannot collate an empty TrainingSample batch")
        if any(not isinstance(sample, TrainingSample) for sample in values):
            raise TypeError("TrainingBatchCollator accepts only TrainingSample objects")
        for sample in values:
            _validate_sample(sample, self.geometry)
        duplicate_groups = (
            _validate_duplicate_conditions(values)
            if self.validate_duplicate_histories
            else ()
        )

        conditions = tuple(sample.conditions for sample in values)
        labels = tuple(sample.label for sample in values)
        batch = len(values)
        device = self.device
        target_normalized_np = np.stack(
            [condition.target_history for condition in conditions], axis=0
        ).astype(np.float32, copy=False)
        target_rate_np = cprecnet_normalized_to_rain_rate(
            target_normalized_np
        ).astype(np.float32, copy=False)
        target_valid_np = np.stack(
            [condition.history_validity for condition in conditions], axis=0
        ).astype(np.bool_, copy=False)
        target_static_np = np.stack(
            [condition.static.target.values for condition in conditions], axis=0
        ).astype(np.float32, copy=False)
        target_coverage_np = np.stack(
            [condition.static.target_static_coverage for condition in conditions], axis=0
        ).astype(np.bool_, copy=False)

        context_dynamic_np = np.stack(
            [condition.context.as_array(dtype=np.float32) for condition in conditions],
            axis=0,
        )
        context_detail_valid_np = np.stack(
            [condition.context.detail_valid for condition in conditions], axis=0
        ).astype(np.bool_, copy=False)
        center_x, center_y = self.geometry.target_center_lcc_km
        context_x, context_y = np.meshgrid(
            self.geometry.context_x_lcc_km,
            self.geometry.context_y_lcc_km,
            indexing="xy",
        )
        context_x_shared, context_y_shared = normalize_lcc_coordinates(
            context_x,
            context_y,
            target_center_x_km=center_x,
            target_center_y_km=center_y,
        )
        context_static_rows = []
        for condition in conditions:
            static = condition.static.context
            context_static_rows.append(
                np.stack(
                    (
                        context_x_shared,
                        context_y_shared,
                        static.source_dx_km,
                        static.source_dy_km,
                        static.nearest_source_distance_km,
                        static.geometry_confidence,
                    ),
                    axis=0,
                ).astype(np.float32, copy=False)
            )
        context_static_np = np.stack(context_static_rows, axis=0)

        raw_era_np = np.stack([condition.era5.values for condition in conditions], axis=0).astype(
            np.float32, copy=True
        )
        data_valid_inst_np = np.stack(
            [condition.era5.data_valid_inst for condition in conditions], axis=0
        ).astype(np.bool_, copy=False)
        tp_valid_np = np.stack(
            [condition.era5.tp_valid for condition in conditions], axis=0
        ).astype(np.bool_, copy=False)
        era_present_np = np.asarray(
            [condition.era5.era_present for condition in conditions], dtype=np.bool_
        )
        tp_present_np = np.asarray(
            [condition.era5.tp_present for condition in conditions], dtype=np.bool_
        )
        era_provenance = tuple(condition.era5.provenance for condition in conditions)
        if self.era_normalization is None:
            era_np = raw_era_np
            value_space: EraValueSpace = "physical_raw"
            normalization_id = None
        else:
            normalized = self.era_normalization.normalize(
                raw_era_np.copy(),
                channel_names=ERA5_CHANNEL_NAMES,
                data_valid_inst=data_valid_inst_np.copy(),
                tp_valid=tp_valid_np.copy(),
                era_present=era_present_np.copy(),
                tp_present=tp_present_np.copy(),
                provenance=era_provenance,
            )
            if not isinstance(normalized, np.ndarray):
                raise TypeError("ERA normalization must return a NumPy array")
            if normalized.shape != raw_era_np.shape or normalized.dtype != np.float32:
                raise TypeError("ERA normalization must preserve shape and return float32")
            if not np.all(np.isfinite(normalized)):
                raise ValueError("ERA normalization returned a non-finite value")
            era_np = np.array(normalized, dtype=np.float32, copy=True, order="C")
            value_space = "explicitly_normalized"
            normalization_id = self.era_normalization.artifact_id

        t0_utc = tuple(condition.t0_utc.astimezone(UTC) for condition in conditions)
        history_rows = tuple(causal_history_times(t0) for t0 in t0_utc)
        lead_hours_np = np.asarray(
            [condition.lead_hours for condition in conditions], dtype=np.float32
        )
        doubled = lead_hours_np.astype(np.float64) * 2.0
        if np.any(np.abs(doubled - np.rint(doubled)) > 1.0e-6):
            raise ValueError("lead hours must be exact 30-minute increments")
        lead_indices_np = np.rint(doubled).astype(np.int64) - 1
        signatures = tuple(condition.condition_signature for condition in conditions)

        target_inputs = TargetEncoderInputs(
            radar_history=_torch_float(target_normalized_np[:, :, None], device=device),
            history_validity=_torch_bool(target_valid_np[:, :, None], device=device),
            static_fields=_torch_float(target_static_np, device=device),
            static_coverage=_torch_bool(target_coverage_np[:, None], device=device),
            static_channel_names=TARGET_STATIC_CHANNEL_NAMES,
        )
        context_dynamic = _torch_float(context_dynamic_np, device=device)
        context_inputs = ContextEncoderInputs(
            dynamic_fields=context_dynamic,
            detail_validity=_torch_bool(context_detail_valid_np[:, :, None], device=device),
            static_fields=_torch_float(context_static_np, device=device),
            dynamic_channel_names=CONTEXT_DYNAMIC_CHANNEL_NAMES,
            static_channel_names=CONTEXT_STATIC_CHANNEL_NAMES,
        )
        causal = CausalAdvectionInput(
            t0_utc=t0_utc,
            history_times_utc=history_rows,
            target_rate_mm_per_hour=_torch_float(target_rate_np, device=device),
            target_valid=_torch_bool(target_valid_np, device=device),
            context_rate_mm_per_hour=context_dynamic[:, :, 0],
            context_valid_fraction=context_dynamic[:, :, 3],
            context_interpolation_confidence=context_dynamic[:, :, 4],
        )
        era_inputs = EraModelInputs(
            values=_torch_float(era_np, device=device),
            delta_hours=_torch_float(
                np.stack([condition.era5.delta_hours for condition in conditions]),
                device=device,
            ),
            tp_interval_center_delta_hours=_torch_float(
                np.stack(
                    [
                        condition.era5.tp_interval_center_delta_hours
                        for condition in conditions
                    ]
                ),
                device=device,
            ),
            data_valid_inst=_torch_bool(data_valid_inst_np, device=device),
            tp_valid=_torch_bool(tp_valid_np, device=device),
            trajectory_window_mask=_torch_bool(
                np.stack(
                    [condition.era5.trajectory_window_mask for condition in conditions]
                ),
                device=device,
            ),
            temporal_access_mask=_torch_bool(
                np.stack(
                    [condition.era5.temporal_access_mask for condition in conditions]
                ),
                device=device,
            ),
            era_present=_torch_bool(era_present_np, device=device),
            tp_present=_torch_bool(tp_present_np, device=device),
            valid_times_utc=tuple(condition.era5.valid_times_utc for condition in conditions),
            tp_intervals_utc=tuple(
                condition.era5.provenance.tp_intervals_utc for condition in conditions
            ),
            condition_signatures=signatures,
            provenance=era_provenance,
            value_space=value_space,
            normalization_artifact_id=normalization_id,
        )
        provenance = BatchProvenance(
            sample_ids=tuple(sample.sample_id for sample in values),
            block_ids=tuple(sample.block_id for sample in values),
            fold_ids=tuple(sample.fold_id for sample in values),
            t0_utc=t0_utc,
            history_times_utc=history_rows,
            condition_signatures=signatures,
            duplicate_condition_groups=duplicate_groups,
            explicit_history_copies=True,
        )
        model = RegressionModelBatch(
            condition_bank=ConditionBankInputs(
                target=target_inputs, context=context_inputs
            ),
            embedding=ConditionEmbeddingInputs(
                lead_hours=_torch_float(lead_hours_np, device=device),
                verification_cyclic=_torch_float(
                    np.asarray(
                        [condition.time_features for condition in conditions],
                        dtype=np.float32,
                    ),
                    device=device,
                ),
            ),
            era=era_inputs,
            advection=AdvectionModelInputs(
                causal=causal,
                geometry=self.geometry.advection_geometry,
                lead_indices=torch.from_numpy(lead_indices_np.copy()).to(device=device),
                context_channels_alias_condition_bank=True,
            ),
            geometry=_regression_geometry(self.geometry, device=device),
            provenance=provenance,
        )
        loss = LossOnlyLabels(
            target_z=_torch_float(
                np.stack([label.z for label in labels])[:, None], device=device
            ),
            target_wet=_torch_bool(
                np.stack([label.wet for label in labels])[:, None], device=device
            ),
            raw_target_mm=_torch_float(
                np.stack([label.raw_accumulation_mm for label in labels])[:, None],
                device=device,
            ),
            model_target_mm=_torch_float(
                np.stack([label.model_accumulation_mm for label in labels])[:, None],
                device=device,
            ),
            target_validity=_torch_bool(
                np.stack([label.target_validity for label in labels])[:, None],
                device=device,
            ),
            omega=_torch_float(
                np.asarray([label.omega for label in labels], dtype=np.float32),
                device=device,
            ),
        )
        if model.batch_size != batch:  # pragma: no cover - defensive invariant.
            raise AssertionError("collator changed the requested batch size")
        return TrainingBatch(model=model, labels=loss)


def collate_training_samples(
    samples: Sequence[TrainingSample],
    *,
    geometry: PhysicalGridSpec,
    era_normalization: EraNormalization | None = None,
    device: torch.device | str = "cpu",
    validate_duplicate_histories: bool = True,
    allow_test_override: bool = False,
) -> TrainingBatch:
    """Functional facade for DataLoader or one-off production conversion."""

    return TrainingBatchCollator(
        geometry,
        era_normalization=era_normalization,
        device=device,
        validate_duplicate_histories=validate_duplicate_histories,
        allow_test_override=allow_test_override,
    )(samples)


__all__ = [
    "AdvectionModelInputs",
    "BatchProvenance",
    "EraModelInputs",
    "EraNormalization",
    "LossOnlyLabels",
    "PhysicalGridSpec",
    "RegressionModelBatch",
    "TrainingBatch",
    "TrainingBatchCollator",
    "collate_training_samples",
]
