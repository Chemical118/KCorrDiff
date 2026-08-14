"""Full-width regridded context-radar encoder from architecture section 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch
from torch import Tensor, nn

from kcorrdiff.data.radar_values import (
    CPRECNET_RAIN_DB_MIN,
    CPRECNET_RAIN_DB_SPAN,
    CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR,
    CPRECNET_RAIN_RATE_MAX_RTOL,
)

from .common import (
    CONTEXT_WIDTHS,
    HISTORY_STEPS,
    PRODUCTION_INPUT_SIZE,
    FeaturePyramid,
    FiveLevelPyramidEncoder,
    StaticSpatialStem,
    TemporalSpatialStem,
    ValidityPyramid,
    build_validity_pyramid,
    require_bool_tensor,
    require_float32_tensor,
    require_module_float32,
    require_no_autocast,
    validate_input_size,
    validate_widths,
)


CONTEXT_DYNAMIC_CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "area_mean_rate_mm_per_hour",
    "detail_rate_mm_per_hour",
    "wet_mask",
    "valid_fraction",
    "interpolation_confidence",
)
CONTEXT_STATIC_CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "x_lcc_shared",
    "y_lcc_shared",
    "source_dx_km",
    "source_dy_km",
    "nearest_source_distance_km",
    "geometry_confidence",
)


@dataclass(frozen=True, slots=True)
class ContextEncoderInputs:
    """Issue-time-only named context tensors on the uniform 256² grid."""

    dynamic_fields: Tensor
    detail_validity: Tensor
    static_fields: Tensor
    dynamic_channel_names: tuple[str, ...] = CONTEXT_DYNAMIC_CHANNEL_NAMES
    static_channel_names: tuple[str, ...] = CONTEXT_STATIC_CHANNEL_NAMES

    def __post_init__(self) -> None:
        require_float32_tensor(
            "context dynamic_fields", self.dynamic_fields, ndim=5
        )
        require_bool_tensor(
            "context detail_validity", self.detail_validity, ndim=5
        )
        require_float32_tensor("context static_fields", self.static_fields, ndim=4)
        batch, steps, channels, height, width = self.dynamic_fields.shape
        if (steps, channels) != (HISTORY_STEPS, 5):
            raise ValueError("dynamic_fields must have shape [B,12,5,H,W]")
        if tuple(self.detail_validity.shape) != (
            batch,
            HISTORY_STEPS,
            1,
            height,
            width,
        ):
            raise ValueError("detail_validity must have shape [B,12,1,H,W]")
        if tuple(self.static_fields.shape) != (batch, 6, height, width):
            raise ValueError("context static_fields must have shape [B,6,H,W]")
        if tuple(self.dynamic_channel_names) != CONTEXT_DYNAMIC_CHANNEL_NAMES:
            raise ValueError(
                "context dynamic channel names/order must match the named schema"
            )
        if tuple(self.static_channel_names) != CONTEXT_STATIC_CHANNEL_NAMES:
            raise ValueError(
                "context static channel names/order must match the named schema"
            )
        tensors = self.dynamic_fields, self.detail_validity, self.static_fields
        if any(tensor.device != self.dynamic_fields.device for tensor in tensors):
            raise ValueError("all context encoder inputs must share one device")
        rain_rate = self.dynamic_fields[:, :, :2]
        if bool((rain_rate < 0.0).any().item()):
            raise ValueError("context rain-rate channels cannot be negative")
        maximum_with_tolerance = CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR * (
            1.0 + CPRECNET_RAIN_RATE_MAX_RTOL
        )
        if bool((rain_rate > maximum_with_tolerance).any().item()):
            raise ValueError(
                "context rain-rate channels exceed the CPrecNet "
                "representable maximum"
            )
        bounded = self.dynamic_fields[:, :, 2:]
        if bool(((bounded < 0.0) | (bounded > 1.0)).any().item()):
            raise ValueError("wet/validity/confidence channels must lie in [0,1]")
        spacing = self.static_fields[:, 2:4]
        if bool((spacing <= 0.0).any().item()):
            raise ValueError("context source spacing must be positive")
        if bool((self.static_fields[:, 4] < 0.0).any().item()):
            raise ValueError("nearest source distance cannot be negative")
        confidence = self.static_fields[:, 5]
        if bool(((confidence < 0.0) | (confidence > 1.0)).any().item()):
            raise ValueError("geometry confidence must lie in [0,1]")


@dataclass(frozen=True, slots=True)
class ContextEncoderOutput:
    features: FeaturePyramid
    validity: ValidityPyramid
    temporal_level_zero: Tensor
    static_level_zero: Tensor

    @property
    def l0(self) -> Tensor:
        return self.features.l0

    @property
    def l1(self) -> Tensor:
        return self.features.l1

    @property
    def l2(self) -> Tensor:
        """Retained even though v1.1.3b does not fuse it by default."""

        return self.features.l2

    @property
    def l3(self) -> Tensor:
        return self.features.l3

    @property
    def l4(self) -> Tensor:
        return self.features.l4

    def detached(self) -> "ContextEncoderOutput":
        return ContextEncoderOutput(
            features=self.features.detached(),
            validity=self.validity.detached(),
            temporal_level_zero=self.temporal_level_zero.detach(),
            static_level_zero=self.static_level_zero.detach(),
        )


def cprecnet_rate_to_normalized(rate_mm_per_hour: Tensor) -> Tensor:
    """Re-apply the CPrecNet model-input transform to linear rain rates.

    Architecture section 4 regrids context radar in linear ``R(mm h-1)``
    space and then re-applies the model-input transform, so the encoder
    consumes the same normalized logarithmic space as the target radar
    history.  Rates at or below the archive's minimum representable rate map
    to exactly zero, mirroring the archive's below-threshold censoring.
    """

    positive = rate_mm_per_hour > 0.0
    rain_db = 10.0 * torch.log10(
        rate_mm_per_hour.clamp_min(torch.finfo(rate_mm_per_hour.dtype).tiny)
    )
    normalized = ((rain_db - CPRECNET_RAIN_DB_MIN) / CPRECNET_RAIN_DB_SPAN).clamp(
        0.0, 1.0
    )
    return torch.where(positive, normalized, torch.zeros_like(normalized))


class ContextEncoder(nn.Module):
    """Context temporal/confidence stems and all five cached pyramid levels."""

    def __init__(
        self,
        *,
        widths: Sequence[int] = CONTEXT_WIDTHS,
        input_size: int = PRODUCTION_INPUT_SIZE,
        activation_checkpoint: bool = False,
        allow_test_override: bool = False,
    ) -> None:
        super().__init__()
        self.widths = validate_widths(
            "context widths",
            widths,
            production=CONTEXT_WIDTHS,
            allow_test_override=allow_test_override,
        )
        self.input_size = validate_input_size(
            input_size, allow_test_override=allow_test_override
        )
        self.allow_test_override = bool(allow_test_override)
        stem_width = self.widths[0] // 2
        self.temporal_stem = TemporalSpatialStem(
            len(CONTEXT_DYNAMIC_CHANNEL_NAMES),
            stem_width,
            activation_checkpoint=activation_checkpoint,
        )
        self.static_stem = StaticSpatialStem(
            len(CONTEXT_STATIC_CHANNEL_NAMES),
            stem_width,
            activation_checkpoint=activation_checkpoint,
        )
        self.pyramid = FiveLevelPyramidEncoder(
            self.widths, activation_checkpoint=activation_checkpoint
        )

    def forward(self, inputs: ContextEncoderInputs) -> ContextEncoderOutput:
        require_no_autocast()
        require_module_float32("ContextEncoder", self)
        if not isinstance(inputs, ContextEncoderInputs):
            raise TypeError("ContextEncoder expects ContextEncoderInputs")
        if tuple(inputs.dynamic_fields.shape[-2:]) != (
            self.input_size,
            self.input_size,
        ):
            raise ValueError(
                f"context spatial shape must be {self.input_size}x{self.input_size}"
            )
        dynamic = inputs.dynamic_fields.clone()
        dynamic[:, :, 1:2] = torch.where(
            inputs.detail_validity, dynamic[:, :, 1:2], 0.0
        )
        # Rain-rate channels arrive in linear mm/h so the advection path can
        # alias the same batch tensor; only the encoder consumes them through
        # the CPrecNet normalized space.
        dynamic[:, :, :2] = cprecnet_rate_to_normalized(dynamic[:, :, :2])
        pooling_validity = dynamic[:, :, 3:4] > 0.0
        temporal = self.temporal_stem(
            dynamic, pooling_validity=pooling_validity
        )
        static = self.static_stem(inputs.static_fields)
        features = self.pyramid(torch.cat((temporal, static), dim=1))
        level_zero_validity = dynamic[:, :, 3:4].mean(dim=1)
        return ContextEncoderOutput(
            features=features,
            validity=build_validity_pyramid(level_zero_validity),
            temporal_level_zero=temporal,
            static_level_zero=static,
        )


__all__ = [
    "CONTEXT_DYNAMIC_CHANNEL_NAMES",
    "CONTEXT_STATIC_CHANNEL_NAMES",
    "ContextEncoder",
    "ContextEncoderInputs",
    "ContextEncoderOutput",
    "cprecnet_rate_to_normalized",
]
