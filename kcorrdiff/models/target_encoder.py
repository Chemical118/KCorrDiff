"""Full-width target-radar temporal/static encoder from architecture section 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import torch
from torch import Tensor, nn

from .common import (
    HISTORY_STEPS,
    PRODUCTION_INPUT_SIZE,
    TARGET_WIDTHS,
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


TARGET_STATIC_CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "dem_elevation",
    "terrain_slope_x",
    "terrain_slope_y",
    "local_terrain_relief",
    "land_sea_mask",
    "x_lcc_shared",
    "y_lcc_shared",
)


@dataclass(frozen=True, slots=True)
class TargetEncoderInputs:
    """Issue-time target tensors; future target/validity fields cannot fit here."""

    radar_history: Tensor
    history_validity: Tensor
    static_fields: Tensor
    static_coverage: Tensor
    static_channel_names: tuple[str, ...] = TARGET_STATIC_CHANNEL_NAMES

    def __post_init__(self) -> None:
        require_float32_tensor("target radar_history", self.radar_history, ndim=5)
        require_bool_tensor(
            "target history_validity", self.history_validity, ndim=5
        )
        require_float32_tensor("target static_fields", self.static_fields, ndim=4)
        require_bool_tensor("target static_coverage", self.static_coverage, ndim=4)
        batch, steps, channels, height, width = self.radar_history.shape
        if (steps, channels) != (HISTORY_STEPS, 1):
            raise ValueError("radar_history must have shape [B,12,1,H,W]")
        if tuple(self.history_validity.shape) != (
            batch,
            HISTORY_STEPS,
            1,
            height,
            width,
        ):
            raise ValueError("history_validity must match radar_history")
        if tuple(self.static_fields.shape) != (batch, 7, height, width):
            raise ValueError("static_fields must have shape [B,7,H,W]")
        if tuple(self.static_coverage.shape) != (batch, 1, height, width):
            raise ValueError("static_coverage must have shape [B,1,H,W]")
        if tuple(self.static_channel_names) != TARGET_STATIC_CHANNEL_NAMES:
            raise ValueError(
                "target static channel names/order must match the v1.1.3b schema"
            )
        tensors = (
            self.radar_history,
            self.history_validity,
            self.static_fields,
            self.static_coverage,
        )
        if any(tensor.device != self.radar_history.device for tensor in tensors):
            raise ValueError("all target encoder inputs must share one device")
        if bool(
            ((self.radar_history < 0.0) | (self.radar_history > 1.0))
            .any()
            .item()
        ):
            raise ValueError("CPrecNet target history must lie in [0,1]")
        outside_coverage = ~self.static_coverage[:, None]
        if bool((self.history_validity & outside_coverage).any().item()):
            raise ValueError("history cannot be valid outside static coverage")


@dataclass(frozen=True, slots=True)
class TargetEncoderOutput:
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
        return self.features.l2

    @property
    def l3(self) -> Tensor:
        return self.features.l3

    @property
    def l4(self) -> Tensor:
        return self.features.l4

    def detached(self) -> "TargetEncoderOutput":
        return TargetEncoderOutput(
            features=self.features.detached(),
            validity=self.validity.detached(),
            temporal_level_zero=self.temporal_level_zero.detach(),
            static_level_zero=self.static_level_zero.detach(),
        )


class TargetEncoder(nn.Module):
    """Target temporal/static stems and the cached five-level target pyramid."""

    def __init__(
        self,
        *,
        widths: Sequence[int] = TARGET_WIDTHS,
        input_size: int = PRODUCTION_INPUT_SIZE,
        activation_checkpoint: bool = False,
        allow_test_override: bool = False,
    ) -> None:
        super().__init__()
        self.widths = validate_widths(
            "target widths",
            widths,
            production=TARGET_WIDTHS,
            allow_test_override=allow_test_override,
        )
        self.input_size = validate_input_size(
            input_size, allow_test_override=allow_test_override
        )
        self.allow_test_override = bool(allow_test_override)
        stem_width = self.widths[0] // 2
        self.temporal_stem = TemporalSpatialStem(
            2,
            stem_width,
            activation_checkpoint=activation_checkpoint,
        )
        self.static_stem = StaticSpatialStem(
            len(TARGET_STATIC_CHANNEL_NAMES),
            stem_width,
            activation_checkpoint=activation_checkpoint,
        )
        self.pyramid = FiveLevelPyramidEncoder(
            self.widths, activation_checkpoint=activation_checkpoint
        )

    def forward(self, inputs: TargetEncoderInputs) -> TargetEncoderOutput:
        require_no_autocast()
        require_module_float32("TargetEncoder", self)
        if not isinstance(inputs, TargetEncoderInputs):
            raise TypeError("TargetEncoder expects TargetEncoderInputs")
        if tuple(inputs.radar_history.shape[-2:]) != (
            self.input_size,
            self.input_size,
        ):
            raise ValueError(
                f"target spatial shape must be {self.input_size}x{self.input_size}"
            )
        valid = inputs.history_validity
        safe_radar = torch.where(valid, inputs.radar_history, 0.0)
        dynamic = torch.cat((safe_radar, valid.to(dtype=torch.float32)), dim=2)
        temporal = self.temporal_stem(
            dynamic, pooling_validity=inputs.history_validity
        )
        safe_static = torch.where(
            inputs.static_coverage, inputs.static_fields, 0.0
        )
        static = self.static_stem(safe_static)
        features = self.pyramid(torch.cat((temporal, static), dim=1))
        level_zero_validity = (
            valid.to(dtype=torch.float32).mean(dim=1)
            * inputs.static_coverage.to(dtype=torch.float32)
        )
        return TargetEncoderOutput(
            features=features,
            validity=build_validity_pyramid(level_zero_validity),
            temporal_level_zero=temporal,
            static_level_zero=static,
        )


__all__ = [
    "TARGET_STATIC_CHANNEL_NAMES",
    "TargetEncoder",
    "TargetEncoderInputs",
    "TargetEncoderOutput",
]
