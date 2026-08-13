"""Full-width lead-conditioned hurdle regression on the 500 m target grid."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from kcorrdiff.data.advection import ADVECTION_MODEL_CHANNEL_NAMES
from kcorrdiff.data.radar_values import A0_MM, A_WET_MM

from .common import (
    CONDITION_DIM,
    CONTEXT_WIDTHS,
    TARGET_WIDTHS,
    count_parameters,
    require_float32_tensor,
    require_no_autocast,
    validate_widths,
)
from .condition_bank import ConditionBankCache
from .era_encoder import ERA_OUTPUT_CHANNELS, EraQueryResult
from .advection import AdvAdapter
from .physical_attention import (
    PhysicalAttentionDiagnostics,
    PhysicalCrossAttention,
    PhysicalTokenGeometry,
)


ADVECTION_CHANNELS = len(ADVECTION_MODEL_CHANNEL_NAMES)


def _run_conditioned_checkpointed(
    module: nn.Module,
    inputs: Tensor,
    e_cond: Tensor,
    *,
    enabled: bool,
) -> Tensor:
    if enabled and module.training and torch.is_grad_enabled():
        return checkpoint(
            module,
            inputs,
            e_cond,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    return module(inputs, e_cond)


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels >= groups and channels % groups == 0:
            return groups
    return 1


class ConditionalResidualBlock(nn.Module):
    """Pre-activation residual block with one e_cond-only FiLM interface."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels, affine=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels, affine=False)
        self.film1 = nn.Linear(CONDITION_DIM, 2 * in_channels)
        self.film2 = nn.Linear(CONDITION_DIM, 2 * out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    @staticmethod
    def _modulate(value: Tensor, parameters: Tensor) -> Tensor:
        scale, shift = parameters.chunk(2, dim=1)
        return value * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, inputs: Tensor, e_cond: Tensor) -> Tensor:
        hidden = self._modulate(self.norm1(inputs), self.film1(e_cond))
        hidden = self.conv1(functional.silu(hidden))
        hidden = self._modulate(self.norm2(hidden), self.film2(e_cond))
        hidden = self.conv2(functional.silu(hidden))
        return self.skip(inputs) + hidden


@dataclass(frozen=True, slots=True)
class RegressionGeometry:
    target_l3: PhysicalTokenGeometry
    target_l4: PhysicalTokenGeometry
    context_l3: PhysicalTokenGeometry
    context_l4: PhysicalTokenGeometry
    era_native: PhysicalTokenGeometry


@dataclass(frozen=True, slots=True)
class RegressionInputs:
    condition_bank: ConditionBankCache
    era: EraQueryResult
    advection_features: Tensor
    e_cond: Tensor
    geometry: RegressionGeometry
    condition_signatures: tuple[str, ...]

    def __post_init__(self) -> None:
        batch = self.condition_bank.batch_size
        require_float32_tensor("regression advection_features", self.advection_features, ndim=4)
        require_float32_tensor("regression e_cond", self.e_cond, ndim=2)
        if tuple(self.advection_features.shape) != (
            batch,
            ADVECTION_CHANNELS,
            self.condition_bank.input_size,
            self.condition_bank.input_size,
        ):
            raise ValueError(
                "advection_features must have shape "
                f"[B,{ADVECTION_CHANNELS},H_target,W_target]"
            )
        if tuple(self.e_cond.shape) != (batch, CONDITION_DIM):
            raise ValueError(f"e_cond must have shape [B,{CONDITION_DIM}]")
        if tuple(self.era.features.shape) != (batch, ERA_OUTPUT_CHANNELS, 33, 33):
            raise ValueError("ERA query must retain [B,128,33,33]")
        if len(self.condition_signatures) != batch or any(
            not value for value in self.condition_signatures
        ):
            raise ValueError("one non-empty condition signature is required per item")
        devices = (
            self.condition_bank.device,
            self.era.features.device,
            self.advection_features.device,
            self.e_cond.device,
        )
        if any(device != devices[0] for device in devices):
            raise ValueError("all regression conditions must share one device")


@dataclass(frozen=True, slots=True)
class RegressionOutput:
    occurrence_logits: Tensor
    probability_wet: Tensor
    wet_amount: Tensor
    transformed_mean: Tensor
    context_l3_contribution: Tensor
    era_l3_contribution: Tensor
    context_l4_contribution: Tensor
    era_l4_contribution: Tensor
    decoder_features: Tensor


@dataclass(frozen=True, slots=True)
class DirectPhysicalRegressionOutput:
    """Output of a separate physical-mean or q50 comparison checkpoint."""

    prediction_mm: Tensor
    statistic: Literal["mean", "q50"]


class RegressionUNet(nn.Module):
    """Original-width target decoder with L3/L4 context and ERA fusion."""

    def __init__(
        self,
        *,
        target_widths: Sequence[int] = TARGET_WIDTHS,
        context_widths: Sequence[int] = CONTEXT_WIDTHS,
        query_chunk_size: int = 128,
        activation_checkpoint: bool = False,
        allow_test_override: bool = False,
    ) -> None:
        super().__init__()
        self.target_widths = validate_widths(
            "target widths",
            target_widths,
            production=TARGET_WIDTHS,
            allow_test_override=allow_test_override,
        )
        self.context_widths = validate_widths(
            "context widths",
            context_widths,
            production=CONTEXT_WIDTHS,
            allow_test_override=allow_test_override,
        )
        self.activation_checkpoint = bool(activation_checkpoint)
        attention = lambda target, source, name: PhysicalCrossAttention(
            target_channels=target,
            source_channels=source,
            condition_dim=CONDITION_DIM,
            source_name=name,
            query_chunk_size=query_chunk_size,
            activation_checkpoint=self.activation_checkpoint,
        )
        self.context_attention_l3 = attention(
            self.target_widths[3], self.context_widths[3], "regression_context_l3"
        )
        self.era_attention_l3 = attention(
            self.target_widths[3], ERA_OUTPUT_CHANNELS, "regression_era_l3"
        )
        self.context_attention_l4 = attention(
            self.target_widths[4], self.context_widths[4], "regression_context_l4"
        )
        self.era_attention_l4 = attention(
            self.target_widths[4], ERA_OUTPUT_CHANNELS, "regression_era_l4"
        )
        self.advection_adapters = nn.ModuleList(
            AdvAdapter(
                target_channels=width,
                advection_channels=ADVECTION_CHANNELS,
                condition_dim=CONDITION_DIM,
            )
            for width in self.target_widths
        )
        self.bottleneck = nn.ModuleList(
            [
                ConditionalResidualBlock(self.target_widths[4], self.target_widths[4]),
                ConditionalResidualBlock(self.target_widths[4], self.target_widths[4]),
            ]
        )
        self.upsample_projections = nn.ModuleList(
            nn.Conv2d(high, low, 3, padding=1, bias=False)
            for high, low in zip(
                self.target_widths[:0:-1], self.target_widths[-2::-1]
            )
        )
        self.decoder_blocks = nn.ModuleList(
            ConditionalResidualBlock(2 * width, width)
            for width in self.target_widths[-2::-1]
        )
        self.output_norm = nn.GroupNorm(_groups(self.target_widths[0]), self.target_widths[0])
        self.occurrence_head = nn.Conv2d(self.target_widths[0], 1, 1)
        self.positive_amount_head = nn.Conv2d(self.target_widths[0], 1, 1)

    @staticmethod
    def _diagnostic(
        result: Tensor | PhysicalAttentionDiagnostics,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(result, PhysicalAttentionDiagnostics):
            raise AssertionError("attention diagnostics were not requested")
        return result.output, result.gated_residual

    def _fuse(
        self,
        inputs: RegressionInputs,
        *,
        level: int,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cache = inputs.condition_bank
        context = cache.context.features.levels[level]
        context_validity = cache.context.validity.levels[level][:, 0] > 0.0
        batch = cache.batch_size
        context_present = context_validity.flatten(1).any(dim=1)
        era_present = ~(inputs.era.used_source_null | inputs.era.used_masked_null)
        era_validity = torch.ones(
            (batch, 33, 33), dtype=torch.bool, device=target.device
        )
        if level == 3:
            context_attention = self.context_attention_l3
            era_attention = self.era_attention_l3
            target_geometry = inputs.geometry.target_l3
            context_geometry = inputs.geometry.context_l3
        elif level == 4:
            context_attention = self.context_attention_l4
            era_attention = self.era_attention_l4
            target_geometry = inputs.geometry.target_l4
            context_geometry = inputs.geometry.context_l4
        else:
            raise ValueError("physical attention is defined only at L3/L4")
        context_result = context_attention(
            target,
            context,
            query_geometry=target_geometry,
            source_geometry=context_geometry,
            source_validity=context_validity,
            source_present=context_present,
            e_cond=inputs.e_cond,
            return_diagnostics=True,
        )
        hidden, context_contribution = self._diagnostic(context_result)
        era_result = era_attention(
            hidden,
            inputs.era.features,
            query_geometry=target_geometry,
            source_geometry=inputs.geometry.era_native,
            source_validity=era_validity,
            source_present=era_present,
            e_cond=inputs.e_cond,
            return_diagnostics=True,
        )
        hidden, era_contribution = self._diagnostic(era_result)
        return hidden, context_contribution, era_contribution

    def forward(self, inputs: RegressionInputs) -> RegressionOutput:
        require_no_autocast()
        if not isinstance(inputs, RegressionInputs):
            raise TypeError("RegressionUNet expects RegressionInputs")
        if inputs.condition_bank.target_widths != self.target_widths:
            raise ValueError("regression and condition-bank target widths differ")
        if inputs.condition_bank.context_widths != self.context_widths:
            raise ValueError("regression and condition-bank context widths differ")
        target = list(inputs.condition_bank.target.features.levels)
        target_with_advection: list[Tensor] = []
        for adapter, feature in zip(self.advection_adapters, target, strict=True):
            resized = functional.interpolate(
                inputs.advection_features,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            confidence = resized[:, 6:7].clamp(0.0, 1.0)
            target_with_advection.append(
                adapter(feature, resized, inputs.e_cond, confidence)
            )
        target = target_with_advection
        target[3], context_l3, era_l3 = self._fuse(
            inputs, level=3, target=target[3]
        )
        target[4], context_l4, era_l4 = self._fuse(
            inputs, level=4, target=target[4]
        )
        hidden = target[4]
        for block in self.bottleneck:
            hidden = _run_conditioned_checkpointed(
                block,
                hidden,
                inputs.e_cond,
                enabled=self.activation_checkpoint,
            )
        skips = target[3::-1]
        for projection, block, skip in zip(
            self.upsample_projections, self.decoder_blocks, skips, strict=True
        ):
            hidden = functional.interpolate(
                hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            hidden = projection(hidden)
            hidden = _run_conditioned_checkpointed(
                block,
                torch.cat((hidden, skip), dim=1),
                inputs.e_cond,
                enabled=self.activation_checkpoint,
            )
        hidden = functional.silu(self.output_norm(hidden))
        logits = self.occurrence_head(hidden)
        probability = torch.sigmoid(logits)
        z_wet = math.log1p(A_WET_MM / A0_MM)
        wet_amount = z_wet + functional.softplus(self.positive_amount_head(hidden))
        transformed_mean = probability * wet_amount
        return RegressionOutput(
            occurrence_logits=logits,
            probability_wet=probability,
            wet_amount=wet_amount,
            transformed_mean=transformed_mean,
            context_l3_contribution=context_l3,
            era_l3_contribution=era_l3,
            context_l4_contribution=context_l4,
            era_l4_contribution=era_l4,
            decoder_features=hidden,
        )

    @property
    def parameter_count(self) -> int:
        return count_parameters(self)


class DirectPhysicalRegression(nn.Module):
    """Separate same-capacity checkpoint for physical mean or median.

    This wrapper owns a complete :class:`RegressionUNet`; it does not reuse a
    trained hurdle checkpoint.  The comparison therefore has the same
    condition encoders, attention and decoder capacity, while its final head
    is optimized directly in physical millimetres and is never supplied to
    residual diffusion.
    """

    def __init__(
        self,
        *,
        statistic: Literal["mean", "q50"],
        target_widths: Sequence[int] = TARGET_WIDTHS,
        context_widths: Sequence[int] = CONTEXT_WIDTHS,
        query_chunk_size: int = 128,
        activation_checkpoint: bool = False,
        allow_test_override: bool = False,
    ) -> None:
        super().__init__()
        if statistic not in {"mean", "q50"}:
            raise ValueError("direct physical statistic must be mean or q50")
        self.statistic = statistic
        self.backbone = RegressionUNet(
            target_widths=target_widths,
            context_widths=context_widths,
            query_chunk_size=query_chunk_size,
            activation_checkpoint=activation_checkpoint,
            allow_test_override=allow_test_override,
        )
        self.physical_head = nn.Conv2d(self.backbone.target_widths[0], 1, 1)

    def forward(self, inputs: RegressionInputs) -> DirectPhysicalRegressionOutput:
        result = self.backbone(inputs)
        raw = self.physical_head(result.decoder_features)
        prediction = (
            functional.softplus(raw)
            if self.statistic == "mean"
            else functional.relu(raw)
        )
        return DirectPhysicalRegressionOutput(
            prediction_mm=prediction,
            statistic=self.statistic,
        )

    @property
    def parameter_count(self) -> int:
        return count_parameters(self)


__all__ = [
    "ADVECTION_CHANNELS",
    "ConditionalResidualBlock",
    "DirectPhysicalRegression",
    "DirectPhysicalRegressionOutput",
    "RegressionGeometry",
    "RegressionInputs",
    "RegressionOutput",
    "RegressionUNet",
]
