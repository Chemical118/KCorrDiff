"""Native-grid ERA5 frame encoding and lead-conditioned temporal query.

The frame cache is deliberately lead-independent.  It retains eight native
hours at 33 x 33, keeps the 23 instantaneous fields and one-hour ``tp`` stem
separate, and records their distinct time centres.  A lead-specific query is
the first operation allowed to consume ``e_cond`` or the temporal-access mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import require_no_autocast


ERA_NATIVE_HOURS = 8
ERA_NATIVE_SHAPE = (33, 33)
ERA_INSTANTANEOUS_CHANNELS = 23
ERA_OUTPUT_CHANNELS = 128


def _require_bool_mask(
    value: Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if value.device != device:
        raise ValueError(f"{name} must be on the ERA tensor device")
    return value


def _time_features(delta: Tensor, frequencies: Tensor) -> Tensor:
    """Continuous Fourier features without a learned discrete-hour index."""

    angle = delta[..., None] * frequencies
    return torch.cat((delta[..., None], torch.sin(angle), torch.cos(angle)), dim=-1)


class _SpatialResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, value: Tensor) -> Tensor:
        residual = F.silu(self.norm1(value))
        residual = self.depthwise(residual)
        residual = self.pointwise(F.silu(self.norm2(residual)))
        return value + residual


@dataclass(frozen=True, slots=True)
class EraFrameCache:
    """Lead-independent native-hour stem outputs and source state."""

    instantaneous: Tensor
    precipitation: Tensor
    encoded: Tensor
    tp_null: Tensor
    delta_hours: Tensor
    tp_interval_center_delta_hours: Tensor
    data_valid_inst: Tensor
    tp_valid: Tensor
    trajectory_window_mask: Tensor
    era_present: Tensor
    tp_present: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.instantaneous.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self.instantaneous.dtype

    @property
    def device(self) -> torch.device:
        return self.instantaneous.device


@dataclass(frozen=True, slots=True)
class EraQueryResult:
    """One lead's 128-channel ERA spatial condition and diagnostics."""

    features: Tensor
    temporal_weights: Tensor
    valid_token_mask: Tensor
    tp_token_mask: Tensor
    used_source_null: Tensor
    used_masked_null: Tensor


class EraEncoder(nn.Module):
    """Encode exact ``[B,8,23/1,33,33]`` ERA inputs and query one lead.

    The native grid is never resized.  ``encode_frames`` may be cached across
    all 12 leads; ``query`` consumes ``e_cond`` and the per-lead intentional
    access mask.  ``era_present=False`` and an ERA-present but entirely masked
    trajectory have distinct learned null outputs.
    """

    def __init__(
        self,
        *,
        condition_dim: int,
        output_channels: int = ERA_OUTPUT_CHANNELS,
        stem_channels: int = 64,
        temporal_heads: int = 8,
        time_frequencies: int = 8,
        spatial_blocks: int = 2,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if output_channels != ERA_OUTPUT_CHANNELS:
            raise ValueError("v1.1.3b ERA output must contain exactly 128 channels")
        if stem_channels <= 0 or spatial_blocks < 0:
            raise ValueError("stem width must be positive and block count non-negative")
        if temporal_heads <= 0 or output_channels % temporal_heads:
            raise ValueError("output_channels must be divisible by temporal_heads")
        if time_frequencies <= 0:
            raise ValueError("time_frequencies must be positive")

        self.condition_dim = int(condition_dim)
        self.output_channels = int(output_channels)
        self.temporal_heads = int(temporal_heads)
        self.head_dim = output_channels // temporal_heads

        self.instantaneous_projection = nn.Conv2d(
            ERA_INSTANTANEOUS_CHANNELS, stem_channels, kernel_size=1
        )
        self.precipitation_projection = nn.Conv2d(1, stem_channels, kernel_size=1)
        self.register_buffer(
            "time_frequencies",
            torch.pi * torch.pow(2.0, torch.arange(time_frequencies)),
            persistent=True,
        )
        time_dimension = 1 + 2 * time_frequencies
        self.instantaneous_time_embedding = nn.Sequential(
            nn.Linear(time_dimension, stem_channels),
            nn.SiLU(),
            nn.Linear(stem_channels, stem_channels),
        )
        self.precipitation_time_embedding = nn.Sequential(
            nn.Linear(time_dimension, stem_channels),
            nn.SiLU(),
            nn.Linear(stem_channels, stem_channels),
        )
        self.tp_null_state = nn.Parameter(torch.empty(1, 1, stem_channels, 1, 1))
        self.fusion = nn.Conv2d(2 * stem_channels, output_channels, kernel_size=1)
        self.spatial_blocks = nn.Sequential(
            *[_SpatialResidualBlock(output_channels) for _ in range(spatial_blocks)]
        )

        self.temporal_norm = nn.LayerNorm(output_channels)
        self.temporal_key = nn.Linear(output_channels, output_channels)
        self.temporal_value = nn.Linear(output_channels, output_channels)
        self.temporal_query = nn.Linear(condition_dim, output_channels)
        self.temporal_time_key = nn.Sequential(
            nn.Linear(time_dimension, output_channels),
            nn.SiLU(),
            nn.Linear(output_channels, output_channels),
        )
        self.temporal_output = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.source_null_state = nn.Parameter(torch.empty(1, output_channels, 1, 1))
        self.masked_null_state = nn.Parameter(torch.empty(1, output_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tp_null_state, std=0.02)
        nn.init.normal_(self.source_null_state, std=0.02)
        nn.init.normal_(self.masked_null_state, std=0.02)

    def encode_frames(
        self,
        instantaneous: Tensor,
        precipitation: Tensor,
        *,
        delta_hours: Tensor,
        data_valid_inst: Tensor,
        tp_valid: Tensor,
        trajectory_window_mask: Tensor,
        era_present: Tensor,
        tp_present: Tensor,
    ) -> EraFrameCache:
        """Build cacheable native-hour frame features without a lead query."""

        require_no_autocast()

        if instantaneous.ndim != 5:
            raise ValueError("instantaneous must have shape [B,8,23,33,33]")
        batch = int(instantaneous.shape[0])
        expected_instantaneous = (
            batch,
            ERA_NATIVE_HOURS,
            ERA_INSTANTANEOUS_CHANNELS,
            *ERA_NATIVE_SHAPE,
        )
        expected_precipitation = (
            batch,
            ERA_NATIVE_HOURS,
            1,
            *ERA_NATIVE_SHAPE,
        )
        if tuple(instantaneous.shape) != expected_instantaneous:
            raise ValueError(
                f"instantaneous must have shape {expected_instantaneous}, "
                f"got {tuple(instantaneous.shape)}"
            )
        if tuple(precipitation.shape) != expected_precipitation:
            raise ValueError(
                f"precipitation must have shape {expected_precipitation}, "
                f"got {tuple(precipitation.shape)}"
            )
        if instantaneous.dtype != torch.float32 or precipitation.dtype != torch.float32:
            raise TypeError("ERA inputs must use float32")
        if instantaneous.device != precipitation.device:
            raise ValueError("instantaneous and precipitation must share a device")
        if not torch.isfinite(instantaneous).all() or not torch.isfinite(
            precipitation
        ).all():
            raise ValueError("ERA inputs must contain only finite values")
        device = instantaneous.device
        if delta_hours.shape != (batch, ERA_NATIVE_HOURS):
            raise ValueError("delta_hours must have shape [B,8]")
        if delta_hours.dtype != torch.float32 or delta_hours.device != device:
            raise TypeError("delta_hours must be float32 on the ERA tensor device")
        if not torch.isfinite(delta_hours).all():
            raise ValueError("delta_hours must be finite")
        masks = {
            "data_valid_inst": _require_bool_mask(
                data_valid_inst,
                name="data_valid_inst",
                shape=(batch, ERA_NATIVE_HOURS),
                device=device,
            ),
            "tp_valid": _require_bool_mask(
                tp_valid,
                name="tp_valid",
                shape=(batch, ERA_NATIVE_HOURS),
                device=device,
            ),
            "trajectory_window_mask": _require_bool_mask(
                trajectory_window_mask,
                name="trajectory_window_mask",
                shape=(batch, ERA_NATIVE_HOURS),
                device=device,
            ),
            "era_present": _require_bool_mask(
                era_present,
                name="era_present",
                shape=(batch,),
                device=device,
            ),
            "tp_present": _require_bool_mask(
                tp_present,
                name="tp_present",
                shape=(batch,),
                device=device,
            ),
        }
        if torch.any(masks["tp_present"] & ~masks["era_present"]):
            raise ValueError("tp_present cannot be true when era_present is false")

        flat_instantaneous = instantaneous.flatten(0, 1)
        flat_precipitation = precipitation.flatten(0, 1)
        inst = self.instantaneous_projection(flat_instantaneous).unflatten(
            0, (batch, ERA_NATIVE_HOURS)
        )
        tp = self.precipitation_projection(flat_precipitation).unflatten(
            0, (batch, ERA_NATIVE_HOURS)
        )
        frequencies = self.time_frequencies.to(device=device, dtype=torch.float32)
        inst_time = self.instantaneous_time_embedding(
            _time_features(delta_hours, frequencies)
        )[..., None, None]
        tp_delta = delta_hours - 0.5
        tp_time = self.precipitation_time_embedding(
            _time_features(tp_delta, frequencies)
        )[..., None, None]
        inst = inst + inst_time
        tp = tp + tp_time
        tp_on = (
            masks["tp_valid"]
            & masks["tp_present"][:, None]
        )[:, :, None, None, None]
        tp_null = self.tp_null_state.expand_as(tp)
        tp_selected = torch.where(tp_on, tp, tp_null)
        fused = torch.cat((inst, tp_selected), dim=2).flatten(0, 1)
        fused = self.fusion(fused)
        fused = self.spatial_blocks(fused).unflatten(
            0, (batch, ERA_NATIVE_HOURS)
        )
        return EraFrameCache(
            instantaneous=inst,
            precipitation=tp,
            encoded=fused,
            tp_null=tp_null,
            delta_hours=delta_hours,
            tp_interval_center_delta_hours=tp_delta,
            data_valid_inst=masks["data_valid_inst"],
            tp_valid=masks["tp_valid"],
            trajectory_window_mask=masks["trajectory_window_mask"],
            era_present=masks["era_present"],
            tp_present=masks["tp_present"],
        )

    def query(
        self,
        cache: EraFrameCache,
        *,
        temporal_access_mask: Tensor,
        e_cond: Tensor,
    ) -> EraQueryResult:
        """Apply one lead's temporal access policy and condition query."""

        require_no_autocast()

        batch = cache.batch_size
        access = _require_bool_mask(
            temporal_access_mask,
            name="temporal_access_mask",
            shape=(batch, ERA_NATIVE_HOURS),
            device=cache.device,
        )
        if e_cond.shape != (batch, self.condition_dim):
            raise ValueError(
                f"e_cond must have shape {(batch, self.condition_dim)}, "
                f"got {tuple(e_cond.shape)}"
            )
        if e_cond.dtype != torch.float32 or e_cond.device != cache.device:
            raise TypeError("e_cond must be float32 on the ERA tensor device")
        if not torch.isfinite(e_cond).all():
            raise ValueError("e_cond must be finite")

        valid = (
            cache.data_valid_inst
            & cache.trajectory_window_mask
            & access
            & cache.era_present[:, None]
        )
        tp_token_mask = (
            cache.tp_valid
            & cache.trajectory_window_mask
            & access
            & cache.tp_present[:, None]
            & cache.era_present[:, None]
        )
        pooled = cache.encoded.mean(dim=(-2, -1))
        pooled = self.temporal_norm(
            self.temporal_key(pooled)
            + self.temporal_time_key(
                _time_features(
                    cache.delta_hours,
                    self.time_frequencies.to(cache.device, torch.float32),
                )
            )
        )
        key = pooled.view(batch, ERA_NATIVE_HOURS, self.temporal_heads, self.head_dim)
        query = self.temporal_query(e_cond).view(
            batch, self.temporal_heads, self.head_dim
        )
        logits = torch.einsum("bhd,bthd->bht", query, key) / math.sqrt(
            self.head_dim
        )
        logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
        any_valid = valid.any(dim=1)
        safe_logits = torch.where(
            any_valid[:, None, None], logits, torch.zeros_like(logits)
        )
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(
            valid[:, None, :], weights, torch.zeros_like(weights)
        )

        value = self.temporal_value(
            cache.encoded.permute(0, 1, 3, 4, 2)
        ).permute(0, 1, 4, 2, 3)
        value = value.view(
            batch,
            ERA_NATIVE_HOURS,
            self.temporal_heads,
            self.head_dim,
            *ERA_NATIVE_SHAPE,
        )
        attended = torch.einsum("bht,bthdxy->bhdxy", weights, value).reshape(
            batch, self.output_channels, *ERA_NATIVE_SHAPE
        )
        attended = self.temporal_output(attended)
        source_missing = ~cache.era_present
        masked = cache.era_present & ~any_valid
        features = torch.where(
            source_missing[:, None, None, None],
            self.source_null_state,
            attended,
        )
        features = torch.where(
            masked[:, None, None, None],
            self.masked_null_state,
            features,
        )
        return EraQueryResult(
            features=features,
            temporal_weights=weights,
            valid_token_mask=valid,
            tp_token_mask=tp_token_mask,
            used_source_null=source_missing,
            used_masked_null=masked,
        )

    def forward(
        self,
        instantaneous: Tensor,
        precipitation: Tensor,
        *,
        delta_hours: Tensor,
        data_valid_inst: Tensor,
        tp_valid: Tensor,
        trajectory_window_mask: Tensor,
        temporal_access_mask: Tensor,
        era_present: Tensor,
        tp_present: Tensor,
        e_cond: Tensor,
    ) -> EraQueryResult:
        cache = self.encode_frames(
            instantaneous,
            precipitation,
            delta_hours=delta_hours,
            data_valid_inst=data_valid_inst,
            tp_valid=tp_valid,
            trajectory_window_mask=trajectory_window_mask,
            era_present=era_present,
            tp_present=tp_present,
        )
        return self.query(
            cache, temporal_access_mask=temporal_access_mask, e_cond=e_cond
        )


__all__ = [
    "ERA_INSTANTANEOUS_CHANNELS",
    "ERA_NATIVE_HOURS",
    "ERA_NATIVE_SHAPE",
    "ERA_OUTPUT_CHANNELS",
    "EraEncoder",
    "EraFrameCache",
    "EraQueryResult",
]
