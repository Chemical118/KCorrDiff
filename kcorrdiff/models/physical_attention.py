"""Chunked physical-coordinate cross-attention for L3/L4 fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .common import require_no_autocast


ATTENTION_DIM = 256
ATTENTION_HEADS = 8


@dataclass(frozen=True, slots=True)
class PhysicalTokenGeometry:
    """Common-LCC token centres and strictly positive physical footprints.

    Each tensor is either unbatched ``[H,W]`` or batched ``[B,H,W]``.
    Positions use the common target-centred scale from architecture §5;
    footprints use one consistent positive physical unit (normally km).
    """

    x_shared: Tensor
    y_shared: Tensor
    footprint_width: Tensor
    footprint_height: Tensor


@dataclass(frozen=True, slots=True)
class PhysicalAttentionDiagnostics:
    output: Tensor
    attention_residual: Tensor
    gated_residual: Tensor
    gate: Tensor


class _ConditionalLayerNorm(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.modulation = nn.Linear(condition_dim, 2 * channels)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        # value [B,N,C], condition [B,E]
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(value) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class PhysicalCrossAttention(nn.Module):
    """Target-query/source-key attention with Fourier physical bias.

    One instance represents one source, scale, and model stage and therefore
    owns its own zero-initialized source gate.  Only that gate is zeroed: the
    attention output projection retains its ordinary nonzero initialization.
    """

    def __init__(
        self,
        *,
        target_channels: int,
        source_channels: int,
        condition_dim: int,
        source_name: str,
        attention_dim: int = ATTENTION_DIM,
        heads: int = ATTENTION_HEADS,
        fourier_frequencies: int = 4,
        geometry_hidden_dim: int = 64,
        query_chunk_size: int = 128,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if min(target_channels, source_channels, condition_dim) <= 0:
            raise ValueError("feature and condition dimensions must be positive")
        if attention_dim != ATTENTION_DIM or heads != ATTENTION_HEADS:
            raise ValueError("v1.1.3b L3/L4 attention requires D=256 and 8 heads")
        if fourier_frequencies <= 0 or geometry_hidden_dim <= 0:
            raise ValueError("geometry embedding dimensions must be positive")
        if query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if not source_name:
            raise ValueError("source_name must be non-empty")
        self.target_channels = int(target_channels)
        self.source_channels = int(source_channels)
        self.condition_dim = int(condition_dim)
        self.source_name = source_name
        self.attention_dim = attention_dim
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.query_chunk_size = int(query_chunk_size)
        self.activation_checkpoint = bool(activation_checkpoint)

        self.query_norm = _ConditionalLayerNorm(target_channels, condition_dim)
        self.source_norm = _ConditionalLayerNorm(source_channels, condition_dim)
        self.query_projection = nn.Linear(target_channels, attention_dim, bias=False)
        self.key_projection = nn.Linear(source_channels, attention_dim, bias=False)
        self.value_projection = nn.Linear(source_channels, attention_dim, bias=False)
        self.output_projection = nn.Linear(attention_dim, target_channels, bias=False)
        self.gate_projection = nn.Linear(condition_dim, target_channels)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)

        self.register_buffer(
            "geometry_frequencies",
            torch.pi * torch.pow(2.0, torch.arange(fourier_frequencies)),
            persistent=True,
        )
        geometry_variables = 7
        geometry_dimension = geometry_variables * (1 + 2 * fourier_frequencies)
        self.geometry_bias = nn.Sequential(
            nn.Linear(geometry_dimension, geometry_hidden_dim),
            nn.SiLU(),
            nn.Linear(geometry_hidden_dim, heads),
        )

    @staticmethod
    def _feature_tokens(value: Tensor, *, channels: int, name: str) -> Tensor:
        if value.ndim != 4 or value.shape[1] != channels:
            raise ValueError(f"{name} must have shape [B,{channels},H,W]")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must use float32")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")
        return value.flatten(2).transpose(1, 2)

    @staticmethod
    def _geometry_tokens(
        geometry: PhysicalTokenGeometry,
        *,
        batch: int,
        shape: tuple[int, int],
        device: torch.device,
        name: str,
    ) -> tuple[Tensor, Tensor]:
        fields = (
            geometry.x_shared,
            geometry.y_shared,
            geometry.footprint_width,
            geometry.footprint_height,
        )
        expected_unbatched = shape
        expected_batched = (batch, *shape)
        expanded: list[Tensor] = []
        for field in fields:
            if field.dtype != torch.float32 or field.device != device:
                raise TypeError(f"{name} geometry must be float32 on the feature device")
            if tuple(field.shape) == expected_unbatched:
                field = field[None].expand(batch, -1, -1)
            elif tuple(field.shape) != expected_batched:
                raise ValueError(
                    f"{name} geometry fields must have shape "
                    f"{expected_unbatched} or {expected_batched}"
                )
            if not torch.isfinite(field).all():
                raise ValueError(f"{name} geometry must be finite")
            expanded.append(field)
        if torch.any(expanded[2] <= 0.0) or torch.any(expanded[3] <= 0.0):
            raise ValueError(f"{name} token footprints must be positive")
        positions = torch.stack((expanded[0], expanded[1]), dim=-1).flatten(1, 2)
        footprints = torch.stack((expanded[2], expanded[3]), dim=-1).flatten(1, 2)
        return positions, footprints

    def _relative_features(
        self,
        query_position: Tensor,
        query_footprint: Tensor,
        key_position: Tensor,
        key_footprint: Tensor,
    ) -> Tensor:
        delta = key_position[:, None, :, :] - query_position[:, :, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        query_footprint_pair = query_footprint[:, :, None, :].expand(
            -1, -1, key_position.shape[1], -1
        )
        key_footprint_pair = key_footprint[:, None, :, :].expand(
            -1, query_position.shape[1], -1, -1
        )
        raw = torch.cat(
            (
                delta,
                distance,
                torch.log(query_footprint_pair),
                torch.log(key_footprint_pair),
            ),
            dim=-1,
        )
        frequencies = self.geometry_frequencies.to(raw.device, torch.float32)
        angle = raw[..., None] * frequencies
        return torch.cat(
            (raw, torch.sin(angle).flatten(-2), torch.cos(angle).flatten(-2)),
            dim=-1,
        )

    def _bias_chunk(
        self,
        query_position: Tensor,
        query_footprint: Tensor,
        key_position: Tensor,
        key_footprint: Tensor,
    ) -> Tensor:
        features = self._relative_features(
            query_position, query_footprint, key_position, key_footprint
        )
        # [B,Q,K,H] -> [B,H,Q,K]
        return self.geometry_bias(features).permute(0, 3, 1, 2)

    @staticmethod
    def _finish_attention_chunk(
        query_chunk: Tensor,
        key_for_scores: Tensor,
        value_for_sum: Tensor,
        valid: Tensor,
        any_valid: Tensor,
        physical_bias: Tensor,
    ) -> Tensor:
        scores = torch.matmul(query_chunk, key_for_scores.transpose(-1, -2))
        scores = scores / math.sqrt(query_chunk.shape[-1])
        scores = scores + physical_bias
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        safe_scores = torch.where(
            any_valid[:, None, None, None], scores, torch.zeros_like(scores)
        )
        weights = torch.softmax(safe_scores, dim=-1)
        weights = torch.where(
            valid[:, None, None, :], weights, torch.zeros_like(weights)
        )
        return torch.matmul(weights, value_for_sum)

    def _dynamic_attention_chunk(
        self,
        query_chunk: Tensor,
        key_for_scores: Tensor,
        value_for_sum: Tensor,
        valid: Tensor,
        any_valid: Tensor,
        query_position: Tensor,
        query_footprint: Tensor,
        key_position: Tensor,
        key_footprint: Tensor,
    ) -> Tensor:
        physical_bias = self._bias_chunk(
            query_position,
            query_footprint,
            key_position,
            key_footprint,
        )
        return self._finish_attention_chunk(
            query_chunk,
            key_for_scores,
            value_for_sum,
            valid,
            any_valid,
            physical_bias,
        )

    def _run_attention_chunk(
        self,
        function: object,
        *arguments: Tensor,
    ) -> Tensor:
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                function,
                *arguments,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return function(*arguments)  # type: ignore[operator, no-any-return]

    def _connected_zero_residual(
        self,
        target: Tensor,
        source: Tensor,
        e_cond: Tensor,
    ) -> Tensor:
        """Return exact zeros while retaining DDP-visible zero gradients.

        An absent source is mathematically an identity branch.  During training,
        however, dropping the branch from autograd would make its parameters and
        the upstream source encoder unused on only some ranks.  Scalar zero
        connections preserve the original zero-gradient behavior without
        constructing Q/K/V, dense geometry features, or attention weights.
        """

        if not torch.is_grad_enabled():
            return torch.zeros_like(target)
        connected = source.reshape(-1)[0] * 0.0 + e_cond.reshape(-1)[0] * 0.0
        for parameter in self.parameters():
            if parameter.requires_grad:
                connected = connected + parameter.reshape(-1)[0] * 0.0
        return torch.zeros_like(target) + connected

    def compute_physical_bias(
        self,
        query_geometry: PhysicalTokenGeometry,
        source_geometry: PhysicalTokenGeometry,
        *,
        batch_size: int,
        query_shape: tuple[int, int],
        source_shape: tuple[int, int],
        device: torch.device | str | None = None,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Materialize cacheable per-head bias, optionally in query chunks."""

        require_no_autocast()

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        selected_device = torch.device(device or self.geometry_frequencies.device)
        query_position, query_footprint = self._geometry_tokens(
            query_geometry,
            batch=batch_size,
            shape=query_shape,
            device=selected_device,
            name="query",
        )
        key_position, key_footprint = self._geometry_tokens(
            source_geometry,
            batch=batch_size,
            shape=source_shape,
            device=selected_device,
            name="source",
        )
        size = self.query_chunk_size if chunk_size is None else int(chunk_size)
        if size <= 0:
            raise ValueError("chunk_size must be positive")
        chunks = [
            self._bias_chunk(
                query_position[:, start:end],
                query_footprint[:, start:end],
                key_position,
                key_footprint,
            )
            for start, end in self._chunks(query_position.shape[1], size)
        ]
        return torch.cat(chunks, dim=2)

    @staticmethod
    def _chunks(length: int, size: int) -> Iterator[tuple[int, int]]:
        for start in range(0, length, size):
            yield start, min(start + size, length)

    def forward(
        self,
        target: Tensor,
        source: Tensor,
        *,
        query_geometry: PhysicalTokenGeometry,
        source_geometry: PhysicalTokenGeometry,
        source_validity: Tensor,
        source_present: Tensor,
        e_cond: Tensor,
        query_chunk_size: int | None = None,
        precomputed_physical_bias: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> Tensor | PhysicalAttentionDiagnostics:
        require_no_autocast()
        if target.device != source.device:
            raise ValueError("target and source features must share a device")
        query_tokens = self._feature_tokens(
            target, channels=self.target_channels, name="target"
        )
        source_tokens = self._feature_tokens(
            source, channels=self.source_channels, name="source"
        )
        batch, _, query_height, query_width = target.shape
        if source.shape[0] != batch:
            raise ValueError("target and source features must share a batch size")
        source_height, source_width = source.shape[-2:]
        if e_cond.shape != (batch, self.condition_dim):
            raise ValueError(
                f"e_cond must have shape {(batch, self.condition_dim)}, "
                f"got {tuple(e_cond.shape)}"
            )
        if e_cond.dtype != torch.float32 or e_cond.device != target.device:
            raise TypeError("e_cond must be float32 on the feature device")
        if not torch.isfinite(e_cond).all():
            raise ValueError("e_cond must be finite")
        if source_validity.shape != (batch, source_height, source_width):
            raise ValueError("source_validity must have shape [B,H_source,W_source]")
        if source_validity.dtype is not torch.bool or source_validity.device != target.device:
            raise TypeError("source_validity must be bool on the feature device")
        if source_present.shape != (batch,):
            raise ValueError("source_present must have shape [B]")
        if source_present.dtype is not torch.bool or source_present.device != target.device:
            raise TypeError("source_present must be bool on the feature device")

        query_position, query_footprint = self._geometry_tokens(
            query_geometry,
            batch=batch,
            shape=(query_height, query_width),
            device=target.device,
            name="query",
        )
        key_position, key_footprint = self._geometry_tokens(
            source_geometry,
            batch=batch,
            shape=(source_height, source_width),
            device=target.device,
            name="source",
        )
        total_queries = query_tokens.shape[1]
        total_keys = source_tokens.shape[1]
        if precomputed_physical_bias is not None:
            expected = (batch, self.heads, total_queries, total_keys)
            if precomputed_physical_bias.shape != expected:
                raise ValueError(
                    f"precomputed_physical_bias must have shape {expected}"
                )
            if (
                precomputed_physical_bias.dtype != torch.float32
                or precomputed_physical_bias.device != target.device
                or not torch.isfinite(precomputed_physical_bias).all()
            ):
                raise ValueError("precomputed physical bias must be finite float32")
        valid = source_validity.flatten(1) & source_present[:, None]
        any_valid = valid.any(dim=1)
        if not bool(any_valid.any().item()):
            residual = self._connected_zero_residual(target, source, e_cond)
            gate = self.gate_projection(e_cond)
            gated = (
                residual
                * gate[:, :, None, None]
                * source_present[:, None, None, None]
            )
            output = target + gated
            if return_diagnostics:
                return PhysicalAttentionDiagnostics(
                    output=output,
                    attention_residual=residual,
                    gated_residual=gated,
                    gate=gate,
                )
            return output

        query = self.query_projection(
            self.query_norm(query_tokens, e_cond)
        ).view(batch, -1, self.heads, self.head_dim)
        source_modulated = self.source_norm(source_tokens, e_cond)
        key = self.key_projection(source_modulated).view(
            batch, -1, self.heads, self.head_dim
        )
        value = self.value_projection(source_modulated).view(
            batch, -1, self.heads, self.head_dim
        )
        chunk_size = self.query_chunk_size if query_chunk_size is None else int(
            query_chunk_size
        )
        if chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")

        attended_chunks: list[Tensor] = []
        key_for_scores = key.permute(0, 2, 1, 3)
        value_for_sum = value.permute(0, 2, 1, 3)
        for start, end in self._chunks(total_queries, chunk_size):
            query_chunk = query[:, start:end].permute(0, 2, 1, 3)
            if precomputed_physical_bias is None:
                attended = self._run_attention_chunk(
                    self._dynamic_attention_chunk,
                    query_chunk,
                    key_for_scores,
                    value_for_sum,
                    valid,
                    any_valid,
                    query_position[:, start:end],
                    query_footprint[:, start:end],
                    key_position,
                    key_footprint,
                )
            else:
                attended = self._run_attention_chunk(
                    self._finish_attention_chunk,
                    query_chunk,
                    key_for_scores,
                    value_for_sum,
                    valid,
                    any_valid,
                    precomputed_physical_bias[:, :, start:end],
                )
            attended_chunks.append(attended.permute(0, 2, 1, 3))
        attended = torch.cat(attended_chunks, dim=1).reshape(
            batch, total_queries, self.attention_dim
        )
        residual_tokens = self.output_projection(attended)
        residual = residual_tokens.transpose(1, 2).reshape_as(target)
        gate = self.gate_projection(e_cond)
        gated = residual * gate[:, :, None, None] * source_present[:, None, None, None]
        output = target + gated
        if return_diagnostics:
            return PhysicalAttentionDiagnostics(
                output=output,
                attention_residual=residual,
                gated_residual=gated,
                gate=gate,
            )
        return output


__all__ = [
    "ATTENTION_DIM",
    "ATTENTION_HEADS",
    "PhysicalAttentionDiagnostics",
    "PhysicalCrossAttention",
    "PhysicalTokenGeometry",
]
