"""Zero-initialized adapter for causal-advection condition fields."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _group_count(width: int) -> int:
    for groups in (8, 4, 2, 1):
        if width % groups == 0:
            return groups
    return 1  # pragma: no cover - the loop always reaches one


class AdvAdapter(nn.Module):
    """Inject full-width advection features through one learned gated residual.

    The final projection alone is initialized to zero, so construction leaves
    the parent target feature exactly unchanged while the first optimization
    step receives a gradient on that projection.  The lead/verification-time
    condition and continuous confidence gates retain ordinary nonzero
    initialization; no hand-written monotone lead curve is present.
    """

    def __init__(
        self,
        *,
        target_channels: int,
        advection_channels: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        if target_channels <= 0 or advection_channels <= 0 or condition_dim <= 0:
            raise ValueError("adapter channel/condition dimensions must be positive")
        self.target_channels = int(target_channels)
        self.advection_channels = int(advection_channels)
        self.condition_dim = int(condition_dim)
        groups = _group_count(self.target_channels)
        self.feature_projection = nn.Conv2d(
            self.advection_channels, self.target_channels, kernel_size=3, padding=1
        )
        self.feature_norm = nn.GroupNorm(groups, self.target_channels)
        self.feature_mixing = nn.Conv2d(
            self.target_channels,
            self.target_channels,
            kernel_size=3,
            padding=1,
            groups=self.target_channels,
        )
        self.mixing_norm = nn.GroupNorm(groups, self.target_channels)
        self.activation = nn.SiLU()
        self.condition_gate = nn.Linear(self.condition_dim, self.target_channels)
        self.confidence_gate = nn.Conv2d(1, self.target_channels, kernel_size=1)
        self.residual_projection = nn.Conv2d(
            self.target_channels, self.target_channels, kernel_size=1
        )
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def _validate(
        self,
        target_features: Tensor,
        advection_features: Tensor,
        e_cond: Tensor,
        confidence: Tensor,
    ) -> None:
        if target_features.ndim != 4 or advection_features.ndim != 4:
            raise ValueError("spatial features must have shape [B,C,H,W]")
        if target_features.shape[1] != self.target_channels:
            raise ValueError("target feature width does not match AdvAdapter")
        if advection_features.shape[1] != self.advection_channels:
            raise ValueError("advection feature width does not match AdvAdapter")
        if target_features.shape[0] != advection_features.shape[0] or target_features.shape[-2:] != advection_features.shape[-2:]:
            raise ValueError("target/advection batch and spatial shapes must match")
        if e_cond.shape != (target_features.shape[0], self.condition_dim):
            raise ValueError("e_cond must have shape [B,condition_dim]")
        if confidence.shape != (
            target_features.shape[0],
            1,
            *target_features.shape[-2:],
        ):
            raise ValueError("confidence must have shape [B,1,H,W]")
        tensors = (target_features, advection_features, e_cond, confidence)
        if any(not value.is_floating_point() for value in tensors):
            raise TypeError("AdvAdapter inputs must be floating tensors")
        if any(value.device != target_features.device for value in tensors[1:]):
            raise ValueError("AdvAdapter inputs must share one device")
        if any(value.dtype != target_features.dtype for value in tensors[1:]):
            raise ValueError("AdvAdapter inputs must share one floating dtype")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("AdvAdapter inputs must contain only finite values")
        if not torch.isfinite(confidence).all() or torch.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError("confidence must contain finite values in [0,1]")

    def residual(
        self,
        advection_features: Tensor,
        e_cond: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        """Return the gated full-width contribution before parent addition."""

        if advection_features.ndim != 4:
            raise ValueError("advection_features must have shape [B,C,H,W]")
        batch, _, height, width = advection_features.shape
        dummy = advection_features.new_zeros(
            (batch, self.target_channels, height, width)
        )
        self._validate(dummy, advection_features, e_cond, confidence)
        features = self.activation(
            self.feature_norm(self.feature_projection(advection_features))
        )
        features = self.activation(
            self.mixing_norm(self.feature_mixing(features))
        )
        projected = self.residual_projection(features)
        condition_gate = torch.sigmoid(self.condition_gate(e_cond))[:, :, None, None]
        # Multiplication by the observed confidence makes confidence=0 an exact
        # off state while the learned mapping itself remains normally initialized.
        spatial_gate = confidence * torch.sigmoid(self.confidence_gate(confidence))
        return condition_gate * spatial_gate * projected

    def forward(
        self,
        target_features: Tensor,
        advection_features: Tensor,
        e_cond: Tensor,
        confidence: Tensor,
    ) -> Tensor:
        """Return target features plus the learned gated advection residual."""

        self._validate(target_features, advection_features, e_cond, confidence)
        return target_features + self.residual(
            advection_features, e_cond, confidence
        )


__all__ = ["AdvAdapter"]
