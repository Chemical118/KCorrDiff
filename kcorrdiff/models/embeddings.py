"""Central condition and EDM-noise embeddings for K-CorrDiff v1.1.3b.

The architecture intentionally has two differently named fusion boundaries:
``ConditionTimeFusionMLP`` combines issue-time information once per lead,
whereas ``DiffusionConditionFusionMLP`` adds the EDM noise level once per
denoising call.  Keeping these modules distinct prevents callers from passing
the internal lead/time embeddings alongside their canonical fused value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import (
    CONDITION_DIM,
    require_float32_tensor,
    require_module_float32,
    require_no_autocast,
)


LEAD_HOURS_MIN = 0.5
LEAD_HOURS_MAX = 6.0
VERIFICATION_FEATURE_DIM = 4
VERIFICATION_EMBED_DIM = 128
SIGMA_FOURIER_BANDS = 32


@dataclass(frozen=True, slots=True)
class ConditionEmbeddingInputs:
    """Lead and issue-time-derived cyclic features; never future observations."""

    lead_hours: Tensor
    verification_cyclic: Tensor

    def __post_init__(self) -> None:
        require_float32_tensor("lead_hours", self.lead_hours, ndim=1)
        require_float32_tensor(
            "verification_cyclic", self.verification_cyclic, ndim=2
        )
        if self.verification_cyclic.shape != (self.lead_hours.shape[0], 4):
            raise ValueError(
                "verification_cyclic must have shape [B,4] matching lead_hours"
            )
        if self.verification_cyclic.device != self.lead_hours.device:
            raise ValueError("condition embedding inputs must share one device")
        if bool(
            (
                (self.lead_hours < LEAD_HOURS_MIN)
                | (self.lead_hours > LEAD_HOURS_MAX)
            ).any().item()
        ):
            raise ValueError("lead_hours must lie in the official [0.5,6.0] range")
        if bool((self.verification_cyclic.abs() > 1.000001).any().item()):
            raise ValueError("verification cyclic features must lie in [-1,1]")


class _LeadEmbedding(nn.Module):
    def __init__(self, *, bands: int = 16) -> None:
        super().__init__()
        if bands <= 0:
            raise ValueError("Fourier bands must be positive")
        self.register_buffer(
            "frequencies",
            torch.arange(1, bands + 1, dtype=torch.float32),
            persistent=True,
        )
        feature_dim = 1 + 2 * bands
        self.input_projection = nn.Linear(feature_dim, CONDITION_DIM)
        self.output_projection = nn.Linear(CONDITION_DIM, CONDITION_DIM)

    def forward(self, lead_hours: Tensor) -> Tensor:
        normalized = lead_hours / LEAD_HOURS_MAX
        phase = 2.0 * math.pi * normalized[:, None] * self.frequencies[None, :]
        features = torch.cat(
            (normalized[:, None], torch.sin(phase), torch.cos(phase)), dim=1
        )
        return self.output_projection(F.silu(self.input_projection(features)))


class _VerificationTimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            VERIFICATION_FEATURE_DIM, VERIFICATION_EMBED_DIM
        )
        self.output_projection = nn.Linear(
            VERIFICATION_EMBED_DIM, VERIFICATION_EMBED_DIM
        )

    def forward(self, cyclic: Tensor) -> Tensor:
        return self.output_projection(F.silu(self.input_projection(cyclic)))


class _ConditionTimeFusionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            CONDITION_DIM + VERIFICATION_EMBED_DIM, CONDITION_DIM
        )
        self.output_projection = nn.Linear(CONDITION_DIM, CONDITION_DIM)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_projection(F.silu(self.input_projection(inputs)))


class ConditionEmbedding(nn.Module):
    """Return only the canonical fused ``e_cond`` tensor of shape ``[B,512]``."""

    output_dim = CONDITION_DIM

    def __init__(self, *, lead_fourier_bands: int = 16) -> None:
        super().__init__()
        self._lead_embedding = _LeadEmbedding(bands=lead_fourier_bands)
        self._verification_embedding = _VerificationTimeEmbedding()
        self._condition_time_fusion = _ConditionTimeFusionMLP()

    def forward(self, inputs: ConditionEmbeddingInputs) -> Tensor:
        require_no_autocast()
        require_module_float32("ConditionEmbedding", self)
        if not isinstance(inputs, ConditionEmbeddingInputs):
            raise TypeError("ConditionEmbedding expects ConditionEmbeddingInputs")
        e_tau = self._lead_embedding(inputs.lead_hours)
        e_time = self._verification_embedding(inputs.verification_cyclic)
        e_cond = self._condition_time_fusion(torch.cat((e_tau, e_time), dim=1))
        if e_cond.dtype is not torch.float32 or e_cond.shape != (
            inputs.lead_hours.shape[0], CONDITION_DIM
        ):
            raise RuntimeError("ConditionEmbedding violated its [B,512] FP32 contract")
        return e_cond


class SigmaEmbedding(nn.Module):
    """Embed a strictly positive EDM noise standard deviation.

    The input contract is the physical ``sigma`` value, not an already
    transformed tensor.  Log-sigma and its Fourier features are created only
    inside this module, which freezes the convention for both training and
    sampling.
    """

    output_dim = CONDITION_DIM

    def __init__(self, *, fourier_bands: int = SIGMA_FOURIER_BANDS) -> None:
        super().__init__()
        if isinstance(fourier_bands, bool) or fourier_bands <= 0:
            raise ValueError("sigma Fourier bands must be a positive integer")
        self.register_buffer(
            "frequencies",
            torch.pow(2.0, torch.arange(fourier_bands, dtype=torch.float32)),
            persistent=True,
        )
        feature_dim = 1 + 2 * fourier_bands
        self.input_projection = nn.Linear(feature_dim, CONDITION_DIM)
        self.output_projection = nn.Linear(CONDITION_DIM, CONDITION_DIM)

    def forward(self, sigma: Tensor) -> Tensor:
        require_no_autocast()
        require_module_float32("SigmaEmbedding", self)
        require_float32_tensor("sigma", sigma, ndim=1)
        if bool((sigma <= 0.0).any().item()):
            raise ValueError("EDM sigma must be strictly positive")
        log_sigma = torch.log(sigma)
        angles = log_sigma[:, None] * self.frequencies[None]
        features = torch.cat(
            (log_sigma[:, None], torch.sin(angles), torch.cos(angles)), dim=1
        )
        result = self.output_projection(F.silu(self.input_projection(features)))
        if result.shape != (sigma.shape[0], CONDITION_DIM):
            raise RuntimeError("SigmaEmbedding violated its [B,512] contract")
        return result


class DiffusionConditionFusionMLP(nn.Module):
    """Combine canonical lead/time condition with the EDM sigma embedding."""

    output_dim = CONDITION_DIM

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2 * CONDITION_DIM, CONDITION_DIM)
        self.output_projection = nn.Linear(CONDITION_DIM, CONDITION_DIM)

    def forward(self, e_cond: Tensor, e_sigma: Tensor) -> Tensor:
        require_no_autocast()
        require_module_float32("DiffusionConditionFusionMLP", self)
        require_float32_tensor("e_cond", e_cond, ndim=2)
        require_float32_tensor("e_sigma", e_sigma, ndim=2)
        if e_cond.shape != e_sigma.shape or e_cond.shape[1:] != (CONDITION_DIM,):
            raise ValueError("e_cond and e_sigma must share shape [B,512]")
        if e_cond.device != e_sigma.device:
            raise ValueError("e_cond and e_sigma must share a device")
        result = self.output_projection(
            F.silu(self.input_projection(torch.cat((e_cond, e_sigma), dim=1)))
        )
        if result.shape != e_cond.shape or result.dtype is not torch.float32:
            raise RuntimeError(
                "DiffusionConditionFusionMLP violated its [B,512] FP32 contract"
            )
        return result


__all__ = [
    "ConditionEmbedding",
    "ConditionEmbeddingInputs",
    "DiffusionConditionFusionMLP",
    "SigmaEmbedding",
]
