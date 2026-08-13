"""Strict masked EDM objective for normalized OOF residuals.

This module is deliberately outside the denoiser model boundary.  The future
target-validity mask is accepted only while constructing the training state
and reducing the loss; it cannot be passed to ``ResidualEDM.denoise``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


SIGMA_DATA = 1.0


def _float_field(name: str, value: Tensor, *, shape: torch.Size) -> None:
    if not isinstance(value, Tensor) or value.dtype is not torch.float32:
        raise TypeError(f"{name} must be a float32 tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")


def _sigma_vector(sigma: Tensor, *, batch: int, device: torch.device) -> None:
    if not isinstance(sigma, Tensor) or sigma.dtype is not torch.float32:
        raise TypeError("sigma must be a float32 tensor")
    if sigma.shape != (batch,) or sigma.device != device:
        raise ValueError("sigma must have shape [B] on the residual device")
    if not bool(torch.isfinite(sigma).all().item()) or bool(
        (sigma <= 0.0).any().item()
    ):
        raise ValueError("sigma must be finite and strictly positive")


def _require_sigma_data(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result != SIGMA_DATA:
        raise ValueError("v1.1.3b normalized-residual sigma_data must equal 1.0")
    return result


@dataclass(frozen=True, slots=True)
class EDMPreconditioning:
    """Broadcastable coefficients from the original EDM parameterization."""

    c_skip: Tensor
    c_out: Tensor
    c_in: Tensor
    c_noise: Tensor

    def __post_init__(self) -> None:
        if self.c_skip.ndim != 4 or self.c_skip.shape[1:] != (1, 1, 1):
            raise ValueError("EDM coefficients must have shape [B,1,1,1]")
        expected = self.c_skip.shape
        for name, value in (
            ("c_out", self.c_out),
            ("c_in", self.c_in),
            ("c_noise", self.c_noise),
        ):
            _float_field(name, value, shape=expected)
            if value.device != self.c_skip.device:
                raise ValueError("EDM coefficients must share one device")


def edm_preconditioning(
    sigma: Tensor, *, sigma_data: float = SIGMA_DATA
) -> EDMPreconditioning:
    """Return EDM coefficients with fixed normalized-residual ``sigma_data``."""

    _require_sigma_data(sigma_data)
    if not isinstance(sigma, Tensor) or sigma.dtype is not torch.float32:
        raise TypeError("sigma must be a float32 tensor")
    if sigma.ndim != 1:
        raise ValueError("sigma must have shape [B]")
    if not bool(torch.isfinite(sigma).all().item()) or bool(
        (sigma <= 0.0).any().item()
    ):
        raise ValueError("sigma must be finite and strictly positive")
    shaped = sigma[:, None, None, None]
    denominator = shaped.square() + 1.0
    return EDMPreconditioning(
        c_skip=1.0 / denominator,
        c_out=shaped / torch.sqrt(denominator),
        c_in=torch.rsqrt(denominator),
        # Karras et al. EDM convention; SigmaEmbedding itself also owns the
        # canonical log-sigma Fourier transform used by this project.
        c_noise=torch.log(shaped) / 4.0,
    )


@dataclass(frozen=True, slots=True)
class NoisyResidualTrainingState:
    """Training-only state; target validity is intentionally not retained."""

    noisy_residual: Tensor
    neutral_filled_clean: Tensor
    noise: Tensor
    sigma: Tensor


def build_noisy_residual_training_state(
    *,
    clean_normalized_residual: Tensor,
    target_validity: Tensor,
    sigma: Tensor,
    noise: Tensor,
) -> NoisyResidualTrainingState:
    """Neutral-fill invalid clean pixels, then add Gaussian noise everywhere."""

    clean = clean_normalized_residual
    if clean.ndim != 4 or clean.shape[1] != 1:
        raise ValueError("clean normalized residual must have shape [B,1,H,W]")
    _float_field("clean normalized residual", clean, shape=clean.shape)
    _float_field("noise", noise, shape=clean.shape)
    if noise.device != clean.device:
        raise ValueError("noise and clean residual must share a device")
    if target_validity.shape != clean.shape or target_validity.dtype is not torch.bool:
        raise TypeError("target_validity must be bool and match the residual shape")
    if target_validity.device != clean.device:
        raise ValueError("target_validity must share the residual device")
    _sigma_vector(sigma, batch=clean.shape[0], device=clean.device)
    if not bool(torch.isfinite(clean[target_validity]).all().item()):
        raise ValueError("valid clean residual contains a non-finite value")
    if not bool(torch.isfinite(noise).all().item()):
        raise ValueError("EDM noise contains a non-finite value")
    neutral = torch.where(target_validity, clean, torch.zeros_like(clean))
    noisy = neutral + sigma[:, None, None, None] * noise
    return NoisyResidualTrainingState(
        noisy_residual=noisy,
        neutral_filled_clean=neutral,
        noise=noise,
        sigma=sigma,
    )


@dataclass(frozen=True, slots=True)
class MaskedEDMLossTerms:
    """Unreduced terms suitable for globally correct DDP reduction."""

    weighted_square_error_sum: Tensor
    valid_importance_mass: Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("weighted_square_error_sum", self.weighted_square_error_sum),
            ("valid_importance_mass", self.valid_importance_mass),
        ):
            if value.shape != () or value.dtype is not torch.float32:
                raise TypeError(f"{name} must be a scalar float32 tensor")

    @property
    def local_mean(self) -> Tensor:
        return self.weighted_square_error_sum / self.valid_importance_mass.clamp_min(
            1.0
        )


def masked_edm_loss_terms(
    *,
    denoised_normalized_residual: Tensor,
    clean_normalized_residual: Tensor,
    target_validity: Tensor,
    omega: Tensor,
    sigma: Tensor,
    sigma_data: float = SIGMA_DATA,
) -> MaskedEDMLossTerms:
    """Compute the exact v1.1.3b masked EDM numerator and denominator.

    Callers doing distributed training must all-reduce both returned scalars
    before division; averaging rank-local normalized losses is not equivalent
    when valid/importance mass differs between ranks.
    """

    _require_sigma_data(sigma_data)
    prediction = denoised_normalized_residual
    clean = clean_normalized_residual
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("EDM residual fields must have shape [B,1,H,W]")
    _float_field("clean normalized residual", clean, shape=prediction.shape)
    _float_field("denoised normalized residual", prediction, shape=prediction.shape)
    if clean.device != prediction.device:
        raise ValueError("EDM prediction and target must share a device")
    if target_validity.shape != prediction.shape or target_validity.dtype is not torch.bool:
        raise TypeError("target_validity must be bool and match prediction")
    if target_validity.device != prediction.device:
        raise ValueError("target_validity must share the prediction device")
    batch = prediction.shape[0]
    _sigma_vector(sigma, batch=batch, device=prediction.device)
    if omega.shape != (batch,) or omega.dtype is not torch.float32:
        raise TypeError("omega must be float32 [B]")
    if omega.device != prediction.device:
        raise ValueError("omega must share the prediction device")
    if not bool(torch.isfinite(omega).all().item()) or bool(
        (omega <= 0.0).any().item()
    ):
        raise ValueError("omega must be finite and strictly positive")
    if not bool(torch.isfinite(prediction).all().item()):
        raise ValueError("EDM prediction contains a non-finite value")
    if not bool(torch.isfinite(clean[target_validity]).all().item()):
        raise ValueError("valid EDM target contains a non-finite value")

    edm_weight = (sigma.square() + 1.0) / sigma.square()
    importance = omega[:, None, None, None] * target_validity.to(torch.float32)
    weighted = importance * edm_weight[:, None, None, None]
    # Multiplication by a zero mask does not neutralize NaN (0 * NaN is NaN),
    # so form the valid-only error explicitly before the weighted reduction.
    error = torch.where(
        target_validity, prediction - clean, torch.zeros_like(prediction)
    )
    numerator = (weighted * error.square()).sum()
    denominator = importance.sum()
    return MaskedEDMLossTerms(
        weighted_square_error_sum=numerator,
        valid_importance_mass=denominator,
    )


__all__ = [
    "EDMPreconditioning",
    "MaskedEDMLossTerms",
    "NoisyResidualTrainingState",
    "SIGMA_DATA",
    "build_noisy_residual_training_state",
    "edm_preconditioning",
    "masked_edm_loss_terms",
]
