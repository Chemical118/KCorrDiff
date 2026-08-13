"""Importance-weighted hurdle regression objectives from protocol v1.1.3b."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as functional

from kcorrdiff.data.radar_values import A0_MM, A_WET_MM


@dataclass(frozen=True, slots=True)
class RegressionLossConfig:
    lambda_occurrence: float = 1.0
    lambda_positive: float = 1.0
    lambda_mean: float = 0.05
    mean_warmup_steps: int = 2_000
    detach_probability_in_mean: bool = False

    def validate(self) -> None:
        values = (
            self.lambda_occurrence,
            self.lambda_positive,
            self.lambda_mean,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("loss coefficients must be finite and non-negative")
        if self.mean_warmup_steps < 0:
            raise ValueError("mean_warmup_steps cannot be negative")


@dataclass(frozen=True, slots=True)
class RegressionLoss:
    total: torch.Tensor
    occurrence: torch.Tensor
    positive_amount: torch.Tensor
    transformed_mean: torch.Tensor
    active_mean_coefficient: float
    valid_weight: torch.Tensor
    wet_weight: torch.Tensor


@dataclass(frozen=True, slots=True)
class GlobalLossDenominators:
    """Global weighted masses for one complete accumulation window."""

    valid_weight: torch.Tensor
    wet_weight: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("valid_weight", self.valid_weight),
            ("wet_weight", self.wet_weight),
        ):
            if value.shape != () or value.dtype is not torch.float64:
                raise TypeError(f"{name} must be a scalar float64 tensor")
            if value.requires_grad or not torch.isfinite(value):
                raise ValueError(f"{name} must be finite and detached")
        if self.valid_weight <= 0.0 or self.wet_weight < 0.0:
            raise ValueError("global valid/wet masses are invalid")


@dataclass(frozen=True, slots=True)
class DistributedRegressionLoss:
    """Local graph contribution and globally reduced scalar diagnostics."""

    local_total: torch.Tensor
    occurrence: torch.Tensor
    positive_amount: torch.Tensor
    transformed_mean: torch.Tensor
    active_mean_coefficient: float
    denominators: GlobalLossDenominators


DetachedSumReducer = Callable[[torch.Tensor], torch.Tensor]


def accumulation_window_denominators(
    labels: Sequence[object],
    *,
    device: torch.device | str,
    reduce_sum: DetachedSumReducer | None = None,
) -> GlobalLossDenominators:
    """Compute weighted masses before forward passes in an accumulation window.

    ``labels`` objects must expose ``target_validity``, ``target_wet`` and
    ``omega`` tensors.  Empty rank-local windows are allowed so explicit
    padding ranks can participate in the same reduction schedule.
    """

    selected = torch.device(device)
    local = torch.zeros(2, dtype=torch.float64, device=selected)
    for item in labels:
        validity = getattr(item, "target_validity", None)
        wet = getattr(item, "target_wet", None)
        omega = getattr(item, "omega", None)
        if not all(isinstance(value, torch.Tensor) for value in (validity, wet, omega)):
            raise TypeError("loss labels do not expose target masks and omega tensors")
        assert isinstance(validity, torch.Tensor)
        assert isinstance(wet, torch.Tensor)
        assert isinstance(omega, torch.Tensor)
        if validity.dtype is not torch.bool or wet.dtype is not torch.bool:
            raise TypeError("target validity/wet masks must be boolean")
        if omega.dtype is not torch.float32 or omega.shape != (validity.shape[0],):
            raise TypeError("omega must be float32 [B]")
        if validity.shape != wet.shape or validity.ndim != 4:
            raise ValueError("target masks must share [B,1,H,W]")
        weights = omega.to(device=selected, dtype=torch.float64)[:, None, None, None]
        valid_on_device = validity.to(device=selected)
        wet_on_device = wet.to(device=selected) & valid_on_device
        local[0] += (weights * valid_on_device).sum()
        local[1] += (weights * wet_on_device).sum()
    global_values = local.detach()
    if reduce_sum is not None:
        global_values = reduce_sum(global_values)
        if not isinstance(global_values, torch.Tensor):
            raise TypeError("reduce_sum must return a tensor")
    if global_values.shape != (2,) or global_values.dtype is not torch.float64:
        raise TypeError("reduced loss masses must be float64 [2]")
    return GlobalLossDenominators(
        valid_weight=global_values[0].detach(),
        wet_weight=global_values[1].detach(),
    )


def _global_detached_metrics(
    local_values: torch.Tensor,
    *,
    reduce_sum: DetachedSumReducer | None,
) -> torch.Tensor:
    detached = local_values.detach().to(dtype=torch.float64)
    result = reduce_sum(detached) if reduce_sum is not None else detached
    if not isinstance(result, torch.Tensor) or result.shape != local_values.shape:
        raise TypeError("reduce_sum changed the metric tensor shape")
    if result.dtype is not torch.float64 or not torch.isfinite(result).all():
        raise TypeError("reduced metrics must be finite float64")
    return result


def distributed_hurdle_loss_contribution(
    *,
    occurrence_logits: torch.Tensor,
    wet_amount: torch.Tensor,
    target_z: torch.Tensor,
    target_wet: torch.Tensor,
    target_validity: torch.Tensor,
    omega: torch.Tensor,
    global_step: int,
    denominators: GlobalLossDenominators,
    config: RegressionLossConfig = RegressionLossConfig(),
    reduce_sum: DetachedSumReducer | None = None,
) -> DistributedRegressionLoss:
    """Return the exact local numerator contribution to a global objective.

    Call backward on ``local_total`` on every rank, then SUM (not average)
    parameter gradients across ranks.  Reusing one ``denominators`` object for
    every microbatch in the accumulation window yields the same objective and
    gradient as concatenating the complete distributed effective batch.
    """

    config.validate()
    if global_step < 0:
        raise ValueError("global_step cannot be negative")
    shape = occurrence_logits.shape
    if occurrence_logits.ndim != 4 or shape[1] != 1:
        raise ValueError("regression fields must have shape [B,1,H,W]")
    for name, tensor in (
        ("wet_amount", wet_amount),
        ("target_z", target_z),
        ("target_wet", target_wet),
        ("target_validity", target_validity),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tensor.shape} does not match {shape}")
    if occurrence_logits.dtype is not torch.float32 or wet_amount.dtype is not torch.float32:
        raise TypeError("model predictions must remain float32")
    if target_z.dtype is not torch.float32:
        raise TypeError("target_z must be float32")
    if target_wet.dtype is not torch.bool or target_validity.dtype is not torch.bool:
        raise TypeError("target wet/validity must be boolean")
    if not torch.isfinite(occurrence_logits).all() or not torch.isfinite(wet_amount).all():
        raise ValueError("regression predictions must be finite")
    valid = target_validity
    if not torch.isfinite(target_z[valid]).all():
        raise ValueError("valid transformed targets must be finite")
    z_wet = math.log1p(A_WET_MM / A0_MM)
    if bool((wet_amount < z_wet - 1.0e-6).any()):
        raise ValueError("wet_amount violates z_wet + softplus support")
    weights = _spatial_weight(omega, valid, dtype=torch.float32)
    wet_mask = target_wet & valid
    wet_weights = weights * wet_mask.to(torch.float32)
    occurrence_values = functional.binary_cross_entropy_with_logits(
        occurrence_logits, target_wet.to(torch.float32), reduction="none"
    )
    probability = torch.sigmoid(occurrence_logits)
    probability_for_mean = (
        probability.detach() if config.detach_probability_in_mean else probability
    )
    local_numerators = torch.stack(
        (
            (occurrence_values * weights).sum(),
            ((target_z - wet_amount).square() * wet_weights).sum(),
            ((target_z - probability_for_mean * wet_amount).square() * weights).sum(),
        )
    )
    valid_denominator = denominators.valid_weight.to(
        device=occurrence_logits.device, dtype=torch.float32
    )
    wet_denominator = denominators.wet_weight.to(
        device=occurrence_logits.device, dtype=torch.float32
    )
    mean_coefficient = (
        config.lambda_mean if global_step >= config.mean_warmup_steps else 0.0
    )
    local_total = (
        config.lambda_occurrence * local_numerators[0] / valid_denominator
        + config.lambda_positive
        * local_numerators[1]
        / wet_denominator.clamp_min(1.0)
        + mean_coefficient * local_numerators[2] / valid_denominator
    )
    global_numerators = _global_detached_metrics(
        local_numerators, reduce_sum=reduce_sum
    )
    return DistributedRegressionLoss(
        local_total=local_total,
        occurrence=(global_numerators[0] / denominators.valid_weight).to(torch.float32),
        positive_amount=(
            global_numerators[1] / denominators.wet_weight.clamp_min(1.0)
        ).to(torch.float32),
        transformed_mean=(
            global_numerators[2] / denominators.valid_weight
        ).to(torch.float32),
        active_mean_coefficient=mean_coefficient,
        denominators=denominators,
    )


def distributed_direct_loss_contribution(
    *,
    statistic: str,
    prediction_mm: torch.Tensor,
    target_mm: torch.Tensor,
    target_validity: torch.Tensor,
    omega: torch.Tensor,
    denominators: GlobalLossDenominators,
    reduce_sum: DetachedSumReducer | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact rank-local contribution for direct physical mean or q50."""

    if statistic not in {"mean", "q50"}:
        raise ValueError("direct statistic must be mean or q50")
    if prediction_mm.shape != target_mm.shape or target_mm.shape != target_validity.shape:
        raise ValueError("direct-loss tensors must have identical shapes")
    if prediction_mm.dtype is not torch.float32 or target_mm.dtype is not torch.float32:
        raise TypeError("direct prediction/target must be float32")
    if target_validity.dtype is not torch.bool:
        raise TypeError("direct target validity must be boolean")
    if bool((prediction_mm < 0.0).any()):
        raise ValueError("physical prediction cannot be negative")
    weights = _spatial_weight(omega, target_validity, dtype=torch.float32)
    if statistic == "mean":
        values = (prediction_mm - target_mm).square()
    else:
        error = target_mm - prediction_mm
        values = torch.maximum(0.5 * error, -0.5 * error)
    local_numerator = (values * weights).sum()
    denominator = denominators.valid_weight.to(
        device=prediction_mm.device, dtype=torch.float32
    )
    local_contribution = local_numerator / denominator
    global_numerator = _global_detached_metrics(
        local_numerator.reshape(1), reduce_sum=reduce_sum
    )[0]
    metric = (global_numerator / denominators.valid_weight).to(torch.float32)
    return local_contribution, metric


def _spatial_weight(
    omega: torch.Tensor,
    mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if omega.ndim == 1:
        omega = omega[:, None, None, None]
    elif omega.ndim == 3:
        omega = omega[:, None]
    if omega.ndim != 4 or omega.shape[0] != mask.shape[0]:
        raise ValueError("omega must have shape [B], [B,H,W], or [B,1,H,W]")
    if not torch.isfinite(omega).all() or bool((omega <= 0).any()):
        raise ValueError("importance weights must be finite and strictly positive")
    return omega.to(device=mask.device, dtype=dtype) * mask.to(dtype=dtype)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weights.sum()
    # A connected zero keeps DDP gradient synchronization well-defined when a
    # microbatch happens to contain no wet pixels.
    result = (values * weights).sum() / denominator.clamp_min(1.0)
    return result, denominator.detach()


def hurdle_regression_loss(
    *,
    occurrence_logits: torch.Tensor,
    wet_amount: torch.Tensor,
    target_z: torch.Tensor,
    target_wet: torch.Tensor,
    target_validity: torch.Tensor,
    omega: torch.Tensor,
    global_step: int,
    config: RegressionLossConfig = RegressionLossConfig(),
) -> RegressionLoss:
    """Compute BCE, wet-amount MSE, and transformed-mean MSE.

    Future-derived ``target_validity`` participates only here. It is never a
    model condition. All reductions use the full item importance weight.
    """

    config.validate()
    if global_step < 0:
        raise ValueError("global_step cannot be negative")
    shape = occurrence_logits.shape
    if occurrence_logits.ndim != 4 or shape[1] != 1:
        raise ValueError("regression fields must have shape [B,1,H,W]")
    for name, tensor in (
        ("wet_amount", wet_amount),
        ("target_z", target_z),
        ("target_wet", target_wet),
        ("target_validity", target_validity),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} shape {tensor.shape} does not match {shape}")
    if occurrence_logits.dtype != torch.float32 or wet_amount.dtype != torch.float32:
        raise TypeError("model predictions must remain float32")
    if target_z.dtype != torch.float32:
        raise TypeError("target_z must be float32")
    if not torch.isfinite(occurrence_logits).all() or not torch.isfinite(wet_amount).all():
        raise ValueError("regression predictions must be finite")
    if not torch.isfinite(target_z[target_validity.bool()]).all():
        raise ValueError("valid transformed targets must be finite")
    z_wet = math.log1p(A_WET_MM / A0_MM)
    if bool((wet_amount < z_wet - 1.0e-6).any()):
        raise ValueError("wet_amount violates z_wet + softplus support")

    valid = target_validity.bool()
    wet = target_wet.bool() & valid
    weights = _spatial_weight(omega, valid, dtype=occurrence_logits.dtype)
    wet_weights = weights * wet.to(weights.dtype)
    occurrence_values = functional.binary_cross_entropy_with_logits(
        occurrence_logits,
        target_wet.to(occurrence_logits.dtype),
        reduction="none",
    )
    occurrence, valid_weight = _weighted_mean(occurrence_values, weights)
    positive, wet_weight = _weighted_mean((target_z - wet_amount).square(), wet_weights)
    probability = torch.sigmoid(occurrence_logits)
    probability_for_mean = probability.detach() if config.detach_probability_in_mean else probability
    transformed_mean = probability_for_mean * wet_amount
    mean_loss, _ = _weighted_mean((target_z - transformed_mean).square(), weights)
    mean_coefficient = (
        config.lambda_mean if global_step >= config.mean_warmup_steps else 0.0
    )
    total = (
        config.lambda_occurrence * occurrence
        + config.lambda_positive * positive
        + mean_coefficient * mean_loss
    )
    return RegressionLoss(
        total=total,
        occurrence=occurrence.detach(),
        positive_amount=positive.detach(),
        transformed_mean=mean_loss.detach(),
        active_mean_coefficient=mean_coefficient,
        valid_weight=valid_weight,
        wet_weight=wet_weight,
    )


def direct_physical_mean_loss(
    prediction_mm: torch.Tensor,
    target_mm: torch.Tensor,
    target_validity: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """MSE whose population optimum is the physical conditional mean."""

    if prediction_mm.shape != target_mm.shape or target_mm.shape != target_validity.shape:
        raise ValueError("direct-mean tensors must have identical shapes")
    if bool((prediction_mm < 0).any()):
        raise ValueError("physical accumulation prediction cannot be negative")
    weights = _spatial_weight(omega, target_validity.bool(), dtype=prediction_mm.dtype)
    result, _ = _weighted_mean((prediction_mm - target_mm).square(), weights)
    return result


def direct_physical_quantile_loss(
    prediction_mm: torch.Tensor,
    target_mm: torch.Tensor,
    target_validity: torch.Tensor,
    omega: torch.Tensor,
    *,
    quantile: float = 0.5,
) -> torch.Tensor:
    """Weighted pinball loss for the explicit direct q50 comparison arm."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly inside (0, 1)")
    if prediction_mm.shape != target_mm.shape or target_mm.shape != target_validity.shape:
        raise ValueError("direct-quantile tensors must have identical shapes")
    error = target_mm - prediction_mm
    pinball = torch.maximum(quantile * error, (quantile - 1.0) * error)
    weights = _spatial_weight(omega, target_validity.bool(), dtype=prediction_mm.dtype)
    result, _ = _weighted_mean(pinball, weights)
    return result


__all__ = [
    "DistributedRegressionLoss",
    "GlobalLossDenominators",
    "RegressionLoss",
    "RegressionLossConfig",
    "direct_physical_mean_loss",
    "direct_physical_quantile_loss",
    "distributed_direct_loss_contribution",
    "distributed_hurdle_loss_contribution",
    "hurdle_regression_loss",
    "accumulation_window_denominators",
]
