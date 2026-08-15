"""Frozen Stage 3 residual and probability calibration primitives.

The functions implement architecture v1.1.3b section 12.4 without selecting
an arm from calibration labels.  A caller must supply the already-frozen
``d_enabled`` decision and exact checkpoint/sampler identities when publishing
the resulting table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor

from kcorrdiff.data.radar_values import A0_MM, A_WET_MM


CALIBRATION_SPLIT = "calibration"
POOLING_ORDER = (
    "full_cell",
    "lead_provider_era_present",
    "lead_provider",
    "lead_only",
)
PoolingLevel = Literal[
    "full_cell",
    "lead_provider_era_present",
    "lead_provider",
    "lead_only",
]


def block_ess_tolerance(block_count: int) -> float:
    """Shared numerical tolerance for the theoretical ESS upper bound."""

    return 1.0e-12 * max(1.0, float(block_count))


def _field(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return result


def _weighted_moments(
    value: ArrayLike, weight: ArrayLike
) -> tuple[float, float, float]:
    values = _field(value, name="values")
    weights = _field(weight, name="weights")
    if values.shape != weights.shape:
        raise ValueError("values and weights must share one shape")
    selected = weights > 0.0
    if not np.any(selected):
        raise ValueError("weighted moment has no positive mass")
    if not np.all(np.isfinite(values[selected])):
        raise ValueError("positive-weight values must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    mass = float(weights.sum())
    if not math.isfinite(mass) or mass <= 0.0:
        raise OverflowError("weighted moment mass overflowed float64")
    mean = float(np.sum(weights * np.where(selected, values, 0.0)) / mass)
    variance = float(
        np.sum(weights * np.where(selected, (values - mean) ** 2, 0.0)) / mass
    )
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise OverflowError("weighted moment statistics overflowed float64")
    return mean, max(variance, 0.0), mass


@dataclass(frozen=True, slots=True)
class IndependentBlockSupport:
    """Independent-block support used by the frozen pooling ladder."""

    block_count: int
    block_ess: float
    positive_support_blocks: int
    negative_support_blocks: int

    def __post_init__(self) -> None:
        integers = (
            self.block_count,
            self.positive_support_blocks,
            self.negative_support_blocks,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in integers
        ):
            raise ValueError("block support counts must be non-negative integers")
        if (
            isinstance(self.block_ess, bool)
            or not isinstance(self.block_ess, Real)
            or not math.isfinite(self.block_ess)
            or self.block_ess < 0.0
        ):
            raise ValueError("block ESS must be finite and non-negative")
        if (self.block_count == 0) != (self.block_ess == 0.0):
            raise ValueError("zero block count and zero block ESS must agree")
        tolerance = block_ess_tolerance(self.block_count)
        if self.block_ess > self.block_count + tolerance:
            raise ValueError("block ESS cannot exceed block count")
        if self.positive_support_blocks > self.block_count:
            raise ValueError("positive block support cannot exceed block count")
        if self.negative_support_blocks > self.block_count:
            raise ValueError("negative block support cannot exceed block count")

    def passes_residual_gate(
        self, *, minimum_blocks: int = 30, minimum_ess: float = 20.0
    ) -> bool:
        return self.block_count >= minimum_blocks and self.block_ess >= minimum_ess

    def passes_probability_gate(
        self,
        *,
        minimum_blocks: int = 30,
        minimum_ess: float = 20.0,
        minimum_positive_blocks: int = 20,
        minimum_negative_blocks: int = 20,
    ) -> bool:
        return (
            self.passes_residual_gate(
                minimum_blocks=minimum_blocks, minimum_ess=minimum_ess
            )
            and self.positive_support_blocks >= minimum_positive_blocks
            and self.negative_support_blocks >= minimum_negative_blocks
        )


def independent_block_support(
    *,
    block_id: Sequence[str],
    weight: ArrayLike,
    observation: ArrayLike | None = None,
) -> IndependentBlockSupport:
    """Compute support after aggregating valid mass within independent blocks.

    A block may support both binary classes when it contains positive-weight
    entries from both classes.  ESS is computed from aggregate block mass,
    never from the much larger number of overlapping pixels/windows.
    """

    weights = np.asarray(weight, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        if len(block_id) != 0:
            raise ValueError("one block_id is required per weight")
        if observation is not None:
            labels = np.asarray(observation, dtype=np.float64).reshape(-1)
            if labels.size != 0:
                raise ValueError("block observations must match weights")
        # A genuinely empty narrow cell is the same absence of support as a
        # populated cell whose weights are all zero.  It must not abort the
        # deterministic fallback to a broader pooling rung.
        return IndependentBlockSupport(0, 0.0, 0, 0)
    if len(block_id) != weights.size or any(
        not isinstance(value, str) or not value or value.strip() != value
        for value in block_id
    ):
        raise ValueError(
            "one canonical non-empty string block_id is required per weight"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("block weights must be finite and non-negative")
    labels: NDArray[np.float64] | None = None
    if observation is not None:
        labels = _field(observation, name="block observations").reshape(-1)
        if labels.shape != weights.shape or not np.all(np.isfinite(labels)):
            raise ValueError("block observations must be finite and match weights")
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("block observations must be binary")
    masses: dict[str, float] = {}
    positive: set[str] = set()
    negative: set[str] = set()
    for index, block in enumerate(block_id):
        item_weight = float(weights[index])
        if item_weight <= 0.0:
            continue
        masses[block] = masses.get(block, 0.0) + item_weight
        if not math.isfinite(masses[block]):
            raise OverflowError("independent-block mass overflowed float64")
        if labels is not None:
            (positive if labels[index] == 1.0 else negative).add(block)
    if not masses:
        # An empty/zero-mass narrow cell is ordinary evidence that this rung
        # cannot pass.  Returning explicit zero support lets the deterministic
        # ladder continue to broader cells instead of aborting calibration.
        return IndependentBlockSupport(0, 0.0, 0, 0)
    values = np.asarray(tuple(masses.values()), dtype=np.float64)
    scale = float(values.max())
    normalized = values / scale
    ess = float(normalized.sum() ** 2 / np.square(normalized).sum())
    if not math.isfinite(ess):
        raise OverflowError("independent-block ESS overflowed float64")
    return IndependentBlockSupport(
        block_count=len(masses),
        block_ess=ess,
        positive_support_blocks=len(positive),
        negative_support_blocks=len(negative),
    )


@dataclass(frozen=True, slots=True)
class PoolingDecision:
    level: PoolingLevel
    support: IndependentBlockSupport
    probability_gate: bool


def select_pooling_level(
    support_by_level: Mapping[str, IndependentBlockSupport],
    *,
    probability_gate: bool,
) -> PoolingDecision:
    """Apply the predeclared pooling order without inspecting fit quality."""

    missing = [level for level in POOLING_ORDER if level not in support_by_level]
    if missing:
        raise ValueError("pooling support is missing: " + ", ".join(missing))
    extras = sorted(set(support_by_level) - set(POOLING_ORDER))
    if extras:
        raise ValueError("unknown pooling support levels: " + ", ".join(extras))
    for raw_level in POOLING_ORDER:
        support = support_by_level[raw_level]
        passed = (
            support.passes_probability_gate()
            if probability_gate
            else support.passes_residual_gate()
        )
        if passed:
            return PoolingDecision(
                level=raw_level,  # type: ignore[arg-type]
                support=support,
                probability_gate=probability_gate,
            )
    raise ValueError("no predeclared calibration pooling level has sufficient support")


@dataclass(frozen=True, slots=True)
class FoldCalibrationMoments:
    fold_id: int
    oof_valid_mass: float
    calibration_mean: float
    calibration_second_moment: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_id, bool)
            or not isinstance(self.fold_id, Integral)
            or self.fold_id < 0
        ):
            raise ValueError("fold_id must be non-negative")
        values = (
            self.oof_valid_mass,
            self.calibration_mean,
            self.calibration_second_moment,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError("fold calibration moments must be real scalars")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("fold calibration moments must be finite")
        if self.oof_valid_mass <= 0.0 or self.calibration_second_moment < 0.0:
            raise ValueError("fold mass must be positive and q_k non-negative")
        squared_mean = float(self.calibration_mean) * float(self.calibration_mean)
        if not math.isfinite(squared_mean):
            raise ValueError("fold mean squared must remain finite")
        tolerance = 1.0e-12 * max(
            1.0, abs(self.calibration_second_moment), squared_mean
        )
        if self.calibration_second_moment + tolerance < squared_mean:
            raise ValueError("fold second moment q_k cannot be smaller than m_k squared")


@dataclass(frozen=True, slots=True)
class LocationScaleCalibration:
    location_b: float
    total_scale_c: float
    fold_weights: tuple[float, ...]
    fold_mixture_mean: float
    fold_mixture_variance: float
    full_mean: float
    full_variance: float

    def __post_init__(self) -> None:
        scalars = (
            self.location_b,
            self.total_scale_c,
            self.fold_mixture_mean,
            self.fold_mixture_variance,
            self.full_mean,
            self.full_variance,
            *self.fold_weights,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in scalars):
            raise TypeError("location/scale calibration must contain real scalars")
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("location/scale calibration must be finite")
        if self.total_scale_c <= 0.0:
            raise ValueError("total residual scale c must be positive")
        if self.fold_mixture_variance < 0.0 or self.full_variance < 0.0:
            raise ValueError("residual variances must be non-negative")
        if (
            not self.fold_weights
            or any(weight < 0.0 for weight in self.fold_weights)
            or not math.isclose(
                sum(self.fold_weights), 1.0, rel_tol=0.0, abs_tol=1.0e-12
            )
        ):
            raise ValueError("fold weights must form a probability vector")


def fit_location_total_scale(
    *,
    folds: Sequence[FoldCalibrationMoments],
    full_residual: ArrayLike,
    calibration_weight: ArrayLike,
    split: str,
    epsilon: float = 1.0e-12,
) -> LocationScaleCalibration:
    """Fit ``b,c`` from fold-mixture and full-regression residual moments."""

    if split != CALIBRATION_SPLIT:
        raise ValueError("final b/c may be fit only on the independent calibration split")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    ordered = tuple(sorted(folds, key=lambda item: item.fold_id))
    if len(ordered) < 2 or len({item.fold_id for item in ordered}) != len(ordered):
        raise ValueError("at least two unique fold moment records are required")
    total_mass = sum(item.oof_valid_mass for item in ordered)
    pi = tuple(item.oof_valid_mass / total_mass for item in ordered)
    mixture_mean = sum(weight * item.calibration_mean for weight, item in zip(pi, ordered))
    mixture_second = sum(
        weight * item.calibration_second_moment
        for weight, item in zip(pi, ordered)
    )
    mixture_variance = max(mixture_second - mixture_mean**2, 0.0)
    full_mean, full_variance, _ = _weighted_moments(
        full_residual, calibration_weight
    )
    scale = math.sqrt(full_variance / max(mixture_variance, epsilon))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("calibration produced an invalid total scale")
    location = full_mean - scale * mixture_mean
    return LocationScaleCalibration(
        location_b=location,
        total_scale_c=scale,
        fold_weights=pi,
        fold_mixture_mean=mixture_mean,
        fold_mixture_variance=mixture_variance,
        full_mean=full_mean,
        full_variance=full_variance,
    )


def identity_location_total_scale(
    *,
    folds: Sequence[FoldCalibrationMoments],
    full_residual: ArrayLike,
    calibration_weight: ArrayLike,
    split: str,
) -> LocationScaleCalibration:
    """Build the documented terminal ``b=0,c=1`` calibration record.

    The moments are retained as audit diagnostics, but they never influence
    the identity parameters.  This is intentionally separate from
    :func:`fit_location_total_scale`: sparse terminal cells must not attempt a
    fit and then silently publish whatever that fit happened to return.
    """

    if split != CALIBRATION_SPLIT:
        raise ValueError(
            "terminal b/c identity may be recorded only on the independent calibration split"
        )
    ordered = tuple(sorted(folds, key=lambda item: item.fold_id))
    if len(ordered) < 2 or len({item.fold_id for item in ordered}) != len(ordered):
        raise ValueError("at least two unique fold moment records are required")
    total_mass = sum(item.oof_valid_mass for item in ordered)
    pi = tuple(item.oof_valid_mass / total_mass for item in ordered)
    mixture_mean = sum(
        weight * item.calibration_mean for weight, item in zip(pi, ordered)
    )
    mixture_second = sum(
        weight * item.calibration_second_moment
        for weight, item in zip(pi, ordered)
    )
    mixture_variance = max(mixture_second - mixture_mean**2, 0.0)
    full_mean, full_variance, _ = _weighted_moments(
        full_residual, calibration_weight
    )
    return LocationScaleCalibration(
        location_b=0.0,
        total_scale_c=1.0,
        fold_weights=pi,
        fold_mixture_mean=mixture_mean,
        fold_mixture_variance=mixture_variance,
        full_mean=full_mean,
        full_variance=full_variance,
    )


@dataclass(frozen=True, slots=True)
class SamplerCalibration:
    sampler_bias_d: float
    spread_gamma: float
    d_enabled: bool
    bias_fraction: float

    def __post_init__(self) -> None:
        values = (self.sampler_bias_d, self.spread_gamma, self.bias_fraction)
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise TypeError("sampler calibration must contain real scalars")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sampler calibration must be finite")
        if not isinstance(self.d_enabled, bool):
            raise TypeError("d_enabled must be a frozen boolean")
        if self.spread_gamma <= 0.0 or self.bias_fraction < 0.0:
            raise ValueError("gamma must be positive and bias_fraction non-negative")
        if not self.d_enabled and self.sampler_bias_d != 0.0:
            raise ValueError("d must be exactly zero when its arm is disabled")


def fit_sampler_bias_and_spread(
    *,
    restored_members: ArrayLike,
    full_residual: ArrayLike,
    calibration_weight: ArrayLike,
    location_scale: LocationScaleCalibration,
    d_enabled: bool,
    split: str,
    epsilon: float = 1.0e-12,
) -> SamplerCalibration:
    """Fit frozen-arm sampler ``d`` and mean-preserving anomaly ``gamma``."""

    if split != CALIBRATION_SPLIT:
        raise ValueError("final d/gamma require the independent calibration split")
    if not isinstance(d_enabled, bool):
        raise TypeError("d_enabled must be a frozen boolean")
    members = _field(restored_members, name="restored_members")
    residual = _field(full_residual, name="full_residual")
    weight = _field(calibration_weight, name="calibration_weight")
    if members.ndim < 2 or members.shape[0] != residual.shape[0]:
        raise ValueError("members must have shape [items,N,...]")
    if members.shape[2:] != residual.shape[1:] or residual.shape != weight.shape:
        raise ValueError("member, residual, and weight shapes disagree")
    member_count = members.shape[1]
    if member_count < 2:
        raise ValueError("gamma requires at least two ensemble members")
    member_mean = members.mean(axis=1)
    e0 = residual - (
        location_scale.location_b + location_scale.total_scale_c * member_mean
    )
    mean_e0, _variance_e0, _ = _weighted_moments(e0, weight)
    rmse = math.sqrt(_weighted_moments(e0**2, weight)[0])
    bias_fraction = abs(mean_e0) / max(rmse, epsilon)
    bias = mean_e0 if d_enabled else 0.0
    calibrated_members = (
        location_scale.location_b
        + bias
        + location_scale.total_scale_c * members
    )
    mean_calibrated = calibrated_members.mean(axis=1)
    error = residual - mean_calibrated
    sample_variance = calibrated_members.var(axis=1, ddof=1)
    numerator = float(np.sum(weight * np.where(weight > 0.0, error**2, 0.0)))
    denominator = float(
        np.sum(weight * (1.0 + 1.0 / member_count) * sample_variance)
    )
    gamma = math.sqrt(numerator / max(denominator, epsilon))
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("calibration produced an invalid spread gamma")
    return SamplerCalibration(
        sampler_bias_d=bias,
        spread_gamma=gamma,
        d_enabled=d_enabled,
        bias_fraction=bias_fraction,
    )


def identity_sampler_bias_and_spread(
    *,
    restored_members: ArrayLike,
    full_residual: ArrayLike,
    calibration_weight: ArrayLike,
    location_scale: LocationScaleCalibration,
    d_enabled: bool,
    split: str,
    epsilon: float = 1.0e-12,
) -> SamplerCalibration:
    """Build documented terminal ``d=0,gamma=1`` parameters.

    Inputs still undergo the same shape, finiteness, and positive-mass checks
    as a real fit.  ``bias_fraction`` remains a diagnostic; it cannot activate
    the frozen ``d`` arm in a terminal cell.
    """

    if split != CALIBRATION_SPLIT:
        raise ValueError(
            "terminal d/gamma identity requires the independent calibration split"
        )
    if not isinstance(d_enabled, bool):
        raise TypeError("d_enabled must be a frozen boolean")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    members = _field(restored_members, name="restored_members")
    residual = _field(full_residual, name="full_residual")
    weight = _field(calibration_weight, name="calibration_weight")
    if members.ndim < 2 or members.shape[0] != residual.shape[0]:
        raise ValueError("members must have shape [items,N,...]")
    if members.shape[2:] != residual.shape[1:] or residual.shape != weight.shape:
        raise ValueError("member, residual, and weight shapes disagree")
    if members.shape[1] < 2:
        raise ValueError("gamma requires at least two ensemble members")
    if not np.all(np.isfinite(members)):
        raise ValueError("restored_members must be finite")
    member_mean = members.mean(axis=1)
    e0 = residual - (
        location_scale.location_b + location_scale.total_scale_c * member_mean
    )
    mean_e0, _variance_e0, _ = _weighted_moments(e0, weight)
    rmse = math.sqrt(_weighted_moments(e0**2, weight)[0])
    return SamplerCalibration(
        sampler_bias_d=0.0,
        spread_gamma=1.0,
        d_enabled=d_enabled,
        bias_fraction=abs(mean_e0) / max(rmse, epsilon),
    )


def apply_residual_calibration(
    restored_members: Tensor,
    *,
    location_b: float,
    total_scale_c: float,
    sampler_bias_d: float,
    spread_gamma: float,
) -> Tensor:
    """Apply ``b,d,c,gamma`` while preserving the post-``b/d/c`` member mean."""

    if restored_members.ndim != 5:
        raise ValueError("residual members must have shape [B,N,1,H,W]")
    if restored_members.dtype is not torch.float32:
        raise TypeError("residual members must remain float32")
    values = (location_b, total_scale_c, sampler_bias_d, spread_gamma)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("calibration parameters must be finite")
    if total_scale_c <= 0.0 or spread_gamma <= 0.0:
        raise ValueError("c and gamma must be positive")
    if not bool(torch.isfinite(restored_members).all().item()):
        raise ValueError("restored_members must be finite")
    if (
        location_b == 0.0
        and total_scale_c == 1.0
        and sampler_bias_d == 0.0
        and spread_gamma == 1.0
    ):
        # The documented terminal identity is a literal raw pass-through.
        # Avoiding an algebraically equivalent mean/anomaly reconstruction is
        # important because that reconstruction is not bitwise neutral in
        # float32.
        return restored_members
    first = location_b + sampler_bias_d + total_scale_c * restored_members
    if not bool(torch.isfinite(first).all().item()):
        raise OverflowError("residual location/scale calibration overflowed float32")
    if spread_gamma == 1.0:
        return first
    member_mean = first.mean(dim=1, keepdim=True)
    result = member_mean + spread_gamma * (first - member_mean)
    if not bool(torch.isfinite(result).all().item()):
        raise OverflowError("residual spread calibration overflowed float32")
    return result


def transformed_members_to_physical(
    mu_z_full: Tensor, calibrated_residual_members: Tensor
) -> Tensor:
    """Member-wise inverse transform followed by the fixed 0.1 mm censor."""

    residual = calibrated_residual_members
    if residual.ndim != 5 or residual.shape[2] != 1:
        raise ValueError("calibrated residuals must have shape [B,N,1,H,W]")
    if mu_z_full.shape != (residual.shape[0], 1, *residual.shape[-2:]):
        raise ValueError("mu_z_full must have shape [B,1,H,W]")
    if residual.dtype is not torch.float32 or mu_z_full.dtype is not torch.float32:
        raise TypeError("transformed forecast fields must remain float32")
    z = mu_z_full[:, None] + residual
    if not bool(torch.isfinite(z).all().item()):
        raise OverflowError("transformed forecast members overflowed float32")
    positive_z = torch.clamp_min(z, 0.0)
    # ``torch.expm1`` returns +inf for otherwise finite float32 inputs above
    # the representable physical-amount range.  A finalized forecast may never
    # contain that silent overflow, so fail before applying the inverse.
    maximum_z = math.log1p(torch.finfo(torch.float32).max / A0_MM)
    if bool((positive_z > maximum_z).any().item()):
        raise OverflowError(
            "finite transformed forecast exceeds the representable float32 physical range"
        )
    amount = A0_MM * torch.expm1(positive_z)
    if not bool(torch.isfinite(amount).all().item()):
        raise OverflowError("physical inverse transform overflowed float32")
    return torch.where(amount >= A_WET_MM, amount, torch.zeros_like(amount))


def empirical_lower_median(members: Tensor) -> Tensor:
    """Return the empirical lower median without even-N interpolation."""

    if members.ndim < 2 or members.shape[1] <= 0:
        raise ValueError("members require a non-empty ensemble axis")
    rank = (members.shape[1] - 1) // 2
    return members.kthvalue(rank + 1, dim=1).values


def monotone_logit_linear_probability(
    probability: Tensor, *, alpha: Tensor | float, beta_raw: Tensor | float
) -> Tensor:
    """Predeclared monotone probability calibration family."""

    if probability.dtype is not torch.float32:
        raise TypeError("probability must remain float32")
    if not bool(torch.isfinite(probability).all().item()) or bool(
        ((probability < 0.0) | (probability > 1.0)).any().item()
    ):
        raise ValueError("probability must be finite and lie in [0,1]")
    clipped = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
    logit = torch.logit(clipped)
    alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32, device=probability.device)
    beta_tensor = torch.as_tensor(beta_raw, dtype=torch.float32, device=probability.device)
    if alpha_tensor.numel() != 1 or beta_tensor.numel() != 1:
        raise ValueError("probability calibration parameters must be scalars")
    if not bool(torch.isfinite(alpha_tensor).all().item()) or not bool(
        torch.isfinite(beta_tensor).all().item()
    ):
        raise OverflowError("probability calibration parameters overflowed float32")
    slope = torch.nn.functional.softplus(beta_tensor)
    if not bool(torch.isfinite(slope).all().item()) or not bool((slope > 0.0).all().item()):
        raise OverflowError(
            "softplus(beta_raw) must remain finite and strictly positive in float32"
        )
    result = torch.sigmoid(alpha_tensor + slope * logit)
    if not bool(torch.isfinite(result).all().item()) or bool(
        ((result < 0.0) | (result > 1.0)).any().item()
    ):
        raise OverflowError("probability calibration mapping produced invalid float32")
    return result


@dataclass(frozen=True, slots=True)
class ProbabilityCalibration:
    """Frozen parameters for the predeclared monotone logit-linear family."""

    alpha: float
    beta_raw: float
    slope: float
    weighted_log_loss: float
    iterations: int
    pooling: PoolingDecision | None
    identity: bool = False

    def __post_init__(self) -> None:
        scalars = self.alpha, self.beta_raw, self.slope, self.weighted_log_loss
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in scalars):
            raise TypeError("probability calibration must contain real scalars")
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("probability calibration parameters must be finite")
        execution_parameters = torch.tensor(
            [self.alpha, self.beta_raw, self.slope], dtype=torch.float32
        )
        if not bool(torch.isfinite(execution_parameters).all().item()):
            raise ValueError(
                "probability calibration parameters must be representable in float32"
            )
        if (
            self.slope <= 0.0
            or execution_parameters[2].item() <= 0.0
            or self.weighted_log_loss < 0.0
        ):
            raise ValueError("probability slope must be positive and loss non-negative")
        # Validate the parameterization in the same dtype used by inference.
        # A double-precision softplus plus a fixed absolute tolerance can
        # approve a tiny stored slope even when float32 softplus underflows to
        # zero and the operational mapping becomes flat.  A relative-only
        # comparison remains meaningful at every nonzero scale.
        expected_slope = float(
            torch.nn.functional.softplus(execution_parameters[1]).item()
        )
        if not math.isfinite(expected_slope) or expected_slope <= 0.0:
            raise ValueError(
                "softplus(beta_raw) must remain finite and strictly positive in float32"
            )
        float32_relative_tolerance = 8.0 * float(torch.finfo(torch.float32).eps)
        if not math.isclose(
            self.slope,
            expected_slope,
            rel_tol=float32_relative_tolerance,
            abs_tol=0.0,
        ):
            raise ValueError(
                "probability slope must equal float32 softplus(beta_raw)"
            )
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, Integral)
            or self.iterations < 0
        ):
            raise ValueError("probability calibration iterations must be non-negative")
        if not isinstance(self.identity, bool):
            raise TypeError("probability identity flag must be boolean")
        if self.identity:
            if self.pooling is not None:
                raise ValueError("terminal probability identity cannot claim a pooling decision")
            if self.alpha != 0.0 or self.slope != 1.0:
                raise ValueError("terminal probability identity must use alpha=0 and slope=1")
        elif self.pooling is None or not self.pooling.probability_gate:
            raise ValueError("probability calibration requires probability support")


def _inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value + math.log1p(-math.exp(-value))
    return math.log(math.expm1(value))


def fit_monotone_logit_linear_probability(
    probability: ArrayLike,
    observation: ArrayLike,
    weight: ArrayLike,
    *,
    split: str,
    pooling: PoolingDecision,
    maximum_iterations: int = 500,
) -> ProbabilityCalibration:
    """Fit the frozen two-parameter probability family on calibration only.

    Optimization is convex in intercept and the constrained positive slope.
    ``beta_raw`` is derived only after fitting so applying the result uses the
    exact public ``softplus(beta_raw)`` parameterization.
    """

    if split != CALIBRATION_SPLIT:
        raise ValueError(
            "final probability mapping requires the independent calibration split"
        )
    if not pooling.probability_gate or not pooling.support.passes_probability_gate():
        raise ValueError("probability mapping lacks independent block support")
    if isinstance(maximum_iterations, bool) or maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be a positive integer")
    forecast = _field(probability, name="probability").reshape(-1)
    target = _field(observation, name="observation").reshape(-1)
    weights = _field(weight, name="probability weight").reshape(-1)
    if not (forecast.shape == target.shape == weights.shape):
        raise ValueError("probability, observation, and weight must share shape")
    if not np.all(np.isfinite(forecast)) or np.any((forecast < 0.0) | (forecast > 1.0)):
        raise ValueError("probability must be finite and lie in [0,1]")
    if not np.all(np.isfinite(target)) or np.any((target != 0.0) & (target != 1.0)):
        raise ValueError("probability observations must be binary")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probability weights must be finite and non-negative")
    selected = weights > 0.0
    if not np.any(selected):
        raise ValueError("probability calibration has no positive weight")
    forecast = np.clip(forecast[selected], 1.0e-6, 1.0 - 1.0e-6)
    target = target[selected]
    weights = weights[selected]
    if not (np.any(target == 0.0) and np.any(target == 1.0)):
        raise ValueError("probability calibration requires both observed classes")
    predictor = np.log(forecast) - np.log1p(-forecast)
    mass = float(weights.sum())
    predictor_mean = float(np.sum(weights * predictor) / mass)
    predictor_variance = float(
        np.sum(weights * (predictor - predictor_mean) ** 2) / mass
    )
    if predictor_variance <= 1.0e-12:
        raise ValueError("probability calibration predictor is not identifiable")
    event_rate = float(np.sum(weights * target) / mass)
    event_rate = min(max(event_rate, 1.0e-6), 1.0 - 1.0e-6)
    initial = np.asarray(
        [math.log(event_rate / (1.0 - event_rate)), 1.0], dtype=np.float64
    )

    def objective(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        alpha, slope = float(parameters[0]), float(parameters[1])
        linear = alpha + slope * predictor
        loss = float(np.sum(weights * (np.logaddexp(0.0, linear) - target * linear)) / mass)
        fitted = np.empty_like(linear)
        positive = linear >= 0.0
        fitted[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
        exponential = np.exp(linear[~positive])
        fitted[~positive] = exponential / (1.0 + exponential)
        residual = fitted - target
        gradient = np.asarray(
            [
                np.sum(weights * residual) / mass,
                np.sum(weights * residual * predictor) / mass,
            ],
            dtype=np.float64,
        )
        return loss, gradient

    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - production dependency gate.
        raise RuntimeError("SciPy is required for probability calibration") from error
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=((None, None), (1.0e-8, None)),
        options={"maxiter": maximum_iterations, "ftol": 1.0e-12, "gtol": 1.0e-9},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"probability calibration optimization failed: {result.message}")
    alpha = float(result.x[0])
    slope = float(result.x[1])
    beta_raw = _inverse_softplus(slope)
    loss, _ = objective(np.asarray([alpha, slope], dtype=np.float64))
    return ProbabilityCalibration(
        alpha=alpha,
        beta_raw=beta_raw,
        slope=slope,
        weighted_log_loss=loss,
        iterations=int(result.nit),
        pooling=pooling,
    )


def identity_monotone_logit_linear_probability(
    probability: ArrayLike,
    observation: ArrayLike,
    weight: ArrayLike,
    *,
    split: str,
) -> ProbabilityCalibration:
    """Return an explicitly flagged raw pass-through terminal mapping."""

    if split != CALIBRATION_SPLIT:
        raise ValueError(
            "terminal probability identity requires the independent calibration split"
        )
    forecast = _field(probability, name="probability").reshape(-1)
    target = _field(observation, name="observation").reshape(-1)
    weights = _field(weight, name="probability weight").reshape(-1)
    if not (forecast.shape == target.shape == weights.shape):
        raise ValueError("probability, observation, and weight must share shape")
    if not np.all(np.isfinite(forecast)) or np.any(
        (forecast < 0.0) | (forecast > 1.0)
    ):
        raise ValueError("probability must be finite and lie in [0,1]")
    if not np.all(np.isfinite(target)) or np.any(
        (target != 0.0) & (target != 1.0)
    ):
        raise ValueError("probability observations must be binary")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probability weights must be finite and non-negative")
    selected = weights > 0.0
    if not np.any(selected):
        raise ValueError("probability calibration has no positive weight")
    clipped = np.clip(forecast[selected], 1.0e-6, 1.0 - 1.0e-6)
    loss = float(
        np.sum(
            weights[selected]
            * (
                -target[selected] * np.log(clipped)
                - (1.0 - target[selected]) * np.log1p(-clipped)
            )
        )
        / np.sum(weights[selected])
    )
    return ProbabilityCalibration(
        alpha=0.0,
        beta_raw=_inverse_softplus(1.0),
        slope=1.0,
        weighted_log_loss=loss,
        iterations=0,
        pooling=None,
        identity=True,
    )


__all__ = [
    "block_ess_tolerance",
    "CALIBRATION_SPLIT",
    "FoldCalibrationMoments",
    "IndependentBlockSupport",
    "LocationScaleCalibration",
    "POOLING_ORDER",
    "PoolingDecision",
    "ProbabilityCalibration",
    "SamplerCalibration",
    "apply_residual_calibration",
    "empirical_lower_median",
    "fit_location_total_scale",
    "identity_location_total_scale",
    "fit_monotone_logit_linear_probability",
    "fit_sampler_bias_and_spread",
    "identity_monotone_logit_linear_probability",
    "identity_sampler_bias_and_spread",
    "independent_block_support",
    "monotone_logit_linear_probability",
    "select_pooling_level",
    "transformed_members_to_physical",
]
