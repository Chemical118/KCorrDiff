"""Causal radar optical flow and semi-Lagrangian persistence.

This module is intentionally a condition builder and deterministic baseline,
not a target builder.  Its input schema contains exactly the twelve radar
frames available through issue time ``t0`` and has no field for a future
label or target-derived validity mask.

The numerical path is shared by CPU and CUDA through Torch operations:

* robust coarse-to-fine Lucas--Kanade flow on the uniform context grid;
* physical ``km h-1`` velocity with positive components along increasing LCC
  x/y axes;
* forward/backward consistency and an independent continuous confidence;
* stationary-velocity RK2 back-trajectories at five-minute intervals;
* target-first source sampling, with context supplying target-boundary inflow;
* twelve exact seven-scan trapezoidal accumulations in linear rain-rate space.

No circular padding is used and no full-archive trajectory cache is provided.
The default streaming mode retains only the last seven forecast scans.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import math
from typing import Literal, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .accumulation import trapezoidal_accumulation_mm
from .radar_values import A0_MM, A_WET_MM


HISTORY_FRAME_COUNT = 12
FIVE_MINUTES = timedelta(minutes=5)
STEP_HOURS = 5.0 / 60.0
FORECAST_STEPS = 72
TRAJECTORY_FRAME_COUNT = FORECAST_STEPS + 1
LEAD_COUNT = 12
LEAD_HOURS = tuple(0.5 * index for index in range(1, LEAD_COUNT + 1))
ADVECTION_MODEL_CHANNEL_NAMES = (
    "z_adv",
    "w_adv",
    "flow_u_km_per_hour",
    "flow_v_km_per_hour",
    "origin_in_domain",
    "domain_residence_fraction",
    "flow_confidence",
    "advection_valid",
)


def _axis(values: Tensor | Sequence[float], *, name: str) -> Tensor:
    axis = torch.as_tensor(values, dtype=torch.float64, device="cpu").clone()
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with at least two cells")
    if not torch.isfinite(axis).all() or not torch.all(torch.diff(axis) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    spacing = torch.diff(axis)
    tolerance = 1.0e-8 * max(1.0, float(torch.max(torch.abs(axis))))
    if not torch.allclose(
        spacing, torch.full_like(spacing, spacing.mean()), rtol=1.0e-6, atol=tolerance
    ):
        raise ValueError(f"{name} must be uniform for the advection kernel")
    return axis


@dataclass(frozen=True, slots=True)
class AdvectionGeometry:
    """Uniform target/context axes in the same physical LCC kilometre frame."""

    target_x_km: Tensor | Sequence[float]
    target_y_km: Tensor | Sequence[float]
    context_x_km: Tensor | Sequence[float]
    context_y_km: Tensor | Sequence[float]

    def __post_init__(self) -> None:
        target_x = _axis(self.target_x_km, name="target_x_km")
        target_y = _axis(self.target_y_km, name="target_y_km")
        context_x = _axis(self.context_x_km, name="context_x_km")
        context_y = _axis(self.context_y_km, name="context_y_km")
        if (
            target_x[0] < context_x[0]
            or target_x[-1] > context_x[-1]
            or target_y[0] < context_y[0]
            or target_y[-1] > context_y[-1]
        ):
            raise ValueError("the target interpolation domain must lie inside context")
        object.__setattr__(self, "target_x_km", target_x)
        object.__setattr__(self, "target_y_km", target_y)
        object.__setattr__(self, "context_x_km", context_x)
        object.__setattr__(self, "context_y_km", context_y)

    @property
    def target_shape(self) -> tuple[int, int]:
        return (self.target_y_km.numel(), self.target_x_km.numel())

    @property
    def context_shape(self) -> tuple[int, int]:
        return (self.context_y_km.numel(), self.context_x_km.numel())

    @property
    def target_spacing_km(self) -> tuple[float, float]:
        return (
            float(self.target_y_km[1] - self.target_y_km[0]),
            float(self.target_x_km[1] - self.target_x_km[0]),
        )

    @property
    def context_spacing_km(self) -> tuple[float, float]:
        return (
            float(self.context_y_km[1] - self.context_y_km[0]),
            float(self.context_x_km[1] - self.context_x_km[0]),
        )


def _as_batch_datetimes(
    values: datetime | Sequence[datetime], *, batch: int, name: str
) -> tuple[datetime, ...]:
    result = (values,) if isinstance(values, datetime) else tuple(values)
    if len(result) != batch:
        raise ValueError(f"{name} must contain one value per batch item")
    canonical: list[datetime] = []
    for value in result:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} values must be timezone-aware")
        utc = value.astimezone(UTC)
        if utc.minute % 5 or utc.second or utc.microsecond:
            raise ValueError(f"{name} values must lie on five-minute slots")
        canonical.append(utc)
    return tuple(canonical)


def _as_history_times(
    values: Sequence[datetime] | Sequence[Sequence[datetime]],
    *,
    batch: int,
) -> tuple[tuple[datetime, ...], ...]:
    sequence = tuple(values)
    if batch == 1 and len(sequence) == HISTORY_FRAME_COUNT and all(
        isinstance(value, datetime) for value in sequence
    ):
        rows = (sequence,)  # type: ignore[assignment]
    else:
        rows = tuple(tuple(row) for row in sequence)  # type: ignore[arg-type]
    if len(rows) != batch:
        raise ValueError("history_times_utc must contain one row per batch item")
    return rows  # type: ignore[return-value]


def causal_history_times(t0_utc: datetime) -> tuple[datetime, ...]:
    """Return exactly ``t0-55 min, ..., t0`` after strict UTC validation."""

    (t0,) = _as_batch_datetimes(t0_utc, batch=1, name="t0_utc")
    return tuple(t0 - timedelta(minutes=5 * offset) for offset in range(11, -1, -1))


@dataclass(frozen=True, slots=True)
class CausalAdvectionInput:
    """Issue-time-only linear-rain-rate tensors.

    Shapes are ``[B,12,H,W]``.  Invalid target values may be neutral-filled;
    ``target_valid`` is authoritative.  Context validity remains a continuous
    fraction and is not confused with a dry rate of zero.
    """

    t0_utc: datetime | Sequence[datetime]
    history_times_utc: Sequence[datetime] | Sequence[Sequence[datetime]]
    target_rate_mm_per_hour: Tensor
    target_valid: Tensor
    context_rate_mm_per_hour: Tensor
    context_valid_fraction: Tensor
    context_interpolation_confidence: Tensor

    def __post_init__(self) -> None:
        target = self.target_rate_mm_per_hour
        context = self.context_rate_mm_per_hour
        if not isinstance(target, Tensor) or not isinstance(context, Tensor):
            raise TypeError("target/context rates must be Torch tensors")
        if target.ndim != 4 or context.ndim != 4:
            raise ValueError("target/context rates must have shape [B,12,H,W]")
        if target.shape[:2] != (context.shape[0], HISTORY_FRAME_COUNT) or context.shape[1] != HISTORY_FRAME_COUNT:
            raise ValueError("target/context histories require the same batch and exactly 12 frames")
        if not target.is_floating_point() or not context.is_floating_point():
            raise TypeError("rain rates must use a floating Torch dtype")
        if target.dtype != context.dtype or target.device != context.device:
            raise ValueError("target/context rates must share dtype and device")
        batch = target.shape[0]
        t0_values = _as_batch_datetimes(self.t0_utc, batch=batch, name="t0_utc")
        history_rows = _as_history_times(self.history_times_utc, batch=batch)
        canonical_rows: list[tuple[datetime, ...]] = []
        for index, (t0, row) in enumerate(zip(t0_values, history_rows, strict=True)):
            if len(row) != HISTORY_FRAME_COUNT:
                raise ValueError("every history row must contain exactly 12 timestamps")
            canonical = tuple(
                _as_batch_datetimes(value, batch=1, name=f"history_times_utc[{index}]")[0]
                for value in row
            )
            expected = causal_history_times(t0)
            if canonical != expected:
                raise ValueError(
                    "history timestamps must be exactly t0-55 min through t0 in five-minute order"
                )
            canonical_rows.append(canonical)

        validity = self.target_valid
        context_valid = self.context_valid_fraction
        context_confidence = self.context_interpolation_confidence
        if validity.shape != target.shape:
            raise ValueError("target_valid must match target rates")
        if validity.dtype != torch.bool:
            raise TypeError("target_valid must be a boolean tensor")
        for name, value in (
            ("context_valid_fraction", context_valid),
            ("context_interpolation_confidence", context_confidence),
        ):
            if value.shape != context.shape:
                raise ValueError(f"{name} must match context rates")
            if not value.is_floating_point() or value.dtype != context.dtype or value.device != context.device:
                raise ValueError(f"{name} must share context dtype/device")
            if not torch.isfinite(value).all() or torch.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"{name} must contain finite values in [0,1]")
        if validity.device != target.device:
            raise ValueError("target_valid must share the target device")
        if not torch.isfinite(target[validity]).all() or torch.any(target[validity] < 0.0):
            raise ValueError("valid target rates must be finite and non-negative")
        context_observed = context_valid > 0.0
        if not torch.isfinite(context[context_observed]).all() or torch.any(
            context[context_observed] < 0.0
        ):
            raise ValueError("observed context rates must be finite and non-negative")
        object.__setattr__(self, "t0_utc", t0_values)
        object.__setattr__(self, "history_times_utc", tuple(canonical_rows))

    @property
    def batch_size(self) -> int:
        return self.target_rate_mm_per_hour.shape[0]


@dataclass(frozen=True, slots=True)
class FlowConfig:
    """Frozen numerical choices for robust stationary context flow."""

    pyramid_levels: int = 4
    warps_per_level: int = 3
    irls_iterations: int = 2
    window_size: int = 7
    gaussian_sigma: float = 1.5
    regularization: float = 1.0e-3
    charbonnier_epsilon: float = 1.0e-2
    minimum_eigenvalue: float = 2.0e-4
    photometric_scale: float = 0.15
    fb_error_scale_km: float = 1.2
    maximum_speed_km_per_hour: float = 150.0
    maximum_increment_pixels: float = 2.0
    recency_half_life_minutes: float = 30.0
    minimum_source_valid_fraction: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.pyramid_levels < 1 or self.warps_per_level < 1 or self.irls_iterations < 1:
            raise ValueError("flow iteration counts must be positive")
        if self.window_size < 3 or self.window_size % 2 != 1:
            raise ValueError("window_size must be odd and at least three")
        positive = (
            self.gaussian_sigma,
            self.regularization,
            self.charbonnier_epsilon,
            self.minimum_eigenvalue,
            self.photometric_scale,
            self.fb_error_scale_km,
            self.maximum_speed_km_per_hour,
            self.maximum_increment_pixels,
            self.recency_half_life_minutes,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("flow scale/configuration values must be finite and positive")
        if not 0.0 <= self.minimum_source_valid_fraction <= 1.0:
            raise ValueError("minimum_source_valid_fraction must lie in [0,1]")

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DenseFlow:
    """Stationary context-grid velocity and independent QC fields."""

    velocity_km_per_hour: Tensor
    backward_velocity_km_per_hour: Tensor
    confidence: Tensor
    forward_backward_error_km: Tensor
    valid_mask: Tensor
    config_hash: str


@dataclass(frozen=True, slots=True)
class AdvectionTrajectory:
    """Optional materialized five-minute target-grid trajectory."""

    rate_mm_per_hour: Tensor
    scan_valid: Tensor
    origin_in_target: Tensor
    origin_in_context: Tensor
    context_inflow: Tensor
    domain_residence_fraction: Tensor
    path_confidence: Tensor


@dataclass(frozen=True, slots=True)
class AdvectionLeadFeatures:
    """Twelve physical baseline fields and model-facing confidence channels."""

    raw_accumulation_mm: Tensor
    model_accumulation_mm: Tensor
    z_adv: Tensor
    w_adv: Tensor
    origin_in_domain_mask: Tensor
    origin_in_target_mask: Tensor
    context_inflow_fraction: Tensor
    domain_residence_fraction: Tensor
    flow_confidence: Tensor
    advection_valid_mask: Tensor
    flow_uv_target_km_per_hour: Tensor
    flow_valid_target: Tensor

    @property
    def channel_names(self) -> tuple[str, ...]:
        return ADVECTION_MODEL_CHANNEL_NAMES

    def as_model_tensor(self) -> Tensor:
        """Return ``[B,12,8,H,W]`` in the documented channel order."""

        batch, leads, height, width = self.z_adv.shape
        flow = self.flow_uv_target_km_per_hour[:, None].expand(
            batch, leads, 2, height, width
        )
        return torch.cat(
            (
                self.z_adv[:, :, None],
                self.w_adv[:, :, None].to(self.z_adv.dtype),
                flow,
                self.origin_in_domain_mask[:, :, None].to(self.z_adv.dtype),
                self.domain_residence_fraction[:, :, None],
                self.flow_confidence[:, :, None],
                self.advection_valid_mask[:, :, None].to(self.z_adv.dtype),
            ),
            dim=2,
        )


@dataclass(frozen=True, slots=True)
class AdvectionProvenance:
    algorithm: str
    flow_config_hash: str
    issue_times_utc: tuple[datetime, ...]
    history_times_utc: tuple[tuple[datetime, ...], ...]
    step_hours: float
    forecast_steps: int
    used_future_observations: bool
    full_trajectory_cached: bool


@dataclass(frozen=True, slots=True)
class AdvectionArtifact:
    flow: DenseFlow
    leads: AdvectionLeadFeatures
    trajectory: AdvectionTrajectory | None
    provenance: AdvectionProvenance


@dataclass(frozen=True, slots=True)
class BaselineArtifact:
    name: Literal["eulerian_persistence", "lagrangian_persistence"]
    raw_accumulation_mm: Tensor
    model_accumulation_mm: Tensor
    z: Tensor
    wet: Tensor
    valid_mask: Tensor
    flow_config_hash: str | None


def _safe_values(values: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid, values, torch.zeros((), dtype=values.dtype, device=values.device))


def _gaussian_kernel(config: FlowConfig, reference: Tensor) -> Tensor:
    radius = config.window_size // 2
    coordinate = torch.arange(
        -radius, radius + 1, device=reference.device, dtype=reference.dtype
    )
    vector = torch.exp(-0.5 * (coordinate / config.gaussian_sigma) ** 2)
    vector = vector / vector.sum()
    return torch.outer(vector, vector).reshape(1, 1, config.window_size, config.window_size)


def _local_sum(values: Tensor, kernel: Tensor) -> Tensor:
    radius = kernel.shape[-1] // 2
    return F.conv2d(F.pad(values, (radius, radius, radius, radius), mode="replicate"), kernel)


def _gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    dtype, device = image.dtype, image.device
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).reshape(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel_x), F.conv2d(padded, kernel_y)


def _pixel_base_grid(batch: int, height: int, width: int, reference: Tensor) -> Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype),
        torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=-1)[None].expand(batch, height, width, 2)


def _warp_pixels(values: Tensor, displacement_xy_pixels: Tensor) -> Tensor:
    batch, _, height, width = values.shape
    grid = _pixel_base_grid(batch, height, width, values).clone()
    if width > 1:
        grid[..., 0] += 2.0 * displacement_xy_pixels[:, 0] / (width - 1)
    if height > 1:
        grid[..., 1] += 2.0 * displacement_xy_pixels[:, 1] / (height - 1)
    return F.grid_sample(
        values, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


def _resize_flow(flow: Tensor, shape: tuple[int, int]) -> Tensor:
    old_height, old_width = flow.shape[-2:]
    new_height, new_width = shape
    resized = F.interpolate(flow, size=shape, mode="bilinear", align_corners=True)
    resized = resized.clone()
    resized[:, 0] *= new_width / old_width
    resized[:, 1] *= new_height / old_height
    return resized


def _pyramidal_lk(
    first: Tensor,
    second: Tensor,
    first_support: Tensor,
    second_support: Tensor,
    config: FlowConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate dense forward displacement in pixels for a batch of pairs."""

    first_pyramid = [first]
    second_pyramid = [second]
    first_valid_pyramid = [first_support]
    second_valid_pyramid = [second_support]
    for _ in range(1, config.pyramid_levels):
        if min(first_pyramid[-1].shape[-2:]) < 8:
            break
        first_pyramid.append(F.avg_pool2d(first_pyramid[-1], 2, 2))
        second_pyramid.append(F.avg_pool2d(second_pyramid[-1], 2, 2))
        first_valid_pyramid.append(F.avg_pool2d(first_valid_pyramid[-1], 2, 2))
        second_valid_pyramid.append(F.avg_pool2d(second_valid_pyramid[-1], 2, 2))

    flow: Tensor | None = None
    for level in range(len(first_pyramid) - 1, -1, -1):
        source = first_pyramid[level]
        destination = second_pyramid[level]
        valid_source = first_valid_pyramid[level]
        valid_destination = second_valid_pyramid[level]
        if flow is None:
            flow = torch.zeros(
                (source.shape[0], 2, *source.shape[-2:]),
                dtype=source.dtype,
                device=source.device,
            )
        elif flow.shape[-2:] != source.shape[-2:]:
            flow = _resize_flow(flow, source.shape[-2:])
        kernel = _gaussian_kernel(config, source)
        for _ in range(config.warps_per_level):
            for _ in range(config.irls_iterations):
                warped = _warp_pixels(destination, flow)
                warped_valid = _warp_pixels(valid_destination, flow).clamp(0.0, 1.0)
                gradient_x, gradient_y = _gradients(warped)
                residual = source - warped
                robust = config.charbonnier_epsilon / torch.sqrt(
                    residual.square() + config.charbonnier_epsilon**2
                )
                weight = valid_source * warped_valid * robust
                support = _local_sum(weight, kernel)
                a_xx = _local_sum(weight * gradient_x.square(), kernel) / support.clamp_min(1.0e-6)
                a_xy = _local_sum(weight * gradient_x * gradient_y, kernel) / support.clamp_min(1.0e-6)
                a_yy = _local_sum(weight * gradient_y.square(), kernel) / support.clamp_min(1.0e-6)
                b_x = _local_sum(weight * gradient_x * residual, kernel) / support.clamp_min(1.0e-6)
                b_y = _local_sum(weight * gradient_y * residual, kernel) / support.clamp_min(1.0e-6)
                a_xx = a_xx + config.regularization
                a_yy = a_yy + config.regularization
                determinant = (a_xx * a_yy - a_xy.square()).clamp_min(
                    config.regularization**2
                )
                delta_x = (a_yy * b_x - a_xy * b_y) / determinant
                delta_y = (a_xx * b_y - a_xy * b_x) / determinant
                delta = torch.cat((delta_x, delta_y), dim=1)
                magnitude = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
                delta = delta * torch.clamp(
                    config.maximum_increment_pixels / magnitude.clamp_min(1.0e-12),
                    max=1.0,
                )
                delta = torch.where(support > 0.05, delta, torch.zeros_like(delta))
                flow = flow + delta

    assert flow is not None
    warped = _warp_pixels(second, flow)
    warped_valid = _warp_pixels(second_support, flow).clamp(0.0, 1.0)
    residual = torch.abs(first - warped)
    gradient_x, gradient_y = _gradients(first)
    kernel = _gaussian_kernel(config, first)
    support_weight = first_support * warped_valid
    support = _local_sum(support_weight, kernel)
    a_xx = _local_sum(support_weight * gradient_x.square(), kernel) / support.clamp_min(1.0e-6)
    a_xy = _local_sum(support_weight * gradient_x * gradient_y, kernel) / support.clamp_min(1.0e-6)
    a_yy = _local_sum(support_weight * gradient_y.square(), kernel) / support.clamp_min(1.0e-6)
    trace = a_xx + a_yy
    discriminant = torch.sqrt(((a_xx - a_yy).square() + 4.0 * a_xy.square()).clamp_min(0.0))
    eigen_min = 0.5 * (trace - discriminant)
    texture = eigen_min / (eigen_min + config.minimum_eigenvalue)
    photo = torch.exp(-residual / config.photometric_scale)
    pair_valid = support > 0.25
    confidence = torch.where(
        pair_valid,
        texture.clamp(0.0, 1.0) * photo.clamp(0.0, 1.0),
        torch.zeros_like(texture),
    )
    return flow, confidence, pair_valid


def _weighted_pair_mean(values: Tensor, weights: Tensor) -> Tensor:
    denominator = weights.sum(dim=1).clamp_min(1.0e-12)
    return (values * weights).sum(dim=1) / denominator


def _clamp_speed(velocity: Tensor, maximum: float) -> Tensor:
    speed = torch.linalg.vector_norm(velocity, dim=1, keepdim=True)
    return velocity * torch.clamp(maximum / speed.clamp_min(1.0e-12), max=1.0)


def estimate_causal_flow(
    inputs: CausalAdvectionInput,
    geometry: AdvectionGeometry,
    *,
    config: FlowConfig | None = None,
) -> DenseFlow:
    """Estimate stationary context flow using only the twelve causal frames."""

    config = config or FlowConfig()
    context = inputs.context_rate_mm_per_hour
    if tuple(context.shape[-2:]) != geometry.context_shape:
        raise ValueError("context tensor shape does not match advection geometry")
    batch, _, height, width = context.shape
    observed = inputs.context_valid_fraction > config.minimum_source_valid_fraction
    safe_context = _safe_values(context, observed)
    flattened = safe_context.reshape(batch, -1)
    # A positive linear scaling improves conditioning without changing the
    # brightness-constancy geometry.  No log or transformed rate is used.
    scale = torch.quantile(flattened, 0.90, dim=1).clamp_min(1.0)
    normalized = safe_context / scale[:, None, None, None]
    support = (
        inputs.context_valid_fraction * inputs.context_interpolation_confidence
    ).clamp(0.0, 1.0)

    first = normalized[:, :-1].reshape(batch * 11, 1, height, width)
    second = normalized[:, 1:].reshape(batch * 11, 1, height, width)
    first_support = support[:, :-1].reshape(batch * 11, 1, height, width)
    second_support = support[:, 1:].reshape(batch * 11, 1, height, width)
    forward_pixels, forward_conf, forward_valid = _pyramidal_lk(
        first, second, first_support, second_support, config
    )
    backward_pixels, backward_conf, backward_valid = _pyramidal_lk(
        second, first, second_support, first_support, config
    )
    forward_pixels = forward_pixels.reshape(batch, 11, 2, height, width)
    backward_pixels = backward_pixels.reshape(batch, 11, 2, height, width)
    forward_conf = forward_conf.reshape(batch, 11, 1, height, width)
    backward_conf = backward_conf.reshape(batch, 11, 1, height, width)
    forward_valid = forward_valid.reshape(batch, 11, 1, height, width)
    backward_valid = backward_valid.reshape(batch, 11, 1, height, width)

    dy_km, dx_km = geometry.context_spacing_km
    scale_xy = context.new_tensor((dx_km / STEP_HOURS, dy_km / STEP_HOURS)).reshape(
        1, 1, 2, 1, 1
    )
    forward_velocity_pairs = forward_pixels * scale_xy
    backward_velocity_pairs = backward_pixels * scale_xy
    age_minutes = torch.arange(10, -1, -1, dtype=context.dtype, device=context.device) * 5.0
    recency = torch.pow(
        context.new_tensor(0.5), age_minutes / config.recency_half_life_minutes
    ).reshape(1, 11, 1, 1, 1)
    forward_weight = recency * forward_conf
    backward_weight = recency * backward_conf
    forward_velocity = _weighted_pair_mean(forward_velocity_pairs, forward_weight)
    backward_velocity = _weighted_pair_mean(backward_velocity_pairs, backward_weight)

    # Low-texture pixels retain low confidence, but a confidence-weighted
    # domain translation fills their velocity.  This keeps a dry pixel valid
    # rather than inventing a missing observation while avoiding patchy
    # displacement outside precipitation gradients.
    def fill_low_texture(velocity: Tensor, weight: Tensor) -> Tensor:
        denominator = weight.sum(dim=(-2, -1), keepdim=True)
        global_velocity = (velocity * weight).sum(dim=(-2, -1), keepdim=True) / denominator.clamp_min(
            1.0e-12
        )
        return torch.where(
            weight > 1.0e-6, velocity, global_velocity.expand_as(velocity)
        )

    forward_velocity = fill_low_texture(
        forward_velocity, forward_weight.sum(dim=1)
    )
    backward_velocity = fill_low_texture(
        backward_velocity, backward_weight.sum(dim=1)
    )
    forward_velocity = _clamp_speed(forward_velocity, config.maximum_speed_km_per_hour)
    backward_velocity = _clamp_speed(backward_velocity, config.maximum_speed_km_per_hour)

    displacement_pixels = forward_velocity.clone()
    displacement_pixels[:, 0] *= STEP_HOURS / dx_km
    displacement_pixels[:, 1] *= STEP_HOURS / dy_km
    backward_at_endpoint = _warp_pixels(backward_velocity, displacement_pixels)
    backward_valid_mean = backward_valid.to(context.dtype).mean(dim=1)
    backward_valid_at_endpoint = _warp_pixels(backward_valid_mean, displacement_pixels)
    fb_error = torch.linalg.vector_norm(
        (forward_velocity + backward_at_endpoint) * STEP_HOURS,
        dim=1,
        keepdim=True,
    )
    pair_observation_weight = recency * forward_valid.to(context.dtype)
    observation_denominator = pair_observation_weight.sum(dim=1)
    aggregate_confidence = (
        (recency * forward_conf).sum(dim=1) / observation_denominator.clamp_min(1.0e-12)
    )
    aggregate_confidence = aggregate_confidence * torch.exp(
        -(fb_error / config.fb_error_scale_km).square()
    )
    latest_observed = inputs.context_valid_fraction[:, -1:, :, :]
    latest_confidence = inputs.context_interpolation_confidence[:, -1:, :, :]
    valid_mask = (
        (latest_observed > config.minimum_source_valid_fraction)
        & (observation_denominator > 0.0)
        & (backward_valid_at_endpoint > 0.5)
    )
    confidence = torch.where(
        valid_mask,
        (aggregate_confidence * latest_observed * latest_confidence).clamp(0.0, 1.0),
        torch.zeros_like(aggregate_confidence),
    )
    forward_velocity = torch.where(valid_mask, forward_velocity, torch.zeros_like(forward_velocity))
    backward_velocity = torch.where(valid_mask, backward_velocity, torch.zeros_like(backward_velocity))
    return DenseFlow(
        velocity_km_per_hour=forward_velocity,
        backward_velocity_km_per_hour=backward_velocity,
        confidence=confidence,
        forward_backward_error_km=fb_error,
        valid_mask=valid_mask,
        config_hash=config.config_hash,
    )


def _physical_mesh(
    x_axis: Tensor, y_axis: Tensor, *, batch: int, reference: Tensor
) -> tuple[Tensor, Tensor]:
    y, x = torch.meshgrid(
        y_axis.to(device=reference.device, dtype=reference.dtype),
        x_axis.to(device=reference.device, dtype=reference.dtype),
        indexing="ij",
    )
    return x[None].expand(batch, -1, -1), y[None].expand(batch, -1, -1)


def _inside(x: Tensor, y: Tensor, x_axis: Tensor, y_axis: Tensor) -> Tensor:
    return (
        (x >= float(x_axis[0]))
        & (x <= float(x_axis[-1]))
        & (y >= float(y_axis[0]))
        & (y <= float(y_axis[-1]))
    )


def _physical_sample(
    values: Tensor,
    x: Tensor,
    y: Tensor,
    x_axis: Tensor,
    y_axis: Tensor,
    *,
    mode: Literal["bilinear", "nearest"] = "bilinear",
) -> Tensor:
    x0, x1 = float(x_axis[0]), float(x_axis[-1])
    y0, y1 = float(y_axis[0]), float(y_axis[-1])
    grid_x = 2.0 * (x - x0) / (x1 - x0) - 1.0
    grid_y = 2.0 * (y - y0) / (y1 - y0) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        values, grid, mode=mode, padding_mode="zeros", align_corners=True
    )


def _validate_flow(flow: DenseFlow, inputs: CausalAdvectionInput, geometry: AdvectionGeometry) -> None:
    batch = inputs.batch_size
    expected_vector = (batch, 2, *geometry.context_shape)
    expected_scalar = (batch, 1, *geometry.context_shape)
    if flow.velocity_km_per_hour.shape != expected_vector or flow.backward_velocity_km_per_hour.shape != expected_vector:
        raise ValueError("flow velocity shape does not match batch/context geometry")
    for name, value in (
        ("confidence", flow.confidence),
        ("forward_backward_error_km", flow.forward_backward_error_km),
        ("valid_mask", flow.valid_mask),
    ):
        if value.shape != expected_scalar:
            raise ValueError(f"flow {name} shape does not match context geometry")
    if flow.valid_mask.dtype != torch.bool:
        raise TypeError("flow valid_mask must be boolean")
    for value in (
        flow.velocity_km_per_hour,
        flow.backward_velocity_km_per_hour,
        flow.confidence,
        flow.forward_backward_error_km,
    ):
        if value.dtype != inputs.target_rate_mm_per_hour.dtype or value.device != inputs.target_rate_mm_per_hour.device:
            raise ValueError("flow tensors must share input dtype/device")
        if not torch.isfinite(value).all():
            raise ValueError("flow tensors must be finite")
    if torch.any((flow.confidence < 0.0) | (flow.confidence > 1.0)):
        raise ValueError("flow confidence must lie in [0,1]")


def _lead_from_window(
    rate: Sequence[Tensor],
    scan_valid: Sequence[Tensor],
    origin_target: Sequence[Tensor],
    origin_context: Sequence[Tensor],
    context_inflow: Sequence[Tensor],
    residence: Sequence[Tensor],
    confidence: Sequence[Tensor],
    *,
    a_wet_mm: float,
    a0_mm: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    rates = torch.stack(tuple(rate), dim=1)
    valid_stack = torch.stack(tuple(scan_valid), dim=1)
    valid = valid_stack.all(dim=1)
    raw = trapezoidal_accumulation_mm(rates, time_axis=1)
    assert isinstance(raw, Tensor)
    raw = torch.where(valid, raw, torch.zeros_like(raw))
    wet = valid & (raw >= a_wet_mm)
    model = torch.where(wet, raw, torch.zeros_like(raw))
    z = torch.log1p(model / a0_mm)
    origin_target_mask = torch.stack(tuple(origin_target), dim=1).all(dim=1)
    origin_context_mask = torch.stack(tuple(origin_context), dim=1).all(dim=1)
    weights = rates.new_tensor((0.5, 1, 1, 1, 1, 1, 0.5)).reshape(1, 7, 1, 1)
    inflow_fraction = (
        torch.stack(tuple(context_inflow), dim=1).to(rates.dtype) * weights
    ).sum(dim=1) / weights.sum()
    residence_min = torch.stack(tuple(residence), dim=1).amin(dim=1)
    confidence_mean = (torch.stack(tuple(confidence), dim=1) * weights).sum(dim=1) / weights.sum()
    return (
        raw,
        model,
        z,
        wet,
        origin_context_mask,
        origin_target_mask,
        inflow_fraction,
        residence_min,
        confidence_mean,
        valid,
    )


def accumulate_advection_leads(
    trajectory: AdvectionTrajectory,
    *,
    flow_uv_target_km_per_hour: Tensor,
    flow_valid_target: Tensor,
    a_wet_mm: float = A_WET_MM,
    a0_mm: float = A0_MM,
) -> AdvectionLeadFeatures:
    """Reduce a materialized 73-frame trajectory to twelve exact windows.

    Lead ``j`` consumes ``trajectory[6*j : 6*j+7]``.  Thus adjacent leads
    share their boundary scan at half weight, exactly like observed targets.
    Invalid seven-scan windows are neutral-filled and never renormalized.
    """

    rate = trajectory.rate_mm_per_hour
    if (
        not math.isfinite(a_wet_mm)
        or a_wet_mm < 0.0
        or not math.isfinite(a0_mm)
        or a0_mm <= 0.0
    ):
        raise ValueError("accumulation thresholds/scales are invalid")
    if rate.ndim != 4 or rate.shape[1] != TRAJECTORY_FRAME_COUNT:
        raise ValueError("trajectory rates must have shape [B,73,H,W]")
    if not rate.is_floating_point() or not torch.isfinite(rate).all() or torch.any(rate < 0.0):
        raise ValueError("trajectory rates must be finite non-negative floating values")
    tensor_fields = (
        trajectory.scan_valid,
        trajectory.origin_in_target,
        trajectory.origin_in_context,
        trajectory.context_inflow,
        trajectory.domain_residence_fraction,
        trajectory.path_confidence,
    )
    if any(value.shape != rate.shape for value in tensor_fields):
        raise ValueError("all materialized trajectory fields must share [B,73,H,W]")
    for name, value in (
        ("scan_valid", trajectory.scan_valid),
        ("origin_in_target", trajectory.origin_in_target),
        ("origin_in_context", trajectory.origin_in_context),
        ("context_inflow", trajectory.context_inflow),
    ):
        if value.dtype != torch.bool:
            raise TypeError(f"trajectory {name} must be boolean")
        if value.device != rate.device:
            raise ValueError(f"trajectory {name} must share the rate device")
    for name, value in (
        ("domain_residence_fraction", trajectory.domain_residence_fraction),
        ("path_confidence", trajectory.path_confidence),
    ):
        if value.dtype != rate.dtype or value.device != rate.device:
            raise ValueError(f"trajectory {name} must share rate dtype/device")
        if not torch.isfinite(value).all() or torch.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"trajectory {name} must contain finite values in [0,1]")
    expected_flow = (rate.shape[0], 2, *rate.shape[-2:])
    expected_valid = (rate.shape[0], *rate.shape[-2:])
    if flow_uv_target_km_per_hour.shape != expected_flow:
        raise ValueError("target flow must have shape [B,2,H,W]")
    if (
        flow_valid_target.shape != expected_valid
        or flow_valid_target.dtype != torch.bool
        or flow_valid_target.device != rate.device
    ):
        raise ValueError("target flow validity must be boolean [B,H,W]")
    if flow_uv_target_km_per_hour.dtype != rate.dtype or flow_uv_target_km_per_hour.device != rate.device:
        raise ValueError("target flow must share trajectory dtype/device")
    if not torch.isfinite(flow_uv_target_km_per_hour).all():
        raise ValueError("target flow must contain only finite values")

    lead_values: list[tuple[Tensor, ...]] = []
    for lead_index in range(LEAD_COUNT):
        start = 6 * lead_index
        stop = start + 7
        lead_values.append(
            _lead_from_window(
                tuple(rate[:, index] for index in range(start, stop)),
                tuple(trajectory.scan_valid[:, index] for index in range(start, stop)),
                tuple(trajectory.origin_in_target[:, index] for index in range(start, stop)),
                tuple(trajectory.origin_in_context[:, index] for index in range(start, stop)),
                tuple(trajectory.context_inflow[:, index] for index in range(start, stop)),
                tuple(
                    trajectory.domain_residence_fraction[:, index]
                    for index in range(start, stop)
                ),
                tuple(trajectory.path_confidence[:, index] for index in range(start, stop)),
                a_wet_mm=a_wet_mm,
                a0_mm=a0_mm,
            )
        )
    stacked = tuple(
        torch.stack([lead[index] for lead in lead_values], dim=1)
        for index in range(10)
    )
    return AdvectionLeadFeatures(
        raw_accumulation_mm=stacked[0],
        model_accumulation_mm=stacked[1],
        z_adv=stacked[2],
        w_adv=stacked[3],
        origin_in_domain_mask=stacked[4],
        origin_in_target_mask=stacked[5],
        context_inflow_fraction=stacked[6],
        domain_residence_fraction=stacked[7],
        flow_confidence=stacked[8],
        advection_valid_mask=stacked[9],
        flow_uv_target_km_per_hour=flow_uv_target_km_per_hour,
        flow_valid_target=flow_valid_target,
    )


def _extrapolate(
    inputs: CausalAdvectionInput,
    geometry: AdvectionGeometry,
    flow: DenseFlow,
    *,
    materialize_trajectory: bool,
    a_wet_mm: float,
    a0_mm: float,
) -> tuple[AdvectionLeadFeatures, AdvectionTrajectory | None]:
    _validate_flow(flow, inputs, geometry)
    if tuple(inputs.target_rate_mm_per_hour.shape[-2:]) != geometry.target_shape:
        raise ValueError("target tensor shape does not match advection geometry")
    if not math.isfinite(a_wet_mm) or a_wet_mm < 0.0 or not math.isfinite(a0_mm) or a0_mm <= 0.0:
        raise ValueError("accumulation thresholds/scales are invalid")
    reference = inputs.target_rate_mm_per_hour
    batch = inputs.batch_size
    x, y = _physical_mesh(
        geometry.target_x_km,
        geometry.target_y_km,
        batch=batch,
        reference=reference,
    )
    target_latest = _safe_values(
        inputs.target_rate_mm_per_hour[:, -1:, :, :],
        inputs.target_valid[:, -1:, :, :],
    )
    target_valid = inputs.target_valid[:, -1:, :, :].to(reference.dtype)
    context_observed = (
        inputs.context_valid_fraction[:, -1:, :, :] > 0.0
    )
    context_latest = _safe_values(
        inputs.context_rate_mm_per_hour[:, -1:, :, :], context_observed
    )
    context_valid_fraction = inputs.context_valid_fraction[:, -1:, :, :]
    context_source_confidence = inputs.context_interpolation_confidence[:, -1:, :, :]
    flow_fields = torch.cat(
        (
            flow.velocity_km_per_hour,
            flow.valid_mask.to(reference.dtype),
            flow.confidence,
        ),
        dim=1,
    )
    target_fields = torch.cat((target_latest, target_valid), dim=1)
    context_fields = torch.cat(
        (context_latest, context_valid_fraction, context_source_confidence), dim=1
    )

    initial_flow_state = _physical_sample(
        flow_fields,
        x,
        y,
        geometry.context_x_km,
        geometry.context_y_km,
    )
    flow_uv_target = initial_flow_state[:, :2]
    flow_valid_target = initial_flow_state[:, 2] >= 1.0 - 1.0e-6
    initial_confidence = initial_flow_state[:, 3]

    rings: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "rate",
            "valid",
            "origin_target",
            "origin_context",
            "inflow",
            "residence",
            "confidence",
        )
    }
    materialized: dict[str, list[Tensor]] | None = (
        {key: [] for key in rings} if materialize_trajectory else None
    )
    lead_values: list[tuple[Tensor, ...]] = []
    path_active = torch.ones_like(x, dtype=torch.bool)
    path_confidence = initial_confidence
    residence_sum = torch.zeros_like(x)

    def append_frame(
        rate_value: Tensor,
        valid_value: Tensor,
        target_origin: Tensor,
        context_origin: Tensor,
        inflow_value: Tensor,
        residence_value: Tensor,
        confidence_value: Tensor,
    ) -> None:
        frame = {
            "rate": rate_value,
            "valid": valid_value,
            "origin_target": target_origin,
            "origin_context": context_origin,
            "inflow": inflow_value,
            "residence": residence_value,
            "confidence": confidence_value,
        }
        for key, value in frame.items():
            rings[key].append(value)
            if len(rings[key]) > 7:
                rings[key].pop(0)
            if materialized is not None:
                materialized[key].append(value)

    in_target = _inside(x, y, geometry.target_x_km, geometry.target_y_km)
    in_context = _inside(x, y, geometry.context_x_km, geometry.context_y_km)
    issue_valid = inputs.target_valid[:, -1]
    issue_rate = torch.where(issue_valid, target_latest[:, 0], torch.zeros_like(x))
    append_frame(
        issue_rate,
        issue_valid,
        in_target,
        in_context,
        torch.zeros_like(in_target),
        torch.ones_like(x),
        initial_confidence,
    )

    for step in range(1, TRAJECTORY_FRAME_COUNT):
        flow_state_1 = _physical_sample(
            flow_fields,
            x,
            y,
            geometry.context_x_km,
            geometry.context_y_km,
        )
        velocity_1 = flow_state_1[:, :2]
        valid_1 = flow_state_1[:, 2] >= 1.0 - 1.0e-6
        confidence_1 = flow_state_1[:, 3]
        mid_x = x - 0.5 * STEP_HOURS * velocity_1[:, 0]
        mid_y = y - 0.5 * STEP_HOURS * velocity_1[:, 1]
        flow_state_mid = _physical_sample(
            flow_fields,
            mid_x,
            mid_y,
            geometry.context_x_km,
            geometry.context_y_km,
        )
        velocity_mid = flow_state_mid[:, :2]
        valid_mid = flow_state_mid[:, 2] >= 1.0 - 1.0e-6
        confidence_mid = flow_state_mid[:, 3]
        next_x = x - STEP_HOURS * velocity_mid[:, 0]
        next_y = y - STEP_HOURS * velocity_mid[:, 1]
        inside_current = _inside(x, y, geometry.context_x_km, geometry.context_y_km)
        inside_mid = _inside(mid_x, mid_y, geometry.context_x_km, geometry.context_y_km)
        inside_next = _inside(next_x, next_y, geometry.context_x_km, geometry.context_y_km)
        residence_sum = residence_sum + 0.5 * (
            inside_mid.to(reference.dtype) + inside_next.to(reference.dtype)
        )
        residence = residence_sum / step
        path_active = (
            path_active
            & inside_current
            & inside_mid
            & inside_next
            & valid_1
            & valid_mid
        )
        path_confidence = torch.minimum(
            path_confidence, torch.minimum(confidence_1, confidence_mid)
        )
        x, y = next_x, next_y

        origin_target = _inside(x, y, geometry.target_x_km, geometry.target_y_km)
        origin_context = _inside(x, y, geometry.context_x_km, geometry.context_y_km)
        target_state = _physical_sample(
            target_fields,
            x,
            y,
            geometry.target_x_km,
            geometry.target_y_km,
        )
        target_value = target_state[:, 0]
        target_valid_at_origin = target_state[:, 1] >= 1.0 - 1.0e-6
        context_state = _physical_sample(
            context_fields,
            x,
            y,
            geometry.context_x_km,
            geometry.context_y_km,
        )
        context_value = context_state[:, 0]
        context_valid_at_origin = context_state[:, 1] > 0.0
        context_confidence_at_origin = context_state[:, 2]
        inflow = (~origin_target) & origin_context
        selected_value = torch.where(origin_target, target_value, context_value)
        selected_valid = torch.where(
            origin_target,
            target_valid_at_origin,
            origin_context & context_valid_at_origin,
        )
        scan_valid = path_active & selected_valid
        rate = torch.where(scan_valid, selected_value, torch.zeros_like(selected_value))
        source_confidence = torch.where(
            origin_target,
            torch.ones_like(context_confidence_at_origin),
            context_confidence_at_origin,
        )
        frame_confidence = path_confidence * source_confidence
        append_frame(
            rate,
            scan_valid,
            origin_target,
            origin_context,
            inflow,
            residence,
            frame_confidence,
        )
        if step >= 6 and step % 6 == 0:
            lead_values.append(
                _lead_from_window(
                    rings["rate"],
                    rings["valid"],
                    rings["origin_target"],
                    rings["origin_context"],
                    rings["inflow"],
                    rings["residence"],
                    rings["confidence"],
                    a_wet_mm=a_wet_mm,
                    a0_mm=a0_mm,
                )
            )

    if len(lead_values) != LEAD_COUNT:  # pragma: no cover - construction invariant
        raise AssertionError("advection must construct exactly twelve lead windows")
    stacked = tuple(torch.stack([lead[index] for lead in lead_values], dim=1) for index in range(10))
    leads = AdvectionLeadFeatures(
        raw_accumulation_mm=stacked[0],
        model_accumulation_mm=stacked[1],
        z_adv=stacked[2],
        w_adv=stacked[3],
        origin_in_domain_mask=stacked[4],
        origin_in_target_mask=stacked[5],
        context_inflow_fraction=stacked[6],
        domain_residence_fraction=stacked[7],
        flow_confidence=stacked[8],
        advection_valid_mask=stacked[9],
        flow_uv_target_km_per_hour=flow_uv_target,
        flow_valid_target=flow_valid_target,
    )
    trajectory = None
    if materialized is not None:
        trajectory = AdvectionTrajectory(
            rate_mm_per_hour=torch.stack(materialized["rate"], dim=1),
            scan_valid=torch.stack(materialized["valid"], dim=1),
            origin_in_target=torch.stack(materialized["origin_target"], dim=1),
            origin_in_context=torch.stack(materialized["origin_context"], dim=1),
            context_inflow=torch.stack(materialized["inflow"], dim=1),
            domain_residence_fraction=torch.stack(materialized["residence"], dim=1),
            path_confidence=torch.stack(materialized["confidence"], dim=1),
        )
        # Exercise the same public materialized-trajectory reduction used by
        # diagnostics.  The streaming path above independently retains only
        # seven frames, and parity is contract-tested.
        leads = accumulate_advection_leads(
            trajectory,
            flow_uv_target_km_per_hour=flow_uv_target,
            flow_valid_target=flow_valid_target,
            a_wet_mm=a_wet_mm,
            a0_mm=a0_mm,
        )
    return leads, trajectory


def semi_lagrangian_trajectory(
    inputs: CausalAdvectionInput,
    geometry: AdvectionGeometry,
    flow: DenseFlow,
    *,
    a_wet_mm: float = A_WET_MM,
    a0_mm: float = A0_MM,
) -> tuple[AdvectionLeadFeatures, AdvectionTrajectory]:
    """Materialize all 73 target-grid rate frames and their lead products."""

    leads, trajectory = _extrapolate(
        inputs,
        geometry,
        flow,
        materialize_trajectory=True,
        a_wet_mm=a_wet_mm,
        a0_mm=a0_mm,
    )
    assert trajectory is not None
    return leads, trajectory


def stream_advection_leads(
    inputs: CausalAdvectionInput,
    geometry: AdvectionGeometry,
    flow: DenseFlow,
    *,
    a_wet_mm: float = A_WET_MM,
    a0_mm: float = A0_MM,
) -> AdvectionLeadFeatures:
    """Build twelve lead products while retaining at most seven rate frames."""

    leads, trajectory = _extrapolate(
        inputs,
        geometry,
        flow,
        materialize_trajectory=False,
        a_wet_mm=a_wet_mm,
        a0_mm=a0_mm,
    )
    assert trajectory is None
    return leads


def build_causal_advection(
    inputs: CausalAdvectionInput,
    geometry: AdvectionGeometry,
    *,
    config: FlowConfig | None = None,
    flow: DenseFlow | None = None,
    materialize_trajectory: bool = False,
    track_grad: bool = False,
    a_wet_mm: float = A_WET_MM,
    a0_mm: float = A0_MM,
) -> AdvectionArtifact:
    """Build one reusable causal flow artifact and all twelve lead features.

    ``track_grad=False`` is the production condition-building default.  The
    same functional path remains differentiable when explicitly requested.
    Supplying a precomputed ``flow`` guarantees that the model condition and
    the Lagrangian baseline share exactly the same motion artifact.
    """

    config = config or FlowConfig()
    context = nullcontext() if track_grad else torch.no_grad()
    with context:
        dense_flow = flow or estimate_causal_flow(inputs, geometry, config=config)
        leads, trajectory = _extrapolate(
            inputs,
            geometry,
            dense_flow,
            materialize_trajectory=materialize_trajectory,
            a_wet_mm=a_wet_mm,
            a0_mm=a0_mm,
        )
    provenance = AdvectionProvenance(
        algorithm="robust_pyramidal_lucas_kanade_stationary_rk2_v1",
        flow_config_hash=dense_flow.config_hash,
        issue_times_utc=tuple(inputs.t0_utc),  # type: ignore[arg-type]
        history_times_utc=tuple(inputs.history_times_utc),  # type: ignore[arg-type]
        step_hours=STEP_HOURS,
        forecast_steps=FORECAST_STEPS,
        used_future_observations=False,
        full_trajectory_cached=False,
    )
    return AdvectionArtifact(
        flow=dense_flow, leads=leads, trajectory=trajectory, provenance=provenance
    )


def eulerian_persistence(
    inputs: CausalAdvectionInput,
    *,
    a_wet_mm: float = A_WET_MM,
    a0_mm: float = A0_MM,
) -> BaselineArtifact:
    """Keep ``R(t0)`` fixed and use the same seven-scan integration helper."""

    latest_valid = inputs.target_valid[:, -1]
    latest = torch.where(
        latest_valid,
        inputs.target_rate_mm_per_hour[:, -1],
        torch.zeros_like(inputs.target_rate_mm_per_hour[:, -1]),
    )
    seven = latest[:, None].expand(-1, 7, -1, -1)
    raw_one = trapezoidal_accumulation_mm(seven, time_axis=1)
    assert isinstance(raw_one, Tensor)
    raw_one = torch.where(latest_valid, raw_one, torch.zeros_like(raw_one))
    raw = raw_one[:, None].expand(-1, LEAD_COUNT, -1, -1)
    valid = latest_valid[:, None].expand_as(raw)
    wet = valid & (raw >= a_wet_mm)
    model = torch.where(wet, raw, torch.zeros_like(raw))
    z = torch.log1p(model / a0_mm)
    return BaselineArtifact(
        name="eulerian_persistence",
        raw_accumulation_mm=raw,
        model_accumulation_mm=model,
        z=z,
        wet=wet,
        valid_mask=valid,
        flow_config_hash=None,
    )


def lagrangian_persistence(artifact: AdvectionArtifact) -> BaselineArtifact:
    """Expose the model's exact advection artifact without neural correction."""

    leads = artifact.leads
    return BaselineArtifact(
        name="lagrangian_persistence",
        raw_accumulation_mm=leads.raw_accumulation_mm,
        model_accumulation_mm=leads.model_accumulation_mm,
        z=leads.z_adv,
        wet=leads.w_adv,
        valid_mask=leads.advection_valid_mask,
        flow_config_hash=artifact.flow.config_hash,
    )


__all__ = [
    "ADVECTION_MODEL_CHANNEL_NAMES",
    "FORECAST_STEPS",
    "FIVE_MINUTES",
    "HISTORY_FRAME_COUNT",
    "LEAD_COUNT",
    "LEAD_HOURS",
    "STEP_HOURS",
    "TRAJECTORY_FRAME_COUNT",
    "AdvectionArtifact",
    "AdvectionGeometry",
    "AdvectionLeadFeatures",
    "AdvectionProvenance",
    "AdvectionTrajectory",
    "BaselineArtifact",
    "CausalAdvectionInput",
    "DenseFlow",
    "FlowConfig",
    "build_causal_advection",
    "accumulate_advection_leads",
    "causal_history_times",
    "estimate_causal_flow",
    "eulerian_persistence",
    "lagrangian_persistence",
    "semi_lagrangian_trajectory",
    "stream_advection_leads",
]
