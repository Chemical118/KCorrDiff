"""Deterministic EDM sampling and one-lead forecast finalization.

This module owns the frozen v1.1.3b inference contracts, but deliberately
does not import a concrete residual model.  A caller supplies a ``denoise``
callable and one already-built, issue-time-available condition cache.  Future
target labels and target-validity masks are consequently absent from every
public sampling interface.

The random stream is generated on CPU with NumPy's counter-based Philox
bit-generator, independently for every ``(sample, lead, member, step)`` key.
It therefore does not depend on DataLoader workers, batch packing, CUDA rank,
or the number of GPUs used for inference.  The model/checkpoint identity is
intentionally *not* part of the random key so candidate models use common
random numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from kcorrdiff.data.radar_values import A_WET_MM
from kcorrdiff.training.calibration import (
    apply_residual_calibration,
    empirical_lower_median,
    transformed_members_to_physical,
)


HEUN_SOLVER = "heun"
KARRAS_SCHEDULE = "karras_rho7"
KARRAS_RHO = 7.0
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
INITIAL_NOISE_PURPOSE = "edm-initial-noise-v1"
COMMON_ENSEMBLE_SEED = 11104
OFFICIAL_Q_THRESHOLDS_MM = (A_WET_MM, 1.0, 5.0)


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    """One exact, protocol-frozen member/step profile."""

    name: str
    members: int
    edm_steps: int
    distilled_only: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("sampling profile name must be non-empty and canonical")
        for name, value in (("members", self.members), ("edm_steps", self.edm_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


_PROFILE_ITEMS = (
    SamplingProfile("development_smoke", members=4, edm_steps=6),
    SamplingProfile("selection_signature", members=16, edm_steps=8),
    SamplingProfile("final_primary_signature", members=32, edm_steps=12),
    SamplingProfile(
        "operational_transition_signature",
        members=8,
        edm_steps=4,
        distilled_only=True,
    ),
)
SAMPLING_PROFILES: Mapping[str, SamplingProfile] = MappingProxyType(
    {profile.name: profile for profile in _PROFILE_ITEMS}
)
_PROFILE_BY_COUNTS = {
    (profile.members, profile.edm_steps): profile for profile in _PROFILE_ITEMS
}


def sampling_profile(name: str) -> SamplingProfile:
    """Resolve a declared profile, failing closed on aliases or new counts."""

    if not isinstance(name, str):
        raise TypeError("sampling profile name must be a string")
    try:
        return SAMPLING_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(SAMPLING_PROFILES)
        raise ValueError(
            f"undeclared sampling profile {name!r}; expected one of {choices}"
        ) from error


@dataclass(frozen=True, slots=True)
class SamplerCoreSignature:
    """Calibration key ``u``; ensemble member count is intentionally absent."""

    checkpoint_id: str
    edm_steps: int
    checkpoint_kind: Literal["diffusion", "distilled"] = "diffusion"
    solver: Literal["heun"] = HEUN_SOLVER
    sigma_schedule: Literal["karras_rho7"] = KARRAS_SCHEDULE
    sigma_min: float = SIGMA_MIN
    sigma_max: float = SIGMA_MAX
    rho: float = KARRAS_RHO

    def __post_init__(self) -> None:
        if (
            not isinstance(self.checkpoint_id, str)
            or not self.checkpoint_id
            or self.checkpoint_id.strip() != self.checkpoint_id
        ):
            raise ValueError("checkpoint_id must be a non-empty canonical string")
        if self.checkpoint_kind not in ("diffusion", "distilled"):
            raise ValueError("checkpoint_kind must be diffusion or distilled")
        if self.solver != HEUN_SOLVER:
            raise ValueError("v1.1.3b supports only the declared Heun solver")
        if self.sigma_schedule != KARRAS_SCHEDULE:
            raise ValueError("v1.1.3b requires the Karras rho=7 schedule")
        if isinstance(self.edm_steps, bool) or not isinstance(self.edm_steps, int):
            raise TypeError("edm_steps must be an integer")
        if self.edm_steps not in {profile.edm_steps for profile in _PROFILE_ITEMS}:
            raise ValueError("edm_steps is not part of a declared sampling profile")
        if (
            float(self.sigma_min) != SIGMA_MIN
            or float(self.sigma_max) != SIGMA_MAX
            or float(self.rho) != KARRAS_RHO
        ):
            raise ValueError("sigma_min=0.002, sigma_max=80, and rho=7 are frozen")
        if self.edm_steps == 4 and self.checkpoint_kind != "distilled":
            raise ValueError("the 4-step operational sampler requires a distilled checkpoint")

    @property
    def canonical(self) -> tuple[str, str, str, int, str, float, float, float]:
        """Exact, hashable calibration/publication signature."""

        return (
            self.checkpoint_id,
            self.checkpoint_kind,
            self.solver,
            self.edm_steps,
            self.sigma_schedule,
            self.sigma_min,
            self.sigma_max,
            self.rho,
        )


@dataclass(frozen=True, slots=True)
class EnsembleSignature:
    """Calibration key ``v=(u,N_members)`` with an exact frozen profile."""

    sampler_core: SamplerCoreSignature
    member_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.sampler_core, SamplerCoreSignature):
            raise TypeError("sampler_core must be a SamplerCoreSignature")
        if isinstance(self.member_count, bool) or not isinstance(self.member_count, int):
            raise TypeError("member_count must be an integer")
        profile = _PROFILE_BY_COUNTS.get(
            (self.member_count, self.sampler_core.edm_steps)
        )
        if profile is None:
            raise ValueError("member/step combination is not a declared sampling profile")
        if profile.distilled_only and self.sampler_core.checkpoint_kind != "distilled":
            raise ValueError("the operational profile requires a distilled checkpoint")

    @property
    def profile(self) -> SamplingProfile:
        return _PROFILE_BY_COUNTS[(self.member_count, self.sampler_core.edm_steps)]

    @property
    def canonical(
        self,
    ) -> tuple[tuple[str, str, str, int, str, float, float, float], int]:
        return (self.sampler_core.canonical, self.member_count)


def build_ensemble_signature(
    *,
    checkpoint_id: str,
    profile_name: str,
    checkpoint_kind: Literal["diffusion", "distilled"] = "diffusion",
) -> EnsembleSignature:
    """Build ``u`` and ``v`` from one named, exact frozen profile."""

    profile = sampling_profile(profile_name)
    core = SamplerCoreSignature(
        checkpoint_id=checkpoint_id,
        checkpoint_kind=checkpoint_kind,
        edm_steps=profile.edm_steps,
    )
    return EnsembleSignature(core, profile.members)


class DenoiseCallable(Protocol):
    """Concrete-model-free residual denoiser boundary.

    ``noisy_normalized_residual`` has shape ``[B*N,1,H,W]`` and ``sigma`` has
    shape ``[B*N]``.  The exact same ``condition_cache`` object is provided at
    every Heun evaluation.  An adapter around the concrete model may use the
    flattened member dimension to index/repeat cached batch features without
    rebuilding source K/V projections.
    """

    def __call__(
        self,
        noisy_normalized_residual: Tensor,
        sigma: Tensor,
        *,
        condition_cache: object,
    ) -> Tensor: ...


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_lead(lead_hours: float) -> str:
    if isinstance(lead_hours, bool):
        raise TypeError("lead_hours must be a real scalar")
    value = float(lead_hours)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("lead_hours must be finite and positive")
    return value.hex()


def _philox_entropy(
    *,
    common_ensemble_seed: int,
    purpose_id: str,
    sample_id: str,
    lead_hours: float,
    member_index: int,
    stochastic_step_index: int,
) -> list[int]:
    if (
        isinstance(common_ensemble_seed, bool)
        or not isinstance(common_ensemble_seed, int)
        or common_ensemble_seed < 0
    ):
        raise ValueError("common_ensemble_seed must be a non-negative integer")
    if not isinstance(purpose_id, str) or not purpose_id:
        raise ValueError("purpose_id must be non-empty")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be non-empty")
    for name, value in (
        ("member_index", member_index),
        ("stochastic_step_index", stochastic_step_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    # JSON length/escaping makes the namespace unambiguous even if IDs contain
    # delimiters.  Checkpoint/model IDs are deliberately absent (common random
    # numbers across candidates are part of the evaluation contract).
    payload = json.dumps(
        (
            common_ensemble_seed,
            purpose_id,
            sample_id,
            _canonical_lead(lead_hours),
            member_index,
            stochastic_step_index,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return [
        int.from_bytes(digest[offset : offset + 4], "little")
        for offset in range(0, len(digest), 4)
    ]


def counter_gaussian_noise(
    *,
    sample_ids: Sequence[str],
    lead_hours: float,
    member_count: int,
    spatial_shape: tuple[int, int],
    common_ensemble_seed: int = COMMON_ENSEMBLE_SEED,
    purpose_id: str = INITIAL_NOISE_PURPOSE,
    stochastic_step_index: int = 0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Return topology-independent float32 Gaussian fields ``[B,N,1,H,W]``."""

    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
        raise TypeError("sample_ids must be a sequence of strings")
    identities = tuple(sample_ids)
    if not identities:
        raise ValueError("at least one sample_id is required")
    members = _positive_integer("member_count", member_count)
    if (
        not isinstance(spatial_shape, tuple)
        or len(spatial_shape) != 2
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in spatial_shape
        )
    ):
        raise ValueError("spatial_shape must contain two positive integers")
    height, width = spatial_shape
    fields = np.empty(
        (len(identities), members, 1, height, width), dtype=np.float32
    )
    for sample_index, sample_id in enumerate(identities):
        for member_index in range(members):
            entropy = _philox_entropy(
                common_ensemble_seed=common_ensemble_seed,
                purpose_id=purpose_id,
                sample_id=sample_id,
                lead_hours=lead_hours,
                member_index=member_index,
                stochastic_step_index=stochastic_step_index,
            )
            generator = np.random.Generator(
                np.random.Philox(np.random.SeedSequence(entropy))
            )
            fields[sample_index, member_index] = generator.standard_normal(
                (1, height, width), dtype=np.float32
            )
    return torch.from_numpy(fields).to(device=torch.device(device))


def karras_sigma_schedule(
    edm_steps: int,
    *,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
    rho: float = KARRAS_RHO,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Construct the exact Karras rho=7 positive levels plus terminal zero."""

    steps = _positive_integer("edm_steps", edm_steps)
    if steps < 2:
        raise ValueError("Karras sampling requires at least two positive levels")
    if (
        float(sigma_min) != SIGMA_MIN
        or float(sigma_max) != SIGMA_MAX
        or float(rho) != KARRAS_RHO
    ):
        raise ValueError("the frozen schedule is sigma_min=0.002, sigma_max=80, rho=7")
    ramp = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    maximum_root = float(sigma_max) ** (1.0 / float(rho))
    minimum_root = float(sigma_min) ** (1.0 / float(rho))
    positive = (maximum_root + ramp * (minimum_root - maximum_root)) ** float(rho)
    # Assignment before the float32 cast makes the declared boundaries exact,
    # rather than relying on endpoint cancellation in the power expression.
    positive[0] = float(sigma_max)
    positive[-1] = float(sigma_min)
    schedule = np.concatenate((positive, np.zeros(1, dtype=np.float64)))
    return torch.tensor(schedule, dtype=torch.float32, device=torch.device(device))


def _validate_solver_inputs(initial_state: Tensor, sigmas: Tensor) -> None:
    if not isinstance(initial_state, Tensor) or initial_state.dtype is not torch.float32:
        raise TypeError("initial_state must be a float32 tensor")
    if initial_state.ndim != 4 or initial_state.shape[1] != 1:
        raise ValueError("initial_state must have shape [B,1,H,W]")
    if not bool(torch.isfinite(initial_state).all().item()):
        raise ValueError("initial_state contains a non-finite value")
    if not isinstance(sigmas, Tensor) or sigmas.dtype is not torch.float32:
        raise TypeError("sigmas must be a float32 tensor")
    if sigmas.ndim != 1 or sigmas.numel() < 3:
        raise ValueError("sigmas must contain at least two positive levels and zero")
    if sigmas.device != initial_state.device:
        raise ValueError("sigmas and initial_state must share one device")
    if not bool(torch.isfinite(sigmas).all().item()):
        raise ValueError("sigmas contain a non-finite value")
    if float(sigmas[-1].item()) != 0.0 or bool((sigmas[:-1] <= 0.0).any().item()):
        raise ValueError("sigmas must contain positive levels followed by exact zero")
    if not bool((sigmas[:-1] > sigmas[1:]).all().item()):
        raise ValueError("sigmas must be strictly decreasing")


def _denoise_checked(
    denoise: DenoiseCallable,
    state: Tensor,
    sigma_value: Tensor,
    *,
    condition_cache: object,
) -> Tensor:
    sigma = sigma_value.expand(state.shape[0])
    result = denoise(state, sigma, condition_cache=condition_cache)
    if not isinstance(result, Tensor):
        raise TypeError("denoise must return a tensor")
    if result.dtype is not torch.float32:
        raise TypeError("denoise output must remain float32")
    if result.shape != state.shape or result.device != state.device:
        raise ValueError("denoise output must match the state shape and device")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("denoise output contains a non-finite value")
    return result


def heun_edm_solve(
    *,
    initial_state: Tensor,
    sigmas: Tensor,
    denoise: DenoiseCallable,
    condition_cache: object,
) -> Tensor:
    """Integrate the deterministic EDM probability-flow ODE with exact Heun.

    Churn is zero.  Every positive-to-positive transition uses the explicit
    trapezoidal (Heun) correction and the terminal transition to sigma zero is
    Euler, matching EDM Algorithm 2 without stochastic augmentation.
    """

    _validate_solver_inputs(initial_state, sigmas)
    if condition_cache is None:
        raise ValueError("condition_cache must be built and passed explicitly")
    state = initial_state
    # Disable ambient autocast even when the caller enabled it around the
    # inference entrypoint.  The denoiser's return contract is checked after
    # every evaluation, preventing an implicit BF16/FP16 path.
    with torch.inference_mode(), torch.autocast(
        device_type=state.device.type, enabled=False
    ):
        for index in range(sigmas.numel() - 1):
            sigma_current = sigmas[index]
            sigma_next = sigmas[index + 1]
            denoised = _denoise_checked(
                denoise,
                state,
                sigma_current,
                condition_cache=condition_cache,
            )
            derivative = (state - denoised) / sigma_current
            delta = sigma_next - sigma_current
            euler = state + delta * derivative
            if float(sigma_next.item()) == 0.0:
                state = euler
                continue
            denoised_next = _denoise_checked(
                denoise,
                euler,
                sigma_next,
                condition_cache=condition_cache,
            )
            derivative_next = (euler - denoised_next) / sigma_next
            state = state + delta * (derivative + derivative_next) * 0.5
    return state


def sample_normalized_residual_ensemble(
    *,
    denoise: DenoiseCallable,
    condition_cache: object,
    sample_ids: Sequence[str],
    lead_hours: float,
    spatial_shape: tuple[int, int],
    ensemble_signature: EnsembleSignature,
    common_ensemble_seed: int = COMMON_ENSEMBLE_SEED,
    noise_purpose_id: str = INITIAL_NOISE_PURPOSE,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Sample normalized residuals with shape ``[B,N,1,H,W]``."""

    if not isinstance(ensemble_signature, EnsembleSignature):
        raise TypeError("ensemble_signature must be declared explicitly")
    if condition_cache is None:
        raise ValueError("condition_cache must be built and passed explicitly")
    identities = tuple(sample_ids)
    noise = counter_gaussian_noise(
        sample_ids=identities,
        lead_hours=lead_hours,
        member_count=ensemble_signature.member_count,
        spatial_shape=spatial_shape,
        common_ensemble_seed=common_ensemble_seed,
        purpose_id=noise_purpose_id,
        stochastic_step_index=0,
        device=device,
    )
    batch, members, _channels, height, width = noise.shape
    schedule = karras_sigma_schedule(
        ensemble_signature.sampler_core.edm_steps,
        sigma_min=ensemble_signature.sampler_core.sigma_min,
        sigma_max=ensemble_signature.sampler_core.sigma_max,
        rho=ensemble_signature.sampler_core.rho,
        device=noise.device,
    )
    initial = (noise * schedule[0]).reshape(batch * members, 1, height, width)
    solved = heun_edm_solve(
        initial_state=initial,
        sigmas=schedule,
        denoise=denoise,
        condition_cache=condition_cache,
    )
    result = solved.reshape(batch, members, 1, height, width)
    if result.dtype is not torch.float32:
        raise AssertionError("internal error: EDM sampler left float32")
    return result


def _scale_for_members(scale: Tensor | float, members: Tensor) -> Tensor:
    if isinstance(scale, Tensor):
        if scale.dtype is not torch.float32:
            raise TypeError("OOF residual scale tensor must be float32")
        if scale.device != members.device:
            raise ValueError("OOF residual scale must share the member device")
        if scale.ndim == 0:
            shaped = scale
        elif scale.shape == (members.shape[0],):
            shaped = scale[:, None, None, None, None]
        elif scale.shape in (
            (members.shape[0], 1, 1, 1),
            (members.shape[0], 1, *members.shape[-2:]),
        ):
            shaped = scale[:, None]
        else:
            raise ValueError("OOF scale must be scalar, [B], [B,1,1,1], or [B,1,H,W]")
    else:
        if isinstance(scale, bool):
            raise TypeError("OOF residual scale must be real")
        shaped = torch.tensor(float(scale), dtype=torch.float32, device=members.device)
    if not bool(torch.isfinite(shaped).all().item()) or bool((shaped <= 0.0).any().item()):
        raise ValueError("OOF residual scale must be finite and positive")
    return shaped


@dataclass(frozen=True, slots=True)
class LeadForecast:
    """One lead's marginal ensemble and its fixed physical summaries."""

    lead_hours: float
    ensemble_signature: EnsembleSignature
    normalized_residual_members: Tensor
    restored_residual_members: Tensor
    calibrated_residual_members: Tensor
    transformed_members: Tensor
    physical_members_mm: Tensor
    ensemble_mean_mm: Tensor
    ensemble_lower_median_mm: Tensor
    q_thresholds_mm: tuple[float, ...]
    q_fractions: Tensor

    def __post_init__(self) -> None:
        _canonical_lead(self.lead_hours)
        members = self.physical_members_mm
        if members.ndim != 5 or members.shape[2] != 1:
            raise ValueError("physical members must have shape [B,N,1,H,W]")
        if members.shape[1] != self.ensemble_signature.member_count:
            raise ValueError("forecast member count disagrees with ensemble signature")
        for name, value in (
            ("normalized_residual_members", self.normalized_residual_members),
            ("restored_residual_members", self.restored_residual_members),
            ("calibrated_residual_members", self.calibrated_residual_members),
            ("transformed_members", self.transformed_members),
            ("physical_members_mm", members),
        ):
            if value.shape != members.shape or value.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32 and match physical members")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")
        point_shape = (members.shape[0], 1, *members.shape[-2:])
        for name, value in (
            ("ensemble_mean_mm", self.ensemble_mean_mm),
            ("ensemble_lower_median_mm", self.ensemble_lower_median_mm),
        ):
            if value.shape != point_shape or value.dtype is not torch.float32:
                raise TypeError(f"{name} must be float32 [B,1,H,W]")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")
        expected_q = (members.shape[0], len(self.q_thresholds_mm), 1, *members.shape[-2:])
        if self.q_fractions.shape != expected_q or self.q_fractions.dtype is not torch.float32:
            raise TypeError("q_fractions must be float32 [B,T,1,H,W]")
        if not bool(torch.isfinite(self.q_fractions).all().item()) or bool(
            ((self.q_fractions < 0.0) | (self.q_fractions > 1.0)).any().item()
        ):
            raise ValueError("q_fractions must be finite and lie in [0,1]")

    @property
    def wet_fraction(self) -> Tensor:
        """Raw ensemble occurrence fraction at the official ``A_wet``."""

        return self.q_fractions[:, 0]


def finalize_lead_forecast(
    *,
    normalized_residual_members: Tensor,
    mu_z_full: Tensor,
    oof_residual_scale: Tensor | float,
    location_b: float,
    total_scale_c: float,
    sampler_bias_d: float,
    spread_gamma: float,
    lead_hours: float,
    ensemble_signature: EnsembleSignature,
    q_thresholds_mm: Sequence[float] = OFFICIAL_Q_THRESHOLDS_MM,
) -> LeadForecast:
    """Apply ``s_oof -> b/c/d -> gamma -> mu -> inverse -> censor`` exactly."""

    members = normalized_residual_members
    if not isinstance(members, Tensor) or members.dtype is not torch.float32:
        raise TypeError("normalized residual members must be a float32 tensor")
    if members.ndim != 5 or members.shape[2] != 1:
        raise ValueError("normalized residual members must have shape [B,N,1,H,W]")
    if not isinstance(ensemble_signature, EnsembleSignature):
        raise TypeError("ensemble_signature must be declared explicitly")
    if members.shape[1] != ensemble_signature.member_count:
        raise ValueError("sampled member count disagrees with ensemble signature")
    if not bool(torch.isfinite(members).all().item()):
        raise ValueError("normalized residual members contain a non-finite value")
    if not isinstance(mu_z_full, Tensor) or mu_z_full.dtype is not torch.float32:
        raise TypeError("mu_z_full must be a float32 tensor")
    expected_mu = (members.shape[0], 1, *members.shape[-2:])
    if mu_z_full.shape != expected_mu or mu_z_full.device != members.device:
        raise ValueError("mu_z_full must be [B,1,H,W] on the member device")
    if not bool(torch.isfinite(mu_z_full).all().item()):
        raise ValueError("mu_z_full contains a non-finite value")
    thresholds = tuple(float(value) for value in q_thresholds_mm)
    if thresholds != OFFICIAL_Q_THRESHOLDS_MM:
        raise ValueError("v1.1.3b q thresholds are frozen at (0.1, 1.0, 5.0) mm")

    scale = _scale_for_members(oof_residual_scale, members)
    restored = members * scale
    if not bool(torch.isfinite(restored).all().item()):
        raise OverflowError("OOF residual scale restoration overflowed float32")
    calibrated = apply_residual_calibration(
        restored,
        location_b=location_b,
        total_scale_c=total_scale_c,
        sampler_bias_d=sampler_bias_d,
        spread_gamma=spread_gamma,
    )
    transformed = mu_z_full[:, None] + calibrated
    if not bool(torch.isfinite(transformed).all().item()):
        raise OverflowError("transformed forecast members overflowed float32")
    physical = transformed_members_to_physical(mu_z_full, calibrated)
    ensemble_mean = physical.mean(dim=1)
    ensemble_median = empirical_lower_median(physical)
    q_fractions = torch.stack(
        [
            (physical >= threshold).to(torch.float32).mean(dim=1)
            for threshold in thresholds
        ],
        dim=1,
    )
    return LeadForecast(
        lead_hours=float(lead_hours),
        ensemble_signature=ensemble_signature,
        normalized_residual_members=members,
        restored_residual_members=restored,
        calibrated_residual_members=calibrated,
        transformed_members=transformed,
        physical_members_mm=physical,
        ensemble_mean_mm=ensemble_mean,
        ensemble_lower_median_mm=ensemble_median,
        q_thresholds_mm=thresholds,
        q_fractions=q_fractions,
    )


def sample_leadwise_forecast(
    *,
    denoise: DenoiseCallable,
    condition_cache: object,
    sample_ids: Sequence[str],
    lead_hours: float,
    mu_z_full: Tensor,
    oof_residual_scale: Tensor | float,
    location_b: float,
    total_scale_c: float,
    sampler_bias_d: float,
    spread_gamma: float,
    ensemble_signature: EnsembleSignature,
    common_ensemble_seed: int = COMMON_ENSEMBLE_SEED,
    noise_purpose_id: str = INITIAL_NOISE_PURPOSE,
) -> LeadForecast:
    """Sample and finalize one lead; member indices are not cross-lead tracks."""

    normalized = sample_normalized_residual_ensemble(
        denoise=denoise,
        condition_cache=condition_cache,
        sample_ids=sample_ids,
        lead_hours=lead_hours,
        spatial_shape=tuple(mu_z_full.shape[-2:]),
        ensemble_signature=ensemble_signature,
        common_ensemble_seed=common_ensemble_seed,
        noise_purpose_id=noise_purpose_id,
        device=mu_z_full.device,
    )
    return finalize_lead_forecast(
        normalized_residual_members=normalized,
        mu_z_full=mu_z_full,
        oof_residual_scale=oof_residual_scale,
        location_b=location_b,
        total_scale_c=total_scale_c,
        sampler_bias_d=sampler_bias_d,
        spread_gamma=spread_gamma,
        lead_hours=lead_hours,
        ensemble_signature=ensemble_signature,
    )


__all__ = [
    "COMMON_ENSEMBLE_SEED",
    "DenoiseCallable",
    "EnsembleSignature",
    "HEUN_SOLVER",
    "INITIAL_NOISE_PURPOSE",
    "KARRAS_RHO",
    "KARRAS_SCHEDULE",
    "LeadForecast",
    "OFFICIAL_Q_THRESHOLDS_MM",
    "SAMPLING_PROFILES",
    "SIGMA_MAX",
    "SIGMA_MIN",
    "SamplerCoreSignature",
    "SamplingProfile",
    "build_ensemble_signature",
    "counter_gaussian_noise",
    "finalize_lead_forecast",
    "heun_edm_solve",
    "karras_sigma_schedule",
    "sample_leadwise_forecast",
    "sample_normalized_residual_ensemble",
    "sampling_profile",
]
