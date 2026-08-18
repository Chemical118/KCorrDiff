from __future__ import annotations

import inspect
import math

import pytest
import torch

from kcorrdiff.training.edm_sampling import (
    KARRAS_RHO,
    SIGMA_MAX,
    SIGMA_MIN,
    EnsembleSignature,
    SamplerCoreSignature,
    build_ensemble_signature,
    counter_gaussian_noise,
    finalize_lead_forecast,
    heun_edm_solve,
    karras_sigma_schedule,
    sample_normalized_residual_ensemble,
    sampling_profile,
)


def _smoke_signature() -> EnsembleSignature:
    return build_ensemble_signature(
        checkpoint_id="sha256:diffusion-test",
        profile_name="development_smoke",
    )


def test_karras_schedule_has_exact_boundaries_terminal_zero_and_rho7() -> None:
    schedule = karras_sigma_schedule(6)
    assert schedule.dtype is torch.float32
    assert schedule.shape == (7,)
    assert schedule[0] == torch.tensor(SIGMA_MAX, dtype=torch.float32)
    assert schedule[-2] == torch.tensor(SIGMA_MIN, dtype=torch.float32)
    assert schedule[-1] == 0.0
    assert torch.all(schedule[:-1] > schedule[1:])
    tuned = karras_sigma_schedule(6, rho=KARRAS_RHO + 1.0)
    assert tuned[0] == torch.tensor(SIGMA_MAX, dtype=torch.float32)
    assert tuned[-2] == torch.tensor(SIGMA_MIN, dtype=torch.float32)


def test_profiles_and_core_vs_ensemble_signatures_fail_closed() -> None:
    selection = build_ensemble_signature(
        checkpoint_id="diffusion-a",
        profile_name="selection_signature",
    )
    assert selection.profile.members == 16
    assert selection.sampler_core.edm_steps == 8
    assert selection.canonical == (selection.sampler_core.canonical, 16)
    assert selection.sampler_core.canonical == SamplerCoreSignature(
        checkpoint_id="diffusion-a", edm_steps=8
    ).canonical

    with pytest.raises(ValueError, match="undeclared sampling profile"):
        sampling_profile("selection")
    with pytest.raises(ValueError, match="member/step combination"):
        EnsembleSignature(selection.sampler_core, 32)
    with pytest.raises(ValueError, match="distilled"):
        build_ensemble_signature(
            checkpoint_id="ordinary-diffusion",
            profile_name="operational_transition_signature",
        )
    operational = build_ensemble_signature(
        checkpoint_id="distilled-a",
        checkpoint_kind="distilled",
        profile_name="operational_transition_signature",
    )
    assert (operational.member_count, operational.sampler_core.edm_steps) == (8, 4)


def test_counter_gaussian_is_batch_order_independent_and_member_prefix_stable() -> None:
    joint = counter_gaussian_noise(
        sample_ids=("sample-a", "sample-b"),
        lead_hours=2.5,
        member_count=5,
        spatial_shape=(3, 4),
    )
    reversed_batch = counter_gaussian_noise(
        sample_ids=("sample-b", "sample-a"),
        lead_hours=2.5,
        member_count=5,
        spatial_shape=(3, 4),
    )
    prefix = counter_gaussian_noise(
        sample_ids=("sample-a",),
        lead_hours=2.5,
        member_count=3,
        spatial_shape=(3, 4),
    )
    assert joint.dtype is torch.float32
    assert torch.equal(joint[0], reversed_batch[1])
    assert torch.equal(joint[1], reversed_batch[0])
    assert torch.equal(joint[0, :3], prefix[0])

    changed_purpose = counter_gaussian_noise(
        sample_ids=("sample-a",),
        lead_hours=2.5,
        member_count=3,
        spatial_shape=(3, 4),
        purpose_id="a-different-purpose",
    )
    assert not torch.equal(prefix, changed_purpose)


class _ConstantDenoiser:
    def __init__(self, value: float, expected_cache: object) -> None:
        self.value = value
        self.expected_cache = expected_cache
        self.cache_ids: list[int] = []
        self.sigmas: list[torch.Tensor] = []
        self.autocast_states: list[bool] = []

    def __call__(
        self,
        noisy_normalized_residual: torch.Tensor,
        sigma: torch.Tensor,
        *,
        condition_cache: object,
    ) -> torch.Tensor:
        assert condition_cache is self.expected_cache
        assert noisy_normalized_residual.dtype is torch.float32
        assert sigma.dtype is torch.float32
        self.cache_ids.append(id(condition_cache))
        self.sigmas.append(sigma.detach().clone())
        self.autocast_states.append(torch.is_autocast_enabled("cpu"))
        return torch.full_like(noisy_normalized_residual, self.value)


def test_exact_heun_hits_constant_denoiser_and_reuses_cache_without_autocast() -> None:
    cache = object()
    denoiser = _ConstantDenoiser(0.25, cache)
    schedule = karras_sigma_schedule(6)
    initial = torch.linspace(-2.0, 2.0, 12, dtype=torch.float32).reshape(2, 1, 2, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = heun_edm_solve(
            initial_state=initial,
            sigmas=schedule,
            denoise=denoiser,
            condition_cache=cache,
        )
    assert torch.allclose(result, torch.full_like(result, 0.25), atol=2.0e-6)
    # N positive levels: two evaluations for N-1 positive transitions, then
    # one Euler evaluation for the terminal transition.
    assert len(denoiser.cache_ids) == 2 * 6 - 1
    assert set(denoiser.cache_ids) == {id(cache)}
    assert not any(denoiser.autocast_states)
    assert all(torch.all(sigma > 0.0) for sigma in denoiser.sigmas)


def test_heun_identity_denoiser_has_exact_zero_derivative() -> None:
    cache = object()

    def identity(
        value: torch.Tensor, sigma: torch.Tensor, *, condition_cache: object
    ) -> torch.Tensor:
        assert condition_cache is cache
        return value

    initial = torch.randn(3, 1, 2, 2, dtype=torch.float32)
    result = heun_edm_solve(
        initial_state=initial,
        sigmas=karras_sigma_schedule(8),
        denoise=identity,
        condition_cache=cache,
    )
    assert torch.equal(result, initial)


def test_sampler_returns_b_n_1_h_w_and_rejects_non_float32_denoising() -> None:
    signature = _smoke_signature()
    cache = object()
    denoiser = _ConstantDenoiser(0.0, cache)
    result = sample_normalized_residual_ensemble(
        denoise=denoiser,
        condition_cache=cache,
        sample_ids=("a", "b"),
        lead_hours=0.5,
        spatial_shape=(2, 3),
        ensemble_signature=signature,
    )
    assert result.shape == (2, 4, 1, 2, 3)
    assert result.dtype is torch.float32
    assert torch.allclose(result, torch.zeros_like(result), atol=1.0e-5)

    def bad_dtype(
        value: torch.Tensor, sigma: torch.Tensor, *, condition_cache: object
    ) -> torch.Tensor:
        return value.to(torch.bfloat16)

    with pytest.raises(TypeError, match="remain float32"):
        sample_normalized_residual_ensemble(
            denoise=bad_dtype,
            condition_cache=cache,
            sample_ids=("a",),
            lead_hours=0.5,
            spatial_shape=(1, 1),
            ensemble_signature=signature,
        )


def test_lead_finalization_orders_scale_calibration_then_physical_mapping() -> None:
    signature = _smoke_signature()
    normalized = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32).view(
        1, 4, 1, 1, 1
    )
    mu = torch.full((1, 1, 1, 1), 0.2, dtype=torch.float32)
    forecast = finalize_lead_forecast(
        normalized_residual_members=normalized,
        mu_z_full=mu,
        oof_residual_scale=2.0,
        location_b=1.0,
        total_scale_c=2.0,
        sampler_bias_d=0.5,
        spread_gamma=3.0,
        lead_hours=1.0,
        ensemble_signature=signature,
    )
    restored = 2.0 * normalized
    first = 1.0 + 0.5 + 2.0 * restored
    expected_calibrated = first.mean(1, keepdim=True) + 3.0 * (
        first - first.mean(1, keepdim=True)
    )
    assert torch.allclose(forecast.restored_residual_members, restored)
    assert torch.allclose(forecast.calibrated_residual_members, expected_calibrated)
    assert torch.allclose(forecast.transformed_members, mu[:, None] + expected_calibrated)
    assert torch.allclose(
        forecast.ensemble_mean_mm,
        forecast.physical_members_mm.mean(dim=1),
    )


def test_memberwise_censor_lower_median_and_raw_q_thresholds() -> None:
    signature = _smoke_signature()
    normalized = torch.tensor(
        [0.0, math.log1p(0.05), math.log1p(0.2), math.log1p(2.0)],
        dtype=torch.float32,
    ).view(1, 4, 1, 1, 1)
    forecast = finalize_lead_forecast(
        normalized_residual_members=normalized,
        mu_z_full=torch.zeros(1, 1, 1, 1, dtype=torch.float32),
        oof_residual_scale=1.0,
        location_b=0.0,
        total_scale_c=1.0,
        sampler_bias_d=0.0,
        spread_gamma=1.0,
        lead_hours=6.0,
        ensemble_signature=signature,
    )
    assert torch.allclose(
        forecast.physical_members_mm.flatten(),
        torch.tensor([0.0, 0.0, 0.2, 2.0]),
        atol=2.0e-7,
    )
    assert forecast.ensemble_mean_mm.item() == pytest.approx(0.55)
    assert forecast.ensemble_lower_median_mm.item() == 0.0
    assert torch.equal(
        forecast.q_fractions.flatten(), torch.tensor([0.5, 0.25, 0.0])
    )
    assert torch.equal(forecast.wet_fraction, forecast.q_fractions[:, 0])


def test_finalization_rejects_finite_latent_that_would_overflow_expm1() -> None:
    signature = _smoke_signature()
    normalized = torch.full((1, 4, 1, 1, 1), 100.0, dtype=torch.float32)
    with pytest.raises(OverflowError, match="representable float32 physical range"):
        finalize_lead_forecast(
            normalized_residual_members=normalized,
            mu_z_full=torch.zeros(1, 1, 1, 1, dtype=torch.float32),
            oof_residual_scale=1.0,
            location_b=0.0,
            total_scale_c=1.0,
            sampler_bias_d=0.0,
            spread_gamma=1.0,
            lead_hours=1.0,
            ensemble_signature=signature,
        )


def test_sampling_api_has_no_future_label_or_mask_argument() -> None:
    parameters = inspect.signature(sample_normalized_residual_ensemble).parameters
    assert not any("target" in name or "mask" in name or "label" in name for name in parameters)
