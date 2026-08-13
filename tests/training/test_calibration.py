from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kcorrdiff.training.calibration import (
    FoldCalibrationMoments,
    IndependentBlockSupport,
    LocationScaleCalibration,
    apply_residual_calibration,
    empirical_lower_median,
    fit_location_total_scale,
    fit_monotone_logit_linear_probability,
    fit_sampler_bias_and_spread,
    independent_block_support,
    monotone_logit_linear_probability,
    select_pooling_level,
    transformed_members_to_physical,
)


def test_fold_mixture_uses_oof_valid_mass_and_population_variance() -> None:
    result = fit_location_total_scale(
        folds=(
            FoldCalibrationMoments(0, 1.0, 0.0, 1.0),
            FoldCalibrationMoments(1, 3.0, 2.0, 5.0),
        ),
        full_residual=np.asarray([1.0, 3.0]),
        calibration_weight=np.ones(2),
        split="calibration",
    )
    # pi=(.25,.75), mixture mean=1.5, second=4, variance=1.75.
    assert result.fold_weights == pytest.approx((0.25, 0.75))
    assert result.fold_mixture_variance == pytest.approx(1.75)
    assert result.total_scale_c == pytest.approx(math.sqrt(1.0 / 1.75))
    assert result.location_b == pytest.approx(
        2.0 - result.total_scale_c * 1.5
    )
    with pytest.raises(ValueError, match="independent calibration"):
        fit_location_total_scale(
            folds=(
                FoldCalibrationMoments(0, 1.0, 0.0, 1.0),
                FoldCalibrationMoments(1, 1.0, 0.0, 1.0),
            ),
            full_residual=np.ones(1),
            calibration_weight=np.ones(1),
            split="model_selection",
        )


def test_sampler_d_is_frozen_and_gamma_is_positive() -> None:
    location = LocationScaleCalibration(0.0, 1.0, (0.5, 0.5), 0.0, 1.0, 0.0, 1.0)
    members = np.asarray([[-1.0, 1.0], [0.0, 2.0]])[:, :, None]
    target = np.asarray([[1.0], [3.0]])
    weight = np.ones_like(target)
    disabled = fit_sampler_bias_and_spread(
        restored_members=members,
        full_residual=target,
        calibration_weight=weight,
        location_scale=location,
        d_enabled=False,
        split="calibration",
    )
    enabled = fit_sampler_bias_and_spread(
        restored_members=members,
        full_residual=target,
        calibration_weight=weight,
        location_scale=location,
        d_enabled=True,
        split="calibration",
    )
    assert disabled.sampler_bias_d == 0.0
    assert enabled.sampler_bias_d == pytest.approx(1.5)
    assert disabled.spread_gamma > 0.0


def test_gamma_changes_anomalies_without_changing_member_mean() -> None:
    members = torch.tensor([0.0, 2.0], dtype=torch.float32).view(1, 2, 1, 1, 1)
    calibrated = apply_residual_calibration(
        members,
        location_b=1.0,
        total_scale_c=2.0,
        sampler_bias_d=-0.5,
        spread_gamma=3.0,
    )
    first = 1.0 - 0.5 + 2.0 * members
    assert torch.allclose(calibrated.mean(1), first.mean(1))
    assert torch.allclose(
        calibrated - calibrated.mean(1, keepdim=True),
        3.0 * (first - first.mean(1, keepdim=True)),
    )


def test_physical_inverse_censor_and_empirical_lower_median() -> None:
    mu = torch.zeros(1, 1, 1, 1)
    residual = torch.tensor(
        [[[[[0.0]]], [[[math.log1p(0.05)]]], [[[math.log1p(0.2)]]], [[[math.log1p(1.0)]]]]]
    )
    amount = transformed_members_to_physical(mu, residual)
    assert torch.equal(amount.flatten(), torch.tensor([0.0, 0.0, 0.2, 1.0]))
    assert empirical_lower_median(amount).item() == pytest.approx(0.0)


def test_probability_mapping_is_monotone_and_bounded() -> None:
    value = torch.linspace(0.0, 1.0, 101)
    mapped = monotone_logit_linear_probability(value, alpha=0.2, beta_raw=-1.0)
    assert torch.all(mapped[1:] >= mapped[:-1])
    assert torch.all((mapped > 0.0) & (mapped < 1.0))


def test_independent_block_support_and_fixed_pooling_ladder() -> None:
    blocks: list[str] = []
    labels: list[float] = []
    for index in range(30):
        blocks.extend([f"block-{index}", f"block-{index}"])
        labels.extend([0.0, 1.0])
    support = independent_block_support(
        block_id=blocks,
        weight=np.ones(len(blocks)),
        observation=labels,
    )
    assert support.block_count == 30
    assert support.block_ess == pytest.approx(30.0)
    assert support.positive_support_blocks == 30
    assert support.negative_support_blocks == 30
    sparse = IndependentBlockSupport(29, 19.0, 19, 19)
    decision = select_pooling_level(
        {
            "lead_provider_era_present": sparse,
            "lead_provider": support,
            "lead_only": IndependentBlockSupport(40, 35.0, 40, 40),
        },
        probability_gate=True,
    )
    assert decision.level == "lead_provider"
    assert decision.support is support
    with pytest.raises(ValueError, match="no predeclared"):
        select_pooling_level(
            {
                "lead_provider_era_present": sparse,
                "lead_provider": sparse,
                "lead_only": sparse,
            },
            probability_gate=False,
        )


def test_probability_fit_recovers_identity_family_on_calibration_only() -> None:
    probabilities: list[float] = []
    observations: list[float] = []
    blocks: list[str] = []
    for index in range(30):
        # Each block supports both classes; empirical frequencies are exactly
        # 0.2 and 0.8 for the two forecast groups.
        probabilities.extend([0.2] * 5 + [0.8] * 5)
        observations.extend([1.0, 0.0, 0.0, 0.0, 0.0])
        observations.extend([1.0, 1.0, 1.0, 1.0, 0.0])
        blocks.extend([f"event-{index}"] * 10)
    weights = np.ones(len(probabilities))
    support = independent_block_support(
        block_id=blocks,
        weight=weights,
        observation=observations,
    )
    pooling = select_pooling_level(
        {level: support for level in (
            "lead_provider_era_present", "lead_provider", "lead_only"
        )},
        probability_gate=True,
    )
    fitted = fit_monotone_logit_linear_probability(
        probabilities,
        observations,
        weights,
        split="calibration",
        pooling=pooling,
    )
    assert fitted.alpha == pytest.approx(0.0, abs=2.0e-5)
    assert fitted.slope == pytest.approx(1.0, abs=2.0e-5)
    mapped = monotone_logit_linear_probability(
        torch.tensor([0.2, 0.8], dtype=torch.float32),
        alpha=fitted.alpha,
        beta_raw=fitted.beta_raw,
    )
    torch.testing.assert_close(
        mapped, torch.tensor([0.2, 0.8]), rtol=2.0e-5, atol=2.0e-5
    )
    with pytest.raises(ValueError, match="independent calibration"):
        fit_monotone_logit_linear_probability(
            probabilities,
            observations,
            weights,
            split="model_selection",
            pooling=pooling,
        )
