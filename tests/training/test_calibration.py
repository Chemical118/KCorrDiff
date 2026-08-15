from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kcorrdiff.training.calibration import (
    FoldCalibrationMoments,
    IndependentBlockSupport,
    LocationScaleCalibration,
    PoolingDecision,
    ProbabilityCalibration,
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


def test_residual_identity_is_a_literal_bitwise_pass_through() -> None:
    generator = torch.Generator().manual_seed(173)
    members = torch.randn(
        (2, 8, 1, 8, 8), generator=generator, dtype=torch.float32
    )
    calibrated = apply_residual_calibration(
        members,
        location_b=0.0,
        total_scale_c=1.0,
        sampler_bias_d=0.0,
        spread_gamma=1.0,
    )
    assert calibrated is members
    assert torch.equal(calibrated, members)


def test_unit_spread_returns_location_scale_result_without_reconstruction() -> None:
    members = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).view(
        1, 3, 1, 1, 1
    )
    expected = 0.25 + 1.5 * members
    calibrated = apply_residual_calibration(
        members,
        location_b=0.5,
        total_scale_c=1.5,
        sampler_bias_d=-0.25,
        spread_gamma=1.0,
    )
    assert torch.equal(calibrated, expected)


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


def test_probability_record_validates_large_inverse_softplus_exactly() -> None:
    slope = 21.0
    beta_raw = math.log(math.expm1(slope))
    support = IndependentBlockSupport(30, 30.0, 20, 20)
    record = ProbabilityCalibration(
        alpha=0.0,
        beta_raw=beta_raw,
        slope=slope,
        weighted_log_loss=0.1,
        iterations=1,
        pooling=PoolingDecision("full_cell", support, probability_gate=True),
    )
    assert record.slope == slope


def test_probability_calibration_rejects_float32_overflow_and_nan_mapping() -> None:
    support = IndependentBlockSupport(30, 30.0, 20, 20)
    pooling = PoolingDecision("full_cell", support, probability_gate=True)
    with pytest.raises(ValueError, match="representable in float32"):
        ProbabilityCalibration(
            alpha=0.0,
            beta_raw=1.0e300,
            slope=1.0e300,
            weighted_log_loss=0.1,
            iterations=1,
            pooling=pooling,
        )
    with pytest.raises(OverflowError, match="parameters overflowed float32"):
        monotone_logit_linear_probability(
            torch.tensor([0.5], dtype=torch.float32),
            alpha=0.0,
            beta_raw=1.0e300,
        )
    with pytest.raises(ValueError, match="strictly positive in float32"):
        ProbabilityCalibration(
            alpha=0.0,
            beta_raw=-1000.0,
            slope=1.0e-20,
            weighted_log_loss=0.1,
            iterations=1,
            pooling=pooling,
        )
    with pytest.raises(OverflowError, match="strictly positive in float32"):
        monotone_logit_linear_probability(
            torch.tensor([0.25, 0.75], dtype=torch.float32),
            alpha=0.0,
            beta_raw=-1000.0,
        )


def test_probability_slope_validation_is_scale_aware_at_fitter_lower_bound() -> None:
    support = IndependentBlockSupport(30, 30.0, 20, 20)
    pooling = PoolingDecision("full_cell", support, probability_gate=True)
    minimum_fitted_slope = 1.0e-8
    beta_raw = math.log(math.expm1(minimum_fitted_slope))
    record = ProbabilityCalibration(
        alpha=0.0,
        beta_raw=beta_raw,
        slope=minimum_fitted_slope,
        weighted_log_loss=0.1,
        iterations=1,
        pooling=pooling,
    )
    assert record.slope == minimum_fitted_slope
    with pytest.raises(ValueError, match="float32 softplus"):
        ProbabilityCalibration(
            alpha=0.0,
            beta_raw=beta_raw,
            slope=minimum_fitted_slope * 1.01,
            weighted_log_loss=0.1,
            iterations=1,
            pooling=pooling,
        )


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
    empty = independent_block_support(
        block_id=("empty-a", "empty-b"),
        weight=np.zeros(2),
        observation=(0.0, 1.0),
    )
    assert empty == IndependentBlockSupport(0, 0.0, 0, 0)
    truly_empty = independent_block_support(
        block_id=(), weight=np.asarray([]), observation=np.asarray([])
    )
    assert truly_empty == IndependentBlockSupport(0, 0.0, 0, 0)
    with pytest.raises(ValueError, match="canonical non-empty string"):
        independent_block_support(
            block_id=(1, "1"),  # type: ignore[arg-type]
            weight=np.ones(2),
        )
    with pytest.raises(ValueError, match="canonical non-empty string"):
        independent_block_support(block_id=(" block-a",), weight=np.ones(1))
    sparse = IndependentBlockSupport(29, 19.0, 19, 19)
    decision = select_pooling_level(
        {
            "full_cell": sparse,
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
                "full_cell": sparse,
                "lead_provider_era_present": sparse,
                "lead_provider": sparse,
                "lead_only": sparse,
            },
            probability_gate=False,
        )


def test_calibration_summary_invariants_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot exceed block count"):
        IndependentBlockSupport(30, 1000.0, 0, 0)
    with pytest.raises(ValueError, match="must agree"):
        IndependentBlockSupport(0, 0.5, 0, 0)
    with pytest.raises(ValueError, match="m_k squared"):
        FoldCalibrationMoments(
            fold_id=0,
            oof_valid_mass=1.0,
            calibration_mean=10.0,
            calibration_second_moment=0.0,
        )
    with pytest.raises(ValueError, match="mean squared"):
        FoldCalibrationMoments(
            fold_id=0,
            oof_valid_mass=1.0,
            calibration_mean=1.0e200,
            calibration_second_moment=1.0,
        )
    with pytest.raises(ValueError, match="probability vector"):
        LocationScaleCalibration(
            0.0, 1.0, (-1.0, 2.0), 0.0, 1.0, 0.0, 1.0
        )
    with pytest.raises(ValueError, match="variances"):
        LocationScaleCalibration(
            0.0, 1.0, (0.5, 0.5), 0.0, -1.0, 0.0, 1.0
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
            "full_cell", "lead_provider_era_present", "lead_provider", "lead_only"
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


def test_physical_inverse_fails_closed_before_finite_expm1_overflow() -> None:
    mu = torch.full((1, 1, 1, 1), 100.0, dtype=torch.float32)
    residual = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
    with pytest.raises(OverflowError, match="representable float32 physical range"):
        transformed_members_to_physical(mu, residual)
