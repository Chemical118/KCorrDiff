from __future__ import annotations

import pytest
import torch

from kcorrdiff.training.edm_loss import (
    build_noisy_residual_training_state,
    edm_preconditioning,
    masked_edm_loss_terms,
)


def test_edm_preconditioning_matches_sigma_data_one_equations() -> None:
    sigma = torch.tensor([0.5, 2.0], dtype=torch.float32)
    result = edm_preconditioning(sigma)
    shaped = sigma[:, None, None, None]
    denominator = shaped.square() + 1.0
    assert torch.allclose(result.c_skip, 1.0 / denominator)
    assert torch.allclose(result.c_out, shaped / torch.sqrt(denominator))
    assert torch.allclose(result.c_in, torch.rsqrt(denominator))
    assert torch.allclose(result.c_noise, torch.log(shaped) / 4.0)
    with pytest.raises(ValueError, match="must equal 1.0"):
        edm_preconditioning(sigma, sigma_data=0.5)


def test_invalid_clean_is_neutral_but_noise_is_present_on_every_pixel() -> None:
    clean = torch.tensor([[[[4.0, float("nan")], [2.0, -8.0]]]])
    valid = torch.tensor([[[[True, False], [True, False]]]])
    sigma = torch.tensor([2.0], dtype=torch.float32)
    noise = torch.tensor([[[[1.0, 3.0], [-1.0, 5.0]]]])
    state = build_noisy_residual_training_state(
        clean_normalized_residual=clean,
        target_validity=valid,
        sigma=sigma,
        noise=noise,
    )
    assert torch.equal(
        state.neutral_filled_clean,
        torch.tensor([[[[4.0, 0.0], [2.0, 0.0]]]]),
    )
    assert torch.equal(
        state.noisy_residual,
        torch.tensor([[[[6.0, 6.0], [0.0, 10.0]]]]),
    )
    assert torch.count_nonzero(state.noisy_residual[~valid]) == 2


def test_masked_edm_terms_match_manual_importance_weighted_formula() -> None:
    prediction = torch.tensor(
        [[[[2.0, 8.0]]], [[[1.0, 5.0]]]], requires_grad=True
    )
    clean = torch.tensor([[[[1.0, 0.0]]], [[[3.0, float("nan")]]]])
    valid = torch.tensor([[[[True, False]]], [[[True, False]]]])
    omega = torch.tensor([2.0, 3.0])
    sigma = torch.tensor([1.0, 2.0])
    terms = masked_edm_loss_terms(
        denoised_normalized_residual=prediction,
        clean_normalized_residual=clean,
        target_validity=valid,
        omega=omega,
        sigma=sigma,
    )
    # EDM weights: 2 for sigma=1 and 1.25 for sigma=2.
    expected_numerator = 2.0 * 2.0 * 1.0**2 + 3.0 * 1.25 * 2.0**2
    assert terms.weighted_square_error_sum.item() == pytest.approx(expected_numerator)
    assert terms.valid_importance_mass.item() == pytest.approx(5.0)
    assert terms.local_mean.item() == pytest.approx(expected_numerator / 5.0)
    terms.local_mean.backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0, 0, 1] == 0.0
    assert prediction.grad[1, 0, 0, 1] == 0.0


def test_rank_local_normalization_is_visibly_not_a_distributed_substitute() -> None:
    # This regression test preserves the reason the API exposes numerator and
    # denominator separately: unequal valid mass makes mean(local means) wrong.
    first = masked_edm_loss_terms(
        denoised_normalized_residual=torch.tensor([[[[2.0]]]]),
        clean_normalized_residual=torch.tensor([[[[0.0]]]]),
        target_validity=torch.tensor([[[[True]]]]),
        omega=torch.tensor([1.0]),
        sigma=torch.tensor([1.0]),
    )
    second = masked_edm_loss_terms(
        denoised_normalized_residual=torch.tensor([[[[1.0, 1.0]]]]),
        clean_normalized_residual=torch.zeros(1, 1, 1, 2),
        target_validity=torch.ones(1, 1, 1, 2, dtype=torch.bool),
        omega=torch.tensor([3.0]),
        sigma=torch.tensor([1.0]),
    )
    local_average = (first.local_mean + second.local_mean) / 2.0
    global_value = (
        first.weighted_square_error_sum + second.weighted_square_error_sum
    ) / (first.valid_importance_mass + second.valid_importance_mass)
    assert not torch.allclose(local_average, global_value)
