from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from kcorrdiff.data.radar_values import A0_MM, A_WET_MM
from kcorrdiff.training.losses import (
    RegressionLossConfig,
    accumulation_window_denominators,
    direct_physical_mean_loss,
    direct_physical_quantile_loss,
    distributed_hurdle_loss_contribution,
    hurdle_regression_loss,
)


def tensors() -> dict[str, torch.Tensor]:
    shape = (2, 1, 2, 2)
    wet = torch.tensor(
        [[[[1, 0], [0, 1]]], [[[0, 0], [1, 0]]]], dtype=torch.bool
    )
    valid = torch.ones(shape, dtype=torch.bool)
    valid[1, :, 0, 0] = False
    target = wet.float() * 0.4
    return {
        "occurrence_logits": torch.zeros(shape, dtype=torch.float32, requires_grad=True),
        "wet_amount": torch.full(shape, 0.4, dtype=torch.float32, requires_grad=True),
        "target_z": target,
        "target_wet": wet,
        "target_validity": valid,
        "omega": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }


def test_hurdle_loss_matches_manual_weighting_and_warmup() -> None:
    values = tensors()
    config = RegressionLossConfig(lambda_mean=0.25, mean_warmup_steps=10)
    warm = hurdle_regression_loss(**values, global_step=9, config=config)
    active = hurdle_regression_loss(**values, global_step=10, config=config)
    assert warm.active_mean_coefficient == 0.0
    assert active.active_mean_coefficient == 0.25
    assert warm.occurrence.item() == pytest.approx(math.log(2.0))
    assert active.total > warm.total
    active.total.backward()
    assert values["occurrence_logits"].grad is not None
    assert values["wet_amount"].grad is not None


def test_empty_wet_microbatch_has_connected_zero_positive_loss() -> None:
    values = tensors()
    values["target_wet"].zero_()
    values["target_z"].zero_()
    result = hurdle_regression_loss(**values, global_step=0)
    assert result.positive_amount.item() == 0.0
    assert result.wet_weight.item() == 0.0
    result.total.backward()
    assert torch.isfinite(values["wet_amount"].grad).all()


def test_support_and_future_mask_contracts_fail_closed() -> None:
    values = tensors()
    values["wet_amount"] = torch.full(
        values["wet_amount"].shape,
        math.log1p(A_WET_MM / A0_MM) - 0.01,
        dtype=torch.float32,
    )
    with pytest.raises(ValueError, match="support"):
        hurdle_regression_loss(**values, global_step=0)


def test_direct_physical_comparison_losses() -> None:
    prediction = torch.tensor([[[[0.0, 2.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    valid = torch.ones_like(target, dtype=torch.bool)
    omega = torch.ones(1)
    assert direct_physical_mean_loss(prediction, target, valid, omega).item() == 2.5
    assert direct_physical_quantile_loss(
        prediction, target, valid, omega, quantile=0.5
    ).item() == 0.75


def test_distributed_numerator_contributions_match_concatenated_gradient() -> None:
    reference = tensors()
    config = RegressionLossConfig(lambda_mean=0.25, mean_warmup_steps=0)
    reference_loss = hurdle_regression_loss(
        **reference, global_step=0, config=config
    )
    reference_loss.total.backward()
    expected_logits = reference["occurrence_logits"].grad.detach().clone()
    expected_amount = reference["wet_amount"].grad.detach().clone()

    split_logits = reference["occurrence_logits"].detach().clone().requires_grad_()
    split_amount = reference["wet_amount"].detach().clone().requires_grad_()
    labels = SimpleNamespace(
        target_validity=reference["target_validity"],
        target_wet=reference["target_wet"],
        omega=reference["omega"],
    )
    denominators = accumulation_window_denominators([labels], device="cpu")
    for row in range(2):
        result = distributed_hurdle_loss_contribution(
            occurrence_logits=split_logits[row : row + 1],
            wet_amount=split_amount[row : row + 1],
            target_z=reference["target_z"][row : row + 1],
            target_wet=reference["target_wet"][row : row + 1],
            target_validity=reference["target_validity"][row : row + 1],
            omega=reference["omega"][row : row + 1],
            global_step=0,
            denominators=denominators,
            config=config,
        )
        result.local_total.backward()
    torch.testing.assert_close(split_logits.grad, expected_logits)
    torch.testing.assert_close(split_amount.grad, expected_amount)


def test_empty_padding_rank_can_reduce_window_mass_without_labels() -> None:
    expected = torch.tensor([8.0, 3.0], dtype=torch.float64)

    def reduce_sum(local: torch.Tensor) -> torch.Tensor:
        assert torch.equal(local, torch.zeros_like(local))
        return expected.clone()

    result = accumulation_window_denominators(
        [], device="cpu", reduce_sum=reduce_sum
    )
    assert result.valid_weight.item() == 8.0
    assert result.wet_weight.item() == 3.0
