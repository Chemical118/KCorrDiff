from __future__ import annotations

import pytest
import torch

from kcorrdiff.models.advection import AdvAdapter


def test_adapter_preserves_full_width_and_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(3)
    adapter = AdvAdapter(target_channels=16, advection_channels=8, condition_dim=12)
    target = torch.randn(2, 16, 32, 32)
    advection = torch.randn(2, 8, 32, 32)
    condition = torch.randn(2, 12)
    confidence = torch.rand(2, 1, 32, 32)
    output = adapter(target, advection, condition, confidence)
    assert output.shape == target.shape
    assert torch.equal(output, target)
    assert torch.count_nonzero(adapter.residual_projection.weight) == 0
    assert torch.count_nonzero(adapter.residual_projection.bias) == 0
    # Gate layers must not be zero-initialized with the residual projection.
    assert torch.count_nonzero(adapter.condition_gate.weight) > 0
    assert torch.count_nonzero(adapter.confidence_gate.weight) > 0


def test_projection_gets_first_step_gradient_and_confidence_zero_is_exact_off() -> None:
    adapter = AdvAdapter(target_channels=8, advection_channels=8, condition_dim=4)
    target = torch.randn(2, 8, 12, 12, requires_grad=True)
    advection = torch.randn(2, 8, 12, 12, requires_grad=True)
    condition = torch.randn(2, 4, requires_grad=True)
    confidence = torch.rand(2, 1, 12, 12)
    adapter(target, advection, condition, confidence).square().mean().backward()
    gradient = adapter.residual_projection.weight.grad
    assert gradient is not None and torch.count_nonzero(gradient) > 0

    with torch.no_grad():
        adapter.residual_projection.weight.normal_()
        adapter.residual_projection.bias.normal_()
    off = adapter(target, advection, condition, torch.zeros_like(confidence))
    assert torch.equal(off, target)


def test_adapter_validates_shapes_and_confidence() -> None:
    adapter = AdvAdapter(target_channels=8, advection_channels=4, condition_dim=6)
    target = torch.zeros(1, 8, 5, 5)
    advection = torch.zeros(1, 4, 5, 5)
    condition = torch.zeros(1, 6)
    confidence = torch.ones(1, 1, 5, 5)
    with pytest.raises(ValueError, match="spatial shapes"):
        adapter(target, advection[:, :, :-1], condition, confidence)
    with pytest.raises(ValueError, match="e_cond"):
        adapter(target, advection, condition[:, :-1], confidence)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        adapter(target, advection, condition, confidence * 2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_adapter_cuda_shape_and_gradient() -> None:
    adapter = AdvAdapter(
        target_channels=32, advection_channels=8, condition_dim=16
    ).cuda()
    target = torch.randn(2, 32, 16, 16, device="cuda")
    advection = torch.randn(2, 8, 16, 16, device="cuda")
    condition = torch.randn(2, 16, device="cuda")
    confidence = torch.rand(2, 1, 16, 16, device="cuda")
    output = adapter(target, advection, condition, confidence)
    assert output.shape == target.shape
    output.sum().backward()
    assert adapter.residual_projection.weight.grad is not None
