from __future__ import annotations

import pytest
import torch

from kcorrdiff.models.embeddings import (
    DiffusionConditionFusionMLP,
    SigmaEmbedding,
)


def test_sigma_embedding_owns_log_sigma_convention_and_is_differentiable() -> None:
    module = SigmaEmbedding()
    sigma = torch.tensor([0.002, 0.5, 1.0, 80.0], dtype=torch.float32)
    output = module(sigma)
    assert output.shape == (4, 512)
    assert output.dtype is torch.float32
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert module.input_projection.weight.grad is not None


def test_sigma_embedding_rejects_invalid_or_implicit_precision() -> None:
    module = SigmaEmbedding()
    with pytest.raises(ValueError, match="strictly positive"):
        module(torch.tensor([0.0], dtype=torch.float32))
    with pytest.raises(ValueError, match="strictly positive"):
        module(torch.tensor([-1.0], dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        module(torch.ones(1, dtype=torch.float64))


def test_diffusion_condition_fusion_is_the_only_sigma_condition_boundary() -> None:
    module = DiffusionConditionFusionMLP()
    condition = torch.randn(3, 512, dtype=torch.float32, requires_grad=True)
    sigma_embedding = torch.randn(3, 512, dtype=torch.float32, requires_grad=True)
    output = module(condition, sigma_embedding)
    assert output.shape == (3, 512)
    assert output.dtype is torch.float32
    output.sum().backward()
    assert condition.grad is not None
    assert sigma_embedding.grad is not None


def test_diffusion_condition_fusion_rejects_shape_or_device_contract_changes() -> None:
    module = DiffusionConditionFusionMLP()
    with pytest.raises(ValueError, match=r"\[B,512\]"):
        module(torch.randn(2, 512), torch.randn(2, 256))


def test_diffusion_embeddings_reject_autocast() -> None:
    sigma = SigmaEmbedding()
    fusion = DiffusionConditionFusionMLP()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            sigma(torch.ones(1, dtype=torch.float32))
        with pytest.raises(RuntimeError, match="autocast"):
            fusion(torch.ones(1, 512), torch.ones(1, 512))
