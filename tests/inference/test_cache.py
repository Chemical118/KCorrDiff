from __future__ import annotations

import pytest
import torch

from kcorrdiff.inference.cache import prepare_residual_edm_sampler
from kcorrdiff.models.common import configure_strict_fp32_runtime
from kcorrdiff.models.residual_edm import ResidualEDM
from kcorrdiff.training.edm_sampling import (
    build_ensemble_signature,
    sample_normalized_residual_ensemble,
)
from tests.models.test_residual_edm import SIZE, _conditions, _config


@pytest.fixture(autouse=True)
def _strict_fp32_runtime():
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    configure_strict_fp32_runtime()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def test_real_residual_edm_adapter_bridges_flattened_sampler_members() -> None:
    torch.manual_seed(9101)
    model = ResidualEDM(_config("edm_a")).eval()
    conditions = _conditions()
    adapter, cache = prepare_residual_edm_sampler(
        model, conditions, sample_ids=conditions.sample_ids
    )
    signature = build_ensemble_signature(
        checkpoint_id="sha256:real-tiny-edm",
        profile_name="development_smoke",
    )

    sampled = sample_normalized_residual_ensemble(
        denoise=adapter,
        condition_cache=cache,
        sample_ids=cache.sample_ids,
        lead_hours=1.5,
        spatial_shape=(SIZE, SIZE),
        ensemble_signature=signature,
    )

    assert sampled.shape == (1, 4, 1, SIZE, SIZE)
    assert sampled.dtype is torch.float32
    assert torch.isfinite(sampled).all()


def test_adapter_rejects_cache_from_another_model() -> None:
    first = ResidualEDM(_config("edm_a")).eval()
    second = ResidualEDM(_config("edm_a")).eval()
    conditions = _conditions()
    adapter, cache = prepare_residual_edm_sampler(
        first, conditions, sample_ids=conditions.sample_ids
    )
    other_adapter, _ = prepare_residual_edm_sampler(
        second, conditions, sample_ids=conditions.sample_ids
    )

    with pytest.raises(ValueError, match="another ResidualEDM"):
        other_adapter(
            torch.randn(4, 1, SIZE, SIZE),
            torch.ones(4),
            condition_cache=cache,
        )
