from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from kcorrdiff.models.common import (
    CONDITION_DIM,
    CONTEXT_WIDTHS,
    TARGET_WIDTHS,
    FullWidthRuntimeContract,
    assert_strict_fp32_runtime,
    configure_strict_fp32_runtime,
)
from kcorrdiff.models.condition_bank import ConditionBankConfig
from kcorrdiff.models.embeddings import (
    ConditionEmbedding,
    ConditionEmbeddingInputs,
)


def embedding_inputs(
    lead: list[float],
    cyclic: list[list[float]] | None = None,
) -> ConditionEmbeddingInputs:
    features = cyclic or [[0.0, 1.0, 0.0, 1.0] for _ in lead]
    return ConditionEmbeddingInputs(
        lead_hours=torch.tensor(lead, dtype=torch.float32),
        verification_cyclic=torch.tensor(features, dtype=torch.float32),
    )


def test_condition_embedding_is_central_fp32_and_sensitive_to_both_inputs() -> None:
    torch.manual_seed(8)
    module = ConditionEmbedding().eval()
    baseline = module(embedding_inputs([1.0]))
    repeated = module(embedding_inputs([1.0]))
    changed_lead = module(embedding_inputs([3.0]))
    changed_time = module(
        embedding_inputs([1.0], [[1.0, 0.0, 0.0, 1.0]])
    )

    assert isinstance(baseline, torch.Tensor)
    assert baseline.shape == (1, CONDITION_DIM)
    assert baseline.dtype is torch.float32
    torch.testing.assert_close(baseline, repeated, rtol=0.0, atol=0.0)
    assert not torch.equal(baseline, changed_lead)
    assert not torch.equal(baseline, changed_time)


def test_condition_embedding_rejects_zero_lead_downcast_and_autocast() -> None:
    with pytest.raises(ValueError, match="official"):
        embedding_inputs([0.0])
    with pytest.raises(TypeError, match="float32"):
        ConditionEmbeddingInputs(
            lead_hours=torch.tensor([1.0], dtype=torch.float64),
            verification_cyclic=torch.zeros(1, 4, dtype=torch.float32),
        )

    module = ConditionEmbedding()
    inputs = embedding_inputs([1.0])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            module(inputs)


def test_condition_embedding_revalidates_inputs_after_tensor_mutation() -> None:
    module = ConditionEmbedding()
    inputs = embedding_inputs([1.0])
    inputs.verification_cyclic[0, 0] = 2.0

    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        module(inputs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("precision", "bfloat16", "precision"),
        ("target_widths", (48, 96, 192, 288, 384), "target-width"),
        ("context_widths", (32, 48, 96, 192, 288), "context-width"),
        ("era_latent_channels", 96, "latent-width"),
        ("era_grid_size", 17, "spatial"),
        ("tf32_enabled", True, "TF32"),
        ("allow_cpu_fallback", True, "fallback"),
        ("allow_model_width_fallback", True, "fallback"),
        ("allow_precision_fallback", True, "fallback"),
        ("allow_era_grid_fallback", True, "fallback"),
    ],
)
def test_runtime_contract_rejects_every_silent_fallback(
    field: str, value: object, message: str
) -> None:
    contract = replace(FullWidthRuntimeContract(), **{field: value})
    with pytest.raises(ValueError, match=message):
        contract.validate()


def test_runtime_contract_rejects_cpu_when_cuda_is_required() -> None:
    with pytest.raises(RuntimeError, match="CPU fallback"):
        FullWidthRuntimeContract().validate(device="cpu", require_cuda=True)


def test_production_config_fails_closed_but_test_override_is_explicit() -> None:
    assert ConditionBankConfig().target_widths == TARGET_WIDTHS
    assert ConditionBankConfig().context_widths == CONTEXT_WIDTHS
    with pytest.raises(ValueError, match="fallback"):
        ConditionBankConfig(target_widths=(4, 8, 12, 16, 20))
    with pytest.raises(ValueError, match="spatial fallback"):
        ConditionBankConfig(input_size=16)
    with pytest.raises(ValueError, match="positive"):
        ConditionBankConfig(
            target_widths=(0, 8, 12, 16, 20),
            input_size=16,
            allow_test_override=True,
        )
    with pytest.raises(TypeError, match="integer widths"):
        ConditionBankConfig(
            target_widths=(4.0, 8, 12, 16, 20),
            input_size=16,
            allow_test_override=True,
        )
    explicit = ConditionBankConfig(
        target_widths=(4, 8, 12, 16, 20),
        context_widths=(4, 8, 12, 16, 20),
        input_size=16,
        allow_test_override=True,
    )
    assert explicit.input_size == 16


def test_explicit_module_downcast_is_rejected() -> None:
    module = ConditionEmbedding().double()
    with pytest.raises(TypeError, match="must remain float32"):
        module(embedding_inputs([1.0]))


def test_strict_fp32_backend_helper_disables_tf32() -> None:
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        configure_strict_fp32_runtime()
        assert_strict_fp32_runtime()
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
