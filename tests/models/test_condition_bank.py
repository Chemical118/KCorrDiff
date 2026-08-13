from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from io import BytesIO
import pickle

import pytest
import torch

from kcorrdiff.models.common import (
    CONTEXT_WIDTHS,
    TARGET_WIDTHS,
    FeaturePyramid,
    count_parameters,
)
from kcorrdiff.models.condition_bank import (
    ConditionBank,
    ConditionBankCache,
    ConditionBankConfig,
)
from kcorrdiff.models.embeddings import ConditionEmbeddingInputs

from .helpers import TEST_CONTEXT_WIDTHS, TEST_TARGET_WIDTHS, make_bank_inputs


def small_config(*, checkpoint: bool = False) -> ConditionBankConfig:
    return ConditionBankConfig(
        target_widths=TEST_TARGET_WIDTHS,
        context_widths=TEST_CONTEXT_WIDTHS,
        input_size=16,
        activation_checkpoint=checkpoint,
        allow_test_override=True,
    )


def test_production_parameter_count_and_width_memory_budget_are_exact() -> None:
    bank = ConditionBank()
    assert bank.config.target_widths == TARGET_WIDTHS
    assert bank.config.context_widths == CONTEXT_WIDTHS
    assert count_parameters(bank.target_encoder) == 20_917_089
    assert count_parameters(bank.context_encoder) == 9_732_049
    assert count_parameters(bank.condition_embedding) == 888_064
    assert bank.parameter_count == 31_537_202
    assert bank.parameter_count * 4 == 126_148_808  # FP32 parameter bytes.


def test_production_pyramid_shape_contract_is_exact_without_fallback_forward() -> None:
    spatial = (256, 128, 64, 32, 16)
    target = FeaturePyramid(
        *(
            torch.empty(1, width, size, size, device="meta")
            for width, size in zip(TARGET_WIDTHS, spatial)
        ),
        widths=TARGET_WIDTHS,
    )
    context = FeaturePyramid(
        *(
            torch.empty(1, width, size, size, device="meta")
            for width, size in zip(CONTEXT_WIDTHS, spatial)
        ),
        widths=CONTEXT_WIDTHS,
    )
    assert [tuple(level.shape) for level in target.levels] == [
        (1, width, size, size)
        for width, size in zip(TARGET_WIDTHS, spatial)
    ]
    assert [tuple(level.shape) for level in context.levels] == [
        (1, width, size, size)
        for width, size in zip(CONTEXT_WIDTHS, spatial)
    ]


def test_cache_is_immutable_lead_independent_and_exposes_context_levels() -> None:
    bank = ConditionBank(small_config()).eval()
    cache = bank(make_bank_inputs(batch=2, size=16))
    assert isinstance(cache, ConditionBankCache)
    assert cache.batch_size == 2
    assert cache.context_l2 is cache.context.l2
    assert cache.context_l3 is cache.context.l3
    assert cache.context_l4 is cache.context.l4
    assert not {
        "future",
        "label",
        "m_target",
        "target_validity",
        "z_tau",
    } & {field.name for field in fields(cache)}
    with pytest.raises(FrozenInstanceError):
        cache.input_size = 32  # type: ignore[misc]

    first = bank.embed_condition(
        ConditionEmbeddingInputs(
            lead_hours=torch.tensor([0.5, 0.5]),
            verification_cyclic=torch.tensor(
                [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
            ),
        )
    )
    second = bank.embed_condition(
        ConditionEmbeddingInputs(
            lead_hours=torch.tensor([6.0, 6.0]),
            verification_cyclic=torch.tensor(
                [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
            ),
        )
    )
    assert not torch.equal(first, second)
    # Per-lead embedding does not mutate or replace shared cached tensors.
    assert cache.context_l3 is cache.context.features.l3


def test_cache_detach_and_bank_freeze_match_section_12_3() -> None:
    bank = ConditionBank(small_config())
    cache = bank(make_bank_inputs(size=16, requires_grad=True))
    assert any(level.requires_grad for level in cache.target.features.levels)
    detached = cache.detached()
    assert not any(level.requires_grad for level in detached.target.features.levels)
    assert not any(level.requires_grad for level in detached.context.features.levels)

    returned = bank.freeze_for_diffusion()
    assert returned is bank
    assert bank.training is False
    assert all(not parameter.requires_grad for parameter in bank.parameters())


def test_state_dict_and_immutable_cache_serialization_round_trip() -> None:
    torch.manual_seed(77)
    bank = ConditionBank(small_config()).eval()
    inputs = make_bank_inputs(size=16)
    expected = bank(inputs)
    expected_embedding = bank.embed_condition(
        ConditionEmbeddingInputs(
            lead_hours=torch.tensor([2.0]),
            verification_cyclic=torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        )
    )

    buffer = BytesIO()
    torch.save(bank.state_dict(), buffer)
    buffer.seek(0)
    restored = ConditionBank(small_config()).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    actual = restored(inputs)
    actual_embedding = restored.embed_condition(
        ConditionEmbeddingInputs(
            lead_hours=torch.tensor([2.0]),
            verification_cyclic=torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        )
    )

    for expected_level, actual_level in zip(
        expected.target.features.levels, actual.target.features.levels
    ):
        torch.testing.assert_close(expected_level, actual_level)
    for expected_level, actual_level in zip(
        expected.context.features.levels, actual.context.features.levels
    ):
        torch.testing.assert_close(expected_level, actual_level)
    torch.testing.assert_close(expected_embedding, actual_embedding)

    cache_round_trip = pickle.loads(pickle.dumps(expected.detached()))
    assert isinstance(cache_round_trip, ConditionBankCache)
    torch.testing.assert_close(cache_round_trip.target.l4, expected.target.l4)


def test_bank_rejects_untyped_inputs_and_has_no_label_argument() -> None:
    bank = ConditionBank(small_config())
    with pytest.raises(TypeError, match="ConditionBankInputs"):
        bank.encode_shared({})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bank(make_bank_inputs(size=16), label=torch.zeros(1))  # type: ignore[call-arg]
