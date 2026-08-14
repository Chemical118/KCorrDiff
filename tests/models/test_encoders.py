from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from kcorrdiff.models.condition_bank import ConditionBankInputs
from kcorrdiff.data.radar_values import (
    CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR,
    CPRECNET_RAIN_RATE_MAX_RTOL,
    rain_rate_to_cprecnet_normalized,
)
from kcorrdiff.models.context_encoder import (
    CONTEXT_DYNAMIC_CHANNEL_NAMES,
    ContextEncoder,
    ContextEncoderInputs,
    cprecnet_rate_to_normalized,
)
from kcorrdiff.models.target_encoder import (
    TARGET_STATIC_CHANNEL_NAMES,
    TargetEncoder,
    TargetEncoderInputs,
)

from .helpers import TEST_CONTEXT_WIDTHS, TEST_TARGET_WIDTHS, make_bank_inputs


def test_full_five_level_shapes_and_context_l2_l3_l4_are_exposed() -> None:
    inputs = make_bank_inputs(batch=2, size=32)
    target_encoder = TargetEncoder(
        widths=TEST_TARGET_WIDTHS,
        input_size=32,
        allow_test_override=True,
    )
    context_encoder = ContextEncoder(
        widths=TEST_CONTEXT_WIDTHS,
        input_size=32,
        allow_test_override=True,
    )
    target = target_encoder(inputs.target)
    context = context_encoder(inputs.context)

    expected_spatial = (32, 16, 8, 4, 2)
    assert [tuple(level.shape) for level in target.features.levels] == [
        (2, width, size, size)
        for width, size in zip(TEST_TARGET_WIDTHS, expected_spatial)
    ]
    assert [tuple(level.shape) for level in context.features.levels] == [
        (2, width, size, size)
        for width, size in zip(TEST_CONTEXT_WIDTHS, expected_spatial)
    ]
    assert context.l2 is context.features.l2
    assert context.l3 is context.features.l3
    assert context.l4 is context.features.l4
    assert [tuple(level.shape) for level in context.validity.levels] == [
        (2, 1, size, size) for size in expected_spatial
    ]


def test_temporal_stem_is_order_sensitive_and_static_is_evaluated_once() -> None:
    inputs = make_bank_inputs(batch=1, size=16)
    encoder = TargetEncoder(
        widths=TEST_TARGET_WIDTHS,
        input_size=16,
        allow_test_override=True,
    ).eval()
    count = 0

    def hook(_module, _args, _output):
        nonlocal count
        count += 1

    handle = encoder.static_stem.register_forward_hook(hook)
    chronological = encoder(inputs.target)
    reversed_input = TargetEncoderInputs(
        radar_history=inputs.target.radar_history.flip(1),
        history_validity=inputs.target.history_validity.flip(1),
        static_fields=inputs.target.static_fields,
        static_coverage=inputs.target.static_coverage,
    )
    reversed_output = encoder(reversed_input)
    handle.remove()

    assert count == 2  # exactly once for each complete encoder call
    assert not torch.equal(
        chronological.temporal_level_zero,
        reversed_output.temporal_level_zero,
    )


def test_rate_transform_matches_data_contract_and_censors_below_threshold() -> None:
    generator = torch.Generator().manual_seed(1207)
    rates = torch.rand(2, 12, 2, 8, 8, generator=generator) * 1000.0
    rates[0, 0, 0, 0, :4] = torch.tensor([0.0, 1.0e-3, 10.0**-1.5, 1000.0])

    normalized = cprecnet_rate_to_normalized(rates)

    expected = rain_rate_to_cprecnet_normalized(rates.double().numpy())
    torch.testing.assert_close(
        normalized.double(),
        torch.from_numpy(expected),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert normalized.min().item() >= 0.0
    assert normalized.max().item() <= 1.0
    assert torch.equal(
        normalized[0, 0, 0, 0, :3], torch.zeros(3)
    )  # below-threshold rates censor to exact zero
    assert normalized[0, 0, 0, 0, 3].item() == pytest.approx(1.0)


def test_context_encoder_consumes_rates_through_the_normalized_space() -> None:
    inputs = make_bank_inputs(batch=1, size=16)
    encoder = ContextEncoder(
        widths=TEST_CONTEXT_WIDTHS,
        input_size=16,
        allow_test_override=True,
    ).eval()

    censored_dynamic = inputs.context.dynamic_fields.clone()
    censored_dynamic[:, :, :2] = torch.where(
        censored_dynamic[:, :, :2] > 10.0**-1.5,
        censored_dynamic[:, :, :2],
        0.0,
    )
    with torch.no_grad():
        from_rates = encoder(inputs.context)
        from_censored = encoder(
            ContextEncoderInputs(
                dynamic_fields=censored_dynamic,
                detail_validity=inputs.context.detail_validity,
                static_fields=inputs.context.static_fields,
            )
        )

    torch.testing.assert_close(
        from_rates.temporal_level_zero, from_censored.temporal_level_zero
    )


def test_schema_and_issue_time_boundary_reject_wrong_or_future_names() -> None:
    inputs = make_bank_inputs(size=16)
    assert {field.name for field in fields(ConditionBankInputs)} == {
        "target",
        "context",
    }
    assert not any(
        token in field.name
        for field in fields(ConditionBankInputs)
        for token in ("future", "label", "m_target", "target_validity")
    )
    with pytest.raises(ValueError, match="static channel"):
        TargetEncoderInputs(
            radar_history=inputs.target.radar_history,
            history_validity=inputs.target.history_validity,
            static_fields=inputs.target.static_fields,
            static_coverage=inputs.target.static_coverage,
            static_channel_names=(*TARGET_STATIC_CHANNEL_NAMES[:-1], "target_z"),
        )
    with pytest.raises(ValueError, match="dynamic channel"):
        ContextEncoderInputs(
            dynamic_fields=inputs.context.dynamic_fields,
            detail_validity=inputs.context.detail_validity,
            static_fields=inputs.context.static_fields,
            dynamic_channel_names=(
                *CONTEXT_DYNAMIC_CHANNEL_NAMES[:-1],
                "future_target_validity",
            ),
        )


def test_input_contract_rejects_float64_bad_validity_and_invalid_coverage() -> None:
    inputs = make_bank_inputs(size=16)
    with pytest.raises(TypeError, match="float32"):
        TargetEncoderInputs(
            radar_history=inputs.target.radar_history.double(),
            history_validity=inputs.target.history_validity,
            static_fields=inputs.target.static_fields,
            static_coverage=inputs.target.static_coverage,
        )
    with pytest.raises(ValueError, match="outside static coverage"):
        TargetEncoderInputs(
            radar_history=inputs.target.radar_history,
            history_validity=torch.ones_like(inputs.target.history_validity),
            static_fields=inputs.target.static_fields,
            static_coverage=torch.zeros_like(inputs.target.static_coverage),
        )
    invalid_context = inputs.context.dynamic_fields.clone()
    invalid_context[:, :, 3] = 1.1
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        ContextEncoderInputs(
            dynamic_fields=invalid_context,
            detail_validity=inputs.context.detail_validity,
            static_fields=inputs.context.static_fields,
        )


def test_context_input_contract_enforces_cprecnet_rain_rate_maximum() -> None:
    inputs = make_bank_inputs(size=16)
    within_tolerance = inputs.context.dynamic_fields.clone()
    within_tolerance[0, 0, 0, 0, 0] = CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR * (
        1.0 + 0.5 * CPRECNET_RAIN_RATE_MAX_RTOL
    )
    ContextEncoderInputs(
        dynamic_fields=within_tolerance,
        detail_validity=inputs.context.detail_validity,
        static_fields=inputs.context.static_fields,
    )

    above_tolerance = inputs.context.dynamic_fields.clone()
    above_tolerance[0, 0, 0, 0, 0] = CPRECNET_RAIN_RATE_MAX_MM_PER_HOUR * (
        1.0 + 2.0 * CPRECNET_RAIN_RATE_MAX_RTOL
    )
    with pytest.raises(ValueError, match="representable maximum"):
        ContextEncoderInputs(
            dynamic_fields=above_tolerance,
            detail_validity=inputs.context.detail_validity,
            static_fields=inputs.context.static_fields,
        )


def test_batch_gradients_reach_temporal_static_and_pyramid_parameters() -> None:
    inputs = make_bank_inputs(batch=2, size=16, requires_grad=True)
    encoder = TargetEncoder(
        widths=TEST_TARGET_WIDTHS,
        input_size=16,
        allow_test_override=True,
    )
    output = encoder(inputs.target)
    loss = sum(level.square().mean() for level in output.features.levels)
    loss.backward()

    assert inputs.target.radar_history.grad is not None
    assert inputs.target.static_fields.grad is not None
    assert encoder.temporal_stem.frame_projection.weight.grad is not None
    assert encoder.temporal_stem.temporal_depthwise.weight.grad is not None
    assert encoder.temporal_stem.temporal_score.weight.grad is not None
    assert encoder.static_stem.projection.weight.grad is not None
    assert encoder.pyramid.downsamples[-1].weight.grad is not None


def test_non_reentrant_checkpoint_matches_forward_and_backward() -> None:
    torch.manual_seed(42)
    plain = TargetEncoder(
        widths=TEST_TARGET_WIDTHS,
        input_size=16,
        activation_checkpoint=False,
        allow_test_override=True,
    )
    checkpointed = TargetEncoder(
        widths=TEST_TARGET_WIDTHS,
        input_size=16,
        activation_checkpoint=True,
        allow_test_override=True,
    )
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    inputs = make_bank_inputs(batch=1, size=16)

    plain_output = plain(inputs.target)
    checkpoint_output = checkpointed(inputs.target)
    for left, right in zip(
        plain_output.features.levels, checkpoint_output.features.levels
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    sum(level.square().mean() for level in plain_output.features.levels).backward()
    sum(
        level.square().mean() for level in checkpoint_output.features.levels
    ).backward()
    for (plain_name, plain_parameter), (check_name, check_parameter) in zip(
        plain.named_parameters(), checkpointed.named_parameters()
    ):
        assert plain_name == check_name
        assert plain_parameter.grad is not None, plain_name
        assert check_parameter.grad is not None, check_name
        torch.testing.assert_close(
            plain_parameter.grad,
            check_parameter.grad,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
