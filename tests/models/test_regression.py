from __future__ import annotations

import math

import pytest
import torch

from kcorrdiff.data.radar_values import A0_MM, A_WET_MM
from kcorrdiff.models.common import FeaturePyramid, ValidityPyramid
from kcorrdiff.models.condition_bank import ConditionBankCache
from kcorrdiff.models.context_encoder import ContextEncoderOutput
from kcorrdiff.models.era_encoder import EraQueryResult
from kcorrdiff.models.physical_attention import PhysicalTokenGeometry
from kcorrdiff.models.regression import (
    DirectPhysicalRegression,
    RegressionGeometry,
    RegressionInputs,
    RegressionUNet,
)
from kcorrdiff.models.target_encoder import TargetEncoderOutput


TARGET = (8, 16, 24, 32, 40)
CONTEXT = (8, 12, 16, 24, 32)


def pyramid(widths: tuple[int, ...], *, batch: int = 1) -> FeaturePyramid:
    size = 16
    levels = []
    for width in widths:
        levels.append(torch.randn(batch, width, size, size))
        size //= 2
    return FeaturePyramid(*levels, widths=widths)


def validity(*, batch: int = 1) -> ValidityPyramid:
    levels = []
    size = 16
    for _ in range(5):
        levels.append(torch.ones(batch, 1, size, size))
        size //= 2
    return ValidityPyramid(*levels)


def geometry(height: int, width: int) -> PhysicalTokenGeometry:
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    return PhysicalTokenGeometry(
        x_shared=x,
        y_shared=y,
        footprint_width=torch.ones_like(x),
        footprint_height=torch.ones_like(y),
    )


def inputs() -> RegressionInputs:
    target_features = pyramid(TARGET)
    context_features = pyramid(CONTEXT)
    target_output = TargetEncoderOutput(
        target_features,
        validity(),
        torch.zeros(1, TARGET[0] // 2, 16, 16),
        torch.zeros(1, TARGET[0] // 2, 16, 16),
    )
    context_output = ContextEncoderOutput(
        context_features,
        validity(),
        torch.zeros(1, CONTEXT[0] // 2, 16, 16),
        torch.zeros(1, CONTEXT[0] // 2, 16, 16),
    )
    cache = ConditionBankCache(
        target_output,
        context_output,
        16,
        TARGET,
        CONTEXT,
    )
    era = EraQueryResult(
        features=torch.randn(1, 128, 33, 33),
        temporal_weights=torch.full((1, 8, 8), 1.0 / 8.0),
        valid_token_mask=torch.ones(1, 8, dtype=torch.bool),
        tp_token_mask=torch.ones(1, 8, dtype=torch.bool),
        used_source_null=torch.zeros(1, dtype=torch.bool),
        used_masked_null=torch.zeros(1, dtype=torch.bool),
    )
    return RegressionInputs(
        condition_bank=cache,
        era=era,
        advection_features=torch.randn(1, 8, 16, 16),
        e_cond=torch.randn(1, 512),
        geometry=RegressionGeometry(
            target_l3=geometry(2, 2),
            target_l4=geometry(1, 1),
            context_l3=geometry(2, 2),
            context_l4=geometry(1, 1),
            era_native=geometry(33, 33),
        ),
        condition_signatures=("era5_oracle:era=1:tp=1:full_trajectory",),
    )


def test_regression_exact_output_support_and_zero_initialized_source_gates() -> None:
    torch.manual_seed(1)
    model = RegressionUNet(
        target_widths=TARGET,
        context_widths=CONTEXT,
        query_chunk_size=2,
        allow_test_override=True,
    )
    result = model(inputs())
    assert result.occurrence_logits.shape == (1, 1, 16, 16)
    assert result.probability_wet.shape == result.occurrence_logits.shape
    assert torch.all((result.probability_wet > 0) & (result.probability_wet < 1))
    assert result.wet_amount.min() >= math.log1p(A_WET_MM / A0_MM)
    assert torch.equal(result.transformed_mean, result.probability_wet * result.wet_amount)
    for contribution in (
        result.context_l3_contribution,
        result.era_l3_contribution,
        result.context_l4_contribution,
        result.era_l4_contribution,
    ):
        assert torch.count_nonzero(contribution) == 0


def test_source_gates_receive_gradient_while_attention_projection_is_not_zeroed() -> None:
    torch.manual_seed(2)
    model = RegressionUNet(
        target_widths=TARGET,
        context_widths=CONTEXT,
        query_chunk_size=2,
        allow_test_override=True,
    )
    values = inputs()
    output = model(values)
    output.transformed_mean.mean().backward()
    gate = model.context_attention_l3.gate_projection
    assert gate.weight.grad is not None
    assert torch.count_nonzero(gate.weight.grad) > 0
    assert torch.count_nonzero(model.context_attention_l3.output_projection.weight) > 0


def test_checkpointed_attention_and_decoder_match_plain_forward_backward() -> None:
    torch.manual_seed(211)
    plain = RegressionUNet(
        target_widths=TARGET,
        context_widths=CONTEXT,
        query_chunk_size=2,
        activation_checkpoint=False,
        allow_test_override=True,
    ).train()
    with torch.no_grad():
        for name, module in plain.named_modules():
            if name.endswith(("attention_l3", "attention_l4")):
                module.gate_projection.bias.fill_(0.3)
    checkpointed = RegressionUNet(
        target_widths=TARGET,
        context_widths=CONTEXT,
        query_chunk_size=2,
        activation_checkpoint=True,
        allow_test_override=True,
    ).train()
    checkpointed.load_state_dict(plain.state_dict())
    values = inputs()

    plain_output = plain(values)
    checkpointed_output = checkpointed(values)
    for name in (
        "occurrence_logits",
        "probability_wet",
        "wet_amount",
        "transformed_mean",
        "context_l3_contribution",
        "era_l3_contribution",
        "context_l4_contribution",
        "era_l4_contribution",
        "decoder_features",
    ):
        torch.testing.assert_close(
            getattr(checkpointed_output, name),
            getattr(plain_output, name),
            rtol=1e-6,
            atol=1e-6,
        )

    plain_output.transformed_mean.mean().backward()
    checkpointed_output.transformed_mean.mean().backward()
    for (plain_name, plain_parameter), (checked_name, checked_parameter) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert plain_name == checked_name
        assert plain_parameter.grad is not None
        assert checked_parameter.grad is not None
        torch.testing.assert_close(
            checked_parameter.grad,
            plain_parameter.grad,
            rtol=3e-5,
            atol=3e-6,
        )
    assert checkpointed.context_attention_l3.activation_checkpoint is True
    assert checkpointed.era_attention_l4.activation_checkpoint is True
    assert checkpointed.parameter_count == plain.parameter_count


def test_missing_era_is_finite_and_cannot_contribute() -> None:
    model = RegressionUNet(
        target_widths=TARGET,
        context_widths=CONTEXT,
        query_chunk_size=2,
        allow_test_override=True,
    )
    values = inputs()
    values.era.used_source_null.fill_(True)
    with torch.no_grad():
        model.era_attention_l3.gate_projection.bias.fill_(1.0)
        model.era_attention_l4.gate_projection.bias.fill_(1.0)
    output = model(values)
    assert torch.isfinite(output.transformed_mean).all()
    assert torch.count_nonzero(output.era_l3_contribution) == 0
    assert torch.count_nonzero(output.era_l4_contribution) == 0


def test_production_model_rejects_width_fallback() -> None:
    with pytest.raises(ValueError, match="fallback"):
        RegressionUNet(target_widths=TARGET, context_widths=CONTEXT)


@pytest.mark.parametrize("statistic", ["mean", "q50"])
def test_direct_physical_comparison_uses_separate_nonnegative_checkpoint(
    statistic: str,
) -> None:
    model = DirectPhysicalRegression(
        statistic=statistic,
        target_widths=TARGET,
        context_widths=CONTEXT,
        query_chunk_size=2,
        allow_test_override=True,
    )
    output = model(inputs())
    assert output.statistic == statistic
    assert output.prediction_mm.shape == (1, 1, 16, 16)
    assert torch.all(output.prediction_mm >= 0.0)
    output.prediction_mm.mean().backward()
    assert model.physical_head.weight.grad is not None


def test_direct_physical_constructor_propagates_activation_checkpoint() -> None:
    model = DirectPhysicalRegression(
        statistic="mean",
        target_widths=TARGET,
        context_widths=CONTEXT,
        query_chunk_size=2,
        activation_checkpoint=True,
        allow_test_override=True,
    )
    assert model.backbone.activation_checkpoint is True
    assert model.backbone.context_attention_l3.activation_checkpoint is True
    assert model.backbone.era_attention_l4.activation_checkpoint is True
