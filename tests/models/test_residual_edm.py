from __future__ import annotations

from dataclasses import fields, replace
import gc

import pytest
import torch

from kcorrdiff.models.common import (
    FeaturePyramid,
    ValidityPyramid,
    configure_strict_fp32_runtime,
)
from kcorrdiff.models.condition_bank import ConditionBankCache
from kcorrdiff.models.context_encoder import ContextEncoderOutput
from kcorrdiff.models.era_encoder import EraQueryResult
from kcorrdiff.models.physical_attention import PhysicalTokenGeometry
from kcorrdiff.models.regression import RegressionGeometry
from kcorrdiff.models.residual_edm import (
    EDM_A_PRODUCTION_PARAMETER_COUNT,
    EDM_B_PRODUCTION_PARAMETER_COUNT,
    ResidualEDM,
    ResidualEDMConditions,
    ResidualEDMConfig,
)
from kcorrdiff.models.target_encoder import TargetEncoderOutput
from kcorrdiff.training.edm_loss import edm_preconditioning


TARGET = (8, 16, 24, 32, 40)
CONTEXT = (8, 12, 16, 24, 32)
SIZE = 16
SIGNATURE = "era5_oracle:era=1:tp=1:full_trajectory"


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


def _geometry(height: int, width: int) -> PhysicalTokenGeometry:
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


def _pyramid(
    widths: tuple[int, ...],
    *,
    shift: float = 0.0,
    requires_grad: bool = False,
) -> FeaturePyramid:
    levels = []
    size = SIZE
    for width in widths:
        value = torch.randn(1, width, size, size) + shift
        value.requires_grad_(requires_grad)
        levels.append(value)
        size //= 2
    return FeaturePyramid(*levels, widths=widths)


def _validity(*, present: bool = True) -> ValidityPyramid:
    levels = []
    size = SIZE
    for _ in range(5):
        levels.append(
            torch.full((1, 1, size, size), float(present), dtype=torch.float32)
        )
        size //= 2
    return ValidityPyramid(*levels)


def _condition_bank(
    *,
    target_shift: float = 0.0,
    context_shift: float = 0.0,
    context_present: bool = True,
    requires_grad: bool = False,
    static: torch.Tensor | None = None,
) -> ConditionBankCache:
    target_features = _pyramid(
        TARGET, shift=target_shift, requires_grad=requires_grad
    )
    context_features = _pyramid(
        CONTEXT, shift=context_shift, requires_grad=requires_grad
    )
    temporal = torch.randn(1, TARGET[0] // 2, SIZE, SIZE)
    temporal.requires_grad_(requires_grad)
    if static is None:
        static = torch.randn(1, TARGET[0] // 2, SIZE, SIZE)
        static.requires_grad_(requires_grad)
    context_temporal = torch.randn(1, CONTEXT[0] // 2, SIZE, SIZE)
    context_temporal.requires_grad_(requires_grad)
    context_static = torch.randn(1, CONTEXT[0] // 2, SIZE, SIZE)
    context_static.requires_grad_(requires_grad)
    return ConditionBankCache(
        target=TargetEncoderOutput(
            features=target_features,
            validity=_validity(),
            temporal_level_zero=temporal,
            static_level_zero=static,
        ),
        context=ContextEncoderOutput(
            features=context_features,
            validity=_validity(present=context_present),
            temporal_level_zero=context_temporal,
            static_level_zero=context_static,
        ),
        input_size=SIZE,
        target_widths=TARGET,
        context_widths=CONTEXT,
    )


def _conditions(
    *,
    bank: ConditionBankCache | None = None,
    context_present: bool = True,
    era_present: bool = True,
    era_shift: float = 0.0,
    requires_grad: bool = False,
) -> ResidualEDMConditions:
    selected_bank = bank or _condition_bank(
        context_present=context_present, requires_grad=requires_grad
    )
    era_features = torch.randn(1, 128, 33, 33) + era_shift
    era_features.requires_grad_(requires_grad)
    era = EraQueryResult(
        features=era_features,
        temporal_weights=torch.full((1, 8, 8), 1.0 / 8.0),
        valid_token_mask=torch.ones(1, 8, dtype=torch.bool),
        tp_token_mask=torch.ones(1, 8, dtype=torch.bool),
        used_source_null=torch.tensor([not era_present]),
        used_masked_null=torch.zeros(1, dtype=torch.bool),
    )
    advection = torch.randn(1, 8, SIZE, SIZE)
    advection[:, 6:7].sigmoid_()
    advection.requires_grad_(requires_grad)
    mu_z = torch.randn(1, 1, SIZE, SIZE)
    mu_z.requires_grad_(requires_grad)
    probability = torch.rand(1, 1, SIZE, SIZE)
    probability.requires_grad_(requires_grad)
    e_cond = torch.randn(1, 512)
    e_cond.requires_grad_(requires_grad)
    return ResidualEDMConditions(
        condition_bank=selected_bank,
        era=era,
        advection_features=advection,
        mu_z=mu_z,
        probability_wet=probability,
        e_cond=e_cond,
        geometry=RegressionGeometry(
            target_l3=_geometry(2, 2),
            target_l4=_geometry(1, 1),
            context_l3=_geometry(2, 2),
            context_l4=_geometry(1, 1),
            era_native=_geometry(33, 33),
        ),
        condition_signatures=(SIGNATURE,),
        lead_indices=torch.tensor([2], dtype=torch.int64),
    )


def _config(
    variant: str = "edm_b", *, activation_checkpoint: bool = False
) -> ResidualEDMConfig:
    return ResidualEDMConfig(
        variant=variant,  # type: ignore[arg-type]
        target_widths=TARGET,
        context_widths=CONTEXT,
        input_size=SIZE,
        query_chunk_size=2,
        activation_checkpoint=activation_checkpoint,
        allow_test_override=True,
    )


def _live_source_gates(model: ResidualEDM, value: float = 0.2) -> None:
    with torch.no_grad():
        for module in (
            model.context_attention_l3,
            model.era_attention_l3,
            model.context_attention_l4,
            model.era_attention_l4,
        ):
            module.gate_projection.bias.fill_(value)


def test_condition_schema_and_target_grid_state_are_fail_closed() -> None:
    names = {field.name for field in fields(ResidualEDMConditions)}
    assert names == {
        "condition_bank",
        "era",
        "advection_features",
        "mu_z",
        "probability_wet",
        "e_cond",
        "geometry",
        "condition_signatures",
        "lead_indices",
    }
    assert not names & {
        "m_tau",
        "m_target",
        "target_validity",
        "future_target",
        "z_tau",
    }
    model = ResidualEDM(_config())
    conditions = _conditions()
    with pytest.raises(ValueError, match="shape"):
        model.denoise(
            torch.randn(1, 2, SIZE, SIZE),
            torch.ones(1),
            conditions,
        )
    with pytest.raises(TypeError):
        model.denoise(  # type: ignore[call-arg]
            torch.randn(1, 1, SIZE, SIZE),
            torch.ones(1),
            conditions,
            target_validity=torch.ones(1, 1, SIZE, SIZE, dtype=torch.bool),
        )


@pytest.mark.parametrize("variant", ["edm_a", "edm_b"])
def test_denoise_shape_fp32_and_exact_edm_preconditioning(variant: str) -> None:
    torch.manual_seed(601)
    model = ResidualEDM(_config(variant)).eval()
    conditions = _conditions()
    noisy = torch.randn(1, 1, SIZE, SIZE)
    sigma = torch.tensor([2.0], dtype=torch.float32)
    captured_c_noise: list[torch.Tensor] = []

    def capture_noise(
        _module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
    ) -> None:
        captured_c_noise.append(arguments[0].detach().clone())

    handle = model.c_noise_adapter.register_forward_pre_hook(capture_noise)
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.bias.fill_(2.0)
        source_cache = model.prepare_source_cache(conditions)
        output = model.denoise(noisy, sigma, conditions, source_cache=source_cache)
    handle.remove()
    coefficients = edm_preconditioning(sigma)
    torch.testing.assert_close(
        output,
        coefficients.c_skip * noisy + coefficients.c_out * 2.0,
    )
    torch.testing.assert_close(captured_c_noise[0], coefficients.c_noise.flatten(1))
    assert output.shape == (1, 1, SIZE, SIZE)
    assert output.dtype is torch.float32


def test_edm_a_excludes_deployment_pyramid_and_context_but_keeps_static() -> None:
    torch.manual_seed(602)
    static = torch.randn(1, TARGET[0] // 2, SIZE, SIZE)
    first_bank = _condition_bank(static=static)
    second_bank = _condition_bank(
        target_shift=10_000.0,
        context_shift=-10_000.0,
        static=static,
    )
    first = _conditions(bank=first_bank)
    second = replace(
        first,
        condition_bank=second_bank,
        # ERA is held fixed so only deployment target/context features change.
    )
    model = ResidualEDM(_config("edm_a")).eval()
    noisy = torch.randn(1, 1, SIZE, SIZE)
    sigma = torch.tensor([0.7])
    with torch.no_grad():
        first_cache = model.prepare_source_cache(first)
        second_cache = model.prepare_source_cache(second)
        first_output = model.denoise(noisy, sigma, first, source_cache=first_cache)
        second_output = model.denoise(noisy, sigma, second, source_cache=second_cache)
    assert first_cache.context_l3.is_exactly_absent
    assert first_cache.context_l4.is_exactly_absent
    torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)


def test_edm_b_uses_detached_deployment_pyramid_and_separate_source_qkv() -> None:
    torch.manual_seed(603)
    model = ResidualEDM(_config("edm_b")).train()
    _live_source_gates(model)
    conditions = _conditions(requires_grad=True)
    noisy = torch.randn(1, 1, SIZE, SIZE, requires_grad=True)
    sigma = torch.tensor([0.8], requires_grad=True)
    source_cache = model.prepare_source_cache(conditions)
    output = model.denoise(noisy, sigma, conditions, source_cache=source_cache)
    output.square().mean().backward()

    assert noisy.grad is not None and torch.count_nonzero(noisy.grad) > 0
    assert sigma.grad is not None and torch.count_nonzero(sigma.grad) > 0
    assert model.sigma_embedding.input_projection.weight.grad is not None
    modules = (
        model.context_attention_l3,
        model.era_attention_l3,
        model.context_attention_l4,
        model.era_attention_l4,
    )
    for module in modules:
        assert module.query_projection.weight.grad is not None
        assert module.key_projection.weight.grad is not None
        assert module.value_projection.weight.grad is not None
        assert module.gate_projection.weight.grad is not None
    # Context L4 has one source token in the 16-pixel test model, so its
    # softmax is identically one and Q/K correctly receive zero gradients.
    for module in (modules[0], modules[1], modules[3]):
        assert torch.count_nonzero(module.query_projection.weight.grad) > 0
        assert torch.count_nonzero(module.key_projection.weight.grad) > 0
        assert torch.count_nonzero(module.value_projection.weight.grad) > 0

    # All deployment/regression/advection conditions are frozen at this model
    # boundary even if a caller hands in tensors that require gradients.
    frozen_inputs = [
        *conditions.condition_bank.target.features.levels,
        *conditions.condition_bank.context.features.levels,
        conditions.condition_bank.target.static_level_zero,
        conditions.era.features,
        conditions.advection_features,
        conditions.mu_z,
        conditions.probability_wet,
        conditions.e_cond,
    ]
    assert all(value.grad is None for value in frozen_inputs)
    assert model.context_attention_l3.key_projection is not model.era_attention_l3.key_projection
    assert model.context_attention_l3.key_projection is not model.context_attention_l4.key_projection


def test_source_kv_cache_is_sigma_free_and_reused_across_noise_levels() -> None:
    torch.manual_seed(604)
    model = ResidualEDM(_config()).eval()
    _live_source_gates(model)
    conditions = _conditions()
    calls = 0

    def counted(*_args: object) -> None:
        nonlocal calls
        calls += 1

    handles = []
    for module in (
        model.context_attention_l3,
        model.era_attention_l3,
        model.context_attention_l4,
        model.era_attention_l4,
    ):
        handles.extend(
            (
                module.key_projection.register_forward_hook(counted),
                module.value_projection.register_forward_hook(counted),
            )
        )
    with torch.no_grad():
        source_cache = model.prepare_source_cache(conditions)
        assert all("sigma" not in field.name for field in fields(source_cache))
        after_prepare = calls
        noisy = torch.randn(1, 1, SIZE, SIZE)
        low = model.denoise(
            noisy,
            torch.tensor([0.1]),
            conditions,
            source_cache=source_cache,
        )
        high = model.denoise(
            noisy,
            torch.tensor([3.0]),
            conditions,
            source_cache=source_cache,
        )
    for handle in handles:
        handle.remove()
    assert after_prepare == 8  # K and V for four independent source/scale modules.
    assert calls == after_prepare
    assert not torch.allclose(low, high)

    wrong_lead = replace(conditions, lead_indices=torch.tensor([3]))
    with pytest.raises(ValueError, match="lead mismatch"):
        model.denoise(
            torch.zeros(1, 1, SIZE, SIZE),
            torch.ones(1),
            wrong_lead,
            source_cache=source_cache,
        )


def test_all_absent_sources_skip_qkv_and_are_exactly_inert() -> None:
    torch.manual_seed(605)
    bank = _condition_bank(context_present=False)
    conditions = _conditions(bank=bank, era_present=False)
    model = ResidualEDM(_config()).eval()
    _live_source_gates(model, value=1.0)
    calls = 0

    def counted(*_args: object) -> None:
        nonlocal calls
        calls += 1

    handles = []
    for module in (
        model.context_attention_l3,
        model.era_attention_l3,
        model.context_attention_l4,
        model.era_attention_l4,
    ):
        handles.extend(
            (
                module.key_projection.register_forward_hook(counted),
                module.value_projection.register_forward_hook(counted),
            )
        )
    with torch.no_grad():
        cache = model.prepare_source_cache(conditions)
    for handle in handles:
        handle.remove()
    assert calls == 0
    assert all(
        item.is_exactly_absent
        for item in (cache.context_l3, cache.era_l3, cache.context_l4, cache.era_l4)
    )

    changed_context = _condition_bank(
        context_shift=1_000_000.0,
        context_present=False,
    )
    changed = replace(
        conditions,
        condition_bank=changed_context,
        era=replace(
            conditions.era,
            features=conditions.era.features + 1_000_000.0,
        ),
    )
    noisy = torch.randn(1, 1, SIZE, SIZE)
    sigma = torch.tensor([1.0])
    with torch.no_grad():
        reference = model.denoise(noisy, sigma, conditions, source_cache=cache)
    # B still consumes the target deployment pyramid, so only compare a change
    # confined to absent context and ERA source tensors.
    changed_same_target = replace(changed, condition_bank=replace(
        changed.condition_bank,
        target=conditions.condition_bank.target,
    ))
    with torch.no_grad():
        changed_same_cache = model.prepare_source_cache(changed_same_target)
        altered = model.denoise(
            noisy, sigma, changed_same_target, source_cache=changed_same_cache
        )
    torch.testing.assert_close(reference, altered, rtol=0.0, atol=0.0)


def test_non_reentrant_checkpointed_model_matches_plain_forward_backward() -> None:
    torch.manual_seed(606)
    plain = ResidualEDM(_config(activation_checkpoint=False)).train()
    _live_source_gates(plain)
    checkpointed = ResidualEDM(_config(activation_checkpoint=True)).train()
    checkpointed.load_state_dict(plain.state_dict())
    conditions = _conditions()
    plain_noisy = torch.randn(1, 1, SIZE, SIZE, requires_grad=True)
    checked_noisy = plain_noisy.detach().clone().requires_grad_(True)
    sigma = torch.tensor([0.9])

    plain_cache = plain.prepare_source_cache(conditions)
    checked_cache = checkpointed.prepare_source_cache(conditions)
    plain_output = plain.denoise(
        plain_noisy, sigma, conditions, source_cache=plain_cache
    )
    checked_output = checkpointed.denoise(
        checked_noisy, sigma, conditions, source_cache=checked_cache
    )
    torch.testing.assert_close(checked_output, plain_output, rtol=2e-5, atol=2e-6)
    plain_output.square().mean().backward()
    checked_output.square().mean().backward()
    torch.testing.assert_close(
        checked_noisy.grad, plain_noisy.grad, rtol=3e-5, atol=3e-6
    )
    for (plain_name, plain_parameter), (checked_name, checked_parameter) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert plain_name == checked_name
        assert plain_parameter.grad is not None
        assert checked_parameter.grad is not None
        torch.testing.assert_close(
            checked_parameter.grad,
            plain_parameter.grad,
            rtol=5e-5,
            atol=5e-6,
        )
    assert checkpointed.context_attention_l3.activation_checkpoint is True
    assert checkpointed.era_attention_l4.activation_checkpoint is True


def test_strict_fp32_sigma_data_and_fallback_contracts() -> None:
    with pytest.raises(ValueError, match="fallback"):
        ResidualEDMConfig(
            target_widths=TARGET,
            context_widths=CONTEXT,
            input_size=SIZE,
        )
    with pytest.raises(ValueError, match="sigma_data"):
        ResidualEDMConfig(sigma_data=0.5)
    model = ResidualEDM(_config())
    conditions = _conditions()
    with pytest.raises(TypeError, match="float32"):
        model.denoise(
            torch.randn(1, 1, SIZE, SIZE),
            torch.ones(1, dtype=torch.float64),
            conditions,
        )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            model.denoise(
                torch.randn(1, 1, SIZE, SIZE), torch.ones(1), conditions
            )
    downcast = ResidualEDM(_config()).double()
    with pytest.raises(TypeError, match="must remain float32"):
        downcast.denoise(
            torch.randn(1, 1, SIZE, SIZE), torch.ones(1), conditions
        )
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(RuntimeError, match="TF32|highest"):
            model.denoise(
                torch.randn(1, 1, SIZE, SIZE), torch.ones(1), conditions
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def test_default_production_parameter_goldens_are_exact() -> None:
    edm_a = ResidualEDM(ResidualEDMConfig(variant="edm_a"))
    assert edm_a.config.target_widths == (64, 128, 256, 384, 512)
    assert edm_a.config.input_size == 256
    assert edm_a.parameter_count == EDM_A_PRODUCTION_PARAMETER_COUNT
    assert edm_a.parameter_count * 4 == EDM_A_PRODUCTION_PARAMETER_COUNT * 4
    del edm_a
    gc.collect()

    edm_b = ResidualEDM(ResidualEDMConfig(variant="edm_b"))
    assert edm_b.parameter_count == EDM_B_PRODUCTION_PARAMETER_COUNT
    assert edm_b.parameter_count * 4 == EDM_B_PRODUCTION_PARAMETER_COUNT * 4
    assert EDM_B_PRODUCTION_PARAMETER_COUNT > EDM_A_PRODUCTION_PARAMETER_COUNT
    del edm_b
    gc.collect()
