from __future__ import annotations

import gc

import pytest
import torch

import kcorrdiff.models.regression_system as regression_system_module
from kcorrdiff.data.advection import DenseFlow, build_causal_advection
from kcorrdiff.data.time_features import OFFICIAL_LEAD_HOURS
from kcorrdiff.models.common import configure_strict_fp32_runtime
from kcorrdiff.models.regression_system import (
    DirectPhysicalRegressionSystem,
    RegressionSystem,
    RegressionSystemConfig,
)
from kcorrdiff.training.batch import TrainingBatch
from kcorrdiff.training.losses import hurdle_regression_loss

from tests.training.test_batch import make_batch


TEST_TARGET_WIDTHS = (4, 8, 12, 16, 20)
TEST_CONTEXT_WIDTHS = (4, 8, 12, 16, 20)


@pytest.fixture(autouse=True)
def strict_fp32_runtime():
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


def small_config() -> RegressionSystemConfig:
    return RegressionSystemConfig(
        target_widths=TEST_TARGET_WIDTHS,
        context_widths=TEST_CONTEXT_WIDTHS,
        input_size=16,
        era_stem_channels=8,
        era_spatial_blocks=0,
        regression_query_chunk_size=2,
        allow_test_override=True,
    )


def zero_flow(batch, config: RegressionSystemConfig) -> DenseFlow:
    causal = batch.model.advection.causal
    batch_size, _, height, width = causal.context_rate_mm_per_hour.shape
    scalar = torch.ones(
        batch_size,
        1,
        height,
        width,
        dtype=torch.float32,
        device=causal.context_rate_mm_per_hour.device,
    )
    vector = torch.zeros(
        batch_size,
        2,
        height,
        width,
        dtype=torch.float32,
        device=causal.context_rate_mm_per_hour.device,
    )
    return DenseFlow(
        velocity_km_per_hour=vector,
        backward_velocity_km_per_hour=vector.clone(),
        confidence=scalar,
        forward_backward_error_km=torch.zeros_like(scalar),
        valid_mask=torch.ones_like(scalar, dtype=torch.bool),
        config_hash=config.flow_config.config_hash,
    )


def test_full_forward_backward_returns_occurrence_mu_and_causal_artifacts() -> None:
    torch.manual_seed(410)
    batch = make_batch(leads=(1.0,))
    model = RegressionSystem(small_config()).train()
    flow = zero_flow(batch, model.config)

    output = model(batch.model, flow_override=flow)

    assert output.occurrence_logits.shape == (1, 1, 16, 16)
    assert output.probability_wet.shape == output.occurrence_logits.shape
    assert output.wet_amount.shape == output.occurrence_logits.shape
    assert output.mu_z is output.regression.transformed_mean
    torch.testing.assert_close(
        output.mu_z, output.probability_wet * output.wet_amount
    )
    assert output.e_cond.shape == (1, 512)
    assert output.condition_cache.batch_size == 1
    assert output.era_frame_cache.encoded.shape == (1, 8, 128, 33, 33)
    assert output.era_query.features.shape == (1, 128, 33, 33)
    assert output.advection.features.shape == (1, 8, 16, 16)
    assert output.advection.provenance.used_future_observations is False
    assert output.advection.provenance.full_trajectory_cached is False
    assert output.advection.provenance.issue_times_utc == (
        batch.model.provenance.t0_utc[0],
    )

    loss = hurdle_regression_loss(
        occurrence_logits=output.occurrence_logits,
        wet_amount=output.wet_amount,
        **batch.labels.hurdle_kwargs(),
        global_step=2_000,
    )
    loss.total.backward()
    assert model.regression.occurrence_head.weight.grad is not None
    assert torch.count_nonzero(model.regression.occurrence_head.weight.grad) > 0
    embedding_parameter = next(model.condition_bank.condition_embedding.parameters())
    assert embedding_parameter.grad is not None
    assert torch.isfinite(embedding_parameter.grad).all()


def test_one_central_condition_embedding_one_era_frame_cache_and_requested_lead() -> None:
    batch = make_batch(leads=(1.5,))
    model = RegressionSystem(small_config()).eval()
    flow = zero_flow(batch, model.config)
    embedding_calls = 0
    era_projection_calls = 0

    def count_embedding(*_args: object) -> None:
        nonlocal embedding_calls
        embedding_calls += 1

    def count_era(*_args: object) -> None:
        nonlocal era_projection_calls
        era_projection_calls += 1

    embedding_handle = model.condition_bank.condition_embedding.register_forward_hook(
        count_embedding
    )
    era_handle = model.era_encoder.instantaneous_projection.register_forward_hook(
        count_era
    )
    with torch.no_grad():
        output = model(batch.model, flow_override=flow)
        direct = build_causal_advection(
            batch.model.advection.causal,
            batch.model.advection.geometry,
            config=model.config.flow_config,
            flow=flow,
            materialize_trajectory=False,
        )
    embedding_handle.remove()
    era_handle.remove()

    assert embedding_calls == 1
    assert era_projection_calls == 1
    assert output.advection.lead_indices.tolist() == [2]
    torch.testing.assert_close(
        output.advection.features, direct.leads.as_model_tensor()[:, 2]
    )
    assert direct.trajectory is None
    assert output.condition_signatures == (
        "era5_oracle:era=1:tp=1:full_trajectory",
    )


def test_issue_time_cache_reuses_shared_work_for_all_twelve_leads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(411)
    batches = tuple(make_batch(leads=(lead,)) for lead in OFFICIAL_LEAD_HOURS)
    model = RegressionSystem(small_config()).eval()

    with torch.inference_mode():
        expected = tuple(
            model(batch.model, flow_override=zero_flow(batch, model.config))
            for batch in batches
        )

    calls = {
        "target": 0,
        "context": 0,
        "era_frames": 0,
        "advection": 0,
        "embedding": 0,
        "regression": 0,
    }

    def increment(name: str):
        def hook(*_args: object) -> None:
            calls[name] += 1

        return hook

    handles = (
        model.condition_bank.target_encoder.register_forward_hook(increment("target")),
        model.condition_bank.context_encoder.register_forward_hook(increment("context")),
        model.era_encoder.instantaneous_projection.register_forward_hook(
            increment("era_frames")
        ),
        model.condition_bank.condition_embedding.register_forward_hook(
            increment("embedding")
        ),
        model.regression.register_forward_hook(increment("regression")),
    )
    original_advection = regression_system_module.build_causal_advection

    def counted_advection(*args, **kwargs):
        calls["advection"] += 1
        return original_advection(*args, **kwargs)

    monkeypatch.setattr(
        regression_system_module, "build_causal_advection", counted_advection
    )
    with torch.inference_mode():
        cache = model._prepare_issue_time_cache(
            batches[0].model,
            flow_override=zero_flow(batches[0], model.config),
        )
        actual = tuple(
            model._forward_from_issue_time_cache(batch.model, cache)
            for batch in batches
        )
    for handle in handles:
        handle.remove()

    assert calls == {
        "target": 1,
        "context": 1,
        "era_frames": 1,
        "advection": 1,
        "embedding": 12,
        "regression": 12,
    }
    assert cache.advection_features_all_leads.shape == (1, 12, 8, 16, 16)
    assert cache.advection_provenance.full_trajectory_cached is False
    for cached, uncached in zip(actual, expected, strict=True):
        assert cached.condition_cache is cache.condition_bank
        assert cached.era_frame_cache is cache.era_frames
        torch.testing.assert_close(cached.e_cond, uncached.e_cond, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            cached.advection.features,
            uncached.advection.features,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            cached.era_query.features,
            uncached.era_query.features,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(cached.mu_z, uncached.mu_z, rtol=0.0, atol=0.0)


def test_issue_time_cache_is_inference_only_and_bound_to_exact_inputs() -> None:
    reference = make_batch(leads=(0.5,))
    later = make_batch(leads=(1.0,))
    model = RegressionSystem(small_config()).eval()

    with pytest.raises(ValueError, match="torch.inference_mode"):
        model._prepare_issue_time_cache(reference.model)
    with torch.inference_mode():
        cache = model._prepare_issue_time_cache(
            reference.model,
            flow_override=zero_flow(reference, model.config),
        )
        mutable_cache = model._prepare_issue_time_cache(
            reference.model,
            flow_override=zero_flow(reference, model.config),
        )
        other = RegressionSystem(small_config()).eval()
        with pytest.raises(ValueError, match="another regression model"):
            other._forward_from_issue_time_cache(later.model, cache)

    mutable_cache.advection_features_all_leads[:, 1].fill_(123.0)
    with torch.inference_mode(), pytest.raises(
        RuntimeError, match="advection.all_leads.*changed after preparation"
    ):
        model._forward_from_issue_time_cache(reference.model, mutable_cache)

    later.model.condition_bank.target.radar_history[0, 0, 0, 0, 0] = 0.4
    with torch.inference_mode(), pytest.raises(
        ValueError, match="target.radar_history.*changed across leads"
    ):
        model._forward_from_issue_time_cache(later.model, cache)

    model.train()
    with torch.inference_mode(), pytest.raises(ValueError, match="eval"):
        model._prepare_issue_time_cache(reference.model)

    model.eval()
    with torch.inference_mode():
        inference_tensor_batch = make_batch(leads=(0.5,))
        with pytest.raises(ValueError, match="must track mutations"):
            model._prepare_issue_time_cache(inference_tensor_batch.model)

    mutation_batch = make_batch(leads=(0.5,))
    mutated_model = RegressionSystem(small_config()).eval()
    with torch.inference_mode():
        model_cache = mutated_model._prepare_issue_time_cache(
            mutation_batch.model,
            flow_override=zero_flow(mutation_batch, mutated_model.config),
        )
    with torch.no_grad():
        next(mutated_model.regression.parameters()).add_(1.0)
    with torch.inference_mode(), pytest.raises(
        RuntimeError, match="parameter.*changed after preparation"
    ):
        mutated_model._forward_from_issue_time_cache(
            mutation_batch.model, model_cache
        )


def test_system_never_accepts_training_batch_or_loss_fields() -> None:
    batch = make_batch(leads=(0.5,))
    assert isinstance(batch, TrainingBatch)
    model = RegressionSystem(small_config())
    with pytest.raises(TypeError, match="RegressionModelBatch"):
        model(batch)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        model(batch.model, labels=batch.labels)  # type: ignore[call-arg]


def test_system_rejects_post_construction_time_embedding_mutation() -> None:
    batch = make_batch(leads=(1.0,))
    model = RegressionSystem(small_config())
    flow = zero_flow(batch, model.config)
    batch.model.embedding.verification_cyclic.zero_()

    with pytest.raises(ValueError, match="canonical FP32 t0/lead"):
        model(batch.model, flow_override=flow)


def test_no_autocast_module_dtype_and_production_flow_override_fail_closed() -> None:
    batch = make_batch(leads=(0.5,))
    model = RegressionSystem(small_config())
    flow = zero_flow(batch, model.config)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            model(batch.model, flow_override=flow)

    downcast = RegressionSystem(small_config()).double()
    with pytest.raises(TypeError, match="must remain float32"):
        downcast(batch.model, flow_override=flow)

    production = RegressionSystem()
    with pytest.raises(ValueError, match="restricted to explicit test"):
        production(batch.model, flow_override=flow)
    del production
    gc.collect()


def test_default_production_parameter_golden_and_explicit_test_override() -> None:
    model = RegressionSystem()
    assert model.config.target_widths == (64, 128, 256, 384, 512)
    assert model.config.context_widths == (32, 64, 128, 256, 384)
    assert model.config.input_size == 256
    assert model.component_parameter_counts == {
        "condition_bank": 31_537_202,
        "era_encoder": 200_000,
        "regression": 30_271_074,
    }
    assert model.parameter_count == 62_008_276
    assert model.parameter_count * 4 == 248_033_104
    del model
    gc.collect()

    with pytest.raises(ValueError, match="fallback"):
        RegressionSystemConfig(
            target_widths=TEST_TARGET_WIDTHS,
            context_widths=TEST_CONTEXT_WIDTHS,
            input_size=16,
        )
    explicit = small_config()
    assert explicit.allow_test_override is True


@pytest.mark.parametrize("statistic", ["mean", "q50"])
def test_direct_system_uses_same_causal_frontend_and_separate_physical_head(
    statistic: str,
) -> None:
    batch = make_batch(leads=(1.0,))
    config = small_config()
    model = DirectPhysicalRegressionSystem(
        statistic=statistic,  # type: ignore[arg-type]
        config=config,
    ).train()
    output = model(batch.model, flow_override=zero_flow(batch, config))
    assert output.statistic == statistic
    assert output.prediction_mm.shape == (1, 1, 16, 16)
    assert torch.all(output.prediction_mm >= 0.0)
    assert output.condition_signatures == batch.model.provenance.condition_signatures
    assert output.advection.provenance.used_future_observations is False
    output.prediction_mm.mean().backward()
    assert model.regression.physical_head.weight.grad is not None
    assert next(model.condition_bank.parameters()).grad is not None


def test_activation_checkpoint_and_query_chunk_propagate_to_every_regression_arm() -> None:
    config = RegressionSystemConfig(
        target_widths=TEST_TARGET_WIDTHS,
        context_widths=TEST_CONTEXT_WIDTHS,
        input_size=16,
        era_stem_channels=8,
        era_spatial_blocks=0,
        regression_query_chunk_size=3,
        activation_checkpoint=True,
        allow_test_override=True,
    )
    hurdle = RegressionSystem(config)
    direct = DirectPhysicalRegressionSystem(statistic="mean", config=config)
    assert hurdle.regression.activation_checkpoint is True
    assert direct.regression.backbone.activation_checkpoint is True
    for backbone in (hurdle.regression, direct.regression.backbone):
        for attention in (
            backbone.context_attention_l3,
            backbone.era_attention_l3,
            backbone.context_attention_l4,
            backbone.era_attention_l4,
        ):
            assert attention.activation_checkpoint is True
            assert attention.query_chunk_size == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_training_batch_to_cuda_and_full_forward_device_contract() -> None:
    batch = make_batch(leads=(0.5,)).to("cuda")
    later = make_batch(leads=(1.0,)).to("cuda")
    model = RegressionSystem(small_config()).cuda().eval()
    flow = zero_flow(batch, model.config)
    with torch.no_grad():
        output = model(batch.model, flow_override=flow)
    with torch.inference_mode():
        cache = model._prepare_issue_time_cache(batch.model, flow_override=flow)
        cached = model._forward_from_issue_time_cache(later.model, cache)
    assert output.mu_z.device.type == "cuda"
    assert output.e_cond.dtype is torch.float32
    assert output.advection.features.device.type == "cuda"
    assert cached.mu_z.device.type == "cuda"
    assert cache.advection_features_all_leads.device.type == "cuda"
    assert all(value.device.type == "cuda" for value in cache.physical_bias.tensors)
