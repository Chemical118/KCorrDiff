from __future__ import annotations

import copy

import pytest
import torch

from kcorrdiff.models.era_encoder import (
    ERA_NATIVE_HOURS,
    ERA_NATIVE_SHAPE,
    ERA_OUTPUT_CHANNELS,
    EraEncoder,
)


CONDITION_DIM = 12


def _inputs(
    *, batch: int = 2, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(104)
    instantaneous = torch.randn(
        batch, 8, 23, 33, 33, generator=generator, dtype=torch.float32
    ).to(device)
    precipitation = torch.randn(
        batch, 8, 1, 33, 33, generator=generator, dtype=torch.float32
    ).to(device)
    delta = torch.arange(8, dtype=torch.float32).sub_(0.25).repeat(batch, 1)
    return {
        "instantaneous": instantaneous,
        "precipitation": precipitation,
        "delta_hours": delta.to(device),
        "data_valid_inst": torch.ones(batch, 8, dtype=torch.bool, device=device),
        "tp_valid": torch.ones(batch, 8, dtype=torch.bool, device=device),
        "trajectory_window_mask": torch.ones(
            batch, 8, dtype=torch.bool, device=device
        ),
        "temporal_access_mask": torch.ones(
            batch, 8, dtype=torch.bool, device=device
        ),
        "era_present": torch.ones(batch, dtype=torch.bool, device=device),
        "tp_present": torch.ones(batch, dtype=torch.bool, device=device),
        "e_cond": torch.randn(
            batch, CONDITION_DIM, generator=generator, dtype=torch.float32
        ).to(device),
    }


def _model(*, device: torch.device | str = "cpu") -> EraEncoder:
    return EraEncoder(
        condition_dim=CONDITION_DIM,
        stem_channels=8,
        spatial_blocks=0,
    ).to(device).eval()


def _encode(model: EraEncoder, values: dict[str, torch.Tensor]):
    return model.encode_frames(
        values["instantaneous"],
        values["precipitation"],
        delta_hours=values["delta_hours"],
        data_valid_inst=values["data_valid_inst"],
        tp_valid=values["tp_valid"],
        trajectory_window_mask=values["trajectory_window_mask"],
        era_present=values["era_present"],
        tp_present=values["tp_present"],
    )


def test_native_grid_fp32_shape_and_interval_centres() -> None:
    model = _model()
    values = _inputs()
    cache = _encode(model, values)
    result = model.query(
        cache,
        temporal_access_mask=values["temporal_access_mask"],
        e_cond=values["e_cond"],
    )

    assert cache.instantaneous.shape == (2, 8, 8, 33, 33)
    assert cache.precipitation.shape == (2, 8, 8, 33, 33)
    assert cache.encoded.shape == (2, 8, ERA_OUTPUT_CHANNELS, 33, 33)
    assert result.features.shape == (2, ERA_OUTPUT_CHANNELS, 33, 33)
    assert result.features.dtype is torch.float32
    assert result.temporal_weights.shape == (2, model.temporal_heads, 8)
    torch.testing.assert_close(
        cache.tp_interval_center_delta_hours,
        values["delta_hours"] - 0.5,
    )
    torch.testing.assert_close(
        result.temporal_weights.sum(dim=-1),
        torch.ones(2, model.temporal_heads),
    )
    assert ERA_NATIVE_HOURS == 8
    assert ERA_NATIVE_SHAPE == (33, 33)

    wrong_grid = dict(values)
    wrong_grid["instantaneous"] = values["instantaneous"][..., :17, :17]
    wrong_grid["precipitation"] = values["precipitation"][..., :17, :17]
    with pytest.raises(ValueError, match="33"):
        _encode(model, wrong_grid)
    double_values = dict(values)
    double_values["instantaneous"] = values["instantaneous"].double()
    with pytest.raises(TypeError, match="float32"):
        _encode(model, double_values)


def test_masks_remain_separate_and_all_masked_states_are_defined() -> None:
    model = _model()
    values = _inputs()
    values["data_valid_inst"] = torch.tensor(
        [[True, True, False, True, True, True, True, True]] * 2
    )
    values["tp_valid"] = torch.tensor(
        [[True, False, True, True, True, True, True, True]] * 2
    )
    values["trajectory_window_mask"] = torch.tensor(
        [[True, True, True, False, True, True, True, True]] * 2
    )
    values["temporal_access_mask"] = torch.tensor(
        [
            [True, True, True, True, False, True, True, True],
            [False, False, False, False, False, False, False, False],
        ]
    )
    values["era_present"] = torch.tensor([False, True])
    values["tp_present"] = torch.tensor([False, True])
    cache = _encode(model, values)
    result = model.query(
        cache,
        temporal_access_mask=values["temporal_access_mask"],
        e_cond=values["e_cond"],
    )

    expected_inst = (
        values["data_valid_inst"]
        & values["trajectory_window_mask"]
        & values["temporal_access_mask"]
        & values["era_present"][:, None]
    )
    expected_tp = (
        values["tp_valid"]
        & values["trajectory_window_mask"]
        & values["temporal_access_mask"]
        & values["tp_present"][:, None]
        & values["era_present"][:, None]
    )
    assert torch.equal(result.valid_token_mask, expected_inst)
    assert torch.equal(result.tp_token_mask, expected_tp)
    assert not result.temporal_weights.any()
    assert result.used_source_null.tolist() == [True, False]
    assert result.used_masked_null.tolist() == [False, True]
    torch.testing.assert_close(
        result.features[0], model.source_null_state[0].expand(-1, 33, 33)
    )
    torch.testing.assert_close(
        result.features[1], model.masked_null_state[0].expand(-1, 33, 33)
    )
    assert not torch.equal(result.features[0], result.features[1])
    assert torch.isfinite(result.features).all()


def test_tp_dropout_uses_null_and_does_not_reencode_or_read_tp_values() -> None:
    model = _model()
    values = _inputs(batch=1)
    values["tp_present"] = torch.tensor([False])
    changed = dict(values)
    changed["precipitation"] = values["precipitation"] + 10_000.0
    cache_a = _encode(model, values)
    cache_b = _encode(model, changed)
    torch.testing.assert_close(cache_a.instantaneous, cache_b.instantaneous)
    torch.testing.assert_close(cache_a.encoded, cache_b.encoded)
    assert torch.equal(cache_a.tp_null, model.tp_null_state.expand_as(cache_a.tp_null))

    enabled = dict(values)
    enabled["tp_present"] = torch.tensor([True])
    enabled["precipitation"] = torch.zeros_like(values["precipitation"])
    cache_enabled = _encode(model, enabled)
    assert not torch.allclose(cache_a.encoded, cache_enabled.encoded)

    invalid_a = dict(enabled)
    invalid_a["tp_valid"] = torch.tensor(
        [[False, True, True, True, True, True, True, True]]
    )
    invalid_b = dict(invalid_a)
    invalid_b["precipitation"] = invalid_a["precipitation"].clone()
    invalid_b["precipitation"][:, 0] = 99_999.0
    cache_invalid_a = _encode(model, invalid_a)
    cache_invalid_b = _encode(model, invalid_b)
    torch.testing.assert_close(
        cache_invalid_a.encoded[:, 0], cache_invalid_b.encoded[:, 0]
    )


def test_frame_cache_is_lead_independent_and_query_is_conditioned() -> None:
    model = _model()
    values = _inputs(batch=1)
    projection_calls = 0

    def count_projection(*_args: object) -> None:
        nonlocal projection_calls
        projection_calls += 1

    handle = model.instantaneous_projection.register_forward_hook(count_projection)
    cache = _encode(model, values)
    first = model.query(
        cache,
        temporal_access_mask=values["temporal_access_mask"],
        e_cond=values["e_cond"],
    )
    later_condition = values["e_cond"] + 3.0
    later_access = values["temporal_access_mask"].clone()
    later_access[:, :3] = False
    later = model.query(
        cache, temporal_access_mask=later_access, e_cond=later_condition
    )
    handle.remove()

    assert projection_calls == 1
    assert cache.encoded.shape[-2:] == (33, 33)
    assert not torch.allclose(first.temporal_weights, later.temporal_weights)
    assert not torch.allclose(first.features, later.features)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_parity() -> None:
    cpu_model = _model()
    cuda_model = copy.deepcopy(cpu_model).cuda()
    cpu_values = _inputs(batch=1)
    cuda_values = {name: value.cuda() for name, value in cpu_values.items()}
    with torch.no_grad():
        cpu_result = cpu_model(**cpu_values)
        cuda_result = cuda_model(**cuda_values)
    torch.testing.assert_close(
        cpu_result.features,
        cuda_result.features.cpu(),
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        cpu_result.temporal_weights,
        cuda_result.temporal_weights.cpu(),
        rtol=2e-4,
        atol=2e-4,
    )
