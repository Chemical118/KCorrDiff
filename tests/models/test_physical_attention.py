from __future__ import annotations

import copy

import pytest
import torch

from kcorrdiff.models.physical_attention import (
    ATTENTION_DIM,
    ATTENTION_HEADS,
    PhysicalAttentionDiagnostics,
    PhysicalCrossAttention,
    PhysicalTokenGeometry,
)


TARGET_CHANNELS = 4
SOURCE_CHANNELS = 5
CONDITION_DIM = 6


def _geometry(
    height: int,
    width: int,
    *,
    x_offset: float = 0.0,
    footprint: float = 1.0,
    device: torch.device | str = "cpu",
) -> PhysicalTokenGeometry:
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    return PhysicalTokenGeometry(
        x_shared=x + x_offset,
        y_shared=y,
        footprint_width=torch.full_like(x, footprint),
        footprint_height=torch.full_like(y, footprint * 1.5),
    )


def _model(
    *,
    device: torch.device | str = "cpu",
    activation_checkpoint: bool = False,
) -> PhysicalCrossAttention:
    torch.manual_seed(205)
    return PhysicalCrossAttention(
        target_channels=TARGET_CHANNELS,
        source_channels=SOURCE_CHANNELS,
        condition_dim=CONDITION_DIM,
        source_name="era_l3",
        query_chunk_size=2,
        activation_checkpoint=activation_checkpoint,
    ).to(device).eval()


def _values(
    *, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor | PhysicalTokenGeometry]:
    generator = torch.Generator(device="cpu").manual_seed(206)
    return {
        "target": torch.randn(
            1, TARGET_CHANNELS, 2, 3, generator=generator
        ).to(device),
        "source": torch.randn(
            1, SOURCE_CHANNELS, 2, 2, generator=generator
        ).to(device),
        "query_geometry": _geometry(2, 3, device=device),
        "source_geometry": _geometry(2, 2, x_offset=0.25, device=device),
        "source_validity": torch.ones(1, 2, 2, dtype=torch.bool, device=device),
        "source_present": torch.ones(1, dtype=torch.bool, device=device),
        "e_cond": torch.ones(1, CONDITION_DIM, dtype=torch.float32, device=device),
    }


def test_zero_gate_is_identity_but_gate_receives_nonzero_gradient() -> None:
    model = _model()
    values = _values()
    diagnostics = model(**values, return_diagnostics=True)
    assert isinstance(diagnostics, PhysicalAttentionDiagnostics)
    assert model.attention_dim == ATTENTION_DIM == 256
    assert model.heads == ATTENTION_HEADS == 8
    assert torch.count_nonzero(model.gate_projection.weight) == 0
    assert torch.count_nonzero(model.gate_projection.bias) == 0
    assert torch.count_nonzero(model.output_projection.weight) > 0
    assert torch.equal(diagnostics.output, values["target"])
    assert torch.count_nonzero(diagnostics.attention_residual) > 0

    loss = (
        diagnostics.output * diagnostics.attention_residual.detach()
    ).sum()
    loss.backward()
    assert model.gate_projection.weight.grad is not None
    assert model.gate_projection.bias.grad is not None
    assert torch.count_nonzero(model.gate_projection.weight.grad) > 0
    assert torch.count_nonzero(model.gate_projection.bias.grad) > 0


def test_q_not_equal_k_chunked_matches_unchunked_and_cached_bias() -> None:
    model = _model()
    values = _values()
    with torch.no_grad():
        model.gate_projection.bias.fill_(0.4)
        chunked = model(**values, query_chunk_size=2)
        unchunked = model(**values, query_chunk_size=64)
        bias = model.compute_physical_bias(
            values["query_geometry"],
            values["source_geometry"],
            batch_size=1,
            query_shape=(2, 3),
            source_shape=(2, 2),
            chunk_size=1,
        )
        cached = model(
            **values,
            query_chunk_size=3,
            precomputed_physical_bias=bias,
        )

    assert bias.shape == (1, 8, 6, 4)  # Deliberately Q != K.
    torch.testing.assert_close(chunked, unchunked, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chunked, cached, rtol=1e-6, atol=1e-6)
    assert chunked.shape == values["target"].shape
    assert chunked.dtype is torch.float32


def test_source_permutation_with_geometry_is_equivariant_and_mask_is_exact() -> None:
    model = _model()
    values = _values()
    with torch.no_grad():
        model.gate_projection.bias.fill_(0.25)
    values["source_validity"] = torch.tensor(
        [[[True, False], [True, True]]], dtype=torch.bool
    )
    reference = model(**values)

    invalid_changed = dict(values)
    invalid_changed["source"] = values["source"].clone()
    invalid_changed["source"][:, :, 0, 1] = 1_000_000.0
    torch.testing.assert_close(model(**invalid_changed), reference)

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(values)
    permuted["source"] = (
        values["source"].flatten(2)[:, :, permutation].reshape_as(values["source"])
    )
    source_geometry = values["source_geometry"]
    assert isinstance(source_geometry, PhysicalTokenGeometry)

    def permute_field(field: torch.Tensor) -> torch.Tensor:
        return field.flatten()[permutation].reshape_as(field)

    permuted["source_geometry"] = PhysicalTokenGeometry(
        *(permute_field(field) for field in (
            source_geometry.x_shared,
            source_geometry.y_shared,
            source_geometry.footprint_width,
            source_geometry.footprint_height,
        ))
    )
    permuted["source_validity"] = (
        values["source_validity"].flatten(1)[:, permutation].reshape(1, 2, 2)
    )
    torch.testing.assert_close(model(**permuted), reference, rtol=1e-5, atol=1e-5)


def test_physical_bias_responds_to_distance_and_both_footprints() -> None:
    model = _model()
    first = model.geometry_bias[0]
    last = model.geometry_bias[2]
    with torch.no_grad():
        first.weight.zero_()
        first.bias.zero_()
        # Raw feature order: dx, dy, distance, q_width, q_height,
        # key_width, key_height.  Make the sensitivity deterministic.
        first.weight[0, 2] = 1.0
        first.weight[1, 3] = 1.0
        first.weight[2, 5] = 1.0
        last.weight.zero_()
        last.bias.zero_()
        last.weight[:, :3] = 1.0

    kwargs = {
        "batch_size": 1,
        "query_shape": (2, 3),
        "source_shape": (2, 2),
        "chunk_size": 2,
    }
    base = model.compute_physical_bias(_geometry(2, 3), _geometry(2, 2), **kwargs)
    far = model.compute_physical_bias(
        _geometry(2, 3), _geometry(2, 2, x_offset=100.0), **kwargs
    )
    wide_query = model.compute_physical_bias(
        _geometry(2, 3, footprint=2.0), _geometry(2, 2), **kwargs
    )
    wide_key = model.compute_physical_bias(
        _geometry(2, 3), _geometry(2, 2, footprint=2.0), **kwargs
    )
    assert not torch.allclose(base, far)
    assert not torch.allclose(base, wide_query)
    assert not torch.allclose(base, wide_key)


def test_all_invalid_or_absent_source_is_finite_exact_identity() -> None:
    model = _model()
    values = _values()
    with torch.no_grad():
        model.gate_projection.bias.fill_(1.0)
    values["source_validity"] = torch.zeros(1, 2, 2, dtype=torch.bool)
    invalid = model(**values, return_diagnostics=True)
    assert isinstance(invalid, PhysicalAttentionDiagnostics)
    assert torch.isfinite(invalid.output).all()
    assert not invalid.attention_residual.any()
    assert torch.equal(invalid.output, values["target"])

    values["source_validity"] = torch.ones(1, 2, 2, dtype=torch.bool)
    values["source_present"] = torch.tensor([False])
    absent = model(**values, return_diagnostics=True)
    assert isinstance(absent, PhysicalAttentionDiagnostics)
    assert torch.isfinite(absent.output).all()
    assert not absent.attention_residual.any()
    assert torch.equal(absent.output, values["target"])


def test_absent_source_skips_dense_attention_but_keeps_ddp_zero_gradients() -> None:
    model = _model().train()
    values = _values()
    values["source_present"] = torch.tensor([False])
    calls = 0

    def counted(*_args: object) -> None:
        nonlocal calls
        calls += 1

    handles = [
        module.register_forward_hook(counted)
        for module in (
            model.query_projection,
            model.key_projection,
            model.value_projection,
            model.geometry_bias,
        )
    ]
    output = model(**values)
    for handle in handles:
        handle.remove()

    assert calls == 0
    assert torch.equal(output, values["target"])
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters())


def test_checkpointed_chunks_match_plain_forward_and_parameter_gradients() -> None:
    plain = _model().train()
    with torch.no_grad():
        plain.gate_projection.bias.fill_(0.4)
    checkpointed = _model(activation_checkpoint=True).train()
    checkpointed.load_state_dict(plain.state_dict())
    values = _values()

    plain_output = plain(**values)
    checkpointed_output = checkpointed(**values)
    torch.testing.assert_close(checkpointed_output, plain_output, rtol=1e-6, atol=1e-6)

    plain_output.square().mean().backward()
    checkpointed_output.square().mean().backward()
    for (plain_name, plain_parameter), (checked_name, checked_parameter) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert plain_name == checked_name
        assert plain_parameter.grad is not None
        assert checked_parameter.grad is not None
        torch.testing.assert_close(
            checked_parameter.grad,
            plain_parameter.grad,
            rtol=2e-5,
            atol=2e-6,
        )


def test_fp32_and_geometry_contracts_fail_closed() -> None:
    model = _model()
    values = _values()
    double_values = dict(values)
    double_values["target"] = values["target"].double()
    with pytest.raises(TypeError, match="float32"):
        model(**double_values)
    bad_footprint = dict(values)
    bad_footprint["source_geometry"] = _geometry(2, 2, footprint=0.0)
    with pytest.raises(ValueError, match="positive"):
        model(**bad_footprint)
    configured = PhysicalCrossAttention(
        target_channels=4,
        source_channels=5,
        condition_dim=6,
        source_name="configured_attention",
        attention_dim=128,
        heads=4,
    )
    assert configured.attention_dim == 128 and configured.heads == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_parity() -> None:
    cpu_model = _model()
    with torch.no_grad():
        cpu_model.gate_projection.bias.fill_(0.5)
    cuda_model = copy.deepcopy(cpu_model).cuda()
    cpu_values = _values()
    cuda_values = {
        name: (
            PhysicalTokenGeometry(
                value.x_shared.cuda(),
                value.y_shared.cuda(),
                value.footprint_width.cuda(),
                value.footprint_height.cuda(),
            )
            if isinstance(value, PhysicalTokenGeometry)
            else value.cuda()
        )
        for name, value in cpu_values.items()
    }
    with torch.no_grad():
        cpu_output = cpu_model(**cpu_values)
        cuda_output = cuda_model(**cuda_values).cpu()
    torch.testing.assert_close(cpu_output, cuda_output, rtol=2e-4, atol=2e-4)
