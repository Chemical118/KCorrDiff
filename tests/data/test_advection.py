from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch

from kcorrdiff.data.accumulation import trapezoidal_accumulation_mm
from kcorrdiff.data.advection import (
    ADVECTION_MODEL_CHANNEL_NAMES,
    AdvectionGeometry,
    CausalAdvectionInput,
    DenseFlow,
    FlowConfig,
    build_causal_advection,
    causal_history_times,
    estimate_causal_flow,
    eulerian_persistence,
    lagrangian_persistence,
)


T0 = datetime(2022, 6, 1, 3, 5, tzinfo=UTC)


def geometry(
    *, target_size: int = 8, context_size: int = 16, context_start: float = -4.0
) -> AdvectionGeometry:
    return AdvectionGeometry(
        target_x_km=torch.arange(target_size, dtype=torch.float64),
        target_y_km=torch.arange(target_size, dtype=torch.float64),
        context_x_km=torch.linspace(
            context_start, context_start + context_size - 1, context_size
        ),
        context_y_km=torch.linspace(
            context_start, context_start + context_size - 1, context_size
        ),
    )


def inputs(
    target: torch.Tensor,
    context: torch.Tensor,
    *,
    target_valid: torch.Tensor | None = None,
    context_valid: torch.Tensor | None = None,
    context_confidence: torch.Tensor | None = None,
    t0: datetime = T0,
    history_times: tuple[datetime, ...] | None = None,
) -> CausalAdvectionInput:
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if context.ndim == 3:
        context = context.unsqueeze(0)
    return CausalAdvectionInput(
        t0_utc=t0,
        history_times_utc=history_times or causal_history_times(t0),
        target_rate_mm_per_hour=target,
        target_valid=(
            torch.ones_like(target, dtype=torch.bool)
            if target_valid is None
            else target_valid
        ),
        context_rate_mm_per_hour=context,
        context_valid_fraction=(
            torch.ones_like(context) if context_valid is None else context_valid
        ),
        context_interpolation_confidence=(
            torch.ones_like(context)
            if context_confidence is None
            else context_confidence
        ),
    )


def prescribed_flow(
    batch: int,
    context_shape: tuple[int, int],
    *,
    u: float = 0.0,
    v: float = 0.0,
    confidence: float = 1.0,
    valid: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> DenseFlow:
    velocity = torch.zeros(batch, 2, *context_shape, dtype=dtype, device=device)
    velocity[:, 0] = u
    velocity[:, 1] = v
    scalar = torch.full(
        (batch, 1, *context_shape), confidence, dtype=dtype, device=device
    )
    validity = torch.full(
        scalar.shape, valid, dtype=torch.bool, device=device
    )
    return DenseFlow(
        velocity_km_per_hour=velocity,
        backward_velocity_km_per_hour=-velocity,
        confidence=scalar,
        forward_backward_error_km=torch.zeros_like(scalar),
        valid_mask=validity,
        config_hash="synthetic-flow",
    )


def test_causal_input_accepts_only_exact_twelve_issue_time_frames() -> None:
    target = torch.zeros(12, 8, 8)
    context = torch.zeros(12, 16, 16)
    value = inputs(target, context)
    assert value.history_times_utc[0] == causal_history_times(T0)
    assert value.history_times_utc[0][-1] == T0
    assert not hasattr(value, "future_target")
    assert not hasattr(value, "target_validity")

    with pytest.raises(ValueError, match="exactly t0-55"):
        inputs(
            target,
            context,
            history_times=(*causal_history_times(T0)[:-1], T0 + timedelta(minutes=5)),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        inputs(target, context, t0=T0.replace(tzinfo=None))
    with pytest.raises(ValueError, match="five-minute"):
        inputs(target, context, t0=T0.replace(minute=3))


def test_robust_flow_recovers_physical_translation_sign_and_confidence() -> None:
    torch.manual_seed(13)
    size = 64
    axis = torch.arange(size, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    frames = []
    # +0.5 x and -0.25 y pixels per five minutes => +6/-3 km h-1.
    for step in range(12):
        value = 10.0 * torch.exp(
            -((x - (20.0 + 0.5 * step)) ** 2 + (y - (35.0 - 0.25 * step)) ** 2)
            / 80.0
        )
        value += 5.0 * torch.exp(
            -((x - (45.0 + 0.5 * step)) ** 2 + (y - (15.0 - 0.25 * step)) ** 2)
            / 20.0
        )
        frames.append(value)
    # One corrupted historical image exercises the 11-pair robust aggregation.
    frames[5] = torch.rand_like(frames[5]) * 5.0
    history = torch.stack(frames)
    geom = AdvectionGeometry(axis, axis, axis, axis)
    flow = estimate_causal_flow(
        inputs(history, history),
        geom,
        config=FlowConfig(pyramid_levels=3),
    )
    selected = flow.confidence[:, 0] > 0.2
    assert selected.sum() > 500
    velocity = flow.velocity_km_per_hour.permute(0, 2, 3, 1)[selected]
    assert torch.allclose(velocity.median(dim=0).values, torch.tensor([6.0, -3.0]), atol=0.35)
    assert torch.all(flow.confidence >= 0.0) and torch.all(flow.confidence <= 1.0)
    assert torch.isfinite(flow.forward_backward_error_km).all()
    assert float(flow.forward_backward_error_km.median()) < 0.35


def test_dry_history_has_zero_flow_zero_confidence_but_remains_observed() -> None:
    history = torch.zeros(12, 32, 32)
    axis = torch.arange(32, dtype=torch.float64)
    flow = estimate_causal_flow(
        inputs(history, history),
        AdvectionGeometry(axis, axis, axis, axis),
        config=FlowConfig(pyramid_levels=3),
    )
    assert not flow.velocity_km_per_hour.any()
    assert not flow.confidence.any()
    assert flow.valid_mask.all()


def test_zero_flow_constant_accumulation_and_baselines_use_shared_helper() -> None:
    target = torch.full((12, 8, 8), 2.0)
    context = torch.full((12, 16, 16), 2.0)
    value = inputs(target, context)
    flow = prescribed_flow(1, (16, 16))
    artifact = build_causal_advection(
        value,
        geometry(),
        flow=flow,
        materialize_trajectory=True,
    )
    assert artifact.trajectory is not None
    assert artifact.trajectory.rate_mm_per_hour.shape == (1, 73, 8, 8)
    assert artifact.trajectory.scan_valid.all()
    assert torch.allclose(
        artifact.leads.raw_accumulation_mm,
        torch.ones(1, 12, 8, 8),
    )
    assert artifact.leads.w_adv.all()
    assert artifact.leads.origin_in_target_mask.all()
    assert artifact.leads.origin_in_domain_mask.all()
    assert torch.all(artifact.leads.domain_residence_fraction == 1.0)
    assert torch.all(artifact.leads.context_inflow_fraction == 0.0)
    assert artifact.leads.as_model_tensor().shape == (1, 12, 8, 8, 8)
    assert artifact.leads.channel_names == ADVECTION_MODEL_CHANNEL_NAMES

    euler = eulerian_persistence(value)
    lagrangian = lagrangian_persistence(artifact)
    assert torch.equal(euler.raw_accumulation_mm, lagrangian.raw_accumulation_mm)
    assert lagrangian.raw_accumulation_mm.data_ptr() == artifact.leads.raw_accumulation_mm.data_ptr()
    assert lagrangian.flow_config_hash == flow.config_hash
    shared = trapezoidal_accumulation_mm(torch.full((7, 8, 8), 2.0))
    assert torch.equal(shared, artifact.leads.raw_accumulation_mm[0, 0])


def test_each_lead_is_the_exact_shared_seven_scan_window() -> None:
    torch.manual_seed(21)
    geom = geometry(target_size=10, context_size=20, context_start=-5.0)
    target = torch.rand(1, 12, 10, 10) * 4.0
    context = torch.rand(1, 12, 20, 20) * 4.0
    artifact = build_causal_advection(
        inputs(target, context),
        geom,
        flow=prescribed_flow(1, geom.context_shape, u=0.5, v=-0.25),
        materialize_trajectory=True,
    )
    trajectory = artifact.trajectory
    assert trajectory is not None
    for lead_index in range(12):
        start = 6 * lead_index
        window = trajectory.rate_mm_per_hour[:, start : start + 7]
        valid = trajectory.scan_valid[:, start : start + 7].all(dim=1)
        expected = trapezoidal_accumulation_mm(window, time_axis=1)
        expected = torch.where(valid, expected, torch.zeros_like(expected))
        assert torch.equal(expected, artifact.leads.raw_accumulation_mm[:, lead_index])
    assert torch.all(
        ~artifact.leads.advection_valid_mask
        | artifact.leads.origin_in_domain_mask
    )
    assert torch.all(
        ~artifact.leads.advection_valid_mask
        | (artifact.leads.domain_residence_fraction == 1.0)
    )


def test_numpy_and_torch_trapezoid_primitive_are_dtype_exact() -> None:
    values = np.arange(2 * 7 * 3, dtype=np.float32).reshape(2, 7, 3)
    numpy_result = trapezoidal_accumulation_mm(values, time_axis=1)
    torch_result = trapezoidal_accumulation_mm(torch.from_numpy(values), time_axis=1)
    assert numpy_result.dtype == np.float32
    assert torch_result.dtype == torch.float32
    assert np.array_equal(numpy_result, torch_result.numpy())
    with pytest.raises(ValueError, match="time axis"):
        trapezoidal_accumulation_mm(np.asarray(1.0, dtype=np.float32))


def test_target_priority_and_context_inflow_have_no_wrap() -> None:
    geom = geometry(target_size=8, context_size=16, context_start=-4.0)
    target = torch.ones(12, 8, 8)
    context = torch.full((12, 16, 16), 9.0)
    # Positive forward u means a future destination backtracks west.  At 12
    # km/h it moves exactly one target cell per 5-minute step.
    artifact = build_causal_advection(
        inputs(target, context),
        geom,
        flow=prescribed_flow(1, geom.context_shape, u=12.0),
        materialize_trajectory=True,
    )
    trajectory = artifact.trajectory
    assert trajectory is not None
    # k=1: target x>=1 comes from target; x=0 originates in context at x=-1.
    assert torch.all(trajectory.rate_mm_per_hour[0, 1, :, 1:] == 1.0)
    assert torch.all(trajectory.rate_mm_per_hour[0, 1, :, 0] == 9.0)
    assert trajectory.context_inflow[0, 1, :, 0].all()
    # k=5 origin x=-5 is outside context lower bound -4: neutral and invalid,
    # never wrapped to the east boundary.
    assert not trajectory.scan_valid[0, 5, :, 0].any()
    assert not trajectory.rate_mm_per_hour[0, 5, :, 0].any()
    assert torch.all(trajectory.domain_residence_fraction[0, 5, :, 0] < 1.0)
    assert not artifact.leads.advection_valid_mask[0, 0, :, 0].any()
    assert not artifact.leads.origin_in_domain_mask[0, 0, :, 0].any()


def test_invalid_target_inside_is_not_replaced_by_context() -> None:
    geom = geometry()
    target = torch.ones(1, 12, 8, 8)
    context = torch.full((1, 12, 16, 16), 9.0)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, -1, 3, 3] = False
    artifact = build_causal_advection(
        inputs(target, context, target_valid=valid),
        geom,
        flow=prescribed_flow(1, geom.context_shape),
        materialize_trajectory=True,
    )
    trajectory = artifact.trajectory
    assert trajectory is not None
    assert trajectory.origin_in_target[0, :, 3, 3].all()
    assert not trajectory.scan_valid[0, :, 3, 3].any()
    assert not trajectory.rate_mm_per_hour[0, :, 3, 3].any()
    assert not artifact.leads.advection_valid_mask[0, :, 3, 3].any()


def test_flow_qc_validity_and_domain_residence_remain_separate() -> None:
    geom = geometry()
    target = torch.ones(12, 8, 8)
    context = torch.ones(12, 16, 16)
    flow = prescribed_flow(1, geom.context_shape, valid=False, confidence=0.0)
    artifact = build_causal_advection(
        inputs(target, context), geom, flow=flow, materialize_trajectory=True
    )
    trajectory = artifact.trajectory
    assert trajectory is not None
    # Origins and geometric residence remain valid, while motion-QC makes
    # forecast frames unusable.  These must never collapse into one mask.
    assert trajectory.origin_in_context.all()
    assert torch.all(trajectory.domain_residence_fraction == 1.0)
    assert not trajectory.scan_valid[:, 1:].any()
    assert not artifact.leads.advection_valid_mask.any()
    assert artifact.leads.origin_in_domain_mask.all()
    assert torch.all(artifact.leads.domain_residence_fraction == 1.0)


def test_physical_velocity_is_preserved_from_1p2km_context_to_500m_target() -> None:
    target_axis = torch.arange(8, dtype=torch.float64) * 0.5
    context_axis = torch.arange(16, dtype=torch.float64) * 1.2 - 7.2
    geom = AdvectionGeometry(target_axis, target_axis, context_axis, context_axis)
    target = torch.ones(12, 8, 8)
    context = torch.ones(12, 16, 16)
    artifact = build_causal_advection(
        inputs(target, context),
        geom,
        flow=prescribed_flow(1, geom.context_shape, u=18.0, v=-6.0),
    )
    expected = torch.tensor([18.0, -6.0]).reshape(1, 2, 1, 1)
    assert torch.allclose(
        artifact.leads.flow_uv_target_km_per_hour,
        expected.expand_as(artifact.leads.flow_uv_target_km_per_hour),
        atol=1.0e-5,
    )


def test_streaming_and_materialized_batch_paths_are_exactly_equal() -> None:
    torch.manual_seed(7)
    geom = geometry()
    target = torch.rand(2, 12, 8, 8)
    context = torch.rand(2, 12, 16, 16)
    value = CausalAdvectionInput(
        t0_utc=(T0, T0 + timedelta(minutes=5)),
        history_times_utc=(
            causal_history_times(T0),
            causal_history_times(T0 + timedelta(minutes=5)),
        ),
        target_rate_mm_per_hour=target,
        target_valid=torch.ones_like(target, dtype=torch.bool),
        context_rate_mm_per_hour=context,
        context_valid_fraction=torch.ones_like(context),
        context_interpolation_confidence=torch.ones_like(context),
    )
    flow = prescribed_flow(2, geom.context_shape, u=1.5, v=-0.5)
    materialized = build_causal_advection(
        value, geom, flow=flow, materialize_trajectory=True
    )
    streaming = build_causal_advection(
        value, geom, flow=flow, materialize_trajectory=False
    )
    assert streaming.trajectory is None
    for name in materialized.leads.__dataclass_fields__:
        left = getattr(materialized.leads, name)
        right = getattr(streaming.leads, name)
        assert torch.equal(left, right), name
    assert materialized.provenance.used_future_observations is False
    assert materialized.provenance.full_trajectory_cached is False


def test_gradient_path_and_conditional_cuda_parity() -> None:
    geom = geometry(target_size=6, context_size=12, context_start=-3.0)
    target = torch.ones(1, 12, 6, 6, requires_grad=True)
    context = torch.ones(1, 12, 12, 12, requires_grad=True)
    value = inputs(target, context)
    velocity = torch.zeros(1, 2, 12, 12, requires_grad=True)
    scalar = torch.ones(1, 1, 12, 12)
    flow = DenseFlow(
        velocity, -velocity, scalar, torch.zeros_like(scalar), scalar.bool(), "grad"
    )
    cpu = build_causal_advection(value, geom, flow=flow, track_grad=True)
    cpu.leads.raw_accumulation_mm.sum().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert velocity.grad is not None and torch.isfinite(velocity.grad).all()

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_target = target.detach().cuda().requires_grad_()
    cuda_context = context.detach().cuda().requires_grad_()
    cuda_value = inputs(cuda_target, cuda_context)
    cuda_velocity = velocity.detach().cuda().requires_grad_()
    cuda_scalar = scalar.cuda()
    cuda_flow = DenseFlow(
        cuda_velocity,
        -cuda_velocity,
        cuda_scalar,
        torch.zeros_like(cuda_scalar),
        cuda_scalar.bool(),
        "grad",
    )
    cuda = build_causal_advection(cuda_value, geom, flow=cuda_flow, track_grad=True)
    assert torch.allclose(
        cpu.leads.raw_accumulation_mm,
        cuda.leads.raw_accumulation_mm.cpu(),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_flow_estimator_cpu_cuda_common_kernel_parity() -> None:
    size = 32
    axis = torch.arange(size, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    history = torch.stack(
        [
            10.0
            * torch.exp(
                -((x - (10.0 + 0.35 * step)) ** 2 + (y - (18.0 - 0.2 * step)) ** 2)
                / 35.0
            )
            for step in range(12)
        ]
    )
    geom = AdvectionGeometry(axis, axis, axis, axis)
    config = FlowConfig(pyramid_levels=3, warps_per_level=2)
    cpu = estimate_causal_flow(inputs(history, history), geom, config=config)
    cuda_history = history.cuda()
    cuda = estimate_causal_flow(
        inputs(cuda_history, cuda_history), geom, config=config
    )
    assert torch.allclose(
        cpu.velocity_km_per_hour,
        cuda.velocity_km_per_hour.cpu(),
        rtol=2.0e-3,
        atol=2.0e-2,
    )
    assert torch.allclose(
        cpu.confidence, cuda.confidence.cpu(), rtol=2.0e-3, atol=2.0e-3
    )
