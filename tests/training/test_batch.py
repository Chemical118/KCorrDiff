from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from kcorrdiff.data.condition_schema import (
    ContextRadarConditions,
    ContextStaticConditions,
    Era5Conditions,
    NamedSpatialFields,
    StaticConditions,
)
from kcorrdiff.data.coordinates import normalize_lcc_coordinates
from kcorrdiff.data.dataset import (
    RadarConditions,
    TrainingLabel,
    TrainingSample,
)
from kcorrdiff.data.era5_reader import Era5WindowProvenance
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES
from kcorrdiff.data.time_features import build_verification_time_features
from kcorrdiff.models.context_encoder import CONTEXT_STATIC_CHANNEL_NAMES
from kcorrdiff.models.target_encoder import TARGET_STATIC_CHANNEL_NAMES
from kcorrdiff.training.batch import (
    LossOnlyLabels,
    PhysicalGridSpec,
    RegressionModelBatch,
    TrainingBatch,
    TrainingBatchCollator,
    collate_training_samples,
)
from kcorrdiff.training.data_factory import RankLocalBatch
from kcorrdiff.training.plan import DrawSlot


SIGNATURE = "era5_oracle:era=1:tp=1:full_trajectory"


def make_grid(*, size: int = 16) -> PhysicalGridSpec:
    target_x = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * 0.5
    target_y = target_x.copy()
    context_x = (np.arange(size, dtype=np.float64) - (size - 1) / 2.0) * 1.2
    context_y = context_x.copy()
    return PhysicalGridSpec(
        target_x_lcc_km=target_x,
        target_y_lcc_km=target_y,
        context_x_lcc_km=context_x,
        context_y_lcc_km=context_y,
        era_latitude_degrees=np.linspace(32.0, 40.0, 33),
        era_longitude_degrees=np.linspace(123.0, 131.0, 33),
        allow_test_override=True,
    )


def _static(grid: PhysicalGridSpec) -> StaticConditions:
    size = grid.input_size
    center_x, center_y = grid.target_center_lcc_km
    x, y = np.meshgrid(
        grid.target_x_lcc_km, grid.target_y_lcc_km, indexing="xy"
    )
    x_shared, y_shared = normalize_lcc_coordinates(
        x,
        y,
        target_center_x_km=center_x,
        target_center_y_km=center_y,
    )
    static_values = np.stack(
        (
            np.full((size, size), 0.25),
            np.zeros((size, size)),
            np.zeros((size, size)),
            np.full((size, size), -0.1),
            np.ones((size, size)),
            x_shared,
            y_shared,
        ),
        axis=0,
    ).astype(np.float32)
    coverage = np.ones((size, size), dtype=np.bool_)
    coverage[0, 0] = False
    source_dx = np.full((size, size), 1.0, dtype=np.float32)
    source_dy = np.full((size, size), 1.1, dtype=np.float32)
    return StaticConditions(
        target=NamedSpatialFields(TARGET_STATIC_CHANNEL_NAMES, static_values),
        target_static_coverage=coverage,
        context=ContextStaticConditions(
            source_dx_km=source_dx,
            source_dy_km=source_dy,
            nearest_source_distance_km=np.full(
                (size, size), 0.2, dtype=np.float32
            ),
            geometry_confidence=np.full(
                (size, size), 0.9, dtype=np.float32
            ),
        ),
    )


def _context(*, size: int) -> ContextRadarConditions:
    shape = (12, size, size)
    mean = np.full(shape, 1.0, dtype=np.float32)
    detail = np.full(shape, 1.5, dtype=np.float32)
    return ContextRadarConditions(
        mean_rate_mm_per_hour=mean,
        detail_rate_mm_per_hour=detail,
        wet_mask=np.ones(shape, dtype=np.bool_),
        valid_fraction=np.ones(shape, dtype=np.float32),
        interpolation_confidence=np.full(shape, 0.8, dtype=np.float32),
        detail_valid=np.ones(shape, dtype=np.bool_),
        detail_mode="local_max",
    )


def _era(t0: datetime, *, lead_hours: float) -> Era5Conditions:
    valid_times = tuple(t0 + timedelta(hours=index) for index in range(8))
    delta = np.arange(8, dtype=np.float32)
    values = np.empty((8, 24, 33, 33), dtype=np.float32)
    for channel in range(24):
        values[:, channel] = np.float32(channel + 1)
    trajectory = np.asarray([True] * 7 + [False], dtype=np.bool_)
    provenance = Era5WindowProvenance(
        provider_id="era5",
        provider_version="synthetic-v1",
        dataset="synthetic",
        cache_format="synthetic-cache-v1",
        cache_manifest="/cache/era5/manifest.json",
        condition_signature=SIGNATURE,
        access_mode="full_trajectory",
        valid_times_utc=valid_times,
        tp_intervals_utc=tuple(
            (value - timedelta(hours=1), value) for value in valid_times
        ),
    )
    return Era5Conditions(
        values=values,
        valid_times_utc=valid_times,
        delta_hours=delta,
        tp_interval_center_delta_hours=delta - np.float32(0.5),
        data_valid_inst=np.ones(8, dtype=np.bool_),
        tp_valid=np.ones(8, dtype=np.bool_),
        trajectory_window_mask=trajectory,
        temporal_access_mask=trajectory.copy(),
        era_present=True,
        tp_present=True,
        condition_signature=SIGNATURE,
        lead_hours=lead_hours,
        provenance=provenance,
    )


def make_sample(
    *,
    lead_hours: float = 0.5,
    sample_id: str = "sample-0",
    grid: PhysicalGridSpec | None = None,
) -> TrainingSample:
    grid = grid or make_grid()
    size = grid.input_size
    t0 = datetime(2022, 1, 1, tzinfo=UTC)
    static = _static(grid)
    normalized = np.full((12, size, size), 1.0 / 3.0, dtype=np.float32)
    history_validity = np.broadcast_to(
        static.target_static_coverage, normalized.shape
    ).copy()
    conditions = RadarConditions(
        target_history=normalized,
        history_validity=history_validity,
        context=_context(size=size),
        era5=_era(t0, lead_hours=lead_hours),
        static=static,
        t0_utc=t0,
        lead_hours=lead_hours,
        condition_signature=SIGNATURE,
        time_features=build_verification_time_features(
            t0, lead_hours
        ).cyclic_features,
    )
    validity = static.target_static_coverage.copy()
    raw = np.where(validity, 0.2, 0.0).astype(np.float64)
    model = raw.copy()
    wet = validity.copy()
    z = np.log1p(model)
    return TrainingSample(
        sample_id=sample_id,
        block_id="event-0",
        fold_id=2,
        conditions=conditions,
        label=TrainingLabel(
            z=z,
            wet=wet,
            raw_accumulation_mm=raw,
            model_accumulation_mm=model,
            target_validity=validity,
            omega=1.25,
        ),
    )


def make_batch(*, leads: tuple[float, ...] = (0.5, 1.0)):
    grid = make_grid()
    samples = tuple(
        make_sample(lead_hours=lead, sample_id=f"sample-{index}", grid=grid)
        for index, lead in enumerate(leads)
    )
    return collate_training_samples(
        samples, geometry=grid, allow_test_override=True
    )


class _WorkerBuiltTrainingSampleDataset(Dataset[TrainingSample]):
    """Build the sample in a worker so collate output crosses the IPC queue."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> TrainingSample:
        if index != 0:
            raise IndexError(index)
        grid = make_grid()
        return make_sample(grid=grid)


def _worker_collate_training_batch(
    samples: list[TrainingSample],
):
    return collate_training_samples(
        samples,
        geometry=make_grid(),
        allow_test_override=True,
    )


def test_causal_model_and_loss_only_boundary_with_explicit_value_spaces() -> None:
    batch = make_batch()

    assert isinstance(batch.model, RegressionModelBatch)
    assert isinstance(batch.labels, LossOnlyLabels)
    assert "labels" not in {field.name for field in fields(RegressionModelBatch)}
    assert "target_validity" not in {
        field.name for field in fields(RegressionModelBatch)
    }
    assert batch.model.condition_bank.target.radar_history.shape == (
        2,
        12,
        1,
        16,
        16,
    )
    assert batch.model.condition_bank.target.radar_history.dtype is torch.float32
    assert batch.model.condition_bank.target.history_validity.dtype is torch.bool
    torch.testing.assert_close(
        batch.model.condition_bank.target.radar_history,
        torch.full((2, 12, 1, 16, 16), 1.0 / 3.0),
    )
    # CPrecNet model input stays archive-normalized, while advection receives
    # the explicit inverse mapping to physical linear rain rate.
    torch.testing.assert_close(
        batch.model.advection.causal.target_rate_mm_per_hour,
        torch.ones(2, 12, 16, 16),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert (
        batch.model.advection.causal.target_rate_mm_per_hour.data_ptr()
        != batch.model.condition_bank.target.radar_history.data_ptr()
    )
    assert batch.labels.target_z.shape == (2, 1, 16, 16)
    assert batch.labels.target_validity.dtype is torch.bool
    assert batch.labels.hurdle_kwargs().keys() == {
        "target_z",
        "target_wet",
        "target_validity",
        "omega",
    }
    assert batch.model.provenance.duplicate_condition_groups == ((0, 1),)
    assert batch.model.provenance.explicit_history_copies is True
    assert batch.model.advection.lead_indices.tolist() == [0, 1]


def test_context_encoder_and_advection_share_channels_across_device_boundary() -> None:
    batch = make_batch(leads=(0.5,))
    dynamic = batch.model.condition_bank.context.dynamic_fields
    causal = batch.model.advection.causal
    assert batch.model.advection.context_channels_alias_condition_bank is True
    for actual, channel in (
        (causal.context_rate_mm_per_hour, 0),
        (causal.context_valid_fraction, 3),
        (causal.context_interpolation_confidence, 4),
    ):
        torch.testing.assert_close(actual, dynamic[:, :, channel])
        assert (
            actual.untyped_storage().data_ptr()
            == dynamic.untyped_storage().data_ptr()
        )

    moved = batch.to("cpu", non_blocking=True)
    moved_dynamic = moved.model.condition_bank.context.dynamic_fields
    moved_causal = moved.model.advection.causal
    for actual, channel in (
        (moved_causal.context_rate_mm_per_hour, 0),
        (moved_causal.context_valid_fraction, 3),
        (moved_causal.context_interpolation_confidence, 4),
    ):
        torch.testing.assert_close(actual, moved_dynamic[:, :, channel])
        assert (
            actual.untyped_storage().data_ptr()
            == moved_dynamic.untyped_storage().data_ptr()
        )

    standalone = batch.model.advection.to("cpu", non_blocking=True)
    assert standalone is not None
    assert standalone.causal.target_rate_mm_per_hour.device.type == "cpu"
    assert standalone.lead_indices.device.type == "cpu"
    assert standalone.context_channels_alias_condition_bank is True


def test_model_batch_row_selection_preserves_single_row_views_and_aliases() -> None:
    batch = make_batch(leads=(0.5, 1.0)).model
    selected = batch.select_rows((1,))

    selected.validate()
    assert selected.batch_size == 1
    assert selected.embedding.lead_hours.tolist() == [1.0]
    assert selected.provenance.sample_ids == (batch.provenance.sample_ids[1],)
    assert selected.provenance.duplicate_condition_groups == ()
    assert (
        selected.condition_bank.target.radar_history.untyped_storage().data_ptr()
        == batch.condition_bank.target.radar_history.untyped_storage().data_ptr()
    )
    dynamic = selected.condition_bank.context.dynamic_fields
    causal = selected.advection.causal
    for actual, channel in (
        (causal.context_rate_mm_per_hour, 0),
        (causal.context_valid_fraction, 3),
        (causal.context_interpolation_confidence, 4),
    ):
        assert actual.untyped_storage().data_ptr() == dynamic.untyped_storage().data_ptr()
        torch.testing.assert_close(actual, dynamic[:, :, channel])


def test_multiprocess_dataloader_retains_context_channel_views() -> None:
    loader = DataLoader(
        _WorkerBuiltTrainingSampleDataset(),
        batch_size=1,
        num_workers=1,
        collate_fn=_worker_collate_training_batch,
        multiprocessing_context="spawn",
        persistent_workers=False,
        timeout=30.0,
    )
    batch = next(iter(loader))
    dynamic = batch.model.condition_bank.context.dynamic_fields
    causal = batch.model.advection.causal

    assert batch.model.advection.context_channels_alias_condition_bank is True
    for actual, channel in (
        (causal.context_rate_mm_per_hour, 0),
        (causal.context_valid_fraction, 3),
        (causal.context_interpolation_confidence, 4),
    ):
        expected = dynamic[:, :, channel]
        torch.testing.assert_close(actual, expected)
        assert (
            actual.untyped_storage().data_ptr()
            == dynamic.untyped_storage().data_ptr()
        )
        assert actual.data_ptr() == expected.data_ptr()
        assert actual.stride() == expected.stride()


def test_pin_memory_retains_context_channel_views() -> None:
    if not torch.cuda.is_available():
        pytest.skip("a CUDA-backed pin-memory allocator is unavailable")

    rank_local = RankLocalBatch(
        training=make_batch(leads=(0.5,)),
        slots=(DrawSlot(logical_position=0, row_index=0),),
        active_slot_indices=(0,),
        loss_weights=torch.ones(1, dtype=torch.float32),
    )
    pinned_rank_local = rank_local.pin_memory()
    assert isinstance(pinned_rank_local.training, TrainingBatch)
    pinned = pinned_rank_local.training
    dynamic = pinned.model.condition_bank.context.dynamic_fields
    causal = pinned.model.advection.causal

    assert dynamic.is_pinned()
    for actual, channel in (
        (causal.context_rate_mm_per_hour, 0),
        (causal.context_valid_fraction, 3),
        (causal.context_interpolation_confidence, 4),
    ):
        expected = dynamic[:, :, channel]
        assert actual.is_pinned()
        assert (
            actual.untyped_storage().data_ptr()
            == dynamic.untyped_storage().data_ptr()
        )
        assert actual.data_ptr() == expected.data_ptr()
        assert actual.stride() == expected.stride()


def test_named_era_masks_signature_provenance_and_raw_values_are_preserved() -> None:
    batch = make_batch(leads=(0.5,))
    era = batch.model.era

    assert era.channel_names == ERA5_CHANNEL_NAMES
    assert era.value_space == "physical_raw"
    assert era.normalization_artifact_id is None
    assert era.values[0, 0, 0, 0, 0].item() == 1.0
    assert era.values[0, 0, 23, 0, 0].item() == 24.0
    assert era.data_valid_inst.dtype is torch.bool
    assert era.tp_valid.dtype is torch.bool
    assert era.trajectory_window_mask.dtype is torch.bool
    assert era.temporal_access_mask.dtype is torch.bool
    assert era.data_valid_inst.data_ptr() != era.tp_valid.data_ptr()
    assert era.condition_signatures == (SIGNATURE,)
    assert era.provenance[0].condition_signature == SIGNATURE
    assert era.valid_times_utc[0] == era.provenance[0].valid_times_utc
    torch.testing.assert_close(
        era.tp_interval_center_delta_hours, era.delta_hours - 0.5
    )


def test_era_condition_signature_is_bound_to_presence_and_access_masks() -> None:
    model = make_batch(leads=(0.5,)).model
    with pytest.raises(ValueError, match="source-presence flags mismatch"):
        replace(model.era, tp_present=torch.zeros_like(model.era.tp_present))

    inaccessible = model.era.temporal_access_mask.clone()
    inaccessible[0, 0] = False
    with pytest.raises(ValueError, match="full-trajectory condition"):
        replace(model.era, temporal_access_mask=inaccessible)

    # Even coherently changing both masks is rejected at the complete model
    # batch boundary because trajectory/access are recomputed from t0+lead.
    shortened = model.era.trajectory_window_mask.clone()
    shortened[0, -2] = False
    forged_era = replace(
        model.era,
        trajectory_window_mask=shortened,
        temporal_access_mask=shortened.clone(),
    )
    with pytest.raises(ValueError, match="trajectory mask disagrees"):
        replace(model, era=forged_era)

    model.era.tp_present.fill_(False)
    with pytest.raises(ValueError, match="source-presence flags mismatch"):
        model.validate()


def test_actual_lcc_axes_define_target_context_and_era_token_footprints() -> None:
    batch = make_batch(leads=(0.5,))
    geometry = batch.model.geometry

    assert geometry.target_l3.x_shared.shape == (2, 2)
    assert geometry.target_l4.x_shared.shape == (1, 1)
    assert geometry.context_l3.x_shared.shape == (2, 2)
    torch.testing.assert_close(
        geometry.target_l3.footprint_width,
        torch.full((2, 2), 4.0),
    )
    torch.testing.assert_close(
        geometry.target_l4.footprint_height,
        torch.full((1, 1), 8.0),
    )
    torch.testing.assert_close(
        geometry.context_l3.footprint_width,
        torch.full((2, 2), 9.6),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        geometry.context_l4.footprint_height,
        torch.full((1, 1), 19.2),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    torch.testing.assert_close(
        geometry.target_l3.x_shared,
        torch.tensor([[-0.0375, 0.0025], [-0.0375, 0.0025]]),
        rtol=0.0,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        geometry.context_l3.x_shared,
        torch.tensor([[-0.09, 0.006], [-0.09, 0.006]]),
        rtol=0.0,
        atol=1.0e-7,
    )
    assert geometry.era_native.x_shared.shape == (33, 33)
    assert torch.all(geometry.era_native.footprint_width > 0.0)
    assert torch.all(geometry.era_native.footprint_height > 0.0)
    context_static = batch.model.condition_bank.context.static_fields
    assert context_static.shape == (1, len(CONTEXT_STATIC_CHANNEL_NAMES), 16, 16)
    # Shared coordinates retain the context's larger physical extent.
    assert context_static[0, 0].abs().max() > geometry.target_l3.x_shared.abs().max()


class ScaleNormalization:
    artifact_id = "outer-train-era-norm-sha256:test"

    def __init__(self) -> None:
        self.calls = 0

    def normalize(self, values: np.ndarray, **kwargs: object) -> np.ndarray:
        self.calls += 1
        assert kwargs["channel_names"] == ERA5_CHANNEL_NAMES
        return np.asarray(values * np.float32(0.25), dtype=np.float32)


def test_era_normalization_is_explicit_and_records_artifact_identity() -> None:
    grid = make_grid()
    normalization = ScaleNormalization()
    batch = TrainingBatchCollator(
        grid,
        era_normalization=normalization,
        allow_test_override=True,
    )([make_sample(grid=grid)])

    assert normalization.calls == 1
    assert batch.model.era.value_space == "explicitly_normalized"
    assert (
        batch.model.era.normalization_artifact_id
        == "outer-train-era-norm-sha256:test"
    )
    assert batch.model.era.values[0, 0, 23, 0, 0].item() == 6.0


def test_duplicate_t0_signature_histories_are_validated_not_deduplicated() -> None:
    grid = make_grid()
    first = make_sample(grid=grid, sample_id="first", lead_hours=0.5)
    second = make_sample(grid=grid, sample_id="second", lead_hours=1.0)
    changed_history = second.conditions.target_history.copy()
    changed_history[-1, 3, 4] += np.float32(0.01)
    second = replace(
        second,
        conditions=replace(second.conditions, target_history=changed_history),
    )
    with pytest.raises(ValueError, match="duplicate t0/signature histories disagree"):
        collate_training_samples(
            [first, second], geometry=grid, allow_test_override=True
        )

    # Disabling the expensive equality audit is explicit and still retains
    # both rows rather than hiding a deduplicated representation.
    retained = collate_training_samples(
        [first, second],
        geometry=grid,
        validate_duplicate_histories=False,
        allow_test_override=True,
    )
    assert retained.model.batch_size == 2
    assert retained.model.provenance.duplicate_condition_groups == ()
    assert not torch.equal(
        retained.model.condition_bank.target.radar_history[0],
        retained.model.condition_bank.target.radar_history[1],
    )


def test_time_dtype_contracts_and_geometry_tuning() -> None:
    grid = make_grid()
    sample = make_sample(grid=grid)
    wrong_time = replace(
        sample,
        conditions=replace(
            sample.conditions, time_features=(0.0, 0.0, 0.0, 0.0)
        ),
    )
    with pytest.raises(ValueError, match="verification-time"):
        collate_training_samples(
            [wrong_time], geometry=grid, allow_test_override=True
        )

    wrong_dtype = replace(
        sample,
        conditions=replace(
            sample.conditions,
            target_history=sample.conditions.target_history.astype(np.float64),
        ),
    )
    with pytest.raises(TypeError, match="archive float32"):
        collate_training_samples(
            [wrong_dtype], geometry=grid, allow_test_override=True
        )

    assert TrainingBatchCollator(grid).geometry is grid


def test_model_batch_binds_canonical_fp32_time_embedding_to_t0_and_lead() -> None:
    original = make_batch(leads=(1.0,)).model
    mismatched_embedding = replace(
        original.embedding,
        verification_cyclic=torch.zeros_like(
            original.embedding.verification_cyclic
        ),
    )

    with pytest.raises(ValueError, match="canonical FP32 t0/lead"):
        replace(original, embedding=mismatched_embedding)

    original.embedding.verification_cyclic.zero_()
    with pytest.raises(ValueError, match="canonical FP32 t0/lead"):
        original.validate()


def test_production_geometry_golden_spacing_and_orientation() -> None:
    target = (np.arange(256, dtype=np.float64) - 127.5) * 0.5
    context = (np.arange(256, dtype=np.float64) - 127.5) * 1.2
    production = PhysicalGridSpec(
        target_x_lcc_km=target,
        target_y_lcc_km=target,
        context_x_lcc_km=context,
        context_y_lcc_km=context,
        era_latitude_degrees=np.linspace(32.0, 40.0, 33),
        era_longitude_degrees=np.linspace(123.0, 131.0, 33),
    )
    assert production.input_size == 256
    assert production.target_center_lcc_km == pytest.approx((0.0, 0.0))
    with pytest.raises(ValueError, match="south-to-north"):
        replace(
            production,
            era_latitude_degrees=np.linspace(40.0, 32.0, 33),
        )
