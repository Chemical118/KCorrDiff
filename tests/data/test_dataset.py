from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from kcorrdiff.data.condition_schema import (
    CONTEXT_DYNAMIC_CHANNEL_NAMES,
    Era5Conditions,
    NamedSpatialFields,
    StaticConditions,
)
from kcorrdiff.data.context_regrid import build_context_regrid_operator
from kcorrdiff.data.dataset import (
    KCorrDiffDataset,
    RadarConditions,
    SharedT0BatchSampler,
    dataloader_worker_options,
)
from kcorrdiff.data.era5_reader import (
    Era5Window,
    Era5WindowProvenance,
    native_hour_times,
)
from kcorrdiff.data.provider_adapter import ConditionSignature
from kcorrdiff.data.sampling import CandidateItem, DrawRow, bounded_draw_manifest


def rows():
    candidates = [
        CandidateItem(
            sample_id=f"s{i}",
            t0_utc="2022-01-01T00:00:00+00:00" if i < 3 else "2022-01-02T00:00:00+00:00",
            lead_hours=0.5 + i * 0.5,
            condition_signature="era5",
            block_id="e",
            stratum="event",
            split="outer_train",
            p_target=0.25,
        )
        for i in range(4)
    ]
    return bounded_draw_manifest(candidates, draws=20, seed=7, purpose_id="x")


def test_shared_t0_sampler_never_mixes_issue_times() -> None:
    values = rows()
    sampler = SharedT0BatchSampler(values, batch_size=3)
    for batch in sampler:
        assert len({values[index].t0_utc for index in batch}) == 1
        leads = [values[index].lead_hours for index in batch]
        assert leads == sorted(leads)


def test_worker_options_enable_persistent_prefetch_only_when_valid() -> None:
    assert dataloader_worker_options(0) == {"num_workers": 0, "pin_memory": True}
    assert dataloader_worker_options(8, 4) == {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }


class RecordingCache:
    def __init__(self, *, context_value: float = 1.0 / 3.0) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.context_value = context_value

    def read_many(self, source: str, timestamps: list[str]) -> np.ndarray:
        self.calls.append((source, tuple(timestamps)))
        value = self.context_value if source == "condition" else 1.0 / 3.0
        return np.full((len(timestamps), 2, 2), value, dtype=np.float32)


class RecordingEra5Cache:
    def __init__(self, *, provenance_signature: str | None = None) -> None:
        self.calls: list[tuple[datetime, float, ConditionSignature, bool]] = []
        self.provenance_signature = provenance_signature

    def read_window(
        self,
        t0: datetime,
        *,
        lead_hours: float,
        signature: ConditionSignature,
        strict_instantaneous: bool,
    ) -> Era5Window:
        self.calls.append((t0, lead_hours, signature, strict_instantaneous))
        valid_times = native_hour_times(t0)
        delta = np.asarray(
            [(value - t0) / timedelta(hours=1) for value in valid_times],
            dtype=np.float32,
        )
        canonical = signature.canonical()
        access_mode = (
            "target_end_causal"
            if canonical.temporal_access_mode == "target_end_causal"
            else "full_trajectory"
        )
        provenance = Era5WindowProvenance(
            provider_id="era5",
            provider_version="test",
            dataset="synthetic",
            cache_format="synthetic-v1",
            cache_manifest="/cache/manifest.json",
            condition_signature=(
                self.provenance_signature or canonical.key
            ),
            access_mode=access_mode,
            valid_times_utc=valid_times,
            tp_intervals_utc=tuple(
                (value - timedelta(hours=1), value) for value in valid_times
            ),
        )
        instantaneous = np.arange(8 * 23 * 2 * 3, dtype=np.float32).reshape(
            8, 23, 2, 3
        )
        tp = np.full((8, 1, 2, 3), 2.0, dtype=np.float32)
        data_valid = np.asarray(
            [True, True, True, True, True, True, False, True]
        )
        tp_valid = np.asarray(
            [True, False, True, True, True, True, True, True]
        )
        trajectory = np.asarray([True] * 7 + [False])
        temporal = np.asarray([True] * 8)
        if not canonical.era_present:
            instantaneous.fill(0.0)
            tp.fill(0.0)
            temporal.fill(False)
        elif not canonical.tp_present:
            tp.fill(0.0)
        return Era5Window(
            instantaneous=instantaneous,
            tp=tp,
            valid_times_utc=valid_times,
            delta_hours=delta,
            tp_interval_center_delta_hours=delta - 0.5,
            data_valid_inst=data_valid,
            tp_valid=tp_valid,
            trajectory_window_mask=trajectory,
            temporal_access_mask=temporal,
            era_present=canonical.era_present,
            tp_present=canonical.era_present and canonical.tp_present,
            provenance=provenance,
        )


def regrid_operator():
    return build_context_regrid_operator([0.0, 1.0], [0.0, 1.0])


def direct_row() -> DrawRow:
    return DrawRow(
        global_example_index=0,
        sample_id="sample",
        t0_utc="2022-01-01T00:00:00+00:00",
        lead_hours=0.5,
        condition_signature="era5_oracle:era=1:tp=1:full_trajectory",
        block_id="event-a",
        stratum="event",
        split="outer_train",
        p_target=1.0,
        p_draw=1.0,
        omega=1.0,
    )


def make_dataset(
    *,
    cache: RecordingCache | None = None,
    era5_cache: RecordingEra5Cache | None = None,
    rows_override: list[object] | None = None,
    fold_ids: dict[object, int | None] | None = None,
) -> KCorrDiffDataset:
    return KCorrDiffDataset(
        cache=cache or RecordingCache(),  # type: ignore[arg-type]
        rows=rows_override or [direct_row()],  # type: ignore[arg-type]
        context_regrid_operator=regrid_operator(),
        era5_cache=era5_cache or RecordingEra5Cache(),  # type: ignore[arg-type]
        static_channels=np.zeros((2, 2, 2), dtype=np.float32),
        static_coverage=np.asarray([[True, False], [True, True]]),
        static_channel_names=("dem_elevation_m", "land_sea_mask"),
        fold_ids=fold_ids,
    )


def test_future_target_values_and_validity_remain_label_only() -> None:
    cache = RecordingCache()
    dataset = make_dataset(cache=cache)

    sample = dataset[0]

    assert [source for source, _ in cache.calls] == [
        "target",
        "condition",
        "target",
    ]
    assert [len(timestamps) for _, timestamps in cache.calls] == [12, 12, 7]
    future_only = set(cache.calls[2][1][1:])
    assert future_only.isdisjoint(cache.calls[0][1])
    assert future_only.isdisjoint(cache.calls[1][1])
    assert "target_validity" not in {field.name for field in fields(RadarConditions)}
    assert "target_validity" not in {
        field.name for field in fields(StaticConditions)
    }
    np.testing.assert_array_equal(
        sample.label.target_validity,
        np.asarray([[True, False], [True, True]]),
    )
    assert not np.shares_memory(
        sample.label.target_validity, sample.conditions.history_validity
    )
    assert not np.shares_memory(
        sample.label.target_validity, sample.conditions.static_channels
    )


def test_context_history_is_inverted_then_regridded_into_named_channels() -> None:
    # CPrecNet normalized 1/3 maps to exactly 1 mm/h.  An identity-grid
    # regrid must therefore emit 1, not the stored normalized value.
    sample = make_dataset(cache=RecordingCache(context_value=1.0 / 3.0))[0]
    context = sample.conditions.context

    assert context.channel_names == CONTEXT_DYNAMIC_CHANNEL_NAMES
    assert context.mean_rate_mm_per_hour.shape == (12, 2, 2)
    np.testing.assert_allclose(context.mean_rate_mm_per_hour, 1.0, rtol=2.0e-7)
    np.testing.assert_allclose(context.detail_rate_mm_per_hour, 1.0, rtol=2.0e-7)
    np.testing.assert_array_equal(context.wet_mask, True)
    np.testing.assert_allclose(context.valid_fraction, 1.0)
    np.testing.assert_allclose(context.interpolation_confidence, 1.0)
    np.testing.assert_array_equal(context.detail_valid, True)
    assert sample.conditions.context_history.shape == (12, 5, 2, 2)


def test_era5_window_values_times_masks_presence_and_provenance_are_wired() -> None:
    era_cache = RecordingEra5Cache()
    sample = make_dataset(era5_cache=era_cache)[0]

    assert len(era_cache.calls) == 1
    t0, lead, signature, strict = era_cache.calls[0]
    assert t0.isoformat() == "2022-01-01T00:00:00+00:00"
    assert lead == 0.5
    assert signature.key == "era5_oracle:era=1:tp=1:full_trajectory"
    assert strict is True

    era = sample.conditions.era5
    assert isinstance(era, Era5Conditions)
    assert era.values.shape == (8, 24, 2, 3)
    assert era.delta_hours.shape == (8,)
    np.testing.assert_allclose(
        era.tp_interval_center_delta_hours, era.delta_hours - 0.5
    )
    assert era.data_valid_inst.tolist() == [True] * 6 + [False, True]
    assert era.tp_valid.tolist() == [True, False] + [True] * 6
    assert era.trajectory_window_mask.tolist() == [True] * 7 + [False]
    assert era.temporal_access_mask.tolist() == [True] * 8
    assert era.era_present is True and era.tp_present is True
    assert era.condition_signature == sample.conditions.condition_signature
    assert era.lead_hours == sample.conditions.lead_hours
    assert era.provenance.provider_id == "era5"
    assert era.provenance.condition_signature == era.condition_signature


def test_row_level_tp_and_whole_source_dropout_drive_distinct_presence_masks() -> None:
    tp_off = "era5_oracle:era=1:tp=0:full_trajectory"
    null = "null_provider:era=0:tp=0:no_era_access"
    base = direct_row()
    rows_override = [
        base,
        replace(
            base,
            global_example_index=1,
            sample_id="tp-off",
            condition_signature=tp_off,
        ),
        replace(
            base,
            global_example_index=2,
            sample_id="era-null",
            condition_signature=null,
        ),
    ]
    era_cache = RecordingEra5Cache()
    dataset = make_dataset(
        era5_cache=era_cache,
        rows_override=rows_override,
    )

    full_sample, tp_off_sample, null_sample = (
        dataset[index] for index in range(3)
    )
    assert full_sample.conditions.era5.era_present is True
    assert full_sample.conditions.era5.tp_present is True
    assert tp_off_sample.conditions.era5.era_present is True
    assert tp_off_sample.conditions.era5.tp_present is False
    assert null_sample.conditions.era5.era_present is False
    assert null_sample.conditions.era5.tp_present is False
    np.testing.assert_array_equal(
        tp_off_sample.conditions.era5.values[:, -1], 0.0
    )
    np.testing.assert_array_equal(null_sample.conditions.era5.values, 0.0)
    np.testing.assert_array_equal(
        null_sample.conditions.era5.temporal_access_mask, False
    )
    assert [call[2].key for call in era_cache.calls] == [
        base.condition_signature,
        tp_off,
        null,
    ]


def test_era_provenance_signature_mismatch_fails_closed() -> None:
    era_cache = RecordingEra5Cache(
        provenance_signature="era5_oracle:era=1:tp=0:full_trajectory"
    )
    with pytest.raises(ValueError, match="provenance/signature mismatch"):
        make_dataset(era5_cache=era_cache)[0]


def test_fold_id_is_duck_typed_from_row_or_separate_mapping() -> None:
    base = direct_row()
    row_with_fold = SimpleNamespace(
        **{
            field.name: getattr(base, field.name)
            for field in fields(DrawRow)
            if field.name != "fold_id"
        },
        fold_id=3,
    )
    assert make_dataset(rows_override=[row_with_fold])[0].fold_id == 3
    assert make_dataset(fold_ids={base.sample_id: 4})[0].fold_id == 4
    assert (
        make_dataset(
            fold_ids={(base.sample_id, base.condition_signature): 5}
        )[0].fold_id
        == 5
    )

    with pytest.raises(ValueError, match="conflicting fold_id"):
        make_dataset(
            rows_override=[row_with_fold], fold_ids={base.sample_id: 2}
        )[0]


def test_named_static_schema_rejects_future_derived_channels() -> None:
    with pytest.raises(ValueError, match="future-derived"):
        NamedSpatialFields(
            ("dem_elevation_m", "target_validity"),
            np.zeros((2, 2, 2), dtype=np.float32),
        )

    sample = make_dataset()[0]
    static = sample.conditions.static
    assert static.target.channel_names == ("dem_elevation_m", "land_sea_mask")
    np.testing.assert_array_equal(
        static.target_static_coverage,
        [[True, False], [True, True]],
    )
    assert static.context.source_dx_km.shape == (2, 2)
