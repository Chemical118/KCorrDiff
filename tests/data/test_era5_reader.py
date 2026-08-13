from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from kcorrdiff.data.era5_reader import (
    ERA5_NATIVE_HOURS,
    Era5MmapCache,
    build_era5_mmap_cache,
    estimate_era5_cache_bytes,
    floor_utc_hour,
    native_hour_times,
    temporal_access_mask,
    trajectory_window_mask,
)
from kcorrdiff.data.provider_adapter import (
    ERA5_CHANNELS,
    ERA5_FULL_TRAJECTORY,
    ConditionSignature,
    Era5ProviderAdapter,
    ProviderMessage,
)


LATITUDE = np.asarray([40.0, 39.0])
LONGITUDE = np.asarray([123.0, 124.0, 125.0])


class SyntheticDecoder:
    def __init__(self, messages: list[ProviderMessage]) -> None:
        self.messages = messages
        self.received_sources: tuple[object, ...] | None = None

    def iter_messages(self, sources: list[object] | tuple[object, ...]):
        self.received_sources = tuple(sources)
        yield from self.messages


def messages_for_hour(valid_time: datetime, ordinal: int) -> list[ProviderMessage]:
    result = []
    for channel in ERA5_CHANNELS:
        if channel.accumulated:
            result.append(
                ProviderMessage(
                    short_name="tp",
                    level_hpa=None,
                    valid_time=valid_time,
                    values=np.full((2, 3), (ordinal + 1) / 1000.0),
                    units="m",
                    step_type="accum",
                    interval_start=valid_time - timedelta(hours=1),
                    interval_end=valid_time,
                    latitude=LATITUDE,
                    longitude=LONGITUDE,
                    source=f"single-levels-{valid_time.year}.grib",
                )
            )
        else:
            result.append(
                ProviderMessage(
                    short_name=channel.short_name,
                    level_hpa=channel.level_hpa,
                    valid_time=valid_time,
                    values=np.full((2, 3), ordinal * 100 + channel.index),
                    units=channel.source_units[0],
                    step_type="instant",
                    latitude=LATITUDE,
                    longitude=LONGITUDE,
                    source=f"instantaneous-{valid_time.year}.grib",
                )
            )
    return result


def build_boundary_cache(tmp_path: Path) -> tuple[Path, tuple[datetime, ...]]:
    t0 = datetime(2020, 12, 31, 23, 35, tzinfo=UTC)
    hours = native_hour_times(t0)
    messages = [message for i, hour in enumerate(hours) for message in messages_for_hour(hour, i)]
    destination = tmp_path / "era-cache"
    decoder = SyntheticDecoder(messages)
    manifest = build_era5_mmap_cache(
        decoder=decoder,
        sources=["2020.grib", "2021.grib"],
        output_dir=destination,
        years=[2020, 2021],
        adapter=Era5ProviderAdapter(grid_shape=(2, 3)),
        require_complete_instantaneous=False,
        require_complete_tp=False,
        reserve_bytes=0,
        verify_sha256=False,
    )
    assert decoder.received_sources == ("2020.grib", "2021.grib")
    assert tuple(shard.year for shard in manifest.shards) == (2020, 2021)
    return destination, hours


def test_native_hours_are_absolute_utc_and_cross_calendar_boundaries() -> None:
    kst = datetime(2021, 1, 1, 8, 35, tzinfo=ZoneInfo("Asia/Seoul"))
    hours = native_hour_times(kst)
    assert len(hours) == ERA5_NATIVE_HOURS
    assert hours[0] == datetime(2020, 12, 31, 23, tzinfo=UTC)
    assert hours[-1] == datetime(2021, 1, 1, 6, tzinfo=UTC)
    assert floor_utc_hour(kst) == hours[0]
    with pytest.raises(ValueError, match="timezone-aware"):
        native_hour_times(kst.replace(tzinfo=None))


def test_trajectory_and_access_masks_are_separate() -> None:
    exact = datetime(2022, 1, 1, tzinfo=UTC)
    non_hour = exact.replace(minute=5)
    assert trajectory_window_mask(exact).tolist() == [True] * 7 + [False]
    assert trajectory_window_mask(non_hour).tolist() == [True] * 8
    assert temporal_access_mask(
        non_hour, lead_hours=0.5, access_mode="full_trajectory"
    ).tolist() == [True] * 8
    assert temporal_access_mask(
        non_hour, lead_hours=0.5, access_mode="target_end_causal"
    ).tolist() == [True, False, False, False, False, False, False, False]
    with pytest.raises(ValueError, match="30-minute"):
        temporal_access_mask(non_hour, lead_hours=0.75, access_mode="full_trajectory")


def test_yearly_cache_round_trip_and_boundary_window(tmp_path: Path) -> None:
    destination, hours = build_boundary_cache(tmp_path)
    cache = Era5MmapCache(destination)
    window = cache.read_window(
        datetime(2020, 12, 31, 23, 35, tzinfo=UTC),
        lead_hours=6.0,
        signature=ERA5_FULL_TRAJECTORY,
    )
    assert window.valid_times_utc == hours
    assert window.instantaneous.shape == (8, 23, 2, 3)
    assert window.tp.shape == (8, 1, 2, 3)
    assert window.values.shape == (8, 24, 2, 3)
    assert window.values.dtype == np.float32
    assert window.data_valid_inst.tolist() == [True] * 8
    assert window.tp_valid.tolist() == [True] * 8
    assert window.trajectory_window_mask.tolist() == [True] * 8
    assert window.temporal_access_mask.tolist() == [True] * 8
    assert np.all(window.instantaneous[3, 10] == 310)
    # Provider rows were north-to-south, while the cache is canonical south-to-north.
    assert np.array_equal(cache.latitude, [39.0, 40.0])
    # Source metres have been converted to canonical millimetres.
    assert np.allclose(window.tp[:, 0, 0, 0], np.arange(1, 9))
    assert np.allclose(window.tp_interval_center_delta_hours, window.delta_hours - 0.5)
    assert window.provenance.provider_id == "era5"
    assert window.provenance.tp_intervals_utc[0] == (
        hours[0] - timedelta(hours=1),
        hours[0],
    )
    assert {path.name for path in destination.glob("instantaneous-*.npy")} == {
        "instantaneous-2020.npy",
        "instantaneous-2021.npy",
    }
    assert {path.name for path in destination.glob("tp-[0-9]*.npy")} == {
        "tp-2020.npy",
        "tp-2021.npy",
    }


def test_target_end_causal_and_source_dropout_do_not_merge_masks(tmp_path: Path) -> None:
    destination, _ = build_boundary_cache(tmp_path)
    cache = Era5MmapCache(destination)
    t0 = datetime(2020, 12, 31, 23, 35, tzinfo=UTC)
    causal = ConditionSignature(
        provider_track="era5_oracle",
        era_present=True,
        tp_present=False,
        temporal_access_mode="target_end_causal",
    )
    window = cache.read_window(t0, lead_hours=0.5, signature=causal)
    assert window.data_valid_inst.tolist() == [True] * 8
    assert window.tp_valid.tolist() == [True] * 8
    assert window.trajectory_window_mask.tolist() == [True] * 8
    assert window.temporal_access_mask.tolist() == [True, True] + [False] * 6
    assert window.era_present is True and window.tp_present is False
    assert not window.tp.any()

    absent = ConditionSignature(
        provider_track="ignored",
        era_present=False,
        tp_present=True,
        temporal_access_mode="full_trajectory",
    )
    dropped = cache.read_window(t0, lead_hours=6.0, signature=absent)
    assert dropped.era_present is False and dropped.tp_present is False
    assert dropped.data_valid_inst.tolist() == [True] * 8
    assert not dropped.temporal_access_mask.any()
    assert not dropped.values.any()
    assert dropped.provenance.condition_signature.startswith("null_provider:")


def test_incomplete_instantaneous_token_is_masked_and_strict_reader_fails(
    tmp_path: Path,
) -> None:
    valid_time = datetime(2022, 4, 5, 0, tzinfo=UTC)
    messages = messages_for_hour(valid_time, 0)
    # Remove one of the 23 instantaneous semantic fields, but retain tp.
    messages = [message for message in messages if not (
        message.short_name == "q" and message.level_hpa == 925
    )]
    destination = tmp_path / "partial"
    build_era5_mmap_cache(
        decoder=SyntheticDecoder(messages),
        sources=["partial.grib"],
        output_dir=destination,
        years=[2022],
        adapter=Era5ProviderAdapter(grid_shape=(2, 3)),
        require_complete_instantaneous=False,
        require_complete_tp=False,
        reserve_bytes=0,
        verify_sha256=False,
    )
    cache = Era5MmapCache(destination)
    with pytest.raises(KeyError, match="missing required"):
        cache.read_window(
            valid_time,
            lead_hours=0.5,
            signature=ERA5_FULL_TRAJECTORY,
        )
    diagnostic = cache.read_window(
        valid_time,
        lead_hours=0.5,
        signature=ERA5_FULL_TRAJECTORY,
        strict_instantaneous=False,
    )
    assert diagnostic.data_valid_inst.tolist() == [False] * 8
    assert diagnostic.tp_valid.tolist() == [True] + [False] * 7
    assert not diagnostic.instantaneous[0].any()


def test_cache_rejects_duplicates_and_never_overwrites_destination(tmp_path: Path) -> None:
    valid_time = datetime(2022, 1, 1, tzinfo=UTC)
    duplicate = messages_for_hour(valid_time, 0)
    duplicate.append(duplicate[0])
    with pytest.raises(ValueError, match="duplicate ERA5"):
        build_era5_mmap_cache(
            decoder=SyntheticDecoder(duplicate),
            sources=["duplicate.grib"],
            output_dir=tmp_path / "duplicate",
            years=[2022],
            adapter=Era5ProviderAdapter(grid_shape=(2, 3)),
            require_complete_instantaneous=False,
            require_complete_tp=False,
            reserve_bytes=0,
            verify_sha256=False,
        )

    owned = tmp_path / "owned"
    owned.mkdir()
    marker = owned / "keep"
    marker.write_text("user")
    with pytest.raises(FileExistsError):
        build_era5_mmap_cache(
            decoder=SyntheticDecoder(messages_for_hour(valid_time, 0)),
            sources=["one.grib"],
            output_dir=owned,
            years=[2022],
            adapter=Era5ProviderAdapter(grid_shape=(2, 3)),
            reserve_bytes=0,
        )
    assert marker.read_text() == "user"


def test_cache_size_estimate_distinguishes_leap_year() -> None:
    non_leap = estimate_era5_cache_bytes([2021], grid_shape=(2, 3))
    leap = estimate_era5_cache_bytes([2020], grid_shape=(2, 3))
    assert leap - non_leap == 24 * (24 * 2 * 3 * 4 + 2)
