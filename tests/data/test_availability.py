from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kcorrdiff.data.availability import (
    availability_stats,
    continuous_runs,
    dependency_times,
    endpoint_times,
    history_times,
    intersection_timestamps,
    is_available,
    parse_radar_timestamp,
    target_times,
)


def dt(hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2022, 1, 1, hour, minute, tzinfo=UTC)


def test_radar_key_is_kst_and_normalized_to_utc() -> None:
    assert parse_radar_timestamp("202201010900") == dt()


def test_history_target_and_dependency_contracts() -> None:
    t0 = dt(1)
    assert len(history_times(t0)) == 12
    assert history_times(t0)[0] == dt(0, 5)
    assert history_times(t0)[-1] == t0
    assert len(target_times(t0, 0.5)) == 7
    assert target_times(t0, 0.5)[0] == t0
    assert target_times(t0, 0.5)[-1] == dt(1, 30)
    deps = dependency_times(t0)
    assert len(deps) == 84
    assert deps[0] == dt(0, 5)
    assert deps[-1] == dt(7)
    assert len(endpoint_times(t0, 0.5)) == 18


def test_endpoint_availability_does_not_require_middle_gap() -> None:
    t0 = dt(1)
    present = set(endpoint_times(t0, 6.0))
    assert is_available(present, t0, 6.0, mode="endpoint")
    assert not is_available(present, t0, 6.0, mode="strict")
    present.remove(target_times(t0, 6.0)[3])
    assert not is_available(present, t0, 6.0, mode="endpoint")


def test_continuous_run_and_reference_counts() -> None:
    first = [dt() + i * timedelta(minutes=5) for i in range(100)]
    second_start = dt() + timedelta(days=1)
    second = [second_start + i * timedelta(minutes=5) for i in range(84)]
    runs = continuous_runs([*first, *second, first[0]])
    assert tuple(map(len, runs)) == (100, 84)
    stats = availability_stats([*first, *second])
    assert stats.timestamps == 184
    assert stats.continuous_runs == 2
    assert stats.strict_six_hour_windows == 18
    assert stats.supporting_runs == 2
    assert stats.nonoverlapping_84_frame_blocks == 2


def test_intersection_and_input_validation() -> None:
    actual = intersection_timestamps(
        ["202201010900", "202201010905"],
        ["202201010905", "202201010910"],
    )
    assert actual == (dt(0, 5),)
    with pytest.raises(ValueError, match="five-minute"):
        history_times(dt(0, 1))
    with pytest.raises(ValueError, match="lead"):
        target_times(dt(), 0.75)
