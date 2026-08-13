"""Timestamp availability contracts for CPrecNet and continuous radar data.

Missing archive keys are missing observations, never dry fields.  This module
therefore works only with timestamp sets and deliberately does not manufacture
values for gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Collection, Iterable, Iterator, Literal, Sequence
from zoneinfo import ZoneInfo


FIVE_MINUTES = timedelta(minutes=5)
HISTORY_OFFSETS = tuple(timedelta(minutes=5 * i) for i in range(-11, 1))
LEADS_HOURS = tuple(0.5 * i for i in range(1, 13))


def parse_radar_timestamp(
    value: str, *, timezone: str = "Asia/Seoul"
) -> datetime:
    """Parse a CPrecNet ``YYYYMMDDHHMM`` key and return an aware UTC time."""

    local = datetime.strptime(value, "%Y%m%d%H%M").replace(
        tzinfo=ZoneInfo(timezone)
    )
    return local.astimezone(UTC)


def format_radar_timestamp(
    value: datetime, *, timezone: str = "Asia/Seoul"
) -> str:
    """Format an aware instant using the archive's declared radar timezone."""

    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(ZoneInfo(timezone)).strftime("%Y%m%d%H%M")


def require_five_minute_slot(value: datetime) -> None:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.minute % 5 or value.second or value.microsecond:
        raise ValueError(f"not a five-minute radar slot: {value.isoformat()}")


def history_times(t0: datetime) -> tuple[datetime, ...]:
    """The 12 causal input scans from ``t0-55 min`` through ``t0``."""

    require_five_minute_slot(t0)
    return tuple(t0 + offset for offset in HISTORY_OFFSETS)


def target_times(t0: datetime, lead_hours: float) -> tuple[datetime, ...]:
    """Seven scan instants for a lead's 30-minute trapezoidal target."""

    require_five_minute_slot(t0)
    if lead_hours not in LEADS_HOURS:
        raise ValueError(f"lead must be one of {LEADS_HOURS}, got {lead_hours}")
    end = t0 + timedelta(hours=lead_hours)
    start = end - timedelta(minutes=30)
    return tuple(start + i * FIVE_MINUTES for i in range(7))


def dependency_times(t0: datetime, max_lead_hours: float = 6.0) -> tuple[datetime, ...]:
    """All 5-minute instants in the maximum dependency interval."""

    require_five_minute_slot(t0)
    if max_lead_hours <= 0:
        raise ValueError("max_lead_hours must be positive")
    start = t0 - timedelta(minutes=55)
    end = t0 + timedelta(hours=max_lead_hours)
    count = int((end - start) / FIVE_MINUTES) + 1
    return tuple(start + i * FIVE_MINUTES for i in range(count))


def endpoint_times(t0: datetime, lead_hours: float) -> tuple[datetime, ...]:
    """Minimum scans needed for one lead-conditioned training item."""

    # The lead=0.5 target begins at t0, so remove the duplicate deterministically.
    return tuple(dict.fromkeys((*history_times(t0), *target_times(t0, lead_hours))))


def is_available(
    timestamps: Collection[datetime],
    t0: datetime,
    lead_hours: float,
    *,
    mode: Literal["endpoint", "strict"] = "endpoint",
) -> bool:
    required = (
        endpoint_times(t0, lead_hours)
        if mode == "endpoint"
        else dependency_times(t0, lead_hours)
    )
    return all(value in timestamps for value in required)


def intersection_timestamps(
    target_keys: Iterable[str],
    condition_keys: Iterable[str],
    *,
    timezone: str = "Asia/Seoul",
) -> tuple[datetime, ...]:
    """Sorted target/condition key intersection in UTC."""

    common = set(target_keys).intersection(condition_keys)
    return tuple(sorted(parse_radar_timestamp(key, timezone=timezone) for key in common))


def continuous_runs(timestamps: Iterable[datetime]) -> tuple[tuple[datetime, ...], ...]:
    """Partition unique timestamps into maximal five-minute runs."""

    ordered = sorted(set(timestamps))
    if not ordered:
        return ()
    runs: list[list[datetime]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - runs[-1][-1] == FIVE_MINUTES:
            runs[-1].append(value)
        else:
            runs.append([value])
    return tuple(tuple(run) for run in runs)


@dataclass(frozen=True, slots=True)
class AvailabilityStats:
    timestamps: int
    continuous_runs: int
    strict_six_hour_windows: int
    supporting_runs: int
    nonoverlapping_84_frame_blocks: int


def availability_stats(timestamps: Iterable[datetime]) -> AvailabilityStats:
    """Compute the reference availability statistics from architecture §18.1."""

    runs = continuous_runs(timestamps)
    window = len(dependency_times(next(iter(runs[0])))) if runs else 84
    supported = [run for run in runs if len(run) >= window]
    return AvailabilityStats(
        timestamps=sum(len(run) for run in runs),
        continuous_runs=len(runs),
        strict_six_hour_windows=sum(len(run) - window + 1 for run in supported),
        supporting_runs=len(supported),
        nonoverlapping_84_frame_blocks=sum(len(run) // window for run in runs),
    )


def iter_eligible_t0(
    timestamps: Collection[datetime],
    *,
    lead_hours: float,
    mode: Literal["endpoint", "strict"] = "endpoint",
) -> Iterator[datetime]:
    """Yield eligible issue times without treating an absent key as zero rain."""

    for value in sorted(timestamps):
        if is_available(timestamps, value, lead_hours, mode=mode):
            yield value
