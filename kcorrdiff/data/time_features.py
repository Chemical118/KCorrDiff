"""UTC indexing and verification-time features for K-CorrDiff v1.1.3b."""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo


TARGET_CENTER_LONGITUDE_DEGREES_EAST = 127.25330141
"""Frozen LCC target-domain center longitude used by v1.1.3b."""

OFFICIAL_LEAD_HOURS = tuple(index / 2.0 for index in range(1, 13))


@dataclass(frozen=True, slots=True)
class VerificationTimeFeatures:
    """Deterministic temporal condition for one official forecast lead."""

    t0_utc: datetime
    verification_center_utc: datetime
    mean_solar_time: datetime
    solar_hour: float
    days_in_year: int
    annual_phase: float
    cyclic_features: tuple[float, float, float, float]


def _timezone(value: str | tzinfo) -> tzinfo:
    if isinstance(value, str):
        return ZoneInfo(value)
    if not isinstance(value, tzinfo):
        raise TypeError("declared_radar_timezone must be an IANA name or tzinfo")
    return value


def to_utc(
    timestamp: datetime,
    *,
    declared_radar_timezone: str | tzinfo | None = None,
) -> datetime:
    """Return an aware UTC datetime without guessing a naive timestamp zone.

    CPrecNet timestamp keys are KST, so callers parsing those naive keys must
    pass ``declared_radar_timezone="Asia/Seoul"``.  An already-aware datetime
    carries its own declared zone and is converted directly.
    """

    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        if declared_radar_timezone is None:
            raise ValueError("a naive timestamp requires declared_radar_timezone")
        timestamp = timestamp.replace(tzinfo=_timezone(declared_radar_timezone))
    return timestamp.astimezone(timezone.utc)


def _official_lead(lead_hours: float) -> float:
    if isinstance(lead_hours, bool):
        raise TypeError("lead_hours must be numeric, not boolean")
    try:
        lead = float(lead_hours)
    except (TypeError, ValueError) as error:
        raise TypeError("lead_hours must be a real number") from error
    doubled = lead * 2.0
    if (
        not math.isfinite(lead)
        or lead < 0.5
        or lead > 6.0
        or not math.isclose(doubled, round(doubled), abs_tol=1e-12)
    ):
        raise ValueError("lead_hours must be one of 0.5, 1.0, ..., 6.0")
    return lead


def _require_five_minute_slot(timestamp: datetime) -> None:
    if timestamp.minute % 5 or timestamp.second or timestamp.microsecond:
        raise ValueError("t0 must lie on a five-minute radar slot")


def verification_center_utc(
    t0: datetime,
    lead_hours: float,
    *,
    declared_radar_timezone: str | tzinfo | None = None,
) -> datetime:
    """Return ``t_c = t0 + tau - 15 minutes`` as an aware UTC datetime."""

    t0_utc = to_utc(t0, declared_radar_timezone=declared_radar_timezone)
    _require_five_minute_slot(t0_utc)
    lead = _official_lead(lead_hours)
    return t0_utc + timedelta(hours=lead, minutes=-15)


def build_verification_time_features(
    t0: datetime,
    lead_hours: float,
    *,
    declared_radar_timezone: str | tzinfo | None = None,
) -> VerificationTimeFeatures:
    """Build the frozen mean-solar-hour and annual-phase feature contract.

    Equation-of-time correction is deliberately absent.  ``mean_solar_time``
    retains a UTC ``tzinfo`` only to keep datetime arithmetic unambiguous; its
    displayed calendar fields represent the longitude-shifted mean-solar
    clock used for the cyclic features.
    """

    t0_utc = to_utc(t0, declared_radar_timezone=declared_radar_timezone)
    _require_five_minute_slot(t0_utc)
    lead = _official_lead(lead_hours)
    center = t0_utc + timedelta(hours=lead, minutes=-15)
    solar = center + timedelta(
        hours=TARGET_CENTER_LONGITUDE_DEGREES_EAST / 15.0
    )
    solar_hour = (
        solar.hour
        + solar.minute / 60.0
        + solar.second / 3600.0
        + solar.microsecond / 3_600_000_000.0
    )
    days_in_year = 366 if calendar.isleap(solar.year) else 365
    zero_based_day = solar.timetuple().tm_yday - 1
    annual_phase = (zero_based_day + solar_hour / 24.0) / days_in_year

    daily_angle = 2.0 * math.pi * solar_hour / 24.0
    annual_angle = 2.0 * math.pi * annual_phase
    cyclic = (
        math.sin(daily_angle),
        math.cos(daily_angle),
        math.sin(annual_angle),
        math.cos(annual_angle),
    )
    return VerificationTimeFeatures(
        t0_utc=t0_utc,
        verification_center_utc=center,
        mean_solar_time=solar,
        solar_hour=solar_hour,
        days_in_year=days_in_year,
        annual_phase=annual_phase,
        cyclic_features=cyclic,
    )


__all__ = [
    "OFFICIAL_LEAD_HOURS",
    "TARGET_CENTER_LONGITUDE_DEGREES_EAST",
    "VerificationTimeFeatures",
    "build_verification_time_features",
    "to_utc",
    "verification_center_utc",
]
