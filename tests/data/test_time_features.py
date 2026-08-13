from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from kcorrdiff.data.time_features import (
    TARGET_CENTER_LONGITUDE_DEGREES_EAST,
    build_verification_time_features,
    to_utc,
    verification_center_utc,
)


def test_naive_cprecnet_time_requires_and_uses_declared_kst() -> None:
    timestamp = datetime(2022, 1, 2, 2, 0)

    with pytest.raises(ValueError, match="declared_radar_timezone"):
        to_utc(timestamp)

    converted = to_utc(timestamp, declared_radar_timezone="Asia/Seoul")
    assert converted == datetime(2022, 1, 1, 17, 0, tzinfo=timezone.utc)


def test_aware_kst_and_utc_instants_produce_identical_features() -> None:
    kst = datetime(2022, 8, 8, 12, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    utc = datetime(2022, 8, 8, 3, 0, tzinfo=timezone.utc)

    from_kst = build_verification_time_features(kst, 6.0)
    from_utc = build_verification_time_features(utc, 6.0)

    assert from_kst == from_utc


def test_verification_center_is_middle_of_target_interval() -> None:
    t0 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)

    assert verification_center_utc(t0, 0.5) == datetime(
        2022, 1, 1, 0, 15, tzinfo=timezone.utc
    )
    assert verification_center_utc(t0, 6.0) == datetime(
        2022, 1, 1, 5, 45, tzinfo=timezone.utc
    )


def test_t0_must_be_a_five_minute_radar_slot() -> None:
    off_slot = datetime(2022, 1, 1, 0, 1, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="five-minute"):
        build_verification_time_features(off_slot, 0.5)


def test_mean_solar_hour_and_non_leap_annual_phase_follow_contract() -> None:
    t0 = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)

    features = build_verification_time_features(t0, 0.5)

    expected_hour = 0.25 + TARGET_CENTER_LONGITUDE_DEGREES_EAST / 15.0
    assert features.solar_hour == pytest.approx(expected_hour, abs=3e-10)
    assert features.days_in_year == 365
    assert features.annual_phase == pytest.approx(
        (expected_hour / 24.0) / 365.0, abs=1e-12
    )
    daily_angle = 2.0 * math.pi * expected_hour / 24.0
    annual_angle = 2.0 * math.pi * features.annual_phase
    assert features.cyclic_features == pytest.approx(
        (
            math.sin(daily_angle),
            math.cos(daily_angle),
            math.sin(annual_angle),
            math.cos(annual_angle),
        )
    )


def test_annual_phase_uses_solar_calendar_and_leap_year() -> None:
    # Center is 2024-02-28 15:45 UTC; longitude shift crosses into Feb 29.
    t0 = datetime(2024, 2, 28, 15, 30, tzinfo=timezone.utc)

    features = build_verification_time_features(t0, 0.5)

    assert features.mean_solar_time.date().isoformat() == "2024-02-29"
    assert features.days_in_year == 366
    expected = (59 + features.solar_hour / 24.0) / 366.0
    assert features.annual_phase == pytest.approx(expected)


def test_solar_year_rollover_selects_new_year_length_and_day_zero() -> None:
    t0 = datetime(2023, 12, 31, 15, 45, tzinfo=timezone.utc)

    features = build_verification_time_features(t0, 0.5)

    assert features.mean_solar_time.year == 2024
    assert features.mean_solar_time.timetuple().tm_yday == 1
    assert features.days_in_year == 366
    assert features.annual_phase == pytest.approx(
        (features.solar_hour / 24.0) / 366.0
    )


@pytest.mark.parametrize("lead", [0.0, 0.75, 6.5, float("nan"), True])
def test_only_twelve_official_half_hour_leads_are_accepted(lead: object) -> None:
    t0 = datetime(2022, 1, 1, tzinfo=timezone.utc)
    with pytest.raises((TypeError, ValueError)):
        build_verification_time_features(t0, lead)  # type: ignore[arg-type]


def test_each_cyclic_pair_lies_on_unit_circle() -> None:
    features = build_verification_time_features(
        datetime(2022, 7, 1, 9, 25, tzinfo=timezone.utc), 4.5
    )
    daily_sin, daily_cos, annual_sin, annual_cos = features.cyclic_features

    assert daily_sin**2 + daily_cos**2 == pytest.approx(1.0)
    assert annual_sin**2 + annual_cos**2 == pytest.approx(1.0)
