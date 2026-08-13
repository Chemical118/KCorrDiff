from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kcorrdiff.data.provider_adapter import (
    ERA5_CHANNEL_NAMES,
    Era5ProviderAdapter,
    ProviderMessage,
    schema_manifest,
)


LATITUDE_NORTH_TO_SOUTH = np.asarray([40.0, 39.0])
LONGITUDE_WEST_TO_EAST = np.asarray([123.0, 124.0, 125.0])
VALID_TIME = datetime(2022, 1, 1, 3, tzinfo=UTC)


def message(**changes: object) -> ProviderMessage:
    values: dict[str, object] = {
        "short_name": "t",
        "level_hpa": 925,
        "valid_time": VALID_TIME,
        "values": np.arange(6, dtype=np.float64).reshape(2, 3),
        "units": "K",
        "step_type": "instant",
        "latitude": LATITUDE_NORTH_TO_SOUTH,
        "longitude": LONGITUDE_WEST_TO_EAST,
        "source": "synthetic.grib",
        "metadata": {"message": 1},
    }
    values.update(changes)
    return ProviderMessage(**values)  # type: ignore[arg-type]


def test_frozen_schema_has_exact_24_channel_order() -> None:
    assert ERA5_CHANNEL_NAMES == (
        "t925", "q925", "u925", "v925",
        "t850", "q850", "u850", "v850",
        "t700", "q700", "u700", "v700",
        "t500", "q500", "u500", "v500",
        "z500", "msl", "t2m", "d2m", "u10", "v10", "tcwv", "tp",
    )
    manifest = schema_manifest()
    assert [entry["index"] for entry in manifest] == list(range(24))
    assert manifest[-1]["accumulated"] is True
    assert manifest[-1]["canonical_units"] == "mm per 1 hour"


def test_instantaneous_field_is_float32_and_south_to_north() -> None:
    adapter = Era5ProviderAdapter(grid_shape=(2, 3))
    field = adapter.canonicalize(message())
    assert field.channel.name == "t925"
    assert field.values.dtype == np.float32
    assert np.array_equal(field.values, np.arange(6).reshape(2, 3)[::-1])
    assert np.array_equal(field.latitude, [39.0, 40.0])
    assert np.array_equal(field.longitude, LONGITUDE_WEST_TO_EAST)
    assert field.provenance.valid_time_utc == VALID_TIME
    assert field.provenance.source == "synthetic.grib"
    assert field.provenance.decoder_metadata == {"message": 1}
    assert "latitude_axis_flipped_north_to_south" in field.provenance.transformations


def test_tp_is_strict_one_hour_interval_and_converted_to_mm() -> None:
    adapter = Era5ProviderAdapter(grid_shape=(2, 3))
    field = adapter.canonicalize(
        message(
            short_name="tp",
            level_hpa=None,
            values=np.full((2, 3), 0.0015, dtype=np.float64),
            units="m",
            step_type="accum",
            interval_start=VALID_TIME - timedelta(hours=1),
            interval_end=VALID_TIME,
        )
    )
    assert field.channel.index == 23
    assert field.values.dtype == np.float32
    assert np.allclose(field.values, 1.5)
    assert field.provenance.interval_start_utc == VALID_TIME - timedelta(hours=1)
    assert field.provenance.interval_end_utc == VALID_TIME
    assert field.provenance.canonical_units == "mm per 1 hour"
    assert "metres_to_millimetres_x1000" in field.provenance.transformations


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"step_type": "instant"}, "step_type"),
        ({"units": "mm"}, "metres"),
        ({"interval_start": VALID_TIME - timedelta(hours=2)}, "exactly"),
        ({"interval_end": VALID_TIME + timedelta(hours=1)}, "exactly"),
        ({"values": np.full((2, 3), -0.001)}, "negative"),
    ],
)
def test_tp_rejects_semantic_contract_violations(
    changes: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "short_name": "tp",
        "level_hpa": None,
        "values": np.full((2, 3), 0.001),
        "units": "m",
        "step_type": "accum",
        "interval_start": VALID_TIME - timedelta(hours=1),
        "interval_end": VALID_TIME,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=match):
        Era5ProviderAdapter(grid_shape=(2, 3)).canonicalize(message(**values))


def test_adapter_rejects_ambiguous_time_grid_and_channel_metadata() -> None:
    adapter = Era5ProviderAdapter(grid_shape=(2, 3))
    with pytest.raises(ValueError, match="timezone-aware"):
        adapter.canonicalize(message(valid_time=VALID_TIME.replace(tzinfo=None)))
    with pytest.raises(ValueError, match="native hour"):
        adapter.canonicalize(message(valid_time=VALID_TIME.replace(minute=30)))
    with pytest.raises(ValueError, match="pressure field"):
        adapter.canonicalize(message(level_hpa=300))
    with pytest.raises(ValueError, match="west-to-east"):
        adapter.canonicalize(message(longitude=LONGITUDE_WEST_TO_EAST[::-1]))
    with pytest.raises(ValueError, match="step_type"):
        adapter.canonicalize(message(step_type="accum"))


@pytest.mark.parametrize(
    ("source_short_name", "canonical_name", "units"),
    [
        ("2t", "t2m", "K"),
        ("2d", "d2m", "K"),
        ("10u", "u10", "m s**-1"),
        ("10v", "v10", "m s**-1"),
    ],
)
def test_eccodes_surface_short_name_aliases_are_canonicalized(
    source_short_name: str, canonical_name: str, units: str
) -> None:
    field = Era5ProviderAdapter(grid_shape=(2, 3)).canonicalize(
        message(short_name=source_short_name, level_hpa=None, units=units)
    )
    assert field.channel.name == canonical_name
    assert field.provenance.source_short_name == source_short_name
