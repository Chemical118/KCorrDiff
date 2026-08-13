from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from kcorrdiff.data.eccodes_decoder import (
    EXPECTED_FIELDS,
    EcCodesDecoder,
    Era5ArchiveSource,
    discover_era5_archive,
)
from kcorrdiff.data.provider_adapter import Era5ProviderAdapter


class FakeEcCodes:
    """Tiny in-memory stand-in for the ecCodes functions used by the decoder."""

    def __init__(self, records_by_path: dict[Path, list[dict[str, object]]]) -> None:
        self.records_by_path = {
            str(path.resolve()): records for path, records in records_by_path.items()
        }
        self.indices: defaultdict[object, int] = defaultdict(int)
        self.released = 0

    def codes_grib_new_from_file(self, stream: object) -> dict[str, object] | None:
        records = self.records_by_path[str(Path(stream.name).resolve())]  # type: ignore[attr-defined]
        index = self.indices[stream]
        if index == len(records):
            return None
        self.indices[stream] += 1
        return records[index]

    def codes_get(self, handle: dict[str, object], key: str) -> object:
        return handle[key]

    def codes_get_values(self, handle: dict[str, object]) -> np.ndarray:
        return np.asarray(handle["values"])

    def codes_release(self, handle: dict[str, object]) -> None:
        self.released += 1


def _date_and_time(value: datetime) -> tuple[int, int]:
    return int(value.strftime("%Y%m%d")), int(value.strftime("%H%M"))


def _record(
    channel_name: str,
    valid_time: datetime,
    *,
    shape: tuple[int, int] = (2, 3),
    changes: dict[str, object] | None = None,
) -> dict[str, object]:
    field = next(field for field in EXPECTED_FIELDS if field.channel_name == channel_name)
    if field.accumulated:
        reference = valid_time - timedelta(hours=1)
        start_step, end_step, step_type = 0, 1, "accum"
    else:
        reference = valid_time
        start_step, end_step, step_type = 0, 0, "instant"
    data_date, data_time = _date_and_time(reference)
    validity_date, validity_time = _date_and_time(valid_time)
    result: dict[str, object] = {
        "shortName": field.raw_short_name,
        "paramId": field.param_id,
        "typeOfLevel": field.type_of_level,
        "level": field.raw_level,
        "units": field.source_units[-1],
        "stepType": step_type,
        "stepUnits": 1,
        "startStep": start_step,
        "endStep": end_step,
        "dataDate": data_date,
        "dataTime": data_time,
        "validityDate": validity_date,
        "validityTime": validity_time,
        "gridType": "regular_ll",
        "Ni": shape[1],
        "Nj": shape[0],
        "numberOfPoints": shape[0] * shape[1],
        "iScansNegatively": 0,
        "jScansPositively": 0,
        "jPointsAreConsecutive": 0,
        "alternativeRowScanning": 0,
        "bitmapPresent": 0,
        "numberOfMissing": 0,
        "latitudeOfFirstGridPointInDegrees": 40.0,
        "latitudeOfLastGridPointInDegrees": 39.0,
        "longitudeOfFirstGridPointInDegrees": 123.0,
        "longitudeOfLastGridPointInDegrees": 125.0,
        "iDirectionIncrementInDegrees": 1.0,
        "jDirectionIncrementInDegrees": 1.0,
        "values": np.arange(shape[0] * shape[1], dtype=np.float64),
    }
    if changes:
        result.update(changes)
    return result


def _source(tmp_path: Path, group: str = "geopotential_500") -> Era5ArchiveSource:
    path = tmp_path / f"{group}.grib"
    path.touch()
    return Era5ArchiveSource(
        path=path,
        request_group=group,  # type: ignore[arg-type]
        year=2020,
        month=1,
    )


def _decoder(path: Path, records: list[dict[str, object]]) -> EcCodesDecoder:
    return EcCodesDecoder(
        eccodes_module=FakeEcCodes({path: records}),
        grid_shape=(2, 3),
        north=40.0,
        south=39.0,
        west=123.0,
        east=125.0,
        spacing_degrees=1.0,
    )


def test_discover_archive_requires_exact_three_files_per_month(tmp_path: Path) -> None:
    for group in ("pressure_core", "geopotential_500", "single_levels"):
        (tmp_path / group).mkdir()
    for month in range(1, 13):
        stamp = f"2020{month:02d}"
        (tmp_path / "pressure_core" / f"era5_pl_tquv_{stamp}.grib").touch()
        (tmp_path / "geopotential_500" / f"era5_pl_z500_{stamp}.grib").touch()
        (tmp_path / "single_levels" / f"era5_sl_{stamp}.grib").touch()

    sources = discover_era5_archive(tmp_path, years=[2020])
    assert len(sources) == 36
    assert {(source.year, source.month) for source in sources} == {
        (2020, month) for month in range(1, 13)
    }
    (tmp_path / "single_levels" / "era5_sl_202006.grib").unlink()
    with pytest.raises(FileNotFoundError, match="202006"):
        discover_era5_archive(tmp_path, years=[2020])


def test_metadata_audit_proves_hourly_month_completeness(tmp_path: Path) -> None:
    source = _source(tmp_path)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    records = [_record("z500", start + timedelta(hours=index)) for index in range(744)]
    decoder = _decoder(source.path, records)

    audit = decoder.validate_sources([source])

    assert audit.source_files == 1
    assert audit.messages == 744
    assert audit.field_counts == (("z500", 744),)
    assert audit.first_valid_time_utc == start
    assert audit.last_valid_time_utc == datetime(2020, 1, 31, 23, tzinfo=UTC)
    assert audit.latitude_orientation == "north_to_south"


@pytest.mark.parametrize("failure", ["missing", "duplicate"])
def test_metadata_audit_rejects_missing_or_duplicate_semantic_hour(
    tmp_path: Path, failure: str
) -> None:
    source = _source(tmp_path)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    records = [_record("z500", start + timedelta(hours=index)) for index in range(744)]
    if failure == "missing":
        records.pop()
        match = "monthly hourly completeness"
    else:
        records.append(records[0])
        match = "duplicate semantic"
    with pytest.raises(ValueError, match=match):
        _decoder(source.path, records).validate_sources([source])


def test_tp_decoding_exposes_exact_interval_and_adapter_converts_units(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, "single_levels")
    valid_time = datetime(2020, 1, 1, tzinfo=UTC)
    decoder = _decoder(source.path, [_record("tp", valid_time)])
    messages = decoder.iter_messages([source])
    message = next(messages)
    messages.close()

    assert message.valid_time == valid_time
    assert message.interval_start == valid_time - timedelta(hours=1)
    assert message.interval_end == valid_time
    assert message.units == "m"
    assert message.metadata["param_id"] == 228
    field = Era5ProviderAdapter(grid_shape=(2, 3)).canonicalize(message)
    assert field.channel.name == "tp"
    assert field.values.dtype == np.float32
    assert np.array_equal(field.latitude, [39.0, 40.0])
    assert np.array_equal(field.values, np.arange(6).reshape(2, 3)[::-1] * 1000.0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"stepType": "instant"}, "stepType='accum'"),
        ({"units": "mm"}, "units"),
        ({"endStep": 2, "validityTime": 100}, "exactly"),
        ({"paramId": 999}, "unexpected parameter identity"),
        ({"jScansPositively": 1}, "jScansPositively"),
        ({"latitudeOfFirstGridPointInDegrees": 39.5}, "latitudeOfFirst"),
    ],
)
def test_decoder_rejects_tp_identity_time_and_grid_contract_violations(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    source = _source(tmp_path, "single_levels")
    record = _record("tp", datetime(2020, 1, 1, tzinfo=UTC), changes=changes)
    with pytest.raises(ValueError, match=match):
        next(_decoder(source.path, [record]).iter_messages([source]))


def test_duplicate_request_group_month_source_is_rejected_before_reading(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    decoder = _decoder(source.path, [])
    with pytest.raises(ValueError, match="duplicate ERA5 request-group/month"):
        decoder.validate_sources([source, source])
