"""Production ecCodes decoder for the fixed K-CorrDiff ERA5 archive.

The Python ``eccodes`` package is imported lazily so importing K-CorrDiff and
running its synthetic tests do not require a native GRIB stack.  This module
owns all GRIB-specific identities and time-key interpretation; downstream
code only receives backend-neutral :class:`ProviderMessage` objects.

The archive contract is deliberately strict.  For each request group and
calendar month, every expected semantic field must occur exactly once at
every UTC hour.  Unexpected parameters, pressure levels, grids, scan order,
units, and step metadata are rejected before they can become model inputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import calendar
import importlib
from pathlib import Path
import re
from types import ModuleType
from typing import Iterator, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .provider_adapter import ERA5_CHANNELS, ProviderMessage


DEFAULT_ARCHIVE_YEARS = tuple(range(2020, 2026))
RequestGroup = Literal["pressure_core", "geopotential_500", "single_levels"]


@dataclass(frozen=True, slots=True)
class _ExpectedField:
    channel_name: str
    channel_index: int
    request_group: RequestGroup
    raw_short_name: str
    param_id: int
    type_of_level: str
    raw_level: int
    source_units: tuple[str, ...]
    accumulated: bool


_GROUP_BY_INDEX: tuple[RequestGroup, ...] = (
    *("pressure_core" for _ in range(16)),
    "geopotential_500",
    *("single_levels" for _ in range(7)),
)
_PARAM_IDS = (
    130,
    133,
    131,
    132,
    130,
    133,
    131,
    132,
    130,
    133,
    131,
    132,
    130,
    133,
    131,
    132,
    129,
    151,
    167,
    168,
    165,
    166,
    137,
    228,
)
_RAW_SHORT_NAMES = {
    "t2m": "2t",
    "d2m": "2d",
    "u10": "10u",
    "v10": "10v",
}


def _build_expected_fields() -> tuple[_ExpectedField, ...]:
    fields: list[_ExpectedField] = []
    for channel, group, param_id in zip(
        ERA5_CHANNELS, _GROUP_BY_INDEX, _PARAM_IDS, strict=True
    ):
        raw_short_name = _RAW_SHORT_NAMES.get(channel.name, channel.short_name)
        is_pressure = channel.level_hpa is not None
        fields.append(
            _ExpectedField(
                channel_name=channel.name,
                channel_index=channel.index,
                request_group=group,
                raw_short_name=raw_short_name,
                param_id=param_id,
                type_of_level="isobaricInhPa" if is_pressure else "surface",
                raw_level=int(channel.level_hpa or 0),
                source_units=channel.source_units,
                accumulated=channel.accumulated,
            )
        )
    identities = {
        (field.raw_short_name, field.param_id, field.type_of_level, field.raw_level)
        for field in fields
    }
    if len(fields) != 24 or len(identities) != 24:  # pragma: no cover
        raise AssertionError("ERA5 raw identities must map one-to-one to 24 channels")
    return tuple(fields)


EXPECTED_FIELDS = _build_expected_fields()
_FIELD_BY_RAW_IDENTITY = {
    (field.raw_short_name, field.param_id, field.type_of_level, field.raw_level): field
    for field in EXPECTED_FIELDS
}
_FIELDS_BY_GROUP: Mapping[RequestGroup, tuple[_ExpectedField, ...]] = {
    group: tuple(field for field in EXPECTED_FIELDS if field.request_group == group)
    for group in ("pressure_core", "geopotential_500", "single_levels")
}

_SOURCE_PATTERNS: Mapping[RequestGroup, re.Pattern[str]] = {
    "pressure_core": re.compile(r"era5_pl_tquv_(?P<year>\d{4})(?P<month>\d{2})\.grib"),
    "geopotential_500": re.compile(
        r"era5_pl_z500_(?P<year>\d{4})(?P<month>\d{2})\.grib"
    ),
    "single_levels": re.compile(r"era5_sl_(?P<year>\d{4})(?P<month>\d{2})\.grib"),
}


@dataclass(frozen=True, slots=True)
class Era5ArchiveSource:
    """One request-group GRIB file for one UTC calendar month."""

    path: Path
    request_group: RequestGroup
    year: int
    month: int

    @property
    def month_start(self) -> datetime:
        return datetime(self.year, self.month, 1, tzinfo=UTC)

    @property
    def month_end_exclusive(self) -> datetime:
        if self.month == 12:
            return datetime(self.year + 1, 1, 1, tzinfo=UTC)
        return datetime(self.year, self.month + 1, 1, tzinfo=UTC)

    @property
    def hours(self) -> int:
        return calendar.monthrange(self.year, self.month)[1] * 24


@dataclass(frozen=True, slots=True)
class Era5ArchiveAudit:
    """Compact, JSON-safe result of a complete metadata pass."""

    source_files: int
    calendar_months: int
    request_group_months: int
    messages: int
    instantaneous_messages: int
    tp_messages: int
    first_valid_time_utc: datetime
    last_valid_time_utc: datetime
    field_counts: tuple[tuple[str, int], ...]
    group_file_counts: tuple[tuple[str, int], ...]
    grid_shape: tuple[int, int]
    latitude_orientation: str
    latitude_bounds: tuple[float, float]
    longitude_bounds: tuple[float, float]

    def to_json(self) -> dict[str, object]:
        return {
            "source_files": self.source_files,
            "calendar_months": self.calendar_months,
            "request_group_months": self.request_group_months,
            "messages": self.messages,
            "instantaneous_messages": self.instantaneous_messages,
            "tp_messages": self.tp_messages,
            "first_valid_time_utc": self.first_valid_time_utc.isoformat(),
            "last_valid_time_utc": self.last_valid_time_utc.isoformat(),
            "field_counts": dict(self.field_counts),
            "group_file_counts": dict(self.group_file_counts),
            "grid_shape": list(self.grid_shape),
            "latitude_orientation": self.latitude_orientation,
            "latitude_bounds": list(self.latitude_bounds),
            "longitude_bounds": list(self.longitude_bounds),
            "tp_contract": {
                "step_type": "accum",
                "source_units": "m",
                "interval": "(valid_time - 1 hour, valid_time]",
            },
        }


def _validate_years(years: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(sorted(set(map(int, years))))
    if not selected:
        raise ValueError("at least one ERA5 archive year is required")
    if selected != tuple(range(selected[0], selected[-1] + 1)):
        raise ValueError("ERA5 archive years must form a contiguous interval")
    if selected[0] < 1900 or selected[-1] > 9998:
        raise ValueError("ERA5 archive year is outside the supported range")
    return selected


def discover_era5_archive(
    root: Path, *, years: Sequence[int] = DEFAULT_ARCHIVE_YEARS
) -> tuple[Era5ArchiveSource, ...]:
    """Resolve the exact three-file-per-month ERA5 archive layout.

    Unknown GRIB filenames in the three owned request-group directories are
    rejected.  Files for years outside ``years`` may coexist and are ignored.
    """

    root = root.resolve()
    selected_years = _validate_years(years)
    discovered: list[Era5ArchiveSource] = []
    expected_paths: set[Path] = set()
    for year in selected_years:
        for month in range(1, 13):
            stamp = f"{year:04d}{month:02d}"
            names: Mapping[RequestGroup, str] = {
                "pressure_core": f"era5_pl_tquv_{stamp}.grib",
                "geopotential_500": f"era5_pl_z500_{stamp}.grib",
                "single_levels": f"era5_sl_{stamp}.grib",
            }
            for group, name in names.items():
                path = root / group / name
                if not path.is_file():
                    raise FileNotFoundError(f"missing ERA5 archive source: {path}")
                expected_paths.add(path.resolve())
                discovered.append(
                    Era5ArchiveSource(
                        path=path.resolve(), request_group=group, year=year, month=month
                    )
                )

    for group, pattern in _SOURCE_PATTERNS.items():
        directory = root / group
        if not directory.is_dir():
            raise FileNotFoundError(f"missing ERA5 request-group directory: {directory}")
        for path in directory.glob("*.grib"):
            match = pattern.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected ERA5 GRIB filename: {path}")
            if int(match.group("year")) in selected_years and path.resolve() not in expected_paths:
                raise ValueError(f"unexpected ERA5 archive source: {path}")

    return tuple(discovered)


def _source_from_path(value: Path | str) -> Era5ArchiveSource:
    path = Path(value).resolve()
    try:
        group: RequestGroup = path.parent.name  # type: ignore[assignment]
        pattern = _SOURCE_PATTERNS[group]
    except KeyError as error:
        raise ValueError(f"cannot infer ERA5 request group from path: {path}") from error
    match = pattern.fullmatch(path.name)
    if match is None:
        raise ValueError(f"cannot infer ERA5 calendar month from path: {path}")
    return Era5ArchiveSource(
        path=path,
        request_group=group,
        year=int(match.group("year")),
        month=int(match.group("month")),
    )


def _prepare_sources(sources: Sequence[object]) -> tuple[Era5ArchiveSource, ...]:
    if not sources:
        raise ValueError("at least one ERA5 GRIB source is required")
    result: list[Era5ArchiveSource] = []
    seen: set[tuple[RequestGroup, int, int]] = set()
    for value in sources:
        if isinstance(value, Era5ArchiveSource):
            source = value
        elif isinstance(value, (Path, str)):
            source = _source_from_path(value)
        else:
            raise TypeError(f"unsupported ERA5 source descriptor: {type(value).__name__}")
        if source.request_group not in _FIELDS_BY_GROUP:
            raise ValueError(f"unsupported ERA5 request group: {source.request_group!r}")
        if not 1 <= source.month <= 12:
            raise ValueError(f"invalid ERA5 source month: {source.month}")
        key = (source.request_group, source.year, source.month)
        if key in seen:
            raise ValueError(
                "duplicate ERA5 request-group/month source: "
                f"{source.request_group} {source.year:04d}-{source.month:02d}"
            )
        if not source.path.is_file():
            raise FileNotFoundError(f"ERA5 GRIB source does not exist: {source.path}")
        seen.add(key)
        result.append(source)
    return tuple(result)


def _load_eccodes() -> ModuleType:
    try:
        return importlib.import_module("eccodes")
    except ImportError as error:
        raise RuntimeError(
            "ecCodes is required to read raw ERA5 GRIB; install the project's "
            "`data` extra in an ephemeral Pod or PVC-backed virtual environment"
        ) from error


def _parse_grib_datetime(date_value: object, time_value: object, *, key: str) -> datetime:
    try:
        date_integer = int(date_value)
        time_integer = int(time_value)
        year = date_integer // 10000
        month = (date_integer // 100) % 100
        day = date_integer % 100
        hour = time_integer // 100
        minute = time_integer % 100
        result = datetime(year, month, day, hour, minute, tzinfo=UTC)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid ecCodes {key} date/time: {date_value!r}/{time_value!r}"
        ) from error
    if minute != 0:
        raise ValueError(f"ecCodes {key} must lie on an exact UTC hour: {result}")
    return result


@dataclass(frozen=True, slots=True)
class _GridContract:
    shape: tuple[int, int] = (33, 33)
    north: float = 40.0
    south: float = 32.0
    west: float = 123.0
    east: float = 131.0
    spacing_degrees: float = 0.25

    @property
    def latitude(self) -> NDArray[np.float64]:
        return np.linspace(self.north, self.south, self.shape[0], dtype=np.float64)

    @property
    def longitude(self) -> NDArray[np.float64]:
        return np.linspace(self.west, self.east, self.shape[1], dtype=np.float64)


class EcCodesDecoder:
    """Decode and validate the production monthly ERA5 GRIB collection."""

    def __init__(
        self,
        *,
        eccodes_module: ModuleType | object | None = None,
        grid_shape: tuple[int, int] = (33, 33),
        north: float = 40.0,
        south: float = 32.0,
        west: float = 123.0,
        east: float = 131.0,
        spacing_degrees: float = 0.25,
    ) -> None:
        if len(grid_shape) != 2 or any(int(size) <= 1 for size in grid_shape):
            raise ValueError("grid_shape must contain two dimensions greater than one")
        if north <= south or east <= west or spacing_degrees <= 0.0:
            raise ValueError("ERA5 grid bounds/spacing are invalid")
        self._eccodes = eccodes_module
        self.grid = _GridContract(
            shape=tuple(map(int, grid_shape)),
            north=float(north),
            south=float(south),
            west=float(west),
            east=float(east),
            spacing_degrees=float(spacing_degrees),
        )
        expected_rows = round((north - south) / spacing_degrees) + 1
        expected_columns = round((east - west) / spacing_degrees) + 1
        if self.grid.shape != (expected_rows, expected_columns):
            raise ValueError("ERA5 grid bounds, spacing, and shape are inconsistent")
        self.last_audit: Era5ArchiveAudit | None = None

    @property
    def eccodes(self) -> object:
        if self._eccodes is None:
            self._eccodes = _load_eccodes()
        return self._eccodes

    def _get(self, handle: object, key: str, *, source: Era5ArchiveSource, number: int) -> object:
        try:
            return self.eccodes.codes_get(handle, key)  # type: ignore[attr-defined]
        except Exception as error:
            raise ValueError(
                f"{source.path}: message {number} lacks required ecCodes key {key!r}"
            ) from error

    def _validate_grid(
        self, handle: object, *, source: Era5ArchiveSource, number: int
    ) -> None:
        rows, columns = self.grid.shape
        expected: tuple[tuple[str, object], ...] = (
            ("gridType", "regular_ll"),
            ("Ni", columns),
            ("Nj", rows),
            ("numberOfPoints", rows * columns),
            ("iScansNegatively", 0),
            ("jScansPositively", 0),
            ("jPointsAreConsecutive", 0),
            ("alternativeRowScanning", 0),
            ("bitmapPresent", 0),
            ("numberOfMissing", 0),
        )
        for key, wanted in expected:
            actual = self._get(handle, key, source=source, number=number)
            if actual != wanted:
                raise ValueError(
                    f"{source.path}: message {number} has {key}={actual!r}, "
                    f"expected {wanted!r}"
                )
        expected_floats = (
            ("latitudeOfFirstGridPointInDegrees", self.grid.north),
            ("latitudeOfLastGridPointInDegrees", self.grid.south),
            ("longitudeOfFirstGridPointInDegrees", self.grid.west),
            ("longitudeOfLastGridPointInDegrees", self.grid.east),
            ("iDirectionIncrementInDegrees", self.grid.spacing_degrees),
            ("jDirectionIncrementInDegrees", self.grid.spacing_degrees),
        )
        for key, wanted in expected_floats:
            actual = float(self._get(handle, key, source=source, number=number))
            if not np.isclose(actual, wanted, rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"{source.path}: message {number} has {key}={actual!r}, "
                    f"expected {wanted!r}"
                )

    def _decode_handle(
        self,
        handle: object,
        *,
        source: Era5ArchiveSource,
        number: int,
        include_values: bool,
    ) -> tuple[_ExpectedField, datetime, ProviderMessage | None]:
        short_name = str(self._get(handle, "shortName", source=source, number=number))
        param_id = int(self._get(handle, "paramId", source=source, number=number))
        type_of_level = str(
            self._get(handle, "typeOfLevel", source=source, number=number)
        )
        raw_level = int(self._get(handle, "level", source=source, number=number))
        identity = (short_name, param_id, type_of_level, raw_level)
        try:
            field = _FIELD_BY_RAW_IDENTITY[identity]
        except KeyError as error:
            raise ValueError(
                f"{source.path}: message {number} has unexpected parameter identity "
                f"shortName={short_name!r}, paramId={param_id}, "
                f"typeOfLevel={type_of_level!r}, level={raw_level}"
            ) from error
        if field.request_group != source.request_group:
            raise ValueError(
                f"{source.path}: message {number} contains {field.channel_name}, "
                f"which belongs to {field.request_group}"
            )

        units = str(self._get(handle, "units", source=source, number=number))
        if units not in field.source_units:
            raise ValueError(
                f"{source.path}: message {number} has units={units!r} for "
                f"{field.channel_name}; expected one of {field.source_units}"
            )
        step_type = str(self._get(handle, "stepType", source=source, number=number))
        step_units = int(self._get(handle, "stepUnits", source=source, number=number))
        start_step = int(self._get(handle, "startStep", source=source, number=number))
        end_step = int(self._get(handle, "endStep", source=source, number=number))
        if step_units != 1:
            raise ValueError(
                f"{source.path}: message {number} stepUnits must be hours (code 1)"
            )
        reference_time = _parse_grib_datetime(
            self._get(handle, "dataDate", source=source, number=number),
            self._get(handle, "dataTime", source=source, number=number),
            key="reference time",
        )
        valid_time = _parse_grib_datetime(
            self._get(handle, "validityDate", source=source, number=number),
            self._get(handle, "validityTime", source=source, number=number),
            key="valid time",
        )
        computed_valid_time = reference_time + timedelta(hours=end_step)
        if computed_valid_time != valid_time:
            raise ValueError(
                f"{source.path}: message {number} validity time disagrees with "
                "reference time + endStep"
            )

        interval_start: datetime | None = None
        interval_end: datetime | None = None
        if field.accumulated:
            if step_type != "accum" or units != "m":
                raise ValueError(
                    f"{source.path}: message {number} tp must use stepType='accum' "
                    "and units='m'"
                )
            interval_start = reference_time + timedelta(hours=start_step)
            interval_end = computed_valid_time
            if end_step - start_step != 1 or interval_start != valid_time - timedelta(hours=1):
                raise ValueError(
                    f"{source.path}: message {number} tp must cover exactly "
                    "(valid_time - 1 hour, valid_time]"
                )
        elif step_type != "instant" or start_step != 0 or end_step != 0:
            raise ValueError(
                f"{source.path}: message {number} instantaneous field "
                f"{field.channel_name} has invalid step metadata"
            )

        self._validate_grid(handle, source=source, number=number)
        message: ProviderMessage | None = None
        if include_values:
            try:
                flat_values = np.asarray(
                    self.eccodes.codes_get_values(handle)  # type: ignore[attr-defined]
                )
            except Exception as error:
                raise ValueError(
                    f"{source.path}: message {number} values could not be decoded"
                ) from error
            if flat_values.size != int(np.prod(self.grid.shape)):
                raise ValueError(
                    f"{source.path}: message {number} decoded {flat_values.size} "
                    f"values, expected {int(np.prod(self.grid.shape))}"
                )
            values = flat_values.reshape(self.grid.shape)
            message = ProviderMessage(
                short_name=short_name,
                level_hpa=field.raw_level if field.type_of_level == "isobaricInhPa" else None,
                valid_time=valid_time,
                values=values,
                units=units,
                step_type=step_type,
                interval_start=interval_start,
                interval_end=interval_end,
                latitude=self.grid.latitude,
                longitude=self.grid.longitude,
                source=str(source.path),
                metadata={
                    "decoder": "eccodes",
                    "message_number": number,
                    "request_group": source.request_group,
                    "param_id": param_id,
                    "type_of_level": type_of_level,
                    "raw_level": raw_level,
                    "reference_time_utc": reference_time.isoformat(),
                    "start_step_hours": start_step,
                    "end_step_hours": end_step,
                },
            )
        return field, valid_time, message

    def _iterate(
        self, sources: Sequence[object], *, include_values: bool
    ) -> Iterator[ProviderMessage]:
        prepared = _prepare_sources(sources)
        self.last_audit = None
        field_counts: Counter[str] = Counter()
        group_counts: Counter[str] = Counter()
        total_messages = 0
        tp_messages = 0
        first_valid: datetime | None = None
        last_valid: datetime | None = None

        for source in prepared:
            fields = _FIELDS_BY_GROUP[source.request_group]
            field_offsets = {field.channel_index: index for index, field in enumerate(fields)}
            seen = np.zeros((source.hours, len(fields)), dtype=np.bool_)
            with source.path.open("rb") as stream:
                number = 0
                while True:
                    try:
                        handle = self.eccodes.codes_grib_new_from_file(stream)  # type: ignore[attr-defined]
                    except Exception as error:
                        raise ValueError(f"failed reading ERA5 GRIB: {source.path}") from error
                    if handle is None:
                        break
                    number += 1
                    try:
                        field, valid_time, message = self._decode_handle(
                            handle,
                            source=source,
                            number=number,
                            include_values=include_values,
                        )
                    finally:
                        self.eccodes.codes_release(handle)  # type: ignore[attr-defined]

                    if not source.month_start <= valid_time < source.month_end_exclusive:
                        raise ValueError(
                            f"{source.path}: message {number} valid_time "
                            f"{valid_time.isoformat()} is outside its calendar month"
                        )
                    hour_offset_float = (valid_time - source.month_start) / timedelta(hours=1)
                    hour_offset = int(hour_offset_float)
                    if hour_offset_float != hour_offset:
                        raise ValueError(
                            f"{source.path}: message {number} valid_time is not hourly"
                        )
                    field_offset = field_offsets[field.channel_index]
                    if seen[hour_offset, field_offset]:
                        raise ValueError(
                            f"{source.path}: duplicate semantic ERA5 message "
                            f"{field.channel_name}@{valid_time.isoformat()}"
                        )
                    seen[hour_offset, field_offset] = True
                    field_counts[field.channel_name] += 1
                    total_messages += 1
                    tp_messages += int(field.accumulated)
                    first_valid = valid_time if first_valid is None else min(first_valid, valid_time)
                    last_valid = valid_time if last_valid is None else max(last_valid, valid_time)
                    if message is not None:
                        yield message

            if not seen.all():
                missing_indices = np.argwhere(~seen)
                examples = [
                    f"{fields[field_index].channel_name}@"
                    f"{(source.month_start + timedelta(hours=int(hour_index))).isoformat()}"
                    for hour_index, field_index in missing_indices[:8]
                ]
                raise ValueError(
                    f"{source.path}: monthly hourly completeness failed; "
                    f"missing {len(missing_indices)} semantic messages "
                    f"(examples: {', '.join(examples)})"
                )
            group_counts[source.request_group] += 1

        if first_valid is None or last_valid is None:  # pragma: no cover - non-empty files required
            raise ValueError("ERA5 archive emitted no GRIB messages")
        self.last_audit = Era5ArchiveAudit(
            source_files=len(prepared),
            calendar_months=len({(source.year, source.month) for source in prepared}),
            request_group_months=len(prepared),
            messages=total_messages,
            instantaneous_messages=total_messages - tp_messages,
            tp_messages=tp_messages,
            first_valid_time_utc=first_valid,
            last_valid_time_utc=last_valid,
            field_counts=tuple(sorted(field_counts.items())),
            group_file_counts=tuple(sorted(group_counts.items())),
            grid_shape=self.grid.shape,
            latitude_orientation="north_to_south",
            latitude_bounds=(self.grid.north, self.grid.south),
            longitude_bounds=(self.grid.west, self.grid.east),
        )

    def iter_messages(self, sources: Sequence[object]) -> Iterator[ProviderMessage]:
        """Yield values after strict per-message and per-month validation."""

        yield from self._iterate(sources, include_values=True)

    def validate_sources(self, sources: Sequence[object]) -> Era5ArchiveAudit:
        """Run a metadata-only validation pass over every GRIB message."""

        for _ in self._iterate(sources, include_values=False):  # pragma: no branch
            pass
        if self.last_audit is None:  # pragma: no cover - assigned on successful exhaustion
            raise AssertionError("ERA5 audit did not complete")
        return self.last_audit


__all__ = [
    "DEFAULT_ARCHIVE_YEARS",
    "EXPECTED_FIELDS",
    "EcCodesDecoder",
    "Era5ArchiveAudit",
    "Era5ArchiveSource",
    "RequestGroup",
    "discover_era5_archive",
]
