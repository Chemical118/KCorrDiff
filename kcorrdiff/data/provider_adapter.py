"""Canonical provider boundary for the 24-channel ERA5 oracle input.

GRIB decoding is intentionally outside this module.  A decoder translates its
backend-specific records (ecCodes, cfgrib, or a test double) into
:class:`ProviderMessage` objects.  The adapter below is then the single place
that enforces channel identity, units, precipitation interval semantics, UTC
valid times, grid orientation, and provenance.

The canonical channel order is frozen by K-CorrDiff protocol v1.1.3b.  The
first 23 fields are instantaneous; ``tp`` is a separate one-hour accumulated
amount and is converted from metres to millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Iterable, Iterator, Literal, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


ERA5_PROVIDER_ID = "era5"
ERA5_PROVIDER_VERSION = "cds-era5-reanalysis"
ERA5_DATASET = "Copernicus Climate Data Store ERA5 reanalysis"
CANONICAL_LATITUDE_ORIENTATION = "south_to_north"
CANONICAL_LONGITUDE_ORIENTATION = "west_to_east"
DEFAULT_GRID_SHAPE = (33, 33)

TemporalAccess = Literal["full_trajectory", "target_end_causal", "no_era_access"]


@dataclass(frozen=True, slots=True)
class ConditionSignature:
    """Canonical source/access identity shared by regression and diffusion."""

    provider_track: str
    era_present: bool
    tp_present: bool
    temporal_access_mode: TemporalAccess

    def canonical(self) -> "ConditionSignature":
        if not self.era_present:
            return ConditionSignature(
                provider_track="null_provider",
                era_present=False,
                tp_present=False,
                temporal_access_mode="no_era_access",
            )
        if self.temporal_access_mode == "no_era_access":
            raise ValueError("ERA-present signature cannot use no_era_access")
        return self

    @property
    def key(self) -> str:
        value = self.canonical()
        return ":".join(
            [
                value.provider_track,
                f"era={int(value.era_present)}",
                f"tp={int(value.tp_present)}",
                value.temporal_access_mode,
            ]
        )


ERA5_FULL_TRAJECTORY = ConditionSignature(
    provider_track="era5_oracle",
    era_present=True,
    tp_present=True,
    temporal_access_mode="full_trajectory",
)


@dataclass(frozen=True, slots=True)
class Era5Channel:
    """One field in the frozen 24-channel semantic schema."""

    index: int
    name: str
    short_name: str
    level_hpa: int | None
    canonical_units: str
    source_units: tuple[str, ...]
    accumulated: bool = False


def _build_schema() -> tuple[Era5Channel, ...]:
    channels: list[Era5Channel] = []
    pressure_variables = (
        ("t", "K", ("K",)),
        ("q", "kg kg-1", ("kg kg-1", "kg kg**-1")),
        ("u", "m s-1", ("m s-1", "m s**-1")),
        ("v", "m s-1", ("m s-1", "m s**-1")),
    )
    for level in (925, 850, 700, 500):
        for short_name, units, source_units in pressure_variables:
            channels.append(
                Era5Channel(
                    index=len(channels),
                    name=f"{short_name}{level}",
                    short_name=short_name,
                    level_hpa=level,
                    canonical_units=units,
                    source_units=source_units,
                )
            )
    channels.append(
        Era5Channel(
            index=len(channels),
            name="z500",
            short_name="z",
            level_hpa=500,
            canonical_units="m2 s-2",
            source_units=("m2 s-2", "m**2 s**-2"),
        )
    )
    for name, units, source_units in (
        ("msl", "Pa", ("Pa",)),
        ("t2m", "K", ("K",)),
        ("d2m", "K", ("K",)),
        ("u10", "m s-1", ("m s-1", "m s**-1")),
        ("v10", "m s-1", ("m s-1", "m s**-1")),
        ("tcwv", "kg m-2", ("kg m-2", "kg m**-2")),
    ):
        channels.append(
            Era5Channel(
                index=len(channels),
                name=name,
                short_name=name,
                level_hpa=None,
                canonical_units=units,
                source_units=source_units,
            )
        )
    channels.append(
        Era5Channel(
            index=len(channels),
            name="tp",
            short_name="tp",
            level_hpa=None,
            canonical_units="mm per 1 hour",
            source_units=("m",),
            accumulated=True,
        )
    )
    if len(channels) != 24 or channels[-1].index != 23:  # pragma: no cover
        raise AssertionError("the ERA5 schema must contain exactly 24 channels")
    return tuple(channels)


ERA5_CHANNELS = _build_schema()
ERA5_INSTANTANEOUS_CHANNELS = ERA5_CHANNELS[:-1]
ERA5_TP_CHANNEL = ERA5_CHANNELS[-1]
ERA5_CHANNEL_NAMES = tuple(channel.name for channel in ERA5_CHANNELS)

_PRESSURE_LOOKUP = {
    (channel.short_name, channel.level_hpa): channel
    for channel in ERA5_CHANNELS
    if channel.level_hpa is not None
}
_SURFACE_LOOKUP = {
    channel.short_name: channel
    for channel in ERA5_CHANNELS
    if channel.level_hpa is None
}
# ecCodes GRIB short names differ from the canonical model-facing names for
# height-specific surface fields.  Accept both spellings, but provenance keeps
# the exact source spelling and the cache manifest keeps the canonical order.
_SURFACE_LOOKUP.update(
    {
        "2t": _SURFACE_LOOKUP["t2m"],
        "2d": _SURFACE_LOOKUP["d2m"],
        "10u": _SURFACE_LOOKUP["u10"],
        "10v": _SURFACE_LOOKUP["v10"],
    }
)


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    """Backend-neutral representation of one decoded provider field.

    ``valid_time`` and interval endpoints must be timezone-aware.  Latitude
    may be a one-dimensional row coordinate or a two-dimensional grid;
    longitude follows the analogous column convention.  A decoder must emit
    one message for one field and one valid time.
    """

    short_name: str
    valid_time: datetime
    values: NDArray[np.generic]
    units: str
    step_type: str
    latitude: NDArray[np.generic]
    longitude: NDArray[np.generic]
    level_hpa: int | None = None
    interval_start: datetime | None = None
    interval_end: datetime | None = None
    source: str = "unknown"
    metadata: Mapping[str, object] = field(default_factory=dict)


class MessageDecoder(Protocol):
    """Pluggable raw-data decoder used by the mmap cache builder."""

    def iter_messages(self, sources: Sequence[object]) -> Iterable[ProviderMessage]:
        """Yield decoded messages from ``sources`` in any order."""


@dataclass(frozen=True, slots=True)
class FieldProvenance:
    provider_id: str
    provider_version: str
    dataset: str
    source: str
    source_short_name: str
    source_units: str
    canonical_units: str
    valid_time_utc: datetime
    interval_start_utc: datetime | None
    interval_end_utc: datetime | None
    source_latitude_orientation: str
    canonical_latitude_orientation: str
    transformations: tuple[str, ...]
    decoder_metadata: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CanonicalField:
    channel: Era5Channel
    valid_time_utc: datetime
    values: NDArray[np.float32]
    latitude: NDArray[np.float64]
    longitude: NDArray[np.float64]
    provenance: FieldProvenance


def _aware_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _axis_orientation(values: NDArray[np.float64], *, axis: int, name: str) -> str:
    if values.ndim == 1:
        vector = values
    elif values.ndim == 2:
        vector = np.mean(values, axis=1 - axis)
    else:
        raise ValueError(f"{name} must be a one- or two-dimensional coordinate")
    differences = np.diff(vector)
    if np.all(differences > 0.0):
        return "increasing"
    if np.all(differences < 0.0):
        return "decreasing"
    raise ValueError(f"{name} is not strictly monotonic")


def _orient_grid(
    values: NDArray[np.float32],
    latitude: NDArray[np.generic],
    longitude: NDArray[np.generic],
) -> tuple[NDArray[np.float32], NDArray[np.float64], NDArray[np.float64], str]:
    rows, columns = values.shape
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    if lat.shape not in {(rows,), (rows, columns)}:
        raise ValueError(
            f"latitude has shape {lat.shape}, expected {(rows,)} or {(rows, columns)}"
        )
    if lon.shape not in {(columns,), (rows, columns)}:
        raise ValueError(
            f"longitude has shape {lon.shape}, expected {(columns,)} or {(rows, columns)}"
        )
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("latitude/longitude contains non-finite values")

    latitude_direction = _axis_orientation(lat, axis=0, name="latitude")
    longitude_direction = _axis_orientation(lon, axis=1, name="longitude")
    if longitude_direction != "increasing":
        raise ValueError("longitude must be west-to-east")

    source_orientation = (
        "south_to_north" if latitude_direction == "increasing" else "north_to_south"
    )
    if latitude_direction == "decreasing":
        values = values[::-1, :]
        lat = lat[::-1] if lat.ndim == 1 else lat[::-1, :]
        if lon.ndim == 2:
            lon = lon[::-1, :]
    return (
        np.ascontiguousarray(values, dtype=np.float32),
        np.ascontiguousarray(lat, dtype=np.float64),
        np.ascontiguousarray(lon, dtype=np.float64),
        source_orientation,
    )


class Era5ProviderAdapter:
    """Validate and transform decoded ERA5 messages into canonical fields."""

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE,
        provider_id: str = ERA5_PROVIDER_ID,
        provider_version: str = ERA5_PROVIDER_VERSION,
        dataset: str = ERA5_DATASET,
    ) -> None:
        if len(grid_shape) != 2 or any(size <= 1 for size in grid_shape):
            raise ValueError("grid_shape must contain two dimensions greater than one")
        self.grid_shape = tuple(map(int, grid_shape))
        self.provider_id = provider_id
        self.provider_version = provider_version
        self.dataset = dataset

    def identify_channel(self, message: ProviderMessage) -> Era5Channel:
        short_name = message.short_name.strip().lower()
        if short_name in {"t", "q", "u", "v", "z"}:
            try:
                return _PRESSURE_LOOKUP[(short_name, message.level_hpa)]
            except KeyError as error:
                raise ValueError(
                    f"unsupported ERA5 pressure field: {short_name}@{message.level_hpa} hPa"
                ) from error
        try:
            channel = _SURFACE_LOOKUP[short_name]
        except KeyError as error:
            raise ValueError(f"unsupported ERA5 field: {message.short_name!r}") from error
        if message.level_hpa not in {None, 0}:
            raise ValueError(
                f"surface field {short_name} must not carry a pressure level, "
                f"got {message.level_hpa}"
            )
        return channel

    def canonicalize(self, message: ProviderMessage) -> CanonicalField:
        channel = self.identify_channel(message)
        valid_time = _aware_utc(message.valid_time, name="valid_time")
        if valid_time.minute or valid_time.second or valid_time.microsecond:
            raise ValueError("ERA5 valid_time must lie on an exact native hour")

        raw = np.asarray(message.values)
        if raw.shape != self.grid_shape:
            raise ValueError(
                f"ERA5 field has shape {raw.shape}, expected {self.grid_shape}"
            )
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise TypeError("ERA5 values must be a real numeric array")
        values = np.asarray(raw, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("ERA5 field contains non-finite values")
        values, latitude, longitude, source_orientation = _orient_grid(
            values, message.latitude, message.longitude
        )

        transformations: list[str] = []
        interval_start: datetime | None = None
        interval_end: datetime | None = None
        if channel.accumulated:
            if message.step_type != "accum":
                raise ValueError("ERA5 tp must have step_type='accum'")
            if message.units not in channel.source_units:
                raise ValueError(
                    f"ERA5 tp source units must be metres, got {message.units!r}"
                )
            if message.interval_start is None or message.interval_end is None:
                raise ValueError("ERA5 tp requires explicit interval_start/interval_end")
            interval_start = _aware_utc(message.interval_start, name="interval_start")
            interval_end = _aware_utc(message.interval_end, name="interval_end")
            expected_start = valid_time - timedelta(hours=1)
            if interval_end != valid_time or interval_start != expected_start:
                raise ValueError(
                    "ERA5 tp must represent exactly (valid_time - 1 h, valid_time]"
                )
            if np.any(values < 0.0):
                raise ValueError("ERA5 tp contains a negative accumulated amount")
            values = np.ascontiguousarray(values * np.float32(1000.0))
            transformations.append("metres_to_millimetres_x1000")
        else:
            if message.step_type != "instant":
                raise ValueError(
                    f"instantaneous ERA5 field {channel.name} must have step_type='instant'"
                )
            if message.interval_start is not None or message.interval_end is not None:
                raise ValueError(
                    f"instantaneous ERA5 field {channel.name} must not have an interval"
                )
            if message.units not in channel.source_units:
                raise ValueError(
                    f"unexpected units for ERA5 {channel.name}: {message.units!r}; "
                    f"expected one of {channel.source_units}"
                )

        if source_orientation != CANONICAL_LATITUDE_ORIENTATION:
            transformations.append("latitude_axis_flipped_north_to_south")
        provenance = FieldProvenance(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            dataset=self.dataset,
            source=message.source,
            source_short_name=message.short_name,
            source_units=message.units,
            canonical_units=channel.canonical_units,
            valid_time_utc=valid_time,
            interval_start_utc=interval_start,
            interval_end_utc=interval_end,
            source_latitude_orientation=source_orientation,
            canonical_latitude_orientation=CANONICAL_LATITUDE_ORIENTATION,
            transformations=tuple(transformations),
            decoder_metadata=dict(message.metadata),
        )
        return CanonicalField(
            channel=channel,
            valid_time_utc=valid_time,
            values=values,
            latitude=latitude,
            longitude=longitude,
            provenance=provenance,
        )

    def iter_canonical(
        self, decoder: MessageDecoder, sources: Sequence[object]
    ) -> Iterator[CanonicalField]:
        for message in decoder.iter_messages(sources):
            yield self.canonicalize(message)


def schema_manifest() -> list[dict[str, object]]:
    """Return a JSON-serializable representation of the frozen schema."""

    return [
        {
            "index": channel.index,
            "name": channel.name,
            "short_name": channel.short_name,
            "level_hpa": channel.level_hpa,
            "canonical_units": channel.canonical_units,
            "accumulated": channel.accumulated,
        }
        for channel in ERA5_CHANNELS
    ]


__all__ = [
    "CANONICAL_LATITUDE_ORIENTATION",
    "CANONICAL_LONGITUDE_ORIENTATION",
    "DEFAULT_GRID_SHAPE",
    "ERA5_CHANNELS",
    "ERA5_CHANNEL_NAMES",
    "ERA5_DATASET",
    "ERA5_INSTANTANEOUS_CHANNELS",
    "ERA5_PROVIDER_ID",
    "ERA5_PROVIDER_VERSION",
    "ERA5_TP_CHANNEL",
    "ERA5_FULL_TRAJECTORY",
    "CanonicalField",
    "ConditionSignature",
    "Era5Channel",
    "Era5ProviderAdapter",
    "FieldProvenance",
    "MessageDecoder",
    "ProviderMessage",
    "TemporalAccess",
    "schema_manifest",
]
