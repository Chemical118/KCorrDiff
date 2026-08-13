"""Absolute-time ERA5 indexing and yearly float32 mmap cache.

The cache stores the 23 instantaneous fields and the one-hour ``tp`` amount in
separate yearly ``.npy`` shards.  GRIB decoding is supplied by a
:class:`~kcorrdiff.data.provider_adapter.MessageDecoder`; this module has no
runtime dependency on ecCodes/cfgrib and can therefore be contract-tested with
synthetic messages.

All indexing is by aware UTC ``valid_time``.  Filenames, message order, and a
request file boundary have no temporal meaning here.  In particular, an
eight-token window may span a day, month, or year boundary without a special
case in the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from .provider_adapter import (
    CANONICAL_LATITUDE_ORIENTATION,
    CANONICAL_LONGITUDE_ORIENTATION,
    DEFAULT_GRID_SHAPE,
    ERA5_CHANNEL_NAMES,
    ERA5_INSTANTANEOUS_CHANNELS,
    ERA5_TP_CHANNEL,
    ConditionSignature,
    Era5ProviderAdapter,
    MessageDecoder,
    schema_manifest,
)


ERA5_CACHE_FORMAT = "kcorrdiff.era5-yearly-mmap.v1"
ERA5_NATIVE_HOURS = 8
INSTANTANEOUS_CHANNEL_COUNT = 23
AccessMode = Literal["full_trajectory", "target_end_causal"]


def _aware_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def floor_utc_hour(value: datetime) -> datetime:
    """Floor an aware timestamp to its absolute UTC native-hour anchor."""

    utc = _aware_utc(value, name="timestamp")
    return utc.replace(minute=0, second=0, microsecond=0)


def ceil_utc_hour(value: datetime) -> datetime:
    utc = _aware_utc(value, name="timestamp")
    floor = utc.replace(minute=0, second=0, microsecond=0)
    return floor if utc == floor else floor + timedelta(hours=1)


def native_hour_times(t0: datetime) -> tuple[datetime, ...]:
    """Return ``floor_hour(t0_UTC) + k hours`` for ``k=0..7``."""

    anchor = floor_utc_hour(t0)
    return tuple(anchor + timedelta(hours=index) for index in range(ERA5_NATIVE_HOURS))


def trajectory_window_mask(t0: datetime) -> NDArray[np.bool_]:
    """Mask tokens needed to bracket every endpoint through ``t0 + 6 h``.

    At an exact native hour, token ``h0+7`` is outside the maximum trajectory;
    for a non-hour issue time it brackets the six-hour endpoint and is active.
    The returned shape is always ``[8]``.
    """

    t0_utc = _aware_utc(t0, name="t0")
    final_bracket = ceil_utc_hour(t0_utc + timedelta(hours=6))
    return np.asarray(
        [valid_time <= final_bracket for valid_time in native_hour_times(t0_utc)],
        dtype=np.bool_,
    )


def temporal_access_mask(
    t0: datetime,
    *,
    lead_hours: float,
    access_mode: AccessMode,
) -> NDArray[np.bool_]:
    """Return the intentional oracle-access policy independently of data QC."""

    t0_utc = _aware_utc(t0, name="t0")
    if not np.isfinite(lead_hours) or lead_hours < 0.5 or lead_hours > 6.0:
        raise ValueError("lead_hours must lie in [0.5, 6.0]")
    if not np.isclose(lead_hours * 2.0, round(lead_hours * 2.0), atol=1e-12):
        raise ValueError("lead_hours must be a 30-minute increment")
    trajectory = trajectory_window_mask(t0_utc)
    if access_mode == "full_trajectory":
        return trajectory
    if access_mode != "target_end_causal":
        raise ValueError(f"unsupported access_mode: {access_mode!r}")
    target_end = t0_utc + timedelta(hours=float(lead_hours))
    causal = np.asarray(
        [valid_time <= target_end for valid_time in native_hour_times(t0_utc)],
        dtype=np.bool_,
    )
    return trajectory & causal


def _hours_in_year(year: int) -> int:
    start = datetime(year, 1, 1, tzinfo=UTC)
    end = datetime(year + 1, 1, 1, tzinfo=UTC)
    return int((end - start) / timedelta(hours=1))


def estimate_era5_cache_bytes(
    years: Sequence[int], *, grid_shape: tuple[int, int] = DEFAULT_GRID_SHAPE
) -> int:
    """Exact mmap payload bytes, excluding small manifests and coordinates."""

    unique_years = tuple(sorted(set(map(int, years))))
    if not unique_years:
        raise ValueError("at least one cache year is required")
    cells = int(np.prod(grid_shape))
    hours = sum(_hours_in_year(year) for year in unique_years)
    # 24 float32 fields plus two persisted boolean validity arrays.
    return hours * (24 * cells * np.dtype(np.float32).itemsize + 2)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class Era5ShardManifest:
    year: int
    start_utc: str
    end_utc_exclusive: str
    hours: int
    instantaneous_file: str
    tp_file: str
    data_valid_inst_file: str
    tp_valid_file: str
    valid_instantaneous_hours: int
    valid_tp_hours: int
    instantaneous_sha256: str
    tp_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "year": self.year,
            "start_utc": self.start_utc,
            "end_utc_exclusive": self.end_utc_exclusive,
            "hours": self.hours,
            "instantaneous_file": self.instantaneous_file,
            "tp_file": self.tp_file,
            "data_valid_inst_file": self.data_valid_inst_file,
            "tp_valid_file": self.tp_valid_file,
            "valid_instantaneous_hours": self.valid_instantaneous_hours,
            "valid_tp_hours": self.valid_tp_hours,
            "instantaneous_sha256": self.instantaneous_sha256,
            "tp_sha256": self.tp_sha256,
        }


@dataclass(frozen=True, slots=True)
class Era5CacheManifest:
    format_version: str
    dtype: str
    grid_shape: tuple[int, int]
    latitude_file: str
    longitude_file: str
    shards: tuple[Era5ShardManifest, ...]
    sources: tuple[str, ...]
    transformations: tuple[str, ...]
    provider_id: str
    provider_version: str
    dataset: str

    def to_json(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "dtype": self.dtype,
            "grid_shape": list(self.grid_shape),
            "latitude_file": self.latitude_file,
            "longitude_file": self.longitude_file,
            "latitude_orientation": CANONICAL_LATITUDE_ORIENTATION,
            "longitude_orientation": CANONICAL_LONGITUDE_ORIENTATION,
            "channels": schema_manifest(),
            "instantaneous_channel_names": list(ERA5_CHANNEL_NAMES[:-1]),
            "tp_channel_name": ERA5_TP_CHANNEL.name,
            "tp_units": ERA5_TP_CHANNEL.canonical_units,
            "tp_interval": "(valid_time - 1 hour, valid_time]",
            "time_zone": "UTC",
            "shards": [shard.to_json() for shard in self.shards],
            "provenance": {
                "provider_id": self.provider_id,
                "provider_version": self.provider_version,
                "dataset": self.dataset,
                "sources": list(self.sources),
                "transformations": list(self.transformations),
            },
        }


@dataclass(slots=True)
class _WritableYear:
    year: int
    instantaneous_file: Path
    tp_file: Path
    instantaneous: np.memmap
    tp: np.memmap
    seen_instantaneous: NDArray[np.bool_]
    seen_tp: NDArray[np.bool_]

    @property
    def start(self) -> datetime:
        return datetime(self.year, 1, 1, tzinfo=UTC)

    @property
    def hours(self) -> int:
        return self.seen_instantaneous.shape[0]


def build_era5_mmap_cache(
    *,
    decoder: MessageDecoder,
    sources: Sequence[object],
    output_dir: Path,
    years: Sequence[int],
    adapter: Era5ProviderAdapter | None = None,
    require_complete_instantaneous: bool = True,
    require_complete_tp: bool = True,
    reserve_bytes: int = 20 * 1024**3,
    verify_sha256: bool = True,
) -> Era5CacheManifest:
    """Decode, validate, and atomically publish separate yearly mmap shards.

    The decoder may yield messages in arbitrary order and from arbitrary file
    groupings.  Duplicate semantic fields are rejected.  Each canonical UTC
    hour is placed by arithmetic offset within its calendar-year shard.

    A caller building a diagnostic subset may disable completeness checks;
    invalid hours remain explicit in the two mask arrays and carry neutral
    zero values.  Official ERA5-oracle cache builds should keep both checks on.
    Existing destinations are never overwritten.
    """

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"ERA5 cache destination already exists: {output_dir}")
    adapter = adapter or Era5ProviderAdapter()
    selected_years = tuple(sorted(set(map(int, years))))
    if not selected_years:
        raise ValueError("at least one cache year is required")
    for year in selected_years:
        if year < 1900 or year > 9998:
            raise ValueError(f"unsupported calendar year: {year}")

    estimated = estimate_era5_cache_bytes(selected_years, grid_shape=adapter.grid_shape)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir.parent).free
    if estimated + reserve_bytes > free:
        raise OSError(
            f"insufficient free space: need cache={estimated} + reserve={reserve_bytes}, "
            f"free={free}"
        )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent)
    )
    writable: dict[int, _WritableYear] = {}
    latitude: NDArray[np.float64] | None = None
    longitude: NDArray[np.float64] | None = None
    source_names: set[str] = set()
    transformations: set[str] = set()
    try:
        rows, columns = adapter.grid_shape
        for year in selected_years:
            hours = _hours_in_year(year)
            instantaneous_file = temporary / f"instantaneous-{year}.npy"
            tp_file = temporary / f"tp-{year}.npy"
            instantaneous = open_memmap(
                instantaneous_file,
                mode="w+",
                dtype=np.float32,
                shape=(hours, INSTANTANEOUS_CHANNEL_COUNT, rows, columns),
            )
            tp = open_memmap(
                tp_file,
                mode="w+",
                dtype=np.float32,
                shape=(hours, 1, rows, columns),
            )
            # POSIX guarantees zero-filled bytes in newly extended files.  Do
            # not eagerly touch the full multi-GiB address range; valid rows
            # are populated below and masks are authoritative.
            writable[year] = _WritableYear(
                year=year,
                instantaneous_file=instantaneous_file,
                tp_file=tp_file,
                instantaneous=instantaneous,
                tp=tp,
                seen_instantaneous=np.zeros(
                    (hours, INSTANTANEOUS_CHANNEL_COUNT), dtype=np.bool_
                ),
                seen_tp=np.zeros(hours, dtype=np.bool_),
            )

        for field in adapter.iter_canonical(decoder, sources):
            valid_time = field.valid_time_utc
            try:
                target = writable[valid_time.year]
            except KeyError as error:
                raise ValueError(
                    f"decoded ERA5 valid_time is outside selected years: {valid_time.isoformat()}"
                ) from error
            offset = int((valid_time - target.start) / timedelta(hours=1))
            if target.start + timedelta(hours=offset) != valid_time:
                raise ValueError(f"non-hour ERA5 valid_time: {valid_time.isoformat()}")
            if not 0 <= offset < target.hours:  # pragma: no cover - guarded by year
                raise ValueError(f"ERA5 valid_time outside shard: {valid_time.isoformat()}")
            if field.channel.accumulated:
                if target.seen_tp[offset]:
                    raise ValueError(f"duplicate ERA5 tp field at {valid_time.isoformat()}")
                target.tp[offset, 0] = field.values
                target.seen_tp[offset] = True
            else:
                index = field.channel.index
                if index >= INSTANTANEOUS_CHANNEL_COUNT:  # pragma: no cover
                    raise AssertionError("instantaneous channel index exceeds 22")
                if target.seen_instantaneous[offset, index]:
                    raise ValueError(
                        f"duplicate ERA5 {field.channel.name} field at "
                        f"{valid_time.isoformat()}"
                    )
                target.instantaneous[offset, index] = field.values
                target.seen_instantaneous[offset, index] = True

            if latitude is None:
                latitude = field.latitude.copy()
                longitude = field.longitude.copy()
            elif not np.array_equal(latitude, field.latitude) or not np.array_equal(
                longitude, field.longitude
            ):
                raise ValueError("ERA5 latitude/longitude grid changed between messages")
            source_names.add(field.provenance.source)
            transformations.update(field.provenance.transformations)

        if latitude is None or longitude is None:
            raise ValueError("decoder emitted no ERA5 messages")
        latitude_file = "latitude.npy"
        longitude_file = "longitude.npy"
        np.save(temporary / latitude_file, latitude, allow_pickle=False)
        np.save(temporary / longitude_file, longitude, allow_pickle=False)

        shards: list[Era5ShardManifest] = []
        for year in selected_years:
            target = writable[year]
            data_valid_inst = np.all(target.seen_instantaneous, axis=1)
            tp_valid = target.seen_tp.copy()
            partial = np.any(target.seen_instantaneous, axis=1) & ~data_valid_inst
            # Partially decoded instantaneous tokens are not model inputs.  A
            # neutral value plus false validity mask prevents accidental use.
            target.instantaneous[~data_valid_inst] = 0.0
            target.tp[~tp_valid] = 0.0
            target.instantaneous.flush()
            target.tp.flush()
            if require_complete_instantaneous and not data_valid_inst.all():
                raise ValueError(
                    f"ERA5 {year} has {int((~data_valid_inst).sum())} incomplete "
                    f"instantaneous hours ({int(partial.sum())} partially decoded)"
                )
            if require_complete_tp and not tp_valid.all():
                raise ValueError(
                    f"ERA5 {year} has {int((~tp_valid).sum())} missing tp hours"
                )

            inst_valid_file = f"data-valid-inst-{year}.npy"
            tp_valid_file = f"tp-valid-{year}.npy"
            np.save(temporary / inst_valid_file, data_valid_inst, allow_pickle=False)
            np.save(temporary / tp_valid_file, tp_valid, allow_pickle=False)
            end = datetime(year + 1, 1, 1, tzinfo=UTC)
            shards.append(
                Era5ShardManifest(
                    year=year,
                    start_utc=target.start.isoformat(),
                    end_utc_exclusive=end.isoformat(),
                    hours=target.hours,
                    instantaneous_file=target.instantaneous_file.name,
                    tp_file=target.tp_file.name,
                    data_valid_inst_file=inst_valid_file,
                    tp_valid_file=tp_valid_file,
                    valid_instantaneous_hours=int(data_valid_inst.sum()),
                    valid_tp_hours=int(tp_valid.sum()),
                    instantaneous_sha256=(
                        _hash_file(target.instantaneous_file) if verify_sha256 else ""
                    ),
                    tp_sha256=_hash_file(target.tp_file) if verify_sha256 else "",
                )
            )
            # Close mmap descriptors before publishing the directory.
            del target.instantaneous
            del target.tp

        manifest = Era5CacheManifest(
            format_version=ERA5_CACHE_FORMAT,
            dtype="float32",
            grid_shape=adapter.grid_shape,
            latitude_file=latitude_file,
            longitude_file=longitude_file,
            shards=tuple(shards),
            sources=tuple(sorted(source_names)),
            transformations=tuple(sorted(transformations)),
            provider_id=adapter.provider_id,
            provider_version=adapter.provider_version,
            dataset=adapter.dataset,
        )
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest.to_json(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
        return manifest
    except BaseException:
        # Preserve expensive partial work for explicit inspection.  Never
        # overwrite or silently remove user-visible cache data.
        raise


@dataclass(frozen=True, slots=True)
class Era5WindowProvenance:
    provider_id: str
    provider_version: str
    dataset: str
    cache_format: str
    cache_manifest: str
    condition_signature: str
    access_mode: AccessMode
    valid_times_utc: tuple[datetime, ...]
    tp_intervals_utc: tuple[tuple[datetime, datetime], ...]


@dataclass(frozen=True, slots=True)
class Era5Window:
    """One fixed-shape ERA5 condition with deliberately separate masks."""

    instantaneous: NDArray[np.float32]
    tp: NDArray[np.float32]
    valid_times_utc: tuple[datetime, ...]
    delta_hours: NDArray[np.float32]
    tp_interval_center_delta_hours: NDArray[np.float32]
    data_valid_inst: NDArray[np.bool_]
    tp_valid: NDArray[np.bool_]
    trajectory_window_mask: NDArray[np.bool_]
    temporal_access_mask: NDArray[np.bool_]
    era_present: bool
    tp_present: bool
    provenance: Era5WindowProvenance

    @property
    def values(self) -> NDArray[np.float32]:
        """Return canonical ``[8,24,H,W]`` order without changing precision."""

        return np.concatenate((self.instantaneous, self.tp), axis=1)


class Era5MmapCache:
    """Read-only absolute-valid-time access to yearly ERA5 mmap shards."""

    def __init__(self, root: Path, *, verify_hashes: bool = False) -> None:
        self.root = root.resolve()
        self.manifest_path = self.root / "manifest.json"
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if raw.get("format_version") != ERA5_CACHE_FORMAT:
            raise ValueError(f"unsupported ERA5 cache at {self.root}")
        if raw.get("dtype") != "float32":
            raise ValueError("ERA5 cache must preserve float32 precision")
        if tuple(raw.get("instantaneous_channel_names", ())) != ERA5_CHANNEL_NAMES[:-1]:
            raise ValueError("ERA5 instantaneous channel schema/order mismatch")
        if raw.get("tp_channel_name") != "tp" or raw.get("tp_units") != "mm per 1 hour":
            raise ValueError("ERA5 tp semantic contract mismatch")
        self.grid_shape = tuple(map(int, raw["grid_shape"]))
        self.latitude = np.load(
            self.root / raw["latitude_file"], mmap_mode="r", allow_pickle=False
        )
        self.longitude = np.load(
            self.root / raw["longitude_file"], mmap_mode="r", allow_pickle=False
        )
        self.provenance: Mapping[str, object] = raw["provenance"]
        self._shards: dict[int, Mapping[str, object]] = {}
        self._arrays: dict[str, np.memmap] = {}
        for shard in raw["shards"]:
            year = int(shard["year"])
            if year in self._shards:
                raise ValueError(f"duplicate ERA5 cache year: {year}")
            if int(shard["hours"]) != _hours_in_year(year):
                raise ValueError(f"ERA5 cache year {year} has an invalid hour count")
            self._shards[year] = shard
            if verify_hashes:
                for filename_key, digest_key in (
                    ("instantaneous_file", "instantaneous_sha256"),
                    ("tp_file", "tp_sha256"),
                ):
                    expected = str(shard[digest_key])
                    if not expected:
                        raise ValueError(f"ERA5 cache has no {digest_key} to verify")
                    if _hash_file(self.root / str(shard[filename_key])) != expected:
                        raise ValueError(f"ERA5 cache hash mismatch: {shard[filename_key]}")

    def _array(self, filename: str) -> np.memmap:
        result = self._arrays.get(filename)
        if result is None:
            result = np.load(self.root / filename, mmap_mode="r", allow_pickle=False)
            self._arrays[filename] = result
        return result

    def _read_hour(
        self, valid_time: datetime
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], bool, bool]:
        valid_time = _aware_utc(valid_time, name="valid_time")
        if valid_time != floor_utc_hour(valid_time):
            raise ValueError("ERA5 valid_time lookup requires an exact UTC hour")
        shard = self._shards.get(valid_time.year)
        if shard is None:
            rows, columns = self.grid_shape
            return (
                np.zeros((INSTANTANEOUS_CHANNEL_COUNT, rows, columns), np.float32),
                np.zeros((1, rows, columns), np.float32),
                False,
                False,
            )
        start = datetime(valid_time.year, 1, 1, tzinfo=UTC)
        offset = int((valid_time - start) / timedelta(hours=1))
        instantaneous = np.asarray(
            self._array(str(shard["instantaneous_file"]))[offset]
        )
        tp = np.asarray(self._array(str(shard["tp_file"]))[offset])
        data_valid = bool(
            self._array(str(shard["data_valid_inst_file"]))[offset]
        )
        tp_valid = bool(self._array(str(shard["tp_valid_file"]))[offset])
        return instantaneous, tp, data_valid, tp_valid

    def read_window(
        self,
        t0: datetime,
        *,
        lead_hours: float,
        signature: ConditionSignature,
        strict_instantaneous: bool = True,
    ) -> Era5Window:
        """Read an eight-token condition under one canonical source signature.

        ``data_valid_inst``, ``tp_valid``, the maximum trajectory bracket, the
        intentional temporal-access policy, and source-presence flags remain
        separate.  A full-trajectory oracle normally uses
        ``strict_instantaneous=True``; missing ``tp`` is represented by its
        mask and does not silently disable the other 23 fields.
        """

        canonical = signature.canonical()
        if canonical.temporal_access_mode == "no_era_access":
            access_mode: AccessMode = "full_trajectory"
        elif canonical.temporal_access_mode in {
            "full_trajectory",
            "target_end_causal",
        }:
            access_mode = canonical.temporal_access_mode
        else:  # pragma: no cover - Literal plus canonical validation
            raise ValueError(
                f"unsupported temporal access: {canonical.temporal_access_mode}"
            )

        t0_utc = _aware_utc(t0, name="t0")
        valid_times = native_hour_times(t0_utc)
        inst_values: list[NDArray[np.float32]] = []
        tp_values: list[NDArray[np.float32]] = []
        inst_valid: list[bool] = []
        tp_valid: list[bool] = []
        for valid_time in valid_times:
            instantaneous, tp, valid_inst, valid_tp = self._read_hour(valid_time)
            inst_values.append(instantaneous)
            tp_values.append(tp)
            inst_valid.append(valid_inst)
            tp_valid.append(valid_tp)

        trajectory = trajectory_window_mask(t0_utc)
        temporal = temporal_access_mask(
            t0_utc, lead_hours=lead_hours, access_mode=access_mode
        )
        data_valid_inst = np.asarray(inst_valid, dtype=np.bool_)
        tp_valid_array = np.asarray(tp_valid, dtype=np.bool_)
        era_present = bool(canonical.era_present)
        tp_present = bool(canonical.tp_present and era_present)
        required_inst = trajectory & temporal
        if strict_instantaneous and era_present and not data_valid_inst[required_inst].all():
            missing = [
                valid_time.isoformat()
                for valid_time, required, valid in zip(
                    valid_times, required_inst, data_valid_inst, strict=True
                )
                if required and not valid
            ]
            raise KeyError(
                "missing required ERA5 instantaneous token(s): " + ", ".join(missing)
            )

        instantaneous_array = np.stack(inst_values, axis=0).astype(np.float32, copy=False)
        tp_array = np.stack(tp_values, axis=0).astype(np.float32, copy=False)
        if not era_present:
            instantaneous_array = np.zeros_like(instantaneous_array)
            tp_array = np.zeros_like(tp_array)
            temporal = np.zeros(ERA5_NATIVE_HOURS, dtype=np.bool_)
        elif not tp_present:
            tp_array = np.zeros_like(tp_array)

        delta_hours = np.asarray(
            [(value - t0_utc) / timedelta(hours=1) for value in valid_times],
            dtype=np.float32,
        )
        tp_intervals = tuple(
            (value - timedelta(hours=1), value) for value in valid_times
        )
        provenance = Era5WindowProvenance(
            provider_id=str(self.provenance["provider_id"]),
            provider_version=str(self.provenance["provider_version"]),
            dataset=str(self.provenance["dataset"]),
            cache_format=ERA5_CACHE_FORMAT,
            cache_manifest=str(self.manifest_path),
            condition_signature=canonical.key,
            access_mode=access_mode,
            valid_times_utc=valid_times,
            tp_intervals_utc=tp_intervals,
        )
        return Era5Window(
            instantaneous=instantaneous_array,
            tp=tp_array,
            valid_times_utc=valid_times,
            delta_hours=delta_hours,
            tp_interval_center_delta_hours=delta_hours - np.float32(0.5),
            data_valid_inst=data_valid_inst,
            tp_valid=tp_valid_array,
            trajectory_window_mask=trajectory,
            temporal_access_mask=temporal,
            era_present=era_present,
            tp_present=tp_present,
            provenance=provenance,
        )

    def close(self) -> None:
        self._arrays.clear()

    def __getstate__(self) -> dict[str, object]:
        # Spawned DataLoader workers reopen their own file descriptors while
        # sharing immutable pages through the kernel page cache.
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state


__all__ = [
    "ERA5_CACHE_FORMAT",
    "ERA5_NATIVE_HOURS",
    "INSTANTANEOUS_CHANNEL_COUNT",
    "AccessMode",
    "Era5CacheManifest",
    "Era5MmapCache",
    "Era5ShardManifest",
    "Era5Window",
    "Era5WindowProvenance",
    "build_era5_mmap_cache",
    "ceil_utc_hour",
    "estimate_era5_cache_bytes",
    "floor_utc_hour",
    "native_hour_times",
    "temporal_access_mask",
    "trajectory_window_mask",
]
