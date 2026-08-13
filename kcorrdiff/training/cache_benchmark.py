"""Real mmap-cache and host-to-device benchmark primitives.

This module deliberately reads the seven future target scans as a benchmark
payload only.  They are never exposed as model conditions: the benchmark is
measuring the complete training-loader I/O path, not constructing a training
sample API.

Radar and ERA cache objects are created lazily in the process which executes
``Dataset.__getitem__``.  Consequently every persistent DataLoader worker owns
one set of read-only mmap handles and reuses it for its full lifetime; no mmap
descriptor is opened in the parent before workers are spawned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Final, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from kcorrdiff.data.availability import (
    format_radar_timestamp,
    history_times,
    require_five_minute_slot,
    target_times,
)
from kcorrdiff.data.cache import FORMAT_VERSION as RADAR_CACHE_FORMAT
from kcorrdiff.data.cache import RadarCache
from kcorrdiff.data.condition_schema import parse_condition_signature
from kcorrdiff.data.era5_reader import (
    ERA5_CACHE_FORMAT,
    ERA5_NATIVE_HOURS,
    Era5MmapCache,
)
from kcorrdiff.data.provider_adapter import ERA5_CHANNEL_NAMES
from kcorrdiff.data.sampling import DrawRow


PRODUCTION_RADAR_SHAPE: Final[tuple[int, int]] = (256, 256)
PRODUCTION_ERA_SHAPE: Final[tuple[int, int]] = (33, 33)
PRODUCTION_TARGET_WIDTHS: Final[tuple[int, ...]] = (64, 128, 256, 384, 512)
PRODUCTION_CONTEXT_WIDTHS: Final[tuple[int, ...]] = (32, 64, 128, 256, 384)
PRODUCTION_ERA_LATENT_CHANNELS: Final[int] = 128
STATIC_FLOAT_CHANNELS: Final[tuple[str, ...]] = (
    "elevation",
    "slope_x",
    "slope_y",
    "local_relief",
    "land_mask",
    "lcc_x_normalized",
    "lcc_y_normalized",
)


def _integer_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain integers")
        converted = int(value)
        if converted <= 0:
            raise ValueError(f"{name} must contain positive values")
        result.append(converted)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CacheBenchmarkContract:
    """Strict production I/O and precision shape contract."""

    radar_shape: tuple[int, int] = PRODUCTION_RADAR_SHAPE
    era_shape: tuple[int, int] = PRODUCTION_ERA_SHAPE
    target_widths: tuple[int, ...] = PRODUCTION_TARGET_WIDTHS
    context_widths: tuple[int, ...] = PRODUCTION_CONTEXT_WIDTHS
    era_latent_channels: int = PRODUCTION_ERA_LATENT_CHANNELS
    precision: str = "float32"
    tf32_enabled: bool = False
    allow_test_override: bool = False

    def __post_init__(self) -> None:
        radar_shape = _integer_tuple("radar_shape", self.radar_shape)
        era_shape = _integer_tuple("era_shape", self.era_shape)
        target_widths = _integer_tuple("target_widths", self.target_widths)
        context_widths = _integer_tuple("context_widths", self.context_widths)
        if len(radar_shape) != 2 or len(era_shape) != 2:
            raise ValueError("radar_shape and era_shape must be two-dimensional")
        if len(target_widths) != 5 or len(context_widths) != 5:
            raise ValueError("target/context widths must contain five levels")
        if isinstance(self.era_latent_channels, bool) or not isinstance(
            self.era_latent_channels, (int, np.integer)
        ):
            raise TypeError("era_latent_channels must be an integer")
        if int(self.era_latent_channels) <= 0:
            raise ValueError("era_latent_channels must be positive")
        object.__setattr__(self, "radar_shape", radar_shape)
        object.__setattr__(self, "era_shape", era_shape)
        object.__setattr__(self, "target_widths", target_widths)
        object.__setattr__(self, "context_widths", context_widths)
        object.__setattr__(self, "era_latent_channels", int(self.era_latent_channels))

        if self.precision != "float32":
            raise ValueError("benchmark precision must be float32")
        if self.tf32_enabled:
            raise ValueError("TF32 must be disabled")
        if not self.allow_test_override:
            expected = (
                PRODUCTION_RADAR_SHAPE,
                PRODUCTION_ERA_SHAPE,
                PRODUCTION_TARGET_WIDTHS,
                PRODUCTION_CONTEXT_WIDTHS,
                PRODUCTION_ERA_LATENT_CHANNELS,
            )
            actual = (
                radar_shape,
                era_shape,
                target_widths,
                context_widths,
                int(self.era_latent_channels),
            )
            if actual != expected:
                raise ValueError(
                    "full-width/spatial fallback is forbidden: "
                    f"expected {expected}, got {actual}"
                )


def _npy_header(path: Path) -> tuple[tuple[int, ...], np.dtype[Any]]:
    """Read only a .npy header; never open a data mmap in the parent."""

    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:  # pragma: no cover - NumPy currently supports versions 1-3.
            raise ValueError(f"unsupported npy format {version}: {path}")
    return tuple(map(int, shape)), np.dtype(dtype)


def _require_file(path: Path, *, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


def load_static_model_inputs(
    path: Path,
    *,
    radar_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the seven v1.1.3b target-static fields plus coverage once/worker."""

    _require_file(path, description="target static NPZ")
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set((*STATIC_FLOAT_CHANNELS, "coverage")) - set(archive.files))
        if missing:
            raise ValueError("target static NPZ is missing: " + ", ".join(missing))
        channels: list[np.ndarray] = []
        for name in STATIC_FLOAT_CHANNELS:
            raw = np.asarray(archive[name])
            if raw.shape != radar_shape:
                raise ValueError(
                    f"static {name} must have shape {radar_shape}, got {raw.shape}"
                )
            if name == "land_mask":
                if not (
                    raw.dtype == np.dtype(np.bool_)
                    or np.issubdtype(raw.dtype, np.number)
                ):
                    raise TypeError(
                        f"static land_mask must be numeric/bool, got {raw.dtype}"
                    )
                if not np.all((raw == 0) | (raw == 1)):
                    raise ValueError("land_mask must contain only 0/1 values")
                value = raw.astype(np.float32)
            else:
                if not np.issubdtype(raw.dtype, np.number):
                    raise TypeError(f"static {name} must be numeric, got {raw.dtype}")
                # The immutable static artifact may retain float64 coordinate
                # geometry. Model-facing loader payloads are always explicitly
                # converted to FP32; this is part of the production contract,
                # not a runtime precision fallback.
                value = raw.astype(np.float32)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"static {name} contains non-finite values")
            channels.append(np.asarray(value, dtype=np.float32))

        raw_coverage = np.asarray(archive["coverage"])
        if raw_coverage.shape != radar_shape:
            raise ValueError("static coverage has the wrong spatial shape")
        if not (
            raw_coverage.dtype == np.dtype(np.bool_)
            or np.issubdtype(raw_coverage.dtype, np.number)
        ):
            raise TypeError("static coverage must be numeric/bool 0/1")
        if not np.all((raw_coverage == 0) | (raw_coverage == 1)):
            raise ValueError("static coverage must contain only 0/1 values")
        coverage = raw_coverage.astype(np.bool_, copy=False)
    return (
        np.ascontiguousarray(np.stack(channels, axis=0), dtype=np.float32),
        np.ascontiguousarray(coverage, dtype=np.bool_),
    )


def validate_cache_artifacts(
    *,
    radar_cache_root: Path,
    era5_cache_root: Path,
    static_path: Path,
    contract: CacheBenchmarkContract,
) -> dict[str, object]:
    """Validate cache manifests and array headers without retaining mmap handles."""

    radar_manifest_path = radar_cache_root / "manifest.json"
    era_manifest_path = era5_cache_root / "manifest.json"
    _require_file(radar_manifest_path, description="radar cache manifest")
    _require_file(era_manifest_path, description="ERA5 cache manifest")
    radar = json.loads(radar_manifest_path.read_text(encoding="utf-8"))
    if radar.get("format_version") != RADAR_CACHE_FORMAT:
        raise ValueError("unsupported radar cache format")
    if radar.get("dtype") != "float32":
        raise ValueError("radar cache must be float32")
    sources: set[str] = set()
    for shard in radar.get("shards", ()):  # format reader performs deeper checks.
        source = str(shard.get("source", ""))
        if source not in {"target", "condition"}:
            raise ValueError(f"unsupported radar cache source: {source!r}")
        sources.add(source)
        if tuple(map(int, shard.get("shape", ()))) != contract.radar_shape:
            raise ValueError(
                f"radar {source} shard is not full spatial size {contract.radar_shape}"
            )
        values_path = radar_cache_root / str(shard["values_file"])
        timestamps_path = radar_cache_root / str(shard["timestamps_file"])
        _require_file(values_path, description="radar values shard")
        _require_file(timestamps_path, description="radar timestamp index")
        shape, dtype = _npy_header(values_path)
        expected = (int(shard["samples"]), *contract.radar_shape)
        if shape != expected or dtype != np.dtype(np.float32):
            raise ValueError(
                f"radar shard header mismatch: expected {expected}/float32, "
                f"got {shape}/{dtype}"
            )
    if sources != {"target", "condition"}:
        raise ValueError("radar cache must contain both target and condition shards")

    era = json.loads(era_manifest_path.read_text(encoding="utf-8"))
    if era.get("format_version") != ERA5_CACHE_FORMAT:
        raise ValueError("unsupported ERA5 cache format")
    if era.get("dtype") != "float32":
        raise ValueError("ERA5 cache must be float32")
    if tuple(map(int, era.get("grid_shape", ()))) != contract.era_shape:
        raise ValueError(
            f"ERA5 grid must retain {contract.era_shape}, got {era.get('grid_shape')}"
        )
    if tuple(era.get("instantaneous_channel_names", ())) != ERA5_CHANNEL_NAMES[:-1]:
        raise ValueError("ERA5 instantaneous schema/order mismatch")
    if era.get("tp_channel_name") != "tp" or era.get("tp_units") != "mm per 1 hour":
        raise ValueError("ERA5 tp semantic contract mismatch")
    for coordinate_key in ("latitude_file", "longitude_file"):
        coordinate_path = era5_cache_root / str(era[coordinate_key])
        _require_file(coordinate_path, description=f"ERA5 {coordinate_key}")
        shape, _dtype = _npy_header(coordinate_path)
        if shape not in {contract.era_shape, (contract.era_shape[0],), (contract.era_shape[1],)}:
            raise ValueError(f"unexpected ERA5 coordinate shape: {shape}")
    if not era.get("shards"):
        raise ValueError("ERA5 cache contains no yearly shards")
    for shard in era["shards"]:
        hours = int(shard["hours"])
        expected_headers = (
            ("instantaneous_file", (hours, 23, *contract.era_shape), np.float32),
            ("tp_file", (hours, 1, *contract.era_shape), np.float32),
            ("data_valid_inst_file", (hours,), np.bool_),
            ("tp_valid_file", (hours,), np.bool_),
        )
        for key, expected_shape, expected_dtype in expected_headers:
            array_path = era5_cache_root / str(shard[key])
            _require_file(array_path, description=f"ERA5 {key}")
            shape, dtype = _npy_header(array_path)
            if shape != expected_shape or dtype != np.dtype(expected_dtype):
                raise ValueError(
                    f"ERA5 {key} header mismatch: expected "
                    f"{expected_shape}/{np.dtype(expected_dtype)}, got {shape}/{dtype}"
                )

    static, coverage = load_static_model_inputs(
        static_path, radar_shape=contract.radar_shape
    )
    return {
        "radar_format": radar["format_version"],
        "radar_shards": len(radar["shards"]),
        "radar_shape": list(contract.radar_shape),
        "era5_format": era["format_version"],
        "era5_shards": len(era["shards"]),
        "era5_shape": list(contract.era_shape),
        "static_channels": list(STATIC_FLOAT_CHANNELS),
        "static_shape": list(static.shape),
        "static_coverage_fraction": float(coverage.mean()),
    }


def _parse_t0(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid draw-manifest t0_utc: {value!r}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("draw-manifest t0_utc must be timezone-aware")
    result = parsed.astimezone(UTC)
    require_five_minute_slot(result)
    return result


@dataclass(frozen=True, slots=True)
class CacheDatasetSpec:
    radar_cache_root: Path
    era5_cache_root: Path
    static_path: Path
    rows: tuple[DrawRow, ...]
    contract: CacheBenchmarkContract = CacheBenchmarkContract()
    radar_timezone: str = "Asia/Seoul"
    verify_cache_hashes: bool = False

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("benchmark requires at least one draw-manifest row")
        for row in self.rows:
            if not row.t0_utc or not row.condition_signature:
                raise ValueError("draw rows require t0 and condition signature")
            t0 = _parse_t0(row.t0_utc)
            parse_condition_signature(row.condition_signature)
            if not math.isfinite(row.lead_hours):
                raise ValueError("draw lead must be finite")
            # Enforce the fixed half-hour lead vocabulary before any worker is
            # started instead of discovering a bad row inside DataLoader.
            target_times(t0, float(row.lead_hours))


class CacheBenchmarkDataset(Dataset[dict[str, object]]):
    """Worker-local lazy access to real radar, ERA, and static artifacts."""

    def __init__(self, spec: CacheDatasetSpec) -> None:
        super().__init__()
        if not isinstance(spec, CacheDatasetSpec):
            raise TypeError("spec must be CacheDatasetSpec")
        self.spec = spec
        self._radar_cache: RadarCache | None = None
        self._era5_cache: Era5MmapCache | None = None
        self._static_fields: np.ndarray | None = None
        self._static_coverage: np.ndarray | None = None
        self._opened_pid: int | None = None

    @property
    def handles_open(self) -> bool:
        return self._radar_cache is not None or self._era5_cache is not None

    def __len__(self) -> int:
        return len(self.spec.rows)

    def _close_handles(self) -> None:
        if self._radar_cache is not None:
            self._radar_cache.close()
        if self._era5_cache is not None:
            self._era5_cache.close()
        self._radar_cache = None
        self._era5_cache = None
        self._static_fields = None
        self._static_coverage = None
        self._opened_pid = None

    def close(self) -> None:
        self._close_handles()

    def _ensure_open(self) -> None:
        pid = os.getpid()
        if self._opened_pid is not None and self._opened_pid != pid:
            # Defensive fork boundary: inherited descriptors are discarded.
            self._close_handles()
        if self._radar_cache is None:
            self._radar_cache = RadarCache(
                self.spec.radar_cache_root,
                verify_hashes=self.spec.verify_cache_hashes,
            )
            self._era5_cache = Era5MmapCache(
                self.spec.era5_cache_root,
                verify_hashes=self.spec.verify_cache_hashes,
            )
            self._static_fields, self._static_coverage = load_static_model_inputs(
                self.spec.static_path,
                radar_shape=self.spec.contract.radar_shape,
            )
            self._opened_pid = pid

    def __getitem__(self, index: int) -> dict[str, object]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("benchmark index must be an integer")
        self._ensure_open()
        assert self._radar_cache is not None
        assert self._era5_cache is not None
        assert self._static_fields is not None
        assert self._static_coverage is not None
        row = self.spec.rows[int(index)]
        t0 = _parse_t0(row.t0_utc)
        signature = parse_condition_signature(row.condition_signature)
        history_keys = tuple(
            format_radar_timestamp(value, timezone=self.spec.radar_timezone)
            for value in history_times(t0)
        )
        # BENCHMARK-ONLY label I/O.  These values never enter a condition API.
        future_keys = tuple(
            format_radar_timestamp(value, timezone=self.spec.radar_timezone)
            for value in target_times(t0, float(row.lead_hours))
        )

        started = time.perf_counter()
        target_history = self._radar_cache.read_many("target", history_keys)
        context_history = self._radar_cache.read_many("condition", history_keys)
        target_future_benchmark_only = self._radar_cache.read_many(
            "target", future_keys
        )
        era = self._era5_cache.read_window(
            t0,
            lead_hours=float(row.lead_hours),
            signature=signature,
            strict_instantaneous=True,
        )
        read_seconds = time.perf_counter() - started

        radar_shape = self.spec.contract.radar_shape
        era_shape = self.spec.contract.era_shape
        expected = {
            "target_history": (12, *radar_shape),
            "context_history": (12, *radar_shape),
            "target_future_benchmark_only": (7, *radar_shape),
            "era_values": (ERA5_NATIVE_HOURS, 24, *era_shape),
        }
        arrays: dict[str, np.ndarray] = {
            "target_history": target_history,
            "context_history": context_history,
            "target_future_benchmark_only": target_future_benchmark_only,
            "era_values": era.values,
            "era_delta_hours": era.delta_hours,
            "era_tp_interval_center_delta_hours": (
                era.tp_interval_center_delta_hours
            ),
            "era_data_valid_inst": era.data_valid_inst,
            "era_tp_valid": era.tp_valid,
            "era_trajectory_window_mask": era.trajectory_window_mask,
            "era_temporal_access_mask": era.temporal_access_mask,
            "era_source_state": np.asarray(
                [era.era_present, era.tp_present], dtype=np.bool_
            ),
            "target_static": self._static_fields,
            "target_static_coverage": self._static_coverage,
            "lead_hours": np.asarray(float(row.lead_hours), dtype=np.float32),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(
                    f"{name} shape mismatch: expected {shape}, got "
                    f"{arrays[name].shape}"
                )
        for name in (
            "target_history",
            "context_history",
            "target_future_benchmark_only",
            "era_values",
            "era_delta_hours",
            "era_tp_interval_center_delta_hours",
            "target_static",
            "lead_hours",
        ):
            if arrays[name].dtype != np.dtype(np.float32):
                raise TypeError(f"{name} must remain float32")
        logical_bytes = sum(array.nbytes for array in arrays.values())
        return {
            "tensors": arrays,
            "identity": (
                row.sample_id,
                t0.isoformat(),
                float(row.lead_hours),
                signature.key,
            ),
            "read_seconds": read_seconds,
            "logical_bytes": logical_bytes,
            "worker_pid": os.getpid(),
        }

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_radar_cache"] = None
        state["_era5_cache"] = None
        state["_static_fields"] = None
        state["_static_coverage"] = None
        state["_opened_pid"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown varies.
        try:
            self._close_handles()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class BenchmarkBatch:
    tensors: Mapping[str, Tensor]
    identities: tuple[tuple[str, str, float, str], ...]
    read_seconds: Tensor
    logical_bytes: Tensor
    worker_pids: Tensor

    @property
    def batch_size(self) -> int:
        return len(self.identities)

    def pin_memory(self) -> "BenchmarkBatch":
        return BenchmarkBatch(
            tensors={name: tensor.pin_memory() for name, tensor in self.tensors.items()},
            identities=self.identities,
            read_seconds=self.read_seconds.pin_memory(),
            logical_bytes=self.logical_bytes.pin_memory(),
            worker_pids=self.worker_pids.pin_memory(),
        )


def collate_benchmark_samples(samples: Sequence[dict[str, object]]) -> BenchmarkBatch:
    if not samples:
        raise ValueError("cannot collate an empty benchmark batch")
    first_tensors = samples[0]["tensors"]
    if not isinstance(first_tensors, Mapping):
        raise TypeError("sample tensors must be a mapping")
    names = tuple(first_tensors)
    tensors: dict[str, Tensor] = {}
    for name in names:
        values: list[np.ndarray] = []
        for sample in samples:
            mapping = sample["tensors"]
            if not isinstance(mapping, Mapping) or tuple(mapping) != names:
                raise ValueError("all samples must have the same tensor schema/order")
            value = mapping[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"sample tensor {name} must be a numpy array")
            values.append(value)
        tensors[name] = torch.from_numpy(np.stack(values, axis=0))
    identities = tuple(sample["identity"] for sample in samples)
    if not all(
        isinstance(identity, tuple) and len(identity) == 4
        for identity in identities
    ):
        raise TypeError("benchmark identities must be four-tuples")
    return BenchmarkBatch(
        tensors=tensors,
        identities=identities,  # type: ignore[arg-type]
        read_seconds=torch.tensor(
            [float(sample["read_seconds"]) for sample in samples],
            dtype=torch.float64,
        ),
        logical_bytes=torch.tensor(
            [int(sample["logical_bytes"]) for sample in samples],
            dtype=torch.int64,
        ),
        worker_pids=torch.tensor(
            [int(sample["worker_pid"]) for sample in samples],
            dtype=torch.int64,
        ),
    )


class CyclingSampler(Sampler[int]):
    """Deterministically repeat manifest rows to make every grid cell comparable."""

    def __init__(self, dataset_size: int, samples: int) -> None:
        if dataset_size <= 0 or samples <= 0:
            raise ValueError("cycling sampler sizes must be positive")
        self.dataset_size = int(dataset_size)
        self.samples = int(samples)

    def __iter__(self) -> Iterator[int]:
        return (index % self.dataset_size for index in range(self.samples))

    def __len__(self) -> int:
        return self.samples


def _worker_init(_worker_id: int) -> None:
    # DataLoader parallelism, not intra-worker OpenMP, is the benchmark axis.
    torch.set_num_threads(1)


def _rss_bytes(pid: int) -> int:
    try:
        with Path(f"/proc/{pid}/status").open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return 0
    return 0


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _move_tensors(
    batch: BenchmarkBatch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, tensor in batch.tensors.items():
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise TypeError(f"floating batch tensor {name} must be float32")
        result[name] = tensor.to(device=device, non_blocking=non_blocking)
    return result


def _shutdown_iterator(iterator: object | None) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def run_loader_candidate(
    spec: CacheDatasetSpec,
    *,
    batch_size: int,
    worker_count: int,
    prefetch_factor: int,
    warmup_batches: int,
    measured_batches: int,
    device: torch.device,
    pin_memory: bool = True,
    allow_cpu_test: bool = False,
) -> dict[str, object]:
    """Measure one batch/worker/prefetch cell with synchronized transfers."""

    if min(batch_size, prefetch_factor, measured_batches) <= 0:
        raise ValueError("batch, prefetch, and measured counts must be positive")
    if worker_count < 0 or warmup_batches < 0:
        raise ValueError("worker and warmup counts cannot be negative")
    if device.type != "cuda" and not allow_cpu_test:
        raise RuntimeError("CPU fallback is forbidden for the production benchmark")
    if device.type == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("benchmark requires exactly one visible CUDA GPU")
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    total_batches = warmup_batches + measured_batches
    dataset = CacheBenchmarkDataset(spec)
    sampler = CyclingSampler(len(dataset), total_batches * batch_size)
    loader_options: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": worker_count,
        "pin_memory": bool(pin_memory),
        "drop_last": True,
        "collate_fn": collate_benchmark_samples,
        "worker_init_fn": _worker_init,
    }
    if worker_count:
        loader_options.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            # CUDA is configured in the parent before measurement. Spawn
            # prevents workers inheriting CUDA runtime state and proves each
            # worker opens its own cache handles.
            multiprocessing_context="spawn",
        )
    loader = DataLoader(**loader_options)
    iterator: object | None = None
    device_batch: dict[str, Tensor] | None = None
    try:
        iterator = iter(loader)

        def fetch_and_transfer() -> tuple[BenchmarkBatch, float, float]:
            wait_started = time.perf_counter()
            batch = next(iterator)  # type: ignore[arg-type]
            data_wait = time.perf_counter() - wait_started
            if not isinstance(batch, BenchmarkBatch):
                raise TypeError("benchmark collate returned an unexpected batch")
            if device.type == "cuda":
                if pin_memory and not all(
                    tensor.is_pinned() for tensor in batch.tensors.values()
                ):
                    raise RuntimeError("DataLoader did not pin every H2D tensor")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                moved = _move_tensors(
                    batch, device=device, non_blocking=True
                )
                end.record()
                end.synchronize()
                transfer = float(start.elapsed_time(end)) / 1000.0
            else:
                transfer_started = time.perf_counter()
                moved = _move_tensors(
                    batch, device=device, non_blocking=False
                )
                transfer = time.perf_counter() - transfer_started
            nonlocal device_batch
            device_batch = moved
            return batch, data_wait, transfer

        for _ in range(warmup_batches):
            fetch_and_transfer()

        waits: list[float] = []
        transfers: list[float] = []
        worker_read_seconds = 0.0
        logical_bytes = 0
        h2d_bytes = 0
        worker_pids: set[int] = set()
        peak_total_rss = 0
        measurement_started = time.perf_counter()
        for _ in range(measured_batches):
            batch, data_wait, transfer = fetch_and_transfer()
            waits.append(data_wait)
            transfers.append(transfer)
            worker_read_seconds += float(batch.read_seconds.sum().item())
            logical_bytes += int(batch.logical_bytes.sum().item())
            h2d_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in batch.tensors.values()
            )
            worker_pids.update(map(int, batch.worker_pids.tolist()))
            main_pid = os.getpid()
            rss = _rss_bytes(main_pid)
            rss += sum(_rss_bytes(pid) for pid in worker_pids if pid != main_pid)
            peak_total_rss = max(peak_total_rss, rss)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - measurement_started
        samples = measured_batches * batch_size
        result: dict[str, object] = {
            "status": "ok",
            "batch_size": batch_size,
            "worker_count": worker_count,
            "prefetch_factor": prefetch_factor,
            "prefetch_effective": prefetch_factor if worker_count else None,
            "warmup_batches": warmup_batches,
            "measured_batches": measured_batches,
            "samples": samples,
            "wall_seconds": wall_seconds,
            "samples_per_second": samples / wall_seconds,
            "batches_per_second": measured_batches / wall_seconds,
            "logical_input_bytes": logical_bytes,
            "logical_input_gib_per_second": logical_bytes / 1024**3 / wall_seconds,
            "host_to_device_bytes": h2d_bytes,
            "data_wait_seconds": _distribution(waits),
            "h2d_seconds": _distribution(transfers),
            "worker_read_seconds_sum": worker_read_seconds,
            "main_rss_bytes": _rss_bytes(os.getpid()),
            "total_rss_peak_bytes": peak_total_rss,
            "observed_worker_pids": len(
                {pid for pid in worker_pids if pid != os.getpid()}
            ),
            "device": str(device),
            "precision": "float32",
            "tf32_enabled": False,
        }
        if device.type == "cuda":
            result.update(
                gpu_memory_allocated_bytes=torch.cuda.memory_allocated(device),
                gpu_memory_reserved_bytes=torch.cuda.memory_reserved(device),
                gpu_memory_allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
                gpu_memory_reserved_peak_bytes=torch.cuda.max_memory_reserved(device),
            )
        else:
            result.update(
                gpu_memory_allocated_bytes=0,
                gpu_memory_reserved_bytes=0,
                gpu_memory_allocated_peak_bytes=0,
                gpu_memory_reserved_peak_bytes=0,
            )
        return result
    finally:
        device_batch = None
        _shutdown_iterator(iterator)
        dataset.close()
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass(frozen=True, slots=True)
class BenchmarkGrid:
    batch_sizes: tuple[int, ...]
    worker_counts: tuple[int, ...]
    prefetch_factors: tuple[int, ...]
    warmup_batches: int
    measured_batches: int

    def __post_init__(self) -> None:
        batch_sizes = _integer_tuple("batch_sizes", self.batch_sizes)
        prefetch = _integer_tuple("prefetch_factors", self.prefetch_factors)
        workers: list[int] = []
        for value in self.worker_counts:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("worker_counts must contain integers")
            if int(value) < 0:
                raise ValueError("worker_counts cannot be negative")
            workers.append(int(value))
        if not workers:
            raise ValueError("worker_counts cannot be empty")
        if any(value & (value - 1) for value in batch_sizes):
            raise ValueError("batch_sizes must be powers of two")
        if len(set(batch_sizes)) != len(batch_sizes):
            raise ValueError("batch_sizes must be unique")
        if len(set(workers)) != len(workers):
            raise ValueError("worker_counts must be unique")
        if len(set(prefetch)) != len(prefetch):
            raise ValueError("prefetch_factors must be unique")
        if self.warmup_batches < 0 or self.measured_batches <= 0:
            raise ValueError("invalid warmup/measured batch count")
        object.__setattr__(self, "batch_sizes", batch_sizes)
        object.__setattr__(self, "worker_counts", tuple(workers))
        object.__setattr__(self, "prefetch_factors", prefetch)


def run_benchmark_grid(
    spec: CacheDatasetSpec,
    grid: BenchmarkGrid,
    *,
    device: torch.device,
    pin_memory: bool = True,
    allow_cpu_test: bool = False,
) -> list[dict[str, object]]:
    """Run every declared grid cell; OOM is recorded without changing shapes."""

    results: list[dict[str, object]] = []
    for batch_size in grid.batch_sizes:
        for worker_count in grid.worker_counts:
            for prefetch_factor in grid.prefetch_factors:
                try:
                    result = run_loader_candidate(
                        spec,
                        batch_size=batch_size,
                        worker_count=worker_count,
                        prefetch_factor=prefetch_factor,
                        warmup_batches=grid.warmup_batches,
                        measured_batches=grid.measured_batches,
                        device=device,
                        pin_memory=pin_memory,
                        allow_cpu_test=allow_cpu_test,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    result = {
                        "status": "oom",
                        "batch_size": batch_size,
                        "worker_count": worker_count,
                        "prefetch_factor": prefetch_factor,
                        "error_type": type(error).__name__,
                        "error": "CUDA out of memory at unchanged full-width contract",
                    }
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                results.append(result)
    if not any(result.get("status") == "ok" for result in results):
        raise RuntimeError("every benchmark grid cell failed or exhausted memory")
    return results


__all__ = [
    "BenchmarkBatch",
    "BenchmarkGrid",
    "CacheBenchmarkContract",
    "CacheBenchmarkDataset",
    "CacheDatasetSpec",
    "CyclingSampler",
    "PRODUCTION_CONTEXT_WIDTHS",
    "PRODUCTION_ERA_LATENT_CHANNELS",
    "PRODUCTION_ERA_SHAPE",
    "PRODUCTION_RADAR_SHAPE",
    "PRODUCTION_TARGET_WIDTHS",
    "STATIC_FLOAT_CHANNELS",
    "collate_benchmark_samples",
    "load_static_model_inputs",
    "run_benchmark_grid",
    "run_loader_candidate",
    "validate_cache_artifacts",
]
