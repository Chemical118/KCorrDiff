"""Full Stage 2 factory, collation, pinning, and H2D benchmark.

Unlike :mod:`kcorrdiff.training.cache_benchmark`, this benchmark measures the
model-facing production object.  Every measured batch is decoded by
``LazyRankSlotDataset``, context-regridded by ``KCorrDiffDataset``, normalized
and nested by ``TrainingBatchCollator``, pinned through
``RankLocalBatch.pin_memory()``, and copied to the GPU without blocking.

The grid runner is deliberately fail-closed.  It materializes one exact CPU
batch for every candidate before starting workers, measures its tensor
storages, and rejects candidates whose conservative worker/prefetch estimate
exceeds the declared host or ``/dev/shm`` budget.  It never changes shape,
precision, batch size, or worker count after a failure.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from kcorrdiff.data.advection import AdvectionGeometry
from kcorrdiff.training.batch import TrainingBatchCollator
from kcorrdiff.training.benchmark_loader import (
    configure_cuda_device,
    derive_jsonl_path,
    preflight_output_paths,
    publish_results_atomic,
)
from kcorrdiff.training.config import Stage2Config, load_stage2_config
from kcorrdiff.training.data_factory import (
    FactoryInputs,
    KCorrDiffDataFactory,
    PlannedSample,
    RankLocalBatch,
    RankSlotCollator,
    build_data_factory,
)
from kcorrdiff.training.plan import DistributedDrawPlan
from kcorrdiff.training.runtime import (
    CONTEXT_WIDTHS,
    ERA_GRID_SIZE,
    ERA_LATENT_CHANNELS,
    TARGET_WIDTHS,
)


SUMMARY_FORMAT = "kcorrdiff.production-loader-benchmark-summary.v1"
RESULT_FORMAT = "kcorrdiff.production-loader-benchmark-result.v1"
PRODUCTION_RADAR_SHAPE = (256, 256)
PRODUCTION_ERA_SHAPE = (33, 33)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FALLBACK_ENVIRONMENT = (
    "KCORRDIFF_ALLOW_CPU_FALLBACK",
    "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK",
    "KCORRDIFF_ALLOW_PRECISION_FALLBACK",
    "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK",
)


def _strict_positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_csv(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not pieces or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    try:
        result = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from error
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("all values must be positive")
    return result


def _nonnegative_csv(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not pieces or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "expected comma-separated non-negative integers"
        )
    try:
        result = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated non-negative integers"
        ) from error
    if any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("all values must be non-negative")
    return result


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LoaderCandidate:
    """One unchanged batch/worker/prefetch grid cell."""

    batch_size: int
    worker_count: int
    prefetch_factor: int

    def __post_init__(self) -> None:
        _strict_positive_integer(self.batch_size, name="batch_size")
        _strict_nonnegative_integer(self.worker_count, name="worker_count")
        _strict_positive_integer(self.prefetch_factor, name="prefetch_factor")

    @property
    def prefetch_effective(self) -> int | None:
        return self.prefetch_factor if self.worker_count else None


@dataclass(frozen=True, slots=True)
class ProductionBenchmarkGrid:
    batch_sizes: tuple[int, ...]
    worker_counts: tuple[int, ...]
    prefetch_factors: tuple[int, ...]
    warmup_batches: int
    measured_batches: int

    def __post_init__(self) -> None:
        if not self.batch_sizes or not self.worker_counts or not self.prefetch_factors:
            raise ValueError("production benchmark grid axes cannot be empty")
        for value in self.batch_sizes:
            _strict_positive_integer(value, name="batch_sizes item")
            if value & (value - 1):
                raise ValueError("batch_sizes must be powers of two")
        for value in self.worker_counts:
            _strict_nonnegative_integer(value, name="worker_counts item")
        for value in self.prefetch_factors:
            _strict_positive_integer(value, name="prefetch_factors item")
        for name, values in (
            ("batch_sizes", self.batch_sizes),
            ("worker_counts", self.worker_counts),
            ("prefetch_factors", self.prefetch_factors),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        _strict_nonnegative_integer(self.warmup_batches, name="warmup_batches")
        _strict_positive_integer(self.measured_batches, name="measured_batches")

    @property
    def cell_count(self) -> int:
        return (
            len(self.batch_sizes)
            * len(self.worker_counts)
            * len(self.prefetch_factors)
        )

    def candidates(self) -> tuple[LoaderCandidate, ...]:
        return tuple(
            LoaderCandidate(batch_size, workers, prefetch)
            for batch_size in self.batch_sizes
            for workers in self.worker_counts
            for prefetch in self.prefetch_factors
        )


@dataclass(frozen=True, slots=True)
class PipelinePayloadLimits:
    """Explicit upper bounds applied before workers or pinned queues exist."""

    maximum_grid_cells: int
    maximum_host_pipeline_bytes: int
    maximum_shm_pipeline_bytes: int
    safety_factor: float = 1.25

    def __post_init__(self) -> None:
        _strict_positive_integer(self.maximum_grid_cells, name="maximum_grid_cells")
        _strict_positive_integer(
            self.maximum_host_pipeline_bytes,
            name="maximum_host_pipeline_bytes",
        )
        _strict_positive_integer(
            self.maximum_shm_pipeline_bytes,
            name="maximum_shm_pipeline_bytes",
        )
        if not math.isfinite(self.safety_factor) or self.safety_factor < 1.0:
            raise ValueError("payload safety_factor must be finite and >= 1")


@dataclass(frozen=True, slots=True)
class StrictBenchmarkContract:
    device: str
    precision: str
    tf32_enabled: bool
    fail_on_fallback: bool
    pin_memory: bool
    logical_world_size: int
    rank: int
    role: str
    fold_id: int | None
    grid: ProductionBenchmarkGrid
    payload_limits: PipelinePayloadLimits


@dataclass(frozen=True, slots=True)
class TensorTreeMetrics:
    """Logical tensor references and deduplicated allocation measurements."""

    tensor_references: int
    unique_tensor_objects: int
    unique_storages: int
    logical_bytes: int
    storage_bytes: int
    pinned_tensor_objects: int
    dtype_tensor_counts: Mapping[str, int]
    device_tensor_counts: Mapping[str, int]

    @property
    def all_cpu_tensors_pinned(self) -> bool:
        cpu = int(self.device_tensor_counts.get("cpu", 0))
        return cpu == 0 or self.pinned_tensor_objects == cpu

    def as_json(self) -> dict[str, object]:
        return {
            "tensor_references": self.tensor_references,
            "unique_tensor_objects": self.unique_tensor_objects,
            "unique_storages": self.unique_storages,
            "logical_bytes": self.logical_bytes,
            "storage_bytes": self.storage_bytes,
            "pinned_tensor_objects": self.pinned_tensor_objects,
            "all_cpu_tensors_pinned": self.all_cpu_tensors_pinned,
            "dtype_tensor_counts": dict(self.dtype_tensor_counts),
            "device_tensor_counts": dict(self.device_tensor_counts),
        }


def _walk_tensor_references(value: object, active: set[int]) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
        return
    if isinstance(value, AdvectionGeometry):
        # These authoritative CPU float64 axes are immutable kernel metadata.
        # ``AdvectionModelInputs.to`` deliberately reuses them and the kernel
        # casts them to the FP32 rate tensor device/dtype when building its
        # mesh.  They are neither pinned nor transferred, so counting them as
        # H2D payload would falsely report a precision/pinning fallback.
        return
    if value is None or isinstance(value, (str, bytes, int, float, bool, Path)):
        return
    identity = id(value)
    if identity in active:
        return
    if is_dataclass(value) and not isinstance(value, type):
        active.add(identity)
        try:
            for field in fields(value):
                yield from _walk_tensor_references(getattr(value, field.name), active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, Mapping):
        active.add(identity)
        try:
            for item in value.values():
                yield from _walk_tensor_references(item, active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, (tuple, list)):
        active.add(identity)
        try:
            for item in value:
                yield from _walk_tensor_references(item, active)
        finally:
            active.remove(identity)


def tensor_tree_metrics(value: object) -> TensorTreeMetrics:
    """Count nested tensors and unique backing storages without copying them."""

    references = tuple(_walk_tensor_references(value, set()))
    unique: dict[int, Tensor] = {}
    for tensor in references:
        unique.setdefault(id(tensor), tensor)
    storage_sizes: dict[tuple[str, int, int], int] = {}
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    pinned = 0
    logical_bytes = 0
    for tensor in unique.values():
        logical_bytes += tensor.numel() * tensor.element_size()
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        device_name = str(tensor.device)
        device_counts[device_name] = device_counts.get(device_name, 0) + 1
        if tensor.device.type == "cpu" and tensor.is_pinned():
            pinned += 1
        storage = tensor.untyped_storage()
        storage_size = int(storage.nbytes())
        storage_key = (device_name, int(storage.data_ptr()), storage_size)
        storage_sizes.setdefault(storage_key, storage_size)
    return TensorTreeMetrics(
        tensor_references=len(references),
        unique_tensor_objects=len(unique),
        unique_storages=len(storage_sizes),
        logical_bytes=logical_bytes,
        storage_bytes=sum(storage_sizes.values()),
        pinned_tensor_objects=pinned,
        dtype_tensor_counts=dict(sorted(dtype_counts.items())),
        device_tensor_counts=dict(sorted(device_counts.items())),
    )


def require_float32_tensor_tree(value: object) -> TensorTreeMetrics:
    """Reject any floating payload tensor which is not exact FP32."""

    metrics = tensor_tree_metrics(value)
    offenders = sorted(
        dtype
        for dtype in metrics.dtype_tensor_counts
        if dtype.startswith(("float", "bfloat", "complex")) and dtype != "float32"
    )
    if offenders:
        raise TypeError(f"production payload contains non-float32 tensors: {offenders}")
    return metrics


@dataclass(frozen=True, slots=True)
class ShmUsage:
    path: str
    total_bytes: int
    used_bytes: int
    available_bytes: int

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def read_shm_usage(path: Path = Path("/dev/shm")) -> ShmUsage:
    selected = path.resolve()
    statistics = os.statvfs(selected)
    total = int(statistics.f_blocks * statistics.f_frsize)
    available = int(statistics.f_bavail * statistics.f_frsize)
    return ShmUsage(
        path=str(selected),
        total_bytes=total,
        used_bytes=max(0, total - available),
        available_bytes=available,
    )


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


@dataclass(frozen=True, slots=True)
class PipelinePayloadEstimate:
    candidate: LoaderCandidate
    tensor_metrics: TensorTreeMetrics
    worker_prefetch_batches: int
    pinned_queue_batches: int
    estimated_shm_bytes: int
    estimated_host_pipeline_bytes: int
    shm_snapshot: ShmUsage
    approved: bool
    rejection_reasons: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "candidate": asdict(self.candidate),
            "tensor_metrics": self.tensor_metrics.as_json(),
            "worker_prefetch_batches": self.worker_prefetch_batches,
            "pinned_queue_batches": self.pinned_queue_batches,
            "estimated_shm_bytes": self.estimated_shm_bytes,
            "estimated_host_pipeline_bytes": self.estimated_host_pipeline_bytes,
            "shm_snapshot": self.shm_snapshot.as_json(),
            "approved": self.approved,
            "rejection_reasons": list(self.rejection_reasons),
        }


def estimate_pipeline_payload(
    metrics: TensorTreeMetrics,
    candidate: LoaderCandidate,
    *,
    limits: PipelinePayloadLimits,
    shm_snapshot: ShmUsage,
    pin_memory: bool,
) -> PipelinePayloadEstimate:
    """Conservatively bound shared and pinned copies of one exact batch."""

    worker_prefetch = (
        candidate.worker_count * candidate.prefetch_factor
        if candidate.worker_count
        else 0
    )
    # The pin thread may hold one batch while the training loop retains the
    # previous/current batch.  Counting two pinned batches is conservative and
    # keeps the estimate independent of implementation-specific queue depths.
    pinned_queue = 2 if pin_memory else 0
    bytes_per_batch = metrics.storage_bytes
    estimated_shm = math.ceil(
        bytes_per_batch * worker_prefetch * limits.safety_factor
    )
    resident_batches = 1 + worker_prefetch + pinned_queue
    estimated_host = math.ceil(
        bytes_per_batch * resident_batches * limits.safety_factor
    )
    reasons: list[str] = []
    if estimated_shm > limits.maximum_shm_pipeline_bytes:
        reasons.append("declared maximum_shm_pipeline_bytes exceeded")
    if estimated_shm > shm_snapshot.available_bytes:
        reasons.append("currently available /dev/shm bytes exceeded")
    if estimated_host > limits.maximum_host_pipeline_bytes:
        reasons.append("declared maximum_host_pipeline_bytes exceeded")
    return PipelinePayloadEstimate(
        candidate=candidate,
        tensor_metrics=metrics,
        worker_prefetch_batches=worker_prefetch,
        pinned_queue_batches=pinned_queue,
        estimated_shm_bytes=estimated_shm,
        estimated_host_pipeline_bytes=estimated_host,
        shm_snapshot=shm_snapshot,
        approved=not reasons,
        rejection_reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class TimedRankLocalBatch:
    """Production rank batch plus worker-side collate and pin measurements."""

    rank_local: RankLocalBatch
    collate_seconds: float
    worker_pid: int
    pin_seconds: float = 0.0
    pin_applied: bool = False

    def __post_init__(self) -> None:
        if self.collate_seconds < 0.0 or self.pin_seconds < 0.0:
            raise ValueError("collate/pin durations cannot be negative")
        if self.worker_pid <= 0:
            raise ValueError("worker_pid must be positive")

    def pin_memory(self) -> "TimedRankLocalBatch":
        started = time.perf_counter()
        pinned = self.rank_local.pin_memory()
        elapsed = time.perf_counter() - started
        return TimedRankLocalBatch(
            rank_local=pinned,
            collate_seconds=self.collate_seconds,
            worker_pid=self.worker_pid,
            pin_seconds=elapsed,
            pin_applied=True,
        )


class TimedRankSlotCollator:
    """Time the exact production nested collator without changing its result."""

    def __init__(self, sample_collator: Callable[[Sequence[object]], object]) -> None:
        self._collator = RankSlotCollator(sample_collator)

    def __call__(self, values: Sequence[PlannedSample]) -> TimedRankLocalBatch:
        started = time.perf_counter()
        batch = self._collator(values)
        return TimedRankLocalBatch(
            rank_local=batch,
            collate_seconds=time.perf_counter() - started,
            worker_pid=os.getpid(),
        )


def _worker_init(_worker_id: int) -> None:
    # Loader workers, rather than nested OpenMP pools, are the tuning axis.
    torch.set_num_threads(1)


class CandidateLoaderBuilder(Protocol):
    def __call__(
        self,
        factory: KCorrDiffDataFactory,
        plan: DistributedDrawPlan,
        *,
        rank: int,
        candidate: LoaderCandidate,
        pin_memory: bool,
        maximum_batches: int,
    ) -> Iterable[TimedRankLocalBatch]: ...


def build_candidate_loader(
    factory: KCorrDiffDataFactory,
    plan: DistributedDrawPlan,
    *,
    rank: int,
    candidate: LoaderCandidate,
    pin_memory: bool,
    maximum_batches: int,
) -> DataLoader[TimedRankLocalBatch]:
    """Build an exactly bounded candidate around the real lazy rank dataset.

    Bounding the subset is operationally important: workers reach the dataset
    end after the last measured batch instead of being interrupted while
    serializing a large prefetched tensor tree.  This removes an artificial
    early-break shutdown path that production epochs do not exercise.
    """

    _strict_positive_integer(maximum_batches, name="maximum_batches")
    rank_dataset = factory.rank_dataset(plan, rank=rank)
    required_items = candidate.batch_size * maximum_batches
    if required_items > len(rank_dataset):
        raise ValueError("bounded candidate exceeds the rank-local dataset")
    dataset = Subset(rank_dataset, range(required_items))
    sample_collator = TrainingBatchCollator(
        factory.artifacts.geometry,
        era_normalization=factory.artifacts.era_normalization,
        device="cpu",
    )
    options: dict[str, object] = {
        "dataset": dataset,
        "batch_size": candidate.batch_size,
        "shuffle": False,
        "drop_last": True,
        "num_workers": candidate.worker_count,
        "pin_memory": pin_memory,
        "collate_fn": TimedRankSlotCollator(sample_collator),
        "worker_init_fn": _worker_init,
    }
    if candidate.worker_count:
        options.update(
            persistent_workers=True,
            prefetch_factor=candidate.prefetch_factor,
            multiprocessing_context="spawn",
        )
    return DataLoader(**options)


def _shutdown_iterator(iterator: object | None) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _iterator_worker_pids(iterator: object | None) -> set[int]:
    workers = getattr(iterator, "_workers", ())
    result: set[int] = set()
    for worker in workers or ():
        pid = getattr(worker, "pid", None)
        if isinstance(pid, int) and pid > 0:
            result.add(pid)
    return result


def _close_loader_dataset(loader: object) -> None:
    dataset = getattr(loader, "dataset", None)
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    close = getattr(dataset, "_close", None)
    if callable(close):
        close()


def _validate_candidate_capacity(
    plan: DistributedDrawPlan,
    *,
    rank: int,
    candidate: LoaderCandidate,
    total_batches: int,
) -> None:
    if not 0 <= rank < plan.world_size:
        raise ValueError("benchmark rank is outside the draw plan")
    required_slots = candidate.batch_size * total_batches
    slots = plan.rank_slots[rank]
    if len(slots) < required_slots:
        raise ValueError(
            "rank-local draw plan has too few slots for the bounded benchmark: "
            f"required={required_slots}, available={len(slots)}"
        )
    if any(slot.is_padding for slot in slots[:required_slots]):
        raise ValueError("benchmark window contains an explicit padding sentinel")


def _validate_measured_batch(
    timed: TimedRankLocalBatch,
    *,
    expected_batch_size: int,
) -> TensorTreeMetrics:
    rank_batch = timed.rank_local
    if rank_batch.is_zero_loss_sentinel or rank_batch.padding_count:
        raise ValueError("production benchmark cannot measure padding sentinels")
    if len(rank_batch.active_slot_indices) != expected_batch_size:
        raise ValueError("collated active batch size changed from the candidate")
    return require_float32_tensor_tree(rank_batch)


@dataclass(frozen=True, slots=True)
class CandidatePreflight:
    payload: PipelinePayloadEstimate
    loader_construction_seconds: float
    iterator_construction_seconds: float
    first_batch_wait_seconds: float

    @property
    def approved(self) -> bool:
        return self.payload.approved

    def as_json(self) -> dict[str, object]:
        return {
            "payload": self.payload.as_json(),
            "loader_construction_seconds": self.loader_construction_seconds,
            "iterator_construction_seconds": self.iterator_construction_seconds,
            "first_batch_wait_seconds": self.first_batch_wait_seconds,
        }


def preflight_candidate_payload(
    factory: KCorrDiffDataFactory,
    plan: DistributedDrawPlan,
    candidate: LoaderCandidate,
    *,
    rank: int,
    limits: PipelinePayloadLimits,
    pin_memory: bool,
    loader_builder: CandidateLoaderBuilder = build_candidate_loader,
    shm_probe: Callable[[], ShmUsage] = read_shm_usage,
) -> CandidatePreflight:
    """Materialize one exact unpinned batch and bound candidate queue memory."""

    _validate_candidate_capacity(plan, rank=rank, candidate=candidate, total_batches=1)
    serial_candidate = LoaderCandidate(
        batch_size=candidate.batch_size,
        worker_count=0,
        prefetch_factor=candidate.prefetch_factor,
    )
    loader_started = time.perf_counter()
    loader = loader_builder(
        factory,
        plan,
        rank=rank,
        candidate=serial_candidate,
        pin_memory=False,
        maximum_batches=1,
    )
    loader_construction = time.perf_counter() - loader_started
    iterator: object | None = None
    try:
        iterator_started = time.perf_counter()
        iterator = iter(loader)
        iterator_construction = time.perf_counter() - iterator_started
        fetch_started = time.perf_counter()
        timed = next(iterator)  # type: ignore[arg-type]
        wait = time.perf_counter() - fetch_started
        if not isinstance(timed, TimedRankLocalBatch):
            raise TypeError("candidate loader must return TimedRankLocalBatch")
        metrics = _validate_measured_batch(
            timed, expected_batch_size=candidate.batch_size
        )
        estimate = estimate_pipeline_payload(
            metrics,
            candidate,
            limits=limits,
            shm_snapshot=shm_probe(),
            pin_memory=pin_memory,
        )
        return CandidatePreflight(
            payload=estimate,
            loader_construction_seconds=loader_construction,
            iterator_construction_seconds=iterator_construction,
            first_batch_wait_seconds=wait,
        )
    finally:
        _shutdown_iterator(iterator)
        _close_loader_dataset(loader)
        del loader
        gc.collect()


@dataclass(frozen=True, slots=True)
class DeviceRankPayload:
    training: object
    loss_weights: Tensor


def move_rank_local_batch(
    batch: RankLocalBatch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> DeviceRankPayload:
    """Move the production training object and complete plan weights together."""

    if batch.training is None:
        raise ValueError("cannot transfer an all-padding rank batch")
    move = getattr(batch.training, "to", None)
    if not callable(move):
        raise TypeError("RankLocalBatch.training must implement to(device)")
    training = move(device, non_blocking=non_blocking)
    weights = batch.loss_weights.to(device=device, non_blocking=non_blocking)
    payload = DeviceRankPayload(training=training, loss_weights=weights)
    require_float32_tensor_tree(payload)
    return payload


def _validate_runtime_device(
    device: torch.device,
    *,
    allow_cpu_test: bool,
) -> None:
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("PyTorch default dtype must remain float32")
    if device.type == "cpu":
        if not allow_cpu_test:
            raise RuntimeError("CPU fallback is forbidden for production benchmarking")
        return
    if device != torch.device("cuda:0"):
        raise RuntimeError("production loader benchmark requires cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible CUDA GPU")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled")


def _timed_transfer(
    batch: RankLocalBatch,
    *,
    device: torch.device,
) -> tuple[DeviceRankPayload, float]:
    if device.type == "cuda":
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        moved = move_rank_local_batch(
            batch,
            device=device,
            non_blocking=True,
        )
        finished.record()
        finished.synchronize()
        seconds = float(started.elapsed_time(finished)) / 1000.0
    else:
        started_cpu = time.perf_counter()
        moved = move_rank_local_batch(
            batch,
            device=device,
            non_blocking=False,
        )
        seconds = time.perf_counter() - started_cpu
    return moved, seconds


def run_production_candidate(
    factory: KCorrDiffDataFactory,
    plan: DistributedDrawPlan,
    candidate: LoaderCandidate,
    *,
    rank: int,
    warmup_batches: int,
    measured_batches: int,
    device: torch.device,
    preflight: CandidatePreflight,
    pin_memory: bool = True,
    allow_cpu_test: bool = False,
    loader_builder: CandidateLoaderBuilder = build_candidate_loader,
    shm_probe: Callable[[], ShmUsage] = read_shm_usage,
) -> dict[str, object]:
    """Measure one approved, unchanged production pipeline candidate."""

    _strict_nonnegative_integer(warmup_batches, name="warmup_batches")
    _strict_positive_integer(measured_batches, name="measured_batches")
    _validate_runtime_device(device, allow_cpu_test=allow_cpu_test)
    if not preflight.approved:
        raise ValueError("cannot execute a candidate rejected by payload preflight")
    if preflight.payload.candidate != candidate:
        raise ValueError("candidate and payload preflight identity disagree")
    if device.type == "cuda" and not pin_memory:
        raise ValueError("production CUDA benchmark requires pinned host memory")
    total_batches = warmup_batches + measured_batches
    _validate_candidate_capacity(
        plan,
        rank=rank,
        candidate=candidate,
        total_batches=total_batches,
    )

    candidate_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    shm_before = shm_probe()
    loader_started = time.perf_counter()
    loader = loader_builder(
        factory,
        plan,
        rank=rank,
        candidate=candidate,
        pin_memory=pin_memory,
        maximum_batches=total_batches,
    )
    loader_construction = time.perf_counter() - loader_started
    iterator: object | None = None
    device_payload: DeviceRankPayload | None = None
    try:
        iterator_started = time.perf_counter()
        iterator = iter(loader)
        iterator_construction = time.perf_counter() - iterator_started

        def fetch_and_transfer() -> tuple[
            TimedRankLocalBatch,
            TensorTreeMetrics,
            TensorTreeMetrics,
            float,
            float,
        ]:
            wait_started = time.perf_counter()
            timed = next(iterator)  # type: ignore[arg-type]
            data_wait = time.perf_counter() - wait_started
            if not isinstance(timed, TimedRankLocalBatch):
                raise TypeError("candidate loader must return TimedRankLocalBatch")
            host_metrics = _validate_measured_batch(
                timed, expected_batch_size=candidate.batch_size
            )
            if device.type == "cuda" and pin_memory:
                if not timed.pin_applied:
                    raise RuntimeError("DataLoader did not invoke RankLocalBatch.pin_memory")
                if not host_metrics.all_cpu_tensors_pinned:
                    raise RuntimeError("not every production H2D tensor is pinned")
            moved, transfer_seconds = _timed_transfer(
                timed.rank_local,
                device=device,
            )
            device_metrics = require_float32_tensor_tree(moved)
            nonlocal device_payload
            device_payload = moved
            return (
                timed,
                host_metrics,
                device_metrics,
                data_wait,
                transfer_seconds,
            )

        for _ in range(warmup_batches):
            fetch_and_transfer()

        waits: list[float] = []
        collates: list[float] = []
        pins: list[float] = []
        transfers: list[float] = []
        worker_pids: set[int] = _iterator_worker_pids(iterator)
        host_signature: tuple[int, int, int, int] | None = None
        device_signature: tuple[int, int, int, int] | None = None
        first_host_metrics: TensorTreeMetrics | None = None
        first_device_metrics: TensorTreeMetrics | None = None
        peak_total_rss = 0
        shm_used_peak = shm_before.used_bytes
        samples = 0
        host_logical_bytes = 0
        host_storage_bytes = 0
        h2d_logical_bytes = 0
        measurement_started = time.perf_counter()
        for _ in range(measured_batches):
            timed, host_metrics, device_metrics, wait, transfer = (
                fetch_and_transfer()
            )
            waits.append(wait)
            collates.append(timed.collate_seconds)
            pins.append(timed.pin_seconds)
            transfers.append(transfer)
            worker_pids.add(timed.worker_pid)
            worker_pids.update(_iterator_worker_pids(iterator))
            samples += len(timed.rank_local.active_slot_indices)
            host_logical_bytes += host_metrics.logical_bytes
            host_storage_bytes += host_metrics.storage_bytes
            h2d_logical_bytes += device_metrics.logical_bytes
            current_host_signature = (
                host_metrics.unique_tensor_objects,
                host_metrics.unique_storages,
                host_metrics.logical_bytes,
                host_metrics.storage_bytes,
            )
            current_device_signature = (
                device_metrics.unique_tensor_objects,
                device_metrics.unique_storages,
                device_metrics.logical_bytes,
                device_metrics.storage_bytes,
            )
            if host_signature is None:
                host_signature = current_host_signature
                device_signature = current_device_signature
                first_host_metrics = host_metrics
                first_device_metrics = device_metrics
            elif (
                host_signature != current_host_signature
                or device_signature != current_device_signature
            ):
                raise RuntimeError("production tensor payload changed between batches")
            main_pid = os.getpid()
            rss = _rss_bytes(main_pid) + sum(
                _rss_bytes(pid) for pid in worker_pids if pid != main_pid
            )
            peak_total_rss = max(peak_total_rss, rss)
            shm_used_peak = max(shm_used_peak, shm_probe().used_bytes)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - measurement_started
        if first_host_metrics is None or first_device_metrics is None:
            raise AssertionError("measurement loop returned no batch")
        expected_samples = measured_batches * candidate.batch_size
        if samples != expected_samples:
            raise AssertionError("measured sample count changed from bounded grid")
        shm_after = shm_probe()
        result: dict[str, object] = {
            "format_version": RESULT_FORMAT,
            "status": "ok",
            **asdict(candidate),
            "prefetch_effective": candidate.prefetch_effective,
            "rank": rank,
            "warmup_batches": warmup_batches,
            "measured_batches": measured_batches,
            "samples": samples,
            "wall_seconds": wall_seconds,
            "samples_per_second": samples / wall_seconds,
            "batches_per_second": measured_batches / wall_seconds,
            "loader_construction_seconds": loader_construction,
            "iterator_construction_seconds": iterator_construction,
            "candidate_elapsed_seconds": time.perf_counter() - candidate_started,
            "data_wait_seconds": _distribution(waits),
            "worker_collate_seconds": _distribution(collates),
            "pin_memory_seconds": _distribution(pins),
            "h2d_seconds": _distribution(transfers),
            "host_batch_tensor_metrics": first_host_metrics.as_json(),
            "device_batch_tensor_metrics": first_device_metrics.as_json(),
            "host_logical_bytes": host_logical_bytes,
            "host_storage_bytes": host_storage_bytes,
            "host_logical_gib_per_second": (
                host_logical_bytes / 1024**3 / wall_seconds
            ),
            "host_storage_gib_per_second": (
                host_storage_bytes / 1024**3 / wall_seconds
            ),
            "host_to_device_logical_bytes": h2d_logical_bytes,
            "host_to_device_gib_per_second": (
                h2d_logical_bytes / 1024**3 / wall_seconds
            ),
            "main_rss_bytes": _rss_bytes(os.getpid()),
            "total_rss_peak_bytes": peak_total_rss,
            "observed_worker_pids": len(
                {pid for pid in worker_pids if pid != os.getpid()}
            ),
            "shm": {
                "before": shm_before.as_json(),
                "after": shm_after.as_json(),
                "used_peak_bytes": shm_used_peak,
                "used_peak_delta_bytes": max(
                    0, shm_used_peak - shm_before.used_bytes
                ),
            },
            "payload_preflight": preflight.as_json(),
            "device": str(device),
            "precision": "float32",
            "tf32_enabled": False,
            "pin_memory": pin_memory,
            "non_blocking_h2d": device.type == "cuda",
        }
        if device.type == "cuda":
            result.update(
                gpu_memory_allocated_bytes=torch.cuda.memory_allocated(device),
                gpu_memory_reserved_bytes=torch.cuda.memory_reserved(device),
                gpu_memory_allocated_peak_bytes=torch.cuda.max_memory_allocated(
                    device
                ),
                gpu_memory_reserved_peak_bytes=torch.cuda.max_memory_reserved(
                    device
                ),
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
        device_payload = None
        _shutdown_iterator(iterator)
        _close_loader_dataset(loader)
        del loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_production_grid(
    factory: KCorrDiffDataFactory,
    plan: DistributedDrawPlan,
    grid: ProductionBenchmarkGrid,
    *,
    rank: int,
    limits: PipelinePayloadLimits,
    device: torch.device,
    pin_memory: bool = True,
    allow_cpu_test: bool = False,
    loader_builder: CandidateLoaderBuilder = build_candidate_loader,
    shm_probe: Callable[[], ShmUsage] = read_shm_usage,
    require_success: bool = True,
    result_callback: Callable[[Mapping[str, object], int], None] | None = None,
) -> list[dict[str, object]]:
    """Preflight and execute every bounded cell without changing a candidate."""

    if grid.cell_count > limits.maximum_grid_cells:
        raise ValueError(
            "benchmark grid exceeds maximum_grid_cells: "
            f"{grid.cell_count} > {limits.maximum_grid_cells}"
        )
    candidates = grid.candidates()
    total_batches = grid.warmup_batches + grid.measured_batches
    for candidate in candidates:
        _validate_candidate_capacity(
            plan,
            rank=rank,
            candidate=candidate,
            total_batches=total_batches,
        )
    results: list[dict[str, object]] = []
    for candidate in candidates:
        preflight = preflight_candidate_payload(
            factory,
            plan,
            candidate,
            rank=rank,
            limits=limits,
            pin_memory=pin_memory,
            loader_builder=loader_builder,
            shm_probe=shm_probe,
        )
        if not preflight.approved:
            result = {
                "format_version": RESULT_FORMAT,
                "status": "preflight_rejected",
                **asdict(candidate),
                "payload_preflight": preflight.as_json(),
                "precision": "float32",
                "tf32_enabled": False,
                "fallback_used": False,
            }
            results.append(result)
            if result_callback is not None:
                result_callback(result, len(results) - 1)
            continue
        try:
            result = run_production_candidate(
                factory,
                plan,
                candidate,
                rank=rank,
                warmup_batches=grid.warmup_batches,
                measured_batches=grid.measured_batches,
                device=device,
                preflight=preflight,
                pin_memory=pin_memory,
                allow_cpu_test=allow_cpu_test,
                loader_builder=loader_builder,
                shm_probe=shm_probe,
            )
        except torch.cuda.OutOfMemoryError as error:
            result = {
                "format_version": RESULT_FORMAT,
                "status": "oom",
                **asdict(candidate),
                "error_type": type(error).__name__,
                "error": "CUDA out of memory at the unchanged production contract",
                "payload_preflight": preflight.as_json(),
                "precision": "float32",
                "tf32_enabled": False,
                "fallback_used": False,
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results.append(result)
        if result_callback is not None:
            result_callback(result, len(results) - 1)
    if require_success and not any(result.get("status") == "ok" for result in results):
        raise RuntimeError("production benchmark produced no successful grid cell")
    return results


@dataclass(frozen=True, slots=True)
class ExpectedArtifactHashes:
    candidate_manifest: str
    draw_manifest: str
    bundle_metadata: str
    target_coordinates: str
    context_coordinates: str

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field.name} must be a lowercase SHA-256 digest")


def verify_expected_manifest_hashes(
    inputs: FactoryInputs,
    expected: ExpectedArtifactHashes,
) -> dict[str, str]:
    """Verify the three large-manifest identities before factory construction."""

    actual = {
        "candidate_manifest": sha256_file(inputs.candidate_manifest),
        "draw_manifest": sha256_file(inputs.draw_manifest),
        "bundle_metadata": sha256_file(inputs.bundle_metadata),
    }
    wanted = {
        "candidate_manifest": expected.candidate_manifest,
        "draw_manifest": expected.draw_manifest,
        "bundle_metadata": expected.bundle_metadata,
    }
    mismatch = [name for name in actual if actual[name] != wanted[name]]
    if mismatch:
        raise ValueError("expected artifact SHA-256 mismatch: " + ", ".join(mismatch))
    return actual


@dataclass(frozen=True, slots=True)
class ProductionBenchmarkContext:
    config: Stage2Config
    inputs: FactoryInputs
    factory: KCorrDiffDataFactory
    plan: DistributedDrawPlan
    expected_hashes: ExpectedArtifactHashes
    validated_manifest_hashes: Mapping[str, str]
    config_load_seconds: float
    artifact_hash_preflight_seconds: float
    factory_construction_seconds: float


def build_production_benchmark_context(
    *,
    config_path: Path,
    radar_cache_root: Path,
    era5_cache_root: Path,
    target_static: Path,
    candidate_manifest: Path,
    draw_manifest: Path,
    bundle_metadata: Path,
    expected_hashes: ExpectedArtifactHashes,
    role: str,
    fold_id: int | None,
    logical_world_size: int,
    verify_cache_hashes: bool = False,
) -> ProductionBenchmarkContext:
    """Build the exact content-addressed factory and logical DDP draw plan."""

    config_started = time.perf_counter()
    config = load_stage2_config(config_path)
    config_seconds = time.perf_counter() - config_started
    if config.runtime.precision != "float32" or config.runtime.tf32:
        raise ValueError("Stage 2 config violates the strict FP32/no-TF32 contract")
    inputs = FactoryInputs(
        radar_cache_root=radar_cache_root,
        era5_cache_root=era5_cache_root,
        target_static=target_static,
        candidate_manifest=candidate_manifest,
        draw_manifest=draw_manifest,
        bundle_metadata=bundle_metadata,
        expected_target_coordinates_sha256=expected_hashes.target_coordinates,
        expected_context_coordinates_sha256=expected_hashes.context_coordinates,
        verify_cache_hashes=verify_cache_hashes,
    )
    hash_started = time.perf_counter()
    validated = verify_expected_manifest_hashes(inputs, expected_hashes)
    hash_seconds = time.perf_counter() - hash_started
    factory_started = time.perf_counter()
    factory = build_data_factory(config, inputs)
    plan = factory.plan(
        role=role,
        fold_id=fold_id,
        world_size=logical_world_size,
    )
    construction_seconds = time.perf_counter() - factory_started
    if factory.artifacts.artifact_hashes.get("target_coordinates") != (
        expected_hashes.target_coordinates
    ):
        raise ValueError("factory target-coordinate hash changed after preflight")
    if factory.artifacts.artifact_hashes.get("context_coordinates") != (
        expected_hashes.context_coordinates
    ):
        raise ValueError("factory context-coordinate hash changed after preflight")
    return ProductionBenchmarkContext(
        config=config,
        inputs=inputs,
        factory=factory,
        plan=plan,
        expected_hashes=expected_hashes,
        validated_manifest_hashes=validated,
        config_load_seconds=config_seconds,
        artifact_hash_preflight_seconds=hash_seconds,
        factory_construction_seconds=construction_seconds,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--radar-cache-root", type=Path, required=True)
    parser.add_argument("--era5-cache-root", type=Path, required=True)
    parser.add_argument("--target-static", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate-manifest-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--draw-manifest", type=Path, required=True)
    parser.add_argument("--draw-manifest-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--bundle-metadata", type=Path, required=True)
    parser.add_argument("--bundle-metadata-sha256", type=_sha256_argument, required=True)
    parser.add_argument(
        "--target-coordinates-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--context-coordinates-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--batch-sizes", type=_positive_csv, required=True)
    parser.add_argument("--worker-counts", type=_nonnegative_csv, required=True)
    parser.add_argument("--prefetch-factors", type=_positive_csv, required=True)
    parser.add_argument("--warmup-batches", type=int, required=True)
    parser.add_argument("--measured-batches", type=int, required=True)
    parser.add_argument("--maximum-grid-cells", type=int, required=True)
    parser.add_argument("--maximum-host-pipeline-bytes", type=int, required=True)
    parser.add_argument("--maximum-shm-pipeline-bytes", type=int, required=True)
    parser.add_argument("--payload-safety-factor", type=float, default=1.25)
    parser.add_argument("--logical-world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--role",
        choices=("fold", "deployment", "direct_mean", "direct_q50"),
        default="deployment",
    )
    parser.add_argument("--fold-id", type=int)
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("auto", "disabled", "offline", "online"),
        default="auto",
    )
    return parser.parse_args(argv)


def _environment_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"invalid boolean environment value: {value!r}")


def validate_strict_arguments(
    arguments: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> StrictBenchmarkContract:
    """Reject every runtime fallback and unbounded candidate grid."""

    environment = os.environ if environ is None else environ
    if arguments.device != "cuda:0":
        raise ValueError("production pipeline benchmark requires device cuda:0")
    if arguments.precision != "float32":
        raise ValueError("production pipeline benchmark requires float32")
    if not arguments.disable_tf32:
        raise ValueError("--disable-tf32 is required")
    if not arguments.fail_on_fallback:
        raise ValueError("--fail-on-fallback is required")
    if not arguments.pin_memory:
        raise ValueError("--pin-memory is required for nonblocking H2D")
    for name in _FALLBACK_ENVIRONMENT:
        raw = environment.get(name)
        if raw is not None and _environment_flag(raw):
            raise ValueError(f"fallback environment flag must be disabled: {name}")
    expected_environment = {
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    for name, expected in expected_environment.items():
        raw = environment.get(name)
        if raw is not None and raw.strip().lower() != expected:
            raise ValueError(
                f"strict environment contract mismatch for {name}: {raw!r}"
            )
    grid = ProductionBenchmarkGrid(
        batch_sizes=tuple(arguments.batch_sizes),
        worker_counts=tuple(arguments.worker_counts),
        prefetch_factors=tuple(arguments.prefetch_factors),
        warmup_batches=arguments.warmup_batches,
        measured_batches=arguments.measured_batches,
    )
    limits = PipelinePayloadLimits(
        maximum_grid_cells=arguments.maximum_grid_cells,
        maximum_host_pipeline_bytes=arguments.maximum_host_pipeline_bytes,
        maximum_shm_pipeline_bytes=arguments.maximum_shm_pipeline_bytes,
        safety_factor=arguments.payload_safety_factor,
    )
    if grid.cell_count > limits.maximum_grid_cells:
        raise ValueError("candidate grid is larger than --maximum-grid-cells")
    world_size = _strict_positive_integer(
        arguments.logical_world_size, name="logical_world_size"
    )
    rank = _strict_nonnegative_integer(arguments.rank, name="rank")
    if rank >= world_size:
        raise ValueError("rank must be smaller than logical_world_size")
    if (arguments.role == "fold") != (arguments.fold_id is not None):
        raise ValueError("fold role requires --fold-id and other roles forbid it")
    if arguments.fold_id is not None and arguments.fold_id < 0:
        raise ValueError("fold_id cannot be negative")
    return StrictBenchmarkContract(
        device=arguments.device,
        precision=arguments.precision,
        tf32_enabled=False,
        fail_on_fallback=True,
        pin_memory=True,
        logical_world_size=world_size,
        rank=rank,
        role=arguments.role,
        fold_id=arguments.fold_id,
        grid=grid,
        payload_limits=limits,
    )


def validate_config_contract(
    config: Stage2Config,
    contract: StrictBenchmarkContract,
) -> dict[str, object]:
    """Bind the CLI candidate grid to the validated full-width Stage 2 config."""

    config.runtime.validate()
    if (
        config.runtime.target_widths != TARGET_WIDTHS
        or config.runtime.context_widths != CONTEXT_WIDTHS
        or config.runtime.era_latent_channels != ERA_LATENT_CHANNELS
        or config.runtime.era_grid_size != ERA_GRID_SIZE
        or config.runtime.precision != "float32"
        or config.runtime.tf32
        or config.runtime.allow_cpu_fallback
        or config.runtime.allow_model_width_fallback
        or config.runtime.allow_precision_fallback
        or config.runtime.allow_era_grid_fallback
    ):
        raise ValueError("Stage 2 config is not the strict full-width FP32 contract")
    data = config.raw.get("data")
    tuning = config.raw.get("loader_tuning")
    if not isinstance(data, Mapping) or not isinstance(tuning, Mapping):
        raise TypeError("Stage 2 config data/loader_tuning sections are invalid")
    candidates = tuning.get("candidates")
    if not isinstance(candidates, Mapping):
        raise TypeError("loader_tuning.candidates must be a mapping")
    expected = {
        "batch_sizes": contract.grid.batch_sizes,
        "worker_counts": contract.grid.worker_counts,
        "prefetch_factors": contract.grid.prefetch_factors,
    }

    def require_declared_ordered_subset(
        selected: tuple[int, ...], declared: tuple[int, ...], *, name: str
    ) -> None:
        positions = {value: index for index, value in enumerate(declared)}
        if any(value not in positions for value in selected):
            raise ValueError(
                f"CLI grid contains undeclared loader_tuning.candidates.{name}"
            )
        indices = tuple(positions[value] for value in selected)
        if indices != tuple(sorted(indices)):
            raise ValueError(
                f"CLI grid reorders loader_tuning.candidates.{name}"
            )

    for name, wanted in expected.items():
        raw = candidates.get(name)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError(f"loader_tuning.candidates.{name} must be a sequence")
        require_declared_ordered_subset(wanted, tuple(raw), name=name)
    if (
        tuple(data.get("target_shape", ())) != PRODUCTION_RADAR_SHAPE
        or tuple(data.get("context_shape", ())) != PRODUCTION_RADAR_SHAPE
        or tuple(data.get("era_shape", ())) != PRODUCTION_ERA_SHAPE
        or data.get("pin_memory") is not True
        or data.get("persistent_workers") is not True
        or data.get("mmap_lazy_per_worker") is not True
    ):
        raise ValueError("Stage 2 data config is not the production loader contract")
    config.validate_topology(world_size=contract.logical_world_size)
    return {
        "target_widths": list(config.runtime.target_widths),
        "context_widths": list(config.runtime.context_widths),
        "era_latent_channels": config.runtime.era_latent_channels,
        "era_grid_size": config.runtime.era_grid_size,
        "precision": config.runtime.precision,
        "tf32": config.runtime.tf32,
        "target_shape": list(PRODUCTION_RADAR_SHAPE),
        "context_shape": list(PRODUCTION_RADAR_SHAPE),
        "era_shape": list(PRODUCTION_ERA_SHAPE),
        "mmap_lazy_per_worker": True,
        "pin_memory": True,
        "persistent_workers": True,
        "candidate_grid": asdict(contract.grid),
        "logical_world_size": contract.logical_world_size,
    }


class _WandbLogger:
    """Optional W&B bridge which never serializes credentials."""

    def __init__(
        self,
        *,
        mode: str,
        safe_config: Mapping[str, object],
        environ: Mapping[str, str] | None = None,
    ) -> None:
        environment = os.environ if environ is None else environ
        selected = mode
        if selected == "auto":
            selected = "online" if environment.get("WANDB_API_KEY") else "disabled"
        if selected == "online" and not environment.get("WANDB_API_KEY"):
            raise RuntimeError("W&B online mode requires WANDB_API_KEY")
        self.mode = selected
        self._run: Any | None = None
        if selected == "disabled":
            return
        try:
            import wandb  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                f"W&B {selected} logging was requested but wandb is unavailable"
            ) from error
        self._run = wandb.init(
            project=environment.get("WANDB_PROJECT", "kcorrdiff"),
            job_type=environment.get(
                "WANDB_JOB_TYPE", "production-loader-benchmark"
            ),
            name=environment.get("WANDB_NAME"),
            id=environment.get("WANDB_RUN_ID"),
            resume="never",
            mode=selected,
            dir=environment.get("WANDB_DIR"),
            config=dict(safe_config),
        )

    def log(self, result: Mapping[str, object], *, index: int) -> None:
        if self._run is None:
            return
        values: dict[str, object] = {
            "grid_index": index,
            "status_ok": int(result.get("status") == "ok"),
        }
        for name in (
            "batch_size",
            "worker_count",
            "prefetch_factor",
            "samples_per_second",
            "batches_per_second",
            "host_storage_gib_per_second",
            "host_to_device_gib_per_second",
            "total_rss_peak_bytes",
            "gpu_memory_allocated_peak_bytes",
            "gpu_memory_reserved_peak_bytes",
        ):
            if name in result:
                values[name] = result[name]
        for prefix in (
            "data_wait_seconds",
            "worker_collate_seconds",
            "pin_memory_seconds",
            "h2d_seconds",
        ):
            distribution = result.get(prefix)
            if isinstance(distribution, Mapping):
                for name, value in distribution.items():
                    values[f"{prefix}/{name}"] = value
        preflight = result.get("payload_preflight")
        if isinstance(preflight, Mapping):
            payload = preflight.get("payload")
            if isinstance(payload, Mapping):
                for name in (
                    "estimated_shm_bytes",
                    "estimated_host_pipeline_bytes",
                ):
                    if name in payload:
                        values[f"preflight/{name}"] = payload[name]
        self._run.log(values, step=index)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def _best_result(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    successful = [result for result in results if result.get("status") == "ok"]
    if not successful:
        raise RuntimeError("production benchmark has no successful candidate")
    best = max(successful, key=lambda result: float(result["samples_per_second"]))
    return {
        name: best[name]
        for name in (
            "batch_size",
            "worker_count",
            "prefetch_factor",
            "samples_per_second",
            "host_storage_gib_per_second",
            "host_to_device_gib_per_second",
        )
    }


def _artifact_paths(
    context: ProductionBenchmarkContext, config_path: Path
) -> dict[str, object]:
    paths = {
        "config": config_path.resolve(),
        "candidate_manifest": context.inputs.candidate_manifest,
        "draw_manifest": context.inputs.draw_manifest,
        "bundle_metadata": context.inputs.bundle_metadata,
        "target_coordinates": context.config.data.target_coordinates.resolve(),
        "context_coordinates": context.config.data.context_coordinates.resolve(),
        "normalization": context.config.data.normalization.resolve(),
        "target_static": context.inputs.target_static,
        "radar_cache_manifest": context.inputs.radar_cache_root / "manifest.json",
        "era5_cache_manifest": context.inputs.era5_cache_root / "manifest.json",
    }
    known_hashes = {
        "candidate_manifest": context.expected_hashes.candidate_manifest,
        "draw_manifest": context.expected_hashes.draw_manifest,
        "bundle_metadata": context.expected_hashes.bundle_metadata,
        "target_coordinates": context.expected_hashes.target_coordinates,
        "context_coordinates": context.expected_hashes.context_coordinates,
        "normalization": context.factory.artifacts.artifact_hashes["normalization"],
        "target_static": context.factory.artifacts.artifact_hashes["target_static"],
        "radar_cache_manifest": context.factory.artifacts.artifact_hashes[
            "radar_cache_manifest"
        ],
        "era5_cache_manifest": context.factory.artifacts.artifact_hashes[
            "era5_cache_manifest"
        ],
    }
    result: dict[str, object] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing benchmark provenance artifact: {path}")
        digest = known_hashes[name] if name in known_hashes else sha256_file(path)
        result[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    started = datetime.now(UTC)
    contract = validate_strict_arguments(arguments)
    summary_path = arguments.output_json
    jsonl_path = derive_jsonl_path(summary_path, arguments.output_jsonl)
    preflight_output_paths(jsonl_path=jsonl_path, summary_path=summary_path)
    expected_hashes = ExpectedArtifactHashes(
        candidate_manifest=arguments.candidate_manifest_sha256,
        draw_manifest=arguments.draw_manifest_sha256,
        bundle_metadata=arguments.bundle_metadata_sha256,
        target_coordinates=arguments.target_coordinates_sha256,
        context_coordinates=arguments.context_coordinates_sha256,
    )
    context = build_production_benchmark_context(
        config_path=arguments.config,
        radar_cache_root=arguments.radar_cache_root,
        era5_cache_root=arguments.era5_cache_root,
        target_static=arguments.target_static,
        candidate_manifest=arguments.candidate_manifest,
        draw_manifest=arguments.draw_manifest,
        bundle_metadata=arguments.bundle_metadata,
        expected_hashes=expected_hashes,
        role=contract.role,
        fold_id=contract.fold_id,
        logical_world_size=contract.logical_world_size,
        verify_cache_hashes=arguments.verify_cache_hashes,
    )
    validated_config = validate_config_contract(context.config, contract)
    device = configure_cuda_device(contract.device)
    safe_wandb_config: dict[str, object] = {
        "benchmark": "stage2-production-pipeline",
        "precision": "float32",
        "tf32": False,
        "target_widths": list(context.config.runtime.target_widths),
        "context_widths": list(context.config.runtime.context_widths),
        "era_latent_channels": context.config.runtime.era_latent_channels,
        "radar_shape": list(PRODUCTION_RADAR_SHAPE),
        "era_shape": list(PRODUCTION_ERA_SHAPE),
        "batch_sizes": list(contract.grid.batch_sizes),
        "worker_counts": list(contract.grid.worker_counts),
        "prefetch_factors": list(contract.grid.prefetch_factors),
        "logical_world_size": contract.logical_world_size,
        "rank": contract.rank,
        "plan_role": contract.role,
        "plan_sha256": context.plan.semantic_sha256,
        "draw_rows": len(context.factory.artifacts.rows),
        "payload_limits": asdict(contract.payload_limits),
    }
    wandb_logger = _WandbLogger(
        mode=arguments.wandb_mode,
        safe_config=safe_wandb_config,
    )
    try:
        results = run_production_grid(
            context.factory,
            context.plan,
            contract.grid,
            rank=contract.rank,
            limits=contract.payload_limits,
            device=device,
            pin_memory=True,
            result_callback=lambda result, index: wandb_logger.log(
                result, index=index
            ),
        )
    finally:
        wandb_logger.finish()

    finished = datetime.now(UTC)
    summary: dict[str, object] = {
        "format_version": SUMMARY_FORMAT,
        "complete": True,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "factory_construction": {
            "config_load_seconds": context.config_load_seconds,
            "artifact_hash_preflight_seconds": (
                context.artifact_hash_preflight_seconds
            ),
            "factory_and_plan_seconds": context.factory_construction_seconds,
        },
        "contract": {
            "device": contract.device,
            "precision": contract.precision,
            "tf32_enabled": contract.tf32_enabled,
            "fail_on_fallback": contract.fail_on_fallback,
            "pin_memory": contract.pin_memory,
            "non_blocking_h2d": True,
            "logical_world_size": contract.logical_world_size,
            "rank": contract.rank,
            "role": contract.role,
            "fold_id": contract.fold_id,
            "grid": asdict(contract.grid),
            "payload_limits": asdict(contract.payload_limits),
        },
        "validated_config": validated_config,
        "artifacts": _artifact_paths(context, arguments.config),
        "expected_artifact_hashes": asdict(context.expected_hashes),
        "validated_manifest_hashes": dict(context.validated_manifest_hashes),
        "stage2_config_canonical_sha256": context.config.sha256,
        "factory_artifact_hashes": dict(context.factory.artifacts.artifact_hashes),
        "draw_plan": {
            "semantic_sha256": context.plan.semantic_sha256,
            "source_rows": len(context.plan.source_row_indices),
            "rank_slots": len(context.plan.rank_slots[contract.rank]),
            "padding_slots_all_ranks": context.plan.padding_slots,
        },
        "grid_cells": len(results),
        "successful_grid_cells": sum(
            result.get("status") == "ok" for result in results
        ),
        "preflight_rejected_grid_cells": sum(
            result.get("status") == "preflight_rejected" for result in results
        ),
        "best": _best_result(results),
        "runtime": {
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(device),
            "visible_cuda_devices": torch.cuda.device_count(),
            "device": str(device),
            "default_dtype": str(torch.get_default_dtype()),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "wandb_mode": wandb_logger.mode,
            "shm": read_shm_usage().as_json(),
        },
    }
    digest = publish_results_atomic(
        jsonl_path=jsonl_path,
        summary_path=summary_path,
        results=results,
        summary=summary,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(summary_path.resolve()),
                "results_jsonl": str(jsonl_path.resolve()),
                "results_sha256": digest,
                "grid_cells": len(results),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CandidatePreflight",
    "DeviceRankPayload",
    "ExpectedArtifactHashes",
    "LoaderCandidate",
    "PipelinePayloadEstimate",
    "PipelinePayloadLimits",
    "ProductionBenchmarkContext",
    "ProductionBenchmarkGrid",
    "RESULT_FORMAT",
    "SUMMARY_FORMAT",
    "ShmUsage",
    "StrictBenchmarkContract",
    "TensorTreeMetrics",
    "TimedRankLocalBatch",
    "TimedRankSlotCollator",
    "build_candidate_loader",
    "build_production_benchmark_context",
    "estimate_pipeline_payload",
    "main",
    "move_rank_local_batch",
    "preflight_candidate_payload",
    "read_shm_usage",
    "require_float32_tensor_tree",
    "run_production_candidate",
    "run_production_grid",
    "sha256_file",
    "tensor_tree_metrics",
    "validate_config_contract",
    "validate_strict_arguments",
    "verify_expected_manifest_hashes",
]


if __name__ == "__main__":
    raise SystemExit(main())
