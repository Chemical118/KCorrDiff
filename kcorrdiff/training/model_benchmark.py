"""Strict full-width Stage 2 training-step benchmark on production batches.

This module complements :mod:`kcorrdiff.training.production_benchmark`.  The
loader benchmark proves that the real mmap-backed factory, collation, pinning,
and host-to-device path are viable.  This benchmark additionally constructs a
fresh production :class:`~kcorrdiff.models.regression_system.RegressionSystem`
for every declared per-rank batch size and executes the complete steady-state
training step: hurdle loss, backward, gradient clipping, and AdamW.

Every candidate is immutable.  In particular, a CUDA OOM is recorded as a
rejected cell; it never triggers a smaller model, lower precision, CPU move, or
another silent fallback.  Candidate-local model, optimizer, loader workers,
gradients, and CUDA allocator cache are released before the next cell.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import time
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Subset

from kcorrdiff.models.regression_system import (
    RegressionSystem,
    RegressionSystemConfig,
)
from kcorrdiff.training.batch import TrainingBatch
from kcorrdiff.training.benchmark_loader import (
    configure_cuda_device,
    derive_jsonl_path,
    preflight_output_paths,
    publish_results_atomic,
)
from kcorrdiff.training.config import Stage2Config
from kcorrdiff.training.data_factory import KCorrDiffDataFactory
from kcorrdiff.training.losses import (
    accumulation_window_denominators,
    distributed_hurdle_loss_contribution,
)
from kcorrdiff.training.plan import DistributedDrawPlan
from kcorrdiff.training.production_benchmark import (
    CandidateLoaderBuilder,
    CandidatePreflight,
    ExpectedArtifactHashes,
    LoaderCandidate,
    PipelinePayloadLimits,
    ProductionBenchmarkContext,
    ProductionBenchmarkGrid,
    StrictBenchmarkContract,
    TimedRankLocalBatch,
    build_candidate_loader,
    build_production_benchmark_context,
    preflight_candidate_payload,
    require_float32_tensor_tree,
    sha256_file,
    validate_config_contract,
)
from kcorrdiff.training.runtime import assert_float32_tree
from kcorrdiff.training.train_stage2 import (
    ModelContract,
    regression_system_config_from_stage2,
)


SUMMARY_FORMAT = "kcorrdiff.stage2-model-benchmark-summary.v1"
RESULT_FORMAT = "kcorrdiff.stage2-model-benchmark-result.v1"
EXPECTED_REGRESSION_PARAMETER_COUNT = 62_008_276
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FALLBACK_ENVIRONMENT = (
    "KCORRDIFF_ALLOW_CPU_FALLBACK",
    "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK",
    "KCORRDIFF_ALLOW_PRECISION_FALLBACK",
    "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK",
)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_csv(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not pieces or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    try:
        result = tuple(int(piece) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from error
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("all values must be positive")
    return result


def _sha256_argument(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256 digest")
    return value


def _environment_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"invalid boolean environment value: {value!r}")


@dataclass(frozen=True, slots=True)
class ModelBenchmarkGrid:
    """Ordered per-rank microbatch candidates and fixed step counts."""

    batch_sizes: tuple[int, ...]
    warmup_steps: int
    measured_steps: int

    def __post_init__(self) -> None:
        if not self.batch_sizes:
            raise ValueError("model benchmark batch-size grid cannot be empty")
        for value in self.batch_sizes:
            _positive_integer(value, name="batch_sizes item")
            if value & (value - 1):
                raise ValueError(
                    "model benchmark batch_sizes must be powers of two"
                )
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must not contain duplicates")
        if tuple(sorted(self.batch_sizes)) != self.batch_sizes:
            raise ValueError("batch_sizes must be strictly increasing")
        # One warmup is required so Adam's two moment tensors are present in
        # every measured step and the measured peak is steady-state training.
        _positive_integer(self.warmup_steps, name="warmup_steps")
        _positive_integer(self.measured_steps, name="measured_steps")

    @property
    def cell_count(self) -> int:
        return len(self.batch_sizes)

    @property
    def total_steps(self) -> int:
        return self.warmup_steps + self.measured_steps


@dataclass(frozen=True, slots=True)
class StrictModelBenchmarkContract:
    device: str
    precision: str
    tf32_enabled: bool
    fail_on_fallback: bool
    pin_memory: bool
    logical_world_size: int
    rank: int
    role: str
    fold_id: int | None
    worker_count: int
    prefetch_factor: int
    maximum_grid_cells: int
    grid: ModelBenchmarkGrid
    payload_limits: PipelinePayloadLimits


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int
    floating_buffer_bytes: int
    component_parameter_counts: Mapping[str, int]

    def as_json(self) -> dict[str, object]:
        return {
            "class": "RegressionSystem",
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameter_bytes": self.parameter_bytes,
            "floating_buffer_bytes": self.floating_buffer_bytes,
            "component_parameter_counts": dict(self.component_parameter_counts),
            "dtype": "float32",
        }


@dataclass(frozen=True, slots=True)
class ForwardLossResult:
    loss: Tensor
    metrics: Mapping[str, Tensor]
    oof_fields: Tensor | None = None


class ModelBuilder(Protocol):
    def __call__(self, config: RegressionSystemConfig) -> nn.Module: ...


class ForwardLossRunner(Protocol):
    def __call__(
        self,
        model: nn.Module,
        training: object,
        config: Stage2Config,
        global_step: int,
    ) -> ForwardLossResult: ...


class CandidateRunner(Protocol):
    def __call__(
        self, candidate: LoaderCandidate, preflight: CandidatePreflight
    ) -> dict[str, object]: ...


class CandidatePreflightRunner(Protocol):
    def __call__(self, candidate: LoaderCandidate) -> CandidatePreflight: ...


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
    parser.add_argument(
        "--draw-manifest-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--bundle-metadata", type=Path, required=True)
    parser.add_argument(
        "--bundle-metadata-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--target-coordinates-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--context-coordinates-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--oof-codec-probe-npy",
        type=Path,
        help="atomically save the first real FP32 [B,2,H,W] prediction batch",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--batch-sizes", type=_positive_csv, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--measured-steps", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, required=True)
    parser.add_argument("--maximum-grid-cells", type=int, required=True)
    parser.add_argument("--maximum-host-pipeline-bytes", type=int, required=True)
    parser.add_argument("--maximum-shm-pipeline-bytes", type=int, required=True)
    parser.add_argument("--payload-safety-factor", type=float, default=1.25)
    parser.add_argument("--logical-world-size", type=int, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--role",
        choices=("fold", "deployment"),
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


def validate_strict_arguments(
    arguments: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
) -> StrictModelBenchmarkContract:
    """Validate a CUDA-only, FP32, bounded, no-fallback invocation."""

    environment = os.environ if environ is None else environ
    if arguments.device != "cuda:0":
        raise ValueError("Stage 2 model benchmark requires device cuda:0")
    if arguments.precision != "float32":
        raise ValueError("Stage 2 model benchmark requires float32")
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
    required_environment = {
        "KCORRDIFF_REQUIRE_FULL_WIDTH": "1",
        "KCORRDIFF_REQUIRE_PRECISION": "float32",
        "KCORRDIFF_REQUIRE_ERA_GRID_SIZE": "33",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    for name, expected in required_environment.items():
        raw = environment.get(name)
        if raw is not None and raw.strip().lower() != expected:
            raise ValueError(
                f"strict environment contract mismatch for {name}: {raw!r}"
            )
    grid = ModelBenchmarkGrid(
        batch_sizes=tuple(arguments.batch_sizes),
        warmup_steps=arguments.warmup_steps,
        measured_steps=arguments.measured_steps,
    )
    maximum_cells = _positive_integer(
        arguments.maximum_grid_cells, name="maximum_grid_cells"
    )
    if grid.cell_count > maximum_cells:
        raise ValueError("batch-size grid is larger than --maximum-grid-cells")
    worker_count = _nonnegative_integer(
        arguments.worker_count, name="worker_count"
    )
    prefetch_factor = _positive_integer(
        arguments.prefetch_factor, name="prefetch_factor"
    )
    logical_world_size = _positive_integer(
        arguments.logical_world_size, name="logical_world_size"
    )
    rank = _nonnegative_integer(arguments.rank, name="rank")
    if rank >= logical_world_size:
        raise ValueError("rank must be smaller than logical_world_size")
    if (arguments.role == "fold") != (arguments.fold_id is not None):
        raise ValueError("fold role requires --fold-id and deployment forbids it")
    if arguments.fold_id is not None and arguments.fold_id < 0:
        raise ValueError("fold_id cannot be negative")
    limits = PipelinePayloadLimits(
        maximum_grid_cells=maximum_cells,
        maximum_host_pipeline_bytes=arguments.maximum_host_pipeline_bytes,
        maximum_shm_pipeline_bytes=arguments.maximum_shm_pipeline_bytes,
        safety_factor=arguments.payload_safety_factor,
    )
    return StrictModelBenchmarkContract(
        device=arguments.device,
        precision="float32",
        tf32_enabled=False,
        fail_on_fallback=True,
        pin_memory=True,
        logical_world_size=logical_world_size,
        rank=rank,
        role=arguments.role,
        fold_id=arguments.fold_id,
        worker_count=worker_count,
        prefetch_factor=prefetch_factor,
        maximum_grid_cells=maximum_cells,
        grid=grid,
        payload_limits=limits,
    )


def validate_stage2_contract(
    config: Stage2Config,
    contract: StrictModelBenchmarkContract,
) -> tuple[ModelContract, dict[str, object]]:
    """Bind the model grid to the exact Stage 2 data/model/optimizer config."""

    loader_contract = StrictBenchmarkContract(
        device=contract.device,
        precision=contract.precision,
        tf32_enabled=False,
        fail_on_fallback=True,
        pin_memory=True,
        logical_world_size=contract.logical_world_size,
        rank=contract.rank,
        role=contract.role,
        fold_id=contract.fold_id,
        grid=ProductionBenchmarkGrid(
            batch_sizes=contract.grid.batch_sizes,
            worker_counts=(contract.worker_count,),
            prefetch_factors=(contract.prefetch_factor,),
            warmup_batches=contract.grid.warmup_steps,
            measured_batches=contract.grid.measured_steps,
        ),
        payload_limits=contract.payload_limits,
    )
    validated_loader = validate_config_contract(config, loader_contract)
    model_contract = regression_system_config_from_stage2(config)
    if not model_contract.system.activation_checkpoint:
        raise ValueError("production activation checkpointing must remain enabled")
    raw_optimization = config.raw.get("optimization")
    if not isinstance(raw_optimization, Mapping):
        raise TypeError("Stage 2 optimization must be a mapping")
    if raw_optimization.get("optimizer") != "AdamW":
        raise ValueError("Stage 2 model benchmark requires production AdamW")
    return model_contract, {
        "loader": validated_loader,
        "model": asdict(model_contract.system),
        "optimizer": {
            "class": "AdamW",
            "learning_rate": config.optimization.learning_rate,
            "weight_decay": config.optimization.weight_decay,
            "betas": list(config.optimization.betas),
            "gradient_clip_norm": config.optimization.gradient_clip_norm,
        },
        "loss": asdict(config.loss),
    }


def validate_model_instance(
    model: nn.Module,
    *,
    expected_parameter_count: int = EXPECTED_REGRESSION_PARAMETER_COUNT,
    require_exact_class: bool = True,
) -> ModelIdentity:
    """Prove model class, size, trainability, and full FP32 state."""

    _positive_integer(expected_parameter_count, name="expected_parameter_count")
    if require_exact_class and type(model) is not RegressionSystem:
        raise TypeError("benchmark model must be the exact RegressionSystem class")
    assert_float32_tree(model)
    parameters = tuple(model.parameters())
    count = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    if count != expected_parameter_count:
        raise ValueError(
            "RegressionSystem parameter count changed: "
            f"expected={expected_parameter_count}, actual={count}"
        )
    if trainable != count:
        raise ValueError("every production RegressionSystem parameter must be trainable")
    floating_buffers = tuple(
        value for value in model.buffers() if value.is_floating_point()
    )
    components = getattr(model, "component_parameter_counts", {})
    if not isinstance(components, Mapping):
        components = {}
    return ModelIdentity(
        parameter_count=count,
        trainable_parameter_count=trainable,
        parameter_bytes=sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        ),
        floating_buffer_bytes=sum(
            value.numel() * value.element_size() for value in floating_buffers
        ),
        component_parameter_counts={
            str(name): int(value) for name, value in components.items()
        },
    )


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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_runtime_device(
    device: torch.device, *, allow_cpu_test: bool
) -> None:
    if torch.get_default_dtype() is not torch.float32:
        raise RuntimeError("PyTorch default dtype must remain float32")
    if device.type == "cpu":
        if not allow_cpu_test:
            raise RuntimeError("CPU fallback is forbidden")
        return
    if device != torch.device("cuda:0"):
        raise RuntimeError("Stage 2 model benchmark requires cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("model benchmark requires exactly one visible CUDA GPU")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled")


def _shutdown_iterator(iterator: object | None) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _close_loader_dataset(loader: object | None) -> None:
    dataset = getattr(loader, "dataset", None)
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    close = getattr(dataset, "_close", None)
    if callable(close):
        close()


def _cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _clear_runtime_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _is_cuda_oom(error: RuntimeError) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    normalized = str(error).lower()
    return "cuda" in normalized and "out of memory" in normalized


def _validate_candidate_capacity(
    plan: DistributedDrawPlan,
    candidate: LoaderCandidate,
    *,
    rank: int,
    total_steps: int,
) -> None:
    if not 0 <= rank < plan.world_size:
        raise ValueError("benchmark rank is outside the draw plan")
    required = candidate.batch_size * total_steps
    slots = plan.rank_slots[rank]
    if len(slots) < required:
        raise ValueError(
            "rank-local draw plan has too few slots for model benchmark: "
            f"required={required}, available={len(slots)}"
        )
    if any(slot.is_padding for slot in slots[:required]):
        raise ValueError("model benchmark window contains a padding sentinel")


def _production_forward_loss(
    model: nn.Module,
    training: object,
    config: Stage2Config,
    global_step: int,
) -> ForwardLossResult:
    if type(model) is not RegressionSystem:
        raise TypeError("production forward requires exact RegressionSystem")
    if not isinstance(training, TrainingBatch):
        raise TypeError("production model benchmark requires TrainingBatch")
    denominators = accumulation_window_denominators(
        [training.labels], device=training.model.device
    )
    output = model(training.model)
    loss = distributed_hurdle_loss_contribution(
        occurrence_logits=output.occurrence_logits,
        wet_amount=output.wet_amount,
        **training.labels.hurdle_kwargs(),
        global_step=global_step,
        denominators=denominators,
        config=config.loss,
    )
    return ForwardLossResult(
        loss=loss.local_total,
        metrics={
            "loss_total": loss.local_total.detach(),
            "loss_occurrence": loss.occurrence,
            "loss_positive": loss.positive_amount,
            "loss_mean": loss.transformed_mean,
        },
        oof_fields=torch.stack(
            (output.probability_wet[:, 0], output.mu_z[:, 0]), dim=1
        ).detach(),
    )


def _atomic_save_oof_codec_probe(path: Path, fields: Tensor) -> None:
    if fields.dtype is not torch.float32 or fields.ndim != 4 or fields.shape[1] != 2:
        raise ValueError("OOF codec probe must be float32 [B,2,H,W]")
    selected = path.resolve()
    if selected.exists():
        return
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{selected.name}.", suffix=".tmp", dir=selected.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(
                stream,
                fields.cpu().numpy().astype(np.float32, copy=False),
                allow_pickle=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        if not selected.exists():
            os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)


def _default_model_builder(config: RegressionSystemConfig) -> nn.Module:
    return RegressionSystem(config)


def run_model_candidate(
    factory: KCorrDiffDataFactory,
    plan: DistributedDrawPlan,
    candidate: LoaderCandidate,
    *,
    rank: int,
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
    preflight: CandidatePreflight,
    config: Stage2Config,
    system_config: RegressionSystemConfig,
    pin_memory: bool = True,
    allow_cpu_test: bool = False,
    loader_builder: CandidateLoaderBuilder = build_candidate_loader,
    model_builder: ModelBuilder = _default_model_builder,
    forward_loss: ForwardLossRunner = _production_forward_loss,
    expected_parameter_count: int = EXPECTED_REGRESSION_PARAMETER_COUNT,
    require_exact_model_class: bool = True,
    oof_codec_probe_path: Path | None = None,
) -> dict[str, object]:
    """Execute warmup and measured full optimizer steps for one candidate."""

    _positive_integer(warmup_steps, name="warmup_steps")
    _positive_integer(measured_steps, name="measured_steps")
    _validate_runtime_device(device, allow_cpu_test=allow_cpu_test)
    if not preflight.approved:
        raise ValueError("cannot execute a payload-preflight-rejected candidate")
    if preflight.payload.candidate != candidate:
        raise ValueError("candidate and payload preflight identity disagree")
    if device.type == "cuda" and not pin_memory:
        raise ValueError("production CUDA benchmark requires pinned host batches")
    total_steps = warmup_steps + measured_steps
    _validate_candidate_capacity(
        plan, candidate, rank=rank, total_steps=total_steps
    )

    loader: object | None = None
    iterator: object | None = None
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    moved_training: object | None = None
    forward_result: ForwardLossResult | None = None
    candidate_started = time.perf_counter()
    try:
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        model_started = time.perf_counter()
        model = model_builder(system_config)
        model.to(device=device, dtype=torch.float32)
        model.train()
        identity = validate_model_instance(
            model,
            expected_parameter_count=expected_parameter_count,
            require_exact_class=require_exact_model_class,
        )
        _synchronize(device)
        model_construction_seconds = time.perf_counter() - model_started

        optimizer_started = time.perf_counter()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
            betas=config.optimization.betas,
        )
        optimizer_construction_seconds = time.perf_counter() - optimizer_started

        loader_started = time.perf_counter()
        loader = loader_builder(
            factory,
            plan,
            rank=rank,
            candidate=candidate,
            pin_memory=pin_memory,
            maximum_batches=total_steps,
        )
        loader_construction_seconds = time.perf_counter() - loader_started
        iterator_started = time.perf_counter()
        iterator = iter(loader)  # type: ignore[arg-type]
        iterator_construction_seconds = time.perf_counter() - iterator_started

        waits: list[float] = []
        collates: list[float] = []
        pins: list[float] = []
        h2d_times: list[float] = []
        zero_times: list[float] = []
        forward_times: list[float] = []
        backward_times: list[float] = []
        optimizer_times: list[float] = []
        compute_step_times: list[float] = []
        losses: dict[str, list[float]] = {}
        first_host_metrics: dict[str, object] | None = None
        first_device_metrics: dict[str, object] | None = None
        samples = 0
        measurement_started: float | None = None
        steady_global_step = max(config.loss.mean_warmup_steps, 0)

        for step_index in range(total_steps):
            measured = step_index >= warmup_steps
            if measured and measurement_started is None:
                measurement_started = time.perf_counter()
            wait_started = time.perf_counter()
            timed = next(iterator)  # type: ignore[arg-type]
            data_wait_seconds = time.perf_counter() - wait_started
            if not isinstance(timed, TimedRankLocalBatch):
                raise TypeError("candidate loader must return TimedRankLocalBatch")
            rank_batch = timed.rank_local
            if rank_batch.is_zero_loss_sentinel or rank_batch.padding_count:
                raise ValueError("model benchmark cannot consume padding")
            if len(rank_batch.active_slot_indices) != candidate.batch_size:
                raise ValueError("collated batch size changed from candidate")
            host_metrics = require_float32_tensor_tree(rank_batch)
            if device.type == "cuda":
                if not timed.pin_applied:
                    raise RuntimeError("DataLoader did not pin the production batch")
                if not host_metrics.all_cpu_tensors_pinned:
                    raise RuntimeError("not every production H2D tensor is pinned")

            compute_started = time.perf_counter()
            _synchronize(device)
            h2d_started = time.perf_counter()
            training = rank_batch.training
            if training is None:
                raise ValueError("model benchmark received an empty rank batch")
            move = getattr(training, "to", None)
            if not callable(move):
                raise TypeError("production training object must implement to(device)")
            moved_training = move(
                device,
                non_blocking=device.type == "cuda",
            )
            moved_weights = rank_batch.loss_weights.to(
                device=device,
                non_blocking=device.type == "cuda",
            )
            _synchronize(device)
            h2d_seconds = time.perf_counter() - h2d_started
            device_metrics = require_float32_tensor_tree(
                {"training": moved_training, "loss_weights": moved_weights}
            )
            if first_host_metrics is None:
                first_host_metrics = host_metrics.as_json()
                first_device_metrics = device_metrics.as_json()

            zero_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            zero_seconds = time.perf_counter() - zero_started

            forward_started = time.perf_counter()
            forward_result = forward_loss(
                model,
                moved_training,
                config,
                steady_global_step + step_index,
            )
            if (
                oof_codec_probe_path is not None
                and forward_result.oof_fields is not None
                and not oof_codec_probe_path.exists()
            ):
                _atomic_save_oof_codec_probe(
                    oof_codec_probe_path, forward_result.oof_fields
                )
            if forward_result.loss.shape != ():
                raise ValueError("model benchmark loss must be scalar")
            if forward_result.loss.dtype is not torch.float32:
                raise TypeError("model benchmark loss must remain float32")
            if not torch.isfinite(forward_result.loss):
                raise FloatingPointError("non-finite model benchmark loss")
            _synchronize(device)
            forward_seconds = time.perf_counter() - forward_started

            backward_started = time.perf_counter()
            forward_result.loss.backward()
            _synchronize(device)
            backward_seconds = time.perf_counter() - backward_started

            optimizer_step_started = time.perf_counter()
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise RuntimeError(
                        "every production parameter must receive a gradient"
                    )
                if parameter.grad.dtype is not torch.float32:
                    raise TypeError("model gradients must remain float32")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimization.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite model benchmark gradient norm")
            optimizer.step()
            _synchronize(device)
            optimizer_seconds = time.perf_counter() - optimizer_step_started
            compute_seconds = time.perf_counter() - compute_started

            if measured:
                waits.append(data_wait_seconds)
                collates.append(timed.collate_seconds)
                pins.append(timed.pin_seconds)
                h2d_times.append(h2d_seconds)
                zero_times.append(zero_seconds)
                forward_times.append(forward_seconds)
                backward_times.append(backward_seconds)
                optimizer_times.append(optimizer_seconds)
                compute_step_times.append(compute_seconds)
                samples += candidate.batch_size
                for name, value in forward_result.metrics.items():
                    if value.shape != () or not torch.isfinite(value):
                        raise FloatingPointError(
                            f"invalid model benchmark metric: {name}"
                        )
                    losses.setdefault(name, []).append(float(value.item()))

            # Drop the graph and device batch before fetching the next one.
            forward_result = None
            moved_training = None
            del moved_weights

        _synchronize(device)
        if measurement_started is None:
            raise AssertionError("measurement window did not start")
        wall_seconds = time.perf_counter() - measurement_started
        expected_samples = measured_steps * candidate.batch_size
        if samples != expected_samples:
            raise AssertionError("measured sample count changed from grid")
        if first_host_metrics is None or first_device_metrics is None:
            raise AssertionError("model benchmark observed no payload")
        memory = _cuda_memory_snapshot(device)
        return {
            "format_version": RESULT_FORMAT,
            "status": "ok",
            "accepted": True,
            "batch_size": candidate.batch_size,
            "worker_count": candidate.worker_count,
            "prefetch_factor": candidate.prefetch_factor,
            "prefetch_effective": candidate.prefetch_effective,
            "rank": rank,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "optimizer_steps_executed": total_steps,
            "samples": samples,
            "wall_seconds": wall_seconds,
            "samples_per_second": samples / wall_seconds,
            "optimizer_steps_per_second": measured_steps / wall_seconds,
            "data_wait_seconds": _distribution(waits),
            "worker_collate_seconds": _distribution(collates),
            "pin_memory_seconds": _distribution(pins),
            "h2d_seconds": _distribution(h2d_times),
            "zero_grad_seconds": _distribution(zero_times),
            "forward_loss_seconds": _distribution(forward_times),
            "backward_seconds": _distribution(backward_times),
            "adamw_step_seconds": _distribution(optimizer_times),
            "compute_step_seconds": _distribution(compute_step_times),
            "losses": {
                name: _distribution(values) for name, values in losses.items()
            },
            "model_construction_seconds": model_construction_seconds,
            "optimizer_construction_seconds": optimizer_construction_seconds,
            "loader_construction_seconds": loader_construction_seconds,
            "iterator_construction_seconds": iterator_construction_seconds,
            "candidate_elapsed_seconds": time.perf_counter() - candidate_started,
            "model": identity.as_json(),
            "optimizer": "AdamW",
            "initial_loss_global_step": steady_global_step,
            "host_batch_tensor_metrics": first_host_metrics,
            "device_batch_tensor_metrics": first_device_metrics,
            "payload_preflight": preflight.as_json(),
            "gpu_memory_allocated_bytes": memory["allocated_bytes"],
            "gpu_memory_reserved_bytes": memory["reserved_bytes"],
            "gpu_memory_allocated_peak_bytes": memory["peak_allocated_bytes"],
            "gpu_memory_reserved_peak_bytes": memory["peak_reserved_bytes"],
            "precision": "float32",
            "tf32_enabled": False,
            "activation_checkpoint": bool(system_config.activation_checkpoint),
            "fallback_used": False,
            "device": str(device),
            "pin_memory": pin_memory,
            "non_blocking_h2d": device.type == "cuda",
        }
    finally:
        forward_result = None
        moved_training = None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        optimizer = None
        model = None
        _shutdown_iterator(iterator)
        iterator = None
        _close_loader_dataset(loader)
        loader = None
        _clear_runtime_state(device)


def run_model_grid(
    grid: ModelBenchmarkGrid,
    *,
    worker_count: int,
    prefetch_factor: int,
    device: torch.device,
    preflight_runner: CandidatePreflightRunner,
    candidate_runner: CandidateRunner,
    result_callback: Callable[[Mapping[str, object], int], None] | None = None,
    memory_probe: Callable[[torch.device], Mapping[str, int]] = _cuda_memory_snapshot,
    reset_peak: Callable[[torch.device], None] = _reset_cuda_peak,
    cleanup: Callable[[torch.device], None] = _clear_runtime_state,
) -> list[dict[str, object]]:
    """Run all immutable cells, recording OOM and continuing after cleanup."""

    _nonnegative_integer(worker_count, name="worker_count")
    _positive_integer(prefetch_factor, name="prefetch_factor")
    results: list[dict[str, object]] = []
    for candidate_index, batch_size in enumerate(grid.batch_sizes):
        candidate = LoaderCandidate(batch_size, worker_count, prefetch_factor)
        preflight = preflight_runner(candidate)
        if not preflight.approved:
            result: dict[str, object] = {
                "format_version": RESULT_FORMAT,
                "status": "rejected_preflight",
                "accepted": False,
                "rejection_reason": "host_or_shm_payload_preflight",
                **asdict(candidate),
                "payload_preflight": preflight.as_json(),
                "precision": "float32",
                "tf32_enabled": False,
                "fallback_used": False,
            }
        else:
            reset_peak(device)
            try:
                result = candidate_runner(candidate, preflight)
                if result.get("status") != "ok" or result.get("accepted") is not True:
                    raise RuntimeError("candidate runner returned a non-success result")
                if (
                    result.get("batch_size") != candidate.batch_size
                    or result.get("worker_count") != candidate.worker_count
                    or result.get("prefetch_factor") != candidate.prefetch_factor
                ):
                    raise RuntimeError("candidate runner changed the grid cell")
                if (
                    result.get("precision") != "float32"
                    or result.get("tf32_enabled") is not False
                    or result.get("fallback_used") is not False
                ):
                    raise RuntimeError("candidate runner violated strict FP32 contract")
            except RuntimeError as error:
                if not _is_cuda_oom(error):
                    raise
                memory = dict(memory_probe(device))
                result = {
                    "format_version": RESULT_FORMAT,
                    "status": "rejected_oom",
                    "accepted": False,
                    "rejection_reason": "cuda_out_of_memory",
                    **asdict(candidate),
                    "error_type": type(error).__name__,
                    "error": "CUDA out of memory at the unchanged production contract",
                    "payload_preflight": preflight.as_json(),
                    "gpu_memory_allocated_bytes": int(
                        memory.get("allocated_bytes", 0)
                    ),
                    "gpu_memory_reserved_bytes": int(memory.get("reserved_bytes", 0)),
                    "gpu_memory_allocated_peak_bytes": int(
                        memory.get("peak_allocated_bytes", 0)
                    ),
                    "gpu_memory_reserved_peak_bytes": int(
                        memory.get("peak_reserved_bytes", 0)
                    ),
                    "precision": "float32",
                    "tf32_enabled": False,
                    "fallback_used": False,
                }
            finally:
                cleanup(device)
            post_cleanup = memory_probe(device)
            result["post_cleanup_gpu_memory_allocated_bytes"] = int(
                post_cleanup.get("allocated_bytes", 0)
            )
            result["post_cleanup_gpu_memory_reserved_bytes"] = int(
                post_cleanup.get("reserved_bytes", 0)
            )
        result["candidate_index"] = candidate_index
        results.append(result)
        if result_callback is not None:
            result_callback(result, candidate_index)
    return results


def summarize_selection(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize observations without changing the production batch contract."""

    accepted = [result for result in results if result.get("status") == "ok"]
    oom = [result for result in results if result.get("status") == "rejected_oom"]
    preflight = [
        result for result in results if result.get("status") == "rejected_preflight"
    ]
    if accepted:
        maximum = max(accepted, key=lambda item: int(item["batch_size"]))
        fastest = max(
            accepted, key=lambda item: float(item["samples_per_second"])
        )
        maximum_accepted: dict[str, object] | None = {
            name: maximum[name]
            for name in (
                "batch_size",
                "samples_per_second",
                "gpu_memory_allocated_peak_bytes",
                "gpu_memory_reserved_peak_bytes",
            )
        }
        throughput_best_accepted: dict[str, object] | None = {
            name: fastest[name]
            for name in (
                "batch_size",
                "samples_per_second",
                "optimizer_steps_per_second",
                "gpu_memory_allocated_peak_bytes",
                "gpu_memory_reserved_peak_bytes",
            )
        }
    else:
        maximum_accepted = None
        throughput_best_accepted = None
    return {
        "scope": "benchmark_observation_only",
        "changes_production_training_config": False,
        "accepted_cells": len(accepted),
        "oom_rejected_cells": len(oom),
        "preflight_rejected_cells": len(preflight),
        "maximum_accepted": maximum_accepted,
        "throughput_best_accepted": throughput_best_accepted,
        "minimum_oom_rejected_batch_size": (
            min(int(item["batch_size"]) for item in oom) if oom else None
        ),
    }


def _model_contract_sha256(config: RegressionSystemConfig) -> str:
    payload = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_provenance(
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
    known = {
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
            raise FileNotFoundError(f"missing model benchmark artifact: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": known.get(name, sha256_file(path)),
        }
    return result


class _WandbLogger:
    """Optional live per-cell W&B reporting without credential serialization."""

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
                "WANDB_JOB_TYPE", "stage2-fullwidth-model-benchmark"
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
        payload: dict[str, object] = {
            "grid/index": index,
            "grid/batch_size": int(result["batch_size"]),
            "grid/accepted": int(result.get("accepted") is True),
            "grid/oom_rejected": int(result.get("status") == "rejected_oom"),
            "grid/preflight_rejected": int(
                result.get("status") == "rejected_preflight"
            ),
        }
        for name in (
            "samples_per_second",
            "optimizer_steps_per_second",
            "gpu_memory_allocated_peak_bytes",
            "gpu_memory_reserved_peak_bytes",
            "post_cleanup_gpu_memory_allocated_bytes",
            "post_cleanup_gpu_memory_reserved_bytes",
        ):
            if name in result:
                payload[f"model/{name}"] = result[name]
        for timing in (
            "data_wait_seconds",
            "h2d_seconds",
            "forward_loss_seconds",
            "backward_seconds",
            "adamw_step_seconds",
            "compute_step_seconds",
        ):
            distribution = result.get(timing)
            if isinstance(distribution, Mapping):
                for statistic, value in distribution.items():
                    payload[f"timing/{timing}/{statistic}"] = value
        losses = result.get("losses")
        if isinstance(losses, Mapping):
            for name, distribution in losses.items():
                if isinstance(distribution, Mapping) and "mean" in distribution:
                    payload[f"loss/{name}"] = distribution["mean"]
        self._run.log(payload, step=index)

    def finish(
        self,
        *,
        selection: Mapping[str, object] | None = None,
        results_sha256: str | None = None,
        exit_code: int = 0,
    ) -> None:
        if self._run is None:
            return
        if selection is not None:
            self._run.summary.update(dict(selection))
        if results_sha256 is not None:
            self._run.summary["results_jsonl_sha256"] = results_sha256
        self._run.finish(exit_code=exit_code)
        self._run = None


def _runtime_json(device: torch.device, *, wandb_mode: str) -> dict[str, object]:
    return {
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
        "wandb_mode": wandb_mode,
    }


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
    model_contract, validated_config = validate_stage2_contract(
        context.config, contract
    )
    # Prove the architecture before allocating CUDA.  Every actual cell checks
    # the same identity again after its fresh model has moved to the GPU.
    probe_model = RegressionSystem(model_contract.system)
    model_identity = validate_model_instance(probe_model)
    del probe_model
    gc.collect()
    device = configure_cuda_device(contract.device)

    safe_wandb_config: dict[str, object] = {
        "benchmark": "stage2-fullwidth-training-step",
        "stage2_config_canonical_sha256": context.config.sha256,
        "model_contract_sha256": _model_contract_sha256(model_contract.system),
        "model_parameter_count": model_identity.parameter_count,
        "precision": "float32",
        "tf32": False,
        "activation_checkpoint": model_contract.system.activation_checkpoint,
        "batch_sizes": list(contract.grid.batch_sizes),
        "warmup_steps": contract.grid.warmup_steps,
        "measured_steps": contract.grid.measured_steps,
        "worker_count": contract.worker_count,
        "prefetch_factor": contract.prefetch_factor,
        "logical_world_size": contract.logical_world_size,
        "rank": contract.rank,
        "role": contract.role,
        "fold_id": contract.fold_id,
        "plan_sha256": context.plan.semantic_sha256,
    }
    wandb_logger = _WandbLogger(
        mode=arguments.wandb_mode, safe_config=safe_wandb_config
    )
    try:
        def preflight_runner(candidate: LoaderCandidate) -> CandidatePreflight:
            _validate_candidate_capacity(
                context.plan,
                candidate,
                rank=contract.rank,
                total_steps=contract.grid.total_steps,
            )
            return preflight_candidate_payload(
                context.factory,
                context.plan,
                candidate,
                rank=contract.rank,
                limits=contract.payload_limits,
                pin_memory=True,
            )

        def candidate_runner(
            candidate: LoaderCandidate, preflight: CandidatePreflight
        ) -> dict[str, object]:
            return run_model_candidate(
                context.factory,
                context.plan,
                candidate,
                rank=contract.rank,
                warmup_steps=contract.grid.warmup_steps,
                measured_steps=contract.grid.measured_steps,
                device=device,
                preflight=preflight,
                config=context.config,
                system_config=model_contract.system,
                pin_memory=True,
                oof_codec_probe_path=arguments.oof_codec_probe_npy,
            )

        results = run_model_grid(
            contract.grid,
            worker_count=contract.worker_count,
            prefetch_factor=contract.prefetch_factor,
            device=device,
            preflight_runner=preflight_runner,
            candidate_runner=candidate_runner,
            result_callback=lambda result, index: wandb_logger.log(
                result, index=index
            ),
        )
        observations = summarize_selection(results)
        finished = datetime.now(UTC)
        summary: dict[str, object] = {
            "format_version": SUMMARY_FORMAT,
            "complete": True,
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "contract": {
                "device": contract.device,
                "precision": contract.precision,
                "tf32_enabled": False,
                "fail_on_fallback": True,
                "fallback_used": False,
                "pin_memory": True,
                "non_blocking_h2d": True,
                "logical_world_size": contract.logical_world_size,
                "rank": contract.rank,
                "role": contract.role,
                "fold_id": contract.fold_id,
                "worker_count": contract.worker_count,
                "prefetch_factor": contract.prefetch_factor,
                "grid": asdict(contract.grid),
                "payload_limits": asdict(contract.payload_limits),
            },
            "benchmark_observations": observations,
            "production_training_config": {
                "per_rank_microbatch_size": (
                    context.config.optimization.per_rank_microbatch_size
                ),
                "gradient_accumulation_steps": (
                    context.config.optimization.gradient_accumulation_steps
                ),
                "global_effective_batch_size": (
                    context.config.optimization.global_effective_batch_size
                ),
                "unchanged_by_benchmark": True,
            },
            "model": model_identity.as_json(),
            "model_contract_sha256": _model_contract_sha256(
                model_contract.system
            ),
            "validated_config": validated_config,
            "stage2_config_canonical_sha256": context.config.sha256,
            "artifacts": _artifact_provenance(context, arguments.config),
            "expected_artifact_hashes": asdict(context.expected_hashes),
            "validated_manifest_hashes": dict(context.validated_manifest_hashes),
            "factory_artifact_hashes": dict(
                context.factory.artifacts.artifact_hashes
            ),
            "factory_construction": {
                "config_load_seconds": context.config_load_seconds,
                "artifact_hash_preflight_seconds": (
                    context.artifact_hash_preflight_seconds
                ),
                "factory_and_plan_seconds": context.factory_construction_seconds,
            },
            "draw_plan": {
                "semantic_sha256": context.plan.semantic_sha256,
                "source_rows": len(context.plan.source_row_indices),
                "rank_slots": len(context.plan.rank_slots[contract.rank]),
                "padding_slots_all_ranks": context.plan.padding_slots,
            },
            "grid_cells": len(results),
            "runtime": _runtime_json(device, wandb_mode=wandb_logger.mode),
        }
        digest = publish_results_atomic(
            jsonl_path=jsonl_path,
            summary_path=summary_path,
            results=results,
            summary=summary,
        )
    except BaseException:
        wandb_logger.finish(exit_code=1)
        raise
    wandb_logger.finish(selection=observations, results_sha256=digest)
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(summary_path.resolve()),
                "results_jsonl": str(jsonl_path.resolve()),
                "results_sha256": digest,
                "benchmark_observations": observations,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "EXPECTED_REGRESSION_PARAMETER_COUNT",
    "ForwardLossResult",
    "ModelBenchmarkGrid",
    "ModelIdentity",
    "RESULT_FORMAT",
    "SUMMARY_FORMAT",
    "StrictModelBenchmarkContract",
    "main",
    "run_model_candidate",
    "run_model_grid",
    "summarize_selection",
    "validate_model_instance",
    "validate_stage2_contract",
    "validate_strict_arguments",
]


if __name__ == "__main__":
    raise SystemExit(main())
