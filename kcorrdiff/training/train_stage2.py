"""Strict two-rank Stage 2 regression, cross-fit OOF and deployment training.

The production entry point is launched with ``torchrun``.  It intentionally
uses explicit SUM gradient reductions instead of wrapping the model in
``DistributedDataParallel``: rank-local weighted numerators are divided by
one all-reduced accumulation-window denominator, padding ranks backpropagate
a graph-connected zero through every parameter, and gradients are summed in
bounded buckets before the identical optimizer step on every rank.  This is
the exact global weighted objective and avoids unequal-forward DDP hangs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Literal, Protocol

import numpy as np
import torch
import torch.distributed as distributed
from torch import Tensor, nn
from torch.utils.data import DataLoader

from kcorrdiff.data.accumulation import TARGET_SCAN_COUNT
from kcorrdiff.data.advection import FlowConfig
from kcorrdiff.data.normalization import sha256_file
from kcorrdiff.data.sampling import DrawRow, oof_dense_byte_budget
from kcorrdiff.models.common import (
    CONTEXT_WIDTHS,
    ERA_GRID_SIZE,
    ERA_LATENT_CHANNELS,
    TARGET_WIDTHS,
)
from kcorrdiff.models.regression_system import (
    DirectPhysicalRegressionSystem,
    RegressionSystem,
    RegressionSystemConfig,
)
from kcorrdiff.training.batch import TrainingBatch, TrainingBatchCollator
from kcorrdiff.training.checkpoints import (
    CheckpointProvenance,
    TrainingCursor,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from kcorrdiff.training.config import Stage2Config, load_stage2_config
from kcorrdiff.training.crossfit import (
    CheckpointRecord,
    checkpoint_record,
    checkpoint_set_hash,
)
from kcorrdiff.training.data_factory import (
    FactoryInputs,
    KCorrDiffDataFactory,
    LazyRankSlotDataset,
    RankLocalBatch,
    RankSlotCollator,
    build_data_factory,
    dataloader_options,
)
from kcorrdiff.training.losses import (
    GlobalLossDenominators,
    accumulation_window_denominators,
    distributed_direct_loss_contribution,
    distributed_hurdle_loss_contribution,
)
from kcorrdiff.training.launch_identity import (
    Stage2LaunchIdentity,
    load_stage2_launch_identity,
)
from kcorrdiff.training.oof import (
    OOFArtifact,
    OOFCompressedBuild,
    OOFIdentity,
    OOFPartitionSpec,
    compressed_oof_storage_plan,
)
from kcorrdiff.training.oof_remote import (
    AuthenticatedHTTPSShardStore,
    RemoteShardStore,
)
from kcorrdiff.training.plan import DrawSlot, unique_oof_rows
from kcorrdiff.training.residual_scales import (
    ResidualScaleAccumulator,
    write_residual_scales,
)
from kcorrdiff.training.runtime import assert_float32_tree, configure_strict_float32
from kcorrdiff.training.selection_contract import load_stage2_selection_contract
from kcorrdiff.training.tracking import TrackingRun, initialize_tracking


TARGET_BUILDER_VERSION = (
    f"v1.1.3b.trapezoid-{TARGET_SCAN_COUNT}-scan.cprecnet-normalized-to-linear"
)
STAGE2_MANIFEST_FORMAT = "kcorrdiff.stage2-training.v1"
PARTIAL_MANIFEST_FORMAT = "kcorrdiff.stage2-partial-training.v1"
RESIDUAL_PARTIAL_FORMAT = "kcorrdiff.oof-residual-partial.v1"
_ALL_PHASES = (
    "crossfit",
    "oof",
    "residual_scales",
    "deployment",
    "direct_mean",
    "direct_q50",
)
_FALLBACK_ENVIRONMENT = (
    "KCORRDIFF_ALLOW_CPU_FALLBACK",
    "KCORRDIFF_ALLOW_MODEL_WIDTH_FALLBACK",
    "KCORRDIFF_ALLOW_PRECISION_FALLBACK",
    "KCORRDIFF_ALLOW_ERA_GRID_FALLBACK",
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _csv(value: str) -> tuple[str, ...]:
    result = tuple(piece.strip() for piece in value.split(","))
    if not result or any(not piece for piece in result):
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(piece) for piece in _csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    return result


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object], *, replace: bool) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _hash_bytes(payload)


def _atomic_text(path: Path, value: str, *, replace: bool) -> None:
    encoded = value.encode("utf-8")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class DistributedRuntime:
    """Small collective boundary used by production NCCL and CPU tests."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized: bool

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.initialized:
            distributed.barrier()

    def reduce_sum(self, value: Tensor) -> Tensor:
        result = value.detach().clone()
        if self.initialized:
            distributed.all_reduce(result, op=distributed.ReduceOp.SUM)
        return result

    def reduce_max(self, value: Tensor) -> Tensor:
        result = value.detach().clone()
        if self.initialized:
            distributed.all_reduce(result, op=distributed.ReduceOp.MAX)
        return result

    def all_gather_objects(self, value: object) -> tuple[object, ...]:
        if not self.initialized:
            return (value,)
        output: list[object] = [None] * self.world_size
        distributed.all_gather_object(output, value)
        return tuple(output)

    def broadcast_model(self, model: nn.Module) -> None:
        if not self.initialized:
            return
        for value in (*model.parameters(), *model.buffers()):
            distributed.broadcast(value, src=0)

    def close(self) -> None:
        if self.initialized and distributed.is_initialized():
            distributed.destroy_process_group()


def _distributed_tracking_log(
    tracking: TrackingRun,
    runtime: DistributedRuntime,
    data: Mapping[str, object],
    *,
    step: int,
) -> None:
    """Propagate a rank-zero external tracking failure before later collectives."""

    failure: str | None = None
    try:
        tracking.log(data, step=step)
    except BaseException as error:
        # Do not broadcast exception text: third-party errors can contain
        # credentials.  The durable audit already records the attempted metric.
        failure = f"rank={runtime.rank},type={type(error).__name__}"
    failures = [
        value
        for value in runtime.all_gather_objects(failure)
        if value is not None
    ]
    if failures:
        raise RuntimeError(f"distributed experiment tracking failed: {failures}")


def _verify_launch_identity_distributed(
    launch_identity: Stage2LaunchIdentity,
    runtime: DistributedRuntime,
) -> None:
    """Make a local identity mutation abort every rank before later collectives."""

    failure: str | None = None
    try:
        launch_identity.verify_current_files()
    except BaseException as error:
        failure = f"rank={runtime.rank},type={type(error).__name__}"
    failures = [
        value for value in runtime.all_gather_objects(failure) if value is not None
    ]
    if failures:
        raise RuntimeError(f"immutable Stage 2 launch identity verification failed: {failures}")


def initialize_distributed_runtime(*, require_world_size: int) -> DistributedRuntime:
    """Initialize the exact torchrun/NCCL production topology."""

    if require_world_size != 2:
        raise ValueError("v1.1.3b Stage 2 production requires exactly two ranks")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"torchrun environment is incomplete: {missing}")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != require_world_size or not 0 <= rank < world_size:
        raise RuntimeError("torchrun rank/world size disagrees with required topology")
    if not 0 <= local_rank < require_world_size:
        raise RuntimeError("LOCAL_RANK is outside the visible two-GPU topology")
    device = configure_strict_float32(
        require_cuda=True, required_visible_gpus=require_world_size
    )
    if device != torch.device("cuda", local_rank):
        raise RuntimeError("strict runtime selected an unexpected CUDA device")
    torch.cuda.set_device(device)
    if distributed.is_initialized():
        if (
            distributed.get_backend() != "nccl"
            or distributed.get_rank() != rank
            or distributed.get_world_size() != world_size
        ):
            raise RuntimeError("preinitialized process group violates NCCL topology")
    else:
        distributed.init_process_group(
            backend="nccl", rank=rank, world_size=world_size
        )
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda"):
        raise RuntimeError("autocast must remain disabled for Stage 2")
    return DistributedRuntime(rank, local_rank, world_size, device, True)


@dataclass(frozen=True, slots=True)
class ModelContract:
    system: RegressionSystemConfig
    direct_mean_required: bool
    direct_q50_required: bool


def regression_system_config_from_stage2(config: Stage2Config) -> ModelContract:
    """Fail-closed parse of the raw model mapping omitted by Stage2Config."""

    raw = _mapping(config.raw.get("model"), name="stage2 model")
    required = {
        "condition_dimension",
        "activation_checkpoint",
        "physical_attention_query_chunk_size",
        "advection",
        "regression",
    }
    if set(raw) != required:
        raise ValueError(
            f"stage2 model schema mismatch: missing={sorted(required-set(raw))}, "
            f"extra={sorted(set(raw)-required)}"
        )
    if raw["condition_dimension"] != 512:
        raise ValueError("condition dimension fallback is forbidden")
    activation = raw["activation_checkpoint"]
    if activation is not True:
        raise ValueError("production activation checkpointing must remain enabled")
    query_chunk = raw["physical_attention_query_chunk_size"]
    if isinstance(query_chunk, bool) or query_chunk != 128:
        raise ValueError("physical attention query chunk must remain 128")
    advection = _mapping(raw["advection"], name="model.advection")
    advection_required = {
        "trajectory_mode",
        "flow_pyramid_levels",
        "flow_warps_per_level",
        "flow_irls_iterations",
        "flow_window_size",
        "maximum_speed_km_per_hour",
    }
    if set(advection) != advection_required:
        raise ValueError("model.advection schema mismatch")
    if advection["trajectory_mode"] != "streaming":
        raise ValueError("full advection trajectory caching is forbidden")
    integer_flow: dict[str, int] = {}
    for name in (
        "flow_pyramid_levels",
        "flow_warps_per_level",
        "flow_irls_iterations",
        "flow_window_size",
    ):
        value = advection[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"model.advection.{name} must be an integer")
        integer_flow[name] = value
    maximum_speed = advection["maximum_speed_km_per_hour"]
    if (
        isinstance(maximum_speed, bool)
        or not isinstance(maximum_speed, (int, float))
        or not math.isfinite(float(maximum_speed))
    ):
        raise TypeError(
            "model.advection.maximum_speed_km_per_hour must be finite numeric"
        )
    flow = FlowConfig(
        pyramid_levels=integer_flow["flow_pyramid_levels"],
        warps_per_level=integer_flow["flow_warps_per_level"],
        irls_iterations=integer_flow["flow_irls_iterations"],
        window_size=integer_flow["flow_window_size"],
        maximum_speed_km_per_hour=float(maximum_speed),
    )
    regression = _mapping(raw["regression"], name="model.regression")
    regression_required = {
        "occurrence_head",
        "wet_amount_support",
        "transformed_mean",
        "direct_physical_mean_checkpoint",
        "direct_physical_q50_checkpoint",
    }
    if set(regression) != regression_required:
        raise ValueError("model.regression schema mismatch")
    if (
        regression["occurrence_head"] is not True
        or regression["wet_amount_support"] != "z_wet_plus_softplus"
        or regression["transformed_mean"] != "probability_times_wet_amount"
    ):
        raise ValueError("hurdle regression output contract changed")
    for name in (
        "direct_physical_mean_checkpoint",
        "direct_physical_q50_checkpoint",
    ):
        if not isinstance(regression[name], bool):
            raise TypeError(f"{name} must be boolean")
    return ModelContract(
        system=RegressionSystemConfig(
            target_widths=config.runtime.target_widths,
            context_widths=config.runtime.context_widths,
            input_size=256,
            activation_checkpoint=True,
            regression_query_chunk_size=128,
            flow_config=flow,
        ),
        direct_mean_required=bool(regression["direct_physical_mean_checkpoint"]),
        direct_q50_required=bool(regression["direct_physical_q50_checkpoint"]),
    )


@dataclass(frozen=True, slots=True)
class RunSelection:
    phases: tuple[str, ...]
    roles: tuple[str, ...]
    max_optimizer_steps: int | None

    def __post_init__(self) -> None:
        unknown_phases = sorted(set(self.phases) - set(_ALL_PHASES))
        if unknown_phases:
            raise ValueError(f"unknown Stage 2 phases: {unknown_phases}")
        if len(self.phases) != len(set(self.phases)):
            raise ValueError("Stage 2 phases cannot repeat")
        allowed_roles = {
            "fold:0",
            "fold:1",
            "fold:2",
            "fold:3",
            "fold:4",
            "deployment",
            "direct_mean",
            "direct_q50",
        }
        unknown_roles = sorted(set(self.roles) - allowed_roles)
        if unknown_roles:
            raise ValueError(f"unknown Stage 2 roles: {unknown_roles}")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError("Stage 2 roles cannot repeat")
        for role in self.roles:
            required_phase = (
                "crossfit" if role.startswith("fold:") else role
            )
            if required_phase not in self.phases:
                raise ValueError(
                    f"selected role {role!r} requires phase {required_phase!r}"
                )
        if self.max_optimizer_steps is not None and (
            isinstance(self.max_optimizer_steps, bool)
            or self.max_optimizer_steps <= 0
        ):
            raise ValueError("max_optimizer_steps must be a positive integer")

    @property
    def partial(self) -> bool:
        return (
            self.max_optimizer_steps is not None
            or self.phases != _ALL_PHASES
            or bool(self.roles)
        )

    def includes_role(self, role: str, fold_id: int | None = None) -> bool:
        if not self.roles:
            return True
        identity = f"fold:{fold_id}" if role == "fold" else role
        return identity in self.roles


def _selected_training_role_count(
    selection: RunSelection, *, folds: int
) -> int:
    if folds <= 0:
        raise ValueError("fold count must be positive")
    count = 0
    if "crossfit" in selection.phases:
        count += sum(
            selection.includes_role("fold", fold) for fold in range(folds)
        )
    for phase, role in (
        ("deployment", "deployment"),
        ("direct_mean", "direct_mean"),
        ("direct_q50", "direct_q50"),
    ):
        if phase in selection.phases and selection.includes_role(role):
            count += 1
    return count


@dataclass(frozen=True, slots=True)
class StoragePreflight:
    checkpoint_bytes: int
    selected_checkpoint_count: int
    include_oof: bool
    oof_dense_bytes: int
    oof_compressed_cap_bytes: int
    oof_staging_bytes: int
    oof_metadata_bytes: int
    oof_peak_bytes: int
    reserve_bytes: int
    required_by_device: Mapping[int, int]
    free_by_device: Mapping[int, int]


def preflight_storage(
    *,
    output_dir: Path,
    oof_output_dir: Path,
    model_parameter_count: int,
    selected_checkpoint_count: int,
    oof_dense_bytes: int,
    include_oof: bool,
    oof_compressed_cap_bytes: int | None = None,
    oof_staging_bytes: int = 0,
    oof_metadata_bytes: int = 0,
    reserve_bytes: int = 2 * 1024**3,
) -> StoragePreflight:
    """Bound checkpoints plus one capped compressed OOF and active shards."""

    if model_parameter_count <= 0 or reserve_bytes <= 0:
        raise ValueError("storage preflight model/reserve counts must be positive")
    if (
        isinstance(selected_checkpoint_count, bool)
        or selected_checkpoint_count < 0
    ):
        raise ValueError("selected checkpoint count cannot be negative")
    if not isinstance(include_oof, bool):
        raise TypeError("include_oof must be boolean")
    if oof_dense_bytes < 0:
        raise ValueError("OOF byte count cannot be negative")
    compressed_cap = (
        oof_dense_bytes
        if oof_compressed_cap_bytes is None
        else oof_compressed_cap_bytes
    )
    if compressed_cap < 0 or oof_staging_bytes < 0 or oof_metadata_bytes < 0:
        raise ValueError("OOF compressed/staging/metadata byte counts cannot be negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_output_dir.mkdir(parents=True, exist_ok=True)
    # A serialized role contains FP32 model (4 B) plus Adam parameter/first/
    # second-moment state (about 12 B total); use 16 B/parameter for container
    # overhead.  Atomic replacement temporarily needs one *additional* complete
    # checkpoint.  Only role-local ``latest.pt`` plus immutable finals exist.
    one_checkpoint = model_parameter_count * 16
    # Selection/step bounds never imply a smaller retained set: a bounded run
    # may cross several complete roles and stop in the next.  Count every role
    # the selected DAG can write plus one atomic replacement temporary.
    atomic_temporary = 1 if selected_checkpoint_count else 0
    checkpoint_bytes = one_checkpoint * (
        selected_checkpoint_count + atomic_temporary
    )
    # Shards are compressed and published in place.  There is one capped final
    # payload plus bounded per-rank staging/archive temporaries, never partial
    # dense D plus final dense D.
    oof_peak = (
        compressed_cap + oof_staging_bytes + oof_metadata_bytes
        if include_oof
        else 0
    )
    paths = (output_dir.resolve(), oof_output_dir.resolve())
    devices = (os.stat(paths[0]).st_dev, os.stat(paths[1]).st_dev)
    required: dict[int, int] = {}
    required[devices[0]] = required.get(devices[0], 0) + checkpoint_bytes
    required[devices[1]] = required.get(devices[1], 0) + oof_peak
    for device in set(devices):
        required[device] += reserve_bytes
    free = {
        device: shutil.disk_usage(paths[devices.index(device)]).free
        for device in set(devices)
    }
    insufficient = {
        device: (required[device], free[device])
        for device in required
        if required[device] > free[device]
    }
    if insufficient:
        raise OSError(f"Stage 2 output byte budget exceeds free space: {insufficient}")
    return StoragePreflight(
        checkpoint_bytes=checkpoint_bytes,
        selected_checkpoint_count=selected_checkpoint_count,
        include_oof=include_oof,
        oof_dense_bytes=oof_dense_bytes,
        oof_compressed_cap_bytes=compressed_cap,
        oof_staging_bytes=oof_staging_bytes,
        oof_metadata_bytes=oof_metadata_bytes,
        oof_peak_bytes=oof_peak,
        reserve_bytes=reserve_bytes,
        required_by_device=required,
        free_by_device=free,
    )


def _sync_cuda(runtime: DistributedRuntime) -> None:
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)


def _connected_zero(model: nn.Module) -> Tensor:
    result: Tensor | None = None
    for parameter in model.parameters():
        term = parameter.sum() * 0.0
        result = term if result is None else result + term
    if result is None:
        raise ValueError("training model has no parameters")
    return result


def reduce_gradients_sum(
    model: nn.Module,
    runtime: DistributedRuntime,
    *,
    bucket_bytes: int = 25 * 1024**2,
) -> None:
    """SUM every float32 gradient in bounded deterministic buckets."""

    if bucket_bytes <= 0:
        raise ValueError("gradient bucket size must be positive")
    bucket: list[nn.Parameter] = []
    bytes_used = 0

    def flush() -> None:
        nonlocal bucket, bytes_used
        if not bucket:
            return
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in bucket])  # type: ignore[union-attr]
        reduced = runtime.reduce_sum(flat)
        cursor = 0
        for parameter in bucket:
            assert parameter.grad is not None
            elements = parameter.numel()
            parameter.grad.copy_(reduced[cursor : cursor + elements].view_as(parameter))
            cursor += elements
        bucket = []
        bytes_used = 0

    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("every parameter must receive a real or connected-zero gradient")
        if parameter.grad.dtype is not torch.float32 or parameter.grad.device != runtime.device:
            raise TypeError("model gradients must remain float32 on the selected device")
        size = parameter.numel() * parameter.grad.element_size()
        if bucket and bytes_used + size > bucket_bytes:
            flush()
        bucket.append(parameter)
        bytes_used += size
        if bytes_used >= bucket_bytes:
            flush()
    flush()


class ForwardLoss(Protocol):
    def __call__(
        self,
        model: nn.Module,
        batch: TrainingBatch,
        denominators: GlobalLossDenominators,
        global_step: int,
    ) -> tuple[Tensor, Tensor]: ...


def _production_forward_loss(
    *, role: str, config: Stage2Config
) -> ForwardLoss:
    if role in {"fold", "deployment"}:
        def hurdle(
            model: nn.Module,
            batch: TrainingBatch,
            denominators: GlobalLossDenominators,
            global_step: int,
        ) -> tuple[Tensor, Tensor]:
            if not isinstance(model, RegressionSystem):
                raise TypeError("hurdle role requires RegressionSystem")
            output = model(batch.model)
            loss = distributed_hurdle_loss_contribution(
                occurrence_logits=output.occurrence_logits,
                wet_amount=output.wet_amount,
                **batch.labels.hurdle_kwargs(),
                global_step=global_step,
                denominators=denominators,
                config=config.loss,
            )
            metrics = torch.stack(
                (loss.occurrence, loss.positive_amount, loss.transformed_mean)
            )
            return loss.local_total, metrics

        return hurdle
    if role not in {"direct_mean", "direct_q50"}:
        raise ValueError(f"unsupported training role: {role}")
    statistic = "mean" if role == "direct_mean" else "q50"

    def direct(
        model: nn.Module,
        batch: TrainingBatch,
        denominators: GlobalLossDenominators,
        global_step: int,
    ) -> tuple[Tensor, Tensor]:
        del global_step
        if not isinstance(model, DirectPhysicalRegressionSystem):
            raise TypeError("direct role requires DirectPhysicalRegressionSystem")
        output = model(batch.model)
        contribution, metric = distributed_direct_loss_contribution(
            statistic=statistic,
            prediction_mm=output.prediction_mm,
            target_mm=batch.labels.model_target_mm,
            target_validity=batch.labels.target_validity,
            omega=batch.labels.omega,
            denominators=denominators,
        )
        return contribution, metric.reshape(1)

    return direct


@dataclass(frozen=True, slots=True)
class OptimizerLoopResult:
    cursor: TrainingCursor
    stopped_by_limit: bool
    optimizer_steps_run: int
    metrics_step: int


def run_optimizer_loop(
    *,
    model: nn.Module,
    loader: Iterable[RankLocalBatch],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    config: Stage2Config,
    runtime: DistributedRuntime,
    start_cursor: TrainingCursor,
    max_optimizer_steps: int | None,
    forward_loss: ForwardLoss,
    tracking: TrackingRun,
    tracking_step: int,
    role_name: str,
    checkpoint_callback: Callable[[TrainingCursor, bool], None],
) -> OptimizerLoopResult:
    """Consume one accumulation-aligned plan with exact global gradients."""

    accumulation = config.optimization.gradient_accumulation_steps
    microbatch = config.optimization.per_rank_microbatch_size
    if start_cursor.microbatches_since_step != 0:
        raise ValueError("resume checkpoints must lie on optimizer-step boundaries")
    expected_rank_local_slots = start_cursor.optimizer_step * microbatch * accumulation
    if start_cursor.global_example_index != expected_rank_local_slots:
        raise ValueError(
            "resume cursor rank-local slot count disagrees with optimizer steps"
        )
    completed_batches = start_cursor.global_example_index // microbatch
    # DataLoader iterator construction draws a base seed from the global CPU
    # generator even when shuffle=False.  Preserve the restored model RNG
    # exactly; worker randomness is forbidden from defining this fixed plan.
    model_cpu_rng = torch.get_rng_state()
    try:
        iterator = iter(loader)
        for _ in range(completed_batches):
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError("resume cursor exceeds the rank-local plan") from error
    finally:
        torch.set_rng_state(model_cpu_rng)
    optimizer_steps_run = 0
    optimizer_step = start_cursor.optimizer_step
    consumed_batches = completed_batches
    stopped = False
    pending_tracking: tuple[dict[str, object], int] | None = None
    while True:
        if max_optimizer_steps is not None and optimizer_steps_run >= max_optimizer_steps:
            stopped = True
            break
        window: list[RankLocalBatch] = []
        wait_start = time.perf_counter()
        for _ in range(accumulation):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        data_wait = time.perf_counter() - wait_start
        if not window:
            break
        if len(window) != accumulation:
            raise RuntimeError("draw plan ended inside a gradient accumulation window")
        local_labels = [
            value.training.labels
            for value in window
            if isinstance(value.training, TrainingBatch)
        ]
        denominators = accumulation_window_denominators(
            local_labels,
            device=runtime.device,
            reduce_sum=runtime.reduce_sum,
        )
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        optimizer.zero_grad(set_to_none=True)
        h2d_seconds = 0.0
        forward_seconds = 0.0
        backward_seconds = 0.0
        local_samples = 0
        local_metrics: Tensor | None = None
        for value in window:
            if value.training is None:
                backward_start = time.perf_counter()
                _connected_zero(model).backward()
                _sync_cuda(runtime)
                backward_seconds += time.perf_counter() - backward_start
                consumed_batches += 1
                continue
            if not isinstance(value.training, TrainingBatch):
                raise TypeError("production loader returned a non-TrainingBatch")
            _sync_cuda(runtime)
            transfer_start = time.perf_counter()
            batch = value.training.to(runtime.device, non_blocking=True)
            _sync_cuda(runtime)
            h2d_seconds += time.perf_counter() - transfer_start
            local_samples += batch.model.batch_size
            forward_start = time.perf_counter()
            local_loss, metrics = forward_loss(
                model, batch, denominators, optimizer_step
            )
            local_loss = local_loss + _connected_zero(model)
            _sync_cuda(runtime)
            forward_seconds += time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            local_loss.backward()
            _sync_cuda(runtime)
            backward_seconds += time.perf_counter() - backward_start
            local_metrics = metrics if local_metrics is None else local_metrics + metrics
            consumed_batches += 1
        reduce_gradients_sum(model, runtime)
        optimizer_start = time.perf_counter()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimization.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite Stage 2 gradient norm")
        optimizer.step()
        scheduler.step()  # type: ignore[attr-defined]
        _sync_cuda(runtime)
        optimizer_seconds = time.perf_counter() - optimizer_start
        optimizer_step += 1
        optimizer_steps_run += 1
        cursor = TrainingCursor(
            epoch=0,
            global_example_index=consumed_batches * microbatch,
            optimizer_step=optimizer_step,
            microbatches_since_step=0,
        )
        if optimizer_step % config.optimization.checkpoint_every_optimizer_steps == 0:
            checkpoint_callback(cursor, False)

        metric_width = 3 if role_name.startswith("fold") or role_name == "deployment" else 1
        if local_metrics is None:
            local_metrics = torch.zeros(
                metric_width, dtype=torch.float32, device=runtime.device
            )
        global_metrics = runtime.reduce_sum(local_metrics)
        timings = runtime.reduce_max(
            torch.tensor(
                [
                    data_wait,
                    h2d_seconds,
                    forward_seconds,
                    backward_seconds,
                    optimizer_seconds,
                ],
                dtype=torch.float64,
                device=runtime.device,
            )
        )
        global_samples = runtime.reduce_sum(
            torch.tensor(float(local_samples), dtype=torch.float64, device=runtime.device)
        )
        total_seconds = float(timings.sum().item())
        if runtime.device.type == "cuda":
            memory = runtime.reduce_max(
                torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(runtime.device),
                        torch.cuda.max_memory_reserved(runtime.device),
                    ],
                    dtype=torch.float64,
                    device=runtime.device,
                )
            )
        else:
            memory = torch.zeros(2, dtype=torch.float64, device=runtime.device)
        metrics_payload: dict[str, object] = {
            f"{role_name}/data_wait_s": float(timings[0].item()),
            f"{role_name}/h2d_s": float(timings[1].item()),
            f"{role_name}/forward_s": float(timings[2].item()),
            f"{role_name}/backward_s": float(timings[3].item()),
            f"{role_name}/optimizer_s": float(timings[4].item()),
            f"{role_name}/samples_per_s": (
                float(global_samples.item()) / max(total_seconds, 1.0e-12)
            ),
            f"{role_name}/gpu_peak_allocated_bytes": int(memory[0].item()),
            f"{role_name}/gpu_peak_reserved_bytes": int(memory[1].item()),
            f"{role_name}/gradient_norm": float(gradient_norm.item()),
            f"{role_name}/optimizer_step": optimizer_step,
        }
        if metric_width == 3:
            metrics_payload.update(
                {
                    f"{role_name}/loss_occurrence": float(global_metrics[0].item()),
                    f"{role_name}/loss_positive": float(global_metrics[1].item()),
                    f"{role_name}/loss_mean": float(global_metrics[2].item()),
                }
            )
        else:
            metrics_payload[f"{role_name}/loss_direct"] = float(global_metrics[0].item())
        pending_tracking = (metrics_payload, optimizer_step)
        should_log = (
            optimizer_step % config.optimization.log_every_optimizer_steps == 0
            or (
                max_optimizer_steps is not None
                and optimizer_steps_run >= max_optimizer_steps
            )
        )
        if should_log:
            tracking_step += 1
            _distributed_tracking_log(
                tracking, runtime, metrics_payload, step=tracking_step
            )
            pending_tracking = None
    if pending_tracking is not None:
        metrics_payload, _ = pending_tracking
        tracking_step += 1
        _distributed_tracking_log(
            tracking, runtime, metrics_payload, step=tracking_step
        )
    cursor = TrainingCursor(
        epoch=0,
        global_example_index=consumed_batches * microbatch,
        optimizer_step=optimizer_step,
        microbatches_since_step=0,
    )
    return OptimizerLoopResult(
        cursor=cursor,
        stopped_by_limit=stopped,
        optimizer_steps_run=optimizer_steps_run,
        metrics_step=tracking_step,
    )


def _role_seed(seed: int, role: str, fold_id: int | None) -> int:
    payload = f"{seed}\0stage2\0{role}\0{fold_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _checkpoint_provenance(
    *,
    config: Stage2Config,
    factory: KCorrDiffDataFactory,
    runtime: DistributedRuntime,
    launch_identity: Stage2LaunchIdentity,
    role: str,
    fold_id: int | None,
) -> CheckpointProvenance:
    return CheckpointProvenance(
        protocol_version=config.protocol_version,
        config_sha256=config.sha256,
        draw_manifest_sha256=factory.artifacts.artifact_hashes["draw_manifest"],
        launch_identity_sha256=launch_identity.artifact_sha256,
        source_tree_sha256=launch_identity.source_tree_sha256,
        container_image_sha256=launch_identity.container_image_sha256,
        runtime_report_sha256=launch_identity.runtime_report_sha256,
        data_contract_sha256=launch_identity.data_contract_sha256,
        role=role,
        fold_id=fold_id,
        world_size=runtime.world_size,
        per_rank_microbatch_size=config.optimization.per_rank_microbatch_size,
        gradient_accumulation_steps=(
            config.optimization.gradient_accumulation_steps
        ),
    )


def _new_model(
    *, role: str, contract: ModelContract, device: torch.device
) -> nn.Module:
    if role in {"fold", "deployment"}:
        model: nn.Module = RegressionSystem(contract.system)
    elif role == "direct_mean":
        model = DirectPhysicalRegressionSystem(
            statistic="mean", config=contract.system
        )
    elif role == "direct_q50":
        model = DirectPhysicalRegressionSystem(
            statistic="q50", config=contract.system
        )
    else:
        raise ValueError(f"unsupported Stage 2 model role: {role}")
    model.to(device=device, dtype=torch.float32)
    assert_float32_tree(model)
    return model


def _role_name(role: str, fold_id: int | None) -> str:
    return f"fold-{fold_id}" if role == "fold" else role.replace("_", "-")


@dataclass(frozen=True, slots=True)
class RoleTrainingResult:
    role: str
    fold_id: int | None
    complete: bool
    checkpoint_path: Path
    checkpoint_sha256: str
    cursor: TrainingCursor
    optimizer_steps_run: int
    metrics_step: int
    record: CheckpointRecord | None


def train_role(
    *,
    role: str,
    fold_id: int | None,
    config: Stage2Config,
    contract: ModelContract,
    factory: KCorrDiffDataFactory,
    runtime: DistributedRuntime,
    launch_identity: Stage2LaunchIdentity,
    output_dir: Path,
    num_workers: int,
    prefetch_factor: int,
    max_optimizer_steps: int | None,
    tracking: TrackingRun,
    tracking_step: int,
) -> RoleTrainingResult:
    """Train or resume one fold/deployment/direct checkpoint.

    ``TrainingCursor.global_example_index`` is deliberately the number of
    rank-local plan slots consumed by *this* rank, including explicit padding.
    Every rank has the same aligned slot count, so the value is identical
    across ranks.  It is not the draw manifest's global example index.
    """

    if config.optimization.epochs != 1:
        raise ValueError(
            "the immutable draw manifest currently supports exactly one configured epoch"
        )
    plan = factory.plan(
        role=role, fold_id=fold_id, world_size=runtime.world_size
    )
    if plan.gradient_accumulation_steps != (
        config.optimization.gradient_accumulation_steps
    ):
        raise AssertionError("draw plan did not retain accumulation alignment")
    loader = factory.rank_dataloader(
        plan,
        rank=runtime.rank,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    identity = _role_name(role, fold_id)
    role_dir = output_dir.resolve() / identity
    role_dir.mkdir(parents=True, exist_ok=True)
    latest = role_dir / "latest.pt"
    final = role_dir / "final.pt"
    provenance = _checkpoint_provenance(
        config=config,
        factory=factory,
        runtime=runtime,
        launch_identity=launch_identity,
        role=role,
        fold_id=fold_id,
    )
    seed = _role_seed(config.seed, role, fold_id)
    torch.manual_seed(seed)
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _new_model(role=role, contract=contract, device=runtime.device)
    runtime.broadcast_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        betas=config.optimization.betas,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=plan.optimizer_steps,
        eta_min=(
            config.optimization.learning_rate
            * config.optimization.scheduler_minimum_ratio
        ),
    )
    training_blocks = tuple(
        sorted(
            {
                factory.artifacts.rows[index].block_id
                for index in plan.source_row_indices
            }
        )
    )
    if final.exists():
        cursor, extra = load_training_checkpoint(
            final,
            model=model,
            optimizer=None,
            scheduler=None,
            expected_provenance=provenance,
            restore_rng=False,
        )
        if (
            extra.get("complete") is not True
            or extra.get("plan_semantic_sha256") != plan.semantic_sha256
            or cursor.optimizer_step != plan.optimizer_steps
            or cursor.global_example_index
            != plan.optimizer_steps
            * config.optimization.per_rank_microbatch_size
            * config.optimization.gradient_accumulation_steps
        ):
            raise ValueError("final role checkpoint is incomplete or belongs to another plan")
        digest = sha256_file(final)
        record = checkpoint_record(
            final,
            fold_id=fold_id,
            role=role,
            training_blocks=training_blocks,
            config_sha256=config.sha256,
            draw_manifest_sha256=provenance.draw_manifest_sha256,
            global_step=cursor.optimizer_step,
        )
        return RoleTrainingResult(
            role,
            fold_id,
            True,
            final,
            digest,
            cursor,
            0,
            tracking_step,
            record,
        )

    cursor = TrainingCursor(0, 0, 0, 0)
    if latest.exists():
        cursor, extra = load_training_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_provenance=provenance,
            restore_rng=False,
        )
        if extra.get("plan_semantic_sha256") != plan.semantic_sha256:
            raise ValueError("latest checkpoint draw plan hash mismatch")
        rank_states = extra.get("rank_rng_states")
        if not isinstance(rank_states, list) or len(rank_states) != runtime.world_size:
            raise ValueError("latest checkpoint has no complete per-rank RNG state")
        state = rank_states[runtime.rank]
        if not isinstance(state, Mapping):
            raise TypeError("rank RNG state must be a mapping")
        restore_rng_state(state)
    if cursor.optimizer_step > plan.optimizer_steps:
        raise ValueError("checkpoint optimizer step exceeds the role plan")
    expected_rank_slots = cursor.optimizer_step * (
        config.optimization.per_rank_microbatch_size
        * config.optimization.gradient_accumulation_steps
    )
    if cursor.global_example_index != expected_rank_slots:
        raise ValueError(
            "checkpoint cursor rank-local slot count disagrees with optimizer steps"
        )

    def save(cursor_value: TrainingCursor, complete: bool) -> None:
        # The checkout and dependency layer live on a writable PVC.  Detect a
        # mid-run mutation before publishing any resumable state.
        _verify_launch_identity_distributed(launch_identity, runtime)
        rank_rng_states = runtime.all_gather_objects(capture_rng_state())
        target = final if complete else latest
        if runtime.is_rank_zero:
            if complete and target.exists():
                raise FileExistsError(target)
            save_training_checkpoint(
                target,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cursor=cursor_value,
                provenance=provenance,
                extra={
                    "complete": complete,
                    "plan_semantic_sha256": plan.semantic_sha256,
                    "plan_optimizer_steps": plan.optimizer_steps,
                    "training_blocks": list(training_blocks),
                    "rank_rng_states": list(rank_rng_states),
                },
            )
        runtime.barrier()

    model.train()
    loop = run_optimizer_loop(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        runtime=runtime,
        start_cursor=cursor,
        max_optimizer_steps=max_optimizer_steps,
        forward_loss=_production_forward_loss(role=role, config=config),
        tracking=tracking,
        tracking_step=tracking_step,
        role_name=identity,
        checkpoint_callback=save,
    )
    complete = loop.cursor.optimizer_step == plan.optimizer_steps
    if complete:
        save(loop.cursor, True)
        if runtime.is_rank_zero:
            latest.unlink(missing_ok=True)
        runtime.barrier()
        checkpoint_path = final
    else:
        save(loop.cursor, False)
        checkpoint_path = latest
    digest = sha256_file(checkpoint_path)
    gathered_hashes = runtime.all_gather_objects(digest)
    if len(set(gathered_hashes)) != 1:
        raise RuntimeError("ranks observed different role checkpoint bytes")
    record = (
        checkpoint_record(
            final,
            fold_id=fold_id,
            role=role,
            training_blocks=training_blocks,
            config_sha256=config.sha256,
            draw_manifest_sha256=provenance.draw_manifest_sha256,
            global_step=loop.cursor.optimizer_step,
        )
        if complete
        else None
    )
    _distributed_tracking_log(
        tracking,
        runtime,
        {
            f"{identity}/checkpoint_sha256": digest,
            f"{identity}/complete": complete,
            f"{identity}/plan_sha256": plan.semantic_sha256,
        },
        step=loop.metrics_step,
    )
    return RoleTrainingResult(
        role=role,
        fold_id=fold_id,
        complete=complete,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=digest,
        cursor=loop.cursor,
        optimizer_steps_run=loop.optimizer_steps_run,
        metrics_step=loop.metrics_step,
        record=record,
    )


def _oof_rank_loader(
    *,
    factory: KCorrDiffDataFactory,
    row_indices: Sequence[int],
    global_indices: Sequence[int],
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader[RankLocalBatch]:
    if len(row_indices) != len(global_indices):
        raise ValueError("OOF row/global index lengths differ")
    slots = tuple(
        DrawSlot(logical_position=global_index, row_index=row_index)
        for row_index, global_index in zip(row_indices, global_indices, strict=True)
    )
    dataset = LazyRankSlotDataset(
        rows=factory.artifacts.rows,
        slots=slots,
        radar_cache_root=factory.inputs.radar_cache_root,
        era5_cache_root=factory.inputs.era5_cache_root,
        context_regrid_operator=factory.artifacts.context_regrid_operator,
        static_conditions=factory.artifacts.static_conditions,
        condition_signature=factory.artifacts.condition_signatures,
        radar_timezone=factory.config.data.radar_timezone,
        context_detail_mode=factory.config.data.context_detail_mode,
        context_wet_threshold_mm_per_hour=(
            factory.config.data.context_wet_threshold_mm_per_hour
        ),
        verify_cache_hashes=factory.inputs.verify_cache_hashes,
    )
    collator = TrainingBatchCollator(
        factory.artifacts.geometry,
        era_normalization=factory.artifacts.era_normalization,
        device="cpu",
    )
    options = dataloader_options(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
    )
    return DataLoader(
        dataset,
        batch_size=factory.config.optimization.per_rank_microbatch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=RankSlotCollator(collator),
        **options,
    )


def _draw_multiplicities(
    rows: Sequence[DrawRow],
) -> dict[tuple[str, float, str], int]:
    # Besides canonical first-occurrence order, this proves every duplicate
    # semantic key retains sample/block/fold/probability/omega invariants.  A
    # multiplicity replay may never multiply one row's weight across changed
    # metadata.
    unique_oof_rows(rows)
    result: dict[tuple[str, float, str], int] = {}
    for row in rows:
        key = (row.t0_utc, row.lead_hours, row.condition_signature)
        result[key] = result.get(key, 0) + 1
    return result


def _selected_semantic_sha256(
    selected: Sequence[tuple[int, DrawRow]],
    multiplicities: Mapping[tuple[str, float, str], int],
) -> str:
    digest = hashlib.sha256()
    for global_index, row in selected:
        key = (row.t0_utc, row.lead_hours, row.condition_signature)
        payload = (
            global_index,
            row.sample_id,
            *key,
            row.block_id,
            row.fold_id,
            row.omega,
            multiplicities[key],
        )
        digest.update(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _fold_rank_oof_rows(
    rows: Sequence[DrawRow], *, fold_id: int, rank: int, world_size: int
) -> tuple[tuple[int, DrawRow], ...]:
    return tuple(
        (global_index, row)
        for global_index, row in enumerate(unique_oof_rows(rows))
        if row.fold_id == fold_id and global_index % world_size == rank
    )


@dataclass(frozen=True, slots=True)
class _CompressedOOFLayout:
    identities: tuple[OOFIdentity, ...]
    partitions: tuple[OOFPartitionSpec, ...]
    selected: Mapping[str, tuple[tuple[int, int, DrawRow], ...]]


def _compressed_oof_preflight_layout(
    rows: Sequence[DrawRow], *, folds: int, world_size: int
) -> tuple[tuple[OOFIdentity, ...], tuple[OOFPartitionSpec, ...]]:
    """Build the exact-size layout before checkpoint digests exist.

    SHA-256 values are fixed-width, so all index/journal/storage byte counts are
    identical after the real fold checkpoint hashes replace these placeholders.
    """

    unique = unique_oof_rows(rows)
    identities: list[OOFIdentity] = []
    partitions: list[OOFPartitionSpec] = []
    placeholder = "0" * 64
    for fold_id in range(folds):
        for rank in range(world_size):
            selected = tuple(
                row
                for index, row in enumerate(unique)
                if row.fold_id == fold_id and index % world_size == rank
            )
            start = len(identities)
            identities.extend(
                OOFIdentity(
                    sample_id=row.sample_id,
                    t0_utc=row.t0_utc,
                    lead_hours=row.lead_hours,
                    condition_signature=row.condition_signature,
                    block_id=row.block_id,
                    fold_id=fold_id,
                    regression_checkpoint_sha256=placeholder,
                )
                for row in selected
            )
            partitions.append(
                OOFPartitionSpec(
                    partition_id=f"fold-{fold_id}-rank-{rank}",
                    fold_id=fold_id,
                    rank=rank,
                    world_size=world_size,
                    row_indices=tuple(range(start, len(identities))),
                    regression_checkpoint_sha256=placeholder,
                )
            )
    if len(identities) != len(unique):
        raise ValueError("OOF preflight partition coverage mismatch")
    return tuple(identities), tuple(partitions)


def _compressed_oof_layout(
    rows: Sequence[DrawRow],
    *,
    fold_results: Mapping[int, RoleTrainingResult],
    folds: int,
    world_size: int,
) -> _CompressedOOFLayout:
    """Order artifact rows by fold/rank so each rank can stream whole shards."""

    identities: list[OOFIdentity] = []
    partitions: list[OOFPartitionSpec] = []
    selected_by_partition: dict[str, tuple[tuple[int, int, DrawRow], ...]] = {}
    for fold_id in range(folds):
        result = fold_results.get(fold_id)
        if result is None or not result.complete or result.record is None:
            raise RuntimeError(
                f"compressed OOF requires complete fold checkpoint {fold_id}"
            )
        for rank in range(world_size):
            source = _fold_rank_oof_rows(
                rows, fold_id=fold_id, rank=rank, world_size=world_size
            )
            partition_id = f"fold-{fold_id}-rank-{rank}"
            partition_rows: list[tuple[int, int, DrawRow]] = []
            start = len(identities)
            for source_index, row in source:
                artifact_index = len(identities)
                identity = OOFIdentity(
                    sample_id=row.sample_id,
                    t0_utc=row.t0_utc,
                    lead_hours=row.lead_hours,
                    condition_signature=row.condition_signature,
                    block_id=row.block_id,
                    fold_id=fold_id,
                    regression_checkpoint_sha256=result.checkpoint_sha256,
                )
                identity.validate()
                identities.append(identity)
                partition_rows.append((artifact_index, source_index, row))
            partitions.append(
                OOFPartitionSpec(
                    partition_id=partition_id,
                    fold_id=fold_id,
                    rank=rank,
                    world_size=world_size,
                    row_indices=tuple(range(start, len(identities))),
                    regression_checkpoint_sha256=result.checkpoint_sha256,
                )
            )
            selected_by_partition[partition_id] = tuple(partition_rows)
    if len(identities) != len(unique_oof_rows(rows)):
        raise ValueError("compressed OOF partitioning lost or duplicated draw rows")
    if len({identity.semantic_key for identity in identities}) != len(identities):
        raise ValueError("compressed OOF partitioning duplicated a semantic row")
    return _CompressedOOFLayout(
        identities=tuple(identities),
        partitions=tuple(partitions),
        selected=selected_by_partition,
    )


def _write_residual_state(
    path: Path,
    accumulator: ResidualScaleAccumulator,
    *,
    config: Stage2Config,
    factory: KCorrDiffDataFactory,
    fold_result: RoleTrainingResult,
    fold_id: int,
    rank: int,
    selected: Sequence[tuple[int, DrawRow]],
    multiplicities: Mapping[tuple[str, float, str], int],
    oof_partial_manifest_sha256: str,
) -> str:
    draw_rows = sum(
        multiplicities[(row.t0_utc, row.lead_hours, row.condition_signature)]
        for _, row in selected
    )
    payload = {
        "format_version": RESIDUAL_PARTIAL_FORMAT,
        "protocol_version": config.protocol_version,
        "config_sha256": config.sha256,
        "draw_manifest_sha256": factory.artifacts.artifact_hashes["draw_manifest"],
        "oof_partial_manifest_sha256": oof_partial_manifest_sha256,
        "regression_checkpoint_sha256": fold_result.checkpoint_sha256,
        "target_builder_version": TARGET_BUILDER_VERSION,
        "fold_id": fold_id,
        "rank": rank,
        "unique_oof_items": len(selected),
        "training_draw_rows": draw_rows,
        "selected_semantic_sha256": _selected_semantic_sha256(
            selected, multiplicities
        ),
        "accumulator_state": accumulator.merge_state(),
    }
    digest = _atomic_json(path, payload, replace=False)
    _atomic_text(
        path.with_suffix(".sha256"),
        f"{digest}  {path.name}\n",
        replace=False,
    )
    return digest


def _read_residual_state(
    path: Path,
    *,
    config: Stage2Config,
    factory: KCorrDiffDataFactory,
    fold_result: RoleTrainingResult,
    fold_id: int,
    rank: int,
    selected: Sequence[tuple[int, DrawRow]],
    multiplicities: Mapping[tuple[str, float, str], int],
    oof_partial_manifest_sha256: str,
) -> ResidualScaleAccumulator:
    hash_path = path.with_suffix(".sha256")
    if not path.is_file() or not hash_path.is_file():
        raise ValueError("OOF residual partial state/hash is missing")
    digest = sha256_file(path)
    if hash_path.read_text(encoding="utf-8") != f"{digest}  {path.name}\n":
        raise ValueError("OOF residual partial state SHA-256 mismatch")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("residual partial state root must be a mapping")
    draw_rows = sum(
        multiplicities[(row.t0_utc, row.lead_hours, row.condition_signature)]
        for _, row in selected
    )
    expected = {
        "format_version": RESIDUAL_PARTIAL_FORMAT,
        "protocol_version": config.protocol_version,
        "config_sha256": config.sha256,
        "draw_manifest_sha256": factory.artifacts.artifact_hashes["draw_manifest"],
        "oof_partial_manifest_sha256": oof_partial_manifest_sha256,
        "regression_checkpoint_sha256": fold_result.checkpoint_sha256,
        "target_builder_version": TARGET_BUILDER_VERSION,
        "fold_id": fold_id,
        "rank": rank,
        "unique_oof_items": len(selected),
        "training_draw_rows": draw_rows,
        "selected_semantic_sha256": _selected_semantic_sha256(
            selected, multiplicities
        ),
    }
    for name, value in expected.items():
        if raw.get(name) != value:
            raise ValueError(f"OOF residual partial provenance mismatch: {name}")
    if set(raw) != {*expected, "accumulator_state"}:
        raise ValueError("OOF residual partial state schema mismatch")
    state = raw["accumulator_state"]
    if not isinstance(state, Mapping):
        raise TypeError("residual accumulator state must be a mapping")
    accumulator = ResidualScaleAccumulator.from_merge_state(state)
    actual_items = sum(
        int(record["items"])
        for record in accumulator.merge_state()["records"]  # type: ignore[index]
    )
    if actual_items != draw_rows:
        raise ValueError("OOF residual partial lost training draw multiplicity")
    return accumulator


def _shutdown_dataloader(loader: DataLoader[object]) -> None:
    """Close persistent workers promptly before the next full-width fold."""

    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        loader._iterator = None  # type: ignore[attr-defined]


def infer_oof_rank_partials(
    *,
    config: Stage2Config,
    contract: ModelContract,
    factory: KCorrDiffDataFactory,
    runtime: DistributedRuntime,
    launch_identity: Stage2LaunchIdentity,
    fold_results: Mapping[int, RoleTrainingResult],
    oof_output_dir: Path,
    num_workers: int,
    prefetch_factor: int,
    remote_store: RemoteShardStore | None = None,
) -> tuple[Path, ...]:
    """Stream rank-disjoint OOF rows into lossless compressed final shards."""

    multiplicities = _draw_multiplicities(factory.artifacts.rows)
    layout = _compressed_oof_layout(
        factory.artifacts.rows,
        fold_results=fold_results,
        folds=config.crossfit.folds,
        world_size=runtime.world_size,
    )
    first_index: dict[tuple[str, float, str], int] = {}
    for index, row in enumerate(factory.artifacts.rows):
        first_index.setdefault(
            (row.t0_utc, row.lead_hours, row.condition_signature), index
        )
    final_dir = oof_output_dir / "artifact"
    initialization_failure: str | None = None
    if runtime.is_rank_zero:
        try:
            OOFCompressedBuild(
                final_dir,
                identities=layout.identities,
                partitions=layout.partitions,
                maximum_compressed_bytes=(
                    config.crossfit.maximum_oof_compressed_bytes
                ),
                maximum_compression_ratio=(
                    config.crossfit.maximum_oof_compression_ratio
                ),
                compression_probe_bytes=(
                    config.crossfit.oof_compression_probe_bytes
                ),
                concurrent_writers=runtime.world_size,
                shard_items=config.crossfit.oof_shard_items,
                protocol_version=config.protocol_version,
                draw_manifest_sha256=(
                    factory.artifacts.artifact_hashes["draw_manifest"]
                ),
                target_builder_version=TARGET_BUILDER_VERSION,
                config_sha256=config.sha256,
                launch_identity_sha256=launch_identity.artifact_sha256,
                remote_store=remote_store,
                local_reserve_bytes=config.crossfit.remote_spill.local_reserve_bytes,
                initialize=True,
            )
        except BaseException as error:
            initialization_failure = (
                f"rank=0,type={type(error).__name__},message={error}"
            )
    failures = [
        value
        for value in runtime.all_gather_objects(initialization_failure)
        if value is not None
    ]
    if failures:
        raise RuntimeError(f"compressed OOF initialization failed: {failures}")
    runtime.barrier()
    build = OOFCompressedBuild(
        final_dir,
        identities=layout.identities,
        partitions=layout.partitions,
        maximum_compressed_bytes=config.crossfit.maximum_oof_compressed_bytes,
        maximum_compression_ratio=config.crossfit.maximum_oof_compression_ratio,
        compression_probe_bytes=config.crossfit.oof_compression_probe_bytes,
        concurrent_writers=runtime.world_size,
        shard_items=config.crossfit.oof_shard_items,
        protocol_version=config.protocol_version,
        draw_manifest_sha256=factory.artifacts.artifact_hashes["draw_manifest"],
        target_builder_version=TARGET_BUILDER_VERSION,
        config_sha256=config.sha256,
        launch_identity_sha256=launch_identity.artifact_sha256,
        remote_store=remote_store,
        local_reserve_bytes=config.crossfit.remote_spill.local_reserve_bytes,
        initialize=False,
    )
    partial_paths: list[Path] = []
    for fold_id in range(config.crossfit.folds):
        result = fold_results.get(fold_id)
        if result is None or not result.complete or result.record is None:
            raise RuntimeError(f"OOF requires complete held-out fold checkpoint {fold_id}")
        partition_id = f"fold-{fold_id}-rank-{runtime.rank}"
        compressed_selected = layout.selected[partition_id]
        selected = tuple(
            (source_index, row)
            for _, source_index, row in compressed_selected
        )
        partial = oof_output_dir / f".partial-fold-{fold_id}-rank-{runtime.rank}"
        partial_paths.append(partial)
        residual_path = partial / "residual-state.json"
        partial_manifest = partial / "manifest.json"
        if final_dir.exists():
            if not partial_manifest.is_file():
                raise ValueError("sealed OOF artifact has no residual partition receipt")
            _read_residual_state(
                residual_path,
                config=config,
                factory=factory,
                fold_result=result,
                fold_id=fold_id,
                rank=runtime.rank,
                selected=selected,
                multiplicities=multiplicities,
                oof_partial_manifest_sha256=sha256_file(
                    partial_manifest
                ),
            )
            continue
        writer = build.partition(partition_id)
        if writer.complete and partial_manifest.is_file() and residual_path.is_file():
            _read_residual_state(
                residual_path,
                config=config,
                factory=factory,
                fold_result=result,
                fold_id=fold_id,
                rank=runtime.rank,
                selected=selected,
                multiplicities=multiplicities,
                oof_partial_manifest_sha256=sha256_file(partial_manifest),
            )
            writer.abort()
            continue
        row_indices = [
            first_index[(row.t0_utc, row.lead_hours, row.condition_signature)]
            for _, row in selected
        ]
        loader = _oof_rank_loader(
            factory=factory,
            row_indices=row_indices,
            global_indices=[index for index, _ in selected],
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        provenance = _checkpoint_provenance(
            config=config,
            factory=factory,
            runtime=runtime,
            launch_identity=launch_identity,
            role="fold",
            fold_id=fold_id,
        )
        model = _new_model(role="fold", contract=contract, device=runtime.device)
        cursor, extra = load_training_checkpoint(
            result.checkpoint_path,
            model=model,
            optimizer=None,
            scheduler=None,
            expected_provenance=provenance,
            restore_rng=False,
        )
        if extra.get("complete") is not True or cursor.optimizer_step <= 0:
            raise ValueError("OOF checkpoint is not a completed fold fit")
        model.eval()
        accumulator = ResidualScaleAccumulator(
            epsilon_scale=config.crossfit.epsilon_residual_scale
        )
        selected_offset = 0
        try:
            with torch.no_grad():
                for rank_batch in loader:
                    if rank_batch.training is None:
                        raise AssertionError("OOF loaders never contain padding")
                    if not isinstance(rank_batch.training, TrainingBatch):
                        raise TypeError("OOF loader returned a non-TrainingBatch")
                    batch = rank_batch.training.to(runtime.device, non_blocking=True)
                    output = model(batch.model)
                    assert isinstance(output, object)
                    sample_ids = batch.model.provenance.sample_ids
                    for batch_index, sample_id in enumerate(sample_ids):
                        if selected_offset >= len(compressed_selected):
                            raise ValueError("OOF loader produced an extra sample")
                        artifact_index, _, expected_row = compressed_selected[
                            selected_offset
                        ]
                        row = factory.artifacts.rows[
                            rank_batch.slots[batch_index].row_index  # type: ignore[index]
                        ]
                        if (
                            row.fold_id != fold_id
                            or row.sample_id != sample_id
                            or (
                                row.t0_utc,
                                row.lead_hours,
                                row.condition_signature,
                            )
                            != (
                                expected_row.t0_utc,
                                expected_row.lead_hours,
                                expected_row.condition_signature,
                            )
                        ):
                            raise ValueError("OOF batch identity/fold mismatch")
                        key = (
                            row.t0_utc,
                            row.lead_hours,
                            row.condition_signature,
                        )
                        writer.append(
                            artifact_index,
                            layout.identities[artifact_index],
                            output.probability_wet[batch_index, 0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False),
                            output.mu_z[batch_index, 0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False),
                        )
                        accumulator.update(
                            lead_hours=row.lead_hours,
                            condition_signature=row.condition_signature,
                            target_z=batch.labels.target_z[batch_index, 0]
                            .detach()
                            .cpu()
                            .numpy(),
                            mu_z_oof=output.mu_z[batch_index, 0]
                            .detach()
                            .cpu()
                            .numpy(),
                            target_validity=batch.labels.target_validity[
                                batch_index, 0
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            omega=float(batch.labels.omega[batch_index].item()),
                            multiplicity=multiplicities[key],
                        )
                        selected_offset += 1
        finally:
            _shutdown_dataloader(loader)
        if selected_offset != len(compressed_selected):
            writer.abort()
            raise ValueError("OOF loader stopped before its declared partition ended")
        partial.mkdir(parents=True, exist_ok=True)
        partial_manifest_sha256 = writer.finish(manifest_path=partial_manifest)
        _write_residual_state(
            residual_path,
            accumulator,
            config=config,
            factory=factory,
            fold_result=result,
            fold_id=fold_id,
            rank=runtime.rank,
            selected=selected,
            multiplicities=multiplicities,
            oof_partial_manifest_sha256=partial_manifest_sha256,
        )
        del model
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()
    runtime.barrier()
    return tuple(partial_paths)


@dataclass(frozen=True, slots=True)
class OOFMergeResult:
    artifact_dir: Path
    manifest_sha256: str
    residual_accumulator: ResidualScaleAccumulator
    items: int


def _validate_oof_identity_sequence(
    *,
    rows: Sequence[DrawRow],
    identities: Sequence[OOFIdentity],
    fold_results: Mapping[int, RoleTrainingResult],
) -> None:
    unique = unique_oof_rows(rows)
    if len(identities) != len(unique):
        raise ValueError("published OOF identity count disagrees with draw universe")
    expected = {
        (row.t0_utc, row.lead_hours, row.condition_signature): row
        for row in unique
    }
    seen: set[tuple[str, float, str]] = set()
    for identity in identities:
        identity.validate()
        if identity.semantic_key in seen:
            raise ValueError("published OOF identities contain a duplicate row")
        seen.add(identity.semantic_key)
        try:
            row = expected[identity.semantic_key]
        except KeyError as error:
            raise ValueError("published OOF identity is outside the draw universe") from error
        fold = row.fold_id
        if fold is None or fold not in fold_results:
            raise ValueError("OOF row has no completed fold checkpoint")
        result = fold_results[fold]
        if not result.complete or result.record is None:
            raise ValueError("OOF identity references an incomplete fold fit")
        if (
            identity.semantic_key
            != (row.t0_utc, row.lead_hours, row.condition_signature)
            or identity.sample_id != row.sample_id
            or identity.block_id != row.block_id
            or identity.fold_id != fold
            or identity.regression_checkpoint_sha256
            != result.checkpoint_sha256
        ):
            raise ValueError("published OOF identity/checkpoint provenance mismatch")
    if seen != set(expected):
        raise ValueError("published OOF identities do not cover the draw universe")


def merge_oof_partials(
    *,
    config: Stage2Config,
    factory: KCorrDiffDataFactory,
    runtime: DistributedRuntime,
    launch_identity: Stage2LaunchIdentity,
    fold_results: Mapping[int, RoleTrainingResult],
    oof_output_dir: Path,
    remote_store: RemoteShardStore | None = None,
) -> OOFMergeResult | None:
    """Rank-zero metadata-only seal and residual-state merge."""

    runtime.barrier()
    if not runtime.is_rank_zero:
        return None
    unique = unique_oof_rows(factory.artifacts.rows)
    layout = _compressed_oof_layout(
        factory.artifacts.rows,
        fold_results=fold_results,
        folds=config.crossfit.folds,
        world_size=runtime.world_size,
    )
    partials = tuple(
        oof_output_dir / f".partial-fold-{fold}-rank-{rank}"
        for fold in range(config.crossfit.folds)
        for rank in range(runtime.world_size)
    )
    multiplicities = _draw_multiplicities(factory.artifacts.rows)
    merged_accumulator = ResidualScaleAccumulator(
        epsilon_scale=config.crossfit.epsilon_residual_scale
    )
    for path in partials:
        name = path.name
        try:
            fold_id = int(name.split("-fold-", 1)[1].split("-rank-", 1)[0])
            rank = int(name.rsplit("-rank-", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"invalid OOF partial directory identity: {name}") from error
        result = fold_results.get(fold_id)
        if result is None:
            raise ValueError("OOF residual state references an unknown fold")
        selected = _fold_rank_oof_rows(
            factory.artifacts.rows,
            fold_id=fold_id,
            rank=rank,
            world_size=runtime.world_size,
        )
        if not (path / "manifest.json").is_file():
            raise ValueError("OOF partition manifest is missing")
        merged_accumulator.merge(
            _read_residual_state(
                path / "residual-state.json",
                config=config,
                factory=factory,
                fold_result=result,
                fold_id=fold_id,
                rank=rank,
                selected=selected,
                multiplicities=multiplicities,
                oof_partial_manifest_sha256=sha256_file(
                    path / "manifest.json"
                ),
            )
        )
    final_dir = oof_output_dir / "artifact"
    build = OOFCompressedBuild(
        final_dir,
        identities=layout.identities,
        partitions=layout.partitions,
        maximum_compressed_bytes=config.crossfit.maximum_oof_compressed_bytes,
        maximum_compression_ratio=config.crossfit.maximum_oof_compression_ratio,
        compression_probe_bytes=config.crossfit.oof_compression_probe_bytes,
        concurrent_writers=runtime.world_size,
        shard_items=config.crossfit.oof_shard_items,
        protocol_version=config.protocol_version,
        draw_manifest_sha256=factory.artifacts.artifact_hashes["draw_manifest"],
        target_builder_version=TARGET_BUILDER_VERSION,
        config_sha256=config.sha256,
        launch_identity_sha256=launch_identity.artifact_sha256,
        remote_store=remote_store,
        local_reserve_bytes=config.crossfit.remote_spill.local_reserve_bytes,
        initialize=False,
    )
    digest = build.seal()
    verified = OOFArtifact(
        final_dir, verify_hashes=True, remote_store=remote_store
    )
    _validate_oof_identity_sequence(
        rows=factory.artifacts.rows,
        identities=verified.identities,
        fold_results=fold_results,
    )
    return OOFMergeResult(final_dir, digest, merged_accumulator, len(unique))


def _cleanup_oof_partials(
    *, oof_output_dir: Path, folds: int, world_size: int
) -> None:
    for fold in range(folds):
        for rank in range(world_size):
            path = oof_output_dir / f".partial-fold-{fold}-rank-{rank}"
            if path.is_dir():
                shutil.rmtree(path)


def _validate_residual_scale_coverage(
    accumulator: ResidualScaleAccumulator, rows: Sequence[DrawRow]
) -> None:
    """Prove one scale cell/update for every canonical OOF identity."""

    expected: dict[tuple[float, str], int] = {}
    for row in rows:
        key = (row.lead_hours, row.condition_signature)
        expected[key] = expected.get(key, 0) + 1
    raw = accumulator.result()
    records = raw.get("records")
    if not isinstance(records, list):
        raise TypeError("residual scale result records must be a list")
    actual: dict[tuple[float, str], int] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("residual scale record must be a mapping")
        key = (float(record["lead_hours"]), str(record["condition_signature"]))
        if key in actual:
            raise ValueError(f"duplicate residual scale result cell: {key}")
        actual[key] = int(record["items"])
    if actual != expected:
        raise ValueError(
            "residual scale cells/items do not exactly cover the training draw universe"
        )


def _validate_existing_residual_scales(
    path: Path,
    accumulator: ResidualScaleAccumulator,
    *,
    oof_manifest_sha256: str,
    regression_checkpoint_set_sha256: str,
) -> str:
    """Accept a resumed scale artifact only if every byte is reproducible."""

    expected = accumulator.result()
    expected["oof_manifest_sha256"] = oof_manifest_sha256
    expected["regression_checkpoint_set_sha256"] = (
        regression_checkpoint_set_sha256
    )
    expected_bytes = (
        json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    actual_bytes = path.read_bytes()
    if actual_bytes != expected_bytes:
        raise ValueError(
            "existing residual-scale values/provenance are not reproducible "
            "from verified OOF sufficient statistics"
        )
    return _hash_bytes(actual_bytes)


@dataclass(frozen=True, slots=True)
class Stage2RunResult:
    partial: bool
    stopped_by_limit: bool
    manifest_path: Path
    manifest_sha256: str
    role_results: tuple[RoleTrainingResult, ...]
    oof_manifest_sha256: str | None
    residual_scales_sha256: str | None


def _tracking_contract(
    config: Stage2Config, *, requested_mode: str, partial: bool
) -> tuple[bool, str, str, str]:
    raw = _mapping(config.raw.get("tracking"), name="tracking config")
    required = {
        "backend",
        "project",
        "job_type",
        "mode",
        "log_gpu_system_metrics",
    }
    if set(raw) != required or raw["backend"] != "wandb":
        raise ValueError("Stage 2 tracking config schema/backend mismatch")
    if raw["log_gpu_system_metrics"] is not True:
        raise ValueError("Stage 2 must retain rank-max GPU system metrics")
    if not isinstance(raw["project"], str) or not raw["project"]:
        raise ValueError("Stage 2 W&B project must be non-empty")
    if not isinstance(raw["job_type"], str) or not raw["job_type"]:
        raise ValueError("Stage 2 W&B job_type must be non-empty")
    configured = str(raw["mode"])
    if configured not in {"online", "offline", "disabled"}:
        raise ValueError("unsupported configured W&B mode")
    effective = configured if requested_mode == "config" else requested_mode
    if effective not in {"online", "offline", "disabled"}:
        raise ValueError("unsupported requested W&B mode")
    if not partial and effective != configured:
        raise ValueError("full Stage 2 run cannot override configured W&B mode")
    return (
        effective != "disabled",
        effective,
        str(raw["project"]),
        str(raw["job_type"]),
    )


def _publication_contract(config: Stage2Config) -> None:
    raw = _mapping(config.raw.get("publication"), name="publication config")
    expected = {
        "require_all_fold_checkpoints": True,
        "require_complete_oof": True,
        "require_residual_scales": True,
        "atomic_stage_manifest": True,
        "immutable_release_requires_manual_promotion": True,
    }
    if dict(raw) != expected:
        raise ValueError("Stage 2 publication contract changed or is incomplete")


def _record_json(record: CheckpointRecord) -> dict[str, object]:
    return asdict(record)


def _role_result_json(result: RoleTrainingResult) -> dict[str, object]:
    """Serialize both immutable finals and resumable ``latest.pt`` states."""

    return {
        "role": result.role,
        "fold_id": result.fold_id,
        "complete": result.complete,
        "checkpoint_path": str(result.checkpoint_path.resolve()),
        "checkpoint_sha256": result.checkpoint_sha256,
        "cursor": asdict(result.cursor),
        "optimizer_steps_run_this_invocation": result.optimizer_steps_run,
    }


_LAUNCH_IDENTITY_KEYS = {
    "artifact_sha256",
    "source_tree_sha256",
    "container_image_sha256",
    "runtime_report_sha256",
    "data_contract_sha256",
}


def _validated_launch_identity_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _LAUNCH_IDENTITY_KEYS:
        raise ValueError("Stage 2 manifest launch identity schema mismatch")
    result: dict[str, str] = {}
    for name in sorted(_LAUNCH_IDENTITY_KEYS):
        digest = value[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Stage 2 manifest has invalid launch identity {name}")
        result[name] = digest
    return result


def _validate_existing_stage2_run_identity(
    *,
    output_dir: Path,
    config: Stage2Config,
    factory: KCorrDiffDataFactory,
    launch_identity: Stage2LaunchIdentity,
) -> None:
    """Reject a resumed output directory from any other immutable launch."""

    expected_identity = launch_identity.provenance()
    expected_draw = factory.artifacts.artifact_hashes["draw_manifest"]
    for name in ("partial-manifest.json", "stage2-manifest.json"):
        path = output_dir.resolve() / name
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"existing {name} is not valid UTF-8 JSON") from error
        if not isinstance(raw, Mapping):
            raise TypeError(f"existing {name} root must be a mapping")
        if raw.get("config_sha256") != config.sha256:
            raise ValueError(f"existing {name} config SHA-256 mismatch")
        if raw.get("draw_manifest_sha256") != expected_draw:
            raise ValueError(f"existing {name} draw-manifest SHA-256 mismatch")
        if _validated_launch_identity_mapping(raw.get("launch_identity")) != expected_identity:
            raise ValueError(f"existing {name} immutable launch identity mismatch")


def load_complete_stage2_manifest(
    path: Path, *, expected_sha256: str
) -> Mapping[str, object]:
    """Strict downstream boundary that can never accept a partial run audit."""

    selected = path.resolve()
    if selected.name != "stage2-manifest.json":
        raise ValueError("downstream consumers require exact stage2-manifest.json")
    if sha256_file(selected) != expected_sha256:
        raise ValueError("Stage 2 manifest SHA-256 mismatch")
    raw = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("Stage 2 manifest root must be a mapping")
    if (
        raw.get("format_version") != STAGE2_MANIFEST_FORMAT
        or raw.get("protocol_version") != "v1.1.3b"
        or raw.get("complete") is not True
        or raw.get("partial") is not False
    ):
        raise ValueError("downstream consumers require a complete Stage 2 manifest")
    _validated_launch_identity_mapping(raw.get("launch_identity"))
    records = raw.get("role_checkpoints")
    if not isinstance(records, list):
        raise TypeError("complete Stage 2 manifest has no checkpoint records")
    identities = {
        (record.get("role"), record.get("fold_id"))
        for record in records
        if isinstance(record, Mapping)
    }
    if len(identities) != len(records):
        raise ValueError("complete Stage 2 manifest has malformed/duplicate checkpoints")
    fold_ids = sorted(
        int(fold)
        for role, fold in identities
        if role == "fold" and isinstance(fold, int) and not isinstance(fold, bool)
    )
    if fold_ids not in ([0, 1, 2], [0, 1, 2, 3, 4]):
        raise ValueError("complete Stage 2 manifest lacks the grouped fold set")
    required = {
        ("deployment", None),
        ("direct_mean", None),
        ("direct_q50", None),
    }
    if not required.issubset(identities):
        raise ValueError("complete Stage 2 manifest lacks required comparison arms")
    for name in (
        "oof_manifest_sha256",
        "residual_scales_sha256",
        "fold_checkpoint_set_sha256",
    ):
        value = raw.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"complete Stage 2 manifest has invalid {name}")
    return raw


def _initialize_tracking_distributed(
    *,
    enabled: bool,
    runtime: DistributedRuntime,
    run_dir: Path,
    run_id: str,
    project: str,
    job_type: str,
    config: Mapping[str, object],
    config_sha256: str,
    launch_identity_sha256: str,
    mode: str,
) -> tuple[TrackingRun, int]:
    tracking: TrackingRun | None = None
    failure: str | None = None
    try:
        tracking = initialize_tracking(
            enabled=enabled,
            rank=runtime.rank,
            run_dir=run_dir,
            run_id=run_id,
            project=project,
            job_type=job_type,
            config=config,
            config_sha256=config_sha256,
            launch_identity_sha256=launch_identity_sha256,
            mode=mode,
            resume="allow",
        )
    except BaseException as error:
        failure = f"rank={runtime.rank},type={type(error).__name__}"
    failures = [
        value
        for value in runtime.all_gather_objects(failure)
        if value is not None
    ]
    if failures:
        raise RuntimeError(f"distributed tracking initialization failed: {failures}")
    assert tracking is not None
    rank_zero_step: int | None = tracking.last_step if runtime.is_rank_zero else None
    gathered_steps = runtime.all_gather_objects(rank_zero_step)
    selected = [value for value in gathered_steps if value is not None]
    if len(selected) != 1 or isinstance(selected[0], bool):
        raise RuntimeError("rank-zero tracking resume step was not uniquely propagated")
    return tracking, int(selected[0])


def run_stage2(
    *,
    config: Stage2Config,
    factory: KCorrDiffDataFactory,
    runtime: DistributedRuntime,
    launch_identity: Stage2LaunchIdentity,
    output_dir: Path,
    oof_output_dir: Path,
    selection: RunSelection,
    num_workers: int,
    prefetch_factor: int,
    wandb_mode: str,
    run_id: str,
    remote_store: RemoteShardStore | None = None,
) -> Stage2RunResult:
    """Execute the selected DAG; publish ``complete`` only for the full DAG."""

    contract = regression_system_config_from_stage2(config)
    _publication_contract(config)
    if not contract.direct_mean_required or not contract.direct_q50_required:
        raise ValueError("full protocol requires direct mean and q50 checkpoints")
    invalid_folds = [
        role
        for role in selection.roles
        if role.startswith("fold:") and int(role.split(":", 1)[1]) >= config.crossfit.folds
    ]
    if invalid_folds:
        raise ValueError(f"selected folds exceed configured fold count: {invalid_folds}")
    enabled, effective_mode, project, job_type = _tracking_contract(
        config, requested_mode=wandb_mode, partial=selection.partial
    )
    _verify_launch_identity_distributed(launch_identity, runtime)
    _validate_existing_stage2_run_identity(
        output_dir=output_dir,
        config=config,
        factory=factory,
        launch_identity=launch_identity,
    )
    tracking_config = dict(config.raw)
    tracking_config["stage2_config_sha256"] = config.sha256
    tracking_config["launch_identity"] = launch_identity.provenance()
    tracking, tracking_step = _initialize_tracking_distributed(
        enabled=enabled,
        runtime=runtime,
        run_dir=output_dir / "tracking",
        run_id=run_id,
        project=project,
        job_type=job_type,
        config=tracking_config,
        config_sha256=config.sha256,
        launch_identity_sha256=launch_identity.artifact_sha256,
        mode=effective_mode,
    )
    role_results: list[RoleTrainingResult] = []
    fold_results: dict[int, RoleTrainingResult] = {}
    remaining = selection.max_optimizer_steps
    stopped = False
    try:
        if "crossfit" in selection.phases:
            for fold in range(config.crossfit.folds):
                if not selection.includes_role("fold", fold):
                    continue
                result = train_role(
                    role="fold",
                    fold_id=fold,
                    config=config,
                    contract=contract,
                    factory=factory,
                    runtime=runtime,
                    launch_identity=launch_identity,
                    output_dir=output_dir,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    max_optimizer_steps=remaining,
                    tracking=tracking,
                    tracking_step=tracking_step,
                )
                role_results.append(result)
                fold_results[fold] = result
                tracking_step = result.metrics_step
                if remaining is not None:
                    remaining -= result.optimizer_steps_run
                    if remaining <= 0:
                        stopped = True
                        break
        # OOF may be run in a later resumed invocation; load complete folds by
        # calling train_role with a zero work budget only after final existence.
        if not stopped and (
            "oof" in selection.phases or "residual_scales" in selection.phases
        ):
            for fold in range(config.crossfit.folds):
                if fold in fold_results and fold_results[fold].complete:
                    continue
                final_path = output_dir / f"fold-{fold}" / "final.pt"
                if not final_path.is_file():
                    raise RuntimeError(f"OOF requires missing fold-{fold}/final.pt")
                result = train_role(
                    role="fold",
                    fold_id=fold,
                    config=config,
                    contract=contract,
                    factory=factory,
                    runtime=runtime,
                    launch_identity=launch_identity,
                    output_dir=output_dir,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    max_optimizer_steps=0,
                    tracking=tracking,
                    tracking_step=tracking_step,
                )
                fold_results[fold] = result

        oof_result: OOFMergeResult | None = None
        scale_hash: str | None = None
        fold_set_hash: str | None = None
        if not stopped and "oof" in selection.phases:
            infer_oof_rank_partials(
                config=config,
                contract=contract,
                factory=factory,
                runtime=runtime,
                launch_identity=launch_identity,
                fold_results=fold_results,
                oof_output_dir=oof_output_dir,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                remote_store=remote_store,
            )
            oof_result = merge_oof_partials(
                config=config,
                factory=factory,
                runtime=runtime,
                launch_identity=launch_identity,
                fold_results=fold_results,
                oof_output_dir=oof_output_dir,
                remote_store=remote_store,
            )
            runtime.barrier()
            metadata = runtime.all_gather_objects(
                (
                    oof_result.manifest_sha256,
                    oof_result.items,
                )
                if oof_result is not None
                else None
            )
            published = [value for value in metadata if value is not None]
            if len(published) != 1:
                raise RuntimeError("rank-zero OOF merge metadata was not unique")
            if oof_result is None:
                digest, items = published[0]  # type: ignore[misc]
                oof_result = OOFMergeResult(
                    oof_output_dir / "artifact",
                    str(digest),
                    ResidualScaleAccumulator(
                        epsilon_scale=config.crossfit.epsilon_residual_scale
                    ),
                    int(items),
                )
            tracking_step += 1
            _distributed_tracking_log(
                tracking,
                runtime,
                {
                    "artifacts/oof_manifest_sha256": oof_result.manifest_sha256,
                    "artifacts/oof_items": oof_result.items,
                },
                step=tracking_step,
            )
        elif not stopped and "residual_scales" in selection.phases:
            final_oof = oof_output_dir / "artifact"
            artifact = OOFArtifact(
                final_oof, verify_hashes=True, remote_store=remote_store
            )
            oof_hash = sha256_file(final_oof / "manifest.json")
            # Residual partial sufficient statistics are intentionally kept
            # until scale publication succeeds.
            merged = ResidualScaleAccumulator(
                epsilon_scale=config.crossfit.epsilon_residual_scale
            )
            multiplicities = _draw_multiplicities(factory.artifacts.rows)
            for fold in range(config.crossfit.folds):
                for rank in range(runtime.world_size):
                    partial_path = (
                        oof_output_dir / f".partial-fold-{fold}-rank-{rank}"
                    )
                    result = fold_results[fold]
                    merged.merge(
                        _read_residual_state(
                            partial_path / "residual-state.json",
                            config=config,
                            factory=factory,
                            fold_result=result,
                            fold_id=fold,
                            rank=rank,
                            selected=_fold_rank_oof_rows(
                                factory.artifacts.rows,
                                fold_id=fold,
                                rank=rank,
                                world_size=runtime.world_size,
                            ),
                            multiplicities=multiplicities,
                            oof_partial_manifest_sha256=sha256_file(
                                partial_path / "manifest.json"
                            ),
                        )
                    )
            oof_result = OOFMergeResult(final_oof, oof_hash, merged, len(artifact.identities))

        if not stopped and "residual_scales" in selection.phases:
            if oof_result is None:
                raise AssertionError("residual scales require an OOF merge result")
            fold_records = tuple(
                fold_results[index].record
                for index in range(config.crossfit.folds)
            )
            if any(record is None for record in fold_records):
                raise RuntimeError("residual scales require every fold checkpoint record")
            fold_set_hash = checkpoint_set_hash(
                [record for record in fold_records if record is not None]
            )
            scale_path = oof_output_dir / "residual-scales.json"
            tracking_step += 1
            if runtime.is_rank_zero:
                _validate_residual_scale_coverage(
                    oof_result.residual_accumulator, factory.artifacts.rows
                )
                if scale_path.exists():
                    scale_hash = _validate_existing_residual_scales(
                        scale_path,
                        oof_result.residual_accumulator,
                        oof_manifest_sha256=oof_result.manifest_sha256,
                        regression_checkpoint_set_sha256=fold_set_hash,
                    )
                else:
                    scale_hash = write_residual_scales(
                        scale_path,
                        oof_result.residual_accumulator,
                        oof_manifest_sha256=oof_result.manifest_sha256,
                        regression_checkpoint_set_sha256=fold_set_hash,
                    )
                _cleanup_oof_partials(
                    oof_output_dir=oof_output_dir,
                    folds=config.crossfit.folds,
                    world_size=runtime.world_size,
                )
            runtime.barrier()
            if not runtime.is_rank_zero:
                scale_hash = sha256_file(scale_path)
            _distributed_tracking_log(
                tracking,
                runtime,
                {"artifacts/residual_scales_sha256": scale_hash},
                step=tracking_step,
            )

        for phase, role in (
            ("deployment", "deployment"),
            ("direct_mean", "direct_mean"),
            ("direct_q50", "direct_q50"),
        ):
            if stopped or phase not in selection.phases or not selection.includes_role(role):
                continue
            if remaining is not None and remaining <= 0:
                stopped = True
                break
            result = train_role(
                role=role,
                fold_id=None,
                config=config,
                contract=contract,
                factory=factory,
                runtime=runtime,
                launch_identity=launch_identity,
                output_dir=output_dir,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                max_optimizer_steps=remaining,
                tracking=tracking,
                tracking_step=tracking_step,
            )
            role_results.append(result)
            tracking_step = result.metrics_step
            if remaining is not None:
                remaining -= result.optimizer_steps_run
                if remaining <= 0:
                    stopped = True
                    break

        partial = selection.partial or stopped
        role_records = [result.record for result in role_results if result.record is not None]
        manifest: dict[str, object] = {
            "format_version": (
                PARTIAL_MANIFEST_FORMAT if partial else STAGE2_MANIFEST_FORMAT
            ),
            "protocol_version": config.protocol_version,
            "partial": partial,
            "complete": not partial,
            "stopped_by_max_optimizer_steps": stopped,
            "selection": {
                "phases": list(selection.phases),
                "roles": list(selection.roles),
                "max_optimizer_steps": selection.max_optimizer_steps,
            },
            "config_sha256": config.sha256,
            "launch_identity": launch_identity.provenance(),
            "draw_manifest_sha256": factory.artifacts.artifact_hashes["draw_manifest"],
            "artifact_hashes": dict(factory.artifacts.artifact_hashes),
            "topology": {
                "world_size": runtime.world_size,
                "per_rank_microbatch_size": config.optimization.per_rank_microbatch_size,
                "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
                "precision": config.runtime.precision,
                "tf32": config.runtime.tf32,
                "cursor_global_example_index_semantics": (
                    "rank_local_plan_slots_consumed_including_padding"
                ),
            },
            "role_states": [_role_result_json(result) for result in role_results],
            "role_checkpoints": [_record_json(record) for record in role_records],
            "oof_manifest_sha256": (
                oof_result.manifest_sha256 if oof_result is not None else None
            ),
            "residual_scales_sha256": scale_hash,
            "fold_checkpoint_set_sha256": fold_set_hash,
            "wandb_mode": effective_mode,
            "wandb_run_id": run_id,
            "tracking_audit_path": (
                str((output_dir / "tracking" / "metrics.jsonl").resolve())
                if enabled
                else None
            ),
            "manual_release_required": True,
        }
        if not partial:
            expected_roles = {
                *(('fold', fold) for fold in range(config.crossfit.folds)),
                ("deployment", None),
                ("direct_mean", None),
                ("direct_q50", None),
            }
            actual_roles = {(record.role, record.fold_id) for record in role_records}
            if actual_roles != expected_roles:
                raise RuntimeError("complete publication lacks required regression checkpoints")
            if oof_result is None or scale_hash is None or fold_set_hash is None:
                raise RuntimeError("complete publication lacks OOF/residual artifacts")
        manifest_path = output_dir / (
            "partial-manifest.json" if partial else "stage2-manifest.json"
        )
        _verify_launch_identity_distributed(launch_identity, runtime)
        if runtime.is_rank_zero:
            manifest_hash = _atomic_json(
                manifest_path, manifest, replace=partial
            )
        else:
            manifest_hash = ""
        runtime.barrier()
        manifest_hash = sha256_file(manifest_path)
        hashes = runtime.all_gather_objects(manifest_hash)
        if len(set(hashes)) != 1:
            raise RuntimeError("ranks observed different Stage 2 manifest bytes")
        return Stage2RunResult(
            partial=partial,
            stopped_by_limit=stopped,
            manifest_path=manifest_path,
            manifest_sha256=manifest_hash,
            role_results=tuple(role_results),
            oof_manifest_sha256=(
                oof_result.manifest_sha256 if oof_result is not None else None
            ),
            residual_scales_sha256=scale_hash,
        )
    except BaseException:
        tracking.finish(exit_code=1)
        raise
    finally:
        # Successful finish occurs after the return expression is evaluated.
        if not getattr(tracking, "finished", False):
            tracking.finish(exit_code=0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launch-identity", type=Path, required=True)
    parser.add_argument("--launch-identity-sha256", required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--radar-cache-root", type=Path, required=True)
    parser.add_argument("--era5-cache-root", type=Path, required=True)
    parser.add_argument("--draw-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--bundle-metadata", type=Path)
    parser.add_argument(
        "--data-contract",
        type=Path,
        required=True,
    )
    parser.add_argument("--static-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oof-output-dir", type=Path, required=True)
    parser.add_argument("--oof-remote-credentials", type=Path, required=True)
    parser.add_argument("--oof-remote-ca-certificate", type=Path, required=True)
    parser.add_argument("--require-world-size", type=int, required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--target-widths", type=_csv_ints, required=True)
    parser.add_argument("--context-widths", type=_csv_ints, required=True)
    parser.add_argument("--era-latent-channels", type=int, required=True)
    parser.add_argument("--era-grid-size", type=int, required=True)
    parser.add_argument("--per-rank-microbatch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, required=True)
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--phases", type=_csv, default=_ALL_PHASES)
    parser.add_argument("--roles", type=_csv, default=())
    parser.add_argument(
        "--wandb-mode",
        choices=("config", "online", "offline", "disabled"),
        default="config",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("WANDB_RUN_ID") or os.environ.get("RUN_ID"),
    )
    parser.add_argument("--storage-reserve-gib", type=float, default=2.0)
    return parser.parse_args(argv)


def _environment_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(f"invalid boolean environment value: {value!r}")


def validate_launch_arguments(
    arguments: argparse.Namespace,
    config: Stage2Config,
    *,
    environ: Mapping[str, str] | None = None,
) -> RunSelection:
    """Bind every k8s flag/environment declaration to the hashed config."""

    environment = os.environ if environ is None else environ
    if (
        not isinstance(arguments.launch_identity_sha256, str)
        or len(arguments.launch_identity_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in arguments.launch_identity_sha256
        )
    ):
        raise ValueError("--launch-identity-sha256 must be a lowercase SHA-256")
    container_digest = arguments.container_image_digest
    if (
        not isinstance(container_digest, str)
        or not container_digest.startswith("sha256:")
        or len(container_digest) != 71
        or any(character not in "0123456789abcdef" for character in container_digest[7:])
    ):
        raise ValueError("--container-image-digest must be immutable sha256:<digest>")
    source_root = arguments.source_root.resolve()
    expected_entrypoint = source_root / "kcorrdiff/training/train_stage2.py"
    expected_identity_module = source_root / "kcorrdiff/training/launch_identity.py"
    if Path(__file__).resolve() != expected_entrypoint:
        raise ValueError("running Stage 2 entrypoint is outside --source-root")
    if Path(load_stage2_launch_identity.__code__.co_filename).resolve() != (
        expected_identity_module
    ):
        raise ValueError("running launch-identity verifier is outside --source-root")
    if arguments.config.resolve() != source_root / "configs/stage2-full-width.yaml":
        raise ValueError("Stage 2 config must be the source-tree production config")
    if arguments.data_contract.resolve() != source_root / "configs/data-contract-v1.1.3b.json":
        raise ValueError("Stage 2 data contract must be the source-tree contract")
    if arguments.require_world_size != 2:
        raise ValueError("Stage 2 requires exactly two torchrun ranks")
    if arguments.precision != "float32" or arguments.precision != config.runtime.precision:
        raise ValueError("Stage 2 precision must remain config-bound float32")
    if not arguments.disable_tf32 or config.runtime.tf32:
        raise ValueError("--disable-tf32 is required and config TF32 must be false")
    if not arguments.fail_on_fallback:
        raise ValueError("--fail-on-fallback is required")
    expected = {
        "target_widths": tuple(config.runtime.target_widths),
        "context_widths": tuple(config.runtime.context_widths),
        "era_latent_channels": config.runtime.era_latent_channels,
        "era_grid_size": config.runtime.era_grid_size,
        "per_rank_microbatch_size": config.optimization.per_rank_microbatch_size,
        "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
    }
    for name, value in expected.items():
        actual = getattr(arguments, name)
        if isinstance(value, tuple):
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"CLI {name} disagrees with hashed config")
    if arguments.num_workers < 0 or arguments.prefetch_factor <= 0:
        raise ValueError("worker/prefetch settings are invalid")
    loader_tuning = _mapping(
        config.raw.get("loader_tuning"), name="loader tuning config"
    )
    selected_workers = loader_tuning.get("selected_num_workers")
    selected_prefetch = loader_tuning.get("selected_prefetch_factor")
    if (
        isinstance(selected_workers, bool)
        or not isinstance(selected_workers, int)
        or isinstance(selected_prefetch, bool)
        or not isinstance(selected_prefetch, int)
    ):
        raise TypeError("hashed loader tuning worker/prefetch choices are invalid")
    if (
        arguments.num_workers != selected_workers
        or arguments.prefetch_factor != selected_prefetch
    ):
        raise ValueError("CLI worker/prefetch settings disagree with hashed config")
    selection_artifact = loader_tuning.get("selection_artifact")
    selection_sha256 = loader_tuning.get("selection_artifact_sha256")
    if (selection_artifact is None) != (selection_sha256 is None):
        raise ValueError("loader selection artifact path/hash must be declared together")
    if selection_artifact is not None:
        if not isinstance(selection_artifact, str) or not selection_artifact:
            raise TypeError("loader selection artifact path must be non-empty")
        if (
            not isinstance(selection_sha256, str)
            or len(selection_sha256) != 64
            or any(character not in "0123456789abcdef" for character in selection_sha256)
        ):
            raise ValueError("loader selection artifact SHA-256 is invalid")
        selection_path = Path(selection_artifact)
        if not selection_path.is_absolute():
            selection_path = source_root / selection_path
        if sha256_file(selection_path) != selection_sha256:
            raise ValueError("loader selection artifact SHA-256 mismatch")
        load_stage2_selection_contract(
            selection_path,
            expected_sha256=selection_sha256,
            config=config,
            world_size=arguments.require_world_size,
            num_workers=arguments.num_workers,
            prefetch_factor=arguments.prefetch_factor,
            # Production benchmark files live on the PVC; their immutable
            # digests are semantically validated here and are verified again
            # after the factory exposes the mounted artifact lineage.
            verify_benchmark_artifacts=False,
        )
    config.validate_topology(world_size=arguments.require_world_size)
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
    for name, expected_value in required_environment.items():
        raw = environment.get(name)
        if raw is None or raw.strip().lower() != expected_value:
            raise ValueError(f"strict environment contract mismatch: {name}")
    if arguments.max_optimizer_steps is not None and arguments.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be explicitly positive")
    if not arguments.run_id:
        raise ValueError("run ID is required via --run-id/WANDB_RUN_ID/RUN_ID")
    return RunSelection(
        phases=tuple(arguments.phases),
        roles=tuple(arguments.roles),
        max_optimizer_steps=arguments.max_optimizer_steps,
    )


def _factory_inputs_from_arguments(
    arguments: argparse.Namespace, config: Stage2Config
) -> FactoryInputs:
    contract = json.loads(arguments.data_contract.read_text(encoding="utf-8"))
    if not isinstance(contract, Mapping):
        raise TypeError("data contract root must be a mapping")
    if (
        contract.get("protocol_version") != config.protocol_version
        or contract.get("research_track") != config.research_track
    ):
        raise ValueError("data contract protocol/research track mismatch")
    geometry = _mapping(contract.get("geometry"), name="data-contract geometry")
    candidate = (
        arguments.candidate_manifest
        if arguments.candidate_manifest is not None
        else arguments.draw_manifest.with_name("candidate-manifest.json")
    )
    bundle = (
        arguments.bundle_metadata
        if arguments.bundle_metadata is not None
        else arguments.draw_manifest.with_name("bundle-metadata.json")
    )
    static_hash = geometry.get("target_static_sha256")
    if not isinstance(static_hash, str) or sha256_file(arguments.static_path) != static_hash:
        raise ValueError("target static hash disagrees with the data contract")
    target_hash = geometry.get("target_coordinates_sha256")
    context_hash = geometry.get("condition_coordinates_sha256")
    if not isinstance(target_hash, str) or not isinstance(context_hash, str):
        raise ValueError("data contract has no coordinate SHA-256 values")
    return FactoryInputs(
        radar_cache_root=arguments.radar_cache_root,
        era5_cache_root=arguments.era5_cache_root,
        target_static=arguments.static_path,
        candidate_manifest=candidate,
        draw_manifest=arguments.draw_manifest,
        bundle_metadata=bundle,
        expected_target_coordinates_sha256=target_hash,
        expected_context_coordinates_sha256=context_hash,
        verify_cache_hashes=bool(arguments.verify_cache_hashes),
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    config = load_stage2_config(arguments.config)
    selection = validate_launch_arguments(arguments, config)
    launch_identity = load_stage2_launch_identity(
        arguments.launch_identity,
        expected_sha256=arguments.launch_identity_sha256,
        source_root=arguments.source_root,
        container_image_digest=arguments.container_image_digest,
        runtime_report=arguments.runtime_report,
        data_contract=arguments.data_contract,
    )
    runtime = initialize_distributed_runtime(
        require_world_size=arguments.require_world_size
    )
    try:
        remote_store = AuthenticatedHTTPSShardStore(
            endpoint=config.crossfit.remote_spill.endpoint,
            remote_prefix=config.crossfit.remote_spill.remote_prefix,
            credential_file=arguments.oof_remote_credentials,
            ca_certificate=arguments.oof_remote_ca_certificate,
            timeout_seconds=config.crossfit.remote_spill.timeout_seconds,
        )
        identities = runtime.all_gather_objects(launch_identity.provenance())
        if len({json.dumps(value, sort_keys=True) for value in identities}) != 1:
            raise RuntimeError("ranks observed different immutable launch identities")
        inputs = _factory_inputs_from_arguments(arguments, config)
        factory = build_data_factory(config, inputs)
        loader_tuning = _mapping(
            config.raw.get("loader_tuning"), name="loader tuning config"
        )
        selection_path = Path(str(loader_tuning["selection_artifact"]))
        if not selection_path.is_absolute():
            selection_path = arguments.source_root / selection_path
        loader_selection = load_stage2_selection_contract(
            selection_path,
            expected_sha256=str(loader_tuning["selection_artifact_sha256"]),
            config=config,
            world_size=runtime.world_size,
            num_workers=arguments.num_workers,
            prefetch_factor=arguments.prefetch_factor,
            verify_benchmark_artifacts=False,
        )
        loader_selection.validate_factory_lineage(factory.artifacts.artifact_hashes)
        loader_selection.verify_benchmark_artifacts()
        contract = regression_system_config_from_stage2(config)
        # Parameter count is architecture-derived, not a stale disk constant.
        counting_model = RegressionSystem(contract.system)
        # A direct arm owns the identical full hurdle backbone plus a separate
        # 1x1 physical head.  Preflight the largest retained role without
        # allocating a second production-width model solely for accounting.
        parameter_count = counting_model.parameter_count + (
            contract.system.target_widths[0] + 1
        )
        del counting_model
        unique = unique_oof_rows(factory.artifacts.rows)
        oof_bytes = oof_dense_byte_budget(unique)
        if oof_bytes > config.crossfit.maximum_oof_bytes:
            raise OSError("OOF logical FP32 fields exceed the configured byte budget")
        planning_identities, planning_partitions = _compressed_oof_preflight_layout(
            factory.artifacts.rows,
            folds=config.crossfit.folds,
            world_size=runtime.world_size,
        )
        oof_plan = compressed_oof_storage_plan(
            planning_identities,
            planning_partitions,
            maximum_compressed_bytes=(
                config.crossfit.maximum_oof_compressed_bytes
            ),
            concurrent_writers=runtime.world_size,
            shard_items=config.crossfit.oof_shard_items,
        )
        preflight_storage(
            output_dir=arguments.output_dir,
            oof_output_dir=arguments.oof_output_dir,
            model_parameter_count=parameter_count,
            selected_checkpoint_count=_selected_training_role_count(
                selection, folds=config.crossfit.folds
            ),
            oof_dense_bytes=oof_bytes,
            include_oof="oof" in selection.phases,
            # Remote spill keeps only a hot subset on the PVC.  The logical
            # 140-GiB artifact cap remains enforced by the writer/manifest,
            # while local preflight reserves only bounded active temporaries.
            oof_compressed_cap_bytes=0,
            oof_staging_bytes=(
                oof_plan.concurrent_writers
                * (
                    oof_plan.maximum_active_shard_bytes
                    + oof_plan.maximum_archive_temporary_bytes
                )
            ),
            oof_metadata_bytes=oof_plan.metadata_budget_bytes,
            reserve_bytes=max(
                int(arguments.storage_reserve_gib * 1024**3),
                config.crossfit.remote_spill.local_reserve_bytes,
            ),
        )
        result = run_stage2(
            config=config,
            factory=factory,
            runtime=runtime,
            launch_identity=launch_identity,
            output_dir=arguments.output_dir,
            oof_output_dir=arguments.oof_output_dir,
            selection=selection,
            num_workers=arguments.num_workers,
            prefetch_factor=arguments.prefetch_factor,
            wandb_mode=arguments.wandb_mode,
            run_id=arguments.run_id,
            remote_store=remote_store,
        )
        if runtime.is_rank_zero:
            print(
                json.dumps(
                    {
                        "partial": result.partial,
                        "stopped_by_limit": result.stopped_by_limit,
                        "manifest": str(result.manifest_path),
                        "manifest_sha256": result.manifest_sha256,
                    },
                    sort_keys=True,
                )
            )
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DistributedRuntime",
    "ModelContract",
    "OptimizerLoopResult",
    "RunSelection",
    "Stage2RunResult",
    "StoragePreflight",
    "TARGET_BUILDER_VERSION",
    "initialize_distributed_runtime",
    "load_complete_stage2_manifest",
    "main",
    "merge_oof_partials",
    "preflight_storage",
    "reduce_gradients_sum",
    "regression_system_config_from_stage2",
    "run_optimizer_loop",
    "run_stage2",
    "train_role",
    "validate_launch_arguments",
]
