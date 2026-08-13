"""Atomic, provenance-bound training checkpoints with exact RNG resume state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = "kcorrdiff.training-checkpoint.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingCursor:
    epoch: int
    global_example_index: int
    optimizer_step: int
    microbatches_since_step: int

    def validate(self) -> None:
        values = (
            self.epoch,
            self.global_example_index,
            self.optimizer_step,
            self.microbatches_since_step,
        )
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("training cursor values must be non-negative integers")


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    protocol_version: str
    config_sha256: str
    draw_manifest_sha256: str
    launch_identity_sha256: str
    source_tree_sha256: str
    container_image_sha256: str
    runtime_report_sha256: str
    data_contract_sha256: str
    role: str
    fold_id: int | None
    world_size: int
    per_rank_microbatch_size: int
    gradient_accumulation_steps: int

    def validate(self) -> None:
        if self.protocol_version != "v1.1.3b":
            raise ValueError("checkpoint protocol mismatch")
        for name, value in (
            ("config_sha256", self.config_sha256),
            ("draw_manifest_sha256", self.draw_manifest_sha256),
            ("launch_identity_sha256", self.launch_identity_sha256),
            ("source_tree_sha256", self.source_tree_sha256),
            ("container_image_sha256", self.container_image_sha256),
            ("runtime_report_sha256", self.runtime_report_sha256),
            ("data_contract_sha256", self.data_contract_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.role not in {"fold", "deployment", "direct_mean", "direct_q50"}:
            raise ValueError("unsupported checkpoint role")
        if (self.role == "fold") != (self.fold_id is not None):
            raise ValueError("fold role/fold_id mismatch")
        if self.fold_id is not None and (
            isinstance(self.fold_id, bool) or self.fold_id < 0
        ):
            raise ValueError("fold_id must be non-negative")
        for name, value in (
            ("world_size", self.world_size),
            ("per_rank_microbatch_size", self.per_rank_microbatch_size),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = []
    return state


def restore_rng_state(state: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("checkpoint RNG state schema mismatch")
    random.setstate(tuple(state["python"]))  # type: ignore[arg-type]
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise TypeError("NumPy RNG state must be a mapping")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    cpu = state["torch_cpu"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8:
        raise TypeError("Torch CPU RNG state must be a uint8 tensor")
    torch.set_rng_state(cpu.cpu())
    cuda = state["torch_cuda"]
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(list(cuda))  # type: ignore[arg-type]


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    cursor: TrainingCursor,
    provenance: CheckpointProvenance,
    extra: Mapping[str, object] | None = None,
) -> str:
    """Atomically publish a full float32 training state and return its hash."""

    cursor.validate()
    provenance.validate()
    floating = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.is_floating_point() and value.dtype is not torch.float32
    ]
    if floating:
        raise TypeError(f"checkpoint contains non-float32 model state: {floating[:8]}")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    state = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "cursor": asdict(cursor),
        "provenance": asdict(provenance),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    try:
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: object | None,
    expected_provenance: CheckpointProvenance,
    restore_rng: bool = True,
) -> tuple[TrainingCursor, Mapping[str, object]]:
    """Load only a checkpoint whose complete immutable provenance matches."""

    expected_provenance.validate()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch.
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping) or state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint")
    provenance = CheckpointProvenance(**state["provenance"])
    provenance.validate()
    if provenance != expected_provenance:
        raise ValueError("checkpoint provenance/topology mismatch")
    cursor = TrainingCursor(**state["cursor"])
    cursor.validate()
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler_state = state["scheduler"]
        if scheduler_state is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(scheduler_state)
    if restore_rng:
        restore_rng_state(state["rng"])
    extra = state.get("extra", {})
    if not isinstance(extra, Mapping):
        raise TypeError("checkpoint extra state must be a mapping")
    return cursor, extra


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointProvenance",
    "TrainingCursor",
    "capture_rng_state",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_training_checkpoint",
]
