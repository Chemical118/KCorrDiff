"""Strict import boundary for independently trained Stage 2 B12 folds.

The importer deliberately keeps the producer checkpoint provenance intact.
It validates the one-GPU/B12 training state and draw plan, then exposes a
model-only load operation for OOF inference.  It is not a training-resume
conversion and never rewrites a checkpoint as a two-GPU/B8 artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import stat

import torch
from torch import nn

from kcorrdiff.data.sampling import DrawRow
from kcorrdiff.training.checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointProvenance,
    TrainingCursor,
    load_training_checkpoint,
)
from kcorrdiff.training.crossfit import CheckpointRecord
from kcorrdiff.training.plan import build_distributed_draw_plan


PARTIAL_MANIFEST_FORMAT = "kcorrdiff.stage2-partial-training.v1"
FOLD_SET_FORMAT = "kcorrdiff.stage2-single-node-fold-set.v1"
VERIFIED_IMPORT_FORMAT = "kcorrdiff.stage2-verified-fold-set-import.v1"
VERIFICATION_RECEIPT_FORMAT = "kcorrdiff.stage2-fold-verification.v1"
PROTOCOL_VERSION = "v1.1.3b"
PRODUCER_NODE_NAME = "porsche"
PRODUCER_WORLD_SIZE = 1
PRODUCER_MICROBATCH_SIZE = 12
PRODUCER_GRADIENT_ACCUMULATION_STEPS = 1
PRODUCER_GLOBAL_BATCH_SIZE = 12
FOLD_COUNT = 3
CURSOR_SEMANTICS = "rank_local_plan_slots_consumed_including_padding"

_CHECKPOINT_KEYS = {
    "format_version",
    "cursor",
    "provenance",
    "model",
    "optimizer",
    "scheduler",
    "rng",
    "extra",
}
_EXTRA_KEYS = {
    "complete",
    "plan_semantic_sha256",
    "plan_optimizer_steps",
    "training_blocks",
    "rank_rng_states",
}
_PARTIAL_KEYS = {
    "format_version",
    "protocol_version",
    "partial",
    "complete",
    "stopped_by_max_optimizer_steps",
    "selection",
    "config_sha256",
    "launch_identity",
    "draw_manifest_sha256",
    "artifact_hashes",
    "topology",
    "role_states",
    "role_checkpoints",
    "oof_manifest_sha256",
    "residual_scales_sha256",
    "fold_checkpoint_set_sha256",
    "wandb_mode",
    "wandb_run_id",
    "tracking_audit_path",
    "manual_release_required",
}
_PARTIAL_OPTIONAL_KEYS = {"imported_fold_set"}
_SELECTION_KEYS = {
    "phases",
    "roles",
    "max_optimizer_steps",
    "bounded_training_year",
    "expected_training_items",
    "single_node_fold_worker",
    "node_name",
}
_TOPOLOGY_KEYS = {
    "world_size",
    "per_rank_microbatch_size",
    "gradient_accumulation_steps",
    "global_effective_batch_size",
    "precision",
    "tf32",
    "single_node_fold_policy_sha256",
    "cursor_global_example_index_semantics",
}
_LAUNCH_KEYS = {
    "artifact_sha256",
    "source_tree_sha256",
    "container_image_sha256",
    "runtime_report_sha256",
    "data_contract_sha256",
}
_ROLE_STATE_KEYS = {
    "role",
    "fold_id",
    "complete",
    "checkpoint_path",
    "checkpoint_sha256",
    "cursor",
    "optimizer_steps_run_this_invocation",
}
_RECORD_KEYS = {
    "fold_id",
    "role",
    "path",
    "sha256",
    "training_blocks_sha256",
    "config_sha256",
    "draw_manifest_sha256",
    "global_step",
}
_CURSOR_KEYS = {
    "epoch",
    "global_example_index",
    "optimizer_step",
    "microbatches_since_step",
}
_FOLD_SET_KEYS = {
    "format_version",
    "folds",
    "node_name",
    "world_size_per_fold",
    "per_rank_microbatch_size",
    "gradient_accumulation_steps",
    "config_sha256",
    "policy_sha256",
    "records",
}
_FOLD_SET_OPTIONAL_KEYS = {"verification_format", "lineage"}
_FOLD_SET_RECORD_KEYS = {
    "fold_id",
    "checkpoint_path",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "partial_manifest_path",
    "partial_manifest_sha256",
}
_FOLD_SET_RECORD_OPTIONAL_KEYS = {"verification"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha256(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, object], *, required: set[str], name: str
) -> None:
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise ValueError(f"{name} schema mismatch: missing={missing}, extra={extra}")


def _keys_with_optional(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _load_json(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {path}") from error
    return _mapping(raw, name=name)


def _require_regular_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink():
        raise ValueError(f"{name} cannot be a symbolic link: {path}")
    try:
        mode = absolute.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(absolute) from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} must be a regular file: {absolute}")
    return absolute.resolve()


def _load_checkpoint_state(path: Path) -> Mapping[str, object]:
    try:
        try:
            raw = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:  # pragma: no cover - older Torch compatibility.
            raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"fold checkpoint is not a safe Torch checkpoint: {path}") from error
    return _mapping(raw, name="fold checkpoint")


def _validated_hash_mapping(value: object, *, name: str) -> dict[str, str]:
    raw = _mapping(value, name=name)
    if not raw:
        raise ValueError(f"{name} cannot be empty")
    result: dict[str, str] = {}
    for key, digest in raw.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = _require_sha256(digest, name=f"{name}.{key}")
    return result


def _validated_launch_identity(value: object) -> dict[str, str]:
    raw = _mapping(value, name="partial manifest launch_identity")
    _exact_keys(raw, required=_LAUNCH_KEYS, name="partial manifest launch_identity")
    return {
        key: _require_sha256(raw[key], name=f"launch_identity.{key}")
        for key in sorted(_LAUNCH_KEYS)
    }


def _validated_cursor(value: object, *, name: str) -> TrainingCursor:
    raw = _mapping(value, name=name)
    _exact_keys(raw, required=_CURSOR_KEYS, name=name)
    try:
        cursor = TrainingCursor(**raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    cursor.validate()
    return cursor


def _validate_rng_state(value: object, *, name: str) -> None:
    raw = _mapping(value, name=name)
    _exact_keys(
        raw,
        required={"python", "numpy", "torch_cpu", "torch_cuda"},
        name=name,
    )
    cpu = raw["torch_cpu"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise TypeError(f"{name}.torch_cpu must be a one-dimensional uint8 tensor")
    if not isinstance(raw["torch_cuda"], list):
        raise TypeError(f"{name}.torch_cuda must be a list")
    numpy_state = _mapping(raw["numpy"], name=f"{name}.numpy")
    _exact_keys(
        numpy_state,
        required={"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"},
        name=f"{name}.numpy",
    )


def _validate_model_state(value: object) -> None:
    state = _mapping(value, name="checkpoint model state")
    if not state:
        raise ValueError("checkpoint model state cannot be empty")
    for name, tensor in state.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint model state keys must be non-empty strings")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint model value is not a tensor: {name}")
        if tensor.is_floating_point():
            if tensor.dtype is not torch.float32:
                raise TypeError(f"checkpoint model tensor is not float32: {name}")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"checkpoint model tensor is non-finite: {name}")


@dataclass(frozen=True, slots=True)
class FoldProducerLineage:
    protocol_version: str
    config_sha256: str
    draw_manifest_sha256: str
    launch_identity: Mapping[str, str]
    artifact_hashes: Mapping[str, str]
    node_name: str
    world_size: int
    per_rank_microbatch_size: int
    gradient_accumulation_steps: int
    global_effective_batch_size: int
    policy_sha256: str

    def audit_json(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "config_sha256": self.config_sha256,
            "draw_manifest_sha256": self.draw_manifest_sha256,
            "launch_identity": dict(self.launch_identity),
            "artifact_hashes": dict(self.artifact_hashes),
            "node_name": self.node_name,
            "world_size": self.world_size,
            "per_rank_microbatch_size": self.per_rank_microbatch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_effective_batch_size": self.global_effective_batch_size,
            "policy_sha256": self.policy_sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedFoldCheckpoint:
    fold_id: int
    checkpoint_path: Path
    checkpoint_bytes: int
    checkpoint_sha256: str
    partial_manifest_path: Path
    partial_manifest_sha256: str
    provenance: CheckpointProvenance
    cursor: TrainingCursor
    plan_semantic_sha256: str
    plan_optimizer_steps: int
    training_blocks: tuple[str, ...]
    training_blocks_sha256: str
    lineage: FoldProducerLineage
    record: CheckpointRecord

    def receipt_json(self) -> dict[str, object]:
        return {
            "format_version": VERIFICATION_RECEIPT_FORMAT,
            "fold_id": self.fold_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "partial_manifest_sha256": self.partial_manifest_sha256,
            "provenance": asdict(self.provenance),
            "cursor": asdict(self.cursor),
            "plan_semantic_sha256": self.plan_semantic_sha256,
            "plan_optimizer_steps": self.plan_optimizer_steps,
            "training_blocks_sha256": self.training_blocks_sha256,
            "lineage": self.lineage.audit_json(),
        }


@dataclass(frozen=True, slots=True)
class VerifiedStage2FoldSet:
    manifest_path: Path
    manifest_sha256: str
    lineage: FoldProducerLineage
    folds: tuple[VerifiedFoldCheckpoint, ...]

    def by_fold(self) -> dict[int, VerifiedFoldCheckpoint]:
        return {fold.fold_id: fold for fold in self.folds}

    def audit_json(self) -> dict[str, object]:
        return {
            "format_version": VERIFIED_IMPORT_FORMAT,
            "manifest_path": str(self.manifest_path.resolve()),
            "manifest_sha256": self.manifest_sha256,
            "producer": self.lineage.audit_json(),
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "checkpoint_path": str(fold.checkpoint_path.resolve()),
                    "checkpoint_bytes": fold.checkpoint_bytes,
                    "checkpoint_sha256": fold.checkpoint_sha256,
                    "partial_manifest_path": str(
                        fold.partial_manifest_path.resolve()
                    ),
                    "partial_manifest_sha256": fold.partial_manifest_sha256,
                    "cursor": asdict(fold.cursor),
                    "plan_semantic_sha256": fold.plan_semantic_sha256,
                    "plan_optimizer_steps": fold.plan_optimizer_steps,
                    "training_blocks_sha256": fold.training_blocks_sha256,
                }
                for fold in self.folds
            ],
        }


def verify_single_node_fold_artifacts(
    *,
    checkpoint_path: Path,
    partial_manifest_path: Path,
    fold_id: int,
    policy_sha256: str,
    rows: Sequence[DrawRow] | None = None,
    expected_config_sha256: str | None = None,
    expected_draw_manifest_sha256: str | None = None,
    expected_artifact_hashes: Mapping[str, str] | None = None,
) -> VerifiedFoldCheckpoint:
    """Deep-verify one completed one-GPU/B12 fold and its partial manifest."""

    if isinstance(fold_id, bool) or not 0 <= fold_id < FOLD_COUNT:
        raise ValueError("fold_id is outside the three-fold contract")
    policy_sha256 = _require_sha256(policy_sha256, name="fold policy SHA-256")
    checkpoint = _require_regular_file(checkpoint_path, name="fold checkpoint")
    partial = _require_regular_file(
        partial_manifest_path, name="fold partial manifest"
    )
    checkpoint_hash_before = _sha256(checkpoint)
    partial_hash = _sha256(partial)
    manifest = _load_json(partial, name="fold partial manifest")
    _keys_with_optional(
        manifest,
        required=_PARTIAL_KEYS,
        optional=_PARTIAL_OPTIONAL_KEYS,
        name="fold partial manifest",
    )
    if (
        manifest["format_version"] != PARTIAL_MANIFEST_FORMAT
        or manifest["protocol_version"] != PROTOCOL_VERSION
        or manifest["partial"] is not True
        or manifest["complete"] is not False
        or manifest["stopped_by_max_optimizer_steps"] is not False
        or manifest["manual_release_required"] is not True
    ):
        raise ValueError("fold partial manifest completion/protocol contract mismatch")
    if any(
        manifest[name] is not None
        for name in (
            "oof_manifest_sha256",
            "residual_scales_sha256",
            "fold_checkpoint_set_sha256",
        )
    ):
        raise ValueError("fold-only partial manifest contains downstream artifacts")
    if manifest.get("imported_fold_set") is not None:
        raise ValueError("fold worker partial manifest cannot import another fold set")
    if (
        manifest["wandb_mode"] not in {"online", "offline", "disabled"}
        or not isinstance(manifest["wandb_run_id"], str)
        or not manifest["wandb_run_id"]
        or (
            manifest["tracking_audit_path"] is not None
            and (
                not isinstance(manifest["tracking_audit_path"], str)
                or not manifest["tracking_audit_path"]
            )
        )
    ):
        raise ValueError("fold partial manifest tracking audit fields are invalid")

    selection = _mapping(manifest["selection"], name="fold selection")
    _exact_keys(selection, required=_SELECTION_KEYS, name="fold selection")
    if dict(selection) != {
        "phases": ["crossfit"],
        "roles": [f"fold:{fold_id}"],
        "max_optimizer_steps": None,
        "bounded_training_year": None,
        "expected_training_items": None,
        "single_node_fold_worker": True,
        "node_name": PRODUCER_NODE_NAME,
    }:
        raise ValueError("fold partial manifest selection is not porsche crossfit-only")

    topology = _mapping(manifest["topology"], name="fold topology")
    _exact_keys(topology, required=_TOPOLOGY_KEYS, name="fold topology")
    for name in (
        "world_size",
        "per_rank_microbatch_size",
        "gradient_accumulation_steps",
        "global_effective_batch_size",
    ):
        _require_int(topology[name], name=f"fold topology {name}", minimum=1)
    if dict(topology) != {
        "world_size": PRODUCER_WORLD_SIZE,
        "per_rank_microbatch_size": PRODUCER_MICROBATCH_SIZE,
        "gradient_accumulation_steps": PRODUCER_GRADIENT_ACCUMULATION_STEPS,
        "global_effective_batch_size": PRODUCER_GLOBAL_BATCH_SIZE,
        "precision": "float32",
        "tf32": False,
        "single_node_fold_policy_sha256": policy_sha256,
        "cursor_global_example_index_semantics": CURSOR_SEMANTICS,
    }:
        raise ValueError("fold partial manifest violates the B12 producer topology")

    config_sha256 = _require_sha256(
        manifest["config_sha256"], name="partial config_sha256"
    )
    draw_sha256 = _require_sha256(
        manifest["draw_manifest_sha256"], name="partial draw_manifest_sha256"
    )
    if expected_config_sha256 is not None and config_sha256 != _require_sha256(
        expected_config_sha256, name="expected config_sha256"
    ):
        raise ValueError("fold config SHA-256 disagrees with the requested config")
    if expected_draw_manifest_sha256 is not None and draw_sha256 != _require_sha256(
        expected_draw_manifest_sha256, name="expected draw_manifest_sha256"
    ):
        raise ValueError("fold draw manifest SHA-256 disagrees with the requested draw")
    launch = _validated_launch_identity(manifest["launch_identity"])
    artifact_hashes = _validated_hash_mapping(
        manifest["artifact_hashes"], name="partial artifact_hashes"
    )
    if artifact_hashes.get("draw_manifest") != draw_sha256:
        raise ValueError("partial artifact hashes do not bind the draw manifest")
    if expected_artifact_hashes is not None:
        expected_artifacts = _validated_hash_mapping(
            expected_artifact_hashes, name="expected artifact_hashes"
        )
        if artifact_hashes != expected_artifacts:
            raise ValueError("fold data artifact lineage disagrees with OOF inputs")

    state = _load_checkpoint_state(checkpoint)
    _exact_keys(state, required=_CHECKPOINT_KEYS, name="fold checkpoint")
    if state["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported fold checkpoint format")
    provenance_raw = _mapping(state["provenance"], name="checkpoint provenance")
    try:
        provenance = CheckpointProvenance(**provenance_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint provenance schema is invalid") from error
    provenance.validate()
    expected_provenance = CheckpointProvenance(
        protocol_version=PROTOCOL_VERSION,
        config_sha256=config_sha256,
        draw_manifest_sha256=draw_sha256,
        launch_identity_sha256=launch["artifact_sha256"],
        source_tree_sha256=launch["source_tree_sha256"],
        container_image_sha256=launch["container_image_sha256"],
        runtime_report_sha256=launch["runtime_report_sha256"],
        data_contract_sha256=launch["data_contract_sha256"],
        role="fold",
        fold_id=fold_id,
        world_size=PRODUCER_WORLD_SIZE,
        per_rank_microbatch_size=PRODUCER_MICROBATCH_SIZE,
        gradient_accumulation_steps=PRODUCER_GRADIENT_ACCUMULATION_STEPS,
    )
    if provenance != expected_provenance:
        raise ValueError("checkpoint provenance disagrees with its producer manifest")

    cursor = _validated_cursor(state["cursor"], name="checkpoint cursor")
    if cursor.epoch != 0 or cursor.microbatches_since_step != 0:
        raise ValueError("completed fold checkpoint cursor is not at a step boundary")
    extra = _mapping(state["extra"], name="checkpoint extra")
    _exact_keys(extra, required=_EXTRA_KEYS, name="checkpoint extra")
    if extra["complete"] is not True:
        raise ValueError("fold checkpoint is not complete")
    plan_sha256 = _require_sha256(
        extra["plan_semantic_sha256"], name="checkpoint plan_semantic_sha256"
    )
    plan_steps = _require_int(
        extra["plan_optimizer_steps"],
        name="checkpoint plan_optimizer_steps",
        minimum=1,
    )
    if cursor.optimizer_step != plan_steps:
        raise ValueError("checkpoint cursor does not finish the declared draw plan")
    expected_slots = (
        plan_steps
        * PRODUCER_MICROBATCH_SIZE
        * PRODUCER_GRADIENT_ACCUMULATION_STEPS
    )
    if cursor.global_example_index != expected_slots:
        raise ValueError("checkpoint cursor has the wrong B12 rank-local slot count")
    raw_blocks = extra["training_blocks"]
    if not isinstance(raw_blocks, list) or not raw_blocks or any(
        not isinstance(block, str) or not block for block in raw_blocks
    ):
        raise ValueError("checkpoint training_blocks must be a non-empty string list")
    training_blocks = tuple(raw_blocks)
    if training_blocks != tuple(sorted(set(training_blocks))):
        raise ValueError("checkpoint training_blocks are not canonical and unique")
    training_blocks_sha256 = _semantic_sha256(list(training_blocks))
    rank_rng_states = extra["rank_rng_states"]
    if not isinstance(rank_rng_states, list) or len(rank_rng_states) != 1:
        raise ValueError("checkpoint has no exact one-rank RNG receipt")
    _validate_rng_state(rank_rng_states[0], name="checkpoint rank RNG state")
    _validate_rng_state(state["rng"], name="checkpoint RNG state")
    _validate_model_state(state["model"])
    optimizer = _mapping(state["optimizer"], name="checkpoint optimizer state")
    if set(optimizer) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer state schema mismatch")
    if not isinstance(optimizer["state"], Mapping) or not isinstance(
        optimizer["param_groups"], list
    ):
        raise TypeError("checkpoint optimizer state is malformed")
    if not isinstance(state["scheduler"], Mapping):
        raise TypeError("completed fold checkpoint requires scheduler state")

    if rows is not None:
        plan = build_distributed_draw_plan(
            rows,
            role="fold",
            fold_id=fold_id,
            world_size=PRODUCER_WORLD_SIZE,
            per_rank_microbatch_size=PRODUCER_MICROBATCH_SIZE,
            gradient_accumulation_steps=(
                PRODUCER_GRADIENT_ACCUMULATION_STEPS
            ),
        )
        expected_blocks = tuple(
            sorted({rows[index].block_id for index in plan.source_row_indices})
        )
        if (
            plan.semantic_sha256 != plan_sha256
            or plan.optimizer_steps != plan_steps
            or expected_blocks != training_blocks
        ):
            raise ValueError("checkpoint does not implement the reconstructed B12 draw plan")

    role_states = manifest["role_states"]
    if not isinstance(role_states, list) or len(role_states) != 1:
        raise ValueError("fold partial manifest must contain exactly one role state")
    role_state = _mapping(role_states[0], name="fold role state")
    _exact_keys(role_state, required=_ROLE_STATE_KEYS, name="fold role state")
    role_cursor = _validated_cursor(role_state["cursor"], name="fold role cursor")
    invocation_steps = _require_int(
        role_state["optimizer_steps_run_this_invocation"],
        name="role optimizer steps this invocation",
    )
    if (
        role_state["role"] != "fold"
        or isinstance(role_state["fold_id"], bool)
        or not isinstance(role_state["fold_id"], int)
        or role_state["fold_id"] != fold_id
        or role_state["complete"] is not True
        or role_state["checkpoint_sha256"] != checkpoint_hash_before
        or role_cursor != cursor
        or invocation_steps > cursor.optimizer_step
        or not isinstance(role_state["checkpoint_path"], str)
        or not role_state["checkpoint_path"]
    ):
        raise ValueError("fold role state disagrees with the completed checkpoint")

    role_records = manifest["role_checkpoints"]
    if not isinstance(role_records, list) or len(role_records) != 1:
        raise ValueError("fold partial manifest must contain exactly one checkpoint record")
    role_record = _mapping(role_records[0], name="fold checkpoint record")
    _exact_keys(role_record, required=_RECORD_KEYS, name="fold checkpoint record")
    record_global_step = _require_int(
        role_record["global_step"],
        name="fold checkpoint record global_step",
        minimum=1,
    )
    if (
        isinstance(role_record["fold_id"], bool)
        or not isinstance(role_record["fold_id"], int)
        or role_record["fold_id"] != fold_id
        or role_record["role"] != "fold"
        or role_record["sha256"] != checkpoint_hash_before
        or role_record["training_blocks_sha256"] != training_blocks_sha256
        or role_record["config_sha256"] != config_sha256
        or role_record["draw_manifest_sha256"] != draw_sha256
        or record_global_step != cursor.optimizer_step
        or not isinstance(role_record["path"], str)
        or not role_record["path"]
    ):
        raise ValueError("fold checkpoint record disagrees with checkpoint contents")

    if _sha256(checkpoint) != checkpoint_hash_before:
        raise RuntimeError("fold checkpoint changed during verification")
    if _sha256(partial) != partial_hash:
        raise RuntimeError("fold partial manifest changed during verification")

    lineage = FoldProducerLineage(
        protocol_version=PROTOCOL_VERSION,
        config_sha256=config_sha256,
        draw_manifest_sha256=draw_sha256,
        launch_identity=launch,
        artifact_hashes=artifact_hashes,
        node_name=PRODUCER_NODE_NAME,
        world_size=PRODUCER_WORLD_SIZE,
        per_rank_microbatch_size=PRODUCER_MICROBATCH_SIZE,
        gradient_accumulation_steps=PRODUCER_GRADIENT_ACCUMULATION_STEPS,
        global_effective_batch_size=PRODUCER_GLOBAL_BATCH_SIZE,
        policy_sha256=policy_sha256,
    )
    record = CheckpointRecord(
        fold_id=fold_id,
        role="fold",
        path=str(checkpoint),
        sha256=checkpoint_hash_before,
        training_blocks_sha256=training_blocks_sha256,
        config_sha256=config_sha256,
        draw_manifest_sha256=draw_sha256,
        global_step=cursor.optimizer_step,
    )
    return VerifiedFoldCheckpoint(
        fold_id=fold_id,
        checkpoint_path=checkpoint,
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=checkpoint_hash_before,
        partial_manifest_path=partial,
        partial_manifest_sha256=partial_hash,
        provenance=provenance,
        cursor=cursor,
        plan_semantic_sha256=plan_sha256,
        plan_optimizer_steps=plan_steps,
        training_blocks=training_blocks,
        training_blocks_sha256=training_blocks_sha256,
        lineage=lineage,
        record=record,
    )


def load_verified_fold_model(
    verified: VerifiedFoldCheckpoint, model: nn.Module
) -> None:
    """Strict-load model weights without restoring optimizer, scheduler, or RNG."""

    before = _sha256(verified.checkpoint_path)
    if before != verified.checkpoint_sha256:
        raise ValueError("verified fold checkpoint bytes changed before model load")
    cursor, extra = load_training_checkpoint(
        verified.checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        expected_provenance=verified.provenance,
        restore_rng=False,
    )
    if (
        cursor != verified.cursor
        or extra.get("complete") is not True
        or extra.get("plan_semantic_sha256") != verified.plan_semantic_sha256
        or extra.get("plan_optimizer_steps") != verified.plan_optimizer_steps
        or tuple(extra.get("training_blocks", ())) != verified.training_blocks
    ):
        raise ValueError("verified fold checkpoint metadata changed during model load")
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        if value.is_floating_point():
            if value.dtype is not torch.float32:
                raise TypeError(f"loaded fold model tensor is not float32: {name}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"loaded fold model tensor is non-finite: {name}")
    if _sha256(verified.checkpoint_path) != before:
        raise RuntimeError("fold checkpoint changed during model-only load")


def verify_fold_set_model_compatibility(
    fold_set: VerifiedStage2FoldSet,
    *,
    model_factory: Callable[[int], nn.Module],
) -> None:
    """Prove every imported state dict strictly fits the current model code."""

    for fold in fold_set.folds:
        model = model_factory(fold.fold_id)
        try:
            load_verified_fold_model(fold, model)
        finally:
            del model


def _resolved_member(root: Path, relative: object, *, expected: str) -> Path:
    if not isinstance(relative, str) or relative != expected:
        raise ValueError(f"fold-set member path must be exactly {expected!r}")
    candidate = root / relative
    current = root
    for part in Path(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("fold-set member directory cannot be a symbolic link")
    _require_regular_file(candidate, name="fold-set member")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("fold-set member escapes the immutable set root")
    return resolved


def verify_stage2_fold_set(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_policy_sha256: str,
    rows: Sequence[DrawRow],
    expected_config_sha256: str,
    expected_artifact_hashes: Mapping[str, str],
    expected_source_tree_sha256: str,
) -> VerifiedStage2FoldSet:
    """Verify and import a complete assembled three-fold B12 checkpoint set."""

    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, name="expected fold-set manifest SHA-256"
    )
    expected_policy_sha256 = _require_sha256(
        expected_policy_sha256, name="expected fold-set policy SHA-256"
    )
    expected_source_tree_sha256 = _require_sha256(
        expected_source_tree_sha256,
        name="expected consumer source-tree SHA-256",
    )
    manifest = _require_regular_file(manifest_path, name="fold-set manifest")
    actual_manifest_sha256 = _sha256(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("fold-set manifest SHA-256 mismatch")
    sidecar = _require_regular_file(
        manifest.with_name("fold-set-manifest.sha256"),
        name="fold-set manifest SHA-256 sidecar",
    )
    try:
        sidecar_payload = sidecar.read_text(encoding="ascii")
    except UnicodeDecodeError as error:
        raise ValueError("fold-set SHA-256 sidecar is not ASCII") from error
    if sidecar_payload != actual_manifest_sha256 + "\n":
        raise ValueError("fold-set SHA-256 sidecar disagrees with the manifest")
    raw = _load_json(manifest, name="fold-set manifest")
    _keys_with_optional(
        raw,
        required=_FOLD_SET_KEYS,
        optional=_FOLD_SET_OPTIONAL_KEYS,
        name="fold-set manifest",
    )
    for name in (
        "folds",
        "world_size_per_fold",
        "per_rank_microbatch_size",
        "gradient_accumulation_steps",
    ):
        _require_int(raw[name], name=f"fold-set {name}", minimum=1)
    if (
        raw["format_version"] != FOLD_SET_FORMAT
        or raw["folds"] != FOLD_COUNT
        or raw["node_name"] != PRODUCER_NODE_NAME
        or raw["world_size_per_fold"] != PRODUCER_WORLD_SIZE
        or raw["per_rank_microbatch_size"] != PRODUCER_MICROBATCH_SIZE
        or raw["gradient_accumulation_steps"]
        != PRODUCER_GRADIENT_ACCUMULATION_STEPS
    ):
        raise ValueError("fold-set producer identity/topology mismatch")
    if raw["config_sha256"] != _require_sha256(
        expected_config_sha256, name="expected Stage 2 config SHA-256"
    ):
        raise ValueError("fold-set config SHA-256 mismatch")
    if raw["policy_sha256"] != expected_policy_sha256:
        raise ValueError("fold-set policy SHA-256 mismatch")
    if "verification_format" in raw and raw["verification_format"] != (
        VERIFICATION_RECEIPT_FORMAT
    ):
        raise ValueError("fold-set verification receipt format mismatch")

    expected_artifacts = _validated_hash_mapping(
        expected_artifact_hashes, name="expected artifact_hashes"
    )
    if "draw_manifest" not in expected_artifacts:
        raise ValueError("expected artifact_hashes has no draw_manifest")
    records = raw["records"]
    if not isinstance(records, list) or len(records) != FOLD_COUNT:
        raise ValueError("fold-set must contain exactly three records")
    verified: list[VerifiedFoldCheckpoint] = []
    root = manifest.parent.resolve()
    seen: set[int] = set()
    for raw_record in records:
        record = _mapping(raw_record, name="fold-set record")
        _keys_with_optional(
            record,
            required=_FOLD_SET_RECORD_KEYS,
            optional=_FOLD_SET_RECORD_OPTIONAL_KEYS,
            name="fold-set record",
        )
        fold_id = _require_int(record["fold_id"], name="fold-set fold_id")
        if fold_id >= FOLD_COUNT or fold_id in seen:
            raise ValueError("fold-set contains a duplicate or invalid fold")
        seen.add(fold_id)
        checkpoint = _resolved_member(
            root,
            record["checkpoint_path"],
            expected=f"fold-{fold_id}/final.pt",
        )
        partial = _resolved_member(
            root,
            record["partial_manifest_path"],
            expected=f"fold-{fold_id}/partial-manifest.json",
        )
        checkpoint_bytes = _require_int(
            record["checkpoint_bytes"], name="fold-set checkpoint_bytes", minimum=1
        )
        checkpoint_sha256 = _require_sha256(
            record["checkpoint_sha256"], name="fold-set checkpoint_sha256"
        )
        partial_sha256 = _require_sha256(
            record["partial_manifest_sha256"],
            name="fold-set partial_manifest_sha256",
        )
        if (
            checkpoint.stat().st_size != checkpoint_bytes
            or _sha256(checkpoint) != checkpoint_sha256
            or _sha256(partial) != partial_sha256
        ):
            raise ValueError(f"fold-{fold_id} assembled file receipt mismatch")
        if checkpoint.stat().st_mode & 0o222:
            raise ValueError(f"fold-{fold_id} checkpoint is not immutable/read-only")
        item = verify_single_node_fold_artifacts(
            checkpoint_path=checkpoint,
            partial_manifest_path=partial,
            fold_id=fold_id,
            policy_sha256=expected_policy_sha256,
            rows=rows,
            expected_config_sha256=expected_config_sha256,
            expected_draw_manifest_sha256=(
                expected_artifacts["draw_manifest"]
            ),
            expected_artifact_hashes=expected_artifacts,
        )
        if (
            item.checkpoint_sha256 != checkpoint_sha256
            or item.partial_manifest_sha256 != partial_sha256
            or item.checkpoint_bytes != checkpoint_bytes
        ):
            raise ValueError(f"fold-{fold_id} deep verification receipt mismatch")
        if "verification" in record and record["verification"] != item.receipt_json():
            raise ValueError(f"fold-{fold_id} embedded verification receipt mismatch")
        verified.append(item)

    if seen != set(range(FOLD_COUNT)):
        raise ValueError("fold-set does not cover folds 0, 1, and 2 exactly once")
    verified.sort(key=lambda item: item.fold_id)
    first_lineage = verified[0].lineage
    if any(item.lineage != first_lineage for item in verified[1:]):
        raise ValueError("fold-set producer lineage disagrees across folds")
    if first_lineage.launch_identity["source_tree_sha256"] != (
        expected_source_tree_sha256
    ):
        raise ValueError(
            "fold-set producer source tree is incompatible with the consumer model code"
        )
    if len({item.checkpoint_sha256 for item in verified}) != FOLD_COUNT:
        raise ValueError("fold-set checkpoint identities are not unique")
    if "lineage" in raw and raw["lineage"] != first_lineage.audit_json():
        raise ValueError("fold-set embedded lineage receipt mismatch")
    if _sha256(manifest) != actual_manifest_sha256:
        raise RuntimeError("fold-set manifest changed during verification")
    return VerifiedStage2FoldSet(
        manifest_path=manifest,
        manifest_sha256=actual_manifest_sha256,
        lineage=first_lineage,
        folds=tuple(verified),
    )


__all__ = [
    "FOLD_COUNT",
    "FOLD_SET_FORMAT",
    "FoldProducerLineage",
    "PRODUCER_GRADIENT_ACCUMULATION_STEPS",
    "PRODUCER_MICROBATCH_SIZE",
    "PRODUCER_NODE_NAME",
    "PRODUCER_WORLD_SIZE",
    "VERIFICATION_RECEIPT_FORMAT",
    "VERIFIED_IMPORT_FORMAT",
    "VerifiedFoldCheckpoint",
    "VerifiedStage2FoldSet",
    "load_verified_fold_model",
    "verify_fold_set_model_compatibility",
    "verify_single_node_fold_artifacts",
    "verify_stage2_fold_set",
]
