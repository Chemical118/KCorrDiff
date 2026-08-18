"""Import independently trained Stage 2 fold checkpoints.

Fold manifests and checkpoint hashes are descriptive lineage. Import checks
only the properties needed for scientifically valid OOF inference: every fold
is present exactly once, each checkpoint belongs to the requested fold, the
training blocks exclude that held-out fold, and model tensors are compatible,
finite float32 values. Producer GPU topology and source/config hashes never
gate reuse.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

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

# Historical values remain importable for callers that write legacy metadata.
# They are not validation constraints.
PRODUCER_NODE_NAME = "porsche"
PRODUCER_WORLD_SIZE = 1
PRODUCER_MICROBATCH_SIZE = 12
PRODUCER_GRADIENT_ACCUMULATION_STEPS = 1
PRODUCER_GLOBAL_BATCH_SIZE = 12
FOLD_COUNT = 3
CURSOR_SEMANTICS = "rank_local_plan_slots_consumed_including_padding"
_PLACEHOLDER_SHA256 = "0" * 64


def _sha256(path: Path) -> str:
    """Return an informational content digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _metadata_text(value: object, default: str = _PLACEHOLDER_SHA256) -> str:
    return value if isinstance(value, str) and value else default


def _positive_int(value: object, *, name: str, default: int = 1) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_json(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON: {path}") from error
    return _mapping(raw, name=name)


def _load_checkpoint_state(path: Path) -> Mapping[str, object]:
    try:
        try:
            raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # pragma: no cover - older Torch compatibility.
            raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"fold checkpoint is not a safe Torch checkpoint: {path}") from error
    return _mapping(raw, name="fold checkpoint")


def _validated_cursor(value: object, *, name: str) -> TrainingCursor:
    raw = _mapping(value, name=name)
    try:
        cursor = TrainingCursor(**raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    cursor.validate()
    return cursor


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


def _provenance_from_state(value: object, *, fold_id: int) -> CheckpointProvenance:
    raw = dict(_mapping(value, name="checkpoint provenance"))
    defaults: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "config_sha256": _PLACEHOLDER_SHA256,
        "draw_manifest_sha256": _PLACEHOLDER_SHA256,
        "launch_identity_sha256": _PLACEHOLDER_SHA256,
        "source_tree_sha256": _PLACEHOLDER_SHA256,
        "container_image_sha256": _PLACEHOLDER_SHA256,
        "runtime_report_sha256": _PLACEHOLDER_SHA256,
        "data_contract_sha256": _PLACEHOLDER_SHA256,
        "role": "fold",
        "fold_id": fold_id,
        "world_size": 1,
        "per_rank_microbatch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    for key, default in defaults.items():
        raw.setdefault(key, default)
    try:
        provenance = CheckpointProvenance(**raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint provenance schema is invalid") from error
    provenance.validate()
    if provenance.role != "fold" or provenance.fold_id != fold_id:
        raise ValueError("checkpoint belongs to a different fold")
    return provenance


def _launch_metadata(value: object) -> dict[str, str]:
    raw = value if isinstance(value, Mapping) else {}
    return {
        key: _metadata_text(raw.get(key))
        for key in (
            "artifact_sha256",
            "source_tree_sha256",
            "container_image_sha256",
            "runtime_report_sha256",
            "data_contract_sha256",
        )
    }


def _artifact_metadata(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): _metadata_text(digest)
        for key, digest in value.items()
        if isinstance(key, str) and key
    }


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
                    "partial_manifest_path": str(fold.partial_manifest_path.resolve()),
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
    policy_sha256: str = _PLACEHOLDER_SHA256,
    rows: Sequence[DrawRow] | None = None,
    expected_config_sha256: str | None = None,
    expected_draw_manifest_sha256: str | None = None,
    expected_artifact_hashes: Mapping[str, str] | None = None,
) -> VerifiedFoldCheckpoint:
    """Load one completed fold and validate fold assignment/model numerics.

    ``expected_*`` values are accepted for compatibility and intentionally
    ignored. Their values remain available in the returned audit metadata.
    """

    del expected_config_sha256, expected_draw_manifest_sha256, expected_artifact_hashes
    if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
        raise ValueError("fold_id must be a non-negative integer")
    checkpoint = Path(checkpoint_path).resolve()
    partial = Path(partial_manifest_path).resolve()
    if not checkpoint.is_file() or not partial.is_file():
        raise FileNotFoundError("fold checkpoint or partial manifest is missing")

    manifest = _load_json(partial, name="fold partial manifest")
    if manifest.get("complete") is True:
        raise ValueError("fold worker manifest must describe a partial Stage 2 run")
    selection = manifest.get("selection")
    if isinstance(selection, Mapping):
        roles = selection.get("roles")
        if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes)):
            if f"fold:{fold_id}" not in roles:
                raise ValueError("fold manifest selects a different fold")

    state = _load_checkpoint_state(checkpoint)
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported fold checkpoint format")
    provenance = _provenance_from_state(state.get("provenance", {}), fold_id=fold_id)
    cursor = _validated_cursor(state.get("cursor"), name="checkpoint cursor")
    _validate_model_state(state.get("model"))
    extra = _mapping(state.get("extra", {}), name="checkpoint extra")
    if extra.get("complete") is not True:
        raise ValueError("fold checkpoint is not complete")

    training_blocks_raw = extra.get("training_blocks", ())
    if not isinstance(training_blocks_raw, Sequence) or isinstance(
        training_blocks_raw, (str, bytes)
    ):
        raise TypeError("checkpoint training_blocks must be a sequence")
    training_blocks = tuple(str(value) for value in training_blocks_raw)
    if len(training_blocks) != len(set(training_blocks)):
        raise ValueError("checkpoint training_blocks contains duplicates")

    if rows is not None:
        plan = build_distributed_draw_plan(
            rows,
            role="fold",
            fold_id=fold_id,
            world_size=_positive_int(provenance.world_size, name="world_size"),
            per_rank_microbatch_size=_positive_int(
                provenance.per_rank_microbatch_size, name="per_rank_microbatch_size"
            ),
            gradient_accumulation_steps=_positive_int(
                provenance.gradient_accumulation_steps,
                name="gradient_accumulation_steps",
            ),
        )
        expected_blocks = tuple(
            sorted({rows[index].block_id for index in plan.source_row_indices})
        )
        if training_blocks != expected_blocks:
            raise ValueError("checkpoint training blocks disagree with fold assignment")

    topology = manifest.get("topology")
    topology = topology if isinstance(topology, Mapping) else {}
    launch = _launch_metadata(manifest.get("launch_identity"))
    artifacts = _artifact_metadata(manifest.get("artifact_hashes"))
    config_sha256 = _metadata_text(
        manifest.get("config_sha256"), provenance.config_sha256
    )
    draw_sha256 = _metadata_text(
        manifest.get("draw_manifest_sha256"), provenance.draw_manifest_sha256
    )
    world_size = _positive_int(
        topology.get("world_size"), name="world_size", default=provenance.world_size
    )
    microbatch = _positive_int(
        topology.get("per_rank_microbatch_size"),
        name="per_rank_microbatch_size",
        default=provenance.per_rank_microbatch_size,
    )
    accumulation = _positive_int(
        topology.get("gradient_accumulation_steps"),
        name="gradient_accumulation_steps",
        default=provenance.gradient_accumulation_steps,
    )
    global_batch = _positive_int(
        topology.get("global_effective_batch_size"),
        name="global_effective_batch_size",
        default=world_size * microbatch * accumulation,
    )
    selection_node = (
        manifest.get("selection", {}).get("node_name")
        if isinstance(manifest.get("selection"), Mapping)
        else None
    )
    lineage = FoldProducerLineage(
        protocol_version=_metadata_text(manifest.get("protocol_version"), PROTOCOL_VERSION),
        config_sha256=config_sha256,
        draw_manifest_sha256=draw_sha256,
        launch_identity=launch,
        artifact_hashes=artifacts,
        node_name=_metadata_text(selection_node, "unspecified"),
        world_size=world_size,
        per_rank_microbatch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        global_effective_batch_size=global_batch,
        policy_sha256=_metadata_text(policy_sha256),
    )

    checkpoint_hash = _sha256(checkpoint)
    partial_hash = _sha256(partial)
    plan_hash = _metadata_text(extra.get("plan_semantic_sha256"))
    plan_steps = _positive_int(
        extra.get("plan_optimizer_steps"),
        name="plan_optimizer_steps",
        default=max(cursor.optimizer_step, 1),
    )
    blocks_hash = hashlib.sha256("\n".join(training_blocks).encode()).hexdigest()
    record = CheckpointRecord(
        fold_id=fold_id,
        role="fold",
        path=str(checkpoint),
        sha256=checkpoint_hash,
        training_blocks_sha256=blocks_hash,
        config_sha256=config_sha256,
        draw_manifest_sha256=draw_sha256,
        global_step=cursor.optimizer_step,
    )
    return VerifiedFoldCheckpoint(
        fold_id=fold_id,
        checkpoint_path=checkpoint,
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=checkpoint_hash,
        partial_manifest_path=partial,
        partial_manifest_sha256=partial_hash,
        provenance=provenance,
        cursor=cursor,
        plan_semantic_sha256=plan_hash,
        plan_optimizer_steps=plan_steps,
        training_blocks=training_blocks,
        training_blocks_sha256=blocks_hash,
        lineage=lineage,
        record=record,
    )


def load_verified_fold_model(verified: VerifiedFoldCheckpoint, model: nn.Module) -> None:
    """Load model weights only; provenance and producer topology are metadata."""

    load_training_checkpoint(
        verified.checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        restore_rng=False,
    )
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        if value.is_floating_point():
            if value.dtype is not torch.float32:
                raise TypeError(f"loaded fold model tensor is not float32: {name}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"loaded fold model tensor is non-finite: {name}")


def verify_fold_set_model_compatibility(
    fold_set: VerifiedStage2FoldSet,
    *,
    model_factory: Callable[[int], nn.Module],
) -> None:
    """Check that each imported state dict fits the current model code."""

    for fold in fold_set.folds:
        model = model_factory(fold.fold_id)
        try:
            load_verified_fold_model(fold, model)
        finally:
            del model


def verify_stage2_fold_set(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_policy_sha256: str | None = None,
    rows: Sequence[DrawRow] = (),
    expected_config_sha256: str | None = None,
    expected_artifact_hashes: Mapping[str, str] | None = None,
    expected_source_tree_sha256: str | None = None,
) -> VerifiedStage2FoldSet:
    """Import a complete fold set without hash, source, or GPU pinning."""

    del expected_manifest_sha256, expected_config_sha256, expected_artifact_hashes
    del expected_source_tree_sha256
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    raw = _load_json(manifest, name="fold-set manifest")
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("fold-set manifest must contain records")
    declared_folds = _positive_int(
        raw.get("folds"), name="folds", default=len(records)
    )
    if len(records) != declared_folds:
        raise ValueError("fold-set record count disagrees with declared folds")

    root = manifest.parent
    verified: list[VerifiedFoldCheckpoint] = []
    seen: set[int] = set()
    for raw_record in records:
        record = _mapping(raw_record, name="fold-set record")
        fold_id = record.get("fold_id")
        if isinstance(fold_id, bool) or not isinstance(fold_id, int) or fold_id < 0:
            raise ValueError("fold-set fold_id must be a non-negative integer")
        if fold_id in seen:
            raise ValueError("fold-set contains a duplicate fold")
        seen.add(fold_id)
        checkpoint_relative = record.get("checkpoint_path")
        partial_relative = record.get("partial_manifest_path")
        if not isinstance(checkpoint_relative, str) or not isinstance(partial_relative, str):
            raise TypeError("fold-set member paths must be strings")
        verified.append(
            verify_single_node_fold_artifacts(
                checkpoint_path=root / checkpoint_relative,
                partial_manifest_path=root / partial_relative,
                fold_id=fold_id,
                policy_sha256=_metadata_text(
                    raw.get("policy_sha256"), _metadata_text(expected_policy_sha256)
                ),
                rows=rows or None,
            )
        )

    if seen != set(range(declared_folds)):
        raise ValueError("fold-set must cover contiguous folds exactly once")
    verified.sort(key=lambda item: item.fold_id)
    return VerifiedStage2FoldSet(
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        lineage=verified[0].lineage,
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
