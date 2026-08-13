"""Manifest-block grouped cross-fit and immutable checkpoint provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence, TypeVar


@dataclass(frozen=True, slots=True)
class FoldItem:
    sample_id: str
    block_id: str
    fold_id: int
    split: str
    condition_signature: str

    def validate(self, *, folds: int) -> None:
        if not self.sample_id or not self.block_id or not self.condition_signature:
            raise ValueError("cross-fit identity fields must be non-empty")
        if self.split != "outer_train":
            raise ValueError("cross-fit consumes outer_train items only")
        if isinstance(self.fold_id, bool) or not 0 <= self.fold_id < folds:
            raise ValueError(f"invalid cross-fit fold: {self.fold_id}")


@dataclass(frozen=True, slots=True)
class FoldPartition:
    fold_id: int
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    training_blocks: tuple[str, ...]
    validation_blocks: tuple[str, ...]


def grouped_fold_partitions(
    items: Sequence[FoldItem], *, folds: int = 3
) -> tuple[FoldPartition, ...]:
    """Return exact held-out partitions while proving block separation."""

    if folds < 2:
        raise ValueError("cross-fit requires at least two folds")
    if not items:
        raise ValueError("cross-fit item list is empty")
    block_folds: dict[str, set[int]] = {}
    sample_ids: set[str] = set()
    for item in items:
        item.validate(folds=folds)
        if item.sample_id in sample_ids:
            raise ValueError(f"duplicate cross-fit sample: {item.sample_id}")
        sample_ids.add(item.sample_id)
        block_folds.setdefault(item.block_id, set()).add(item.fold_id)
    leaking = sorted(block for block, values in block_folds.items() if len(values) != 1)
    if leaking:
        raise ValueError(f"manifest blocks cross folds: {leaking[:5]}")
    populated = {next(iter(values)) for values in block_folds.values()}
    if populated != set(range(folds)):
        raise ValueError(f"not every cross-fit fold is populated: {sorted(populated)}")
    result: list[FoldPartition] = []
    all_indices = tuple(range(len(items)))
    for fold in range(folds):
        validation = tuple(index for index, item in enumerate(items) if item.fold_id == fold)
        training = tuple(index for index in all_indices if index not in set(validation))
        validation_blocks = tuple(sorted({items[index].block_id for index in validation}))
        training_blocks = tuple(sorted({items[index].block_id for index in training}))
        if set(validation_blocks) & set(training_blocks):
            raise AssertionError("cross-fit block leakage survived validation")
        result.append(
            FoldPartition(
                fold_id=fold,
                training_indices=training,
                validation_indices=validation,
                training_blocks=training_blocks,
                validation_blocks=validation_blocks,
            )
        )
    held_out = [index for partition in result for index in partition.validation_indices]
    if sorted(held_out) != list(range(len(items))) or len(held_out) != len(set(held_out)):
        raise AssertionError("cross-fit must hold out every item exactly once")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    fold_id: int | None
    role: str
    path: str
    sha256: str
    training_blocks_sha256: str
    config_sha256: str
    draw_manifest_sha256: str
    global_step: int


def checkpoint_record(
    checkpoint_path: Path,
    *,
    fold_id: int | None,
    role: str,
    training_blocks: Iterable[str],
    config_sha256: str,
    draw_manifest_sha256: str,
    global_step: int,
) -> CheckpointRecord:
    if role not in {"fold", "deployment", "direct_mean", "direct_q50"}:
        raise ValueError("unsupported regression checkpoint role")
    if (role == "fold") != (fold_id is not None):
        raise ValueError("fold checkpoint role/fold_id mismatch")
    if global_step < 0:
        raise ValueError("global_step cannot be negative")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    blocks = tuple(sorted(set(training_blocks)))
    if not blocks:
        raise ValueError("checkpoint training block set is empty")
    return CheckpointRecord(
        fold_id=fold_id,
        role=role,
        path=str(checkpoint_path.resolve()),
        sha256=_hash_file(checkpoint_path),
        training_blocks_sha256=_semantic_hash(blocks),
        config_sha256=config_sha256,
        draw_manifest_sha256=draw_manifest_sha256,
        global_step=global_step,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_hash(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def checkpoint_set_hash(records: Sequence[CheckpointRecord]) -> str:
    if not records:
        raise ValueError("checkpoint record set is empty")
    identities = {(record.role, record.fold_id) for record in records}
    if len(identities) != len(records):
        raise ValueError("duplicate checkpoint role/fold identity")
    return _semantic_hash(asdict(record) for record in sorted(records, key=lambda item: (item.role, item.fold_id if item.fold_id is not None else -1)))


__all__ = [
    "CheckpointRecord",
    "FoldItem",
    "FoldPartition",
    "checkpoint_record",
    "checkpoint_set_hash",
    "grouped_fold_partitions",
]
