"""Topology-independent draw consumption and explicit DDP padding slots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence

from kcorrdiff.data.sampling import DrawRow


@dataclass(frozen=True, slots=True)
class DrawSlot:
    """One rank-local loader slot; ``row_index=None`` is explicit zero-loss padding."""

    logical_position: int
    row_index: int | None

    @property
    def is_padding(self) -> bool:
        return self.row_index is None


@dataclass(frozen=True, slots=True)
class DistributedDrawPlan:
    role: str
    fold_id: int | None
    source_row_indices: tuple[int, ...]
    world_size: int
    per_rank_microbatch_size: int
    rank_slots: tuple[tuple[DrawSlot, ...], ...]
    semantic_sha256: str
    gradient_accumulation_steps: int = 1

    @property
    def synchronized_microbatches(self) -> int:
        local_slots = len(self.rank_slots[0])
        if local_slots % self.per_rank_microbatch_size:
            raise AssertionError("rank slots do not form complete microbatches")
        return local_slots // self.per_rank_microbatch_size

    @property
    def optimizer_steps(self) -> int:
        batches = self.synchronized_microbatches
        if batches % self.gradient_accumulation_steps:
            raise AssertionError("rank slots do not form complete accumulation windows")
        return batches // self.gradient_accumulation_steps

    @property
    def padding_slots(self) -> int:
        return sum(slot.is_padding for slots in self.rank_slots for slot in slots)


def _validate_role(role: str, fold_id: int | None) -> None:
    if role not in {"fold", "deployment", "direct_mean", "direct_q50"}:
        raise ValueError("unsupported training role")
    if (role == "fold") != (fold_id is not None):
        raise ValueError("fold role/fold_id mismatch")
    if fold_id is not None and (isinstance(fold_id, bool) or fold_id < 0):
        raise ValueError("fold_id must be non-negative")


def training_row_indices(
    rows: Sequence[DrawRow], *, role: str, fold_id: int | None = None
) -> tuple[int, ...]:
    """Filter the fixed draw sequence without reordering or resampling it."""

    _validate_role(role, fold_id)
    if not rows:
        raise ValueError("draw manifest is empty")
    for position, row in enumerate(rows):
        if row.global_example_index != position:
            raise ValueError("draw rows are not in canonical global-index order")
        if row.split != "outer_train":
            raise ValueError("Stage 2 training accepts outer_train rows only")
        if row.fold_id is None:
            raise ValueError("every cross-fit draw row requires a fold_id")
    if role == "fold":
        selected = tuple(
            index for index, row in enumerate(rows) if row.fold_id != fold_id
        )
    else:
        selected = tuple(range(len(rows)))
    if not selected:
        raise ValueError("training role selected no draw rows")
    return selected


def build_distributed_draw_plan(
    rows: Sequence[DrawRow],
    *,
    role: str,
    fold_id: int | None,
    world_size: int,
    per_rank_microbatch_size: int,
    gradient_accumulation_steps: int = 1,
) -> DistributedDrawPlan:
    """Assign each manifest draw once, with visible zero-loss tail padding.

    Padding does not invent a sample ID or alter ``P_draw``.  A training loop
    must turn it into a connected zero loss; it may not silently repeat a row.
    """

    if (
        world_size <= 0
        or per_rank_microbatch_size <= 0
        or gradient_accumulation_steps <= 0
    ):
        raise ValueError("world size, microbatch and accumulation must be positive")
    selected = training_row_indices(rows, role=role, fold_id=fold_id)
    synchronized_width = (
        world_size * per_rank_microbatch_size * gradient_accumulation_steps
    )
    padded_length = (
        (len(selected) + synchronized_width - 1) // synchronized_width
    ) * synchronized_width
    flat: list[DrawSlot] = [
        DrawSlot(logical_position=position, row_index=row_index)
        for position, row_index in enumerate(selected)
    ]
    flat.extend(
        DrawSlot(logical_position=position, row_index=None)
        for position in range(len(flat), padded_length)
    )
    rank_slots: list[tuple[DrawSlot, ...]] = []
    for rank in range(world_size):
        local: list[DrawSlot] = []
        for start in range(0, padded_length, synchronized_width):
            for accumulation_index in range(gradient_accumulation_steps):
                microbatch_start = (
                    start
                    + accumulation_index * world_size * per_rank_microbatch_size
                )
                rank_start = microbatch_start + rank * per_rank_microbatch_size
                local.extend(
                    flat[rank_start : rank_start + per_rank_microbatch_size]
                )
        rank_slots.append(tuple(local))
    actual = [
        slot.row_index
        for slots in rank_slots
        for slot in slots
        if slot.row_index is not None
    ]
    if sorted(actual) != sorted(selected) or len(actual) != len(set(actual)):
        raise AssertionError("DDP plan duplicated or dropped a draw position")
    payload = {
        "role": role,
        "fold_id": fold_id,
        "source_row_indices": selected,
        "world_size": world_size,
        "per_rank_microbatch_size": per_rank_microbatch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "padding_slots": padded_length - len(selected),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return DistributedDrawPlan(
        role=role,
        fold_id=fold_id,
        source_row_indices=selected,
        world_size=world_size,
        per_rank_microbatch_size=per_rank_microbatch_size,
        rank_slots=tuple(rank_slots),
        semantic_sha256=digest,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def unique_oof_rows(rows: Sequence[DrawRow]) -> tuple[DrawRow, ...]:
    """Return first-draw order unique OOF identities with metadata consistency."""

    seen: dict[tuple[str, float, str], DrawRow] = {}
    ordered: list[DrawRow] = []
    for row in rows:
        key = (row.t0_utc, row.lead_hours, row.condition_signature)
        previous = seen.get(key)
        if previous is None:
            seen[key] = row
            ordered.append(row)
            continue
        invariant = (
            "sample_id",
            "block_id",
            "stratum",
            "split",
            "fold_id",
            "p_target",
            "p_draw",
            "omega",
        )
        if any(getattr(previous, name) != getattr(row, name) for name in invariant):
            raise ValueError(f"repeated OOF identity changed metadata: {key}")
    if not ordered:
        raise ValueError("draw manifest has no OOF rows")
    return tuple(ordered)


def rank_oof_rows(
    rows: Sequence[DrawRow], *, rank: int, world_size: int
) -> tuple[tuple[int, DrawRow], ...]:
    """Deterministically shard unique OOF inference while retaining global order."""

    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed OOF rank/world size")
    unique = unique_oof_rows(rows)
    return tuple((index, unique[index]) for index in range(rank, len(unique), world_size))


__all__ = [
    "DistributedDrawPlan",
    "DrawSlot",
    "build_distributed_draw_plan",
    "rank_oof_rows",
    "training_row_indices",
    "unique_oof_rows",
]
