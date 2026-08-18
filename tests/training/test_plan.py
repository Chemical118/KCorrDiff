from __future__ import annotations

from dataclasses import replace

import pytest

from kcorrdiff.data.sampling import DrawRow
from kcorrdiff.training.plan import (
    build_distributed_draw_plan,
    rank_oof_rows,
    unique_oof_rows,
)


def row(index: int, *, fold: int, identity: int | None = None) -> DrawRow:
    semantic = index if identity is None else identity
    return DrawRow(
        global_example_index=index,
        sample_id=f"sample-{semantic}",
        t0_utc=f"2020-01-01T{semantic:02d}:00:00+00:00",
        lead_hours=0.5,
        condition_signature="era5_oracle:era=1:tp=1:full_trajectory",
        block_id=f"block-{fold}",
        stratum="event",
        split="outer_train",
        p_target=0.1,
        p_draw=0.1,
        omega=1.0,
        fold_id=fold,
    )


def test_ddp_plan_consumes_every_selected_draw_and_marks_padding() -> None:
    rows = tuple(row(index, fold=index % 3) for index in range(7))
    plan = build_distributed_draw_plan(
        rows,
        role="fold",
        fold_id=1,
        world_size=2,
        per_rank_microbatch_size=1,
    )
    assert plan.source_row_indices == (0, 2, 3, 5, 6)
    assert plan.padding_slots == 1
    assert plan.synchronized_microbatches == 3
    actual = sorted(
        slot.row_index
        for slots in plan.rank_slots
        for slot in slots
        if not slot.is_padding
    )
    assert actual == [0, 2, 3, 5, 6]
    assert len(plan.semantic_sha256) == 64


def test_plan_padding_completes_whole_gradient_accumulation_window() -> None:
    rows = tuple(row(index, fold=index % 3) for index in range(7))
    plan = build_distributed_draw_plan(
        rows,
        role="deployment",
        fold_id=None,
        world_size=2,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=4,
    )
    assert plan.optimizer_steps == 1
    assert plan.synchronized_microbatches == 4
    assert plan.padding_slots == 1
    assert [slot.logical_position for slot in plan.rank_slots[0]] == [0, 2, 4, 6]
    assert [slot.logical_position for slot in plan.rank_slots[1]] == [1, 3, 5, 7]


def test_plan_repeats_each_draw_once_per_configured_epoch() -> None:
    rows = tuple(row(index, fold=index % 3) for index in range(5))
    plan = build_distributed_draw_plan(
        rows,
        role="deployment",
        fold_id=None,
        world_size=1,
        per_rank_microbatch_size=2,
        epochs=3,
    )
    assert plan.epochs == 3
    assert plan.optimizer_steps == 9
    assert sorted(
        slot.row_index for slot in plan.rank_slots[0] if not slot.is_padding
    ) == sorted(tuple(range(5)) * 3)


def test_unique_oof_is_first_draw_order_and_rank_complete() -> None:
    first = row(0, fold=0)
    second = row(1, fold=1)
    duplicate = replace(first, global_example_index=2)
    rows = (first, second, duplicate)
    unique = unique_oof_rows(rows)
    assert unique == (first, second)
    shards = rank_oof_rows(rows, rank=0, world_size=2) + rank_oof_rows(
        rows, rank=1, world_size=2
    )
    assert sorted(index for index, _ in shards) == [0, 1]

    changed = replace(duplicate, block_id="other")
    with pytest.raises(ValueError, match="changed metadata"):
        unique_oof_rows((first, changed))
