from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kcorrdiff.training.crossfit import (
    FoldItem,
    checkpoint_record,
    checkpoint_set_hash,
    grouped_fold_partitions,
)


def items() -> list[FoldItem]:
    return [
        FoldItem(
            sample_id=f"sample-{index}",
            block_id=f"block-{index // 2}",
            fold_id=(index // 2) % 3,
            split="outer_train",
            condition_signature="era5",
        )
        for index in range(6)
    ]


def test_grouped_crossfit_holds_every_item_once_without_block_leakage() -> None:
    values = items()
    partitions = grouped_fold_partitions(values)
    assert len(partitions) == 3
    held_out = [index for partition in partitions for index in partition.validation_indices]
    assert sorted(held_out) == list(range(6))
    for partition in partitions:
        assert set(partition.training_blocks).isdisjoint(partition.validation_blocks)


def test_grouped_crossfit_rejects_block_split_across_folds() -> None:
    values = items()
    values[1] = replace(values[1], fold_id=2)
    with pytest.raises(ValueError, match="cross folds"):
        grouped_fold_partitions(values)


def test_checkpoint_records_bind_content_blocks_and_manifests(tmp_path: Path) -> None:
    checkpoint = tmp_path / "fold.pt"
    checkpoint.write_bytes(b"checkpoint")
    record = checkpoint_record(
        checkpoint,
        fold_id=0,
        role="fold",
        training_blocks=["b", "a"],
        config_sha256="c" * 64,
        draw_manifest_sha256="d" * 64,
        global_step=10,
    )
    assert len(record.sha256) == 64
    assert len(checkpoint_set_hash([record])) == 64
    with pytest.raises(ValueError, match="role/fold"):
        checkpoint_record(
            checkpoint,
            fold_id=None,
            role="fold",
            training_blocks=["a"],
            config_sha256="c",
            draw_manifest_sha256="d",
            global_step=0,
        )
