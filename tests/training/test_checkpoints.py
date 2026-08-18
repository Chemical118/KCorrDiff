from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from kcorrdiff.training.checkpoints import (
    CheckpointProvenance,
    TrainingCursor,
    load_training_checkpoint,
    save_training_checkpoint,
)


def provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        protocol_version="v1.1.3b",
        config_sha256="a" * 64,
        draw_manifest_sha256="b" * 64,
        launch_identity_sha256="c" * 64,
        source_tree_sha256="d" * 64,
        container_image_sha256="e" * 64,
        runtime_report_sha256="f" * 64,
        data_contract_sha256="0" * 64,
        role="deployment",
        fold_id=None,
        world_size=2,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=4,
    )


def test_atomic_checkpoint_roundtrip_restores_training_state_and_rng(
    tmp_path: Path,
) -> None:
    torch.manual_seed(12)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    loss = model(torch.ones(2, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    cursor = TrainingCursor(1, 8, 1, 0)
    path = tmp_path / "checkpoint.pt"
    digest = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        provenance=provenance(),
        extra={"draw_rows": 8},
    )
    expected_random = torch.rand(4)
    with torch.no_grad():
        model.weight.zero_()
    loaded_cursor, extra = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_provenance=provenance(),
    )
    assert loaded_cursor == cursor
    assert extra["draw_rows"] == 8
    assert torch.equal(torch.rand(4), expected_random)
    assert len(digest) == 64
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_loads_regardless_of_provenance_and_rejects_non_float32(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        cursor=TrainingCursor(0, 0, 0, 0),
        provenance=provenance(),
    )
    # Provenance is informational: a checkpoint from another topology or
    # launch identity still loads (research code — no lineage gating).
    cursor, _ = load_training_checkpoint(
        path,
        model=model,
        optimizer=None,
        scheduler=None,
        expected_provenance=replace(
            provenance(), world_size=1, launch_identity_sha256="1" * 64
        ),
    )
    assert cursor == TrainingCursor(0, 0, 0, 0)
    cursor, _ = load_training_checkpoint(
        path, model=model, optimizer=None, scheduler=None
    )
    assert cursor == TrainingCursor(0, 0, 0, 0)
    with pytest.raises(TypeError, match="non-float32"):
        save_training_checkpoint(
            tmp_path / "half.pt",
            model=model.half(),
            optimizer=optimizer,
            scheduler=None,
            cursor=TrainingCursor(0, 0, 0, 0),
            provenance=provenance(),
        )
