from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest
import torch
from torch import nn

from kcorrdiff.data.sampling import DrawRow, write_draw_manifest
from kcorrdiff.training.checkpoints import (
    CheckpointProvenance,
    TrainingCursor,
    capture_rng_state,
    save_training_checkpoint,
)
from kcorrdiff.training.config import load_stage2_config
from kcorrdiff.training.crossfit import checkpoint_record
from kcorrdiff.training.plan import build_distributed_draw_plan
from scripts.collect_stage2_folds import collect, mark_complete


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "stage2-full-width.yaml"
POLICY_SHA256 = "a" * 64
LAUNCH_IDENTITY = {
    "artifact_sha256": "d" * 64,
    "source_tree_sha256": "e" * 64,
    "container_image_sha256": "f" * 64,
    "runtime_report_sha256": "0" * 64,
    "data_contract_sha256": "1" * 64,
}


@dataclass(frozen=True, slots=True)
class FoldSetFixture:
    config_path: Path
    config_sha256: str
    draw_manifest: Path
    draw_manifest_sha256: str
    rows: tuple[DrawRow, ...]
    artifact_hashes: Mapping[str, str]
    policy_sha256: str
    launch_identity: Mapping[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_rows() -> tuple[DrawRow, ...]:
    return tuple(
        DrawRow(
            global_example_index=fold_id,
            sample_id=f"sample-{fold_id}",
            t0_utc=f"2020-01-0{fold_id + 1}T00:00:00+00:00",
            lead_hours=0.5,
            condition_signature="era5_oracle:era=1:tp=1:full_trajectory",
            block_id=f"block-{fold_id}",
            stratum="event",
            split="outer_train",
            p_target=1.0 / 3.0,
            p_draw=1.0 / 3.0,
            omega=1.0,
            fold_id=fold_id,
        )
        for fold_id in range(3)
    )


def _write_fold_set_inputs(root: Path) -> FoldSetFixture:
    rows = _draw_rows()
    draw_manifest = root / "training-draw-manifest.jsonl"
    draw_sha256 = write_draw_manifest(draw_manifest, rows)
    config = load_stage2_config(CONFIG_PATH)
    return FoldSetFixture(
        config_path=CONFIG_PATH,
        config_sha256=config.sha256,
        draw_manifest=draw_manifest,
        draw_manifest_sha256=draw_sha256,
        rows=rows,
        artifact_hashes={
            "draw_manifest": draw_sha256,
            "candidate_manifest": "2" * 64,
            "normalization": "3" * 64,
        },
        policy_sha256=POLICY_SHA256,
        launch_identity=LAUNCH_IDENTITY,
    )


def _write_valid_fold(
    run_root: Path,
    fixture: FoldSetFixture,
    fold_id: int,
    *,
    launch_identity: Mapping[str, str] | None = None,
) -> tuple[Path, Path, dict[str, torch.Tensor]]:
    launch = dict(launch_identity or fixture.launch_identity)
    worker = run_root / "workers" / f"fold-{fold_id}"
    checkpoint = worker / f"fold-{fold_id}" / "final.pt"
    checkpoint.parent.mkdir(parents=True)

    plan = build_distributed_draw_plan(
        fixture.rows,
        role="fold",
        fold_id=fold_id,
        world_size=1,
        per_rank_microbatch_size=12,
        gradient_accumulation_steps=1,
    )
    training_blocks = tuple(
        sorted(
            {
                fixture.rows[index].block_id
                for index in plan.source_row_indices
            }
        )
    )
    model = nn.Linear(2, 1, dtype=torch.float32)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(
                [[fold_id + 0.25, fold_id + 0.5]], dtype=torch.float32
            )
        )
        model.bias.fill_(fold_id + 0.75)
    expected_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=plan.optimizer_steps
    )
    provenance = CheckpointProvenance(
        protocol_version="v1.1.3b",
        config_sha256=fixture.config_sha256,
        draw_manifest_sha256=fixture.draw_manifest_sha256,
        launch_identity_sha256=launch["artifact_sha256"],
        source_tree_sha256=launch["source_tree_sha256"],
        container_image_sha256=launch["container_image_sha256"],
        runtime_report_sha256=launch["runtime_report_sha256"],
        data_contract_sha256=launch["data_contract_sha256"],
        role="fold",
        fold_id=fold_id,
        world_size=1,
        per_rank_microbatch_size=12,
        gradient_accumulation_steps=1,
    )
    cursor = TrainingCursor(
        epoch=0,
        global_example_index=plan.optimizer_steps * 12,
        optimizer_step=plan.optimizer_steps,
        microbatches_since_step=0,
    )
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        provenance=provenance,
        extra={
            "complete": True,
            "plan_semantic_sha256": plan.semantic_sha256,
            "plan_optimizer_steps": plan.optimizer_steps,
            "training_blocks": list(training_blocks),
            "rank_rng_states": [capture_rng_state()],
        },
    )
    digest = _sha256(checkpoint)
    record = checkpoint_record(
        checkpoint,
        fold_id=fold_id,
        role="fold",
        training_blocks=training_blocks,
        config_sha256=fixture.config_sha256,
        draw_manifest_sha256=fixture.draw_manifest_sha256,
        global_step=cursor.optimizer_step,
    )
    partial = worker / "partial-manifest.json"
    partial.write_text(
        json.dumps(
            {
                "format_version": "kcorrdiff.stage2-partial-training.v1",
                "protocol_version": "v1.1.3b",
                "partial": True,
                "complete": False,
                "stopped_by_max_optimizer_steps": False,
                "selection": {
                    "phases": ["crossfit"],
                    "roles": [f"fold:{fold_id}"],
                    "max_optimizer_steps": None,
                    "bounded_training_year": None,
                    "expected_training_items": None,
                    "single_node_fold_worker": True,
                    "node_name": "porsche",
                },
                "config_sha256": fixture.config_sha256,
                "launch_identity": launch,
                "draw_manifest_sha256": fixture.draw_manifest_sha256,
                "artifact_hashes": dict(fixture.artifact_hashes),
                "topology": {
                    "world_size": 1,
                    "per_rank_microbatch_size": 12,
                    "gradient_accumulation_steps": 1,
                    "global_effective_batch_size": 12,
                    "precision": "float32",
                    "tf32": False,
                    "single_node_fold_policy_sha256": fixture.policy_sha256,
                    "cursor_global_example_index_semantics": (
                        "rank_local_plan_slots_consumed_including_padding"
                    ),
                },
                "role_states": [
                    {
                        "role": "fold",
                        "fold_id": fold_id,
                        "complete": True,
                        "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_sha256": digest,
                        "cursor": asdict(cursor),
                        "optimizer_steps_run_this_invocation": (
                            plan.optimizer_steps
                        ),
                    }
                ],
                "role_checkpoints": [asdict(record)],
                "oof_manifest_sha256": None,
                "residual_scales_sha256": None,
                "fold_checkpoint_set_sha256": None,
                "wandb_mode": "online",
                "wandb_run_id": f"fold-{fold_id}",
                "tracking_audit_path": None,
                "manual_release_required": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, partial, expected_state


def _mark_fold(
    run_root: Path, fixture: FoldSetFixture, fold_id: int
) -> None:
    mark_complete(
        argparse.Namespace(
            run_root=run_root,
            worker_root=run_root / "workers" / f"fold-{fold_id}",
            fold_id=fold_id,
            policy_sha256=fixture.policy_sha256,
            config=fixture.config_path,
            draw_manifest=fixture.draw_manifest,
        )
    )


def _collect(run_root: Path) -> None:
    collect(
        argparse.Namespace(
            run_root=run_root,
            poll_seconds=0.01,
            timeout_hours=0.01,
        )
    )


def _assemble_valid_fold_set(
    root: Path,
) -> tuple[Path, FoldSetFixture, dict[int, dict[str, torch.Tensor]]]:
    run_root = root / "run"
    fixture = _write_fold_set_inputs(root)
    expected_states: dict[int, dict[str, torch.Tensor]] = {}
    for fold_id in range(3):
        _, _, expected_states[fold_id] = _write_valid_fold(
            run_root, fixture, fold_id
        )
        _mark_fold(run_root, fixture, fold_id)
    _collect(run_root)
    return (
        run_root / "assembled" / "fold-set-v1" / "fold-set-manifest.json",
        fixture,
        expected_states,
    )


def test_single_node_markers_are_verified_and_assembled(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    fixture = _write_fold_set_inputs(tmp_path)
    source_inodes: list[int] = []
    for fold_id in range(3):
        checkpoint, _, _ = _write_valid_fold(run_root, fixture, fold_id)
        source_inodes.append(checkpoint.stat().st_ino)
        _mark_fold(run_root, fixture, fold_id)

    _collect(run_root)

    assembled = run_root / "assembled" / "fold-set-v1"
    manifest = json.loads((assembled / "fold-set-manifest.json").read_text())
    assert manifest["node_name"] == "porsche"
    assert manifest["per_rank_microbatch_size"] == 12
    assert manifest["lineage"]["draw_manifest_sha256"] == (
        fixture.draw_manifest_sha256
    )
    assert [record["fold_id"] for record in manifest["records"]] == [0, 1, 2]
    assert all("verification" in record for record in manifest["records"])
    assert [
        (assembled / f"fold-{fold_id}" / "final.pt").stat().st_ino
        for fold_id in range(3)
    ] != source_inodes
    declared = (assembled / "fold-set-manifest.sha256").read_text().strip()
    assert declared == _sha256(assembled / "fold-set-manifest.json")
    assert assembled.stat().st_mode & 0o200
    pointer = run_root / "assembled" / "fold-set-pointer.json"
    assert pointer.is_file()


def test_mark_complete_rejects_arbitrary_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    fixture = _write_fold_set_inputs(tmp_path)
    checkpoint, _, _ = _write_valid_fold(run_root, fixture, 0)
    checkpoint.write_bytes(b"not-a-torch-checkpoint")

    with pytest.raises(ValueError, match="safe Torch checkpoint"):
        _mark_fold(run_root, fixture, 0)

    assert not (run_root / "completion" / "fold-0.json").exists()


def test_collect_records_cross_fold_producer_lineage_without_gating(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    fixture = _write_fold_set_inputs(tmp_path)
    for fold_id in range(3):
        launch = dict(fixture.launch_identity)
        if fold_id == 1:
            launch["source_tree_sha256"] = "9" * 64
        _write_valid_fold(
            run_root,
            fixture,
            fold_id,
            launch_identity=launch,
        )
        _mark_fold(run_root, fixture, fold_id)

    _collect(run_root)
    assert (run_root / "assembled" / "fold-set-v1" / "fold-set-manifest.json").is_file()
