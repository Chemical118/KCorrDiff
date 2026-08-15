from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kcorrdiff.data.dataset import TrainingLabel, TrainingSample
from kcorrdiff.data.sampling import DrawRow, write_draw_manifest
from kcorrdiff.data.split_manifest import canonical_sample_id
from kcorrdiff.models.common import configure_strict_fp32_runtime
from kcorrdiff.models.regression_system import RegressionSystem
from kcorrdiff.training.batch import collate_training_samples
from kcorrdiff.training.crossfit import (
    CheckpointRecord,
    checkpoint_record,
    checkpoint_set_hash,
)
from kcorrdiff.training.data_factory import DatasetDependencies, KCorrDiffDataFactory
from kcorrdiff.training.edm_loss import build_noisy_residual_training_state
from kcorrdiff.training.oof import (
    OOF_COMPRESSED_FORMAT_VERSION,
    OOFCompressedBuild,
    OOFIdentity,
    OOFPartitionSpec,
    OOFShardWriter,
)
from kcorrdiff.training.residual_scales import (
    DEFAULT_MINIMUM_BLOCK_ESS,
    DEFAULT_MINIMUM_INDEPENDENT_BLOCKS,
    ResidualScaleAccumulator,
    write_residual_scales,
)
from kcorrdiff.training.stage3_data import (
    DiffusionScaleUnsupportedError,
    STAGE2_TARGET_BUILDER_VERSION,
    Stage3ArtifactInputs,
    Stage3BatchCollator,
    Stage2FactoryTargetLoaderFactory,
    Stage3DiffusionTarget,
    Stage3ModelConditions,
    Stage3RankDataset,
    build_stage3_distributed_plan,
    load_stage3_artifacts,
)

from tests.models.test_regression_system import small_config, zero_flow
from tests.training.test_batch import SIGNATURE, make_grid, make_sample


GRID_SIZE = 16


def _canonical_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _draws() -> tuple[DrawRow, ...]:
    t0 = datetime(2022, 1, 1, tzinfo=UTC)
    specifications = (
        (0.5, "event-0", 0),
        (1.0, "event-1", 1),
        (0.5, "event-0", 0),  # Exact duplicate: retained by Stage 3 draws.
        (1.5, "event-2", 2),
        (1.0, "event-1", 1),  # A second exact duplicate.
    )
    rows: list[DrawRow] = []
    for index, (lead, block, fold) in enumerate(specifications):
        rows.append(
            DrawRow(
                global_example_index=index,
                sample_id=canonical_sample_id(t0, lead, SIGNATURE),
                t0_utc=t0.isoformat(),
                lead_hours=lead,
                condition_signature=SIGNATURE,
                block_id=block,
                stratum="event",
                split="outer_train",
                p_target=0.2,
                p_draw=0.2,
                omega=1.0,
                fold_id=fold,
            )
        )
    return tuple(rows)


def _unique(rows: tuple[DrawRow, ...]) -> tuple[DrawRow, ...]:
    seen: set[tuple[str, float, str]] = set()
    result: list[DrawRow] = []
    for row in rows:
        key = (row.t0_utc, row.lead_hours, row.condition_signature)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return tuple(result)


def _mean(lead: float) -> np.ndarray:
    return np.full((GRID_SIZE, GRID_SIZE), np.float32(lead / 5.0), dtype=np.float32)


def _probability(lead: float) -> np.ndarray:
    return np.full((GRID_SIZE, GRID_SIZE), np.float32(0.2 + lead / 20.0), dtype=np.float32)


def _target() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validity = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.bool_)
    validity[0, 0] = False
    z = np.full((GRID_SIZE, GRID_SIZE), np.float32(0.8), dtype=np.float32)
    z[0, 0] = np.nan
    wet = validity.copy()
    return z, wet, validity


def _records(tmp_path: Path, draw_hash: str) -> tuple[CheckpointRecord, ...]:
    config_hash = "c" * 64
    result: list[CheckpointRecord] = []
    for role, fold in (
        ("fold", 0),
        ("fold", 1),
        ("fold", 2),
        ("deployment", None),
        ("direct_mean", None),
        ("direct_q50", None),
    ):
        identity = f"{role}-{fold}" if fold is not None else role
        path = tmp_path / "checkpoints" / identity / "final.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"exact checkpoint {identity}\n".encode())
        result.append(
            checkpoint_record(
                path,
                fold_id=fold,
                role=role,
                training_blocks=("event-0", "event-1", "event-2"),
                config_sha256=config_hash,
                draw_manifest_sha256=draw_hash,
                global_step=7,
            )
        )
    return tuple(result)


def _role_state(record: CheckpointRecord) -> dict[str, object]:
    return {
        "role": record.role,
        "fold_id": record.fold_id,
        "complete": True,
        "checkpoint_path": record.path,
        "checkpoint_sha256": record.sha256,
        "cursor": {
            "epoch": 1,
            "global_example_index": 5,
            "optimizer_step": record.global_step,
            "microbatches_since_step": 0,
        },
        "optimizer_steps_run_this_invocation": record.global_step,
    }


def _replace_stage2_hash(inputs: Stage3ArtifactInputs) -> Stage3ArtifactInputs:
    return replace(
        inputs,
        expected_stage2_manifest_sha256=hashlib.sha256(
            inputs.stage2_manifest.read_bytes()
        ).hexdigest(),
    )


def _imported_fold_set(
    published: SimpleNamespace, tmp_path: Path
) -> dict[str, object]:
    manifest_path = tmp_path / "verified-fold-set-manifest.json"
    manifest_hash = _canonical_json(
        manifest_path,
        {"format_version": "kcorrdiff.stage2-verified-fold-set.v1"},
    )
    states = {
        (state["role"], state["fold_id"]): state
        for state in published.stage2_payload["role_states"]
    }
    folds: list[dict[str, object]] = []
    for record in published.records:
        if record.role != "fold":
            continue
        assert record.fold_id is not None
        partial_path = tmp_path / f"fold-{record.fold_id}-partial-manifest.json"
        partial_hash = _canonical_json(partial_path, {"fold_id": record.fold_id})
        folds.append(
            {
                "fold_id": record.fold_id,
                "checkpoint_path": record.path,
                "checkpoint_bytes": Path(record.path).stat().st_size,
                "checkpoint_sha256": record.sha256,
                "partial_manifest_path": str(partial_path.resolve()),
                "partial_manifest_sha256": partial_hash,
                "cursor": dict(states[("fold", record.fold_id)]["cursor"]),
                "plan_semantic_sha256": str(record.fold_id) * 64,
                "plan_optimizer_steps": record.global_step,
                "training_blocks_sha256": record.training_blocks_sha256,
            }
        )
    return {
        "format_version": "kcorrdiff.stage2-verified-fold-set-import.v1",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "producer": {
            "protocol_version": "v1.1.3b",
            "config_sha256": published.stage2_payload["config_sha256"],
            "draw_manifest_sha256": published.stage2_payload[
                "draw_manifest_sha256"
            ],
            "launch_identity": {
                "artifact_sha256": "2" * 64,
                "source_tree_sha256": "3" * 64,
                "container_image_sha256": "4" * 64,
                "runtime_report_sha256": "5" * 64,
                "data_contract_sha256": "6" * 64,
            },
            "artifact_hashes": dict(published.stage2_payload["artifact_hashes"]),
            "node_name": "porsche",
            "world_size": 1,
            "per_rank_microbatch_size": 12,
            "gradient_accumulation_steps": 1,
            "global_effective_batch_size": 12,
            "policy_sha256": "7" * 64,
        },
        "folds": folds,
    }


@pytest.fixture()
def published(tmp_path: Path) -> SimpleNamespace:
    rows = _draws()
    draw = tmp_path / "training-draw-manifest.jsonl"
    draw_hash = write_draw_manifest(draw, rows)
    records = _records(tmp_path, draw_hash)
    fold_records = tuple(record for record in records if record.role == "fold")
    fold_set_hash = checkpoint_set_hash(fold_records)

    oof = tmp_path / "oof" / "artifact"
    writer = OOFShardWriter(
        oof,
        expected_items=len(_unique(rows)),
        maximum_bytes=len(_unique(rows)) * 2 * GRID_SIZE * GRID_SIZE * 4,
        height=GRID_SIZE,
        width=GRID_SIZE,
        shard_items=1,
        protocol_version="v1.1.3b",
        draw_manifest_sha256=draw_hash,
        target_builder_version=STAGE2_TARGET_BUILDER_VERSION,
    )
    fold_hashes = {
        record.fold_id: record.sha256 for record in fold_records
    }
    multiplicities: dict[tuple[str, float, str], int] = {}
    for row in rows:
        key = (row.t0_utc, row.lead_hours, row.condition_signature)
        multiplicities[key] = multiplicities.get(key, 0) + 1
    z, _, validity = _target()
    scales = ResidualScaleAccumulator(
        epsilon_scale=1.0e-3,
        minimum_independent_blocks=1,
        minimum_block_ess=1.0,
    )
    for row in _unique(rows):
        mean = _mean(row.lead_hours)
        writer.append(
            OOFIdentity(
                sample_id=row.sample_id,
                t0_utc=row.t0_utc,
                lead_hours=row.lead_hours,
                condition_signature=row.condition_signature,
                block_id=row.block_id,
                fold_id=int(row.fold_id),
                regression_checkpoint_sha256=fold_hashes[row.fold_id],
            ),
            _probability(row.lead_hours),
            mean,
        )
        scales.update(
            lead_hours=row.lead_hours,
            condition_signature=row.condition_signature,
            block_id=row.block_id,
            target_z=z,
            mu_z_oof=mean,
            target_validity=validity,
            omega=row.omega,
            multiplicity=multiplicities[
                (row.t0_utc, row.lead_hours, row.condition_signature)
            ],
        )
    oof_hash = writer.close()
    scale_path = tmp_path / "oof" / "residual-scales.json"
    scale_hash = write_residual_scales(
        scale_path,
        scales,
        oof_manifest_sha256=oof_hash,
        regression_checkpoint_set_sha256=fold_set_hash,
    )

    stage2 = tmp_path / "stage2-manifest.json"
    stage2_payload = {
        "format_version": "kcorrdiff.stage2-training.v1",
        "protocol_version": "v1.1.3b",
        "partial": False,
        "complete": True,
        "stopped_by_max_optimizer_steps": False,
        "selection": {
            "phases": [
                "crossfit",
                "oof",
                "residual_scales",
                "deployment",
                "direct_mean",
                "direct_q50",
            ],
            "roles": [],
            "max_optimizer_steps": None,
            "bounded_training_year": None,
            "expected_training_items": None,
            "single_node_fold_worker": False,
            "node_name": None,
        },
        "config_sha256": "c" * 64,
        "launch_identity": {
            "artifact_sha256": "d" * 64,
            "source_tree_sha256": "e" * 64,
            "container_image_sha256": "f" * 64,
            "runtime_report_sha256": "0" * 64,
            "data_contract_sha256": "1" * 64,
        },
        "draw_manifest_sha256": draw_hash,
        "artifact_hashes": {"draw_manifest": draw_hash},
        "topology": {
            "world_size": 2,
            "per_rank_microbatch_size": 1,
            "gradient_accumulation_steps": 2,
            "global_effective_batch_size": 4,
            "precision": "float32",
            "tf32": False,
            "single_node_fold_policy_sha256": None,
            "cursor_global_example_index_semantics": (
                "rank_local_plan_slots_consumed_including_padding"
            ),
        },
        "role_states": [_role_state(record) for record in records],
        "role_checkpoints": [asdict(record) for record in records],
        "oof_manifest_sha256": oof_hash,
        "residual_scales_sha256": scale_hash,
        "fold_checkpoint_set_sha256": fold_set_hash,
        "wandb_mode": "online",
        "wandb_run_id": "stage2-complete-test",
        "tracking_audit_path": str(tmp_path / "tracking.jsonl"),
        "manual_release_required": True,
        "imported_fold_set": None,
    }
    stage2_hash = _canonical_json(stage2, stage2_payload)
    inputs = Stage3ArtifactInputs(
        stage2_manifest=stage2,
        expected_stage2_manifest_sha256=stage2_hash,
        draw_manifest=draw,
        oof_artifact=oof,
        residual_scales=scale_path,
        expected_grid_shape=(GRID_SIZE, GRID_SIZE),
        expected_residual_scale_minimum_independent_blocks=1,
        expected_residual_scale_minimum_block_ess=1.0,
    )
    return SimpleNamespace(
        rows=rows,
        inputs=inputs,
        records=records,
        fold_set_hash=fold_set_hash,
        stage2_payload=stage2_payload,
    )


def _samples(rows: tuple[DrawRow, ...]) -> tuple[TrainingSample, ...]:
    grid = make_grid(size=GRID_SIZE)
    z, wet, validity = _target()
    result: list[TrainingSample] = []
    for row in rows:
        original = make_sample(
            lead_hours=row.lead_hours,
            sample_id=row.sample_id,
            grid=grid,
        )
        label = TrainingLabel(
            z=z.copy(),
            wet=wet.copy(),
            raw_accumulation_mm=np.where(validity, 1.0, np.nan),
            model_accumulation_mm=np.where(validity, 1.0, np.nan),
            target_validity=validity.copy(),
            omega=row.omega,
        )
        result.append(
            replace(
                original,
                block_id=row.block_id,
                fold_id=row.fold_id,
                label=label,
            )
        )
    return tuple(result)


class _AdapterCache:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _AdapterDecodedDataset:
    def __init__(self, rows: tuple[DrawRow, ...]) -> None:
        self.samples = _samples(rows)

    def __getitem__(self, index: int) -> TrainingSample:
        return self.samples[index]


def _adapter_cache(path: Path, verify_hashes: bool) -> _AdapterCache:
    assert isinstance(path, Path) and verify_hashes is False
    return _AdapterCache()


def _adapter_dataset(**kwargs: object) -> _AdapterDecodedDataset:
    return _AdapterDecodedDataset(tuple(kwargs["rows"]))  # type: ignore[arg-type]


def _target_factory_adapter(bundle) -> Stage2FactoryTargetLoaderFactory:
    config = SimpleNamespace(
        sha256=bundle.provenance.stage2_config_sha256,
        protocol_version=bundle.provenance.protocol_version,
        data=SimpleNamespace(
            context_detail_mode="local_max",
            context_wet_threshold_mm_per_hour=0.0,
            radar_timezone="Asia/Seoul",
        ),
    )
    inputs = SimpleNamespace(
        radar_cache_root=Path("/synthetic/radar"),
        era5_cache_root=Path("/synthetic/era5"),
        verify_cache_hashes=False,
    )
    artifacts = SimpleNamespace(
        rows=bundle.rows,
        artifact_hashes={"draw_manifest": bundle.provenance.draw_manifest_sha256},
        context_regrid_operator=object(),
        static_conditions=object(),
    )
    factory = KCorrDiffDataFactory(
        config,  # type: ignore[arg-type]
        inputs,  # type: ignore[arg-type]
        artifacts,  # type: ignore[arg-type]
        dependencies=DatasetDependencies(
            radar_cache=_adapter_cache,
            era5_cache=_adapter_cache,
            dataset=_adapter_dataset,
        ),
    )
    return Stage2FactoryTargetLoaderFactory(factory)


def test_complete_publication_is_bound_and_dense_oof_stays_lazy(
    published: SimpleNamespace,
) -> None:
    bundle = load_stage3_artifacts(published.inputs)

    assert bundle.rows == published.rows
    assert len(bundle.rows) == 5
    assert len(bundle.oof.identities) == 3
    assert bundle.oof.dense_cache_open is False
    assert bundle.provenance.stage2_manifest_sha256 == (
        published.inputs.expected_stage2_manifest_sha256
    )
    assert bundle.provenance.stage2_launch_identity_sha256 == "d" * 64
    assert bundle.provenance.stage2_source_tree_sha256 == "e" * 64
    assert bundle.provenance.stage2_container_image_sha256 == "f" * 64
    assert bundle.provenance.stage2_runtime_report_sha256 == "0" * 64
    assert bundle.provenance.stage2_data_contract_sha256 == "1" * 64
    assert bundle.provenance.fold_checkpoint_set_sha256 == published.fold_set_hash
    assert bundle.provenance.deployment_checkpoint_sha256 == next(
        record.sha256 for record in published.records if record.role == "deployment"
    )
    assert len(bundle.provenance.semantic_sha256) == 64

    first = bundle.rows[0]
    _, probability, mean = bundle.oof.read_key(
        (first.t0_utc, first.lead_hours, first.condition_signature)
    )
    assert bundle.oof.dense_cache_open is True
    assert bundle.oof.open_shard_index == 0
    np.testing.assert_array_equal(probability, _probability(0.5))
    np.testing.assert_array_equal(mean, _mean(0.5))
    second = bundle.rows[1]
    bundle.oof.read_key((second.t0_utc, second.lead_hours, second.condition_signature))
    assert bundle.oof.open_shard_index == 1


def _portable_checkpoint_overrides(
    published: SimpleNamespace, root: Path
) -> tuple[tuple[str, int | None, Path], ...]:
    result: list[tuple[str, int | None, Path]] = []
    for record in published.records:
        identity = record.role if record.fold_id is None else f"fold-{record.fold_id}"
        destination = root / "checkpoints" / identity / "final.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record.path, destination)
        result.append((record.role, record.fold_id, destination))
    return tuple(result)


def test_stage3_uses_complete_relocated_checkpoint_set_and_rejects_wrong_role(
    published: SimpleNamespace, tmp_path: Path
) -> None:
    overrides = _portable_checkpoint_overrides(
        published, tmp_path / "portable-release"
    )
    shutil.rmtree(tmp_path / "checkpoints")
    inputs = replace(published.inputs, checkpoint_path_overrides=overrides)

    bundle = load_stage3_artifacts(inputs)
    assert all(
        str(record.path).startswith(str(tmp_path / "portable-release"))
        for record in bundle.checkpoints
    )

    swapped = list(overrides)
    deployment = next(
        index for index, value in enumerate(swapped) if value[:2] == ("deployment", None)
    )
    direct_mean = next(
        index for index, value in enumerate(swapped) if value[:2] == ("direct_mean", None)
    )
    deployment_path = swapped[deployment][2]
    direct_mean_path = swapped[direct_mean][2]
    swapped[deployment] = ("deployment", None, direct_mean_path)
    swapped[direct_mean] = ("direct_mean", None, deployment_path)
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_stage3_artifacts(
            replace(published.inputs, checkpoint_path_overrides=tuple(swapped))
        )


def test_stage3_relocates_imported_b12_audit_after_original_tree_is_gone(
    published: SimpleNamespace, tmp_path: Path
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    imported = _imported_fold_set(published, tmp_path)
    payload["imported_fold_set"] = imported
    _canonical_json(published.inputs.stage2_manifest, payload)
    portable = tmp_path / "portable-release"
    overrides = _portable_checkpoint_overrides(published, portable)
    promoted_manifest = portable / "imported" / "fold-set.json"
    promoted_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(imported["manifest_path"], promoted_manifest)
    promoted_partials: list[tuple[int, Path]] = []
    for fold in imported["folds"]:
        fold_id = int(fold["fold_id"])
        destination = portable / "imported" / f"fold-{fold_id}-partial.json"
        shutil.copyfile(fold["partial_manifest_path"], destination)
        promoted_partials.append((fold_id, destination))

    shutil.rmtree(tmp_path / "checkpoints")
    Path(imported["manifest_path"]).unlink()
    for fold in imported["folds"]:
        Path(fold["partial_manifest_path"]).unlink()
    inputs = replace(
        _replace_stage2_hash(published.inputs),
        checkpoint_path_overrides=overrides,
        imported_fold_set_manifest_override=promoted_manifest,
        imported_fold_partial_manifest_overrides=tuple(promoted_partials),
    )

    bundle = load_stage3_artifacts(inputs)
    assert len(bundle.checkpoints) == 6


def test_terminal_scale_cell_is_auditable_and_excluded_from_diffusion_plan(
    published: SimpleNamespace,
) -> None:
    unsupported = ResidualScaleAccumulator(epsilon_scale=1.0e-3)
    z, _, validity = _target()
    multiplicities: dict[tuple[str, float, str], int] = {}
    for row in published.rows:
        key = (row.t0_utc, row.lead_hours, row.condition_signature)
        multiplicities[key] = multiplicities.get(key, 0) + 1
    for row in _unique(published.rows):
        unsupported.update(
            lead_hours=row.lead_hours,
            condition_signature=row.condition_signature,
            block_id=row.block_id,
            target_z=z,
            mu_z_oof=_mean(row.lead_hours),
            target_validity=validity,
            omega=row.omega,
            multiplicity=multiplicities[
                (row.t0_utc, row.lead_hours, row.condition_signature)
            ],
        )
    scale_hash = write_residual_scales(
        published.inputs.residual_scales,
        unsupported,
        oof_manifest_sha256=str(published.stage2_payload["oof_manifest_sha256"]),
        regression_checkpoint_set_sha256=published.fold_set_hash,
    )
    stage2 = dict(published.stage2_payload)
    stage2["residual_scales_sha256"] = scale_hash
    _canonical_json(published.inputs.stage2_manifest, stage2)
    unsupported_inputs = replace(
        _replace_stage2_hash(published.inputs),
        expected_residual_scale_minimum_independent_blocks=(
            DEFAULT_MINIMUM_INDEPENDENT_BLOCKS
        ),
        expected_residual_scale_minimum_block_ess=DEFAULT_MINIMUM_BLOCK_ESS,
    )
    bundle = load_stage3_artifacts(unsupported_inputs)

    row = bundle.rows[0]
    audit = bundle.residual_scale_record(
        lead_hours=row.lead_hours,
        condition_signature=row.condition_signature,
    )
    assert audit.diffusion_scale_unsupported and audit.scale is None
    with pytest.raises(DiffusionScaleUnsupportedError, match="unsupported"):
        bundle.scale_record(row)
    assert bundle.diffusion_supported_row_indices == ()
    with pytest.raises(ValueError, match="no diffusion-supported"):
        build_stage3_distributed_plan(
            bundle,
            per_rank_microbatch_size=1,
            gradient_accumulation_steps=2,
        )

    # Even a coherently re-hashed artifact may not replace the terminal state
    # with a convenient identity/global scale.
    tampered = json.loads(published.inputs.residual_scales.read_text())
    tampered["records"][0]["scale"] = 1.0
    tampered_scale_hash = _canonical_json(
        published.inputs.residual_scales, tampered
    )
    stage2["residual_scales_sha256"] = tampered_scale_hash
    _canonical_json(published.inputs.stage2_manifest, stage2)
    with pytest.raises(ValueError, match="must not substitute"):
        load_stage3_artifacts(
            replace(
                _replace_stage2_hash(published.inputs),
                expected_residual_scale_minimum_independent_blocks=(
                    DEFAULT_MINIMUM_INDEPENDENT_BLOCKS
                ),
                expected_residual_scale_minimum_block_ess=(
                    DEFAULT_MINIMUM_BLOCK_ESS
                ),
            )
        )


def test_two_rank_plan_retains_draw_multiplicity_and_explicit_padding(
    published: SimpleNamespace,
) -> None:
    bundle = load_stage3_artifacts(published.inputs)
    first = build_stage3_distributed_plan(
        bundle,
        world_size=2,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )
    second = build_stage3_distributed_plan(
        bundle,
        world_size=2,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )

    assert first == second
    assert first.optimizer_steps == 2
    assert first.padding_slots == 3
    assert [slot.row_index for slot in first.rank_slots[0]] == [0, 2, 4, None]
    assert [slot.row_index for slot in first.rank_slots[1]] == [1, 3, None, None]
    retained = sorted(
        slot.row_index
        for rank in first.rank_slots
        for slot in rank
        if slot.row_index is not None
    )
    assert retained == list(range(5))
    assert bundle.rows[0].sample_id == bundle.rows[2].sample_id
    with pytest.raises(ValueError, match="exactly two ranks"):
        build_stage3_distributed_plan(
            bundle,
            world_size=1,
            per_rank_microbatch_size=1,
            gradient_accumulation_steps=2,
        )


def test_dataset_and_typed_collate_build_float32_neutral_oof_residual(
    published: SimpleNamespace,
) -> None:
    bundle = load_stage3_artifacts(published.inputs)
    plan = build_stage3_distributed_plan(
        bundle,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )
    samples = _samples(published.rows)
    dataset = Stage3RankDataset(
        bundle,
        plan,
        rank=0,
        target_loader_factory=lambda: samples,
    )
    assert dataset.target_loader_open is False
    first, duplicate = dataset[0], dataset[1]
    assert dataset.target_loader_open is True
    assert first.row.global_example_index == 0
    assert duplicate.row.global_example_index == 2
    assert first.occurrence_probability_oof is not duplicate.occurrence_probability_oof
    np.testing.assert_array_equal(
        first.occurrence_probability_oof, duplicate.occurrence_probability_oof
    )

    grid = make_grid(size=GRID_SIZE)
    collator = Stage3BatchCollator(
        artifact_provenance=bundle.provenance,
        plan_sha256=plan.semantic_sha256,
        training_collator=lambda values: collate_training_samples(
            values,
            geometry=grid,
            allow_test_override=True,
        ),
    )
    batch = collator([first, duplicate])
    assert batch.is_zero_loss_sentinel is False
    assert batch.active_slot_indices == (0, 1)
    assert isinstance(batch.model_conditions, Stage3ModelConditions)
    assert isinstance(batch.diffusion_target, Stage3DiffusionTarget)
    assert batch.provenance.global_example_indices == (0, 2)
    assert batch.provenance.lead_hours == (0.5, 0.5)
    target = batch.diffusion_target
    conditions = batch.model_conditions
    assert target.clean_normalized_residual.dtype is torch.float32
    assert target.target_z.dtype is torch.float32
    assert target.clean_normalized_residual[:, 0, 0, 0].tolist() == [0.0, 0.0]
    expected = np.float32(
        (0.8 - 0.1) / bundle.scale_record(bundle.rows[0]).scale
    )
    torch.testing.assert_close(
        target.clean_normalized_residual[:, :, 1:, 1:],
        torch.full_like(target.clean_normalized_residual[:, :, 1:, 1:], expected),
    )
    torch.testing.assert_close(
        conditions.mu_z_oof,
        torch.full_like(conditions.mu_z_oof, 0.1),
    )
    assert not bool(target.target_validity[:, :, 0, 0].any().item())
    assert not bool(target.target_wet[:, :, 0, 0].any().item())
    assert "target_validity" not in {field.name for field in fields(Stage3ModelConditions)}
    assert not ({"b", "c", "d", "gamma", "calibration"} & {
        field.name for field in fields(Stage3DiffusionTarget)
    })

    noise_state = build_noisy_residual_training_state(
        clean_normalized_residual=target.clean_normalized_residual,
        target_validity=target.target_validity,
        sigma=torch.ones(2, dtype=torch.float32),
        noise=torch.ones_like(target.clean_normalized_residual),
    )
    assert torch.all(noise_state.noisy_residual[:, :, 0, 0] == 1.0)


def test_all_padding_collate_is_a_graph_connected_zero_sentinel(
    published: SimpleNamespace,
) -> None:
    bundle = load_stage3_artifacts(published.inputs)
    plan = build_stage3_distributed_plan(
        bundle,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )
    dataset = Stage3RankDataset(
        bundle,
        plan,
        rank=1,
        target_loader_factory=lambda: _samples(published.rows),
    )
    collator = Stage3BatchCollator(
        artifact_provenance=bundle.provenance,
        plan_sha256=plan.semantic_sha256,
        training_collator=lambda values: pytest.fail("padding must not load targets"),
    )
    padding = collator([dataset[2], dataset[3]])
    assert padding.is_zero_loss_sentinel
    assert padding.slot_loss_weights.tolist() == [0.0, 0.0]
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    loss = padding.connected_zero_loss([parameter])
    loss.backward()
    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


def test_oof_conditions_replace_deployment_heads_but_keep_causal_caches(
    published: SimpleNamespace,
) -> None:
    configure_strict_fp32_runtime()
    bundle = load_stage3_artifacts(published.inputs)
    plan = build_stage3_distributed_plan(
        bundle,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )
    samples = _samples(published.rows)
    dataset = Stage3RankDataset(
        bundle,
        plan,
        rank=0,
        target_loader_factory=lambda: samples,
    )
    grid = make_grid(size=GRID_SIZE)
    batch = Stage3BatchCollator(
        artifact_provenance=bundle.provenance,
        plan_sha256=plan.semantic_sha256,
        training_collator=lambda values: collate_training_samples(
            values, geometry=grid, allow_test_override=True
        ),
    )([dataset[0]])
    assert batch.model_conditions is not None
    deployment = RegressionSystem(small_config()).eval().requires_grad_(False)
    with torch.no_grad():
        output = deployment(
            batch.model_conditions.regression,
            flow_override=zero_flow(
                SimpleNamespace(model=batch.model_conditions.regression),
                deployment.config,
            ),
        )
    edm = batch.model_conditions.residual_edm_conditions(output)
    assert edm.mu_z is batch.model_conditions.mu_z_oof
    assert edm.probability_wet is batch.model_conditions.occurrence_probability_oof
    assert edm.condition_bank is output.condition_cache
    assert edm.era is output.era_query
    assert edm.advection_features is output.advection.features


def test_fail_closed_on_incomplete_manifest_checkpoint_and_dense_shard_tamper(
    published: SimpleNamespace,
) -> None:
    payload = dict(published.stage2_payload)
    payload["complete"] = False
    _canonical_json(published.inputs.stage2_manifest, payload)
    incomplete = _replace_stage2_hash(published.inputs)
    with pytest.raises(ValueError, match="exact complete"):
        load_stage3_artifacts(incomplete)

    # Restore the trusted manifest, then mutate an exact checkpoint.
    _canonical_json(published.inputs.stage2_manifest, published.stage2_payload)
    restored = _replace_stage2_hash(published.inputs)
    checkpoint = Path(published.records[0].path)
    checkpoint.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        load_stage3_artifacts(restored)


def test_requires_exact_complete_stage2_manifest_filename(
    published: SimpleNamespace,
) -> None:
    renamed = published.inputs.stage2_manifest.with_name("renamed-complete.json")
    published.inputs.stage2_manifest.rename(renamed)
    inputs = replace(published.inputs, stage2_manifest=renamed)
    with pytest.raises(ValueError, match="exact complete stage2-manifest.json filename"):
        load_stage3_artifacts(inputs)


def test_stage2_manifest_accepts_legacy_selection_and_topology_schema(
    published: SimpleNamespace,
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    payload.pop("imported_fold_set")
    for name in (
        "bounded_training_year",
        "expected_training_items",
        "single_node_fold_worker",
        "node_name",
    ):
        payload["selection"].pop(name)
    for name in (
        "global_effective_batch_size",
        "single_node_fold_policy_sha256",
    ):
        payload["topology"].pop(name)
    _canonical_json(published.inputs.stage2_manifest, payload)

    bundle = load_stage3_artifacts(_replace_stage2_hash(published.inputs))

    assert bundle.provenance.stage2_manifest_sha256 == hashlib.sha256(
        published.inputs.stage2_manifest.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("selection", "single_node_fold_worker", True, "partial-only selection"),
        ("selection", "node_name", "porsche", "partial-only selection"),
        ("topology", "global_effective_batch_size", 16, "execution metadata"),
        (
            "topology",
            "single_node_fold_policy_sha256",
            "a" * 64,
            "execution metadata",
        ),
    ),
)
def test_stage2_manifest_rejects_invalid_current_execution_metadata(
    published: SimpleNamespace,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    payload[section][field] = value
    _canonical_json(published.inputs.stage2_manifest, payload)

    with pytest.raises(ValueError, match=message):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))


def test_stage2_manifest_rejects_hybrid_selection_or_topology_schema(
    published: SimpleNamespace,
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    payload["selection"].pop("node_name")
    _canonical_json(published.inputs.stage2_manifest, payload)
    with pytest.raises(ValueError, match="selection schema mismatch"):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))

    payload = json.loads(json.dumps(published.stage2_payload))
    payload["topology"].pop("global_effective_batch_size")
    _canonical_json(published.inputs.stage2_manifest, payload)
    with pytest.raises(ValueError, match="topology schema mismatch"):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))


def test_stage2_manifest_accepts_and_cross_binds_imported_fold_set_audit(
    published: SimpleNamespace, tmp_path: Path
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    payload["imported_fold_set"] = _imported_fold_set(published, tmp_path)
    _canonical_json(published.inputs.stage2_manifest, payload)

    bundle = load_stage3_artifacts(_replace_stage2_hash(published.inputs))

    assert len(bundle.checkpoints) == 6
    assert payload["imported_fold_set"]["producer"]["launch_identity"] != payload[
        "launch_identity"
    ]


def test_stage2_manifest_rejects_imported_fold_set_schema_and_lineage_mismatch(
    published: SimpleNamespace, tmp_path: Path
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    imported = _imported_fold_set(published, tmp_path)
    imported["unexpected"] = True
    payload["imported_fold_set"] = imported
    _canonical_json(published.inputs.stage2_manifest, payload)
    with pytest.raises(ValueError, match="imported_fold_set schema mismatch"):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))

    imported = _imported_fold_set(published, tmp_path)
    imported["producer"]["config_sha256"] = "8" * 64
    payload["imported_fold_set"] = imported
    _canonical_json(published.inputs.stage2_manifest, payload)
    with pytest.raises(ValueError, match="producer lineage mismatch"):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))


def test_stage2_manifest_rejects_imported_fold_identity_and_record_mismatch(
    published: SimpleNamespace, tmp_path: Path
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    imported = _imported_fold_set(published, tmp_path)
    imported["folds"][1]["fold_id"] = 0
    payload["imported_fold_set"] = imported
    _canonical_json(published.inputs.stage2_manifest, payload)
    with pytest.raises(ValueError, match="fold IDs must be exactly"):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))

    imported = _imported_fold_set(published, tmp_path)
    imported["folds"][0]["plan_optimizer_steps"] += 1
    payload["imported_fold_set"] = imported
    _canonical_json(published.inputs.stage2_manifest, payload)
    with pytest.raises(ValueError, match="audit/checkpoint mismatch"):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))


def test_stage2_manifest_rejects_tampered_imported_fold_set_artifact(
    published: SimpleNamespace, tmp_path: Path
) -> None:
    payload = json.loads(json.dumps(published.stage2_payload))
    imported = _imported_fold_set(published, tmp_path)
    payload["imported_fold_set"] = imported
    Path(imported["manifest_path"]).write_bytes(b"tampered fold-set manifest\n")
    _canonical_json(published.inputs.stage2_manifest, payload)

    with pytest.raises(ValueError, match="fold-set manifest SHA-256 mismatch"):
        load_stage3_artifacts(_replace_stage2_hash(published.inputs))


def test_fail_closed_on_oof_shard_and_residual_provenance_mismatch(
    published: SimpleNamespace,
) -> None:
    shard = published.inputs.oof_artifact / "fields-00000.npy"
    with shard.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="OOF shard .* SHA-256 mismatch"):
        load_stage3_artifacts(published.inputs)

    # Rebuild from a fresh fixture is handled by pytest.  Here prove the scale
    # lineage itself is checked, even when outer file hashes are made coherent.


def test_fail_closed_on_residual_scale_cross_provenance(
    published: SimpleNamespace,
) -> None:
    scale = json.loads(published.inputs.residual_scales.read_text())
    scale["oof_manifest_sha256"] = "f" * 64
    scale_hash = _canonical_json(published.inputs.residual_scales, scale)
    stage2 = dict(published.stage2_payload)
    stage2["residual_scales_sha256"] = scale_hash
    _canonical_json(published.inputs.stage2_manifest, stage2)
    coherent_outer = _replace_stage2_hash(published.inputs)
    with pytest.raises(ValueError, match="exact provenance"):
        load_stage3_artifacts(coherent_outer)


def test_residual_scale_support_gates_are_bound_to_consumer_contract(
    published: SimpleNamespace,
) -> None:
    stricter_consumer = replace(
        published.inputs,
        expected_residual_scale_minimum_independent_blocks=2,
        expected_residual_scale_minimum_block_ess=2.0,
    )
    with pytest.raises(ValueError, match="support gates differ"):
        load_stage3_artifacts(stricter_consumer)

    with pytest.raises(ValueError, match="frozen at 30 blocks/ESS 20"):
        replace(
            published.inputs,
            expected_grid_shape=(256, 256),
            expected_residual_scale_minimum_independent_blocks=1,
            expected_residual_scale_minimum_block_ess=1.0,
        )


def test_fail_closed_when_target_loader_changes_exact_signature_join(
    published: SimpleNamespace,
) -> None:
    bundle = load_stage3_artifacts(published.inputs)
    plan = build_stage3_distributed_plan(
        bundle,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )
    samples = list(_samples(published.rows))
    samples[0] = replace(
        samples[0],
        conditions=replace(samples[0].conditions, condition_signature="radar_only:era=0:tp=0:none"),
    )
    dataset = Stage3RankDataset(
        bundle,
        plan,
        rank=0,
        target_loader_factory=lambda: samples,
    )
    with pytest.raises(ValueError, match="exactly join"):
        dataset[0]


def test_public_stage2_factory_adapter_is_lazy_bound_and_closes_caches(
    published: SimpleNamespace,
) -> None:
    bundle = load_stage3_artifacts(published.inputs)
    adapter = _target_factory_adapter(bundle)
    plan = build_stage3_distributed_plan(
        bundle,
        per_rank_microbatch_size=1,
        gradient_accumulation_steps=2,
    )
    dataset = Stage3RankDataset(
        bundle,
        plan,
        rank=0,
        target_loader_factory=adapter,
    )
    assert not dataset.target_loader_open
    assert dataset[0].sample.sample_id == bundle.rows[0].sample_id
    assert dataset.target_loader_open
    loader = adapter()
    radar, era5 = loader.radar_cache, loader.era5_cache
    loader.close()
    assert radar.closed and era5.closed

    adapter.factory.artifacts.artifact_hashes["draw_manifest"] = "0" * 64
    with pytest.raises(ValueError, match="artifact hashes differ"):
        Stage3RankDataset(
            bundle,
            plan,
            rank=0,
            target_loader_factory=adapter,
        )


def test_stage3_lazy_reader_accepts_lossless_v2_reordered_partitions(
    published: SimpleNamespace,
) -> None:
    rows = published.rows
    unique = _unique(rows)
    fold_hashes = {
        int(record.fold_id): record.sha256
        for record in published.records
        if record.role == "fold" and record.fold_id is not None
    }
    # Deliberately use fold-contiguous artifact order rather than first-draw
    # order; the Stage3 semantic index join must remain exact.
    ordered_rows = tuple(sorted(unique, key=lambda row: int(row.fold_id)))
    identities = tuple(
        OOFIdentity(
            sample_id=row.sample_id,
            t0_utc=row.t0_utc,
            lead_hours=row.lead_hours,
            condition_signature=row.condition_signature,
            block_id=row.block_id,
            fold_id=int(row.fold_id),
            regression_checkpoint_sha256=fold_hashes[int(row.fold_id)],
        )
        for row in ordered_rows
    )
    partitions = tuple(
        OOFPartitionSpec(
            partition_id=f"fold-{fold}-rank-0",
            fold_id=fold,
            rank=0,
            world_size=1,
            row_indices=tuple(
                index
                for index, identity in enumerate(identities)
                if identity.fold_id == fold
            ),
            regression_checkpoint_sha256=fold_hashes[fold],
        )
        for fold in range(3)
    )
    old_root = published.inputs.oof_artifact
    new_root = old_root.parent / "artifact-v2"
    build = OOFCompressedBuild(
        new_root,
        identities=identities,
        partitions=partitions,
        maximum_compressed_bytes=10_000_000,
        maximum_compression_ratio=10.0,
        compression_probe_bytes=1,
        concurrent_writers=1,
        height=GRID_SIZE,
        width=GRID_SIZE,
        shard_items=1,
        protocol_version="v1.1.3b",
        draw_manifest_sha256=hashlib.sha256(
            published.inputs.draw_manifest.read_bytes()
        ).hexdigest(),
        target_builder_version=STAGE2_TARGET_BUILDER_VERSION,
        config_sha256="c" * 64,
        launch_identity_sha256="d" * 64,
        initialize=True,
    )
    for partition in partitions:
        writer = build.partition(partition.partition_id)
        for index in partition.row_indices:
            row = ordered_rows[index]
            writer.append(
                index,
                identities[index],
                _probability(row.lead_hours),
                _mean(row.lead_hours),
            )
        writer.finish()
    oof_hash = build.seal()
    scale_payload = json.loads(published.inputs.residual_scales.read_text())
    scale_payload["oof_manifest_sha256"] = oof_hash
    scale_hash = _canonical_json(published.inputs.residual_scales, scale_payload)
    stage2 = dict(published.stage2_payload)
    stage2["oof_manifest_sha256"] = oof_hash
    stage2["residual_scales_sha256"] = scale_hash
    _canonical_json(published.inputs.stage2_manifest, stage2)
    inputs = replace(
        _replace_stage2_hash(published.inputs),
        oof_artifact=new_root,
    )
    bundle = load_stage3_artifacts(inputs)
    assert json.loads((new_root / "manifest.json").read_text())["format_version"] == (
        OOF_COMPRESSED_FORMAT_VERSION
    )
    assert not bundle.oof.dense_cache_open
    for row in unique:
        _, probability, mean = bundle.oof.read_key(
            (row.t0_utc, row.lead_hours, row.condition_signature)
        )
        assert np.array_equal(probability, _probability(row.lead_hours))
        assert np.array_equal(mean, _mean(row.lead_hours))
    assert bundle.oof.dense_cache_open
