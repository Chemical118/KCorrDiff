from __future__ import annotations

from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from kcorrdiff.data.condition_augmentation import ConditionAugmentationPolicy
from kcorrdiff.training.config import load_stage2_config
from kcorrdiff.training.crossfit import CheckpointRecord, checkpoint_set_hash
from kcorrdiff.training.release_index import (
    PATH_CONTRACT,
    RELEASE_INDEX_FORMAT,
    Stage2ReleaseCheckpointPaths,
    Stage2ImportedFoldSetPaths,
    Stage2ReleasePaths,
    load_stage2_release_index,
    resolve_stage3_release_inputs,
    write_stage2_release_index,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY_SOURCE = REPOSITORY_ROOT / "configs/condition-augmentation-production-v1.json"
CONFIG_SOURCE = REPOSITORY_ROOT / "configs/stage2-full-width.yaml"


def _canonical(path: Path, value: object) -> str:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(path: Path, value: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _hash(path),
        "bytes": path.stat().st_size,
    }


@pytest.fixture
def release(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "KCorrDiff"
    config_path = source / "configs/stage2-full-width.yaml"
    contract = source / "configs/data-contract-v1.1.3b.json"
    _canonical(
        contract,
        {
            "protocol_version": "v1.1.3b",
            "research_track": (
                "cprecnet_event_conditioned_era5_full_trajectory_oracle"
            ),
        },
    )

    bundle_root = root / "data/manifests/event-pretrain-production-v1"
    candidate = _file(bundle_root / "candidate-manifest.jsonl", b"candidate\n")
    draw = _file(bundle_root / "training-draw-manifest.jsonl", b"draw\n")
    policy_path = bundle_root / "condition-policy.json"
    policy_raw = json.loads(POLICY_SOURCE.read_text(encoding="utf-8"))
    _canonical(policy_path, policy_raw)
    policy = ConditionAugmentationPolicy.from_mapping(policy_raw)
    normalization = bundle_root / "normalization.json"
    _canonical(normalization, {"format_version": "normalization-fixture"})

    radar_root = root / "data/cache/radar-production-v1"
    era_root = root / "data/cache/era5-production-v1"
    radar_manifest = radar_root / "manifest.json"
    era_manifest = era_root / "manifest.json"
    _canonical(radar_manifest, {"format_version": "radar-fixture"})
    _canonical(era_manifest, {"format_version": "era-fixture"})
    target_static = _file(root / "data/static/target-static.npz", b"static")
    target_coordinates = _file(
        root / "data/coordinates/target-coordinates.npz", b"target-coordinates"
    )
    context_coordinates = _file(
        root / "data/coordinates/context-coordinates.npz", b"context-coordinates"
    )

    raw_config = yaml.safe_load(CONFIG_SOURCE.read_text(encoding="utf-8"))
    raw_config["data"]["condition_augmentation"] = policy_raw
    raw_config["data"]["normalization"] = str(normalization)
    raw_config["data"]["target_coordinates"] = str(target_coordinates)
    raw_config["data"]["context_coordinates"] = str(context_coordinates)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8"
    )
    config = load_stage2_config(config_path)

    bundle_metadata = bundle_root / "bundle-metadata.json"
    _canonical(
        bundle_metadata,
        {
            "protocol_version": "v1.1.3b",
            "condition_augmentation_policy_sha256": policy.semantic_sha256,
            "artifacts": {
                "candidate_manifest": _artifact(candidate, bundle_root),
                "training_draw_manifest": _artifact(draw, bundle_root),
                "condition_policy": _artifact(policy_path, bundle_root),
            },
        },
    )

    selected = root / "releases/stage2/selected"
    oof_root = selected / "oof/artifact"
    oof_manifest = oof_root / "manifest.json"
    _canonical(
        oof_manifest,
        {
            "format_version": "kcorrdiff.regression-oof.v2",
            "protocol_version": "v1.1.3b",
            "draw_manifest_sha256": _hash(draw),
            "shards": [],
        },
    )
    residual_scales = selected / "oof/residual-scales.json"
    _canonical(
        residual_scales,
        {
            "format_version": "kcorrdiff.residual-scales.v1",
            "oof_manifest_sha256": _hash(oof_manifest),
            "records": [],
        },
    )
    runtime_report = selected / "provenance/runtime-report.json"
    _canonical(runtime_report, {"format_version": "runtime-fixture"})
    launch_identity = selected / "provenance/launch-identity.json"
    launch_payload = {
        "format_version": "kcorrdiff.stage2-launch-identity.v1",
        "stage": "stage2",
        "source_tree": {
            "algorithm": "sha256-canonical-file-map-v1",
            "semantic_sha256": "1" * 64,
            "files": {"fixture.py": "2" * 64},
        },
        "container_image_sha256": "3" * 64,
        "runtime_report_sha256": _hash(runtime_report),
        "data_contract_sha256": _hash(contract),
    }
    _canonical(launch_identity, launch_payload)

    artifact_hashes = {
        "candidate_manifest": _hash(candidate),
        "draw_manifest": _hash(draw),
        "bundle_metadata": _hash(bundle_metadata),
        "condition_augmentation_policy": policy.semantic_sha256,
        "normalization": _hash(normalization),
        "radar_cache_manifest": _hash(radar_manifest),
        "era5_cache_manifest": _hash(era_manifest),
        "target_static": _hash(target_static),
        "target_coordinates": _hash(target_coordinates),
        "context_coordinates": _hash(context_coordinates),
    }
    original_run = tmp_path / "original-stage2-run"
    release_models = selected / "models"
    checkpoint_records: list[CheckpointRecord] = []
    released_checkpoints: dict[tuple[str, int | None], Path] = {}
    identities = [
        *(("fold", fold) for fold in range(3)),
        ("deployment", None),
        ("direct_mean", None),
        ("direct_q50", None),
    ]
    for role, fold_id in identities:
        name = role if fold_id is None else f"fold-{fold_id}"
        payload = f"checkpoint for {name}\n".encode()
        original_checkpoint = _file(original_run / name / "final.pt", payload)
        released_checkpoint = _file(release_models / name / "final.pt", payload)
        digest = _hash(original_checkpoint)
        checkpoint_records.append(
            CheckpointRecord(
                fold_id=fold_id,
                role=role,
                path=str(original_checkpoint.resolve()),
                sha256=digest,
                training_blocks_sha256=hashlib.sha256(name.encode()).hexdigest(),
                config_sha256=config.sha256,
                draw_manifest_sha256=_hash(draw),
                global_step=7,
            )
        )
        released_checkpoints[(role, fold_id)] = released_checkpoint
    stage2_manifest = selected / "stage2-manifest.json"
    stage2_payload = {
        "format_version": "kcorrdiff.stage2-training.v1",
        "protocol_version": "v1.1.3b",
        "complete": True,
        "partial": False,
        "stopped_by_max_optimizer_steps": False,
        "config_sha256": config.sha256,
        "draw_manifest_sha256": _hash(draw),
        "artifact_hashes": artifact_hashes,
        "launch_identity": {
            "artifact_sha256": _hash(launch_identity),
            "source_tree_sha256": "1" * 64,
            "container_image_sha256": "3" * 64,
            "runtime_report_sha256": _hash(runtime_report),
            "data_contract_sha256": _hash(contract),
        },
        "role_checkpoints": [asdict(record) for record in checkpoint_records],
        "oof_manifest_sha256": _hash(oof_manifest),
        "residual_scales_sha256": _hash(residual_scales),
        "fold_checkpoint_set_sha256": checkpoint_set_hash(
            [record for record in checkpoint_records if record.role == "fold"]
        ),
        "manual_release_required": True,
    }
    _canonical(stage2_manifest, stage2_payload)

    paths = Stage2ReleasePaths(
        stage2_config=config_path,
        launch_identity=launch_identity,
        data_contract=contract,
        augmented_candidate_manifest=candidate,
        augmented_draw_manifest=draw,
        augmented_bundle_metadata=bundle_metadata,
        condition_augmentation_policy=policy_path,
        normalization=normalization,
        radar_cache_manifest=radar_manifest,
        era5_cache_manifest=era_manifest,
        target_static=target_static,
        target_coordinates=target_coordinates,
        context_coordinates=context_coordinates,
        stage2_manifest=stage2_manifest,
        oof_artifact_manifest=oof_manifest,
        residual_scales=residual_scales,
        runtime_report=runtime_report,
        radar_cache_root=radar_root,
        era5_cache_root=era_root,
        oof_artifact_root=oof_root,
        model_checkpoints=Stage2ReleaseCheckpointPaths(
            fold_0=released_checkpoints[("fold", 0)],
            fold_1=released_checkpoints[("fold", 1)],
            fold_2=released_checkpoints[("fold", 2)],
            deployment=released_checkpoints[("deployment", None)],
            direct_mean=released_checkpoints[("direct_mean", None)],
            direct_q50=released_checkpoints[("direct_q50", None)],
        ),
    )
    index = selected / "release-index.json"
    return SimpleNamespace(
        root=root,
        paths=paths,
        index=index,
        stage2_payload=stage2_payload,
        policy_sha256=policy.semantic_sha256,
        original_run=original_run,
        checkpoint_records=tuple(checkpoint_records),
    )


def _attach_imported_fold_audit(
    release: SimpleNamespace, tmp_path: Path
) -> None:
    original = tmp_path / "original-b12-audit"
    promoted = release.index.parent / "imported-fold-audit"
    original_manifest = original / "fold-set-manifest.json"
    promoted_manifest = promoted / "fold-set-manifest.json"
    manifest_payload = {"format_version": "kcorrdiff.stage2-single-node-fold-set.v1"}
    manifest_hash = _canonical(original_manifest, manifest_payload)
    _canonical(promoted_manifest, manifest_payload)
    states = {
        (record.role, record.fold_id): {
            "epoch": 0,
            "global_example_index": 84,
            "optimizer_step": record.global_step,
            "microbatches_since_step": 0,
        }
        for record in release.checkpoint_records
    }
    folds: list[dict[str, object]] = []
    promoted_partials: dict[int, Path] = {}
    for record in release.checkpoint_records:
        if record.role != "fold":
            continue
        assert record.fold_id is not None
        original_partial = original / f"fold-{record.fold_id}-partial.json"
        promoted_partial = promoted / f"fold-{record.fold_id}-partial.json"
        partial_payload = {"fold_id": record.fold_id}
        partial_hash = _canonical(original_partial, partial_payload)
        _canonical(promoted_partial, partial_payload)
        promoted_partials[record.fold_id] = promoted_partial
        folds.append(
            {
                "fold_id": record.fold_id,
                "checkpoint_path": record.path,
                "checkpoint_bytes": Path(record.path).stat().st_size,
                "checkpoint_sha256": record.sha256,
                "partial_manifest_path": str(original_partial.resolve()),
                "partial_manifest_sha256": partial_hash,
                "cursor": states[("fold", record.fold_id)],
                "plan_semantic_sha256": str(record.fold_id + 1) * 64,
                "plan_optimizer_steps": record.global_step,
                "training_blocks_sha256": record.training_blocks_sha256,
            }
        )
    release.stage2_payload["imported_fold_set"] = {
        "format_version": "kcorrdiff.stage2-verified-fold-set-import.v1",
        "manifest_path": str(original_manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "producer": {},
        "folds": folds,
    }
    _canonical(release.paths.stage2_manifest, release.stage2_payload)
    release.paths = replace(
        release.paths,
        imported_fold_set=Stage2ImportedFoldSetPaths(
            manifest=promoted_manifest,
            fold_0_partial=promoted_partials[0],
            fold_1_partial=promoted_partials[1],
            fold_2_partial=promoted_partials[2],
        ),
    )
    release.original_import_audit = original


def _relocated_release_paths(
    paths: Stage2ReleasePaths, *, source: Path, destination: Path
) -> Stage2ReleasePaths:
    def moved(path: Path) -> Path:
        return destination / Path(path).relative_to(source)

    scalar_paths = {
        field.name: moved(getattr(paths, field.name))
        for field in fields(Stage2ReleasePaths)
        if field.name not in {"model_checkpoints", "imported_fold_set"}
    }
    checkpoints = Stage2ReleaseCheckpointPaths(
        **{
            field.name: moved(getattr(paths.model_checkpoints, field.name))
            for field in fields(Stage2ReleaseCheckpointPaths)
        }
    )
    imported = paths.imported_fold_set
    relocated_imported = (
        None
        if imported is None
        else Stage2ImportedFoldSetPaths(
            **{
                field.name: moved(getattr(imported, field.name))
                for field in fields(Stage2ImportedFoldSetPaths)
            }
        )
    )
    return Stage2ReleasePaths(
        **scalar_paths,
        model_checkpoints=checkpoints,
        imported_fold_set=relocated_imported,
    )


def test_roundtrip_is_canonical_portable_and_resolves_exact_stage3_cli(
    release: SimpleNamespace,
) -> None:
    digest = write_stage2_release_index(
        release.index,
        containment_root=release.root,
        release_paths=release.paths,
    )
    raw = json.loads(release.index.read_text(encoding="utf-8"))
    assert raw["format_version"] == RELEASE_INDEX_FORMAT
    assert raw["path_contract"] == PATH_CONTRACT
    assert all(not record["path"].startswith("/") for record in raw["artifacts"].values())
    assert release.index.read_bytes() == (
        json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")

    loaded = load_stage2_release_index(
        release.index, expected_sha256=digest, containment_root=release.root
    )
    assert loaded.condition_augmentation_policy_semantic_sha256 == release.policy_sha256
    assert len(loaded.semantic_sha256) == 64
    loaded.verify_current_files()
    cli = loaded.stage3_cli_inputs()
    assert cli.draw_manifest == release.paths.augmented_draw_manifest
    assert cli.oof_artifact == release.paths.oof_artifact_root
    assert cli.stage2_manifest_sha256 == _hash(release.paths.stage2_manifest)
    assert resolve_stage3_release_inputs(
        release.index, expected_sha256=digest, containment_root=release.root
    ) == cli
    assert {(item.role, item.fold_id) for item in cli.checkpoints} == {
        ("fold", 0),
        ("fold", 1),
        ("fold", 2),
        ("deployment", None),
        ("direct_mean", None),
        ("direct_q50", None),
    }
    assert set(cli.as_cli_mapping()) == {
        "--stage2-release-index",
        "--stage2-release-index-sha256",
        "--stage2-release-root",
    }


def test_release_tree_relocates_after_original_paths_are_unavailable(
    release: SimpleNamespace, tmp_path: Path
) -> None:
    _attach_imported_fold_audit(release, tmp_path)
    self_contained = tmp_path / "self-contained-release"
    shutil.copytree(release.root, self_contained)
    self_contained_paths = _relocated_release_paths(
        release.paths, source=release.root, destination=self_contained
    )
    relative_index = release.index.relative_to(release.root)
    self_contained_index = self_contained / relative_index
    digest = write_stage2_release_index(
        self_contained_index,
        containment_root=self_contained,
        release_paths=self_contained_paths,
    )
    relocated = tmp_path / "relocated-release"
    shutil.copytree(self_contained, relocated)

    shutil.rmtree(self_contained)
    shutil.rmtree(release.root)
    shutil.rmtree(release.original_run)
    shutil.rmtree(release.original_import_audit)

    loaded = load_stage2_release_index(
        relocated / relative_index,
        expected_sha256=digest,
        containment_root=relocated,
    )
    cli = loaded.stage3_cli_inputs()
    assert all(item.path.is_file() for item in cli.checkpoints)
    assert all(str(item.path).startswith(str(relocated)) for item in cli.checkpoints)
    assert cli.imported_fold_set is not None
    assert cli.imported_fold_set.manifest.is_file()
    assert all(path.is_file() for _, path in cli.imported_fold_set.partials)


def test_release_checkpoint_missing_tampered_and_wrong_role_fail_closed(
    release: SimpleNamespace,
) -> None:
    digest = write_stage2_release_index(
        release.index,
        containment_root=release.root,
        release_paths=release.paths,
    )
    release.paths.model_checkpoints.fold_0.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_stage2_release_index(
            release.index, expected_sha256=digest, containment_root=release.root
        )

    # Restore exact bytes, then forge a self-consistent index that assigns the
    # direct-mean bytes to the deployment role.  Content hashing alone is not
    # enough; the Stage 2 role lineage must reject this substitution.
    fold_record = next(
        record
        for record in release.checkpoint_records
        if record.role == "fold" and record.fold_id == 0
    )
    release.paths.model_checkpoints.fold_0.write_bytes(
        Path(fold_record.path).read_bytes()
    )
    raw = json.loads(release.index.read_text(encoding="utf-8"))
    deployment = next(
        record for record in raw["model_checkpoints"] if record["role"] == "deployment"
    )
    direct_mean = next(
        record for record in raw["model_checkpoints"] if record["role"] == "direct_mean"
    )
    for key in ("path", "sha256", "bytes"):
        deployment[key], direct_mean[key] = direct_mean[key], deployment[key]
    semantic_payload = {
        key: value for key, value in raw.items() if key != "semantic_sha256"
    }
    raw["semantic_sha256"] = hashlib.sha256(
        (
            json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    _canonical(release.index, raw)
    with pytest.raises(ValueError, match="role/content mismatch"):
        load_stage2_release_index(
            release.index,
            expected_sha256=_hash(release.index),
            containment_root=release.root,
        )


def test_atomic_publication_is_no_clobber_and_leaves_no_temporary(
    release: SimpleNamespace,
) -> None:
    write_stage2_release_index(
        release.index,
        containment_root=release.root,
        release_paths=release.paths,
    )
    original = release.index.read_bytes()
    with pytest.raises(FileExistsError):
        write_stage2_release_index(
            release.index,
            containment_root=release.root,
            release_paths=release.paths,
        )
    assert release.index.read_bytes() == original
    assert not list(release.index.parent.glob(".release-index.json.*.tmp"))


@pytest.mark.parametrize(
    "role",
    ("target_static", "radar_cache_manifest", "oof_artifact_manifest"),
)
def test_consumer_revalidates_every_bound_content(
    release: SimpleNamespace, role: str
) -> None:
    digest = write_stage2_release_index(
        release.index,
        containment_root=release.root,
        release_paths=release.paths,
    )
    with Path(getattr(release.paths, role)).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="byte count|SHA-256"):
        load_stage2_release_index(
            release.index, expected_sha256=digest, containment_root=release.root
        )


def test_writer_rejects_escape_and_any_symlinked_artifact(
    release: SimpleNamespace, tmp_path: Path
) -> None:
    outside = _file(tmp_path / "outside.json", b"outside")
    with pytest.raises(ValueError, match="escapes"):
        write_stage2_release_index(
            release.index,
            containment_root=release.root,
            release_paths=replace(release.paths, runtime_report=outside),
        )
    link = release.root / "linked-static.npz"
    link.symlink_to(release.paths.target_static)
    with pytest.raises(ValueError, match="symlink"):
        write_stage2_release_index(
            release.index,
            containment_root=release.root,
            release_paths=replace(release.paths, target_static=link),
        )


def test_consumer_rejects_symlinked_index_even_when_target_bytes_match(
    release: SimpleNamespace,
) -> None:
    digest = write_stage2_release_index(
        release.index,
        containment_root=release.root,
        release_paths=release.paths,
    )
    real = release.index.with_name("saved-release-index.json")
    release.index.rename(real)
    release.index.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        load_stage2_release_index(
            release.index, expected_sha256=digest, containment_root=release.root
        )


def test_writer_rejects_stage2_cross_lineage_substitution(
    release: SimpleNamespace,
) -> None:
    changed = dict(release.stage2_payload)
    changed["draw_manifest_sha256"] = "f" * 64
    _canonical(release.paths.stage2_manifest, changed)
    with pytest.raises(ValueError, match="draw SHA-256"):
        write_stage2_release_index(
            release.index,
            containment_root=release.root,
            release_paths=release.paths,
        )


def test_writer_rejects_config_policy_substitution(release: SimpleNamespace) -> None:
    raw = yaml.safe_load(release.paths.stage2_config.read_text(encoding="utf-8"))
    raw["data"].pop("condition_augmentation")
    release.paths.stage2_config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="no production augmentation policy"):
        write_stage2_release_index(
            release.index,
            containment_root=release.root,
            release_paths=release.paths,
        )


def test_loader_rejects_noncanonical_and_exact_schema_changes(
    release: SimpleNamespace,
) -> None:
    write_stage2_release_index(
        release.index,
        containment_root=release.root,
        release_paths=release.paths,
    )
    raw = json.loads(release.index.read_text(encoding="utf-8"))
    release.index.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON"):
        load_stage2_release_index(
            release.index,
            expected_sha256=_hash(release.index),
            containment_root=release.root,
        )

    raw["unexpected"] = True
    semantic_payload = {
        key: value for key, value in raw.items() if key != "semantic_sha256"
    }
    raw["semantic_sha256"] = hashlib.sha256(
        (
            json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    _canonical(release.index, raw)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_stage2_release_index(
            release.index,
            expected_sha256=_hash(release.index),
            containment_root=release.root,
        )


def test_loader_rejects_semantically_rehashed_path_escape(
    release: SimpleNamespace,
) -> None:
    write_stage2_release_index(
        release.index,
        containment_root=release.root,
        release_paths=release.paths,
    )
    raw = json.loads(release.index.read_text(encoding="utf-8"))
    raw["artifacts"]["runtime_report"]["path"] = "../escape.json"
    semantic_payload = {
        key: value for key, value in raw.items() if key != "semantic_sha256"
    }
    raw["semantic_sha256"] = hashlib.sha256(
        (
            json.dumps(semantic_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    _canonical(release.index, raw)
    with pytest.raises(ValueError, match="contained canonical relative path"):
        load_stage2_release_index(
            release.index,
            expected_sha256=_hash(release.index),
            containment_root=release.root,
        )
